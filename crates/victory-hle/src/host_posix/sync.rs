//! Critical sections, slim reader/writer locks, condition variables and
//! waits on several objects, for the native host.
//!
//! The first three live in the title's own memory (`CRITICAL_SECTION` is 40
//! bytes, the other two 8), so they are built on `wait_on_address` over a
//! word of the structure itself: no handle, nothing to free, and a
//! structure zeroed by the title (`SRWLOCK_INIT`) is a valid, free lock.

#![allow(clippy::not_unsafe_ptr_arg_deref)]

use std::ffi::c_void;
use std::sync::{Arc, Mutex};
use std::sync::atomic::{AtomicU8, AtomicU16, AtomicU32, AtomicU64, Ordering};
use std::time::{Duration, Instant};

use victory_platform::sync::{wait_on_address, wake_all, wake_one};
use victory_platform::thread;

use super::{ERROR_INVALID_PARAMETER, INFINITE, WAIT_OBJECT_0, WAIT_TIMEOUT, Waitable, set_error};

const ERROR_TIMEOUT: u32 = 1460;

// Waits on several objects. Everything the title can signal goes through
// `signalled`, which bumps a counter that waiters sleep on; each waiter
// then looks at its own objects.

static EPOCH: AtomicU32 = AtomicU32::new(0);
/// Wait-all takers go one at a time: one that took some objects and could
/// not take the rest gives them back, and two doing it would starve.
static WAIT_ALL: Mutex<()> = Mutex::new(());

/// Threads in [`wait_multiple`]: with none, a signal makes no wake system call.
static MULTI_WAITERS: AtomicU32 = AtomicU32::new(0);

#[cfg(test)]
pub(super) fn multi_waiters() -> u32 {
    MULTI_WAITERS.load(Ordering::SeqCst)
}

/// Something a wait could be waiting for was signalled.
pub(super) fn signalled() {
    // SeqCst on both sides: a waiter that registers after this load reads the new epoch.
    EPOCH.fetch_add(1, Ordering::SeqCst);
    if MULTI_WAITERS.load(Ordering::SeqCst) != 0 {
        wake_all(&EPOCH);
    }
}

/// `WaitForMultipleObjects` over `objects`: the index of the first one
/// signalled (any), or `WAIT_OBJECT_0` once all are taken (all).
pub(super) fn wait_multiple(objects: &[Waitable], all: bool, timeout: Option<Duration>) -> u32 {
    MULTI_WAITERS.fetch_add(1, Ordering::SeqCst);
    let result = wait_registered(objects, all, timeout);
    MULTI_WAITERS.fetch_sub(1, Ordering::SeqCst);
    result
}

fn wait_registered(objects: &[Waitable], all: bool, timeout: Option<Duration>) -> u32 {
    let deadline = timeout.map(|t| Instant::now() + t);
    loop {
        let mut seen = EPOCH.load(Ordering::SeqCst);
        if all {
            let restored = {
                let _one_at_a_time = WAIT_ALL.lock().unwrap_or_else(|e| e.into_inner());
                let taken = objects.iter().take_while(|o| o.try_take()).count();
                if taken == objects.len() {
                    return WAIT_OBJECT_0;
                }
                let mut restored = false;
                for o in &objects[..taken] {
                    restored |= o.give_back();
                }
                restored
            };
            if restored {
                let after = EPOCH.fetch_add(1, Ordering::SeqCst).wrapping_add(1);
                if MULTI_WAITERS.load(Ordering::SeqCst) > 1 {
                    wake_all(&EPOCH);
                }
                if after == seen.wrapping_add(1) {
                    seen = after;
                }
            }
        } else if let Some(i) = objects.iter().position(Waitable::try_take) {
            return WAIT_OBJECT_0 + i as u32;
        }
        let left = match deadline {
            None => None,
            Some(d) => match d.checked_duration_since(Instant::now()) {
                Some(left) if !left.is_zero() => Some(left),
                _ => return WAIT_TIMEOUT,
            },
        };
        wait_on_address(&EPOCH, seen, left);
    }
}

// Critical sections.

const CS_LOCK_WORD: usize = 8;
const CS_RECURSION: usize = 12;
const CS_OWNER: usize = 16;

struct Cs(*mut u8);

impl Cs {
    fn word(&self) -> &AtomicU32 {
        // SAFETY: the title's initialised CRITICAL_SECTION, 4-byte aligned.
        unsafe { &*self.0.add(CS_LOCK_WORD).cast::<AtomicU32>() }
    }

    fn owner(&self) -> &AtomicU64 {
        // SAFETY: as above.
        unsafe { &*self.0.add(CS_OWNER).cast::<AtomicU64>() }
    }

    fn recursion(&self) -> *mut i32 {
        // SAFETY: as above.
        unsafe { self.0.add(CS_RECURSION).cast() }
    }

    fn take(&self) {
        let word = self.word();
        if word
            .compare_exchange(0, 1, Ordering::Acquire, Ordering::Relaxed)
            .is_ok()
        {
            return;
        }
        // 2 means locked with waiters, so unlocking knows to wake one.
        while word.swap(2, Ordering::Acquire) != 0 {
            wait_on_address(word, 2, None);
        }
    }

    fn release(&self) {
        if self.word().swap(0, Ordering::Release) == 2 {
            wake_one(self.word());
        }
    }
}

pub extern "system" fn InitializeCriticalSection(cs: *mut u8) {
    // SAFETY: the title's 40-byte structure.
    unsafe { std::ptr::write_bytes(cs, 0, 40) };
}

pub extern "system" fn InitializeCriticalSectionAndSpinCount(cs: *mut u8, _spin: u32) -> i32 {
    InitializeCriticalSection(cs);
    1
}

pub extern "system" fn InitializeCriticalSectionEx(cs: *mut u8, _spin: u32, _flags: u32) -> i32 {
    InitializeCriticalSection(cs);
    1
}

pub extern "system" fn SetCriticalSectionSpinCount(_cs: *mut u8, _spin: u32) -> u32 {
    0
}

pub extern "system" fn DeleteCriticalSection(_cs: *mut u8) {}

pub extern "system" fn EnterCriticalSection(cs: *mut u8) {
    let cs = Cs(cs);
    let me = u64::from(thread::current_id());
    if cs.owner().load(Ordering::Relaxed) == me {
        // SAFETY: only the owner touches the count.
        unsafe { cs.recursion().write(cs.recursion().read() + 1) };
        return;
    }
    cs.take();
    cs.owner().store(me, Ordering::Relaxed);
    // SAFETY: we hold it.
    unsafe { cs.recursion().write(1) };
}

pub extern "system" fn TryEnterCriticalSection(cs: *mut u8) -> i32 {
    let cs = Cs(cs);
    let me = u64::from(thread::current_id());
    if cs.owner().load(Ordering::Relaxed) == me {
        // SAFETY: only the owner touches the count.
        unsafe { cs.recursion().write(cs.recursion().read() + 1) };
        return 1;
    }
    if cs
        .word()
        .compare_exchange(0, 1, Ordering::Acquire, Ordering::Relaxed)
        .is_err()
    {
        return 0;
    }
    cs.owner().store(me, Ordering::Relaxed);
    // SAFETY: we hold it.
    unsafe { cs.recursion().write(1) };
    1
}

pub extern "system" fn LeaveCriticalSection(cs: *mut u8) {
    let cs = Cs(cs);
    // SAFETY: the owner's count.
    let left = unsafe {
        let n = cs.recursion().read() - 1;
        cs.recursion().write(n);
        n
    };
    if left == 0 {
        cs.owner().store(0, Ordering::Relaxed);
        cs.release();
    }
}

// Slim reader/writer locks: bit 31 held exclusively, bit 30 someone is
// waiting, the rest the reader count.

const EXCLUSIVE: u32 = 1 << 31;
const WAITERS: u32 = 1 << 30;
const READERS: u32 = WAITERS - 1;

fn srw<'a>(lock: *mut c_void) -> &'a AtomicU32 {
    // SAFETY: the title's 8-byte SRWLOCK, whose low word we use.
    unsafe { &*lock.cast::<AtomicU32>() }
}

pub extern "system" fn InitializeSRWLock(lock: *mut c_void) {
    // SAFETY: the title's 8-byte structure.
    unsafe { lock.cast::<u64>().write(0) };
}

fn srw_try(word: &AtomicU32, shared: bool) -> Result<(), u32> {
    let s = word.load(Ordering::Relaxed);
    let free = if shared {
        s & EXCLUSIVE == 0
    } else {
        s & (EXCLUSIVE | READERS) == 0
    };
    if !free {
        return Err(s);
    }
    let next = if shared { s + 1 } else { s | EXCLUSIVE };
    word.compare_exchange(s, next, Ordering::Acquire, Ordering::Relaxed)
        .map(|_| ())
        .or(Err(word.load(Ordering::Relaxed)))
}

fn srw_acquire(word: &AtomicU32, shared: bool) {
    loop {
        let Err(seen) = srw_try(word, shared) else {
            return;
        };
        srw_wait(word, seen, shared);
    }
}

fn srw_wait(word: &AtomicU32, seen: u32, shared: bool) {
    // srw_try's acquisition CAS can lose a race, then observe the owner's
    // release. Do not sleep on an available lock: nobody owns it to wake us.
    // A reader can also retry immediately while other readers hold it.
    let incompatible = if shared {
        EXCLUSIVE
    } else {
        EXCLUSIVE | READERS
    };
    if seen & incompatible == 0 {
        return;
    }
    let held = seen | WAITERS;
    let marked = seen & WAITERS != 0
        || word
            .compare_exchange(seen, held, Ordering::Relaxed, Ordering::Relaxed)
            .is_ok();
    if marked {
        wait_on_address(word, held, None);
    }
}

pub extern "system" fn AcquireSRWLockExclusive(lock: *mut c_void) {
    srw_acquire(srw(lock), false);
}

pub extern "system" fn AcquireSRWLockShared(lock: *mut c_void) {
    srw_acquire(srw(lock), true);
}

pub extern "system" fn TryAcquireSRWLockExclusive(lock: *mut c_void) -> u8 {
    u8::from(srw_try(srw(lock), false).is_ok())
}

pub extern "system" fn TryAcquireSRWLockShared(lock: *mut c_void) -> u8 {
    u8::from(srw_try(srw(lock), true).is_ok())
}

pub extern "system" fn ReleaseSRWLockExclusive(lock: *mut c_void) {
    let word = srw(lock);
    if word.swap(0, Ordering::Release) & WAITERS != 0 {
        wake_all(word);
    }
}

pub extern "system" fn ReleaseSRWLockShared(lock: *mut c_void) {
    let word = srw(lock);
    let before = word.fetch_sub(1, Ordering::Release);
    if before & READERS == 1 && before & WAITERS != 0 {
        wake_all(word);
    }
}

// Condition variables: a counter that waking bumps.

fn condition<'a>(cv: *mut c_void) -> &'a AtomicU32 {
    // SAFETY: the title's 8-byte CONDITION_VARIABLE, whose low word we use.
    unsafe { &*cv.cast::<AtomicU32>() }
}

pub extern "system" fn InitializeConditionVariable(cv: *mut c_void) {
    // SAFETY: the title's 8-byte structure.
    unsafe { cv.cast::<u64>().write(0) };
}

fn sleep_condition(
    cv: *mut c_void,
    ms: u32,
    release: impl FnOnce(),
    reacquire: impl FnOnce(),
) -> i32 {
    let word = condition(cv);
    let ticket = word.load(Ordering::Acquire);
    release();
    let deadline = (ms != u32::MAX).then(|| Instant::now() + Duration::from_millis(u64::from(ms)));
    loop {
        // A wake between the load and the wait changes the word, and the
        // wait returns at once. A spurious return leaves the word alone:
        // with a deadline, only a changed word is a wake, so the wait goes
        // on until the time is up (wait_on_address's callers re-check, as
        // here). Without a deadline any return is taken as a wake, which is
        // legal for a condition variable: the caller re-checks its own
        // predicate.
        let left = match deadline {
            None => None,
            Some(d) => match d.checked_duration_since(Instant::now()) {
                Some(left) if !left.is_zero() => Some(left),
                _ => {
                    reacquire();
                    set_error(ERROR_TIMEOUT);
                    return 0;
                }
            },
        };
        if !wait_on_address(word, ticket, left) {
            reacquire();
            set_error(ERROR_TIMEOUT);
            return 0;
        }
        if deadline.is_none() || word.load(Ordering::Acquire) != ticket {
            reacquire();
            return 1;
        }
        // The word did not change, a spurious return. wait_on_address also
        // returns at once on EFAULT-type errors, so back off instead of
        // spinning until the deadline.
        std::thread::sleep(Duration::from_micros(100));
    }
}

pub extern "system" fn SleepConditionVariableCS(cv: *mut c_void, cs: *mut u8, ms: u32) -> i32 {
    let me = u64::from(thread::current_id());
    let held = Cs(cs);
    // SAFETY: the caller owns the section.
    let depth = unsafe { held.recursion().read() };
    sleep_condition(
        cv,
        ms,
        || {
            held.owner().store(0, Ordering::Relaxed);
            // SAFETY: still the owner's.
            unsafe { held.recursion().write(0) };
            held.release();
        },
        || {
            held.take();
            held.owner().store(me, Ordering::Relaxed);
            // SAFETY: owned again.
            unsafe { held.recursion().write(depth) };
        },
    )
}

pub extern "system" fn SleepConditionVariableSRW(
    cv: *mut c_void,
    lock: *mut c_void,
    ms: u32,
    flags: u32,
) -> i32 {
    let shared = flags & 1 != 0;
    sleep_condition(
        cv,
        ms,
        || {
            if shared {
                ReleaseSRWLockShared(lock);
            } else {
                ReleaseSRWLockExclusive(lock);
            }
        },
        || srw_acquire(srw(lock), shared),
    )
}

pub extern "system" fn WakeConditionVariable(cv: *mut c_void) {
    let word = condition(cv);
    word.fetch_add(1, Ordering::AcqRel);
    wake_one(word);
}

pub extern "system" fn WakeAllConditionVariable(cv: *mut c_void) {
    let word = condition(cv);
    word.fetch_add(1, Ordering::AcqRel);
    wake_all(word);
}

// WaitOnAddress / WakeByAddress*: 1, 2, 4, or 8 byte values keyed by exact
// address (`WakeByAddress*` carries no size argument, while Darwin's
// `os_sync_wake_by_address_*` requires an aligned 4- or 8-byte size).

const ADDR_STRIPES: usize = 32;

struct AddrWaiter {
    addr: usize,
    woken: AtomicU32,
}

static ADDR_WAITERS: [Mutex<Vec<Arc<AddrWaiter>>>; ADDR_STRIPES] =
    [const { Mutex::new(Vec::new()) }; ADDR_STRIPES];

fn addr_stripe(addr: usize) -> &'static Mutex<Vec<Arc<AddrWaiter>>> {
    &ADDR_WAITERS[((addr >> 2) ^ (addr >> 12)) & (ADDR_STRIPES - 1)]
}

unsafe fn read_addr_pair(
    address: *const c_void,
    compare: *const c_void,
    size: usize,
) -> Option<(u64, u64)> {
    let addr = address as usize;
    // SAFETY: caller checked non-null; the title supplies valid `size`-byte regions.
    unsafe {
        Some(match size {
            1 => (
                u64::from((&*address.cast::<AtomicU8>()).load(Ordering::SeqCst)),
                u64::from(compare.cast::<u8>().read_unaligned()),
            ),
            2 => (
                u64::from(if addr.is_multiple_of(2) {
                    (&*address.cast::<AtomicU16>()).load(Ordering::SeqCst)
                } else {
                    address.cast::<u16>().read_unaligned()
                }),
                u64::from(compare.cast::<u16>().read_unaligned()),
            ),
            4 => (
                u64::from(if addr.is_multiple_of(4) {
                    (&*address.cast::<AtomicU32>()).load(Ordering::SeqCst)
                } else {
                    address.cast::<u32>().read_unaligned()
                }),
                u64::from(compare.cast::<u32>().read_unaligned()),
            ),
            8 => (
                if addr.is_multiple_of(8) {
                    (&*address.cast::<AtomicU64>()).load(Ordering::SeqCst)
                } else {
                    address.cast::<u64>().read_unaligned()
                },
                compare.cast::<u64>().read_unaligned(),
            ),
            _ => return None,
        })
    }
}

pub extern "system" fn WaitOnAddress(
    address: *const c_void,
    compare: *const c_void,
    size: usize,
    ms: u32,
) -> i32 {
    if address.is_null() || compare.is_null() || !matches!(size, 1 | 2 | 4 | 8) {
        set_error(ERROR_INVALID_PARAMETER);
        return 0;
    }
    let deadline =
        (ms != INFINITE).then(|| Instant::now() + Duration::from_millis(u64::from(ms)));
    let addr = address as usize;
    let waiter = {
        let mut stripe = super::lock(addr_stripe(addr));
        // SAFETY: `address` and `compare` are non-null and `size` is 1, 2, 4, or 8.
        let (cur, exp) = unsafe { read_addr_pair(address, compare, size) }.unwrap();
        if cur != exp {
            return 1;
        }
        if ms == 0 {
            drop(stripe);
            set_error(ERROR_TIMEOUT);
            return 0;
        }
        let waiter = Arc::new(AddrWaiter {
            addr,
            woken: AtomicU32::new(0),
        });
        stripe.push(Arc::clone(&waiter));
        waiter
    };
    while waiter.woken.load(Ordering::Acquire) == 0 {
        let left = match deadline {
            None => None,
            Some(d) => match d.checked_duration_since(Instant::now()) {
                Some(left) if !left.is_zero() => Some(left),
                _ => break,
            },
        };
        wait_on_address(&waiter.woken, 0, left);
    }
    if waiter.woken.load(Ordering::Acquire) != 0 {
        return 1;
    }
    let mut stripe = super::lock(addr_stripe(addr));
    if waiter.woken.load(Ordering::Acquire) != 0 {
        return 1;
    }
    stripe.retain(|w| !Arc::ptr_eq(w, &waiter));
    drop(stripe);
    set_error(ERROR_TIMEOUT);
    0
}

pub extern "system" fn WakeByAddressSingle(address: *mut c_void) {
    if address.is_null() {
        return;
    }
    let addr = address as usize;
    let woken = {
        let mut stripe = super::lock(addr_stripe(addr));
        stripe.iter().position(|w| w.addr == addr).map(|i| {
            let w = stripe.remove(i);
            w.woken.store(1, Ordering::Release);
            w
        })
    };
    if let Some(w) = woken {
        wake_one(&w.woken);
    }
}

pub extern "system" fn WakeByAddressAll(address: *mut c_void) {
    if address.is_null() {
        return;
    }
    let addr = address as usize;
    let mut woken = Vec::new();
    {
        let mut stripe = super::lock(addr_stripe(addr));
        let mut i = 0;
        while i < stripe.len() {
            if stripe[i].addr == addr {
                let w = stripe.remove(i);
                w.woken.store(1, Ordering::Release);
                woken.push(w);
            } else {
                i += 1;
            }
        }
    }
    for w in woken {
        wake_one(&w.woken);
    }
}

// Yielding.

pub extern "system" fn SwitchToThread() -> i32 {
    std::thread::yield_now();
    1
}

pub extern "system" fn SleepEx(ms: u32, _alertable: i32) -> u32 {
    super::Sleep(ms);
    0
}

/// Names this module answers for.
pub(super) fn lookup(name: &str) -> Option<usize> {
    macro_rules! table {
        ($($f:ident),* $(,)?) => {
            match name {
                $(stringify!($f) => Some($f as *const () as usize),)*
                _ => None,
            }
        };
    }
    table!(
        InitializeCriticalSection,
        InitializeCriticalSectionAndSpinCount,
        InitializeCriticalSectionEx,
        SetCriticalSectionSpinCount,
        DeleteCriticalSection,
        EnterCriticalSection,
        TryEnterCriticalSection,
        LeaveCriticalSection,
        InitializeSRWLock,
        AcquireSRWLockExclusive,
        AcquireSRWLockShared,
        TryAcquireSRWLockExclusive,
        TryAcquireSRWLockShared,
        ReleaseSRWLockExclusive,
        ReleaseSRWLockShared,
        InitializeConditionVariable,
        SleepConditionVariableCS,
        SleepConditionVariableSRW,
        WakeConditionVariable,
        WakeAllConditionVariable,
        WaitOnAddress,
        WakeByAddressSingle,
        WakeByAddressAll,
        SwitchToThread,
        SleepEx,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::mpsc;

    #[test]
    fn srw_does_not_sleep_on_an_available_lock_after_a_failed_try() {
        // A failed acquisition CAS can be followed by the owner's release,
        // so srw_try can return Err with an available word. WAITERS alone
        // is also available (the last shared owner leaves that bit set).
        for (seen, shared) in [
            (0, false),
            (WAITERS, false),
            (0, true),
            (WAITERS, true),
            (1, true),
            (WAITERS | 1, true),
        ] {
            let word = Arc::new(AtomicU32::new(seen));
            let waiter_word = Arc::clone(&word);
            let (tx, rx) = mpsc::channel();
            let waiter = std::thread::spawn(move || {
                srw_wait(&waiter_word, seen, shared);
                tx.send(()).unwrap();
            });
            let completed = rx.recv_timeout(Duration::from_secs(1)).is_ok();
            let after = word.load(Ordering::Relaxed);
            // Rescue a broken implementation before asserting, including if
            // it has not reached wait_on_address yet. Never leave a sleeper.
            word.store(EXCLUSIVE, Ordering::Release);
            wake_all(&word);
            waiter.join().unwrap();
            assert!(
                completed,
                "slept on available SRW word {seen:#x}, shared={shared}"
            );
            assert_eq!(after, seen, "marked an available lock as waiting");
        }
    }

    /// A CV in "title memory", shared by the waiter thread and this one.
    fn cv_fixture() -> (*mut c_void, *mut u8) {
        let cv = Box::leak(Box::new(0u64)) as *mut u64 as *mut c_void;
        let cs = Box::leak(Box::new([0u8; 64])) as *mut [u8; 64] as *mut u8;
        InitializeCriticalSection(cs);
        InitializeConditionVariable(cv);
        (cv, cs)
    }

    #[test]
    fn a_wake_that_changes_nothing_is_not_a_condition_variable_signal() {
        // wake_one wakes a waiter without changing the word; only a changed
        // word is a signal, so the wait must still end as a time-out (the
        // native_probe waits test saw these under load, 1 run in 3).
        let (cv, cs) = cv_fixture();
        let (cv_addr, cs_addr) = (cv as usize, cs as usize);
        let waiter = std::thread::spawn(move || {
            let (cv, cs) = (cv_addr as *mut c_void, cs_addr as *mut u8);
            EnterCriticalSection(cs);
            let start = Instant::now();
            let slept = SleepConditionVariableCS(cv, cs, 1_000);
            LeaveCriticalSection(cs);
            (slept, start.elapsed())
        });
        // Wake word-less, repeatedly, so one lands while the waiter is parked
        // even on a busy host.
        for _ in 0..10 {
            std::thread::sleep(Duration::from_millis(50));
            wake_one(condition(cv));
        }
        let (slept, waited) = waiter.join().unwrap();
        assert_eq!(slept, 0, "a word-less wake is not a signal");
        // Loose for slow hosts: the wait ran past the wakes to its deadline.
        assert!(waited >= Duration::from_millis(500), "{waited:?}");
    }

    #[test]
    fn a_condition_variable_wake_ends_the_wait_at_once() {
        let (cv, cs) = cv_fixture();
        let (cv_addr, cs_addr) = (cv as usize, cs as usize);
        let waiter = std::thread::spawn(move || {
            let (cv, cs) = (cv_addr as *mut c_void, cs_addr as *mut u8);
            EnterCriticalSection(cs);
            let start = Instant::now();
            let slept = SleepConditionVariableCS(cv, cs, 30_000);
            LeaveCriticalSection(cs);
            (slept, start.elapsed())
        });
        std::thread::sleep(Duration::from_millis(100));
        WakeConditionVariable(cv);
        let (slept, waited) = waiter.join().unwrap();
        assert_eq!(slept, 1);
        assert!(waited < Duration::from_secs(5), "{waited:?}");
    }

    #[test]
    fn wait_on_address_handles_sizes_timeouts_and_isolated_wakes() {
        for dll in [
            "api-ms-win-core-synch-l1-2-0.dll",
            "kernelbase.dll",
            "kernel32.dll",
        ] {
            for sym in ["WaitOnAddress", "WakeByAddressSingle", "WakeByAddressAll"] {
                assert!(super::super::lookup(dll, sym).is_some(), "{dll}!{sym}");
            }
        }

        let mut word64 = 0x0102_0304_0506_0708u64;
        let ptr = (&raw mut word64).cast::<c_void>();
        assert_eq!(WaitOnAddress(std::ptr::null(), ptr, 4, 0), 0);
        assert_eq!(super::super::GetLastError(), ERROR_INVALID_PARAMETER);
        assert_eq!(WaitOnAddress(ptr, std::ptr::null(), 4, 0), 0);
        assert_eq!(super::super::GetLastError(), ERROR_INVALID_PARAMETER);
        for bad_size in [0usize, 3, 5, 16] {
            assert_eq!(WaitOnAddress(ptr, ptr, bad_size, 0), 0);
            assert_eq!(super::super::GetLastError(), ERROR_INVALID_PARAMETER);
        }

        // Immediate mismatch returns 1 across 1, 2, 4, and 8 bytes, plus unaligned 4-byte reads;
        // matching values with ms == 0 or a short deadline return 0 with ERROR_TIMEOUT.
        let zero64 = 0u64;
        for size in [1usize, 2, 4, 8] {
            assert_eq!(WaitOnAddress(ptr, (&raw const zero64).cast(), size, 5_000), 1);
            assert_eq!(WaitOnAddress(ptr, ptr, size, 0), 0);
            assert_eq!(super::super::GetLastError(), ERROR_TIMEOUT);
        }
        let unaligned = (&raw const word64).cast::<u8>().wrapping_add(1).cast::<c_void>();
        assert_eq!(WaitOnAddress(unaligned, (&raw const zero64).cast(), 4, 5_000), 1);
        assert_eq!(WaitOnAddress(ptr, ptr, 8, 25), 0);
        assert_eq!(super::super::GetLastError(), ERROR_TIMEOUT);

        // Two distinct addresses in the same stripe: WakeByAddressSingle(a) wakes exactly one `a`
        // waiter, WakeByAddressAll(a) wakes the remaining `a` waiters without waking `b`, and
        // WakeByAddressSingle(b) then wakes `b`.
        let buf = Box::leak(Box::new([0u64; (ADDR_STRIPES / 2) + 2]));
        let a_addr = buf.as_mut_ptr() as usize;
        let b_addr = a_addr + (ADDR_STRIPES << 2);
        assert!(std::ptr::eq(addr_stripe(a_addr), addr_stripe(b_addr)));

        let spawn_waiter = |target: usize, size: usize| {
            let (tx, rx) = mpsc::channel();
            let handle = std::thread::spawn(move || {
                let zero = 0u64;
                let r = WaitOnAddress(target as *const c_void, (&raw const zero).cast(), size, 5_000);
                tx.send(r).unwrap();
            });
            (rx, handle)
        };
        let (rx_a1, h_a1) = spawn_waiter(a_addr, 1);
        let (rx_a2, h_a2) = spawn_waiter(a_addr, 1);
        let (rx_a3, h_a3) = spawn_waiter(a_addr, 1);
        let (rx_b, h_b) = spawn_waiter(b_addr, 8);

        let deadline = Instant::now() + Duration::from_secs(1);
        while super::super::lock(addr_stripe(a_addr)).len() < 4 && Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(2));
        }
        assert_eq!(super::super::lock(addr_stripe(a_addr)).len(), 4);

        // SAFETY: `a_addr` points to the first byte of `buf`.
        unsafe { (&*(a_addr as *const AtomicU8)).store(1, Ordering::SeqCst) };
        WakeByAddressSingle(a_addr as *mut c_void);
        assert_eq!(rx_a1.recv_timeout(Duration::from_secs(1)).unwrap(), 1);
        assert!(rx_a2.recv_timeout(Duration::from_millis(30)).is_err());
        assert!(rx_b.recv_timeout(Duration::from_millis(10)).is_err());

        WakeByAddressAll(a_addr as *mut c_void);
        assert_eq!(rx_a2.recv_timeout(Duration::from_secs(1)).unwrap(), 1);
        assert_eq!(rx_a3.recv_timeout(Duration::from_secs(1)).unwrap(), 1);
        assert!(rx_b.recv_timeout(Duration::from_millis(30)).is_err());

        // SAFETY: `b_addr` is 8-byte aligned within `buf`.
        unsafe { (&*(b_addr as *const AtomicU64)).store(9, Ordering::SeqCst) };
        WakeByAddressSingle(b_addr as *mut c_void);
        assert_eq!(rx_b.recv_timeout(Duration::from_secs(1)).unwrap(), 1);

        for h in [h_a1, h_a2, h_a3, h_b] {
            h.join().unwrap();
        }
    }
}
