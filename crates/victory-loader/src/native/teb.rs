//! The thread environment block title code expects at `gs:`.
//!
//! Title code reads `gs:[0x30]` (TEB self), `gs:[0x58]` (the thread's static
//! TLS array), `gs:[0x60]` (the PEB), `gs:[0x10]` (stack limit) and so on. A
//! macOS x86-64 process, Rosetta included, keeps pthread's thread-specific
//! data at `gs:` instead, and slots 11 and 12 of it (our `0x58` and `0x60`)
//! belong to libSystem's own keys. So each thread has two GS bases: the
//! host's, while host code runs, and this TEB, while title code runs. The
//! bridge (`bridge.rs`) swaps them with the `thread_fast_set_cthread_self`
//! machdep call at every crossing, which Rosetta supports (about 190 ns
//! each way on an M-series Mac). `docs/native-host.md` has the numbers and
//! the alternatives.

use std::alloc::{Layout, alloc_zeroed, dealloc};
use std::cell::RefCell;
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

use super::image::TlsTemplate;

pub const TEB_SIZE: usize = 0x2000;
pub const OFF_STACK_BASE: usize = 0x08;
pub const OFF_STACK_LIMIT: usize = 0x10;
pub const OFF_SELF: usize = 0x30;
pub const OFF_PROCESS_ID: usize = 0x40;
pub const OFF_THREAD_ID: usize = 0x48;
pub const OFF_TLS_ARRAY: usize = 0x58;
pub const OFF_PEB: usize = 0x60;
pub const OFF_LAST_ERROR: usize = 0x68;
pub const OFF_TLS_SLOTS: usize = 0x1480;
pub const TLS_SLOT_COUNT: usize = 64;
/// Ours, in the part of the block Windows leaves unused: the thread's host
/// GS base, which the bridge swaps back to.
pub const OFF_HOST_TSD: usize = 0x1F00;
/// Ours too: what the bridge writes while a thread is in a host call, for
/// the thread dump (`dump.rs`): the target (0 when in title code), the
/// title's return address, the stub's context word and the bridge's `rbp`.
pub const OFF_IN_HOST: usize = 0x1F10;
pub const OFF_IN_HOST_RET: usize = 0x1F18;
pub const OFF_IN_HOST_CTX: usize = 0x1F20;
pub const OFF_IN_HOST_RBP: usize = 0x1F28;
/// Modules with static TLS one thread can hold (the array's length).
pub const MAX_TLS_MODULES: usize = 64;

core::arch::global_asm!(
    // GetLastError() and SetLastError(code), Windows ABI, for title code: both touch only the
    // TEB at the caller's GS base, so they run with no bridge swap and cost the call itself
    // instead of the ~370 ns GS swap each way. Bound with `bridge::declare_direct`.
    ".p2align 4",
    ".globl _victory_GetLastError",
    "_victory_GetLastError:",
    "mov eax, gs:[{last_error}]",
    "ret",
    ".p2align 4",
    ".globl _victory_SetLastError",
    "_victory_SetLastError:",
    "mov gs:[{last_error}], ecx",
    "ret",
    // TlsGetValue(index) / TlsSetValue(index, value), same story: the 64 slots at
    // gs:[OFF_TLS_SLOTS] plus LastError. An out-of-range index sets ERROR_INVALID_PARAMETER
    // (87) and fails, as the bridged implementation does.
    ".p2align 4",
    ".globl _victory_TlsGetValue",
    "_victory_TlsGetValue:",
    "cmp ecx, {slots}",
    "jae 1f",
    "mov dword ptr gs:[{last_error}], 0",
    "mov rax, gs:[rcx*8 + {tls_slots}]",
    "ret",
    "1:",
    "mov dword ptr gs:[{last_error}], 87",
    "xor eax, eax",
    "ret",
    ".p2align 4",
    ".globl _victory_TlsSetValue",
    "_victory_TlsSetValue:",
    "cmp ecx, {slots}",
    "jae 1f",
    "mov gs:[rcx*8 + {tls_slots}], rdx",
    "mov dword ptr gs:[{last_error}], 0",
    "mov eax, 1",
    "ret",
    "1:",
    "mov dword ptr gs:[{last_error}], 87",
    "xor eax, eax",
    "ret",
    // GetCurrentThreadId() / GetCurrentProcessId(): the ids the TEB was made with
    // (gs:[OFF_THREAD_ID] / gs:[OFF_PROCESS_ID]), so no swap needed either.
    ".p2align 4",
    ".globl _victory_GetCurrentThreadId",
    "_victory_GetCurrentThreadId:",
    "mov eax, gs:[{thread_id}]",
    "ret",
    ".p2align 4",
    ".globl _victory_GetCurrentProcessId",
    "_victory_GetCurrentProcessId:",
    "mov eax, gs:[{process_id}]",
    "ret",
    // GetCurrentProcess() / GetCurrentThread(): the constant pseudo-handles -1 and -2.
    ".p2align 4",
    ".globl _victory_GetCurrentProcess",
    "_victory_GetCurrentProcess:",
    "mov rax, -1",
    "ret",
    ".p2align 4",
    ".globl _victory_GetCurrentThread",
    "_victory_GetCurrentThread:",
    "mov rax, -2",
    "ret",
    last_error = const OFF_LAST_ERROR,
    tls_slots = const OFF_TLS_SLOTS,
    slots = const TLS_SLOT_COUNT,
    thread_id = const OFF_THREAD_ID,
    process_id = const OFF_PROCESS_ID,
);

unsafe extern "win64" {
    fn victory_GetLastError() -> u32;
    fn victory_SetLastError(code: u32);
    fn victory_TlsGetValue(index: u32) -> usize;
    fn victory_TlsSetValue(index: u32, value: usize) -> u32;
    fn victory_GetCurrentThreadId() -> u32;
    fn victory_GetCurrentProcessId() -> u32;
    fn victory_GetCurrentProcess() -> isize;
    fn victory_GetCurrentThread() -> isize;
}

/// The address title code calls for `GetLastError` with no bridge: reads `gs:[OFF_LAST_ERROR]`
/// of the TEB its GS base points at.
pub fn direct_get_last_error_address() -> usize {
    victory_GetLastError as *const () as usize
}

/// The address title code calls for `SetLastError` with no bridge: writes `ecx` to
/// `gs:[OFF_LAST_ERROR]` of that TEB.
pub fn direct_set_last_error_address() -> usize {
    victory_SetLastError as *const () as usize
}

/// The address title code calls for `TlsGetValue` with no bridge: the slot at
/// `gs:[OFF_TLS_SLOTS + 8*index]`, or null with `LastError` 87 when `index >= TLS_SLOT_COUNT`.
pub fn direct_tls_get_value_address() -> usize {
    victory_TlsGetValue as *const () as usize
}

/// The address title code calls for `TlsSetValue` with no bridge: writes `rdx` to the slot
/// above, or fails with `LastError` 87 when `index >= TLS_SLOT_COUNT`.
pub fn direct_tls_set_value_address() -> usize {
    victory_TlsSetValue as *const () as usize
}

/// The address title code calls for `GetCurrentThreadId` with no bridge: `gs:[OFF_THREAD_ID]`.
pub fn direct_get_current_thread_id_address() -> usize {
    victory_GetCurrentThreadId as *const () as usize
}

/// The address title code calls for `GetCurrentProcessId` with no bridge: `gs:[OFF_PROCESS_ID]`.
pub fn direct_get_current_process_id_address() -> usize {
    victory_GetCurrentProcessId as *const () as usize
}

/// The address title code calls for `GetCurrentProcess` with no bridge: always the pseudo-handle -1.
pub fn direct_get_current_process_address() -> usize {
    victory_GetCurrentProcess as *const () as usize
}

/// The address title code calls for `GetCurrentThread` with no bridge: always the pseudo-handle -2.
pub fn direct_get_current_thread_address() -> usize {
    victory_GetCurrentThread as *const () as usize
}
const PEB_SIZE: usize = 0x800;
const OFF_PEB_IMAGE_BASE: usize = 0x10;

struct LiveThread {
    teb: usize,
    port: super::threadctl::ThreadPort,
    tid: u32,
    tls_array: usize,
    /// TLS blocks allocated after this thread started (for late-loaded DLLs),
    /// freed when its TEB is dropped.
    late_blocks: Vec<(usize, Layout)>,
}

/// Every live thread with a TEB. Lock order with `TLS_MODULES`: always lock
/// `TLS_MODULES` before `LIVE`.
static LIVE: Mutex<Vec<LiveThread>> = Mutex::new(Vec::new());

/// The threads that have a TEB now (for the thread dump).
pub fn live_threads() -> Vec<(usize, super::threadctl::ThreadPort, u32)> {
    LIVE.lock()
        .unwrap_or_else(|e| e.into_inner())
        .iter()
        .map(|t| (t.teb, t.port, t.tid))
        .collect()
}

static TLS_MODULES: Mutex<Vec<TlsTemplate>> = Mutex::new(Vec::new());
static PEB: AtomicUsize = AtomicUsize::new(0);

fn alloc_tls_block(t: &TlsTemplate) -> (*mut u8, Layout) {
    let layout = Layout::from_size_align(t.block_len().max(1), 64).expect("layout");
    // SAFETY: a non-zero layout; the template is in a mapped image.
    let block = unsafe { alloc_zeroed(layout) };
    assert!(!block.is_null(), "out of memory for a TLS block");
    let init = t.end.saturating_sub(t.start);
    if init > 0 {
        // SAFETY: the template's initial data is `init` readable bytes.
        unsafe { std::ptr::copy_nonoverlapping(t.start as *const u8, block, init) };
    }
    (block, layout)
}

/// Gives a mapped image its TLS index, writes it where the image reads
/// `_tls_index`, populates a block in every already-live thread's TLS array,
/// and arranges for every thread started from now on to get a block. Call
/// before [`super::NativeImage::finalize`] makes that word read-only.
pub fn register_tls(template: TlsTemplate) -> u32 {
    let mut modules = TLS_MODULES.lock().unwrap_or_else(|e| e.into_inner());
    let index = modules.len();
    assert!(index < MAX_TLS_MODULES, "too many modules with static TLS");
    modules.push(template);
    // SAFETY: the template's index word is in the image the caller mapped.
    unsafe { (template.index_at as *mut u32).write(index as u32) };
    let mut live = LIVE.lock().unwrap_or_else(|e| e.into_inner());
    for thread in live.iter_mut() {
        let (block, layout) = alloc_tls_block(&template);
        // SAFETY: `tls_array` has `MAX_TLS_MODULES` entries and stays alive until
        // `ThreadTeb::drop` removes this entry under `LIVE`.
        unsafe {
            let slot = (thread.tls_array as *const AtomicUsize).add(index);
            (*slot).store(block as usize, Ordering::Release);
        }
        thread.late_blocks.push((block as usize, layout));
    }
    index as u32
}

fn peb() -> usize {
    let p = PEB.load(Ordering::Acquire);
    if p != 0 {
        return p;
    }
    let layout = Layout::from_size_align(PEB_SIZE, 0x1000).expect("layout");
    // SAFETY: a non-zero layout.
    let fresh = unsafe { alloc_zeroed(layout) } as usize;
    match PEB.compare_exchange(0, fresh, Ordering::AcqRel, Ordering::Acquire) {
        Ok(_) => fresh,
        Err(winner) => {
            // SAFETY: ours, never shared.
            unsafe { dealloc(fresh as *mut u8, layout) };
            winner
        }
    }
}

/// Records the title's main image in the PEB (`ImageBaseAddress`).
pub fn set_image_base(base: usize) {
    // SAFETY: inside our PEB block.
    unsafe { ((peb() + OFF_PEB_IMAGE_BASE) as *mut usize).write(base) };
}

struct ThreadTeb {
    teb: *mut u8,
    tls_array: Box<[AtomicUsize; MAX_TLS_MODULES]>,
    blocks: Vec<(*mut u8, Layout)>,
}

impl ThreadTeb {
    /// Builds the calling thread's TEB. Must run with the host GS base.
    fn new() -> ThreadTeb {
        super::seh::install();
        let layout = Layout::from_size_align(TEB_SIZE, 0x1000).expect("layout");
        // SAFETY: a non-zero layout.
        let teb = unsafe { alloc_zeroed(layout) };
        assert!(!teb.is_null(), "out of memory for a TEB");
        let tls_array: Box<[AtomicUsize; MAX_TLS_MODULES]> =
            Box::new(std::array::from_fn(|_| AtomicUsize::new(0)));
        let mut blocks = Vec::new();
        let modules = TLS_MODULES.lock().unwrap_or_else(|e| e.into_inner());
        for (i, t) in modules.iter().enumerate() {
            let (block, layout) = alloc_tls_block(t);
            tls_array[i].store(block as usize, Ordering::Relaxed);
            blocks.push((block, layout));
        }
        let (top, size) = stack_bounds();
        let put = |off: usize, v: usize| {
            // SAFETY: inside the TEB block.
            unsafe { teb.add(off).cast::<usize>().write(v) }
        };
        put(OFF_STACK_BASE, top);
        put(OFF_STACK_LIMIT, top - size);
        put(OFF_SELF, teb as usize);
        put(OFF_PROCESS_ID, std::process::id() as usize);
        put(
            OFF_THREAD_ID,
            victory_platform::thread::current_id() as usize,
        );
        put(OFF_TLS_ARRAY, tls_array.as_ptr() as usize);
        put(OFF_PEB, peb());
        put(OFF_HOST_TSD, host_tsd_base());
        LIVE.lock()
            .unwrap_or_else(|e| e.into_inner())
            .push(LiveThread {
                teb: teb as usize,
                port: super::threadctl::ThreadPort::current(),
                tid: victory_platform::thread::current_id(),
                tls_array: tls_array.as_ptr() as usize,
                late_blocks: Vec::new(),
            });
        drop(modules);
        ThreadTeb {
            teb,
            tls_array,
            blocks,
        }
    }
}

impl Drop for ThreadTeb {
    fn drop(&mut self) {
        let _ = &self.tls_array;
        let late_blocks = {
            let mut live = LIVE.lock().unwrap_or_else(|e| e.into_inner());
            if let Some(pos) = live.iter().position(|t| t.teb == self.teb as usize) {
                live.swap_remove(pos).late_blocks
            } else {
                Vec::new()
            }
        };
        for (block, layout) in late_blocks {
            // SAFETY: allocated by `alloc_tls_block` in `register_tls`.
            unsafe { dealloc(block as *mut u8, layout) };
        }
        for (block, layout) in self.blocks.drain(..) {
            // SAFETY: allocated in `new` with this layout.
            unsafe { dealloc(block, layout) };
        }
        let layout = Layout::from_size_align(TEB_SIZE, 0x1000).expect("layout");
        // SAFETY: allocated in `new`.
        unsafe { dealloc(self.teb, layout) };
    }
}

thread_local! {
    static CURRENT: RefCell<Option<ThreadTeb>> = const { RefCell::new(None) };
}

/// The calling thread's TEB, made on first use. Call it from host code
/// (the host GS base).
pub fn current() -> *mut u8 {
    CURRENT.with(|c| c.borrow_mut().get_or_insert_with(ThreadTeb::new).teb)
}

/// `GetLastError`'s variable for the calling thread.
pub fn last_error_slot() -> *mut u32 {
    // SAFETY: inside the TEB block.
    unsafe { current().add(OFF_LAST_ERROR).cast() }
}

/// Zeroes TLS slot `index` (of the 64 `TlsAlloc` hands out) in every live thread, as `TlsFree`
/// does on Windows: a slot handed out again reads 0 everywhere.
pub fn clear_tls_slot(index: usize) {
    assert!(index < TLS_SLOT_COUNT);
    for thread in LIVE.lock().unwrap_or_else(|e| e.into_inner()).iter() {
        let slot = (thread.teb + OFF_TLS_SLOTS + index * 8) as *const AtomicUsize;
        // SAFETY: inside a TEB that stays alive while it is in `LIVE`, which is locked.
        unsafe { (*slot).store(0, Ordering::Relaxed) };
    }
}

/// The 64 slots `TlsAlloc` hands out, for the calling thread.
pub fn tls_slots() -> *mut usize {
    // SAFETY: inside the TEB block.
    unsafe { current().add(OFF_TLS_SLOTS).cast() }
}

unsafe extern "C" {
    fn pthread_self() -> usize;
    fn pthread_get_stackaddr_np(thread: usize) -> usize;
    fn pthread_get_stacksize_np(thread: usize) -> usize;
}

/// Top of the stack and its size.
fn stack_bounds() -> (usize, usize) {
    // SAFETY: queries about the calling thread.
    unsafe {
        let me = pthread_self();
        (pthread_get_stackaddr_np(me), pthread_get_stacksize_np(me))
    }
}

/// Reads `gs:[8 * slot]` of the host's thread-specific data.
fn gs_slot(slot: usize) -> usize {
    let value: usize;
    // SAFETY: slots below 16 exist in every pthread's TSD.
    unsafe {
        std::arch::asm!("mov {}, gs:[{}]", out(reg) value, in(reg) slot * 8, options(nostack, readonly))
    };
    value
}

/// The calling thread's host GS base: the address of the pthread's
/// thread-specific data, whose first word is `pthread_self()`.
///
/// The TSD array sits inside the pthread structure at an offset that differs
/// between macOS versions (and the structure holds other words equal to
/// itself), so each candidate has to match the first words read through GS.
fn host_tsd_base() -> usize {
    let me = gs_slot(0);
    (0..0x400)
        .step_by(8)
        .map(|off| me + off)
        // SAFETY: inside the pthread structure, which is larger than 0x400 bytes.
        .find(|&p| (0..12).all(|i| unsafe { (p as *const usize).add(i).read() } == gs_slot(i)))
        .expect("cannot locate the pthread TSD base")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn late_register_tls_populates_already_live_threads() {
        let teb = current();
        let init: [u8; 8] = [0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88];
        let mut tls_index: u32 = u32::MAX;
        let template = TlsTemplate {
            start: init.as_ptr() as usize,
            end: init.as_ptr() as usize + init.len(),
            zero_fill: 8,
            index_at: (&mut tls_index as *mut u32) as usize,
        };
        let index = register_tls(template);
        assert_eq!(tls_index, index);

        // SAFETY: `teb` is this thread's live TEB and `OFF_TLS_ARRAY` points to its TLS array.
        let block = unsafe {
            let array = *(teb.add(OFF_TLS_ARRAY) as *const *const AtomicUsize);
            (*array.add(index as usize)).load(Ordering::Acquire) as *const u8
        };
        assert!(!block.is_null(), "live thread must receive a TLS block for late-loaded DLL");
        // SAFETY: `block` has `init.len() + zero_fill` bytes allocated by `alloc_tls_block`.
        let bytes = unsafe { std::slice::from_raw_parts(block, 16) };
        assert_eq!(&bytes[..8], &init);
        assert_eq!(&bytes[8..16], &[0u8; 8]);
    }

    #[test]
    fn direct_last_error_reads_and_writes_the_callers_gs_base() {
        // SAFETY: swaps GS to `base` with the same machdep call the bridge uses.
        unsafe fn set_gs(base: usize) {
            #[allow(unused_unsafe)]
            unsafe {
                std::arch::asm!(
                    "mov rdi, {base}",
                    "mov eax, 0x3000003",
                    "syscall",
                    base = in(reg) base,
                    out("rax") _,
                    lateout("rcx") _,
                    lateout("r11") _,
                    options(nostack),
                );
            }
        }
        let mut fake = vec![0u8; TEB_SIZE];
        fake[OFF_LAST_ERROR..OFF_LAST_ERROR + 4].copy_from_slice(&0xBEEFu32.to_le_bytes());
        let fake_base = fake.as_mut_ptr() as usize;
        let host_gs = host_tsd_base();
        struct Restore(usize);
        impl Drop for Restore {
            fn drop(&mut self) {
                // SAFETY: puts this thread's host GS base back.
                unsafe { set_gs(self.0) };
            }
        }
        let _restore = Restore(host_gs);
        // A value in TLS slot 3, to read back through gs:.
        fake[OFF_TLS_SLOTS + 3 * 8..OFF_TLS_SLOTS + 4 * 8].copy_from_slice(&0xABCDusize.to_le_bytes());
        fake[OFF_THREAD_ID..OFF_THREAD_ID + 4].copy_from_slice(&0x7EADu32.to_le_bytes());
        fake[OFF_PROCESS_ID..OFF_PROCESS_ID + 4].copy_from_slice(&0xC0DEu32.to_le_bytes());
        let got;
        let wrote;
        let (tls_ok, tls_bad, tls_wrote, tls_err, tid, pid, hproc, hthread);
        // SAFETY: GS points at `fake`, which outlives the calls, and is restored before any
        // host code (TLS access) runs again.
        unsafe {
            set_gs(fake_base);
            got = victory_GetLastError();
            victory_SetLastError(0x1234);
            wrote = u32::from_le_bytes(fake[OFF_LAST_ERROR..OFF_LAST_ERROR + 4].try_into().unwrap());
            tls_ok = victory_TlsGetValue(3);
            tls_wrote = victory_TlsSetValue(4, 0x5678);
            tls_bad = victory_TlsGetValue(TLS_SLOT_COUNT as u32);
            tls_err = u32::from_le_bytes(fake[OFF_LAST_ERROR..OFF_LAST_ERROR + 4].try_into().unwrap());
            tid = victory_GetCurrentThreadId();
            pid = victory_GetCurrentProcessId();
            hproc = victory_GetCurrentProcess();
            hthread = victory_GetCurrentThread();
            set_gs(host_gs);
        }
        assert_eq!(got, 0xBEEF);
        assert_eq!(wrote, 0x1234);
        assert_eq!(tls_ok, 0xABCD);
        assert_eq!(tls_wrote, 1);
        assert_eq!(
            usize::from_le_bytes(fake[OFF_TLS_SLOTS + 4 * 8..OFF_TLS_SLOTS + 5 * 8].try_into().unwrap()),
            0x5678
        );
        assert_eq!(tls_bad, 0, "an out-of-range index returns null");
        assert_eq!(tls_err, 87, "and sets ERROR_INVALID_PARAMETER");
        assert_eq!(tid, 0x7EAD);
        assert_eq!(pid, 0xC0DE);
        assert_eq!(hproc, -1);
        assert_eq!(hthread, -2);
    }

    #[test]
    fn alloc_tls_block_accepts_zero_init_template_with_null_start() {
        let template = TlsTemplate {
            start: 0,
            end: 0,
            zero_fill: 16,
            index_at: 0,
        };
        let (block, layout) = alloc_tls_block(&template);
        assert!(!block.is_null());
        // SAFETY: `alloc_tls_block` allocated and zeroed `layout.size() >= 16` bytes.
        unsafe {
            assert_eq!(std::slice::from_raw_parts(block, 16), &[0u8; 16]);
            dealloc(block, layout);
        }
    }
}
