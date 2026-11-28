//! `VICTORY_LOOKUP_EXIT=1`: a way out of a title lookup that never ends (#456).
//!
//! The title keeps a hash table of pooled objects by handle (pool slot, pool id and the slot's
//! generation). One of its job functions looks objects up there with a probe that has no "not
//! there" exit: on a miss in one group of sixteen entries it goes to the next group, for good.
//! That is fine while every object it is given is in the table. Here the title sometimes frees
//! an object while such a job still holds it; the handle rebuilt from the freed slot carries the
//! next generation, is in no group, and the job thread spins. Another job then waits for the
//! lookups to finish before it empties the table, the main thread waits for that job, and the
//! title stops: on the loading screen when a save is loaded, or some minutes into the world.
//!
//! What orders the free behind the lookups on the console is not known (guess: timing the host
//! does not reproduce). Until it is, this gives the probe the exit it lacks. Before the title
//! runs, the first byte of the probe's miss path becomes an `int3`. Its handler does what the two
//! instructions there did (count the group up, go round again), unless the group just probed has
//! an empty entry: then the function returns through its own epilogue, the way it does for an
//! object with nothing to draw.
//!
//! Guesses, and what they rest on (`docs/native-host.md`, "A title lookup that never ends"):
//!
//! * A group with an empty control byte (0x80) ends a probe: a key that is in the table is never
//!   beyond one. The title's insert takes the first control byte with bit 7 set that it meets on
//!   the same probe sequence, and the job that empties the table sets every control byte to 0x80.
//!   Premise, from the dumps only: nothing removes single entries (a freed object's entry stays,
//!   with its old generation), so no other bit-7 value was seen. Only 0x80 ends the probe here:
//!   if the title does mark removed entries with another value, the probe goes on past them as
//!   the title's own code does.
//! * Read from the function, not traced: at the miss path `eax` is the group number, masked at
//!   the loop head, and `rdx` the table; the loop does not move `rsp`; the loop head's first
//!   instruction sets the flags again before anything reads them; the epilogue sets no return
//!   value and the one caller on the stuck chain reads none.
//! * Not checked: what else the early return leaves out for that object in that frame.
//!
//! Off by default and specific to one build of the title: the three image offsets below are
//! where that build has the miss path, the loop head and the epilogue, and nothing is patched
//! unless the miss path is the `inc eax` and the jump back to the loop head expected there.

use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

use victory_platform::mem::{self, Protect};

use super::seh::context::{Context, EXCEPTION_BREAKPOINT, ExceptionRecord, RAX, RCX, RDX};

/// The probe's miss path: `inc eax`, then a jump back to [`LOOP`] (seven bytes).
const MISS: usize = 0x50e_b9d1;
/// The head of the probe loop, which masks the group number and compares its control bytes.
const LOOP: usize = 0x50e_b80f;
/// The function's epilogue.
const RETURN: usize = 0x50e_b910;
/// One group: sixteen control bytes, then sixteen eight-byte entries.
const GROUP: usize = 0x90;
/// The control byte of an entry never used since the table was last emptied.
const EMPTY: u8 = 0x80;

/// The image base once the miss path is patched, else 0.
static ARMED: AtomicUsize = AtomicUsize::new(0);
/// Lookups that returned through the exit, and miss paths that went on to the next group.
static EXITS: AtomicU64 = AtomicU64::new(0);
static STEPS: AtomicU64 = AtomicU64::new(0);

fn switch(value: Option<&str>) -> bool {
    value.is_some_and(|v| v.trim() == "1")
}

fn enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| switch(std::env::var("VICTORY_LOOKUP_EXIT").ok().as_deref()))
}

/// Whether `code`, the seven bytes at `base + MISS`, is `inc eax` and a near jump to
/// `base + LOOP`: the miss path this patch stands in for.
fn is_miss_path(code: &[u8], base: usize) -> bool {
    let [0xff, 0xc0, 0xe9, d0, d1, d2, d3] = *code else {
        return false;
    };
    let after = (base + MISS + 7) as i64;
    after + i64::from(i32::from_le_bytes([d0, d1, d2, d3])) == (base + LOOP) as i64
}

/// Puts the `int3` at the miss path of the image at `base` (`len` bytes), if the code there is
/// the one expected. The page's protection is left as it was.
///
/// # Safety
/// `base` and `len` are a mapped image no thread is running yet.
unsafe fn plant(base: usize, len: usize) -> Result<(), &'static str> {
    if len < MISS + 7 {
        return Err("the image is too small for this build's lookup");
    }
    let at = (base + MISS) as *mut u8;
    // SAFETY: seven bytes inside the mapped image.
    if !is_miss_path(unsafe { std::slice::from_raw_parts(at, 7) }, base) {
        return Err("the code there is not this build's lookup");
    }
    // Only the first of the seven bytes is written: one page.
    let size = mem::page_size();
    let page = ((base + MISS) & !(size - 1)) as *mut u8;
    // SAFETY: a whole page of the mapped image, opened for one byte.
    let Ok(old) = (unsafe { mem::protect(page, size, Protect::READ_WRITE) }) else {
        return Err("the code page could not be opened");
    };
    // SAFETY: the page is writable now, and gets back the protection it had. Left writable and
    // not executable it would fault the title at its first instruction there, with nothing in
    // the log to say why: stop here instead.
    unsafe {
        at.write(0xcc);
        if let Err(e) = mem::protect(page, size, old) {
            panic!("VICTORY_LOOKUP_EXIT: the code page could not be closed again: {e}");
        }
        mem::flush_icache(page, size);
    }
    Ok(())
}

/// Patches the miss path of the image at `base` (`len` bytes) if the switch is set and the code
/// there is the one expected. Call it before any title code runs: Rosetta translates what it has
/// not yet run from the bytes it then finds.
pub fn arm(base: usize, len: usize) {
    if !enabled() {
        return;
    }
    // SAFETY: the caller's image, before its first thread.
    match unsafe { plant(base, len) } {
        Ok(()) => {
            ARMED.store(base, Ordering::Relaxed);
            tracing::info!("VICTORY_LOOKUP_EXIT: armed at image+{MISS:#x}");
        }
        Err(why) => tracing::warn!("VICTORY_LOOKUP_EXIT: {why} (image+{MISS:#x}); not armed"),
    }
}

/// What the miss path does for a group with these control bytes.
#[derive(Debug, PartialEq, Eq)]
enum Step {
    /// An empty entry in the group: the probe ends without its key.
    Return,
    /// No empty entry: the key may be in the next group.
    NextGroup,
}

fn step(control: [u8; 16]) -> Step {
    if control.contains(&EMPTY) {
        Step::Return
    } else {
        Step::NextGroup
    }
}

/// The breakpoint at `base + MISS` in `ctx`: sets the registers the two replaced instructions
/// would have set, or returns from the function. `control` reads a group's sixteen control bytes.
/// `None`, with `ctx` untouched, for any other address.
fn handle(
    ctx: &mut Context,
    base: usize,
    control: impl Fn(usize) -> Option<[u8; 16]>,
) -> Option<Step> {
    if ctx.rip as usize != base + MISS {
        return None;
    }
    // `eax` is the group number, already masked; `rdx` the table.
    let group = (ctx.gpr[RDX] as usize).wrapping_add(ctx.gpr[RAX] as u32 as usize * GROUP);
    // Control bytes that cannot be read: the title's own next step, never the exit.
    let next = control(group).map_or(Step::NextGroup, step);
    match next {
        Step::Return => ctx.rip = (base + RETURN) as u64,
        Step::NextGroup => {
            // `inc eax` clears the upper half of `rax`. Its flags are not set here: the loop
            // head's first instruction sets them all again.
            ctx.gpr[RAX] = u64::from((ctx.gpr[RAX] as u32).wrapping_add(1));
            ctx.rip = (base + LOOP) as u64;
        }
    }
    Some(next)
}

fn control_at(group: usize) -> Option<[u8; 16]> {
    let (a, b) = (
        super::dump::read_checked(group)?,
        super::dump::read_checked(group + 8)?,
    );
    let mut out = [0; 16];
    out[..8].copy_from_slice(&a.to_le_bytes());
    out[8..].copy_from_slice(&b.to_le_bytes());
    Some(out)
}

/// The title's breakpoint exception `rec` with registers `ctx`: true if it was the patched miss
/// path, and `ctx` then says where the thread goes on. Costs one atomic load when not armed.
pub fn on_breakpoint(ctx: &mut Context, rec: &ExceptionRecord) -> bool {
    let base = ARMED.load(Ordering::Relaxed);
    if base == 0 || rec.code != EXCEPTION_BREAKPOINT {
        return false;
    }
    match handle(ctx, base, control_at) {
        None => false,
        Some(Step::NextGroup) => {
            STEPS.fetch_add(1, Ordering::Relaxed);
            true
        }
        Some(Step::Return) => {
            let n = EXITS.fetch_add(1, Ordering::Relaxed) + 1;
            // The first few, then at each power of two: a stuck lookup used to be one handle for
            // good.
            if n <= 8 || n.is_power_of_two() {
                tracing::warn!(
                    handle = format_args!("{:#x}", ctx.gpr[RCX] as u32),
                    exits = n,
                    steps = STEPS.load(Ordering::Relaxed),
                    "VICTORY_LOOKUP_EXIT: a lookup met an empty entry before its handle; it returns"
                );
            }
            true
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const BASE: usize = 0x1_4000_0000;

    fn miss_path(base: usize, target: usize) -> [u8; 7] {
        let rel = (target as i64 - (base + MISS + 7) as i64) as i32;
        let d = rel.to_le_bytes();
        [0xff, 0xc0, 0xe9, d[0], d[1], d[2], d[3]]
    }

    #[test]
    fn only_the_expected_miss_path_is_recognised() {
        assert!(is_miss_path(&miss_path(BASE, BASE + LOOP), BASE));
        // A jump elsewhere, another instruction, another length: another build.
        assert!(!is_miss_path(&miss_path(BASE, BASE + LOOP + 1), BASE));
        let mut other = miss_path(BASE, BASE + LOOP);
        other[1] = 0xc1;
        assert!(!is_miss_path(&other, BASE));
        assert!(!is_miss_path(&miss_path(BASE, BASE + LOOP)[..6], BASE));
        // The jump is relative: the same bytes are the miss path wherever the image is mapped.
        assert!(is_miss_path(&miss_path(BASE, BASE + LOOP), BASE + 0x1_0000));
    }

    #[test]
    fn the_switch_is_on_only_for_1() {
        assert!(switch(Some("1")) && switch(Some(" 1 ")));
        assert!(!switch(None) && !switch(Some("0")) && !switch(Some("")) && !switch(Some("yes")));
    }

    #[test]
    fn a_group_with_an_empty_entry_ends_the_probe() {
        assert_eq!(step([0x12; 16]), Step::NextGroup);
        let mut one_empty = [0x7f; 16];
        one_empty[9] = EMPTY;
        assert_eq!(step(one_empty), Step::Return);
        assert_eq!(step([EMPTY; 16]), Step::Return);
        // Another value with bit 7 set is not the emptied table's: the probe goes on.
        let mut other = [0x7f; 16];
        other[3] = 0xfe;
        assert_eq!(step(other), Step::NextGroup);
    }

    #[test]
    fn a_missing_key_leaves_the_loop_at_the_first_group_with_an_empty_entry() {
        // Three groups: full, full, one empty entry. The key is in none of them.
        let mut table = vec![0x11u8; 3 * GROUP];
        table[2 * GROUP + 5] = EMPTY;
        let at = table.as_ptr() as usize;
        let control = |group: usize| {
            let k = group - at;
            Some(<[u8; 16]>::try_from(&table[k..k + 16]).unwrap())
        };
        let mut ctx = Context::zeroed();
        ctx.gpr[RDX] = at as u64;
        // Upper half set: `inc eax` clears it.
        ctx.gpr[RAX] = 0xdead_beef_0000_0000;
        ctx.gpr[RCX] = 0x88_0eab;
        for expected in [1u64, 2] {
            ctx.rip = (BASE + MISS) as u64;
            assert_eq!(handle(&mut ctx, BASE, control), Some(Step::NextGroup));
            assert_eq!(ctx.rip as usize, BASE + LOOP);
            assert_eq!(ctx.gpr[RAX], expected);
        }
        ctx.rip = (BASE + MISS) as u64;
        assert_eq!(handle(&mut ctx, BASE, control), Some(Step::Return));
        assert_eq!(ctx.rip as usize, BASE + RETURN);
        // The group number and the key are as the loop left them.
        assert_eq!((ctx.gpr[RAX], ctx.gpr[RCX]), (2, 0x88_0eab));
    }

    #[test]
    fn other_breakpoints_are_left_alone_and_unreadable_tables_take_the_titles_own_step() {
        let mut ctx = Context::zeroed();
        ctx.rip = (BASE + MISS + 1) as u64;
        ctx.gpr[RAX] = 7;
        assert_eq!(handle(&mut ctx, BASE, |_| Some([EMPTY; 16])), None);
        assert_eq!((ctx.rip as usize, ctx.gpr[RAX]), (BASE + MISS + 1, 7));
        // Control bytes that cannot be read: the title's own next step, never the exit.
        ctx.rip = (BASE + MISS) as u64;
        assert_eq!(handle(&mut ctx, BASE, |_| None), Some(Step::NextGroup));
        assert_eq!((ctx.rip as usize, ctx.gpr[RAX]), (BASE + LOOP, 8));
        // The group number wraps in 32 bits, as `inc eax` does.
        ctx.rip = (BASE + MISS) as u64;
        ctx.gpr[RAX] = 0xffff_ffff;
        assert_eq!(handle(&mut ctx, BASE, |_| None), Some(Step::NextGroup));
        assert_eq!(ctx.gpr[RAX], 0);
    }

    #[test]
    fn nothing_is_handled_before_the_patch_is_in() {
        // No test arms the process-wide patch.
        let mut ctx = Context::zeroed();
        ctx.rip = (BASE + MISS) as u64;
        ctx.gpr[RAX] = 3;
        let rec = ExceptionRecord::new(EXCEPTION_BREAKPOINT, 0, BASE + MISS);
        assert!(!on_breakpoint(&mut ctx, &rec));
        assert_eq!((ctx.rip as usize, ctx.gpr[RAX]), (BASE + MISS, 3));
    }

    #[test]
    fn control_bytes_are_read_from_memory_and_unmapped_memory_reads_as_none() {
        let mut group = [0u8; 16];
        for (i, b) in group.iter_mut().enumerate() {
            *b = 0x70 + i as u8;
        }
        assert_eq!(control_at(group.as_ptr() as usize), Some(group));
        // Nothing is mapped in the first page.
        assert_eq!(control_at(0x10), None);
    }

    #[test]
    fn the_patch_changes_one_byte_and_keeps_the_pages_protection() {
        // A stand-in image: only the page with the miss path is ever touched.
        let len = (MISS + 7 + 0xffff) & !0xffff;
        // SAFETY: a fresh anonymous mapping of our own, released at the end.
        unsafe {
            let base = mem::alloc(None, len, Protect::READ_WRITE).unwrap() as usize;
            let at = (base + MISS) as *mut u8;
            let size = mem::page_size();
            let page = ((base + MISS) & !(size - 1)) as *mut u8;

            // Not the expected code: nothing is written.
            assert!(plant(base, len).is_err());
            assert_eq!(at.read(), 0);
            // Too small an image for this build.
            assert!(plant(base, MISS + 6).is_err());

            let code = miss_path(base, base + LOOP);
            std::ptr::copy_nonoverlapping(code.as_ptr(), at, 7);
            mem::protect(page, size, Protect::READ_EXECUTE).unwrap();
            assert!(plant(base, len).is_ok());
            assert_eq!(at.read(), 0xcc);
            assert_eq!(std::slice::from_raw_parts(at.add(1), 6), &code[1..]);
            assert_eq!(
                mem::query(base + MISS).unwrap().protect,
                Protect::READ_EXECUTE
            );
            // Patched once: the code there is no longer the miss path.
            assert!(plant(base, len).is_err());

            mem::release(base as *mut u8, len).unwrap();
        }
    }
}
