//! Opt-in recovery of the title's pool exhaustion traps, ported from the Wine handler.

use super::context::{Context, EXCEPTION_BREAKPOINT, ExceptionRecord};
use super::unwind;
#[cfg(target_arch = "x86_64")]
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};

/// `xor r, r` (`31 /r` or `33 /r`), including a REX prefix that extends
/// both register fields or neither. Memory operands and different registers do not match.
fn zeroes_a_register(behind: &[u8]) -> bool {
    let (rex, rest) = match behind {
        [rex @ 0x40..=0x4f, rest @ ..] => (*rex, rest),
        rest => (0, rest),
    };
    if (rex >> 2 & 1) != (rex & 1) {
        return false;
    }
    matches!(rest, [0x31 | 0x33, modrm, ..] if modrm >> 6 == 3 && (modrm >> 3 & 7) == (modrm & 7))
}

/// Runs on the host dispatcher, never inside the signal handler. Nothing
/// changes unless the switch includes `pool` and the trapped bytes match.
///
/// # Safety
/// The context/record describe the faulting thread; its image stays mapped.
#[cfg(target_arch = "x86_64")]
pub(super) unsafe fn step_over(ctx: &mut Context, rec: &ExceptionRecord) -> bool {
    static POOLS: OnceLock<bool> = OnceLock::new();
    let enabled = *POOLS.get_or_init(|| {
        std::env::var("VICTORY_STEP_OVER")
            .unwrap_or_default()
            .split(',')
            .any(|v| v.trim().eq_ignore_ascii_case("pool"))
    });
    // SAFETY: forwarded from the caller.
    enabled && unsafe { step_over_pool(ctx, rec) }
}

/// # Safety
/// As for `step_over`: the registered image containing the trap remains mapped.
unsafe fn step_over_pool(ctx: &mut Context, rec: &ExceptionRecord) -> bool {
    if rec.code != EXCEPTION_BREAKPOINT || rec.flags & 1 != 0 {
        return false;
    }
    let at = rec.address;
    let Some(last) = at.checked_add(3) else {
        return false;
    };
    let Some(base) = unwind::image_base_of(at) else {
        return false;
    };
    if unwind::image_base_of(last) != Some(base)
        || (ctx.rip != at as u64 && ctx.rip != at as u64 + 1)
    {
        return false;
    }
    // SAFETY: all four bytes lie within the registered, mapped title image.
    let code = unsafe { std::slice::from_raw_parts(at as *const u8, 4) };
    if code[0] != 0xcc || !zeroes_a_register(&code[1..]) {
        return false;
    }
    // signal.rs normalizes RIP to the int3. Leave the following xor to run,
    // producing the NULL return exactly as the title intended.
    ctx.rip = at as u64 + 1;
    static STEPS: AtomicU64 = AtomicU64::new(0);
    let n = STEPS.fetch_add(1, Relaxed) + 1;
    if n <= 3 || n.is_power_of_two() {
        tracing::warn!(
            offset = format_args!("{:#x}", at - base),
            n,
            "stepped over the title's pool breakpoint (VICTORY_STEP_OVER=pool)"
        );
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_same_register_xors_match() {
        for code in [
            &[0x31, 0xc0][..],
            &[0x33, 0xdb],
            &[0x48, 0x31, 0xc0],
            &[0x4d, 0x33, 0xff],
        ] {
            assert!(zeroes_a_register(code));
        }
        for code in [
            &[][..],
            &[0x31],
            &[0x31, 0xc1],
            &[0x31, 0x00],
            &[0x41, 0x31, 0xc0],
            &[0x44, 0x33, 0xc0],
            &[0x90, 0x31, 0xc0],
        ] {
            assert!(!zeroes_a_register(code));
        }
    }

    #[test]
    fn pool_recovery_is_bounded_and_advances_only_rip() {
        let mut code = [0xccu8, 0x48, 0x31, 0xc0, 0xcc];
        let base = code.as_mut_ptr() as usize;
        unwind::add_image(base, code.len(), None);
        let mut ctx = Context::zeroed();
        ctx.rip = base as u64;
        ctx.gpr = [42; 16];
        let mut rec = ExceptionRecord::new(EXCEPTION_BREAKPOINT, 0, base);
        // SAFETY: our registered, live byte array; no title code is executed.
        unsafe {
            assert!(step_over_pool(&mut ctx, &rec));
            assert_eq!(ctx.rip, base as u64 + 1);
            assert_eq!(ctx.gpr, [42; 16]);
            assert!(step_over_pool(&mut ctx, &rec)); // already past int3
            assert_eq!(ctx.rip, base as u64 + 1);
            rec.code = super::super::context::EXCEPTION_ACCESS_VIOLATION;
            assert!(!step_over_pool(&mut ctx, &rec));
            rec.code = EXCEPTION_BREAKPOINT;
            rec.address = base + 4; // not enough bytes inside this image
            ctx.rip = rec.address as u64;
            assert!(!step_over_pool(&mut ctx, &rec));
            rec.address = base;
            ctx.rip = base as u64;
            code[0] = 0x90;
            assert!(!step_over_pool(&mut ctx, &rec));
            unwind::remove_image(base);
            assert!(!step_over_pool(&mut ctx, &rec));
        }
    }
}
