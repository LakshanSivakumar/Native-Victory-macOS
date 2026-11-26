//! Crash-time help from crates the loader does not depend on.

use std::sync::OnceLock;

/// Describes a title address at a crash: whether GPU memory holds it and which buffer views
/// name it (set by the app from the GPU crate). Must not block.
pub static DESCRIBE_ADDRESS: OnceLock<fn(u64) -> Option<String>> = OnceLock::new();

/// [`DESCRIBE_ADDRESS`]'s text for `at`, if it is set and has something to say.
pub fn describe(at: u64) -> Option<String> {
    DESCRIBE_ADDRESS.get().and_then(|f| f(at))
}

/// Reads `out.len()` bytes at `at`, false if they cannot be read (the kernel checks; nothing
/// faults).
pub fn read(at: u64, out: &mut [u8]) -> bool {
    #[cfg(target_os = "macos")]
    {
        unsafe extern "C" {
            static mach_task_self_: u32;
            fn mach_vm_read_overwrite(task: u32, address: u64, size: u64, data: u64, out: *mut u64) -> i32;
        }
        let mut got = 0u64;
        // SAFETY: the kernel copies at most `out.len()` bytes into `out`.
        let kr = unsafe {
            mach_vm_read_overwrite(mach_task_self_, at, out.len() as u64, out.as_mut_ptr() as u64, &mut got)
        };
        kr == 0 && got == out.len() as u64
    }
    #[cfg(not(target_os = "macos"))]
    {
        let _ = (at, out);
        false
    }
}
