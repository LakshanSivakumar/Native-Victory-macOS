//! pthread implementation of `thread` (through `std::thread`, which wraps
//! them and takes the stack size).

use std::io;
use std::os::unix::thread::JoinHandleExt;
use std::time::Duration;

unsafe extern "C" {
    fn pthread_threadid_np(thread: usize, id: *mut u64) -> i32;
    fn pthread_self() -> usize;
    fn pthread_set_qos_class_self_np(qos: u32, relative_priority: i32) -> i32;
    fn pthread_get_qos_class_np(thread: usize, qos: *mut u32, relative_priority: *mut i32) -> i32;
}

/// `QOS_CLASS_USER_INTERACTIVE` in `<sys/qos.h>`.
pub const QOS_CLASS_USER_INTERACTIVE: u32 = 0x21;

/// A thread started by [`spawn`].
#[derive(Debug)]
pub struct Thread {
    handle: std::thread::JoinHandle<u32>,
    id: u32,
}

/// Starts a thread named `name` with a stack of `stack_size` bytes that
/// runs `f`; `f`'s result is the thread's exit code.
pub fn spawn(
    name: &str,
    stack_size: usize,
    f: impl FnOnce() -> u32 + Send + 'static,
) -> io::Result<Thread> {
    let handle = std::thread::Builder::new()
        .name(name.to_owned())
        .stack_size(stack_size)
        .spawn(f)?;
    let mut id = 0;
    // SAFETY: a live thread's pthread_t and a local out-pointer.
    unsafe { pthread_threadid_np(handle.as_pthread_t(), &mut id) };
    Ok(Thread {
        handle,
        id: id as u32,
    })
}

impl Thread {
    /// The OS thread id, as `current_id` returns on that thread.
    pub fn id(&self) -> u32 {
        self.id
    }

    /// Waits for the thread and returns its exit code (`u32::MAX` if it panicked).
    pub fn join(self) -> u32 {
        self.handle.join().unwrap_or(u32::MAX)
    }
}

/// The calling thread's OS id.
pub fn current_id() -> u32 {
    let mut id = 0;
    // SAFETY: 0 names the calling thread; local out-pointer.
    unsafe { pthread_threadid_np(0, &mut id) };
    id as u32
}

pub fn sleep(time: Duration) {
    std::thread::sleep(time);
}

/// Schedules the calling thread at `QOS_CLASS_USER_INTERACTIVE` so macOS treats
/// title frame-driving threads as interactive work (#533).
pub fn set_user_interactive() {
    // SAFETY: documented libSystem call for the calling thread only.
    unsafe {
        pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
    }
}

/// The calling thread's macOS QoS class (`<sys/qos.h>`).
pub fn current_qos_class() -> u32 {
    let (mut qos, mut rel) = (0u32, 0i32);
    // SAFETY: `pthread_self()` names the calling thread; local out-pointers.
    unsafe {
        pthread_get_qos_class_np(pthread_self(), &mut qos, &mut rel);
    }
    qos
}
