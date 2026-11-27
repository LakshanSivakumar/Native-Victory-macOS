//! The one native host the title-facing objects call on macOS: `queue.rs`, `fence.rs`, `list.rs`,
//! `descriptor.rs` and `resource.rs` forward to the functions here (the same call sites that go to
//! `crate::host` on Windows).
//!
//! Threads. The title calls from many threads; the backend's objects and the `Rc`s inside
//! [`Title`] are not thread safe, so everything that touches them runs under one lock (`TITLE`),
//! always taken before a list's own lock, and objects holding such `Rc`s (lists) are dropped under
//! it. The GPU never waits on the lock: submissions are asynchronous, and a timer thread moves held
//! work and fence values along.

use super::memory::{PoolObserver, PoolTable};
use super::state::Graphics;
use super::title::{Guessed, Pipe, TargetFormats, Title, TitleList, Wait};
use crate::object::This;
use std::rc::Rc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use victory_backend::resources::{HeapKind};
use victory_metal::frame_time::{self, Metric, Timer};
use victory_metal::metal::Metal;

pub type MetalTitle = Title<Metal>;
pub type MetalList = TitleList<Metal>;

struct Shared(MetalTitle);
// SAFETY: every access to the title's `Rc`s and Objective-C objects is under `TITLE`.
unsafe impl Send for Shared {}

static TITLE: Mutex<Option<Shared>> = Mutex::new(None);
static FAILED: AtomicBool = AtomicBool::new(false);
static POOLS: OnceLock<Arc<PoolTable>> = OnceLock::new();

/// The table of the title's memory pools. Feed it through [`pool_observer`].
pub fn pools() -> &'static Arc<PoolTable> {
    POOLS.get_or_init(Default::default)
}

/// What `victory_hle`'s physical-memory observer forwards its four calls to
/// (`pool_created`, `pool_released`, `mapped`, `unmapped`): a plain handle on the pool table the
/// native host reads, safe to call from any title thread, before or after the GPU exists.
pub fn pool_observer() -> Arc<dyn PoolObserver> {
    pools().clone()
}

/// Called with the event handle (as the title gave it) when a fence a title waits on with
/// `SetEventOnCompletion` is reached. The runtime sets it; without one, events are not signalled.
static EVENT_HOOK: OnceLock<fn(usize)> = OnceLock::new();

pub fn set_event_hook(hook: fn(usize)) {
    let _ = EVENT_HOOK.set(hook);
}

pub(crate) fn signal_event(event: usize) {
    if let Some(hook) = EVENT_HOOK.get() {
        hook(event);
    }
}

struct TitleGuard {
    guard: std::sync::MutexGuard<'static, Option<Shared>>,
    held: Option<std::time::Instant>,
    kind: Metric,
}

impl TitleGuard {
    fn kind(mut self, kind: Metric) -> Self {
        self.kind = kind;
        self
    }
}

impl std::ops::Deref for TitleGuard {
    type Target = Option<Shared>;

    fn deref(&self) -> &Self::Target {
        &self.guard
    }
}

impl std::ops::DerefMut for TitleGuard {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.guard
    }
}

impl Drop for TitleGuard {
    fn drop(&mut self) {
        if let Some(held) = self.held {
            frame_time::lock_released(held, self.kind);
        }
        // The original mutex guard unlocks after this accounting, at the same scope boundary.
    }
}

fn lock() -> TitleGuard {
    let waiting = frame_time::start();
    let guard = TITLE.lock().unwrap_or_else(|e| e.into_inner());
    let held = waiting.map(frame_time::lock_acquired);
    TitleGuard { guard, held, kind: Metric::HoldOther }
}

/// Runs `f` on the native host, made on first use; `None` if there is no Metal device (the call
/// is then ignored, as when there is no host GPU on Windows).
pub(crate) fn with_title<R>(f: impl FnOnce(&mut MetalTitle) -> R) -> Option<R> {
    with_title_kind(Metric::HoldOther, f)
}

fn with_title_kind<R>(kind: Metric, f: impl FnOnce(&mut MetalTitle) -> R) -> Option<R> {
    let mut guard = lock().kind(kind);
    if guard.is_none() {
        if FAILED.load(Ordering::Relaxed) {
            return None;
        }
        match Metal::new() {
            Ok(metal) => {
                let console = crate::console::current();
                let (gpu_optimal, standard) = console.title_memory();
                let budget =
                    super::textures::texture_budget(std::env::var("VICTORY_METAL_TEXTURE_BUDGET_MB").ok().as_deref());
                let suggested = super::textures::suggested_texture_budget(
                    metal.recommended_working_set(),
                    (gpu_optimal + standard) as u64,
                );
                tracing::info!(
                    console = console.name(),
                    title_memory_mib = (gpu_optimal + standard) / (1 << 20),
                    recommended_working_set_mib = metal.recommended_working_set() / (1 << 20),
                    texture_budget_mib = budget >> 20,
                    suggested_texture_budget_mib = suggested >> 20,
                    "native host: Metal"
                );
                let mut title = Title::new(metal, pools().clone());
                title.set_texture_budget(budget);
                *guard = Some(Shared(title));
                start_timer();
            }
            Err(e) => {
                FAILED.store(true, Ordering::Relaxed);
                tracing::warn!(error = %e, "no native host GPU");
                return None;
            }
        }
    }
    guard.as_mut().map(|t| f(&mut t.0))
}

/// Moves held work along and raises the title's copies of fence values as the GPU signals them
/// (titles read a fence's value from memory), without a call of the title's to drive it.
fn start_timer() {
    let _ = std::thread::Builder::new()
        .name("victory native host".into())
        .spawn(|| {
            loop {
                std::thread::sleep(std::time::Duration::from_millis(1));
                if let Some(t) = lock().kind(Metric::HoldTimer).as_mut() {
                    frame_time::count(Metric::TimerTicks, 1);
                    refresh(&mut t.0);
                }
            }
        });
}

fn refresh(t: &mut MetalTitle) {
    t.pump();
    static LAST: Mutex<Option<std::time::Instant>> = Mutex::new(None);
    {
        let mut last = LAST.lock().unwrap_or_else(|e| e.into_inner());
        if last.is_none_or(|l| l.elapsed().as_secs() >= 5) {
            let stuck = t.stuck(std::time::Duration::from_secs(2));
            if !stuck.is_empty() {
                *last = Some(std::time::Instant::now());
                tracing::warn!("pending queue work or incomplete fence: {stuck:?}");
                // What a held queue or a fence waits for may be owed by a command buffer the GPU
                // lost (out of memory): those are paid from the CPU so the title goes on.
                let lost = victory_metal::gfx::recover_lost();
                if lost > 0 {
                    tracing::warn!("recovered {lost} command buffers the GPU never reported on");
                }
                // Nothing is submitted while every queue is held, so the per-submission report
                // of command buffers the GPU has not finished would stay silent.
                if env_set!("VICTORY_TRACE_GPU_FAILS") {
                    let flying = victory_metal::gfx::report_in_flight();
                    let order: Vec<String> = t
                        .backend_queues()
                        .map(|(k, q)| (k, q.order_state()))
                        .filter(|(_, (done, made))| done != made)
                        .map(|(k, (done, made))| format!("queue {k:#x} at {done} of {made}"))
                        .collect();
                    tracing::warn!(
                        "command buffers in flight while work is held: {flying}; order events behind: {order:?}; commands dropped at unmapped memory so far: {}",
                        t.host.unmapped()
                    );
                }
            }
        }
    }
    for (fence, value) in t.completed() {
        crate::fence::raise(fence as This, value);
    }
}

/// Runs `f` on the native host and a title list's recording.
pub(crate) fn with_list<R>(
    this: This,
    f: impl FnOnce(&mut MetalTitle, &mut MetalList) -> R,
) -> Option<R> {
    if this.is_null() {
        return None;
    }
    with_title_kind(Metric::HoldList, |t| {
        let list = crate::list::native(this);
        let mut list = list.lock().unwrap_or_else(|e| e.into_inner());
        f(t, &mut list)
    })
}

// --- queues and fences ---

/// `ExecuteCommandLists`.
pub(crate) fn submit(queue: This, lists: &[This]) {
    frame_time::count(Metric::Submits, 1);
    if env_set!("VICTORY_TRACE_WRITES") {
        tracing::warn!("submit to queue {queue:p}: lists {lists:?}");
    }
    // One submission at a time, in the order they came: the lock below is let go while the
    // textures are read, and the next submission must not overtake this one meanwhile.
    static ORDER: Mutex<()> = Mutex::new(());
    let order_wait = Timer::new(Metric::OrderWait);
    let _order = ORDER.lock().unwrap_or_else(|e| e.into_inner());
    drop(order_wait);
    // Tests share one title and count the uploads of their own `sync_dirty` calls, which another
    // test's submission with textures read but not yet uploaded would upset: there the lock is
    // kept from the probes to the uploads (`Title::submit`).
    let unlocked = !cfg!(test);
    // The lists' snapshots hold `Rc`s the recording threads share: they are taken and dropped
    // under the title lock, once here and once for the submission below.
    let segments = |t: &MetalTitle| {
        let mut segments = Vec::new();
        for &l in lists.iter().filter(|l| !l.is_null()) {
            let list = crate::list::native(l);
            segments.extend(list.lock().unwrap_or_else(|e| e.into_inner()).segments(t));
        }
        segments
    };
    let mut named = Default::default();
    if unlocked {
        let jobs = with_title_kind(Metric::HoldSync, |t| {
            if t.names_wanted() {
                named = t.keep_named(segments(t).iter().map(|(l, _)| l));
            }
            t.sync_jobs()
        })
        .unwrap_or_default();
        for batch in super::textures::batches(jobs) {
            // Without the lock: the title's other threads go on recording.
            let prepare = Timer::new(Metric::SyncPrepare);
            let prepared = super::textures::prepare(&batch);
            drop(prepare);
            with_title_kind(Metric::HoldSync, |t| t.sync_apply(prepared));
        }
    }
    with_title_kind(Metric::HoldSubmit, |t| {
        // What the GPU drew, cleared or wrote through views is no longer filled from title
        // memory, which the title changes under its render targets (noise uploaded over them).
        let drew = std::mem::take(&mut *DREW.lock().unwrap_or_else(|e| e.into_inner()));
        for key in drew.into_iter().chain(t.host.take_gpu_written()) {
            t.gpu_written(key);
        }
        if t.ensure_queue(queue as usize).is_err() {
            t.let_named_go(named);
            return;
        }
        let segments = segments(t);
        if unlocked {
            t.submit_synced(queue as usize, segments);
        } else {
            t.submit(queue as usize, segments);
        }
        t.let_named_go(named);
        // `VICTORY_TRACE_DESCRIPTORS=sites`: the heaps' pages are closed again for the next
        // interval, so the title's first store into each is seen.
        if super::descriptor_trace::sites() {
            victory_platform::watch::watch(&t.host.heap_ranges());
        }
    });
}

pub(crate) fn queue_signal(queue: This, fence: This, value: u64) {
    let initial = crate::fence::initial(fence);
    with_title(|t| {
        let _ = t.ensure_queue(queue as usize);
        t.signal_queue(queue as usize, fence as usize, initial, value);
    });
}

pub(crate) fn queue_wait(queue: This, fence: This, value: u64) {
    let initial = crate::fence::initial(fence);
    with_title(|t| {
        let _ = t.ensure_queue(queue as usize);
        t.wait_queue(queue as usize, fence as usize, initial, value);
    });
}

pub(crate) fn queue_released(queue: This) {
    with_title(|t| t.forget_queue(queue as usize));
}

/// `ID3D12Fence::Signal`.
pub(crate) fn fence_signal(fence: This, value: u64) {
    let initial = crate::fence::initial(fence);
    with_title(|t| t.signal_cpu(fence as usize, initial, value));
}

/// The value fence `fence` has completed, queues pumped first.
pub(crate) fn fence_poll(fence: This) -> u64 {
    frame_time::count(Metric::FencePolls, 1);
    let initial = crate::fence::initial(fence);
    with_title(|t| {
        t.pump();
        t.fence_value(fence as usize, initial)
    })
    .unwrap_or(initial)
}

/// `GetCompletedValue` of a fence that has no GPU behind it yet: nothing to ask.
pub(crate) fn fence_released(fence: This) {
    with_title(|t| t.forget_fence(fence as usize));
}

/// Blocks until fence `fence` reaches `value` (the title's `SetEventOnCompletion` with no event):
/// held work keeps moving while it waits. False after `timeout`.
pub(crate) fn fence_wait(fence: This, value: u64, timeout: std::time::Duration) -> bool {
    let _wait = Timer::counted(Metric::FenceWait, Metric::FenceWaits);
    let initial = crate::fence::initial(fence);
    let start = std::time::Instant::now();
    loop {
        match with_title(|t| t.try_wait(fence as usize, initial, value)) {
            Some(true) => return true,
            None => return false,
            Some(false) => {}
        }
        if start.elapsed() >= timeout {
            return false;
        }
        std::thread::sleep(std::time::Duration::from_micros(200));
    }
}

// --- present ---

/// Gives the host the app's drawable layer (a `CAMetalLayer`, by address; it must outlive the
/// host): frames the title presents are drawn into it from then on. Without one, `PresentX` shows
/// nothing.
pub fn set_layer(layer: *mut std::ffi::c_void) -> Result<(), String> {
    with_title(|t| t.set_surface(layer)).unwrap_or_else(|| Err("no native host GPU".into()))
}

/// A layer drawing to the host's device, `width` by `height`, for an app that has none (or a test):
/// hand [`victory_metal::present::Layer::as_ptr`] to [`set_layer`] and keep the layer alive.
pub fn create_layer(width: usize, height: usize) -> Result<victory_metal::present::Layer, String> {
    with_title(|t| t.host.backend.create_layer(width, height)).ok_or_else(|| "no native host GPU".into())
}

/// `VICTORY_TRACE_TARGETS=1`: per frame, what drew into or cleared which title texture, and what
/// was presented (logged at each present).
pub(crate) fn tracing_targets() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| env_set!("VICTORY_TRACE_TARGETS"))
}

/// Draws and clears by title texture since the last present, and dropped texture operations.
static TRACE: Mutex<Option<std::collections::HashMap<usize, [u32; 3]>>> = Mutex::new(None);
/// Draws dropped since the last present, by reason (with `VICTORY_TRACE_TARGETS`).
static DROP_REASONS: Mutex<Vec<(&'static str, u32)>> = Mutex::new(Vec::new());
/// Draws dropped since the last present for want of a host pipeline, by shaders and targets.
static DROP_PIPES: Mutex<Vec<(String, u32)>> = Mutex::new(Vec::new());
static DROPPED_OPS: Mutex<Vec<String>> = Mutex::new(Vec::new());

/// Textures a recorded draw or clear writes, made the GPU's ([`Title::gpu_written`]) at the next
/// submission.
static DREW: Mutex<std::collections::BTreeSet<usize>> = Mutex::new(std::collections::BTreeSet::new());

/// Counts texture `key` for the frame trace: `which` 0 drawn or copied into, 1 cleared, 2 a draw
/// into it dropped. Drawn and cleared, it is also the GPU's from the next submission on ([`DREW`]).
fn trace_bump(key: usize, which: usize) {
    if which < 2 {
        DREW.lock().unwrap_or_else(|e| e.into_inner()).insert(key);
    }
    if tracing_targets() {
        TRACE.lock().unwrap_or_else(|e| e.into_inner()).get_or_insert_default().entry(key).or_default()[which] += 1;
    }
}

/// `VICTORY_DUMP_TITLE=<dir>,<seconds>,<seconds>...`: at the first present after each time, the
/// title memory behind every 8-bit colour target drawn to since the last present is read whole,
/// the fraction of non-zero bytes logged, and the memory detiled into a scratch texture and
/// written as `<dir>/m<time>_<n>.ppm`: what the title (CPU, DMA) put there, whatever the GPU did.
#[inline(never)]
fn dump_title_memory(stats: &std::collections::HashMap<usize, [u32; 3]>) {
    static START: OnceLock<std::time::Instant> = OnceLock::new();
    static DONE: Mutex<Vec<u64>> = Mutex::new(Vec::new());
    let start = START.get_or_init(std::time::Instant::now);
    let Ok(v) = std::env::var("VICTORY_DUMP_TITLE") else { return };
    let mut parts = v.split(',');
    let dir = parts.next().unwrap_or(".").to_owned();
    let now = start.elapsed().as_secs();
    let Some(time) = parts
        .filter_map(|p| p.parse::<u64>().ok())
        .find(|&t| now >= t && !DONE.lock().unwrap_or_else(|e| e.into_inner()).contains(&t))
    else {
        return;
    };
    DONE.lock().unwrap_or_else(|e| e.into_inner()).push(time);
    with_title(|t| {
        for (n, key) in stats.keys().enumerate() {
            let Some(info) = crate::resource::texture_info(*key as This) else { continue };
            if !matches!(info.format, 28 | 87 | 91 | 29) {
                continue;
            }
            let tiled = super::textures::read_memory(info.memory, info.size);
            let nz = tiled.iter().filter(|&&b| b != 0).count();
            tracing::info!(time, n, key = format!("{key:#x}"), at = format!("{:#x}", info.memory as usize), size = info.size, nonzero = nz, fraction = nz as f64 / tiled.len().max(1) as f64, "title memory of a colour target");
            let Some(desc) = info.desc() else { continue };
            let Ok(tex) = t.host.texture(&desc) else { continue };
            if super::textures::upload(&t.host.backend, &tex, &info, &tiled).is_err() {
                continue;
            }
            let (w, h) = (info.width as usize, info.height as usize);
            let mut px = vec![0u8; w * h * 4];
            victory_backend::resources::ResourceBackend::read_texture(&t.host.backend, &tex, 0, 0, &mut px, w * 4, w * h * 4);
            let bgra = matches!(info.format, 87 | 91);
            let mut out = format!("P6\n{w} {h}\n255\n").into_bytes();
            out.extend(px.as_chunks::<4>().0.iter().flat_map(|p| if bgra { [p[2], p[1], p[0]] } else { [p[0], p[1], p[2]] }));
            let _ = std::fs::write(std::path::Path::new(&dir).join(format!("m{time}_{n}.ppm")), out);
        }
    });
}

/// `VICTORY_DUMP_TARGETS=<dir>,<seconds>`: once, at the first present after that many seconds,
/// every 8-bit colour target drawn to since the last present is written as `<dir>/t<n>.ppm`; HDR
/// and 10-bit targets (DXGI 2, 10, 24, 26) are tone-mapped into `..._f<format>.ppm`, their range logged.
fn dump_targets(stats: &std::collections::HashMap<usize, [u32; 3]>) {
    use std::sync::atomic::AtomicBool;
    static DONE: AtomicBool = AtomicBool::new(false);
    static LAST: Mutex<Option<u64>> = Mutex::new(None);
    static START: OnceLock<std::time::Instant> = OnceLock::new();
    let start = START.get_or_init(std::time::Instant::now);
    let Ok(v) = std::env::var("VICTORY_DUMP_TARGETS") else { return };
    // `<dir>,<seconds>[,<every>]`: from that second on, once or every that many seconds.
    let mut parts = v.split(',');
    let dir = parts.next().unwrap_or(".").to_owned();
    let secs = parts.next().and_then(|s| s.parse::<u64>().ok()).unwrap_or(25);
    let every = parts.next().and_then(|s| s.parse::<u64>().ok());
    let now = start.elapsed().as_secs();
    let due = match (every, *LAST.lock().unwrap_or_else(|e| e.into_inner())) {
        (Some(e), Some(last)) => now >= last + e,
        _ => true,
    };
    if now < secs || !due || (every.is_none() && DONE.swap(true, Ordering::Relaxed)) {
        return;
    }
    *LAST.lock().unwrap_or_else(|e| e.into_inner()) = Some(now);
    with_title(|t| {
        for (n, key) in stats.keys().enumerate() {
            let Some(info) = crate::resource::texture_info(*key as This) else { continue };
            let (w, h) = (info.width as usize, info.height as usize);
            if matches!(info.format, 2 | 10 | 24 | 26) {
                // HDR and 10-bit targets: tone-mapped to 8 bits, with their range logged.
                let Some(tex) = t.texture_of(*key) else { continue };
                let bpp = match info.format { 2 => 16, 10 => 8, _ => 4 };
                let mut raw = vec![0u8; w * h * bpp];
                victory_backend::resources::ResourceBackend::read_texture(&t.host.backend, &tex, 0, 0, &mut raw, w * bpp, w * h * bpp);
                let rgb: Vec<[f32; 3]> = raw.chunks_exact(bpp).map(|p| decode_hdr(info.format, p)).collect();
                let finite = rgb.iter().flatten().filter(|v| v.is_finite());
                let (mut max, mut sum, mut count, mut lit) = (0f32, 0f64, 0usize, 0usize);
                for v in finite {
                    max = max.max(*v);
                    sum += f64::from(*v);
                    count += 1;
                    lit += usize::from(*v > 1e-4);
                }
                let mut out = format!("P6\n{w} {h}\n255\n").into_bytes();
                out.extend(rgb.iter().flat_map(|c| c.map(|v| if v.is_finite() && v > 0.0 { ((v / (1.0 + v)).sqrt() * 255.0) as u8 } else { 0 })));
                let _ = std::fs::write(std::path::Path::new(&dir).join(format!("t{n}_{now}_{:x}_f{}.ppm", info.memory as usize, info.format)), out);
                tracing::info!(n, key = format!("{key:#x}"), w, h, format = info.format, max, mean = sum / count.max(1) as f64, lit = lit as f64 / (rgb.len() * 3).max(1) as f64, "target dumped");
                continue;
            }
            // 16- and 32-bit single-channel targets (depth and shadow maps among them): no image,
            // but how much of the target differs from its first texel says whether anything was
            // rasterised into it. Depth with stencil (DXGI 19, 20, 44 to 47) cannot be read back.
            if let Some(bpp) = match info.format { 53..=59 => Some(2usize), 39..=43 => Some(4), _ => None } {
                let Some(tex) = t.texture_of(*key) else { continue };
                let mut raw = vec![0u8; w * h * bpp];
                victory_backend::resources::ResourceBackend::read_texture(&t.host.backend, &tex, 0, 0, &mut raw, w * bpp, w * h * bpp);
                let first = raw[..bpp].to_vec();
                let differ = raw.chunks_exact(bpp).filter(|p| *p != first.as_slice()).count();
                let distinct = raw.chunks_exact(bpp).step_by(7).collect::<std::collections::HashSet<_>>().len();
                let grey = |p: &[u8]| if bpp == 2 { p[1] } else { (f32::from_le_bytes([p[0], p[1], p[2], p[3]]).clamp(0.0, 1.0) * 255.0) as u8 };
                let mut out = format!("P6\n{w} {h}\n255\n").into_bytes();
                out.extend(raw.chunks_exact(bpp).flat_map(|p| [grey(p); 3]));
                let _ = std::fs::write(std::path::Path::new(&dir).join(format!("t{n}_{now}_{:x}_single{}.ppm", info.memory as usize, info.format)), out);
                tracing::info!(n, key = format!("{key:#x}"), w, h, format = info.format, first = format!("{first:02x?}"), differ = differ as f64 / (w * h).max(1) as f64, distinct, "single-channel target dumped");
                continue;
            }
            if !matches!(info.format, 28 | 87 | 91 | 29) {
                tracing::info!(n, key = format!("{key:#x}"), w, h, format = info.format, draws = ?stats.get(key), "target not dumped (format not read back)");
                continue;
            }
            let Some(tex) = t.texture_of(*key) else { continue };
            let mut px = vec![0u8; w * h * 4];
            victory_backend::resources::ResourceBackend::read_texture(&t.host.backend, &tex, 0, 0, &mut px, w * 4, w * h * 4);
            let bgra = matches!(info.format, 87 | 91);
            let mut out = format!("P6\n{w} {h}\n255\n").into_bytes();
            out.extend(px.as_chunks::<4>().0.iter().flat_map(|p| if bgra { [p[2], p[1], p[0]] } else { [p[0], p[1], p[2]] }));
            let _ = std::fs::write(std::path::Path::new(&dir).join(format!("t{n}_{now}_{:x}.ppm", info.memory as usize)), out);
            // The alpha channel as grey.
            let mut alpha = format!("P6\n{w} {h}\n255\n").into_bytes();
            alpha.extend(px.as_chunks::<4>().0.iter().flat_map(|p| [p[3], p[3], p[3]]));
            let _ = std::fs::write(std::path::Path::new(&dir).join(format!("t{n}_{now}_{:x}_alpha.ppm", info.memory as usize)), alpha);
            tracing::info!(n, key = format!("{key:#x}"), w, h, "target dumped");
        }
    });
}

/// One texel of an HDR or 10-bit colour target (DXGI 2, 10, 24, 26) as linear RGB.
fn decode_hdr(format: u32, p: &[u8]) -> [f32; 3] {
    let word = |i: usize| u32::from_le_bytes([p[i], p[i + 1], p[i + 2], p[i + 3]]);
    // An unsigned small float: `mbits` mantissa bits over a 5-bit exponent.
    let small = |v: u32, mbits: u32| {
        let (e, m) = ((v >> mbits) & 0x1f, v & ((1 << mbits) - 1));
        if e == 0 { m as f32 * 2f32.powi(-14 - mbits as i32) } else { (1.0 + m as f32 / (1 << mbits) as f32) * 2f32.powi(e as i32 - 15) }
    };
    match format {
        2 => [0, 4, 8].map(|i| f32::from_bits(word(i))),
        10 => [0, 2, 4].map(|i| {
            let h = u32::from(u16::from_le_bytes([p[i], p[i + 1]]));
            let v = small(h & 0x7fff, 10);
            if h >> 15 == 1 { -v } else if (h >> 10) & 0x1f == 0x1f { f32::NAN } else { v }
        }),
        24 => [0, 10, 20].map(|s| ((word(0) >> s) & 0x3ff) as f32 / 1023.0),
        _ => {
            let v = word(0);
            [small(v & 0x7ff, 6), small((v >> 11) & 0x7ff, 6), small(v >> 22, 5)]
        }
    }
}

/// A texture operation the native host does not do (copies, resolves), counted by the trace.
pub(crate) fn dropped_op(what: &str, dst: This, src: This) {
    TEXTURE_COPIES_DROPPED.fetch_add(1, Ordering::Relaxed);
    warn_once("a texture copy or resolve");
    // Once per operation and title formats (0: a buffer or not a texture), so a refused kind of
    // copy shows what it was.
    static SEEN: Mutex<Vec<(String, u32, u32)>> = Mutex::new(Vec::new());
    let format = |r: This| crate::resource::texture_info(r).map_or(0, |i| i.format);
    let key = (what.to_owned(), format(src), format(dst));
    let mut seen = SEEN.lock().unwrap_or_else(|e| e.into_inner());
    if !seen.contains(&key) {
        tracing::warn!(what, src_format = key.1, dst_format = key.2, "a texture copy refused (first of this kind)");
        seen.push(key);
    }
    drop(seen);
    if tracing_targets() {
        DROPPED_OPS.lock().unwrap_or_else(|e| e.into_inner()).push(format!("{what} {src:?}->{dst:?}"));
    }
}

/// `VICTORY_TRACE_RECORD_WRITERS`: one compute program's dispatches since the last present, run
/// or dropped: the buffers its tables hold for unordered access (title memory start and size),
/// and the textures they hold as shader resources and for unordered access.
#[derive(Default)]
struct RecordDispatches {
    buffers: std::collections::BTreeSet<(u64, u64)>,
    reads: std::collections::BTreeSet<usize>,
    writes: std::collections::BTreeSet<usize>,
}

/// `VICTORY_TRACE_RECORD_WRITERS`: the compute dispatches since the last present by program and
/// outcome, and each indirect mesh execution's first record address with its records and how
/// many had a group count of 0 in title memory as the execution was recorded.
#[derive(Default)]
struct RecordTrace {
    dispatches: std::collections::BTreeMap<String, RecordDispatches>,
    executions: std::collections::BTreeMap<u64, [u64; 2]>,
}

static RECORD_TRACE: Mutex<Option<RecordTrace>> = Mutex::new(None);

fn tracing_record_writers() -> bool {
    env_set!("VICTORY_TRACE_RECORD_WRITERS")
}

/// `VICTORY_TRACE_RECORD_WRITERS=<hash>[,<hash>...]`: the records watched are those of the
/// indirect dispatches of these compute shaders, not the mesh records (`None`: the mesh records).
fn record_writers_of() -> Option<&'static str> {
    static OF: OnceLock<Option<String>> = OnceLock::new();
    OF.get_or_init(|| std::env::var("VICTORY_TRACE_RECORD_WRITERS").ok().filter(|v| v.len() >= 16)).as_deref()
}

/// The records and zero records of the executions whose first record is inside one of `buffers`.
fn records_in(buffers: &std::collections::BTreeSet<(u64, u64)>, executions: &std::collections::BTreeMap<u64, [u64; 2]>) -> [u64; 2] {
    executions
        .iter()
        .filter(|(at, _)| buffers.iter().any(|&(start, size)| (start..start.saturating_add(size)).contains(at)))
        .fold([0; 2], |a, (_, n)| [a[0] + n[0], a[1] + n[1]])
}

#[cfg(test)]
#[test]
fn records_are_counted_for_the_buffers_that_hold_their_first_record() {
    let executions = [(0x1000, [4, 4]), (0x1ff4, [2, 1]), (0x2000, [8, 0])].into_iter().collect();
    assert_eq!(records_in(&[(0x1000, 0x1000)].into_iter().collect(), &executions), [6, 5]);
    assert_eq!(records_in(&[(0x1000, 0x1000), (0x1800, 0x1000)].into_iter().collect(), &executions), [14, 5]);
    assert_eq!(records_in(&[(0x3000, 0x10), (u64::MAX, 2)].into_iter().collect(), &executions), [0, 0]);
}

fn trace_record_dispatch(t: &MetalTitle, l: &MetalList, outcome: &str) {
    if !tracing_record_writers() {
        return;
    }
    let (buffers, reads, writes) = t.host.table_views(l.list.compute_args());
    let mut trace = RECORD_TRACE.lock().unwrap_or_else(|e| e.into_inner());
    let d = trace.get_or_insert_default().dispatches.entry(format!("{:?} {outcome}", stage_hashes(l.pipeline))).or_default();
    d.buffers.extend(buffers);
    d.reads.extend(reads);
    d.writes.extend(writes);
}

/// `VICTORY_TRACE_RECORD_WRITERS`, at each present: the frame's indirect mesh records and how
/// many were 0 as recorded (the GPU may still write them: a culling pass that ran fills them
/// later, one that was dropped never does); each compute program whose tables hold a record
/// buffer for unordered access, with the records in those buffers and the textures it reads;
/// and for each such texture, the compute programs that hold it for unordered access.
#[inline(never)]
fn trace_record_writers() {
    let Some(trace) = RECORD_TRACE.lock().unwrap_or_else(|e| e.into_inner()).take() else { return };
    let total = trace.executions.values().fold([0u64; 2], |a, n| [a[0] + n[0], a[1] + n[1]]);
    if total[0] == 0 {
        return;
    }
    let texture = |r: usize| {
        let i = crate::resource::texture_info(r as This);
        let (f, w, h) = i.map_or((0, 0, 0), |i| (i.format, i.width, i.height));
        format!("{r:#x} fmt {f} {w}x{h}")
    };
    tracing::info!(records = total[0], zero = total[1], "mesh records");
    let mut read = std::collections::BTreeSet::new();
    for (name, d) in &trace.dispatches {
        let n = records_in(&d.buffers, &trace.executions);
        if n[0] == 0 {
            continue;
        }
        read.extend(&d.reads);
        let reads: Vec<String> = d.reads.iter().map(|&r| texture(r)).collect();
        tracing::info!(records = n[0], zero = n[1], holder = name, ?reads, "mesh record buffer holder");
    }
    for r in read {
        let writers: Vec<&String> = trace.dispatches.iter().filter(|(_, d)| d.writes.contains(&r)).map(|(n, _)| n).collect();
        tracing::info!(texture = texture(r), ?writers, "texture read by a mesh record buffer holder");
    }
}

#[inline(never)]
fn trace_present(resource: This, planes: &[This]) {
        let stats = TRACE.lock().unwrap_or_else(|e| e.into_inner()).take().unwrap_or_default();
        let ops = std::mem::take(&mut *DROPPED_OPS.lock().unwrap_or_else(|e| e.into_inner()));
        let reasons = std::mem::take(&mut *DROP_REASONS.lock().unwrap_or_else(|e| e.into_inner()));
        let mut pipes = std::mem::take(&mut *DROP_PIPES.lock().unwrap_or_else(|e| e.into_inner()));
        pipes.sort_by_key(|(_, n)| std::cmp::Reverse(*n));
        for (what, n) in &pipes {
            tracing::info!(n, what, "draws dropped (no host pipeline, or an indirect execution)");
        }
        let described: Vec<String> = with_title(|t| {
            stats
                .iter()
                .map(|(k, [d, c, x])| {
                    let i = crate::resource::texture_info(*k as This);
                    let (f, w, h, a) = i.map_or((0, 0, 0, 0), |i| (i.format, i.width, i.height, i.memory as usize));
                    let mark = if *k == resource as usize { " PRESENTED" } else { "" };
                    let _ = &t;
                    format!("{k:#x} fmt {f} {w}x{h} at {a:#x} draws {d} clears {c} dropped {x}{mark}")
                })
                .collect()
        })
        .unwrap_or_default();
        dump_targets(&stats);
        dump_title_memory(&stats);
        let pi = crate::resource::texture_info(resource);
        tracing::info!(
            present = ?resource,
            ?planes,
            at = pi.as_ref().map_or(0, |i| i.memory as usize),
            fmt = pi.as_ref().map_or(0, |i| i.format),
            targets = ?described,
            dropped_ops = ?ops.iter().take(8).collect::<Vec<_>>(),
            dropped = ops.len(),
            copies = COPIES[1].swap(0, Ordering::Relaxed),
            ?reasons,
            "frame targets"
        );
}

/// `VICTORY_TRACE_PRESENT=<every>`: the title resources `PresentX` has shown.
static PRESENTED: Mutex<Vec<usize>> = Mutex::new(Vec::new());

fn present_every() -> Option<u32> {
    static EVERY: OnceLock<Option<u32>> = OnceLock::new();
    *EVERY.get_or_init(|| std::env::var("VICTORY_TRACE_PRESENT").ok().and_then(|v| v.parse().ok()))
}

/// `VICTORY_TRACE_PRESENT=<every>`: a draw into a texture the title has presented (its present
/// pass), the first and then every `<every>`th, at most 12: its pipeline's shaders, root
/// parameters and binding, which textures its tables hold and how much of each is non-zero, and
/// the first bytes of its constant buffers, all as the draw is recorded.
fn trace_present_draw(t: &MetalTitle, l: &MetalList) {
    use std::sync::atomic::AtomicU32;
    static N: AtomicU32 = AtomicU32::new(0);
    let Some(every) = present_every().map(|e| e.max(1)) else { return };
    {
        let shown = PRESENTED.lock().unwrap_or_else(|e| e.into_inner());
        if !l.target_keys.iter().any(|k| shown.contains(k)) {
            return;
        }
    }
    let n = N.fetch_add(1, Ordering::Relaxed);
    if n % every != 0 || n / every >= 12 {
        return;
    }
    let shaders = stage_hashes(l.pipeline);
    let targets: Vec<String> = l.target_keys.iter().map(|k| format!("{k:#x}")).collect();
    tracing::info!(n, pipeline = format!("{:#x}", l.pipeline), ?shaders, ?targets, "present draw");
    let (root, binding) = l.list.graphics_args();
    if let Some(b) = binding {
        // Filter, address u v w, mip LOD bias, anisotropy, comparison, border, min and max LOD.
        let ps = b.ps.as_ref().map(|p| &p.static_samplers);
        tracing::info!(vs = ?b.static_samplers, ?ps, "present draw static samplers");
    }
    t.host.trace_args(root, binding);
}

/// `PresentX` on title queue `queue`: title textures `planes` over the layer, the first under the
/// others (`Title::present`). The traces name the first.
pub(crate) fn present(queue: This, planes: &[This]) {
    let Some(&resource) = planes.first() else { return };
    let present = Timer::new(Metric::Present);
    if present_every().is_some() {
        let mut shown = PRESENTED.lock().unwrap_or_else(|e| e.into_inner());
        for &p in planes {
            if !shown.contains(&(p as usize)) {
                shown.push(p as usize);
            }
        }
    }
    // Diagnostic: VICTORY_PRESENT_DELAY_MS slows every present, to tell a trigger that follows
    // frames from one that follows time.
    static DELAY: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    let ms = *DELAY.get_or_init(|| {
        std::env::var("VICTORY_PRESENT_DELAY_MS").ok().and_then(|v| v.parse().ok()).unwrap_or(0)
    });
    if ms > 0 {
        std::thread::sleep(std::time::Duration::from_millis(ms));
    }
    flow_present();
    if tracing_record_writers() {
        trace_record_writers();
    }
    if tracing_targets() {
        trace_present(resource, planes);
    }
    pass_present(resource);
    let shown = with_title(|t| {
        let _ = t.ensure_queue(queue as usize);
        let keys: Vec<usize> = planes.iter().map(|&p| p as usize).collect();
        t.present(queue as usize, &keys)
    });
    match shown {
        Some(Err(e)) => tracing::warn!(error = %e, "a frame was not presented"),
        Some(Ok(false)) => {
            NO_TEXTURE.fetch_add(1, Ordering::Relaxed);
        }
        _ => {}
    }
    drop(present);
    frame_time::report();
}

/// Frames presented to the layer so far.
pub fn frames_presented() -> u64 {
    with_title(|t| t.surface().map_or(0, |s| s.frames())).unwrap_or(0)
}

/// The pixels (BGRA) of the last frame presented, once the GPU has drawn it; for tests.
pub fn last_frame() -> Option<(usize, usize, Vec<u8>)> {
    with_title(|t| t.surface().and_then(|s| s.read_last())).flatten()
}

// --- resources and descriptors ---

/// A buffer resource object `res` of `len` bytes at title address `addr`.
/// `own`: its memory is the host's (committed, or in a heap), not title memory a pool holds.
pub(crate) fn buffer_created(res: This, addr: u64, len: u64, own: bool) {
    with_title(|t| match own {
        true => t.host.register_own_buffer(res as usize, addr, len),
        false => t.host.register_buffer(res as usize, addr, len),
    });
}

/// A pipeline object's last reference went: the host pipelines made for it are forgotten, as the
/// title makes thousands of variants (device slot 0x1b8) a minute and a new one often gets a freed
/// one's address (which drew text with the old variant's blend state: glyphs as boxes).
pub(crate) fn pipeline_released(pipeline: This) {
    let key = pipeline as usize;
    with_title(|t| {
        t.pipelines.remove(&key);
        t.rect_pipes.retain(|(k, _), _| *k != key);
        t.console_pipes.retain(|(k, _), _| *k != key);
        t.thread_pipes.remove(&key);
    });
}

pub(crate) fn resource_released(res: This) {
    with_title(|t| {
        t.host.release_resource(res as usize);
        t.texture_released(res as usize);
    });
}

/// A title texture object: its host texture is made, and filled from title memory when work is
/// next submitted.
pub(crate) fn texture_created(res: This) {
    if let Some(info) = crate::resource::texture_info(res) {
        with_title(|t| t.texture_created(res as usize, info));
    }
}

/// `OMSetRenderTargets`: the host textures of the views' resources, in the views' formats.
pub(crate) fn targets(this: This, color: &[u64], depth: u64) {
    with_list(this, |t, l| {
        let views: Vec<_> = color.iter().map(|&h| crate::descriptor::rtv_record(h)).collect();
        let depth_key = crate::descriptor::dsv_record(depth).map(|r| r.resource as usize);
        let depth = depth_key.and_then(|k| t.texture_of(k));
        l.target_keys = views.iter().flatten().map(|r| r.resource as usize).chain(depth_key).collect();
        // A view the host cannot make keeps its slot: binding the rest would move every later
        // target down one, and `SV_TargetN` would write a surface this pass does not own.
        let made: Option<Vec<_>> = views
            .iter()
            .map(|slot| slot.and_then(|r| t.target_of(r)))
            .collect();
        let Some(made) = made else {
            l.targets_ready = false;
            l.targets = TargetFormats::default();
            l.target_keys.clear();
            l.list.set_render_targets(&[], None);
            warn_once("a render target the host cannot make; the passes that use it are refused");
            return;
        };
        l.targets_ready = true;
        l.targets = TargetFormats {
            color: made.iter().map(|(_, format)| *format).collect(),
            depth: depth_key.and_then(|k| t.format_of(k)),
        };
        let color: Vec<_> = made.into_iter().map(|(texture, _)| texture).collect();
        // Debugging: no depth attachment at all.
        let drop_depth = env_set!("VICTORY_DEBUG_NO_DEPTH_ATTACH");
        if drop_depth {
            l.targets.depth = None;
        }
        l.list.set_render_targets(&color, depth.as_ref().filter(|_| !drop_depth));
    });
}

/// `ClearUnorderedAccessViewUint` / `...Float` of the view at descriptor slot `cpu` (ignored on
/// the native host before): a texture view's level is cleared as a render pass's load action
/// does, if the texture is also a render target; a depth texture's on the plane of its
/// writable twin the view names; a buffer view is filled, when the value is one byte repeated
/// (what a fill can write). Grand Theft Auto VI was not seen calling these in its first world
/// frames (2026-11-27), so this is exercised by the GPU test only.
/// shortcut: the whole view, whatever the rectangles; slice 0 of an array; a texture that is
/// unordered access alone is not cleared (`NativeHost::unordered_clear_target`).
pub(crate) fn clear_unordered(this: This, cpu: u64, values: [u32; 4], float: bool) {
    let Some(view) = crate::descriptor::view(cpu) else {
        warn_once("an unordered access clear of a slot without a view");
        return;
    };
    with_list(this, |t, l| {
        // The texture the view names is realized first (the host only knows realized ones).
        if let crate::descriptor::View::Unordered(r, ..) = &view
            && crate::resource::is_resource(*r as This)
            && crate::resource::texture_info(*r as This).is_some()
        {
            let _ = t.texture_of(*r);
        }
        let Some((target, format, size)) = t.host.unordered_clear_target(&view) else {
            warn_once("an unordered access clear of a view the host cannot make");
            return;
        };
        if tracing_targets() {
            static SEEN: Mutex<Vec<(String, [u32; 4], bool)>> = Mutex::new(Vec::new());
            let what = format!("{format} {size:?}");
            let mut seen = SEEN.lock().unwrap_or_else(|e| e.into_inner());
            if seen.len() < 96 && !seen.contains(&(what.clone(), values, float)) {
                tracing::info!(what, values = format!("{values:08x?}"), float, "unordered access clear (first with these values)");
                seen.push((what, values, float));
            }
        }
        match target {
            super::host::ClearTarget::Texture(texture) => {
                let color = if float { values.map(f32::from_bits) } else { values.map(|v| v as f32) };
                l.list.clear_render_target(&texture, color);
            }
            super::host::ClearTarget::Buffer(addr, len) => {
                let bytes = values[0].to_le_bytes();
                if bytes.iter().all(|&b| b == bytes[0]) {
                    l.list.fill(addr, len, bytes[0]);
                } else {
                    warn_once("an unordered access clear of a buffer to a value that is not one byte repeated: not done");
                }
            }
        }
    });
}

/// `ClearRenderTargetView` against the attachment named by the descriptor.
pub(crate) fn clear_rtv(this: This, handle: u64, color: [f32; 4], rects: &[crate::list::Rect]) {
    with_list(this, |t, l| {
        if let Some(record) = crate::descriptor::rtv_record(handle)
            && let Some((texture, _)) = t.target_of(record)
        {
            trace_bump(record.resource as usize, 1);
            if pass_on() {
                pass_note_op(l, &format!("clear-rtv {color:?} rects={}", rects.len()), None, Some(record.resource as usize));
            }
            if rects.is_empty() {
                l.list.clear_render_target(&texture, color);
            } else {
                l.list.clear_render_target_rects(&texture, color, rects);
            }
        }
    });
}

#[inline(never)]
fn trace_clear_dsv(flags: u32, depth: f32) {
    static N: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
    let n = N.fetch_add(1, Ordering::Relaxed);
    if n < 6 || (n % 40 == 0 && n < 3000) {
        tracing::info!(flags, depth, n, "ClearDepthStencilView");
    }
}

/// `ClearDepthStencilView`: depth (flag 1) and stencil (flag 2) are independent.
pub(crate) fn clear_dsv(this: This, handle: u64, flags: u32, depth: f32, stencil: u8, count: u32) {
    if tracing_targets() {
        trace_clear_dsv(flags, depth);
    }
    // Neither rectangle bounds nor attachment mip/layer selection are represented by
    // NativeList's clear command. Do not widen an unsupported clear to the base texture.
    if count != 0 || flags & 3 == 0 {
        return;
    }
    let Some(record) = crate::descriptor::dsv_record(handle).filter(|r| r.native_dsv_base_view()) else {
        return;
    };
    if !crate::resource::native_dsv_clear_supported(record.resource) {
        return;
    }
    with_list(this, |t, l| {
        if let Some(tex) = t.texture_of(record.resource as usize) {
            trace_bump(record.resource as usize, 1);
            if pass_on() {
                pass_note_op(l, &format!("clear-dsv flags={flags} depth={depth}"), None, Some(record.resource as usize));
            }
            l.list.clear_depth_stencil(
                &tex,
                (flags & 1 != 0).then_some(depth),
                (flags & 2 != 0).then_some(stencil),
            );
        }
    });
}

/// Shader-visible descriptor heaps by address range: start, end, sampler heap, count. The host
/// heap is made when a list first names the heap, from what the title wrote by then.
static HEAPS: Mutex<Vec<(u64, u64, bool, usize)>> = Mutex::new(Vec::new());

pub(crate) fn heap_created(start: u64, samplers: bool, count: usize) {
    let step = if samplers { 16 } else { 32 };
    HEAPS
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .push((start, start + step * count as u64, samplers, count));
}

pub(crate) fn heap_released(start: u64) {
    HEAPS
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .retain(|h| h.0 != start);
    with_title(|t| t.host.drop_heap(start));
}

/// The host heap the descriptor at `handle` is in (made now if it was not) and its index.
fn heap_of(t: &mut MetalTitle, handle: u64) -> Option<(Rc<super::list::NativeHeap<Metal>>, usize)> {
    if let Some(found) = t.host.heap_at(handle) {
        return Some(found);
    }
    let (start, _, samplers, count) = HEAPS
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .iter()
        .copied()
        .find(|h| (h.0..h.1).contains(&handle))?;
    let kind = if samplers {
        HeapKind::Sampler
    } else {
        HeapKind::Resource
    };
    let step = if samplers { 16 } else { 32 };
    match t.host.heap(kind, start, count) {
        Ok(heap) => {
            // What the title wrote before the heap was made.
            for i in 0..count as u64 {
                let _ = t.host.mirror(start + i * step);
            }
            t.host.heap_at(handle).or(Some((heap, 0)))
        }
        Err(e) => {
            tracing::warn!(error = %e, "no host descriptor heap");
            None
        }
    }
}

/// The title created a view (`placement`) or copied an existing descriptor to `handle`.
pub(crate) fn mirror(handle: u64, placement: bool) {
    let _mirror = Timer::counted(Metric::DeviceMirror, Metric::DeviceMirrors);
    with_title(|t| {
        match crate::descriptor::view(handle) {
            // A texture the title placed in its own memory has no resource object: the host
            // reuses a GPU-written object's texture there, or makes one at the first view.
            Some(crate::descriptor::View::Placed(_, res, d)) if res[0] > 1 => {
                let address = u64::from(d[10]) | u64::from(d[11]) << 32;
                let key = super::textures::placed_key(address, &res);
                if (placement || !t.host.knows(key))
                    && let Some(info) = crate::resource::placed_info(address, &res)
                    && let Some((pool, _)) = t.host.memory().resolve(address, 1)
                {
    let _realize = Timer::new(Metric::Realize);
                    t.placed_view(address, key, pool.id, info, placement);
                }
            }
            // A view of a texture object that no list has used yet.
            Some(
                crate::descriptor::View::Shader(r, _) | crate::descriptor::View::Unordered(r, ..),
            ) => {
                let _realize = Timer::new(Metric::Realize);
                t.realize(r);
            }
            _ => {}
        }
        if let Err(e) = t.host.mirror(handle) {
            static N: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
            if N.fetch_add(1, Ordering::Relaxed) < 3000 {
                tracing::info!(error = %e, handle = format!("{handle:#x}"), "a descriptor was not mirrored");
            }
        }
    });
}

// --- list recording ---

pub(crate) fn set_pipeline(this: This, pipeline: This) {
    with_list(this, |_, l| l.pipeline = pipeline as usize);
}

/// The host pipeline of title pipeline `key` for a draw into `targets`, made at first use. A
/// console pipeline's state names no target formats, so it is made again for each set of targets.
fn pipe(t: &mut MetalTitle, key: usize, targets: &TargetFormats, rect: bool) -> Pipe<Metal> {
    if rect && let Some(p) = t.rect_pipes.get(&(key, targets.clone())) {
        return p.clone();
    }
    if !rect && let Some(p) = t.pipelines.get(&key) {
        return p.clone();
    }
    let console_key = (key, targets.clone());
    if !rect && let Some(p) = t.console_pipes.get(&console_key) {
        return p.clone();
    }
    // A cold first world frame binds ~1,300 pipelines; translating them all here holds the frame
    // past the title's watchdog. Once the frame has spent its budget, a pipeline whose shaders are
    // not translated yet is put off: its work is dropped this time (not cached, so a later draw
    // makes it), and its shaders go to the front of the prewarm pool.
    let frame = crate::frame::current();
    let left = frame_left();
    if left.is_zero()
        && let Some((stages, _)) = crate::pipeline::native_stages(key as This)
        && stages.iter().any(|(_, s)| !s.native_ready())
    {
        for (_, s) in stages.iter().filter(|(_, s)| !s.native_ready()) {
            crate::pipeline::prewarm_next(s);
        }
        let n = PUT_OFF.fetch_add(1, Ordering::Relaxed) + 1;
        if n.is_power_of_two() {
            tracing::info!(frame, put_off = n, "pipelines put off for their translations so far");
        }
        return Pipe::Refused;
    }
    // A pipeline Metal is still compiling when the budget ends is put off the same way.
    t.compile_wait = compile_wait(t);
    let put_off = t.shaders.put_off;
    let (made, console) = make_pipe(t, key, targets, rect);
    if t.shaders.put_off != put_off {
        return made;
    }
    if rect {
        t.rect_pipes.insert(console_key, made.clone());
    } else if console {
        t.console_pipes.insert(console_key, made.clone());
    } else {
        t.pipelines.insert(key, made.clone());
    }
    made
}

/// What the current frame has left of its budget for new pipelines (zero: spent).
fn frame_left() -> std::time::Duration {
    BUDGET.lock().unwrap_or_else(|e| e.into_inner()).left(crate::frame::current(), std::time::Instant::now(), crate::prewarm::FIRST_BIND_BUDGET)
}

/// [`frame_left`] for tests, which read it to see whether a compile spent its frame.
#[cfg(test)]
pub(crate) fn frame_left_now() -> std::time::Duration {
    frame_left()
}

/// Spends what the current frame has left of its budget for new pipelines: from here on [`pipe`]
/// puts off the frame's pipelines whose shaders are not translated yet, and [`compile_wait`] waits
/// 20 ms a pipeline then nothing, until the next frame starts with the budget whole again.
pub(crate) fn spend_frame_budget() {
    BUDGET.lock().unwrap_or_else(|e| e.into_inner()).spend(crate::frame::current(), std::time::Instant::now());
}

/// A pipeline compile that ended in `error`: if Metal's compiler died on it rather than refusing
/// a shader the host cannot run ([`victory_metal::is_compiler_crash`]), the frame spends the rest
/// of its pipeline budget. One frame on 27 November spent 7.3 s on nine crashes of the same
/// pipeline's compiles (issue #181), about 0.85 s each; with the budget spent, that frame's other
/// new pipelines are put off and retried in later frames, where the crash memory refuses the bad
/// ones at once. A single crash can still cost the frame the ~0.85 s it takes, or on macOS 15 the
/// ~30 s the compiler retries for, out of its budget.
pub(crate) fn compile_failed(name: &str, error: &str) {
    if !victory_metal::is_compiler_crash(error) {
        return;
    }
    spend_frame_budget();
    // The frame this was last logged for: one warning a frame, however many pipelines crash.
    static WARNED: Mutex<u64> = Mutex::new(u64::MAX);
    let frame = crate::frame::current();
    let mut warned = WARNED.lock().unwrap_or_else(|e| e.into_inner());
    if *warned != frame {
        *warned = frame;
        tracing::warn!(frame, pipeline = name, "a compile crashed Metal's compiler; no new pipelines this frame");
    }
}

/// A compile that ended in a crash of Metal's compiler spends the rest of the frame's budget
/// (issue #181: nine of them held one frame for 7.3 s) and is what `compile_wait` sees, so the
/// frame is handed no new pipeline; one refused for a reason of the host's spends nothing.
#[cfg(test)]
#[test]
fn a_compiler_crash_spends_the_frame_budget() {
    fn whole_frame() {
        *BUDGET.lock().unwrap_or_else(|e| e.into_inner()) = crate::prewarm::FrameBudget::new();
    }
    struct Whole;
    impl Drop for Whole {
        fn drop(&mut self) {
            whole_frame();
        }
    }
    // The budget is the process's and every other test reads it under `TITLE`: hold it, and start
    // the frame here, so no test of theirs runs while this one has spent the budget.
    let _title = lock();
    whole_frame();
    let _whole = Whole;
    let before = frame_left_now();
    assert!(!before.is_zero(), "a frame starts with its budget");
    // A refusal of `make_render`'s, not a crash.
    compile_failed("a guessed pipeline", "a target format the host does not know");
    let left = frame_left_now();
    assert!(!left.is_zero(), "a refusal is not a crash");
    assert!(left <= before);
    assert!(!frame_spent_by_crash());

    compile_failed(
        "a guessed pipeline",
        "Compilation failed due to an interrupted connection: XPC_ERROR_CONNECTION_INTERRUPTED. \
         This error occurred after multiple retries.",
    );
    assert!(frame_left_now().is_zero(), "a crash spends the frame");
    assert!(frame_spent_by_crash(), "and `compile_wait` tells it from a frame out of time");
}

/// How long a new pipeline may hold the current frame while Metal makes it: what the frame has
/// left of its budget; past it, the few ms a pipeline the disk cache and the archive hold takes
/// (so a warm frame that outlasts its budget puts none off), until one is not made in them; then
/// nothing. A cold world frame asks for some 800 past its budget: 20 ms each would be 16 s. A
/// frame a crash spent gets nothing: the compiler service is down for the rest of it, and the
/// 20 ms window let 12 pipelines of one frame fail within 0.22 s, each refused and remembered as
/// a crash of its own.
fn compile_wait(t: &MetalTitle) -> std::time::Duration {
    const ARCHIVED: std::time::Duration = std::time::Duration::from_millis(20);
    /// The frame last past its budget, and the pipelines put off before that.
    static PAST: Mutex<(u64, u64)> = Mutex::new((0, 0));
    let left = frame_left();
    if !left.is_zero() {
        return left;
    }
    if frame_spent_by_crash() {
        return std::time::Duration::ZERO;
    }
    let frame = crate::frame::current();
    let mut past = PAST.lock().unwrap_or_else(|e| e.into_inner());
    if past.0 != frame {
        *past = (frame, t.shaders.put_off);
    }
    if past.1 == t.shaders.put_off { ARCHIVED } else { left }
}

/// Whether a compile that crashed Metal's compiler spent the current frame ([`frame_left`] is
/// zero for a frame that is merely past its time too; this tells the two apart).
fn frame_spent_by_crash() -> bool {
    BUDGET.lock().unwrap_or_else(|e| e.into_inner()).spent_by_crash()
}

/// When this frame made its first new pipeline, and pipelines put off since ([`pipe`]).
static BUDGET: Mutex<crate::prewarm::FrameBudget> = Mutex::new(crate::prewarm::FrameBudget::new());
static PUT_OFF: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Runs a translation `what` names on a thread of its own, once at a time per name, for a
/// pipeline put off past the frame's budget ([`guess`]).
fn put_off(what: String, translate: impl FnOnce() + Send + 'static) {
    static RUNNING: Mutex<Option<std::collections::HashSet<String>>> = Mutex::new(None);
    if !RUNNING.lock().unwrap_or_else(|e| e.into_inner()).get_or_insert_with(Default::default).insert(what.clone()) {
        return;
    }
    let n = PUT_OFF.fetch_add(1, Ordering::Relaxed) + 1;
    if n.is_power_of_two() {
        tracing::info!(put_off = n, "pipelines put off for their translations so far");
    }
    let _ = std::thread::Builder::new().name("shader-put-off".into()).spawn(move || {
        crate::prewarm::lower_priority();
        translate();
        if let Some(r) = RUNNING.lock().unwrap_or_else(|e| e.into_inner()).as_mut() {
            r.remove(&what);
        }
    });
}

// Pass trace (`VICTORY_TRACE_PASSES=<dir>,<skip>,<presents>`, docs/native-host.md): every draw,
// dispatch, clear and copy of a few world frames in queue order (`PASS`), the commands the host
// could not prepare (`PASSFAIL`), and at the window's last present each texture they named
// (`PASSTEX`, with a .ppm in <dir>). Off unless the variable is set: callers test `pass_on()`
// before building a line, so with it unset no line is formatted and nothing is kept.
use super::pass_on;
static PASS_KEYS: Mutex<Vec<usize>> = Mutex::new(Vec::new());
// Why the last draw on this thread was dropped, for its `PASS` line.
thread_local! { static LAST_WHY: std::cell::Cell<&'static str> = const { std::cell::Cell::new("") }; }

/// `VICTORY_TRACE_PASSES`' value: the folder for images, how many world presents to let pass
/// first, and how many presents the window lasts after the one that opens it.
fn pass_config(v: &str) -> (String, u64, i64) {
    let mut parts = v.split(',');
    let dir = parts.next().filter(|d| !d.is_empty()).unwrap_or(".").to_owned();
    let skip = parts.next().and_then(|s| s.trim().parse().ok()).unwrap_or(100);
    let presents = parts.next().and_then(|s| s.trim().parse().ok()).unwrap_or(4);
    (dir, skip, presents)
}

#[cfg(test)]
#[test]
fn pass_trace_reads_its_switch_and_stays_off_without_it() {
    assert_eq!(pass_config("/tmp/p,8,5"), ("/tmp/p".into(), 8, 5));
    assert_eq!(pass_config("/tmp/p"), ("/tmp/p".into(), 100, 4));
    assert_eq!(pass_config(",x,"), (".".into(), 100, 4));
    // No test sets `VICTORY_TRACE_PASSES`, so the window never opens and nothing is noted.
    assert!(!pass_on());
}

fn pass_tex(k: usize) -> String {
    match crate::resource::texture_info(k as This) {
        Some(i) => format!("{k:#x}:f{}:{}x{}@{:#x}", i.format, i.width, i.height, i.memory as usize),
        None => format!("{k:#x}:?"),
    }
}

/// One piece of work, as it is recorded: its shaders, targets and the textures its tables name.
fn pass_note(t: &MetalTitle, l: &mut MetalList, kind: &str, outcome: &str, compute: bool, n: u32) {
    let sh = stage_hashes(l.pipeline);
    let (reads, writes, none, keys) = if compute {
        t.host.pass_views(l.list.compute_args(), &[])
    } else {
        let (root, binding) = l.list.graphics_args();
        let mut limits: Vec<(u8, usize)> = Vec::new();
        if let Some(b) = binding.filter(|_| outcome.starts_with("ran")) {
            limits.extend(b.table_entries.iter().copied());
            if let Some(ps) = &b.ps {
                limits.extend(ps.table_entries.iter().copied());
            }
            if limits.is_empty() { limits.push((255, 0)); }
        }
        t.host.pass_views(root, &limits)
    };
    let tg: Vec<String> = if compute { Vec::new() } else { l.target_keys.iter().map(|&k| pass_tex(k)).collect() };
    {
        let mut ks = PASS_KEYS.lock().unwrap_or_else(|e| e.into_inner());
        for k in keys.iter().chain(if compute { [].iter() } else { l.target_keys.iter() }) {
            if !ks.contains(k) { ks.push(*k); }
        }
    }
    let at = l.list.cmds.len();
    l.list.notes.push((at, format!("kind={kind} n={n} out={outcome:?} sh={sh:?} tg={tg:?} rd={reads:?} wr={writes:?} none={none} topo={}", l.topology)));
}

fn pass_note_op(l: &mut MetalList, what: &str, src: Option<usize>, dst: Option<usize>) {
    {
        let mut ks = PASS_KEYS.lock().unwrap_or_else(|e| e.into_inner());
        for k in src.iter().chain(dst.iter()) {
            if !ks.contains(k) { ks.push(*k); }
        }
    }
    let at = l.list.cmds.len();
    l.list.notes.push((at, format!("kind={what} src={:?} dst={:?}", src.map(pass_tex), dst.map(pass_tex))));
}

/// At each present: switch the window on and off, and while on write every texture the window's
/// work named: its range, how much is non-zero and not finite, and (colour formats) an image.
fn pass_present(resource: This) {
    static START: OnceLock<std::time::Instant> = OnceLock::new();
    static LEFT: std::sync::atomic::AtomicI64 = std::sync::atomic::AtomicI64::new(i64::MIN);
    static NUM: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    static CONFIG: OnceLock<Option<(String, u64, i64)>> = OnceLock::new();
    let Some((dir, skip, presents)) =
        CONFIG.get_or_init(|| std::env::var("VICTORY_TRACE_PASSES").ok().map(|v| pass_config(&v)))
    else {
        return;
    };
    let (skip, presents) = (*skip, *presents);
    let num = NUM.fetch_add(1, Ordering::Relaxed);
    let now = START.get_or_init(std::time::Instant::now).elapsed().as_secs();
    let left = LEFT.load(Ordering::Relaxed);
    // `skip` counts world presents (500 draws or more recorded since the last present).
    static LASTREC: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    static WORLD: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let rec = RECORDED.load(Ordering::Relaxed);
    if rec - LASTREC.swap(rec, Ordering::Relaxed) >= 500 {
        WORLD.fetch_add(1, Ordering::Relaxed);
    }
    if left == i64::MIN {
        if WORLD.load(Ordering::Relaxed) >= skip {
            LEFT.store(presents, Ordering::Relaxed);
            super::PASS_ON.store(true, Ordering::Relaxed);
            eprintln!("PASSWINDOW start present={num} t={now}");
        }
        return;
    }
    if left < 0 {
        return;
    }
    eprintln!("PASSPRESENT present={num} left={left} resource={}", pass_tex(resource as usize));
    let dump = left == 0 || std::env::var("VICTORY_TRACE_PASSES_DUMP_EACH").is_ok();
    if left == 0 {
        super::PASS_ON.store(false, Ordering::Relaxed);
    }
    LEFT.store(left - 1, Ordering::Relaxed);
    if !dump {
        return;
    }
    let mut keys = PASS_KEYS.lock().unwrap_or_else(|e| e.into_inner()).clone();
    if !keys.contains(&(resource as usize)) { keys.push(resource as usize); }
    with_title(|t| {
        for key in keys {
            let Some(info) = crate::resource::texture_info(key as This) else { continue };
            let (w, h) = (info.width as usize, info.height as usize);
            let bpp = match info.format { 2 => 16, 10 | 11 | 16 => 8, 24 | 26 | 28 | 29 | 34 | 41 | 42 | 67 | 87 | 91 => 4, 49 | 54 | 56 | 57 => 2, 61 | 62 | 65 => 1, _ => 0 };
            // A 3D texture (a colour grading table) is read whole; arrays are skipped.
            let d = if info.dimension == victory_formats::texture::Dimension::Tex3d { info.depth_or_layers.max(1) as usize } else { 1 };
            if bpp == 0 || w * h == 0 || w * h > 3840 * 2160 || d == 1 && info.depth_or_layers > 1 && info.format != 28 {
                eprintln!("PASSTEX present={num} key={key:#x} fmt={} {w}x{h} layers={} skipped", info.format, info.depth_or_layers);
                continue;
            }
            let Some(tex) = t.texture_of(key) else {
                eprintln!("PASSTEX present={num} key={key:#x} fmt={} {w}x{h} no host texture", info.format);
                continue;
            };
            let mut raw = vec![0u8; w * h * d * bpp];
            victory_backend::resources::ResourceBackend::read_texture(&t.host.backend, &tex, 0, 0, &mut raw, w * bpp, w * h * bpp);
            let nonzero = raw.chunks_exact(bpp).filter(|p| p.iter().any(|&b| b != 0)).count() as f64 / (w * h * d) as f64;
            let mut slices = String::new();
            if d > 1 {
                let lit = raw.chunks_exact(w * h * bpp).filter(|s| s.iter().any(|&b| b != 0)).count();
                let distinct = raw.chunks_exact(bpp).collect::<std::collections::HashSet<_>>().len();
                slices = format!(" depth={d} slices_nonzero={lit} distinct_texels={distinct}");
            }
            let name = format!("p{num}_{key:x}_f{}_{w}x{h}.ppm", info.format);
            let mut out = format!("P6\n{w} {}\n255\n", h * d).into_bytes();
            let mut extra = String::new();
            match info.format {
                2 | 10 | 24 | 26 => {
                    let rgb: Vec<[f32; 3]> = raw.chunks_exact(bpp).map(|p| decode_hdr(info.format, p)).collect();
                    let (mut max, mut min, mut sum, mut bad, mut neg) = (f32::MIN, f32::MAX, 0f64, 0usize, 0usize);
                    for v in rgb.iter().flatten() {
                        if !v.is_finite() { bad += 1; continue; }
                        max = max.max(*v); min = min.min(*v); sum += f64::from(*v); neg += usize::from(*v < 0.0);
                    }
                    // format 10: count infinities and NaNs by their bits (decode_hdr folds inf).
                    let mut nan16 = 0usize;
                    if info.format == 10 {
                        nan16 = raw.chunks_exact(2).filter(|p| u16::from_le_bytes([p[0], p[1]]) & 0x7c00 == 0x7c00).count();
                    }
                    extra = format!(" max={max} min={min} mean={} notfinite={bad} infnan16={nan16} negative={neg}", sum / (rgb.len() * 3).max(1) as f64);
                    // A few texels of 32-bit floats (exposure and histogram results): every channel, alpha too.
                    if info.format == 2 && w * h <= 16 {
                        let v: Vec<f32> = raw.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect();
                        extra += &format!(" values={v:?}");
                    }
                    out.extend(rgb.iter().flat_map(|c| c.map(|v| if v.is_finite() && v > 0.0 { ((v / (1.0 + v)).sqrt() * 255.0) as u8 } else { 0 })));
                }
                28 | 29 | 87 | 91 => {
                    let bgra = matches!(info.format, 87 | 91);
                    out.extend(raw.chunks_exact(4).flat_map(|p| if bgra { [p[2], p[1], p[0]] } else { [p[0], p[1], p[2]] }));
                }
                _ => {
                    // Any other: the low byte of up to three channels, as grey or colour.
                    out.extend(raw.chunks_exact(bpp).flat_map(|p| [p[0], p[bpp / 2], p[bpp - 1]]));
                }
            }
            let _ = std::fs::write(std::path::Path::new(dir).join(&name), out);
            eprintln!("PASSTEX present={num} key={key:#x} fmt={} {w}x{h} at={:#x} nonzero={nonzero:.4}{slices}{extra} file={name}", info.format, info.memory as usize);
        }
    });
}

/// Work dropped because its pipeline has no host pipeline.
static DROPPED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// A draw into `l`'s targets that the host drops, counted per target for the trace.
fn trace_drop(l: &MetalList) {
    l.target_keys.iter().for_each(|&k| trace_bump(k, 2));
}

/// A dropped draw or execution, by what it was (its shaders and targets), for the frame trace.
pub(crate) fn trace_drop_pipe(key: String) {
    let mut d = DROP_PIPES.lock().unwrap_or_else(|e| e.into_inner());
    match d.iter_mut().find(|(k, _)| *k == key) {
        Some((_, n)) => *n += 1,
        None => d.push((key, 1)),
    }
}

/// Work dropped since the last present, by reason, for the `VICTORY_TRACE_TARGETS` frame line.
pub(crate) fn trace_reason(why: &'static str) {
    if tracing_targets() {
        let mut r = DROP_REASONS.lock().unwrap_or_else(|e| e.into_inner());
        match r.iter_mut().find(|(w, _)| *w == why) {
            Some((_, n)) => *n += 1,
            None => r.push((why, 1)),
        }
    }
}

/// Counts a draw or dispatch the host drops, and warns the first time for each `why`.
fn dropped(why: &'static str) {
    DROPPED.fetch_add(1, Ordering::Relaxed);
    if pass_on() {
        LAST_WHY.with(|w| w.set(why));
    }
    warn_once(why);
    trace_reason(why);
}

/// A warning logged once per distinct `what` for the run.
pub(crate) fn warn_once(what: &'static str) {
    static SEEN: std::sync::Mutex<Vec<&'static str>> = std::sync::Mutex::new(Vec::new());
    let mut seen = SEEN.lock().unwrap_or_else(|e| e.into_inner());
    if !seen.contains(&what) {
        seen.push(what);
        tracing::warn!("native host drops work: {what} (first time; counted in \"native draws so far\")");
    }
}

/// Draws the title recorded, and those that went on to the host list.
static RECORDED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static ENCODED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub(super) static TEXTURE_COPIES_DROPPED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
/// Presents whose resource had no host texture.
static NO_TEXTURE: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// `(draws recorded, draws handed to the host list, draws dropped for want of a host pipeline,
/// presents that found no host texture)`, for the frame log.
pub(crate) fn draw_counts() -> (u64, u64, u64, u64) {
    (
        RECORDED.load(Ordering::Relaxed),
        ENCODED.load(Ordering::Relaxed),
        DROPPED.load(Ordering::Relaxed),
        NO_TEXTURE.load(Ordering::Relaxed),
    )
}


/// The frame log's line of native draw counters, and the commands dropped by kind (kept out of
/// `PresentX`, whose frame it would otherwise enlarge).
#[inline(never)]
pub(crate) fn log_draw_counts(frame: u64) {
    let (recorded, encoded, no_pipeline, no_texture) = draw_counts();
    let [mesh, mesh_indirect] = MESH_DROPPED.each_ref().map(|c| c.load(Ordering::Relaxed));
    let copies_recorded = COPIES[0].load(Ordering::Relaxed);
    let copies = TEXTURE_COPIES_DROPPED.load(Ordering::Relaxed);
    let [mesh_drawn, mesh_refused_over_limit, mesh_refused_conversion, mesh_refused_no_pipeline] = MESH_COUNTS.each_ref().map(|c| c.load(Ordering::Relaxed));
    let indirect = crate::list::indirect_dropped();
    let bvh_builder_skipped = BVH_BUILDER_SKIPPED.load(Ordering::Relaxed);
    let (r, heap_cleared, [dropped_unset_root, dropped_unmapped, dropped_failed, typed_views, resource_heaps, twin_copies], metal_mb) = with_title(|t| {
        let h = &t.host;
        (h.backend.residency_counts(), h.cleared(), [h.unset_roots(), h.unmapped(), h.failed(), h.typed_views(), h.resource_heaps() as u64, h.twin_copies()], victory_backend::backend::ComputeBackend::allocated_bytes(&h.backend) >> 20)
    })
    .unwrap_or_default();
    // Where the memory is when command buffers stall (#79): Metal's own count and the process footprint.
    #[cfg(target_os = "macos")]
    let footprint_mb = victory_platform::mem::footprint() >> 20;
    #[cfg(not(target_os = "macos"))]
    let footprint_mb = 0usize;
    tracing::info!(frame, recorded, encoded, no_pipeline, no_texture, mesh, mesh_indirect, indirect, texture_copies = copies, copies_recorded, mesh_drawn, mesh_refused_over_limit, mesh_refused_conversion, mesh_refused_no_pipeline, bvh_builder_skipped, twin_copies, use_resource = r.use_resource, heap_resources = r.heap_resources, heap_entries = r.heap_entries, heap_allocations = r.heap_allocations, heap_set_mb = r.heap_set_bytes >> 20, metal_mb, footprint_mb, heap_cleared, dropped_unset_root, dropped_unmapped, dropped_failed, typed_views, resource_heaps, "native draws so far");
    if let Some(n) = super::host::heap_set_trace()
        && let Some([all, idle, never]) = with_title(|t| t.host.heap_set_report(n)).flatten()
    {
        let mb = |b: u64| b >> 20;
        tracing::info!(frame, idle_frames = n, textures = all.0, textures_mb = mb(all.1), buffers = all.2, buffers_mb = mb(all.3), idle_textures = idle.0, idle_textures_mb = mb(idle.1), idle_buffers = idle.2, idle_buffers_mb = mb(idle.3), never_textures = never.0, never_textures_mb = mb(never.1), never_buffers = never.2, never_buffers_mb = mb(never.3), "heap set: allocations (once over all heaps) no bound table named for that many frames (idle), or ever (never)");
    }
    if dropped_unmapped > 0 && env_set!("VICTORY_TRACE_TARGETS") {
        let (kinds, top) = with_title(|t| t.host.unmapped_kinds(12)).unwrap_or_default();
        tracing::info!(frame, kinds, ?top, "commands dropped at unmapped title memory so far, by kind");
    }
    static TRACE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    if *TRACE.get_or_init(|| std::env::var_os("VICTORY_TRACE_SHADER_CACHE").is_some_and(|v| !v.is_empty() && v != "0")) {
        let (shaders_cached, shaders_translated) = crate::shader_cache::counts();
        let (libraries_cached, libraries_converted) = victory_metal::disk::counts();
        tracing::info!(frame, shaders_cached, shaders_translated, libraries_cached, libraries_converted, "native shader cache so far");
    }
    let (typed_views, typed_views_live) =
        with_title(|t| (t.host.typed_views(), t.host.live_typed_views())).unwrap_or_default();
    if typed_views > 0 {
        tracing::info!(frame, typed_views, typed_views_live, "native typed buffer views");
    }
    super::descriptor_trace::log(frame);
}

/// `VICTORY_DEBUG_DEPTH_ALWAYS=<hash>[+w],...`: pipelines with a stage of one of these hashes
/// draw with the depth test ALWAYS (`+w`: and write depth), to tell a depth test that fails from
/// a pixel shader that writes nothing. `Some(write)` for such a pipeline.
fn debug_depth_always(stages: &[(&'static str, std::sync::Arc<crate::pipeline::Shader>)]) -> Option<bool> {
    static LIST: OnceLock<Vec<(String, bool)>> = OnceLock::new();
    let list = LIST.get_or_init(|| {
        std::env::var("VICTORY_DEBUG_DEPTH_ALWAYS")
            .map(|v| v.split(',').filter(|h| !h.is_empty()).map(|h| (h.trim_end_matches("+w").to_owned(), h.ends_with("+w"))).collect())
            .unwrap_or_default()
    });
    stages.iter().find_map(|(_, sh)| list.iter().find(|(h, _)| sh.hash.ends_with(h.as_str())).map(|&(_, w)| w))
}

/// A title pipeline object's host pipeline: its shaders translated for the host, then converted.
/// The second result says the pipeline depends on the draw's targets.
fn make_pipe(t: &mut MetalTitle, key: usize, targets: &TargetFormats, rect: bool) -> (Pipe<Metal>, bool) {
    let Some((stages, graphics)) = crate::pipeline::native_stages(key as This) else {
        return (Pipe::Refused, false);
    };
    let graphics = graphics.map(|g| {
        if g.names_targets() {
            (g, false)
        } else {
            (super::console_state::graphics_state_for(&g, targets.color.clone(), targets.depth), true)
        }
    });
    let console = graphics.as_ref().is_some_and(|g| g.1);
    if console && env_set!("VICTORY_TRACE_BIND") {
        tracing::info!(pipeline = key, ?targets, state = ?graphics.as_ref().map(|g| &g.0), "console pipeline state at its draw");
    }
    let mut graphics = graphics.map(|g| g.0);
    if let (Some(g), Some(write)) = (graphics.as_mut(), debug_depth_always(&stages)) {
        g.debug_depth_always(write);
    }
    (make_host_pipe(t, stages, graphics, rect), console)
}

/// The host pipeline of compute pipeline `key` for dispatches of thread counts, made at first
/// use: its shader translated for them (`Options::thread_counts`).
fn thread_pipe(t: &mut MetalTitle, key: usize) -> Pipe<Metal> {
    if let Some(p) = t.thread_pipes.get(&key) {
        return p.clone();
    }
    if frame_left().is_zero()
        && let Some(s) = crate::pipeline::compute_shader(key as This)
        && !s.threads_ready()
    {
        put_off(format!("{} threads", s.hash), move || drop(s.native_threads_shader()));
        return Pipe::Refused;
    }
    let (hash, shader) = crate::pipeline::native_threads(key as This);
    let name = format!("compute {hash} by thread counts");
    t.compile_wait = compile_wait(t);
    let put_off = t.shaders.put_off;
    let made = match shader {
        Ok(cs) => t.compute_pipe(&cs, &name),
        Err(e) => {
            t.refused_translation(&name, e);
            Pipe::Refused
        }
    };
    if t.shaders.put_off == put_off {
        t.thread_pipes.insert(key, made.clone());
    }
    made
}

fn make_host_pipe(
    t: &mut MetalTitle,
    stages: Vec<(&'static str, Arc<crate::pipeline::Shader>)>,
    graphics: Option<Graphics>,
    rect: bool,
) -> Pipe<Metal> {
    if stages.is_empty() && graphics.as_ref().is_some_and(|g| g.is_guessed()) {
        return Pipe::Skipped;
    }
    let name = stages
        .iter()
        .map(|(stage, s)| format!("{stage} {}", s.hash))
        .collect::<Vec<_>>()
        .join(" ");
    if let Some((_, mesh)) = stages.iter().find(|s| s.0 == "mesh") {
        if stages.iter().any(|s| !matches!(s.0, "mesh" | "pixel")) {
            t.refused_translation(&name, "console amplification/other stages are not translated".into());
            return Pipe::MeshRefused(super::title::MeshRefusal::ConversionFailed);
        }
        let pixel = stages.iter().find(|s| s.0 == "pixel").map(|s| s.1.as_ref());
        return match (graphics.as_ref(), mesh.native_linked(pixel)) {
            // Use the source mesh identity for the once-per-shader output-limit warning,
            // regardless of the stream's stage order or which pixel shader it pairs with.
            (Some(g), Ok((ms, ps))) => t.render_pipe(&ms, ps.as_deref(), g, &format!("mesh {}", mesh.hash)),
            (_, Err(e)) => {
                t.refused_translation(&name, e);
                Pipe::MeshRefused(super::title::MeshRefusal::ConversionFailed)
            }
            _ => Pipe::MeshRefused(super::title::MeshRefusal::NoPipeline),
        };
    }
    // A pixel shader drawing into integer targets is translated for them (Metal refuses a float
    // output for an integer attachment).
    let integer = graphics.as_ref().and_then(|g| g.color_formats().ok()).map_or(0, |c| super::shaders::integer_targets(&c));
    let mut hosts = Vec::new();
    for (stage, shader) in &stages {
        let rectangle = rect && *stage == "vertex" && graphics.as_ref().is_some_and(|g| !g.is_guessed());
        if rectangle && frame_left().is_zero() && !shader.rect_ready() {
            let shader = shader.clone();
            put_off(format!("{} rect", shader.hash), move || { let _ = shader.native_rect(); });
            t.shaders.put_off += 1;
            return Pipe::Refused;
        }
        let made = (if rectangle { shader.native_rect() } else { shader.native() }).map(Arc::clone).map_err(str::to_owned);
        let made = match made {
            Ok(_) if *stage == "pixel" && integer != 0 => {
                if frame_left().is_zero() && !shader.targets_cached(integer) {
                    let s = shader.clone();
                    put_off(format!("{} targets {integer}", s.hash), move || drop(s.translate_targets(integer)));
                    // Counted as put off, so `pipe` does not keep the refusal.
                    t.shaders.put_off += 1;
                    return Pipe::Refused;
                }
                shader.translate_targets(integer).map_err(|e| format!("integer targets {integer:#b}: {e}"))
            }
            made => made,
        };
        match made {
            Ok(h) => hosts.push((*stage, h)),
            Err(e) => {
                t.refused_translation(&name, e.to_string());
                return Pipe::Refused;
            }
        }
    }
    let find = |stage: &str| hosts.iter().find(|h| h.0 == stage).map(|h| &*h.1);
    match graphics {
        None => match find("compute") {
            Some(cs) => t.compute_pipe(cs, &name),
            None => Pipe::Refused,
        },
        Some(g) if g.is_guessed() => match hosts.iter().find(|h| h.0 == "vertex") {
            Some((_, vs)) => Pipe::Guess(Rc::new(Guessed {
                vs_source: stages.iter().find(|s| s.0 == "vertex").expect("a vertex stage").1.clone(),
                vs: vs.clone(),
                ps: hosts.iter().find(|h| h.0 == "pixel").map(|h| h.1.clone()),
                name,
                state: g.clone(),
                made: Default::default(),
            })),
            None => Pipe::Refused,
        },
        Some(g) => match find("vertex") {
            Some(vs) => t.render_pipe(vs, find("pixel"), &g, &name),
            None => {
                t.refused_translation(&name, "a graphics pipeline without a vertex shader".into());
                Pipe::Refused
            }
        },
    }
}

/// The shader path's counters: pipelines made and cached, refusals by reason.
pub fn shader_stats() -> Option<super::shaders::ShaderStats> {
    with_title(|t| t.shaders.clone())
}

/// Makes title pipeline `key` the compute pipeline of a pre-translated DXIL container with the
/// given group size (what a translated shader's pipeline is made of); for diagnostics and tests.
pub fn install_compute_pipeline(key: usize, dxil: &[u8], threads: [usize; 3]) -> Result<(), String> {
    with_title(|t| {
        let pso = t.host.compute_pipeline(dxil, threads)?;
        t.pipelines.insert(key, Pipe::Compute(pso));
        Ok(())
    })
    .unwrap_or_else(|| Err("no native host GPU".into()))
}

/// Requested dispatches suppressed by the builder policy; indirect counts are the maximum
/// records requested, since the count buffer belongs to the GPU and is never read on the CPU.
static BVH_BUILDER_SKIPPED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

fn skip_builder_dispatches(pipeline: usize, count: u32) -> bool {
    if crate::pipeline::is_builder(pipeline as This) && !super::shaders::builder_runs() {
        BVH_BUILDER_SKIPPED.fetch_add(u64::from(count), Ordering::Relaxed);
        return true;
    }
    false
}


/// Debugging, `VICTORY_DEBUG_FILL_INPUT=<shader hash>:<root slot>.<entry>:<byte>`: each time a
/// compute shader with that hash is dispatched, level 0 of the texture its table at that root
/// slot shows at that entry is filled with the byte, from the CPU, at once (not ordered with
/// the GPU's work: for a texture nothing else writes). Stands in for a pass the host does not
/// run yet: `5741294cf872ffb9:0.95:255` gives the tiled lighting a sun shadow mask of "lit
/// everywhere" where the tiled cascade shadow shaders that write it are refused.
fn debug_fill_input(t: &mut MetalTitle, l: &MetalList) {
    static SPEC: OnceLock<Option<(String, usize, usize, u8)>> = OnceLock::new();
    let Some((hash, slot, entry, byte)) = SPEC.get_or_init(|| {
        let v = std::env::var("VICTORY_DEBUG_FILL_INPUT").ok()?;
        let mut parts = v.split(':');
        let (hash, at, byte) = (parts.next()?, parts.next()?, parts.next()?);
        let (slot, entry) = at.split_once('.')?;
        Some((hash.to_owned(), slot.parse().ok()?, entry.parse().ok()?, byte.parse().ok()?))
    }) else {
        return;
    };
    if !stage_hashes(l.pipeline).iter().any(|s| s.ends_with(hash.as_str())) {
        return;
    }
    let Some(key) = t.host.table_texture(l.list.compute_args(), *slot, *entry) else { return };
    if !crate::resource::is_resource(key as This) {
        return;
    }
    let Some(info) = crate::resource::texture_info(key as This) else { return };
    let (Some((bytes, 1)), Some(tex)) = (crate::resource::element(info.format), t.texture_of(key)) else { return };
    let (w, h) = (info.width as usize, info.height as usize);
    let row = w * bytes as usize;
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| tracing::warn!(shader = hash, slot, entry, byte, format = info.format, w, h, "VICTORY_DEBUG_FILL_INPUT: a compute shader's input texture is filled before each of its dispatches"));
    victory_backend::resources::ResourceBackend::upload_texture(&t.host.backend, &tex, 0, 0, &vec![*byte; row * h], row, row * h);
}

#[cfg(test)]
pub(crate) fn install_test_render_pipeline(
    key: usize,
    pso: Rc<<Metal as victory_backend::resources::GraphicsBackend>::RenderPipeline>,
) {
    with_title(|t| {
        t.pipelines.insert(key, Pipe::Render(pso, None));
    });
}

pub(crate) fn dispatch(this: This, x: u32, y: u32, z: u32) {
    with_list(this, |t, l| {
        if skip_builder_dispatches(l.pipeline, 1) {
            flow_snap_compute_missed(l, "dropped (builder policy)");
            return;
        }
        match pipe(t, l.pipeline, &TargetFormats::default(), false) {
            Pipe::Compute(pso) => {
                trace_record_dispatch(t, l, "ran");
                if pass_on() {
                    pass_note(t, l, &format!("dispatch[{x},{y},{z}]"), "ran", true, 1);
                }
                trace_flow_compute(t, l, pso.binding.as_ref().map(|b| b.table_entries.as_slice()), "encoded", &format!("groups [{x}, {y}, {z}]"), None);
                debug_fill_input(t, l);
                l.list.set_compute_pipeline(&pso);
                if let Err(e) = flow_snap_compute(t, l, pso.binding.as_ref().map(|b| b.table_entries.as_slice()), ![x, y, z].contains(&0), |l| l.list.dispatch(x, y, z)) {
                    tracing::warn!(error = %e, "dispatch");
                }
            }
            p => {
                flow_snap_compute_missed(l, if matches!(p, Pipe::Refused) { "refused" } else { "dropped" });
                dropped_dispatch(t, l, &format!("groups [{x}, {y}, {z}]"), None);
            }
        }
    });
}

/// Encoded mesh dispatches and refused dispatches (over limit, conversion, no pipeline).
static MESH_COUNTS: [std::sync::atomic::AtomicU64; 4] = [const { std::sync::atomic::AtomicU64::new(0) }; 4];

/// Refused direct and indirect mesh dispatches, alongside their detailed reasons.
pub(crate) static MESH_DROPPED: [std::sync::atomic::AtomicU64; 2] = [const { std::sync::atomic::AtomicU64::new(0) }; 2];

pub(crate) fn refuse_mesh(reason: super::title::MeshRefusal, indirect: bool) {
    MESH_DROPPED[usize::from(indirect)].fetch_add(1, Ordering::Relaxed);
    use super::title::MeshRefusal::*;
    let i = match reason { OverLimit => 1, ConversionFailed => 2, NoPipeline => 3 };
    MESH_COUNTS[i].fetch_add(1, Ordering::Relaxed);
}

/// Mesh work uses graphics root arguments and targets, with counts kept on the GPU for indirect work.
pub(crate) fn dispatch_mesh(this: This, groups: [u32; 3], record: Option<u64>) {
    dispatch_mesh_from(this, groups, record, None);
}

/// `VICTORY_TRACE_MESH_COUNTS=1`: what the mesh dispatches ask for, read from title memory when
/// they are recorded: the thread-group counts of each record and the value of its count buffer.
/// The GPU writes both (culling), so these are the values of the frame before; a tally every
/// 20,000 dispatches says how many ask for no groups at all, and the first 24 are listed.
fn trace_mesh_counts(groups: [u32; 3], record: Option<u64>, count: Option<(u64, u32)>) {
    use std::sync::atomic::AtomicU64;
    static ON: OnceLock<bool> = OnceLock::new();
    if !*ON.get_or_init(|| env_set!("VICTORY_TRACE_MESH_COUNTS")) {
        return;
    }
    static N: AtomicU64 = AtomicU64::new(0);
    static EMPTY: AtomicU64 = AtomicU64::new(0);
    static PAST_COUNT: AtomicU64 = AtomicU64::new(0);
    static UNREAD: AtomicU64 = AtomicU64::new(0);
    static GROUPS: AtomicU64 = AtomicU64::new(0);
    let read3 = |at: u64| {
        let mut b = [0u8; 12];
        super::safe_read(at as *const u8, &mut b).then(|| [0, 4, 8].map(|i| u32::from_le_bytes([b[i], b[i + 1], b[i + 2], b[i + 3]])))
    };
    let asked = match record { Some(at) => read3(at), None => Some(groups) };
    let limit = count.and_then(|(at, _)| {
        let mut b = [0u8; 4];
        super::safe_read(at as *const u8, &mut b).then(|| u32::from_le_bytes(b))
    });
    let n = N.fetch_add(1, Ordering::Relaxed) + 1;
    match asked {
        None => drop(UNREAD.fetch_add(1, Ordering::Relaxed)),
        Some(g) if g.contains(&0) => drop(EMPTY.fetch_add(1, Ordering::Relaxed)),
        Some(g) => drop(GROUPS.fetch_add(g.iter().map(|&x| u64::from(x)).product(), Ordering::Relaxed)),
    }
    if let (Some((_, index)), Some(limit)) = (count, limit) && index >= limit {
        PAST_COUNT.fetch_add(1, Ordering::Relaxed);
    }
    if n <= 24 {
        tracing::info!(n, ?asked, record = record.map(|r| format!("{r:#x}")), count_value = ?limit, index = count.map(|c| c.1), "mesh dispatch asked");
    }
    if n % 20_000 == 0 {
        tracing::info!(
            dispatches = n,
            no_groups = EMPTY.load(Ordering::Relaxed),
            past_count = PAST_COUNT.load(Ordering::Relaxed),
            unread = UNREAD.load(Ordering::Relaxed),
            groups = GROUPS.load(Ordering::Relaxed),
            "mesh dispatches so far (values of the frame before)"
        );
    }
}

fn dispatch_mesh_from(this: This, groups: [u32; 3], record: Option<u64>, count: Option<(u64, u32)>) {
    RECORDED.fetch_add(1, Ordering::Relaxed);
    trace_mesh_counts(groups, record, count);
    with_list(this, |t, l| {
        use super::title::MeshRefusal;
        let targets = l.targets.clone();
        // A described mesh stage (a stream pipeline) is made by `pipe`; a guessed pipeline whose
        // vertex program exports primitives is made for this dispatch's targets by `guess`.
        // Both are mesh pipelines of the host: `Pipe::Render` with a mesh thread-group size.
        let made = match pipe(t, l.pipeline, &targets, false) {
            Pipe::Guess(g) if l.targets_ready => guess(t, l, &g, false),
            p => p,
        };
        let failure = match made {
            Pipe::Render(pso, binding) if l.targets_ready && pso.mesh_threads().is_some() => {
                // A mesh shader declares its own topology and ignores the list's.
                l.list.set_graphics_pipeline(&pso, victory_backend::resources::Topology::Triangles);
                l.list.set_graphics_binding(binding);
                if pass_on() {
                    pass_note(t, l, &format!("mesh{groups:?} record={record:?}"), "ran", false, 1);
                }
                let [x, y, z] = groups;
                let result = match record {
                    Some(record) => l.list.dispatch_mesh_indirect(record, count),
                    None => l.list.dispatch_mesh(x, y, z),
                };
                match result {
                    Ok(()) => {
                        ENCODED.fetch_add(1, Ordering::Relaxed);
                        MESH_COUNTS[0].fetch_add(1, Ordering::Relaxed);
                        l.target_keys.iter().for_each(|&k| trace_bump(k, 0));
                        trace_present_draw(t, l);
                        return;
                    }
                    Err(e) => tracing::warn!(error = %e, "mesh dispatch"),
                }
                MeshRefusal::NoPipeline
            }
            Pipe::MeshRefused(r) => r,
            Pipe::Skipped => {
                warn_once("a mesh dispatch whose guessed 0x1c8 stage registers are undecoded");
                MeshRefusal::NoPipeline
            }
            _ => MeshRefusal::NoPipeline,
        };
        // The frame trace's "dropped" of the targets this dispatch would have drawn into.
        if pass_on() {
            pass_note(t, l, "mesh", &format!("refused {failure:?}"), false, 1);
        }
        trace_drop(l);
        refuse_mesh(failure, record.is_some());
    });
}

/// `VICTORY_TRACE_FLOW=<seconds>[,<frames>]`: every compute dispatch and indirect mesh execution
/// recorded in `frames` (default 2) presents, from the first present `seconds` after the first
/// one on: its shaders, whether it was encoded or dropped, its group counts or the address and
/// values of its indirect record (read from title memory as it is when recorded), and the views
/// its root parameters name (`NativeHost::root_views`). The lines are "flow: ..." in recording
/// order, so what writes the buffers a later dispatch reads can be followed through a frame.
fn flow_capturing() -> bool {
    FLOW_ON.load(Ordering::Relaxed)
}

static FLOW_ON: AtomicBool = AtomicBool::new(false);
/// The indirect records the capture saw, to be read again once the GPU has run the frame
/// (`flow_late`): sequence number, whether a mesh execution's, address.
static FLOW_RECORDS: Mutex<Vec<(u32, bool, u64)>> = Mutex::new(Vec::new());
static FLOW_SEQ: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
/// The R32_FLOAT textures the captured compute work names (depth pyramids among them), read
/// back with the late records.
static FLOW_TEXTURES: Mutex<Vec<FlowTexture>> = Mutex::new(Vec::new());

/// Keep the view's level-0 extent and address: placed keys have no resource object to query later.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct FlowTexture {
    key: usize,
    address: u64,
    format: u32,
    width: usize,
    height: usize,
    depth: usize,
    dim: victory_backend::resources::TexDim,
}

/// Both `sN.M tex 0x...` and `uN.M placed tex 0x...` from `NativeHost::root_views`.
fn flow_texture_view(view: &str) -> Option<(&str, FlowTexture)> {
    let (head, rest) = view.split_once(" tex 0x")?;
    let name = head.strip_suffix(" placed").unwrap_or(head);
    let (slot, entry) = name.strip_prefix(['s', 'u'])?.split_once('.')?;
    slot.parse::<usize>().ok()?;
    entry.parse::<usize>().ok()?;
    let (key, rest) = rest.split_once(" at 0x")?;
    let (address, rest) = rest.split_once(" fmt ")?;
    let (format, rest) = rest.split_once(' ')?;
    let (size, mip) = rest.split_once(" mip ")?;
    let (width, height) = size.split_once('x')?;
    mip.parse::<usize>().ok()?;
    Some((name, FlowTexture {
        key: usize::from_str_radix(key, 16).ok()?,
        address: u64::from_str_radix(address, 16).ok()?,
        format: format.parse().ok()?,
        width: width.parse().ok()?,
        height: height.parse().ok()?,
        depth: 1,
        dim: victory_backend::resources::TexDim::D2,
    }))
}

fn flow_resource_texture(key: usize) -> Option<FlowTexture> {
    // `texture_info` dereferences an object; a placed key must never reach it.
    let info = crate::resource::is_resource(key as This).then(|| crate::resource::texture_info(key as This)).flatten()?;
    let desc = info.desc()?;
    Some(FlowTexture {
        key, address: info.memory as u64, format: info.format, width: desc.width, height: desc.height,
        depth: if desc.dim == victory_backend::resources::TexDim::D3 { desc.depth_or_layers } else { 1 },
        dim: desc.dim,
    })
}

/// Canonical DXGI numbers for placed keys, whose format comes from `Title::format_of`.
fn flow_dxgi_format(format: victory_backend::resources::Format) -> u32 {
    use victory_backend::resources::Format::*;
    match format {
        Rgba8Unorm => 28,
        Rgba8UnormSrgb => 29,
        Bgra8Unorm => 87,
        Bgra8UnormSrgb => 91,
        Rgba16Float => 10,
        R32Float => 41,
        D32Float => 40,
        D32FloatS8 => 20,
        Bc1 => 71,
        Bc2 => 74,
        Bc3 => 77,
        Bc4 => 80,
        Bc5 => 83,
        Bc6h => 95,
        Bc7 => 98,
        R8Unorm => 61,
        A8Unorm => 65,
        R16Unorm => 56,
        Rgba32Float => 2,
        Rg32Float => 16,
        Rgb10a2Unorm => 24,
        Rg11b10Float => 26,
        Rgba8Uint => 30,
        Rg16Float => 34,
        Rg16Unorm => 35,
        R32Uint => 42,
        R8Uint => 62,
        Rgb9e5Float => 67,
        Bc7Srgb => 99,
        Rgba32Uint => 3,
        Rgba16Unorm => 11,
        Rgba16Snorm => 13,
        Rg32Uint => 17,
        Rg16Uint => 36,
        Rg8Unorm => 49,
        Rg8Uint => 50,
        Rg8Snorm => 51,
        R16Float => 54,
        R16Uint => 57,
        D16Unorm => 55,
        Bc1Srgb => 72,
        Bc3Srgb => 78,
        Bgr5a1Unorm => 86,
        Abgr4Unorm => 115,
        Bc6hSigned => 96,
        // Formats only buffer views use have no texture statistics.
        _ => 0,
    }
}

/// Called after realization or registry lookup. Shared placed views keep their named format.
fn flow_texture_info(t: &MetalTitle, named: FlowTexture) -> Option<FlowTexture> {
    let desc = t.texture_desc(named.key)?;
    Some(FlowTexture {
        format: t.format_of(named.key).map_or(named.format, flow_dxgi_format),
        width: desc.width,
        height: desc.height,
        depth: if desc.dim == victory_backend::resources::TexDim::D3 { desc.depth_or_layers } else { 1 },
        dim: desc.dim,
        ..flow_resource_texture(named.key).unwrap_or(named)
    })
}

fn flow_note_textures(views: &[String]) {
    let mut seen = FLOW_TEXTURES.lock().unwrap_or_else(|e| e.into_inner());
    for (_, texture) in views.iter().filter_map(|v| flow_texture_view(v)) {
        if texture.format == 41
            && !seen.iter().any(|t| t.key == texture.key)
            && seen.len() < 12
        {
            seen.push(texture);
        }
    }
}

/// Level 0 of each noted texture: its range, the share of texels that are 0 and that are 1 or
/// more, and the mean of each cell of an 8 x 4 grid.
fn flow_late_textures(after: u32) {
    let keys = FLOW_TEXTURES.lock().unwrap_or_else(|e| e.into_inner()).clone();
    with_title(|t| {
        for named in keys {
            let key = named.key;
            let Some(tex) = t.texture_of(key).or_else(|| t.host.texture_at(key)) else { continue };
            let Some(info) = flow_texture_info(t, named) else { continue };
            let (w, h, depth) = (info.width, info.height, info.depth);
            if info.format != 41 || flow_texels([w, h, depth]).is_none() {
                continue;
            }
            let mut raw = vec![0u8; w * h * depth * 4];
            victory_backend::resources::ResourceBackend::read_texture(&t.host.backend, &tex, 0, 0, &mut raw, w * 4, w * h * 4);
            let px: Vec<f32> = raw.chunks_exact(4).map(|p| f32::from_le_bytes([p[0], p[1], p[2], p[3]])).collect();
            let (mut min, mut max, mut zero, mut one) = (f32::INFINITY, f32::NEG_INFINITY, 0usize, 0usize);
            for &v in &px {
                min = min.min(v);
                max = max.max(v);
                zero += usize::from(v == 0.0);
                one += usize::from(v >= 1.0);
            }
            let grid: Vec<String> = (0..4)
                .map(|gy| {
                    (0..8)
                        .map(|gx| {
                            let (x0, x1, y0, y1) = (gx * w / 8, ((gx + 1) * w / 8).max(gx * w / 8 + 1), gy * h / 4, ((gy + 1) * h / 4).max(gy * h / 4 + 1));
                            let (mut sum, mut n) = (0f64, 0u32);
                            for z in 0..depth {
                                for y in y0..y1.min(h) {
                                    for x in x0..x1.min(w) {
                                        sum += f64::from(px[(z * h + y) * w + x]);
                                        n += 1;
                                    }
                                }
                            }
                            format!("{:.5}", sum / f64::from(n.max(1)))
                        })
                        .collect::<Vec<_>>()
                        .join(" ")
                })
                .collect();
            let n = (w * h * depth) as f64;
            tracing::info!(after, key = format!("{key:#x}"), at = format!("{:#x}", info.address), w, h, depth = format!("depth {depth}"), min, max, zero = zero as f64 / n, one_or_more = one as f64 / n, ?grid, "flow: late texture");
            // The other levels (a depth pyramid's coarse levels are what large objects are tested
            // against): each level's range, and its texels where there are 32 or fewer.
            let levels = crate::resource::is_resource(key as This)
                .then(|| crate::resource::texture_info(key as This)).flatten()
                .map_or(1, |i| i.layout.mips.len().min(12));
            let mut ranges = Vec::new();
            for mip in 1..levels {
                let (mw, mh, md) = ((w >> mip).max(1), (h >> mip).max(1), (depth >> mip).max(1));
                let mut raw = vec![0u8; mw * mh * md * 4];
                victory_backend::resources::ResourceBackend::read_texture(&t.host.backend, &tex, mip, 0, &mut raw, mw * 4, mw * mh * 4);
                let px: Vec<f32> = raw.chunks_exact(4).map(|p| f32::from_le_bytes([p[0], p[1], p[2], p[3]])).collect();
                let lo = px.iter().copied().fold(f32::INFINITY, f32::min);
                let hi = px.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let zero = px.iter().filter(|v| **v == 0.0).count() as f64 / px.len().max(1) as f64;
                let nan = px.iter().filter(|v| !v.is_finite()).count();
                let texels = if px.len() <= 32 { format!(" {px:?}") } else { String::new() };
                ranges.push(format!("mip {mip} {mw}x{mh} depth {md} min {lo} max {hi} zero {zero:.4} not_finite {nan}{texels}"));
            }
            if !ranges.is_empty() {
                tracing::info!(after, key = format!("{key:#x}"), ?ranges, "flow: late texture levels");
            }
        }
    });
}

/// The textures the captured draws into HDR targets name or draw into (any format), read back
/// with the late records (`flow_late_draw_textures`).
static FLOW_DRAW_TEXTURES: Mutex<Vec<FlowTexture>> = Mutex::new(Vec::new());

/// `VICTORY_TRACE_FLOW_DUMP=<hash>,<hash>`: for these shaders, the words (at most 96) of every
/// buffer of `limit` bytes or fewer they name, as title memory has them when the dispatch or
/// draw is recorded (constants the title's CPU wrote).
fn flow_buffer_words(hashes: &[String], views: &[String], limit: u64) {
    static DUMP: OnceLock<Vec<String>> = OnceLock::new();
    let dump = DUMP.get_or_init(|| {
        std::env::var("VICTORY_TRACE_FLOW_DUMP").map(|v| v.split(',').map(str::to_owned).collect()).unwrap_or_default()
    });
    if !hashes.iter().any(|h| dump.iter().any(|d| h.ends_with(d.as_str()))) {
        return;
    }
    for v in views {
        let mut parts = v.split(' ');
        let (Some(name), Some(range)) = (parts.next(), parts.next()) else { continue };
        let Some((at, len)) = range.split_once('+') else { continue };
        let parse = |x: &str| u64::from_str_radix(x.trim_start_matches("0x"), 16).ok();
        let (Some(at), Some(len)) = (parse(at), parse(len)) else { continue };
        if len == 0 || len > limit {
            continue;
        }
        let mut bytes = vec![0u8; len.min(384) as usize];
        if super::safe_read(at as *const u8, &mut bytes) {
            let words: Vec<String> = bytes.chunks_exact(4).map(|b| format!("{:08x}", u32::from_le_bytes([b[0], b[1], b[2], b[3]]))).collect();
            tracing::info!(shader = ?hashes, view = name, at = format!("{at:#x}"), words = words.join(" "), "flow: buffer words");
        }
    }
}

/// `VICTORY_TRACE_FLOW` with `VICTORY_TRACE_FLOW_DRAWS=1`: every draw the capture records into
/// a set of targets with an HDR colour target (DXGI 10 or 26): its shaders, the call, whether
/// it was encoded, its targets and the views its root parameters name; the pipeline's depth,
/// stencil, cull and colour state the first time the capture meets it.
fn trace_flow_draw(t: &MetalTitle, l: &MetalList, state: &str, how: &str) {
    if !flow_capturing() || !env_set!("VICTORY_TRACE_FLOW_DRAWS") {
        return;
    }
    let hdr = l.target_keys.iter().any(|&k| crate::resource::texture_info(k as This).is_some_and(|i| matches!(i.format, 10 | 26)));
    if !hdr {
        return;
    }
    let hashes = stage_hashes(l.pipeline);
    let (root, binding) = l.list.graphics_args();
    let entries: Option<Vec<(u8, usize)>> =
        binding.map(|b| b.table_entries.iter().chain(b.ps.iter().flat_map(|ps| &ps.table_entries)).copied().collect());
    let views = t.host.root_views(root, entries.as_deref());
    let targets: Vec<String> = l
        .target_keys
        .iter()
        .map(|&k| {
            let i = crate::resource::texture_info(k as This);
            let (f, w, h) = i.map_or((0, 0, 0), |i| (i.format, i.width, i.height));
            format!("{k:#x} fmt {f} {w}x{h}")
        })
        .collect();
    {
        let mut seen = FLOW_DRAW_TEXTURES.lock().unwrap_or_else(|e| e.into_inner());
        let named = views.iter().filter_map(|v| flow_texture_view(v).map(|(_, t)| t));
        for texture in l.target_keys.iter().filter_map(|&k| flow_resource_texture(k)).chain(named) {
            if !seen.iter().any(|t| t.key == texture.key) && seen.len() < 400 {
                seen.push(texture);
            }
        }
    }
    static STATES: Mutex<Vec<(usize, TargetFormats)>> = Mutex::new(Vec::new());
    let id = (l.pipeline, l.targets.clone());
    let new = {
        let mut seen = STATES.lock().unwrap_or_else(|e| e.into_inner());
        !seen.contains(&id) && {
            seen.push(id);
            true
        }
    };
    let seq = flow_seq(false, None);
    if new && let Some((_, Some(g))) = crate::pipeline::native_stages(l.pipeline as This) {
        let described = g.names_targets();
        let g = if described { g } else { super::console_state::graphics_state_for(&g, l.targets.color.clone(), l.targets.depth) };
        tracing::info!(seq, pipeline = ?hashes, described, guessed = g.is_guessed(), depth = ?g.depth_test(), stencil = ?g.stencil(), cull = ?g.cull(), color = ?g.color_state(), "flow: draw pipeline state");
    }
    tracing::info!(seq, pipeline = ?hashes, state, how, topology = l.topology, ?targets, depth_target = ?l.targets.depth, ?views, "flow: draw");
    // The constants of the first few draws of the shaders `VICTORY_TRACE_FLOW_DUMP` names.
    static DUMPED: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
    if DUMPED.load(Ordering::Relaxed) < 6 {
        let constants: Vec<String> = views.iter().filter(|v| v.starts_with('c')).cloned().collect();
        flow_buffer_words(&hashes, &constants, 0x1000);
        DUMPED.fetch_add(1, Ordering::Relaxed);
    }
}

/// Where in a `w` x `h` level the texels `hit` names lie: 8 rows of 14 cells, each the
/// percentage of its texels that are hits (a tile of the title's 7 x 4 lighting tiles is two
/// cells by two, so a half-drawn tile shows as three cells of its four).
fn texel_grid(w: usize, h: usize, hit: impl Fn(usize) -> bool) -> Vec<String> {
    let (cols, rows) = (14usize, 8usize);
    let mut cells = vec![(0u32, 0u32); cols * rows];
    for y in 0..h {
        for x in 0..w {
            let c = &mut cells[(y * rows / h.max(1)) * cols + x * cols / w.max(1)];
            c.1 += 1;
            c.0 += u32::from(hit(y * w + x));
        }
    }
    cells.chunks(cols).map(|r| r.iter().map(|&(n, of)| format!("{:3}", n * 100 / of.max(1))).collect::<Vec<_>>().join("")).collect()
}

/// The eight most common values of `values`, with their counts.
fn top_values(values: impl Iterator<Item = u64>) -> String {
    let mut counts = std::collections::HashMap::<u64, u32>::new();
    values.for_each(|v| *counts.entry(v).or_default() += 1);
    let mut list: Vec<_> = counts.into_iter().collect();
    list.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    list.iter().take(8).map(|(v, n)| format!("{v:x}x{n}")).collect::<Vec<_>>().join(" ")
}

/// What a texture level holds, for the flow trace: the share of texels that are not zero, the
/// bitwise OR of all texels (first 8 bytes), every texel if there are 64 or fewer, else the
/// eight most common texels with their counts (hexadecimal, as little-endian words).
fn texel_stats(raw: &[u8], bpp: usize) -> String {
    let texels = raw.len() / bpp.max(1);
    let hex = |p: &[u8]| p.iter().rev().map(|b| format!("{b:02x}")).collect::<String>();
    let nonzero = raw.chunks_exact(bpp).filter(|p| p.iter().any(|&b| b != 0)).count();
    let mut or = [0u8; 8];
    for p in raw.chunks_exact(bpp) {
        p.iter().take(8).enumerate().for_each(|(i, b)| or[i] |= b);
    }
    let detail = if texels <= 64 {
        format!("all [{}]", raw.chunks_exact(bpp).map(hex).collect::<Vec<_>>().join(" "))
    } else {
        let mut counts: std::collections::HashMap<&[u8], usize> = std::collections::HashMap::new();
        for p in raw.chunks_exact(bpp) {
            *counts.entry(p).or_default() += 1;
        }
        let distinct = counts.len();
        let mut top: Vec<_> = counts.into_iter().collect();
        top.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(b.0)));
        top.truncate(8);
        format!("distinct {distinct} top [{}]", top.iter().map(|(p, n)| format!("{}x{n}", hex(p))).collect::<Vec<_>>().join(" "))
    };
    format!("nonzero {:.5} or {} {detail}", nonzero as f64 / texels.max(1) as f64, hex(&or[..bpp.min(8)]))
}

/// Level 0 of each texture the captured HDR draws name (and the compute shaders of
/// `VICTORY_TRACE_FLOW_TEXTURES=<hash>,...`), of every depth twin plane, and of the copies
/// `VICTORY_TRACE_FLOW_SNAP` made: [`texel_stats`], and the largest finite channel of the HDR
/// formats (DXGI 10, 26).
fn flow_late_draw_textures(after: u32) {
    let keys = FLOW_DRAW_TEXTURES.lock().unwrap_or_else(|e| e.into_inner()).clone();
    with_title(|t| {
        let hdr_max = |format: u32, raw: &[u8], bpp: usize| {
            matches!(format, 10 | 26).then(|| raw.chunks_exact(bpp).flat_map(|p| decode_hdr(format, p)).filter(|v| v.is_finite()).fold(0f32, f32::max))
        };
        for named in keys {
            let key = named.key;
            let Some(tex) = t.texture_of(key).or_else(|| t.host.texture_at(key)) else { continue };
            let Some(info) = flow_texture_info(t, named) else { continue };
            let (w, h, depth) = (info.width, info.height, info.depth);
            let Some((bytes, block)) = crate::resource::element(info.format) else { continue };
            // Depth cannot be read back this way; block formats are source art.
            if block != 1 || matches!(info.format, 19 | 20 | 39 | 40 | 44..=47 | 53 | 55 | 118..=120) || flow_texels([w, h, depth]).is_none() {
                continue;
            }
            let bpp = bytes as usize;
            let mut raw = vec![0u8; w * h * depth * bpp];
            victory_backend::resources::ResourceBackend::read_texture(&t.host.backend, &tex, 0, 0, &mut raw, w * bpp, w * h * bpp);
            tracing::info!(after, key = format!("{key:#x}"), at = format!("{:#x}", info.address), format = info.format, w, h, depth = format!("depth {depth}"), stats = texel_stats(&raw, bpp), hdr_max = ?hdr_max(info.format, &raw, bpp), "flow: late draw texture");
            // The title's material number is the low five bits of a G-buffer target's alpha.
            if after == 1 && matches!(info.format, 28 | 29) && w * h > 1 << 16 {
                tracing::info!(key = format!("{key:#x}"), format = info.format, alpha = top_values(raw.chunks_exact(4).map(|p| u64::from(p[3]))), alpha_low5 = top_values(raw.chunks_exact(4).map(|p| u64::from(p[3] & 31))), "flow: late draw texture alpha");
            }
            if after == 1 && depth == 1 && matches!(info.format, 10) && w * h > 1 << 16 {
                tracing::info!(key = format!("{key:#x}"), w, h, grid = ?texel_grid(w, h, |i| raw[i * bpp..][..bpp].iter().any(|&b| b != 0)), "flow: late draw texture where not zero");
            }
        }
        if after == 1 {
            for (key, plane, tex, [w, h, bpp]) in t.host.twin_planes() {
                if bpp == 0 || w * h > 1 << 22 {
                    continue;
                }
                let info = crate::resource::is_resource(key as This).then(|| crate::resource::texture_info(key as This)).flatten();
                let mut raw = vec![0u8; w * h * bpp];
                victory_backend::resources::ResourceBackend::read_texture(&t.host.backend, &tex, 0, 0, &mut raw, w * bpp, w * h * bpp);
                tracing::info!(key = format!("{key:#x}"), at = format!("{:#x}", info.as_ref().map_or(0, |i| i.memory as usize)), format = info.as_ref().map_or(0, |i| i.format), plane, w, h, stats = texel_stats(&raw, bpp), "flow: late twin plane");
            }
            // Each twin's depth texture next to its twin's depth plane, both as they are now:
            // the texels that differ, what the depth texture holds there (with
            // `VICTORY_DEBUG_DEPTH_ALWAYS=<hash>+w` the depth those draws rasterised), and where.
            for (key, depth, plane, [w, h, bpp]) in t.host.twin_depths() {
                if bpp != 4 || w * h == 0 || w * h > 1 << 22 {
                    continue;
                }
                let (mut d, mut p) = (vec![0u8; w * h * 4], vec![0u8; w * h * 4]);
                if !t.host.backend.debug_read_depth(&depth, &mut d) {
                    continue;
                }
                victory_backend::resources::ResourceBackend::read_texture(&t.host.backend, &plane, 0, 0, &mut p, w * 4, w * h * 4);
                let word = |b: &[u8], i: usize| u32::from_le_bytes([b[i * 4], b[i * 4 + 1], b[i * 4 + 2], b[i * 4 + 3]]);
                let differ: Vec<usize> = (0..w * h).filter(|&i| word(&d, i) != word(&p, i)).collect();
                tracing::info!(
                    key = format!("{key:#x}"),
                    w,
                    h,
                    differ = differ.len() as f64 / (w * h) as f64,
                    depth = texel_stats(&d, 4),
                    depth_where_differs = top_values(differ.iter().map(|&i| u64::from(word(&d, i)))),
                    twin_where_differs = top_values(differ.iter().map(|&i| u64::from(word(&p, i)))),
                    grid = ?texel_grid(w, h, |i| word(&d, i) != word(&p, i)),
                    "flow: late depth texture against its twin"
                );
            }
            for (seq, what, tex, format, [w, h, depth]) in std::mem::take(&mut t.flow_snaps) {
                let Some((bytes, _)) = crate::resource::element(format) else { continue };
                let Some(texels) = flow_texels([w, h, depth]) else { continue };
                let bpp = bytes as usize;
                let mut raw = vec![0u8; texels * bpp];
                victory_backend::resources::ResourceBackend::read_texture(&t.host.backend, &tex, 0, 0, &mut raw, w * bpp, w * h * bpp);
                tracing::info!(seq, what, format, w, h, depth = format!("depth {depth}"), stats = texel_stats(&raw, bpp), hdr_max = ?hdr_max(format, &raw, bpp), grid = ?(depth == 1).then(|| texel_grid(w, h, |i| raw[i * bpp..][..bpp].iter().any(|&b| b != 0))), "flow: target as the GPU had it");
            }
        }
    });
}

/// `VICTORY_TRACE_FLOW_SNAP=<hash>,...`: after the 1st, 30th, 60th, 90th and 120th captured
/// draw of a pipeline with one of these stage hashes (`VICTORY_TRACE_FLOW_SNAP_AT=<n>,<n>,...`
/// names other draws, at most 16), a copy of its first colour target, made on the GPU where the
/// draw is (read back after the capture: "flow: target as the GPU had it"). With pixel shader
/// hashes the count is of those pipelines' draws alone.
fn flow_snap_target(t: &mut MetalTitle, l: &mut MetalList) {
    static AT: OnceLock<Vec<u32>> = OnceLock::new();
    static DRAWS: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
    let hashes = flow_snap_hashes();
    if hashes.is_empty() || !flow_capturing() {
        return;
    }
    let stages = stage_hashes(l.pipeline);
    if !stages.iter().any(|s| hashes.iter().any(|h| s.ends_with(h.as_str()))) {
        return;
    }
    let n = DRAWS.fetch_add(1, Ordering::Relaxed) + 1;
    let at = AT.get_or_init(|| snap_draws(std::env::var("VICTORY_TRACE_FLOW_SNAP_AT").ok().as_deref()));
    if !at.contains(&n) {
        return;
    }
    let Some(&key) = l.target_keys.first() else { return };
    let Some(info) = crate::resource::texture_info(key as This) else { return };
    let (Some(desc), Some(src)) = (info.desc(), t.texture_of(key)) else { return };
    let (w, h) = (desc.width, desc.height);
    if !flow_snap_copyable(desc.format, [w, h, 1]) { return };
    let Ok(copy) = t.host.texture(&victory_backend::resources::TextureDesc { mips: 1, depth_or_layers: 1, render_target: false, unordered_access: false, ..desc }) else { return };
    l.list.copy_texture_region(&src, 0, 0, [0; 3], [w, h, 1], &copy, 0, 0, [0; 3]);
    let seq = FLOW_SEQ.load(Ordering::Relaxed);
    t.flow_snaps.push((seq, format!("draw {n} of {stages:?}"), copy, info.format, [w, h, 1]));
}

fn flow_snap_hashes() -> &'static [String] {
    static HASHES: OnceLock<Vec<String>> = OnceLock::new();
    HASHES.get_or_init(|| std::env::var("VICTORY_TRACE_FLOW_SNAP").map(|v| v.split(',').map(str::to_owned).collect()).unwrap_or_default())
}

/// Claim a compute hash once, independently of the draw counter and `_SNAP_AT`.
fn snap_compute_hash(stages: &[String], hashes: &[String], seen: &mut Vec<String>) -> Option<String> {
    let hash = stages.iter().filter_map(|s| s.strip_prefix("compute "))
        .find(|s| hashes.iter().any(|h| s.ends_with(h.as_str())))?;
    if seen.iter().any(|s| s == hash) {
        return None;
    }
    seen.push(hash.to_owned());
    Some(hash.to_owned())
}

fn flow_snap_compute_hash(pipeline: usize) -> Option<String> {
    if !flow_capturing() || flow_snap_hashes().is_empty() {
        return None;
    }
    static SEEN: Mutex<Vec<String>> = Mutex::new(Vec::new());
    snap_compute_hash(&stage_hashes(pipeline), flow_snap_hashes(), &mut SEEN.lock().unwrap_or_else(|e| e.into_inner()))
}

fn flow_snap_compute_missed(l: &MetalList, state: &str) {
    if let Some(hash) = flow_snap_compute_hash(l.pipeline) {
        tracing::info!(hash, state, "flow: snapshot dispatch was not encoded");
    }
}

/// Only table UAV textures, with repeated views of one key represented by the first name.
fn flow_uav_textures(views: &[String]) -> Vec<(&str, FlowTexture)> {
    let mut seen = std::collections::HashSet::new();
    views.iter().filter_map(|v| flow_texture_view(v))
        .filter(|(name, texture)| name.starts_with('u') && seen.insert(texture.key)).collect()
}

/// Whole mips only: never mistake a truncated readback for empty depth slices.
fn flow_texels([w, h, depth]: [usize; 3]) -> Option<usize> {
    w.checked_mul(h)?.checked_mul(depth).filter(|&n| n > 0 && n <= 1 << 22)
}

fn flow_snap_copyable(format: victory_backend::resources::Format, size: [usize; 3]) -> bool {
    !format.is_depth() && format.block().0 == 1 && flow_texels(size).is_some()
}

fn flow_snap_copy(t: &mut MetalTitle, l: &mut MetalList, src: &Rc<victory_metal::gfx::Texture>, info: FlowTexture, what: String) {
    let Some(format) = super::descriptors::format(info.format) else { return };
    let size = [info.width, info.height, info.depth];
    if !flow_snap_copyable(format, size) { return };
    let desc = victory_backend::resources::TextureDesc {
        dim: info.dim,
        depth_or_layers: info.depth,
        ..victory_backend::resources::TextureDesc::d2(format, info.width, info.height)
    };
    let Ok(copy) = t.host.texture(&desc) else { return };
    l.list.copy_texture_region(src, 0, 0, [0; 3], size, &copy, 0, 0, [0; 3]);
    t.flow_snaps.push((FLOW_SEQ.load(Ordering::Relaxed), what, copy, info.format, size));
}

/// Bracket the first captured dispatch of each requested compute hash, including one indirect
/// record, with at most four level-0 UAV copies on each side. `CopyTexture` commands split the
/// host's compute batches into separate blit / compute / blit passes, so no copy is encoded
/// inside a compute encoder that still binds the source as a UAV.
fn flow_snap_compute(
    t: &mut MetalTitle,
    l: &mut MetalList,
    entries: Option<&[(u8, usize)]>,
    has_work: bool,
    encode: impl FnOnce(&mut MetalList) -> Result<(), String>,
) -> Result<(), String> {
    let Some(hash) = flow_snap_compute_hash(l.pipeline) else { return encode(l) };
    if !has_work {
        tracing::info!(hash, state = "zero groups", "flow: snapshot dispatch was not encoded");
        return encode(l);
    }
    let views = t.host.root_views(l.list.compute_args(), entries);
    let mut sources = Vec::new();
    let textures: Vec<_> = flow_uav_textures(&views).into_iter().filter_map(|(name, named)| {
        let src = t.texture_of(named.key).or_else(|| t.host.texture_at(named.key))?;
        let info = flow_texture_info(t, named)?;
        let format = super::descriptors::format(info.format)?;
        if !flow_snap_copyable(format, [info.width, info.height, info.depth]) || sources.iter().any(|s| Rc::ptr_eq(s, &src)) {
            return None;
        }
        sources.push(src.clone());
        Some((name, info, src))
    }).take(4).collect();
    let start = t.flow_snaps.len();
    for (name, info, src) in &textures {
        flow_snap_copy(t, l, src, *info, format!("before compute {hash} {name}"));
    }
    let result = encode(l);
    if let Err(error) = &result {
        // The diagnostic copies stay ordered in the list, but are not reported as a dispatch.
        t.flow_snaps.truncate(start);
        tracing::info!(hash, state = "refused", %error, "flow: snapshot dispatch was not encoded");
    } else {
        for (name, info, src) in &textures {
            flow_snap_copy(t, l, src, *info, format!("after compute {hash} {name}"));
        }
    }
    result
}

/// The draws `VICTORY_TRACE_FLOW_SNAP` copies the target after: the numbers `list` names (from
/// 1, at most 16 of them), or the 1st, 30th, 60th, 90th and 120th.
fn snap_draws(list: Option<&str>) -> Vec<u32> {
    let named: Vec<u32> =
        list.map(|v| v.split(',').filter_map(|n| n.trim().parse().ok()).filter(|&n| n > 0).take(16).collect()).unwrap_or_default();
    if named.is_empty() { vec![1, 30, 60, 90, 120] } else { named }
}

fn flow_seq(mesh: bool, record: Option<u64>) -> u32 {
    let seq = FLOW_SEQ.fetch_add(1, Ordering::Relaxed);
    if let Some(r) = record {
        FLOW_RECORDS.lock().unwrap_or_else(|e| e.into_inner()).push((seq, mesh, r));
    }
    seq
}

/// The captured records as title memory has them `after` presents after the capture ended:
/// what the GPU wrote there, where the GPU writes them. One line per compute record, and the
/// mesh records as counts (asking for no groups, asking for some, unreadable) with the sequence
/// numbers of up to 400 that ask for some.
fn flow_late(after: u32) {
    let records = FLOW_RECORDS.lock().unwrap_or_else(|e| e.into_inner()).clone();
    let (mut none, mut some, mut unread, mut groups) = (0u32, Vec::new(), 0u32, 0u64);
    let mut persistent = (0u32, 0u32);
    for (seq, mesh, at) in records {
        let now = flow_words(at);
        if !mesh {
            tracing::info!(after, seq, record = format!("{at:#x}"), ?now, "flow: late compute record");
            continue;
        }
        // Records in the title's persistent buffers (the high pools here) still hold what the
        // GPU wrote; per-frame memory may have been reused by now.
        if at >= 0xa_0000_0000 {
            match now {
                Some(g) if !g.contains(&0) => persistent.1 += 1,
                Some(_) => persistent.0 += 1,
                None => {}
            }
        }
        match now {
            None => unread += 1,
            Some(g) if g.contains(&0) => none += 1,
            Some(g) => {
                groups += g.iter().map(|&x| u64::from(x)).product::<u64>();
                some.push((seq, g[0]));
            }
        }
    }
    some.truncate(400);
    tracing::info!(after, none, some = some.len(), unread, persistent_none = persistent.0, persistent_some = persistent.1, groups, with_groups = ?some, "flow: late mesh records");
}

/// At a present: starts and stops the capture.
fn flow_present() {
    use std::sync::atomic::AtomicU32;
    static OPTIONS: OnceLock<Option<(u64, u32)>> = OnceLock::new();
    static START: OnceLock<std::time::Instant> = OnceLock::new();
    static LEFT: AtomicU32 = AtomicU32::new(u32::MAX);
    let Some((after, frames)) = *OPTIONS.get_or_init(|| {
        let v = std::env::var("VICTORY_TRACE_FLOW").ok()?;
        let mut parts = v.split(',');
        let after = parts.next()?.parse().ok()?;
        Some((after, parts.next().and_then(|f| f.parse().ok()).unwrap_or(2)))
    }) else {
        return;
    };
    if START.get_or_init(std::time::Instant::now).elapsed().as_secs() < after {
        return;
    }
    match LEFT.load(Ordering::Relaxed) {
        u32::MAX => {
            LEFT.store(frames, Ordering::Relaxed);
            FLOW_ON.store(true, Ordering::Relaxed);
            tracing::info!("flow: capture starts");
        }
        0 => {
            // The presents after the capture: the records again, three times.
            static AFTER: AtomicU32 = AtomicU32::new(0);
            let after = AFTER.fetch_add(1, Ordering::Relaxed) + 1;
            if after <= 3 {
                flow_late(after);
                flow_late_snapshots(after);
                flow_late_textures(after);
                flow_late_draw_textures(after);
            }
        }
        n => {
            LEFT.store(n - 1, Ordering::Relaxed);
            tracing::info!(left = n - 1, "flow: present");
            if n == 1 {
                FLOW_ON.store(false, Ordering::Relaxed);
            }
        }
    }
}

fn flow_words(at: u64) -> Option<[u32; 3]> {
    let mut b = [0u8; 12];
    super::safe_read(at as *const u8, &mut b).then(|| [0, 4, 8].map(|i| u32::from_le_bytes([b[i], b[i + 1], b[i + 2], b[i + 3]])))
}

/// Record pages as the GPU has them when the captured mesh work reads them. The values of
/// persistent record buffers read from the CPU say little: the title's reset pass and its
/// cullers rewrite them every frame, and the GPU runs behind the recording. So the first time a
/// captured mesh execution takes a record from a 64 KiB page for a set of targets, a GPU copy of
/// the page into memory of the host's own is recorded just before it, in order with the cullers
/// that wrote it; [`flow_late_snapshots`] reads the copies.
struct FlowSnapshots {
    /// The copies' memory (a pool of its own in the pool table, shown at its host address).
    base: usize,
    /// `(sequence number of the mesh execution, title page, targets, first target)` per slot.
    taken: Vec<(u32, u64, usize, usize)>,
    seen: std::collections::HashSet<(u64, Vec<usize>)>,
}

const FLOW_SNAPSHOT_SLOTS: usize = 128;
const FLOW_PAGE: usize = 1 << 16;
static FLOW_SNAPSHOTS: Mutex<Option<FlowSnapshots>> = Mutex::new(None);

/// Records the copy of `records`' page for `targets` if this capture has none yet.
fn flow_snapshot(l: &mut MetalList, seq: u32, records: u64) {
    use super::memory::{PoolInfo, PoolObserver};
    let mut guard = FLOW_SNAPSHOTS.lock().unwrap_or_else(|e| e.into_inner());
    if guard.is_none() {
        let len = FLOW_SNAPSHOT_SLOTS * FLOW_PAGE;
        let Some(region) = victory_metal::mem::Region::new(len, None) else { return };
        let base = region.ptr as usize;
        // Kept for the run: the GPU writes it.
        std::mem::forget(region);
        let info = PoolInfo { id: usize::MAX - 0xF10, base, len, memory_entry: None };
        pools().pool_created(&info);
        pools().mapped(base, len, &info, 0);
        *guard = Some(FlowSnapshots { base, taken: Vec::new(), seen: Default::default() });
    }
    let snaps = guard.as_mut().expect("made above");
    let page = records & !(FLOW_PAGE as u64 - 1);
    if snaps.taken.len() >= FLOW_SNAPSHOT_SLOTS || !snaps.seen.insert((page, l.target_keys.clone())) {
        return;
    }
    let slot = snaps.taken.len();
    snaps.taken.push((seq, page, l.target_keys.len(), l.target_keys.first().copied().unwrap_or(0)));
    l.list.copy_buffer_region((snaps.base + slot * FLOW_PAGE) as u64, page, FLOW_PAGE);
}

/// The snapshot pages: for each, the captured mesh records in it from its execution on (up to
/// the next snapshot of the same page), how many ask for groups, and the first of those.
fn flow_late_snapshots(after: u32) {
    let guard = FLOW_SNAPSHOTS.lock().unwrap_or_else(|e| e.into_inner());
    let Some(snaps) = guard.as_ref() else { return };
    let records = FLOW_RECORDS.lock().unwrap_or_else(|e| e.into_inner()).clone();
    for (slot, &(seq, page, targets, first)) in snaps.taken.iter().enumerate() {
        let until = snaps.taken[slot + 1..].iter().find(|s| s.1 == page).map_or(u32::MAX, |s| s.0);
        let (mut n, mut some, mut groups, mut list) = (0u32, 0u32, 0u64, Vec::new());
        for &(rseq, mesh, at) in &records {
            if !mesh || rseq < seq || rseq >= until || at & !(FLOW_PAGE as u64 - 1) != page {
                continue;
            }
            n += 1;
            let copy = snaps.base + slot * FLOW_PAGE + (at - page) as usize;
            if (at - page) as usize + 12 > FLOW_PAGE {
                continue;
            }
            // SAFETY: inside the snapshot memory, which is never unmapped.
            let g: [u32; 3] = unsafe { (copy as *const [u32; 3]).read_unaligned() };
            if !g.contains(&0) {
                some += 1;
                groups = groups.saturating_add(g.iter().map(|&x| u64::from(x)).fold(1u64, u64::saturating_mul));
                if list.len() < 40 {
                    list.push((rseq, g[0]));
                }
            }
        }
        tracing::info!(after, slot, seq, page = format!("{page:#x}"), targets, first_target = format!("{first:#x}"), records = n, with_groups = some, groups, first = ?list, "flow: record page as the GPU had it");
    }
}

fn trace_flow_mesh(this: This, records: u64, stride: u32, max: u32) {
    if !flow_capturing() {
        return;
    }
    with_list(this, |t, l| {
        let hashes = stage_hashes(l.pipeline);
        let (root, binding) = l.list.graphics_args();
        let entries: Option<Vec<(u8, usize)>> = binding.map(|b| {
            b.table_entries.iter().chain(b.ps.iter().flat_map(|ps| &ps.table_entries)).copied().collect()
        });
        let views = t.host.root_views(root, entries.as_deref());
        let targets: Vec<String> = l.target_keys.iter().map(|k| format!("{k:#x}")).collect();
        let seq = flow_seq(true, Some(records));
        flow_snapshot(l, seq, records);
        // The pipeline's depth and stencil state, the first time the capture meets it.
        static STATES: Mutex<Vec<usize>> = Mutex::new(Vec::new());
        let new = {
            let mut seen = STATES.lock().unwrap_or_else(|e| e.into_inner());
            !seen.contains(&l.pipeline) && {
                seen.push(l.pipeline);
                true
            }
        };
        if new && let Some((_, Some(g))) = crate::pipeline::native_stages(l.pipeline as This) {
            tracing::info!(seq, pipeline = ?hashes, depth = ?g.depth_test(), stencil = ?g.stencil(), color = ?g.color_state(), "flow: mesh pipeline state");
        }
        tracing::info!(seq, pipeline = ?hashes, record = format!("{records:#x}"), stride, max, asked = ?flow_words(records), ?targets, ?views, "flow: mesh");
    });
}

fn trace_flow_compute(t: &MetalTitle, l: &MetalList, entries: Option<&[(u8, usize)]>, state: &str, how: &str, record: Option<u64>) {
    if !flow_capturing() {
        return;
    }
    let hashes = stage_hashes(l.pipeline);
    let views = t.host.root_views(l.list.compute_args(), entries);
    flow_note_textures(&views);
    // `VICTORY_TRACE_FLOW_TEXTURES=<hash>,...`: every texture these shaders name is read back
    // with the draws' ("flow: late draw texture").
    static NOTE: OnceLock<Vec<String>> = OnceLock::new();
    let note = NOTE.get_or_init(|| std::env::var("VICTORY_TRACE_FLOW_TEXTURES").map(|v| v.split(',').map(str::to_owned).collect()).unwrap_or_default());
    if hashes.iter().any(|h| note.iter().any(|d| h.ends_with(d.as_str()))) {
        let mut seen = FLOW_DRAW_TEXTURES.lock().unwrap_or_else(|e| e.into_inner());
        for (_, texture) in views.iter().filter_map(|v| flow_texture_view(v)) {
            if !seen.iter().any(|t| t.key == texture.key) && seen.len() < 400 {
                seen.push(texture);
            }
        }
    }
    flow_buffer_words(&hashes, &views, 0x200);
    let seq = flow_seq(false, record);
    tracing::info!(seq, shader = ?hashes, state, how, record = record.map(|r| format!("{r:#x}")), asked = ?record.and_then(flow_words), ?views, "flow: compute");
}

/// A dispatch dropped for want of a host compute pipeline; with `VICTORY_TRACE_TARGETS`, counted
/// by its shader, `how` (group counts, or indirect) and the textures its tables hold as
/// unordered-access views (what it may have written) for the frame line.
fn dropped_dispatch(t: &MetalTitle, l: &mut MetalList, how: &str, record: Option<u64>) {
    dropped("a dispatch without a host compute pipeline");
    trace_flow_compute(t, l, None, "dropped", how, record);
    if pass_on() {
        pass_note(t, l, &format!("dispatch {how}"), "a dispatch without a host compute pipeline", true, 1);
    }
    trace_record_dispatch(t, l, "dropped");
    if tracing_targets() {
        let uavs: Vec<String> = t
            .host
            .table_uav_textures(l.list.compute_args())
            .into_iter()
            .map(|r| {
                let i = crate::resource::texture_info(r as This);
                let (f, w, h) = i.map_or((0, 0, 0), |i| (i.format, i.width, i.height));
                format!("{r:#x} fmt {f} {w}x{h}")
            })
            .collect();
        trace_drop_pipe(format!("{:?} {how} uavs {uavs:?}", stage_hashes(l.pipeline)));
    }
}

/// The stages of title pipeline `pipeline` with their shader hashes (for traces).
fn stage_hashes(pipeline: usize) -> Vec<String> {
    crate::pipeline::native_stages(pipeline as This)
        .map(|(stages, _)| stages.iter().map(|(s, sh)| format!("{s} {}", sh.hash)).collect())
        .unwrap_or_default()
}

/// `DispatchX` with thread counts (`crate::list::dispatch_x`). Without the translation for
/// them it is dropped: run as group counts, its lanes past the counts would write over memory
/// that is not theirs.
pub(crate) fn dispatch_threads(this: This, threads: [u32; 3]) {
    with_list(this, |t, l| {
        if skip_builder_dispatches(l.pipeline, 1) {
            flow_snap_compute_missed(l, "dropped (builder policy)");
            return;
        }
        match thread_pipe(t, l.pipeline) {
            Pipe::Compute(pso) => {
                if pass_on() {
                    pass_note(t, l, &format!("dispatch-threads{threads:?}"), "ran", true, 1);
                }
                trace_flow_compute(t, l, pso.binding.as_ref().map(|b| b.table_entries.as_slice()), "encoded", &format!("threads {threads:?}"), None);
                l.list.set_compute_pipeline(&pso);
                if let Err(e) = flow_snap_compute(t, l, pso.binding.as_ref().map(|b| b.table_entries.as_slice()), !threads.contains(&0), |l| l.list.dispatch_threads(threads)) {
                    tracing::warn!(error = %e, "dispatch of thread counts");
                }
            }
            p => {
                dropped("a dispatch of thread counts without a host compute pipeline");
                if pass_on() {
                    pass_note(t, l, &format!("dispatch-threads{threads:?}"), "a dispatch of thread counts without a host compute pipeline", true, 1);
                }
                flow_snap_compute_missed(l, if matches!(p, Pipe::Refused) { "refused" } else { "dropped" });
            }
        }
    });
}

/// `IASetPrimitiveTopology`.
pub(crate) fn topology(this: This, topology: u32) {
    with_list(this, |_, l| l.topology = topology);
}

/// `RSSetViewports`: the first one.
pub(crate) fn viewports(this: This, first: Option<[f32; 6]>) {
    if tracing_targets() {
        static N: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
        if N.fetch_add(1, Ordering::Relaxed) < 8 {
            tracing::info!(?first, "viewport set");
        }
    }
    with_list(this, |_, l| {
        l.list.set_viewport(first.map(|v| victory_backend::resources::Viewport {
            x: v[0],
            y: v[1],
            width: v[2],
            height: v[3],
            min_depth: v[4],
            max_depth: v[5],
        }))
    });
}

/// `RSSetScissorRects`: the first one (left, top, right, bottom).
pub(crate) fn scissors(this: This, first: Option<[i32; 4]>) {
    with_list(this, |_, l| l.list.set_scissor(first));
}

pub(crate) fn blend_factor(this: This, factor: Option<[f32; 4]>) {
    with_list(this, |_, l| l.list.set_blend_factor(factor.unwrap_or([1.0; 4])));
}

pub(crate) fn stencil_ref(this: This, reference: u32) {
    with_list(this, |_, l| l.list.set_stencil_ref(reference));
}

/// `IASetIndexBuffer`: the buffer's address and whether its indices are 32 bits.
pub(crate) fn index_buffer(this: This, view: Option<(u64, bool)>) {
    with_list(this, |_, l| match view {
        Some((address, wide)) => l.list.set_index_buffer(address, wide),
        None => l.list.clear_index_buffer(),
    });
}

/// Title ABI for `D3D12_VERTEX_BUFFER_VIEW`.
#[repr(C)]
pub(crate) struct VertexBufferView {
    address: u64,
    size: u32,
    stride: u32,
}

pub(crate) fn vertex_buffers(this: This, start: u32, count: u32, views: &[VertexBufferView]) {
    with_list(this, |_, l| {
        for i in 0..count as usize {
            let slot = start as usize + i;
            let (address, stride) = match views.get(i) {
                Some(v) if v.address != 0 && v.size != 0 => (v.address, v.stride),
                _ => (0, 0),
            };
            l.list.set_vertex_buffer(slot, address);
            if l.strides.len() <= slot {
                l.strides.resize(slot + 1, 0);
            }
            l.strides[slot] = stride;
        }
    });
}

/// The host pipeline of a guessed pipeline for the targets and vertex buffers the list has bound.
fn guess(t: &mut MetalTitle, l: &MetalList, g: &Guessed<Metal>, rect: bool) -> Pipe<Metal> {
    let key = format!("{:?}{rect}", l.strides);
    if let Some(p) = g.made.borrow().get(&key) {
        return p.clone();
    }
    // As in `pipe`: past the frame's budget, a translation not made yet goes to a worker and the
    // draw is dropped this time (nothing is cached; the worker's result is, on disk or in the
    // shader).
    let spent = frame_left().is_zero();
    if spent && rect && !g.vs_source.rect_ready() {
        let s = g.vs_source.clone();
        put_off(format!("{} rect", s.hash), move || drop(s.native_rect()));
        return Pipe::Refused;
    }
    let rect_vs = if rect {
        match g.vs_source.native_rect() {
            Ok(h) => Some(h.clone()),
            Err(e) => {
                t.refused_translation(&g.name, format!("rectangle list: {e}"));
                return Pipe::Refused;
            }
        }
    } else {
        None
    };
    // Parameters the pixel shader reads and the vertex shader does not write: written (undefined)
    // too, as the converter refuses a pipeline whose fragment inputs the vertex stage lacks.
    let base_vs = rect_vs.as_deref().unwrap_or(&g.vs);
    let extra = g.ps.as_ref().map_or(0, |ps| {
        super::state::texcoords(&ps.dxil, victory_shader::FourCC::ISG1)
            & !super::state::texcoords(&base_vs.dxil, victory_shader::FourCC::OSG1)
    });
    let widened = if extra != 0 {
        let mut copy = [0u8; 32];
        for &(orig, new) in g.ps.as_ref().map_or(&[][..], |ps| &ps.splits[..]) {
            if orig < 32 && new < 32 {
                copy[new as usize] = orig as u8 + 1;
            }
        }
        if spent && !g.vs_source.extra_cached(rect, extra, copy) {
            let s = g.vs_source.clone();
            put_off(format!("{} {rect} {extra} {copy:?}", s.hash), move || drop(s.translate_extra(rect, extra, copy)));
            return Pipe::Refused;
        }
        match g.vs_source.translate_extra(rect, extra, copy) {
            Ok(h) => Some(h),
            Err(e) => {
                t.refused_translation(&g.name, format!("extra outputs: {e}"));
                return Pipe::Refused;
            }
        }
    } else {
        None
    };
    let vs = widened.as_deref().unwrap_or(base_vs);
    let reads = super::state::vertex_inputs(&vs.dxil).unwrap_or_default();
    let state = g.state.for_draw(l.targets.color.clone(), l.targets.depth, l.strides.clone(), &reads, rect);
    t.compile_wait = compile_wait(t);
    let put_off = t.shaders.put_off;
    let made = t.render_pipe(vs, g.ps.as_deref(), &state, &g.name);
    // Put off, it is asked for again at every draw until Metal has made it: logged once made.
    if t.shaders.put_off == put_off {
        tracing::info!(name = %g.name, ?reads, targets = ?l.targets, strides = ?l.strides, "guessed pipeline");
        if tracing_targets() {
            tracing::info!(name = %g.name, ?state, depth_test = ?state.depth_test(), stencil = ?state.stencil(), cull = ?state.cull(), color = ?state.color_state(), "guessed pipeline state");
        }
        g.made.borrow_mut().insert(key, made.clone());
    }
    made
}

/// Sets pipeline, topology and bindings. Returns whether the draw uses guessed rectangle strips.
/// `mesh`: the call is a mesh dispatch, which only a mesh pipeline runs (and a draw only a vertex
/// pipeline).
fn graphics(t: &mut MetalTitle, l: &mut MetalList, mesh: bool) -> Option<bool> {
    if !pass_on() {
        return graphics_inner(t, l, mesh);
    }
    LAST_WHY.with(|w| w.set(""));
    let r = graphics_inner(t, l, mesh);
    let why = LAST_WHY.with(|w| w.get());
    let out = match r {
        Some(true) => "ran-rect",
        Some(false) => "ran",
        None if why.is_empty() => "dropped: a mesh pipeline without a vertex stage",
        None => why,
    };
    pass_note(t, l, if mesh { "mesh" } else { "draw" }, out, false, 1);
    r
}

fn graphics_inner(t: &mut MetalTitle, l: &mut MetalList, mesh: bool) -> Option<bool> {
    if !l.targets_ready {
        dropped("a draw of a pass whose render targets the host cannot make");
        trace_drop(l);
        return None;
    }
    // Guessed pipelines keep their attribute-based strips. Described pipelines replay the
    // vertex program and complete its outputs; the backend expands their draw arguments.
    let rect = l.topology == 17;
    let targets = l.targets.clone();
    let (made, strip) = match pipe(t, l.pipeline, &targets, rect) {
        Pipe::Guess(g) => (guess(t, l, &g, rect), rect),
        Pipe::Skipped => {
            warn_once("a draw of a guessed pipeline without a vertex stage (a mesh pipeline)");
            trace_reason("a mesh pipeline");
            trace_drop(l);
            return None;
        }
        p => (p, false),
    };
    let Pipe::Render(pso, binding) = made else {
        dropped("a draw without a host render pipeline (refused or not translated)");
        trace_drop(l);
        if tracing_targets() {
            trace_drop_pipe(format!("{:?} into {:?}", stage_hashes(l.pipeline), l.targets));
        }
        return None;
    };
    if pso.mesh_threads().is_some() != mesh {
        dropped(if mesh { "a mesh dispatch of a vertex pipeline" } else { "a draw of a mesh pipeline" });
        trace_drop(l);
        return None;
    }
    let topology = if strip {
        Some(victory_backend::resources::Topology::TriangleStrip)
    } else if rect && !mesh {
        Some(victory_backend::resources::Topology::Rectangles)
    } else if mesh {
        // A mesh shader declares its own topology and ignores the list's.
        Some(victory_backend::resources::Topology::Triangles)
    } else {
        super::state::topology(l.topology)
    };
    let Some(topology) = topology else {
        static SEEN: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        if l.topology < 64 && SEEN.fetch_or(1 << l.topology, Ordering::Relaxed) >> l.topology & 1 == 0 {
            tracing::warn!(topology = l.topology, "a draw dropped for its primitive topology");
        }
        dropped("a draw of a primitive topology the host does not draw");
        trace_drop(l);
        return None;
    };
    l.list.set_graphics_pipeline(&pso, topology);
    l.list.set_graphics_binding(binding);
    Some(strip)
}

pub(crate) fn draw(this: This, vertices: u32, instances: u32, first: u32, first_instance: u32) {
    RECORDED.fetch_add(1, Ordering::Relaxed);
    if tracing_targets() {
        static N: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
        static START: OnceLock<std::time::Instant> = OnceLock::new();
        let late = START.get_or_init(std::time::Instant::now).elapsed().as_secs() >= 26;
        let n = N.fetch_add(1, Ordering::Relaxed);
        if n < 8 || (late && n % 7 == 0 && n < 4000) {
            with_list(this, |_, l| {
                let mut head = [0u8; 72];
                let at = l.list.vertex_buffer_address(0);
                let ok = at.is_some_and(|a| super::safe_read(a as *const u8, &mut head));
                tracing::info!(vertices, instances, first, topology = l.topology, vb0 = ?at, ok, head = ?head, strides = ?l.strides, formats = ?l.targets, "draw");
            });
        }
    }
    // Debugging: only indexed draws (the content), not the boot passes.
    if env_set!("VICTORY_DEBUG_ONLY_INDEXED") {
        return;
    }
    with_list(this, |t, l| {
        let made = graphics(t, l, false);
        trace_flow_draw(t, l, if made.is_some() { "encoded" } else { "dropped" }, &format!("draw vertices {vertices} instances {instances} first {first} first_instance {first_instance}"));
        if let Some(rect) = made {
            if rect && env_set!("VICTORY_TRACE_RECTS") {
                static SEEN: Mutex<Vec<usize>> = Mutex::new(Vec::new());
                let mut seen = SEEN.lock().unwrap();
                if seen.iter().filter(|&&p| p == l.pipeline as usize).count() < 3 {
                    seen.push(l.pipeline as usize);
                    let stride = l.strides.first().copied().unwrap_or(0) as usize;
                    let mut head = vec![0u8; (stride * 3).clamp(12, 192)];
                    let at = l.list.vertex_buffer_address(0).map(|a| a + (first as u64) * stride as u64);
                    let ok = at.is_some_and(|a| super::safe_read(a as *const u8, &mut head));
                    let floats: Vec<f32> = head.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();
                    tracing::info!(pipeline = ?l.pipeline, vertices, first, strides = ?l.strides, ok, ?floats, "rect draw");
                }
            }
            let call = if rect {
                l.list.draw_instanced(4, vertices / 3, 0, first / 3)
            } else {
                l.list.draw_instanced(vertices, instances, first, first_instance)
            };
            match call {
                Ok(()) => {
                    ENCODED.fetch_add(1, Ordering::Relaxed);
                    trace_present_draw(t, l);
                    l.target_keys.iter().for_each(|&k| trace_bump(k, 0));
                }
                Err(e) => tracing::warn!(error = %e, "draw"),
            }
        }
    });
}

/// A direct mesh dispatch (`DispatchMesh`, or `DispatchMeshX` without an argument address): the
/// bound mesh pipeline runs `[x, y, z]` groups, as an indirect one runs its record's (#353). With
/// no mesh pipeline it is dropped for the reason `graphics` gives, and counted in the frame line's
/// `mesh=` as before.

pub(crate) fn draw_indexed(this: This, indices: u32, instances: u32, first: u32, base: i32, first_instance: u32) {
    RECORDED.fetch_add(1, Ordering::Relaxed);
    with_list(this, |t, l| {
        let made = graphics(t, l, false);
        trace_flow_draw(t, l, if made.is_some() { "encoded" } else { "dropped" }, &format!("indexed indices {indices} instances {instances} first {first} base {base}"));
        if let Some(rect) = made {
            if rect || l.topology == 17 {
                // Indexed expansion also needs the other two indices, not just a vertex ID.
                dropped("an indexed rectangle-list draw");
                if pass_on() {
                    pass_note_op(l, "DROPPED-indexed-rect-list", None, None);
                }
                trace_drop(l);
                return;
            }
            match l.list.draw_indexed_instanced(indices, instances, first, base, first_instance) {
                Ok(()) => {
                    ENCODED.fetch_add(1, Ordering::Relaxed);
                    trace_present_draw(t, l);
                    l.target_keys.iter().for_each(|&k| trace_bump(k, 0));
                }
                Err(e) => tracing::warn!(error = %e, "indexed draw"),
            }
        }
    });
}

/// `ExecuteIndirect` of a single-command signature: `max` draws or dispatches, record `i` at title
/// address `records + i * stride`, each read on the GPU; with a count buffer at title address
/// `count`, record `i` runs only if `i` is below the count, also read on the GPU (D3D12 runs
/// min(count, max) records). False when the host cannot run it; the caller logs
/// that. Indexed rectangle lists drop here.
pub(crate) fn execute_indirect(
    this: This,
    command: crate::signature::Command,
    stride: u32,
    max: u32,
    records: u64,
    count: Option<u64>,
) -> bool {
    use crate::signature::Command;
    let at = |i: u32| records + u64::from(i) * u64::from(stride);
    match command {
        Command::DispatchMesh => {
            trace_flow_mesh(this, records, stride, max);
            if tracing_record_writers() && record_writers_of().is_none() {
                let zero = (0..max)
                    .filter(|&i| super::textures::read_memory(at(i) as *const u8, 4).iter().all(|&b| b == 0))
                    .count() as u64;
                let mut trace = RECORD_TRACE.lock().unwrap_or_else(|e| e.into_inner());
                let n = trace.get_or_insert_default().executions.entry(at(0)).or_default();
                n[0] += u64::from(max);
                n[1] += zero;
            }
            for i in 0..max { dispatch_mesh_from(this, [0; 3], Some(at(i)), count.map(|c| (c, i))); }
            true
        },
        Command::Dispatch => {
            with_list(this, |t, l| {
                if skip_builder_dispatches(l.pipeline, max) {
                    if max > 0 {
                        flow_snap_compute_missed(l, "dropped (builder policy)");
                    }
                    return;
                }
                match pipe(t, l.pipeline, &TargetFormats::default(), false) {
                    Pipe::Compute(pso) => {
                        trace_record_dispatch(t, l, "ran");
                        if pass_on() {
                            pass_note(t, l, "indirect dispatch", "ran", true, max);
                        }
                        trace_flow_compute(t, l, pso.binding.as_ref().map(|b| b.table_entries.as_slice()), "encoded", "indirect", Some(records));
                        l.list.set_compute_pipeline(&pso);
                        for i in 0..max {
                        if let Some(of) = record_writers_of()
                            && stage_hashes(l.pipeline).iter().any(|h| h.strip_prefix("compute ").is_some_and(|h| of.contains(h)))
                        {
                            let zero = super::textures::read_memory(at(i) as *const u8, 4).iter().all(|&b| b == 0);
                            let mut trace = RECORD_TRACE.lock().unwrap_or_else(|e| e.into_inner());
                            let n = trace.get_or_insert_default().executions.entry(at(i)).or_default();
                            n[0] += 1;
                            n[1] += u64::from(zero);
                        }
                        if crate::list::trace_dispatch_flags() {
                            static SEEN: Mutex<Vec<(usize, bool)>> = Mutex::new(Vec::new());
                            let mut seen = SEEN.lock().unwrap();
                            let key = (l.pipeline as usize, count.is_some());
                            if seen.iter().filter(|k| **k == key).count() < 3 {
                                seen.push(key);
                                // What title memory holds now, not what the GPU will read.
                                let now: Vec<u32> = super::textures::read_memory(at(i) as *const u8, 12)
                                    .chunks_exact(4)
                                    .map(|w| u32::from_le_bytes(w.try_into().unwrap()))
                                    .collect();
                                tracing::warn!(shader = ?stage_hashes(l.pipeline), ?command, counted = count.is_some(), stride, record = format_args!("{:#x}", at(i)), ?now, "an indirect dispatch (the first three of each pipeline)");
                            }
                        }
                            let r = flow_snap_compute(t, l, pso.binding.as_ref().map(|b| b.table_entries.as_slice()), true, |l| match count {
                                Some(c) => l.list.dispatch_indirect_counted(at(i), c, i),
                                None => l.list.dispatch_indirect(at(i)),
                            });
                            if let Err(e) = r {
                                tracing::warn!(error = %e, "indirect dispatch");
                            }
                        }
                    }
                    p => {
                        // A zero maximum records no dispatch, so it cannot consume the hash's snapshot.
                        if max > 0 {
                            flow_snap_compute_missed(l, if matches!(p, Pipe::Refused) { "refused" } else { "dropped" });
                        }
                        dropped_dispatch(t, l, "indirect", Some(records));
                    }
                }
            });
            true
        }
        Command::Draw | Command::DrawIndexed => {
            RECORDED.fetch_add(u64::from(max), Ordering::Relaxed);
            with_list(this, |t, l| match graphics(t, l, false) {
                // shortcut: an indexed rectangle list is not drawn, as directly.
                Some(rect) if (rect || l.topology == 17) && command == Command::DrawIndexed => {
                    dropped("an indirect rectangle-list draw");
                    if pass_on() {
                        pass_note_op(l, &format!("DROPPED-indirect-rect-list n={max}"), None, None);
                    }
                    trace_drop(l);
                    true
                }
                Some(rect) => {
                    trace_flow_draw(t, l, "encoded", &format!("indirect max {max} records {records:#x} now {:?}", flow_words(records)));
                    for i in 0..max {
                        let indexed = command == Command::DrawIndexed;
                        let r = match count {
                            // As a direct one: four-vertex instances, the record rewritten on the GPU.
                            _ if rect => l.list.draw_rects_indirect(at(i), count.map(|c| (c, i))),
                            Some(c) => l.list.draw_indirect_counted(at(i), indexed, c, i),
                            None => l.list.draw_indirect(at(i), indexed),
                        };
                        match r {
                            Ok(()) => {
                                ENCODED.fetch_add(1, Ordering::Relaxed);
                                l.target_keys.iter().for_each(|&k| trace_bump(k, 0));
                                flow_snap_target(t, l);
                            }
                            Err(e) => tracing::warn!(error = %e, "indirect draw"),
                        }
                    }
                    true
                }
                None => true,
            })
            .unwrap_or(true)
        }
    }
}

pub(crate) fn root_constants(this: This, compute: bool, slot: u32, offset: u32, words: &[u32]) {
    with_list(this, |_, l| {
        l.list
            .set_root_constants_at(compute, slot as usize, offset as usize, words)
    });
}

pub(crate) fn root_view(this: This, compute: bool, slot: u32, address: u64) {
    with_list(this, |_, l| {
        if compute {
            l.list.set_compute_root_view(slot as usize, address)
        } else {
            l.list.set_graphics_root_view(slot as usize, address)
        }
    });
}

pub(crate) fn root_table(this: This, compute: bool, slot: u32, handle: u64) {
    with_list(this, |t, l| {
        let Some((heap, index)) = heap_of(t, handle) else {
            return;
        };
        if compute {
            l.list.set_compute_root_table(slot as usize, &heap, index)
        } else {
            l.list.set_graphics_root_table(slot as usize, &heap, index)
        }
    });
}

/// `SetDescriptorHeaps`: the title's heap objects.
pub(crate) fn set_heaps(this: This, heaps: &[This]) {
    with_list(this, |t, l| {
        let (mut views, mut samplers) = (None, None);
        for &h in heaps {
            let Some((start, sampler, _)) = crate::descriptor::span(h) else {
                continue;
            };
            let Some((heap, _)) = heap_of(t, start) else {
                continue;
            };
            if sampler {
                samplers = Some(heap);
            } else {
                views = Some(heap);
            }
        }
        l.list.set_heaps(views.as_ref(), samplers.as_ref());
    });
}

/// `CopyBufferRegion` between title buffers.
pub(crate) fn copy_buffer(this: This, dst: (This, u64), src: (This, u64), len: u64) {
    let (Some((to, to_size)), Some((from, from_size))) = (
        crate::resource::buffer_bytes(dst.0),
        crate::resource::buffer_bytes(src.0),
    ) else {
        return;
    };
    if dst.1.saturating_add(len) > to_size || src.1.saturating_add(len) > from_size {
        tracing::warn!(len, "copy outside its buffer");
        return;
    }
    // SAFETY: inside the buffers, checked above.
    let (to, from) = unsafe { (to.add(dst.1 as usize), from.add(src.1 as usize)) };
    copy_memory(this, to as u64, from as u64, len);
}

/// `CopyMemoryToMemoryX` and buffer copies: title addresses.
pub(crate) fn copy_memory(this: This, dst: u64, src: u64, len: u64) {
    if flow_capturing() {
        let seq = flow_seq(false, None);
        tracing::info!(seq, dst = format!("{dst:#x}"), src = format!("{src:#x}"), len, "flow: copy");
    }
    with_list(this, |_, l| {
        l.list.copy_buffer_region(dst, src, len as usize);
    });
}

fn subresource_mip_slice(info: &super::textures::TextureInfo, subresource: u32) -> Option<(usize, usize)> {
    let mips = info.layout.mips.len().max(1);
    let slices = if info.dimension == victory_formats::texture::Dimension::Tex3d {
        1
    } else {
        info.layout.slices.max(1) as usize
    };
    let sub = subresource as usize;
    (sub < mips * slices).then(|| (sub % mips, sub / mips))
}

/// The host texture's level for title level `mip`: `None` for a top level the host texture was
/// made without (`VICTORY_TEXTURE_DROP_TOP_MIP`). Host level `mip - dropped` has the title level's
/// size. Shortcut: a copy to or from a dropped level is skipped, not scaled; the switch is off by
/// default.
fn host_mip(info: &super::textures::TextureInfo, mip: usize) -> Option<usize> {
    mip.checked_sub(info.dropped_levels())
}

fn mip_dims(info: &super::textures::TextureInfo, mip: usize) -> [usize; 3] {
    [
        (info.width as usize >> mip).max(1),
        (info.height.max(1) as usize >> mip).max(1),
        info.layout.subresource_depth(mip as u32).max(1) as usize,
    ]
}

fn copy_box(
    info: &super::textures::TextureInfo,
    mip: usize,
    source_box: Option<[u32; 6]>,
) -> Option<([usize; 3], [usize; 3])> {
    let dims = mip_dims(info, mip);
    let Some([l, t, f, r, b, back]) = source_box else {
        return Some(([0, 0, 0], dims));
    };
    if r < l || b < t || back < f {
        return None;
    }
    let origin = [l as usize, t as usize, f as usize];
    let size = [(r - l) as usize, (b - t) as usize, (back - f) as usize];
    (0..3).all(|i| origin[i].saturating_add(size[i]) <= dims[i]).then_some((origin, size))
}

fn footprint_span(
    res: This,
    fp: &crate::resource::PlacedFootprint,
    at: [u32; 3],
    size: [u32; 3],
) -> Option<(u64, usize, usize, usize)> {
    // 115 (B4G4R4A4_UNORM) needs CPU nibble rotation; depth formats need plane-separated copies.
    if fp.format == 115 {
        return None;
    }
    let fmt = super::descriptors::format(fp.format)?;
    // A depth and stencil footprint names both planes in one element; the host copies the depth
    // plane alone (as a 4-byte format: `R32_TYPELESS` and the like).
    if fmt == victory_backend::resources::Format::D32FloatS8 {
        return None;
    }
    let (_, block) = crate::resource::element(fp.format)?;
    let (base, buf_size) = crate::resource::buffer_bytes(res)?;
    let (start, end) = fp.span(at, size)?;
    let byte_end = fp.offset.checked_add(end)?;
    if byte_end > buf_size {
        return None;
    }
    let addr = (base as u64).checked_add(fp.offset + start)?;
    let len = usize::try_from(end - start).ok()?;
    let row_pitch = fp.row_pitch as usize;
    let slice_pitch = row_pitch.checked_mul((fp.height as usize).div_ceil(block as usize))?;
    Some((addr, len, row_pitch, slice_pitch))
}

/// Texture copies recorded so far, and since the last present (the `VICTORY_TRACE_TARGETS` frame
/// line).
static COPIES: [std::sync::atomic::AtomicU64; 2] = [const { std::sync::atomic::AtomicU64::new(0) }; 2];

/// A texture copy was recorded: counted, logged once per kind with `VICTORY_TRACE_TARGETS`, and its
/// destination texture (if any) is the GPU's from now on ([`Title::gpu_written`]).
fn copied(t: &mut MetalTitle, what: &str, src: This, dst: Option<This>) {
    COPIES[0].fetch_add(1, Ordering::Relaxed);
    COPIES[1].fetch_add(1, Ordering::Relaxed);
    if let Some(d) = dst {
        t.gpu_written(d as usize);
        trace_bump(d as usize, 0);
    }
    if tracing_targets() {
        static KINDS: Mutex<Vec<(String, Option<victory_backend::resources::Format>, Option<victory_backend::resources::Format>)>> = Mutex::new(Vec::new());
        let kind = (what.to_owned(), t.format_of(src as usize), dst.and_then(|d| t.format_of(d as usize)));
        let mut kinds = KINDS.lock().unwrap_or_else(|e| e.into_inner());
        if !kinds.contains(&kind) {
            tracing::info!(what, src = ?kind.1, dst = ?kind.2, "a texture copy (first of its kind)");
            kinds.push(kind);
        }
    }
}

/// Whether texels of `src` copy into `dst` on the host: one blit (the same format, or plain
/// formats of one size through a view), or through a buffer, formats of one block size where one
/// is block-compressed or depth (plain texels into a GPU encoder's compressed blocks, a depth
/// plane into a 4-byte float), as D3D12 copies them.
fn copy_formats(src: victory_backend::resources::Format, dst: victory_backend::resources::Format) -> bool {
    use victory_backend::resources::Format;
    let bytes = |f: Format| if f == Format::D32FloatS8 { 4 } else { f.block().1 };
    let special = |f: Format| f.block().0 > 1 || f.is_depth();
    victory_metal::gfx::direct_copy(src, dst) || ((special(src) || special(dst)) && bytes(src) == bytes(dst))
}

/// `size` source texels in the destination's texels: a plain texel is a whole compressed block
/// (cut at the destination's edge, from `at`), and a block one texel.
fn dest_size(src: victory_backend::resources::Format, dst: victory_backend::resources::Format, size: [usize; 3], at: [usize; 3], dims: [usize; 3]) -> [usize; 3] {
    let (sw, dw) = (src.block().0, dst.block().0);
    std::array::from_fn(|i| match i {
        2 => size[2],
        _ if dw > sw => (size[i] * dw).min(dims[i].saturating_sub(at[i])),
        _ if sw > dw => size[i].div_ceil(sw),
        _ => size[i],
    })
}

/// Whether a footprint's rows hold the texels of a texture (a depth and stencil texture's: its
/// 4-byte depth plane).
fn rows_match(fp: &crate::resource::PlacedFootprint, info: &super::textures::TextureInfo) -> bool {
    match (super::descriptors::format(fp.format), info.desc()) {
        (Some(f), Some(d)) => copy_formats(f, d.format),
        _ => false,
    }
}

/// `ResolveSubresource`: host textures have one sample, so a resolve copies what both
/// subresources hold (the console clips a larger source to the destination, as the D3D12 host
/// does); the format argument names a view the copy makes when it needs one.
pub(crate) fn resolve(this: This, dst: This, dst_sub: u32, src: This, src_sub: u32) -> bool {
    let (Some(si), Some(di)) = (crate::resource::texture_info(src), crate::resource::texture_info(dst)) else {
        return false;
    };
    let (Some((sm, _)), Some((dm, _))) = (subresource_mip_slice(&si, src_sub), subresource_mip_slice(&di, dst_sub)) else {
        return false;
    };
    let (a, b) = (mip_dims(&si, sm), mip_dims(&di, dm));
    let region = [0, 0, 0, a[0].min(b[0]) as u32, a[1].min(b[1]) as u32, a[2].min(b[2]) as u32];
    copy_texture_region(this, crate::list::CopyLocation::sub(dst, dst_sub), [0; 3], crate::list::CopyLocation::sub(src, src_sub), Some(region))
}

/// `CopyResource` between two title textures. Returns `false` if the host cannot copy them.
pub(crate) fn copy_texture_resource(this: This, dst: This, src: This) -> bool {
    let (Some(src_info), Some(dst_info)) = (
        crate::resource::texture_info(src),
        crate::resource::texture_info(dst),
    ) else {
        return false;
    };
    let mips = src_info.layout.mips.len().max(1).min(dst_info.layout.mips.len().max(1));
    let src_slices = if src_info.dimension == victory_formats::texture::Dimension::Tex3d {
        1
    } else {
        src_info.layout.slices.max(1) as usize
    };
    let dst_slices = if dst_info.dimension == victory_formats::texture::Dimension::Tex3d {
        1
    } else {
        dst_info.layout.slices.max(1) as usize
    };
    let slices = src_slices.min(dst_slices);
    with_list(this, |t, l| {
        let (Some(src_tex), Some(dst_tex)) = (t.texture_of(src as usize), t.texture_of(dst as usize)) else {
            return false;
        };
        let (Some(sf), Some(df)) = (t.format_of(src as usize), t.format_of(dst as usize)) else {
            return false;
        };
        if !copy_formats(sf, df) {
            return false;
        }
        for slice in 0..slices {
            for mip in 0..mips {
                let (Some(src_mip), Some(dst_mip)) = (host_mip(&src_info, mip), host_mip(&dst_info, mip)) else {
                    continue;
                };
                let s = mip_dims(&src_info, mip);
                let d = mip_dims(&dst_info, mip);
                let size = [s[0].min(d[0]), s[1].min(d[1]), s[2].min(d[2])];
                l.list.copy_texture_region(
                    &src_tex,
                    src_mip,
                    slice,
                    [0, 0, 0],
                    size,
                    &dst_tex,
                    dst_mip,
                    slice,
                    [0, 0, 0],
                );
            }
        }
        copied(t, "CopyResource", src, Some(dst));
        if pass_on() {
            pass_note_op(l, "copy-resource", Some(src as usize), Some(dst as usize));
        }
        true
    })
    .unwrap_or(false)
}

/// `CopyTextureRegion` on macOS: texture-to-texture and placed footprint <-> texture.
pub(crate) fn copy_texture_region(
    this: This,
    dst: crate::list::CopyLocation,
    origin: [u32; 3],
    src: crate::list::CopyLocation,
    source_box: Option<[u32; 6]>,
) -> bool {
    const SUBRESOURCE: u32 = 0;
    const FOOTPRINT: u32 = 1;
    let dst_origin = [origin[0] as usize, origin[1] as usize, origin[2] as usize];
    match (dst.kind, src.kind) {
        (SUBRESOURCE, SUBRESOURCE) => {
            let (Some(src_info), Some(dst_info)) = (
                crate::resource::texture_info(src.resource),
                crate::resource::texture_info(dst.resource),
            ) else {
                return false;
            };
            let Some((src_mip, src_slice)) = subresource_mip_slice(&src_info, src.footprint.offset as u32) else {
                return false;
            };
            let Some((dst_mip, dst_slice)) = subresource_mip_slice(&dst_info, dst.footprint.offset as u32) else {
                return false;
            };
            let Some((src_origin, size)) = copy_box(&src_info, src_mip, source_box) else {
                return false;
            };
            let (Some(sf), Some(df)) = (src_info.desc().map(|d| d.format), dst_info.desc().map(|d| d.format)) else {
                return false;
            };
            if !copy_formats(sf, df) {
                return false;
            }
            let dst_dims = mip_dims(&dst_info, dst_mip);
            let to = dest_size(sf, df, size, dst_origin, dst_dims);
            if (0..3).any(|i| dst_origin[i].saturating_add(to[i]) > dst_dims[i]) {
                return false;
            }
            if size.contains(&0) {
                return true;
            }
            let (Some(src_mip), Some(dst_mip)) = (host_mip(&src_info, src_mip), host_mip(&dst_info, dst_mip)) else {
                return true;
            };
            with_list(this, |t, l| {
                let (Some(src_tex), Some(dst_tex)) =
                    (t.texture_of(src.resource as usize), t.texture_of(dst.resource as usize))
                else {
                    return false;
                };
                l.list.copy_texture_region(
                    &src_tex,
                    src_mip,
                    src_slice,
                    src_origin,
                    size,
                    &dst_tex,
                    dst_mip,
                    dst_slice,
                    dst_origin,
                );
                copied(t, "CopyTextureRegion", src.resource, Some(dst.resource));
                if pass_on() {
                    pass_note_op(l, "copy-region", Some(src.resource as usize), Some(dst.resource as usize));
                }
                true
            })
            .unwrap_or(false)
        }
        (SUBRESOURCE, FOOTPRINT) => {
            let Some(dst_info) = crate::resource::texture_info(dst.resource).filter(|i| rows_match(&src.footprint, i)) else {
                return false;
            };
            let Some((dst_mip, dst_slice)) = subresource_mip_slice(&dst_info, dst.footprint.offset as u32) else {
                return false;
            };
            let (src_at, size_u32) = match source_box {
                Some([l, t, f, r, b, back]) if r >= l && b >= t && back >= f => {
                    ([l, t, f], [r - l, b - t, back - f])
                }
                Some(_) => return false,
                None => (
                    [0, 0, 0],
                    [src.footprint.width, src.footprint.height, src.footprint.depth],
                ),
            };
            let size = size_u32.map(|v| v as usize);
            if size.contains(&0) {
                return true;
            }
            let dst_dims = mip_dims(&dst_info, dst_mip);
            if (0..3).any(|i| dst_origin[i].saturating_add(size[i]) > dst_dims[i]) {
                return false;
            }
            let Some(dst_mip) = host_mip(&dst_info, dst_mip) else {
                return true;
            };
            let Some((src_addr, len, row_pitch, slice_pitch)) =
                footprint_span(src.resource, &src.footprint, src_at, size_u32)
            else {
                return false;
            };
            with_list(this, |t, l| {
                let Some(dst_tex) = t.texture_of(dst.resource as usize) else {
                    return false;
                };
                l.list.copy_buffer_to_texture(
                    src_addr,
                    len,
                    row_pitch,
                    slice_pitch,
                    size,
                    &dst_tex,
                    dst_mip,
                    dst_slice,
                    dst_origin,
                );
                copied(t, "CopyTextureRegion", src.resource, Some(dst.resource));
                if pass_on() {
                    pass_note_op(l, "copy-region", Some(src.resource as usize), Some(dst.resource as usize));
                }
                true
            })
            .unwrap_or(false)
        }
        (FOOTPRINT, SUBRESOURCE) => {
            let Some(src_info) = crate::resource::texture_info(src.resource).filter(|i| rows_match(&dst.footprint, i)) else {
                return false;
            };
            let Some((src_mip, src_slice)) = subresource_mip_slice(&src_info, src.footprint.offset as u32) else {
                return false;
            };
            let Some((src_origin, size)) = copy_box(&src_info, src_mip, source_box) else {
                return false;
            };
            if size.contains(&0) {
                return true;
            }
            let Some(src_mip) = host_mip(&src_info, src_mip) else {
                return true;
            };
            let size_u32 = size.map(|v| v as u32);
            let Some((dst_addr, len, row_pitch, slice_pitch)) =
                footprint_span(dst.resource, &dst.footprint, origin, size_u32)
            else {
                return false;
            };
            with_list(this, |t, l| {
                let Some(src_tex) = t.texture_of(src.resource as usize) else {
                    return false;
                };
                l.list.copy_texture_to_buffer(
                    &src_tex,
                    src_mip,
                    src_slice,
                    src_origin,
                    size,
                    dst_addr,
                    len,
                    row_pitch,
                    slice_pitch,
                );
                copied(t, "CopyTextureRegion", src.resource, None);
                if pass_on() {
                    pass_note_op(l, "copy-to-buffer", Some(src.resource as usize), None);
                }
                true
            })
            .unwrap_or(false)
        }
        _ => false,
    }
}

/// A bottom-of-pipe value write of `value`, `bytes` long.
pub(crate) fn write_value(this: This, address: u64, value: u64, bytes: usize) -> bool {
    if env_set!("VICTORY_TRACE_WRITES") {
        let inside = pools().resolve(address, bytes as u64).is_some();
        tracing::warn!("list write {value:#x} ({bytes} bytes) at {address:#x}, in a pool: {inside}, list {this:p}");
    }
    with_list(this, |_, l| l.list.write_value(address, value, bytes)).is_some()
}

/// A wait for a memory value, recorded at the current position of the list.
pub(crate) fn wait_value(this: This, w: Wait) {
    if env_set!("VICTORY_TRACE_WRITES") {
        tracing::warn!("list wait at {:#x} cmp {} ref {:#x}, list {this:p}", w.address, w.comparison, w.reference);
    }
    with_list(this, |_, l| l.wait(w));
}

/// `Reset`: forgets what was recorded (the host objects it named go with it, under the lock).
pub(crate) fn reset_list(this: This) {
    with_list(this, |_, l| l.reset());
}

/// Drops a list object, whose `Rc`s must be dropped under the lock.
pub(crate) fn locked<R>(f: impl FnOnce() -> R) -> R {
    let mut f = Some(f);
    match with_title(|_| (f.take().expect("once"))()) {
        Some(r) => r,
        None => (f.take().expect("not run"))(),
    }
}

/// Counters of the host for tests and the frame line.
pub fn stats() -> Option<super::title::Stats> {
    with_title(|t| t.stats.clone())
}

#[cfg(test)]
mod flow_tests {
    use super::{FlowTexture, flow_dxgi_format, flow_snap_copyable, flow_texture_view, flow_uav_textures, snap_compute_hash, snap_draws, texel_grid, top_values};

    #[test]
    fn texture_views_keep_keys_and_level_zero_extents_including_placed_views() {
        assert_eq!(flow_texture_view("s2.7 tex 0x123 at 0x9000 fmt 10 128x64 mip 3"), Some(("s2.7", FlowTexture {
            key: 0x123, address: 0x9000, format: 10, width: 128, height: 64, depth: 1, dim: victory_backend::resources::TexDim::D2,
        })));
        assert_eq!(flow_texture_view("u0.95 placed tex 0x8000000000000123 at 0xa00000000 fmt 41 7x4 mip 0"), Some(("u0.95", FlowTexture {
            key: 0x8000000000000123, address: 0xa00000000, format: 41, width: 7, height: 4, depth: 1, dim: victory_backend::resources::TexDim::D2,
        })));
        for view in [
            "u0.1 0x123+0x100", "r0 0x123", "k0 [00000000]",
            "s0.0 placed tex at 0x123 fmt 10 2x2 mip 0",
            "s0.0 tex 0xno at 0x123 fmt 10 2x2 mip 0",
            "u0.0 tex 0x1 at 0xno fmt 10 2x2 mip 0",
            "u0.0 tex 0x1 at 0x123 fmt bad 2x2 mip 0",
            "u0.0 tex 0x1 at 0x123 fmt 10 bad mip 0",
            "u0.0 tex 0x1 at 0x123 fmt 10 2x2 mip bad",
            "u.0 tex 0x1 at 0x123 fmt 10 2x2 mip 0",
            "c0.0 tex 0x1 at 0x123 fmt 10 2x2 mip 0",
        ] {
            assert_eq!(flow_texture_view(view), None, "{view}");
        }
    }

    #[test]
    fn snapshot_uavs_exclude_srvs_buffers_and_duplicate_texture_keys() {
        let views: Vec<String> = [
            "s0.0 tex 0x1 at 0x1000 fmt 10 16x16 mip 0",
            "u0.1 0x2000+0x400",
            "u0.2 tex 0x1 at 0x1000 fmt 10 16x16 mip 0",
            "u1.4 tex 0x1 at 0x1000 fmt 10 16x16 mip 1",
            "u1.5 placed tex 0x8000000000000002 at 0x3000 fmt 41 8x4 mip 2",
            "s1.6 placed tex 0x8000000000000003 at 0x4000 fmt 61 7x4 mip 0",
        ].map(str::to_owned).into();
        let picked: Vec<_> = flow_uav_textures(&views).into_iter().map(|(name, t)| (name, t.key, t.width, t.height)).collect();
        assert_eq!(picked, [("u0.2", 1, 16, 16), ("u1.5", 0x8000000000000002, 8, 4)]);
    }

    #[test]
    fn snapshot_texture_limits_skip_depth_blocks_empty_and_overflowing_extents() {
        use victory_backend::resources::Format;
        assert!(flow_snap_copyable(Format::Rgba16Float, [2048, 1024, 2]));
        assert!(flow_snap_copyable(Format::R32Float, [7, 4, 17]));
        for size in [[2048, 1025, 2], [1, 1, 0], [0, 1, 1], [1, 0, 1], [usize::MAX, 2, 1], [2, 1, usize::MAX]] {
            assert!(!flow_snap_copyable(Format::R8Unorm, size));
        }
        for format in [Format::D32Float, Format::D32FloatS8, Format::D16Unorm, Format::Bc1, Format::Bc7] {
            assert!(!flow_snap_copyable(format, [16, 16, 4]));
        }
    }

    #[test]
    fn snapshot_compute_hashes_are_once_per_shader_and_separate_from_draws() {
        let hashes = ["abcd".to_owned(), "5678".to_owned()];
        let mut seen = Vec::new();
        assert_eq!(snap_compute_hash(&["pixel 0000abcd".into()], &hashes, &mut seen), None);
        assert_eq!(snap_compute_hash(&["compute 0000ffff".into()], &hashes, &mut seen), None);
        assert!(seen.is_empty());
        assert_eq!(snap_compute_hash(&["compute 0000abcd".into()], &hashes, &mut seen).as_deref(), Some("0000abcd"));
        assert_eq!(snap_compute_hash(&["compute 0000abcd".into()], &hashes, &mut seen), None);
        assert_eq!(snap_compute_hash(&["compute 00005678".into()], &hashes, &mut seen).as_deref(), Some("00005678"));
        assert_eq!(snap_compute_hash(&["compute 1111abcd".into()], &[], &mut seen), None);
        assert_eq!(seen, ["0000abcd", "00005678"]);
    }

    #[test]
    fn placed_host_formats_keep_their_texel_interpretation() {
        for dxgi in 0..=191 {
            if let Some(format) = super::super::descriptors::format(dxgi) {
                let canonical = flow_dxgi_format(format);
                assert_eq!(super::super::descriptors::format(canonical), Some(format), "DXGI {dxgi}");
                assert!(crate::resource::element(canonical).is_some());
            }
        }
        use victory_backend::resources::Format;
        assert_eq!(flow_dxgi_format(Format::R32Float), 41);
        assert_eq!(flow_dxgi_format(Format::Rgba16Float), 10);
        assert_eq!(flow_dxgi_format(Format::Rg11b10Float), 26);
    }

    /// The snapshots are after the draws the switch names, or after the usual five.
    #[test]
    fn the_snapshot_draws_are_the_ones_named() {
        assert_eq!(snap_draws(None), [1, 30, 60, 90, 120]);
        assert_eq!(snap_draws(Some("")), [1, 30, 60, 90, 120]);
        assert_eq!(snap_draws(Some("2, 38,x,0,44")), [2, 38, 44]);
        assert_eq!(snap_draws(Some(&(1..40).map(|n| n.to_string()).collect::<Vec<_>>().join(","))).len(), 16);
    }

    /// The grid is 8 rows of 14 cells with the percentage of hits in each: a 28 x 16 level whose
    /// upper-left triangle of the first 4 x 4 block is hit shows it in that block's cells only.
    #[test]
    fn the_texel_grid_places_hits_in_their_cells() {
        let (w, h) = (28usize, 16usize);
        let grid = texel_grid(w, h, |i| {
            let (x, y) = (i % w, i / w);
            x + y < 4
        });
        assert_eq!(grid.len(), 8);
        assert!(grid.iter().all(|row| row.len() == 14 * 3));
        // Cells are 2 x 2 texels: (0, 0) all hit, (1, 0) and (0, 1) three of four, (1, 1) none.
        assert_eq!(&grid[0][..9], "100 75  0");
        assert_eq!(&grid[1][..6], " 75  0");
        assert!(grid[2..].iter().all(|row| row.split_whitespace().all(|c| c == "0")));
        // A level smaller than the grid does not divide by zero.
        assert_eq!(texel_grid(1, 1, |_| true).len(), 8);
    }

    #[test]
    fn the_top_values_are_ordered_by_count_then_value() {
        assert_eq!(top_values([6u64, 0, 6, 30, 6, 0, 7].into_iter()), "6x3 0x2 7x1 1ex1");
        assert_eq!(top_values(std::iter::empty()), "");
    }
}
