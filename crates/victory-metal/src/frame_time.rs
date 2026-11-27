//! Optional native-host wall-time accounting. Durations are inclusive, attributed when a
//! scope ends; GPU work is attributed when its completion handler runs. Relaxed swaps give
//! approximate interval boundaries across threads, not an atomic snapshot of the whole host.

use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

static ORIGIN: OnceLock<Instant> = OnceLock::new();

#[inline]
pub fn enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        let on = std::env::var_os("VICTORY_TRACE_FRAME_TIME").is_some_and(|v| v == "1");
        if on {
            let _ = ORIGIN.set(Instant::now());
        }
        on
    })
}

#[derive(Clone, Copy)]
enum Unit {
    Ms,
    Count,
}

macro_rules! metrics {
    ($($metric:ident => ($name:literal, $unit:ident)),+ $(,)?) => {
        #[derive(Clone, Copy)]
        pub enum Metric { $($metric),+ }
        const FIELDS: &[(&str, Unit)] = &[$(($name, Unit::$unit)),+];
    };
}

metrics! {
    LockHeld => ("lock_held", Ms),
    LockWait => ("lock_wait", Ms),
    LockTakes => ("lock_takes", Count),
    LockMaxHold => ("lock_max_hold", Ms),
    HoldSubmit => ("hold_submit", Ms),
    HoldSync => ("hold_sync", Ms),
    HoldTimer => ("hold_timer", Ms),
    HoldList => ("hold_list", Ms),
    HoldOther => ("hold_other", Ms),
    Submits => ("submits", Count),
    OrderWait => ("order_wait", Ms),
    SyncPrepare => ("sync_prepare", Ms),
    Pump => ("pump", Ms),
    Pumps => ("pumps", Count),
    Execute => ("execute", Ms),
    Executes => ("executes", Count),
    Prep => ("prep", Ms),
    Cmds => ("cmds", Count),
    Mirror => ("mirror", Ms),
    Mirrors => ("mirrors", Count),
    MirrorSame => ("mirror_same", Count),
    StampRevalidated => ("stamp_revalidated", Count),
    MirrorBuffer => ("mirror_buffer", Count),
    MirrorTyped => ("mirror_typed", Count),
    MirrorImage => ("mirror_image", Count),
    MirrorSampler => ("mirror_sampler", Count),
    MirrorNull => ("mirror_null", Count),
    ViewMade => ("view_made", Count),
    ViewCached => ("view_cached", Count),
    TexbufMade => ("texbuf_made", Count),
    TexbufCached => ("texbuf_cached", Count),
    EntrySame => ("entry_same", Count),
    SetCommits => ("set_commits", Count),
    SetAdds => ("set_adds", Count),
    SetRemoves => ("set_removes", Count),
    SetCommit => ("set_commit", Ms),
    SafeReads => ("safe_reads", Count),
    SafeRead => ("safe_read", Ms),
    DeviceMirrors => ("device_mirrors", Count),
    DeviceMirror => ("device_mirror", Ms),
    Realize => ("realize", Ms),
    Remirror => ("remirror", Ms),
    Remirrors => ("remirrors", Count),
    RefreshSpans => ("refresh_spans", Ms),
    Twin => ("twin", Ms),
    Backend => ("backend", Ms),
    Cbs => ("cbs", Count),
    PassesCompute => ("passes_compute", Count),
    PassesRender => ("passes_render", Count),
    PassesBlit => ("passes_blit", Count),
    PassesClear => ("passes_clear", Count),
    Dispatches => ("dispatches", Count),
    Draws => ("draws", Count),
    EncCompute => ("enc_compute", Ms),
    EncRender => ("enc_render", Ms),
    EncBlit => ("enc_blit", Ms),
    Commit => ("commit", Ms),
    HeapSets => ("heap_sets", Ms),
    Gpu => ("gpu", Ms),
    Kernel => ("kernel", Ms),
    CbsDone => ("cbs_done", Count),
    CbLatency => ("cb_latency", Ms),
    CbLatencyMax => ("cb_latency_max", Ms),
    Present => ("present", Ms),
    Drawable => ("drawable", Ms),
    FenceWait => ("fence_wait", Ms),
    FenceWaits => ("fence_waits", Count),
    FencePolls => ("fence_polls", Count),
    TimerTicks => ("timer_ticks", Count),
}

const N: usize = FIELDS.len();

struct Counters {
    values: [AtomicU64; N],
    gpu_first: AtomicU64,
    gpu_last: AtomicU64,
}

impl Counters {
    const fn new() -> Self {
        Self {
            values: [const { AtomicU64::new(0) }; N],
            gpu_first: AtomicU64::new(u64::MAX),
            gpu_last: AtomicU64::new(0),
        }
    }

    fn add(&self, metric: Metric, value: u64) {
        self.values[metric as usize].fetch_add(value, Relaxed);
    }

    fn max(&self, metric: Metric, value: u64) {
        self.values[metric as usize].fetch_max(value, Relaxed);
    }

    fn gpu_range(&self, first: u64, last: u64) {
        self.gpu_first.fetch_min(first, Relaxed);
        self.gpu_last.fetch_max(last, Relaxed);
    }

    fn snapshot(&self) -> Snapshot {
        let values = std::array::from_fn(|i| self.values[i].swap(0, Relaxed));
        let first = self.gpu_first.swap(u64::MAX, Relaxed);
        let last = self.gpu_last.swap(0, Relaxed);
        Snapshot {
            values,
            gpu_span: last.saturating_sub(first),
        }
    }
}

static TOTAL: Counters = Counters::new();

struct Snapshot {
    values: [u64; N],
    gpu_span: u64,
}

impl fmt::Display for Snapshot {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for ((name, unit), value) in FIELDS.iter().zip(self.values) {
            match unit {
                Unit::Ms => write!(f, " {name}={:.1}", value as f64 / 1_000_000.0)?,
                Unit::Count => write!(f, " {name}={value}")?,
            }
        }
        write!(f, " gpu_span={:.1}", self.gpu_span as f64 / 1_000_000.0)
    }
}

#[inline]
pub fn count(metric: Metric, n: u64) {
    if enabled() {
        TOTAL.add(metric, n);
    }
}

#[inline]
pub fn start() -> Option<Instant> {
    enabled().then(Instant::now)
}

fn nanos(time: Duration) -> u64 {
    time.as_nanos().min(u128::from(u64::MAX)) as u64
}

/// A disabled timer neither reads the clock nor updates counters. Counted scopes record
/// their time and count together on exit, including early returns and errors.
pub struct Timer {
    started: Option<Instant>,
    metric: Metric,
    count: Option<Metric>,
}

impl Timer {
    #[inline]
    pub fn new(metric: Metric) -> Self {
        Self {
            started: start(),
            metric,
            count: None,
        }
    }

    #[inline]
    pub fn counted(metric: Metric, count: Metric) -> Self {
        Self {
            started: start(),
            metric,
            count: Some(count),
        }
    }

    #[inline]
    pub fn started(&self) -> Option<Instant> {
        self.started
    }
}

impl Drop for Timer {
    #[inline]
    fn drop(&mut self) {
        if let Some(started) = self.started {
            TOTAL.add(self.metric, nanos(started.elapsed()));
            if let Some(count) = self.count {
                TOTAL.add(count, 1);
            }
        }
    }
}

/// Called just after TITLE.lock(), only when `start()` returned a timestamp.
pub fn lock_acquired(waiting: Instant) -> Instant {
    let held = Instant::now();
    TOTAL.add(Metric::LockWait, nanos(held.duration_since(waiting)));
    TOTAL.add(Metric::LockTakes, 1);
    held
}

/// Called immediately before the TITLE guard unlocks.
pub fn lock_released(held: Instant, kind: Metric) {
    let ns = nanos(held.elapsed());
    TOTAL.add(Metric::LockHeld, ns);
    TOTAL.add(kind, ns);
    TOTAL.max(Metric::LockMaxHold, ns);
}

/// Only called by an enabled completion handler. Metal timestamps are seconds in one
/// monotonic clock domain; failed buffers may report zero or unavailable intervals.
pub fn completed(
    latency: Duration,
    gpu_start: f64,
    gpu_end: f64,
    kernel_start: f64,
    kernel_end: f64,
) {
    TOTAL.add(Metric::CbsDone, 1);
    let latency = nanos(latency);
    TOTAL.add(Metric::CbLatency, latency);
    TOTAL.max(Metric::CbLatencyMax, latency);
    let valid = |first: f64, last: f64| {
        first.is_finite() && last.is_finite() && first > 0.0 && last >= first
    };
    if valid(gpu_start, gpu_end) {
        TOTAL.add(Metric::Gpu, ((gpu_end - gpu_start) * 1e9) as u64);
        TOTAL.gpu_range((gpu_start * 1e9) as u64, (gpu_end * 1e9) as u64);
    }
    if valid(kernel_start, kernel_end) {
        TOTAL.add(Metric::Kernel, ((kernel_end - kernel_start) * 1e9) as u64);
    }
}

/// One line at the end of every glue::present, after its TITLE guard and present timer drop.
pub fn report() {
    if !enabled() {
        return;
    }
    // Serialize reports only; writers keep using relaxed atomics without taking this mutex.
    static LAST: Mutex<Option<Instant>> = Mutex::new(None);
    let mut last = LAST.lock().unwrap_or_else(|e| e.into_inner());
    let now = Instant::now();
    let wall = now.duration_since(last.replace(now).unwrap_or(*ORIGIN.get().expect("enabled")));
    let snapshot = TOTAL.snapshot();
    tracing::warn!("frame time: wall={:.1}{snapshot}", wall.as_secs_f64() * 1e3);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn add_and_snapshot_reset_totals_maxima_and_gpu_bounds() {
        // Local counters, independent of both the process environment and concurrent users.
        let counters = Counters::new();
        counters.add(Metric::Submits, 1);
        counters.add(Metric::Submits, 2);
        counters.add(Metric::LockHeld, 1_500_000);
        counters.add(Metric::LockHeld, 2_000_000);
        counters.max(Metric::LockMaxHold, 2_000_000);
        counters.max(Metric::LockMaxHold, 1_500_000);
        counters.gpu_range(10_000_000, 12_000_000);
        counters.gpu_range(9_000_000, 11_000_000);
        let snapshot = counters.snapshot();
        assert_eq!(snapshot.values[Metric::Submits as usize], 3);
        assert_eq!(snapshot.values[Metric::LockHeld as usize], 3_500_000);
        assert_eq!(snapshot.values[Metric::LockMaxHold as usize], 2_000_000);
        assert_eq!(snapshot.gpu_span, 3_000_000);
        let line = snapshot.to_string();
        assert!(line.contains(" lock_held=3.5"));
        assert!(line.contains(" submits=3 "));
        assert!(line.ends_with(" gpu_span=3.0"));
        let empty = counters.snapshot();
        assert_eq!(empty.values, [0; N]);
        assert_eq!(empty.gpu_span, 0);
        counters.gpu_range(20_000_000, 21_000_000);
        assert_eq!(counters.snapshot().gpu_span, 1_000_000);
    }
}
