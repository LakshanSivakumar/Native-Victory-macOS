//! `VICTORY_PASS_TIMING=1`: GPU time per pass (#460), off by default.
//!
//! The host otherwise times whole command buffers (`GPUEndTime - GPUStartTime`), so a slow pass in
//! a world frame cannot be told from the others. With the switch, each compute, render and blit pass
//! of a submission gets two timestamp samples, at the start and the end of its encoder, through its
//! pass descriptor (`sampleBufferAttachments`; Apple GPUs sample at stage boundaries only, not
//! between dispatches or draws). One `MTLCounterSampleBuffer` per command buffer holds them; when
//! the command buffer completes, they are resolved and each pass's time is added to its key: the
//! names of its pipelines (or their addresses) and its dispatch or draw count. About every 10
//! seconds one `pass timing` line lists the passes that took the most GPU time since the last line,
//! with the total and the share of the command buffers' own GPU time that the passes cover.
//!
//! A device without timestamp counters at stage boundaries logs that once and times nothing. A
//! submission split into pieces (`VICTORY_PASSES_PER_BUFFER`) is not timed.
//!
//! GPU timestamps are converted to nanoseconds by the device's own correlation of its GPU and CPU
//! clocks (`sampleTimestamps:gpuTimestamp:`), taken when timing starts and again at each line.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSRange;
use objc2_metal::*;

/// How often a `pass timing` line is logged.
const EVERY: Duration = Duration::from_secs(10);
/// How many passes a line names.
const TOP: usize = 12;

/// Whether the switch is on.
pub(crate) fn asked() -> bool {
    std::env::var("VICTORY_PASS_TIMING").is_ok_and(|v| v.trim() == "1")
}

/// What the timed passes added up to since the last line.
#[derive(Default)]
pub(crate) struct Totals {
    /// Per key: passes seen, GPU ticks.
    passes: HashMap<String, (u64, u64)>,
    /// The command buffers' own GPU time (seconds) and how many there were.
    buffers: (f64, u64),
    since: Option<Instant>,
}

/// Timing for one device.
pub(crate) struct PassTiming {
    set: Retained<ProtocolObject<dyn MTLCounterSet>>,
    totals: Arc<Mutex<Totals>>,
    /// The device's CPU (ns) and GPU clocks when timing started.
    start: (u64, u64),
}

/// Two clocks read together: CPU in nanoseconds, GPU in its ticks.
fn clocks(dev: &ProtocolObject<dyn MTLDevice>) -> (u64, u64) {
    let (mut cpu, mut gpu) = (0u64, 0u64);
    // SAFETY: both pointers are valid for one write each.
    unsafe {
        dev.sampleTimestamps_gpuTimestamp(std::ptr::NonNull::from(&mut cpu), std::ptr::NonNull::from(&mut gpu))
    };
    (cpu, gpu)
}

impl PassTiming {
    /// Timing on `dev` if the switch is on and the device can sample timestamps at stage
    /// boundaries; otherwise `None` (logged once when the switch is on).
    pub(crate) fn new(dev: &ProtocolObject<dyn MTLDevice>) -> Option<PassTiming> {
        if !asked() {
            return None;
        }
        Self::on(dev)
    }

    /// [`Self::new`] whatever the switch says.
    pub(crate) fn on(dev: &ProtocolObject<dyn MTLDevice>) -> Option<PassTiming> {
        if !dev.supportsCounterSampling(MTLCounterSamplingPoint::AtStageBoundary) {
            tracing::warn!("pass timing: this GPU samples no counters at stage boundaries; nothing is timed");
            return None;
        }
        // SAFETY: a static string of the framework.
        let wanted = unsafe { MTLCommonCounterSetTimestamp };
        let sets = dev.counterSets()?;
        let set = (0..sets.count()).map(|i| sets.objectAtIndex(i)).find(|s| s.name().to_string() == wanted.to_string());
        let Some(set) = set else {
            tracing::warn!("pass timing: this GPU has no timestamp counter set; nothing is timed");
            return None;
        };
        tracing::info!("pass timing: on (VICTORY_PASS_TIMING); a line about every {}s", EVERY.as_secs());
        Some(PassTiming { set, totals: Arc::default(), start: clocks(dev) })
    }

    /// A sample buffer for a command buffer of `passes` passes.
    pub(crate) fn samples(&self, dev: &ProtocolObject<dyn MTLDevice>, passes: usize) -> Option<Samples> {
        if passes == 0 {
            return None;
        }
        let d = MTLCounterSampleBufferDescriptor::new();
        d.setCounterSet(Some(&self.set));
        d.setStorageMode(MTLStorageMode::Shared);
        // SAFETY: two samples per pass; Metal checks the count against its own limit.
        unsafe { d.setSampleCount(2 * passes) };
        let buf = match dev.newCounterSampleBufferWithDescriptor_error(&d) {
            Ok(b) => b,
            Err(e) => {
                static ONCE: std::sync::Once = std::sync::Once::new();
                ONCE.call_once(|| tracing::warn!("pass timing: no sample buffer: {}", e.localizedDescription()));
                return None;
            }
        };
        Some(Samples { buf, keys: Vec::new(), cap: passes })
    }

    /// Each key's passes and GPU ticks since the last line, and the command buffers counted.
    #[cfg(test)]
    pub(crate) fn snapshot(&self) -> (Vec<(String, u64, u64)>, u64) {
        let t = self.totals.lock().unwrap();
        (t.passes.iter().map(|(k, &(n, ticks))| (k.clone(), n, ticks)).collect(), t.buffers.1)
    }

    /// What to move into a command buffer's completed handler.
    pub(crate) fn sink(&self, dev: &Retained<ProtocolObject<dyn MTLDevice>>) -> Sink {
        Sink { totals: self.totals.clone(), dev: dev.clone(), start: self.start }
    }
}

/// The samples of one command buffer's passes, in order.
pub(crate) struct Samples {
    buf: Retained<ProtocolObject<dyn MTLCounterSampleBuffer>>,
    keys: Vec<String>,
    cap: usize,
}

impl Samples {
    /// The start and end sample indices of the next pass, named `key`; `None` when full.
    fn next(&mut self, key: String) -> Option<(usize, usize)> {
        if self.keys.len() >= self.cap {
            return None;
        }
        let n = self.keys.len();
        self.keys.push(key);
        Some((2 * n, 2 * n + 1))
    }

    /// A compute pass descriptor that samples the next pass.
    pub(crate) fn compute(&mut self, key: String) -> Option<Retained<MTLComputePassDescriptor>> {
        let (a, b) = self.next(key)?;
        let d = MTLComputePassDescriptor::computePassDescriptor();
        // SAFETY: index 0 of the attachment array always exists; the indices are inside the buffer.
        unsafe {
            let s = d.sampleBufferAttachments().objectAtIndexedSubscript(0);
            s.setSampleBuffer(Some(&self.buf));
            s.setStartOfEncoderSampleIndex(a);
            s.setEndOfEncoderSampleIndex(b);
        }
        Some(d)
    }

    /// A blit pass descriptor that samples the next pass.
    pub(crate) fn blit(&mut self, key: String) -> Option<Retained<MTLBlitPassDescriptor>> {
        let (a, b) = self.next(key)?;
        let d = MTLBlitPassDescriptor::blitPassDescriptor();
        // SAFETY: as for `compute`.
        unsafe {
            let s = d.sampleBufferAttachments().objectAtIndexedSubscript(0);
            s.setSampleBuffer(Some(&self.buf));
            s.setStartOfEncoderSampleIndex(a);
            s.setEndOfEncoderSampleIndex(b);
        }
        Some(d)
    }

    /// Samples the next pass, a render pass described by `d`: from the start of its vertex stage
    /// to the end of its fragment stage.
    pub(crate) fn render(&mut self, d: &MTLRenderPassDescriptor, key: String) {
        let Some((a, b)) = self.next(key) else { return };
        // SAFETY: as for `compute`; the stages not named take no sample.
        unsafe {
            let s = d.sampleBufferAttachments().objectAtIndexedSubscript(0);
            s.setSampleBuffer(Some(&self.buf));
            s.setStartOfVertexSampleIndex(a);
            s.setEndOfVertexSampleIndex(MTLCounterDontSample);
            s.setStartOfFragmentSampleIndex(MTLCounterDontSample);
            s.setEndOfFragmentSampleIndex(b);
        }
    }

    /// Each sampled pass's key and GPU ticks (a pass whose samples did not resolve is left out).
    fn resolve(&self) -> Vec<(String, u64)> {
        let n = self.keys.len();
        if n == 0 {
            return Vec::new();
        }
        // SAFETY: the range is inside the buffer, which the completed command buffer wrote.
        let Some(data) = (unsafe { self.buf.resolveCounterRange(NSRange::new(0, 2 * n)) }) else {
            return Vec::new();
        };
        let bytes = data.to_vec();
        let at = |i: usize| bytes.get(8 * i..8 * i + 8).map(|b| u64::from_le_bytes(b.try_into().unwrap()));
        self.keys
            .iter()
            .enumerate()
            .filter_map(|(i, k)| {
                let (a, b) = (at(2 * i)?, at(2 * i + 1)?);
                // An omitted or failed sample reads 0 or `MTLCounterErrorValue` (all ones).
                (a != 0 && b != u64::MAX && b >= a).then(|| (k.clone(), b - a))
            })
            .collect()
    }
}

/// Adds a completed command buffer's passes to the totals, and logs a line when one is due.
pub(crate) struct Sink {
    totals: Arc<Mutex<Totals>>,
    dev: Retained<ProtocolObject<dyn MTLDevice>>,
    start: (u64, u64),
}

impl Sink {
    pub(crate) fn add(&self, samples: &Samples, buffer_gpu_seconds: f64) {
        let times = samples.resolve();
        let mut t = self.totals.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        for (k, ticks) in times {
            let e = t.passes.entry(k).or_default();
            e.0 += 1;
            e.1 += ticks;
        }
        t.buffers.0 += buffer_gpu_seconds;
        t.buffers.1 += 1;
        let since = *t.since.get_or_insert_with(Instant::now);
        if since.elapsed() < EVERY {
            return;
        }
        let ns_per_tick = ns_per_tick(self.start, clocks(&self.dev));
        let line = report(&t, ns_per_tick);
        *t = Totals { since: Some(Instant::now()), ..Totals::default() };
        drop(t);
        tracing::info!("pass timing: {line}");
    }
}

/// Nanoseconds per GPU tick from two readings of both clocks (1 when the GPU clock did not move).
fn ns_per_tick(a: (u64, u64), b: (u64, u64)) -> f64 {
    let (cpu, gpu) = (b.0.saturating_sub(a.0), b.1.saturating_sub(a.1));
    if gpu == 0 { 1.0 } else { cpu as f64 / gpu as f64 }
}

/// The line for `t`: the total, the share of command-buffer time, then the top passes.
fn report(t: &Totals, ns_per_tick: f64) -> String {
    let mut passes: Vec<(&String, u64, f64)> =
        t.passes.iter().map(|(k, &(n, ticks))| (k, n, ticks as f64 * ns_per_tick / 1e6)).collect();
    passes.sort_by(|a, b| b.2.total_cmp(&a.2));
    let total: f64 = passes.iter().map(|p| p.2).sum();
    let buffers_ms = t.buffers.0 * 1e3;
    let share = if buffers_ms > 0.0 { 100.0 * total / buffers_ms } else { 0.0 };
    let mut line = format!(
        "{:.1} ms in {} passes over {} command buffers ({:.1} ms, {share:.0}% covered); top:",
        total,
        passes.iter().map(|p| p.1).sum::<u64>(),
        t.buffers.1,
        buffers_ms
    );
    for (k, n, ms) in passes.iter().take(TOP) {
        line.push_str(&format!(" [{ms:.2} ms x{n}: {k}]"));
    }
    line
}

/// The key of a pass: its pipelines' names (or addresses), each once with its count, and what it
/// holds in all (`what`: "dispatches", "draws", "ops").
pub(crate) fn key(kind: &str, what: &str, count: usize, pipes: &[(String, usize)]) -> String {
    let mut k = format!("{kind} {count} {what}");
    for (name, n) in pipes.iter().take(4) {
        k.push_str(&format!(" {name}x{n}"));
    }
    if pipes.len() > 4 {
        k.push_str(&format!(" +{}", pipes.len() - 4));
    }
    k
}

/// Counts the distinct values of `ids`, in first-seen order.
pub(crate) fn tally(ids: impl Iterator<Item = usize>) -> Vec<(usize, usize)> {
    let mut seen: Vec<(usize, usize)> = Vec::new();
    for k in ids {
        match seen.iter_mut().find(|(p, _)| *p == k) {
            Some((_, n)) => *n += 1,
            None => seen.push((k, 1)),
        }
    }
    seen
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_report_ranks_passes_by_time_and_shares_the_buffers_time() {
        let mut t = Totals::default();
        t.passes.insert("compute 1 dispatches slowx1".into(), (2, 3_000_000));
        t.passes.insert("blit 1 ops".into(), (1, 1_000_000));
        t.buffers = (0.008, 2);
        let line = report(&t, 1.0);
        assert!(line.starts_with("4.0 ms in 3 passes over 2 command buffers (8.0 ms, 50% covered)"), "{line}");
        let slow = line.find("slowx1").unwrap();
        let blit = line.find("blit 1 ops").unwrap();
        assert!(slow < blit, "the longest first: {line}");
    }

    #[test]
    fn keys_name_each_pipeline_once_with_its_count() {
        let t = tally([5, 7, 5, 5].into_iter());
        assert_eq!(t, vec![(5, 3), (7, 1)]);
        let pipes: Vec<(String, usize)> = t.iter().map(|&(k, n)| (format!("p{k}"), n)).collect();
        assert_eq!(key("render", "draws", 4, &pipes), "render 4 draws p5x3 p7x1");
    }

    #[test]
    fn ticks_convert_by_the_two_clocks() {
        assert_eq!(ns_per_tick((100, 10), (300, 110)), 2.0);
        assert_eq!(ns_per_tick((100, 10), (300, 10)), 1.0);
    }
}
