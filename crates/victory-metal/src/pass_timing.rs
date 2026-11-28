//! `VICTORY_PASS_TIMING`: GPU time per pass, sampled at each encoder's boundary (#460, step 1).
//! The whole command buffer's GPU time (`GPUEndTime - GPUStartTime`) cannot tell a slow pass
//! from the others; an `MTLCounterSampleBuffer` of the timestamp counter set, sampled at each
//! pass encoder's start and end, can. Off unless the variable is set: `=1` logs a frame's 10
//! most expensive passes, `=<n>` logs `n`. At each present (a frame's boundary) the passes
//! timed since the last one are logged, most expensive first, by pipeline and work.
//!
//! Only encoder kinds the device reports as sampling at their own boundary are sampled: Metal
//! asserts (and aborts) on `sampleCountersInBuffer` for a kind it does not support, which is
//! not the same on every GPU family (a G15X samples dispatch and draw boundaries, not blit).

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::{NSRange, NSString};
use objc2_metal::*;
use std::collections::HashMap;
use std::ptr::NonNull;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};

/// `MTLCounterErrorValue` (the C macro `~0ULL`, not exported to Rust): written into a sample
/// that errored while resolving.
const ERROR_VALUE: u64 = u64::MAX;

/// Which encoder kinds the device samples at their own boundary.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct Points {
    pub compute: bool,
    pub render: bool,
    pub blit: bool,
}

impl Points {
    fn any(&self) -> bool {
        self.compute || self.render || self.blit
    }
}

/// What one timed pass was: its kind, the pipeline it ran (by pointer; `pipe_names` in gfx.rs
/// names it), and how much work it held (draws, thread groups, blit ops).
#[derive(Clone, Debug)]
pub(crate) struct PassLabel {
    pub kind: &'static str,
    pub pipe: usize,
    pub work: u64,
}

/// A timed pass whose samples resolved: the label, and the GPU nanoseconds between its samples.
#[derive(Clone, Debug)]
pub(crate) struct PassSample {
    pub label: PassLabel,
    pub nanos: u64,
}

/// The device's timestamp counter set and the encoder kinds it samples: makes one sample buffer
/// per command buffer.
pub(crate) struct PassTimer {
    dev: Retained<ProtocolObject<dyn MTLDevice>>,
    set: Retained<ProtocolObject<dyn MTLCounterSet>>,
    /// How many of a frame's most expensive passes to log.
    pub top: usize,
    points: Points,
}

impl PassTimer {
    /// `VICTORY_PASS_TIMING` asks for this: `=1` logs a frame's 10 most expensive passes, `=<n>`
    /// logs `n`. `None` when unset, or when the device samples no encoder kind.
    pub(crate) fn asked(dev: &Retained<ProtocolObject<dyn MTLDevice>>) -> Option<PassTimer> {
        let v = std::env::var("VICTORY_PASS_TIMING").ok()?;
        let top = match v.trim() {
            "1" => 10,
            s => s.parse::<usize>().ok().filter(|&n| n > 0)?,
        };
        let timer = Self::supported(dev, top);
        match &timer {
            Some(t) => {
                let p = t.points();
                tracing::info!(
                    compute = p.compute,
                    render = p.render,
                    blit = p.blit,
                    top = t.top,
                    "metal: VICTORY_PASS_TIMING on: GPU time per pass, sampled at each encoder's boundary"
                );
            }
            None => eprintln!("metal: VICTORY_PASS_TIMING={v} but this device samples no counter boundary"),
        }
        timer
    }

    /// The device's timestamp counter set, if it has one, and the encoder kinds it samples.
    pub(crate) fn supported(dev: &Retained<ProtocolObject<dyn MTLDevice>>, top: usize) -> Option<PassTimer> {
        let points = Points {
            compute: dev.supportsCounterSampling(MTLCounterSamplingPoint::AtDispatchBoundary),
            render: dev.supportsCounterSampling(MTLCounterSamplingPoint::AtDrawBoundary),
            blit: dev.supportsCounterSampling(MTLCounterSamplingPoint::AtBlitBoundary),
        };
        if !points.any() {
            return None;
        }
        let set = dev
            .counterSets()?
            .iter()
            .find(|s| {
                // SAFETY: reading Metal's constant name strings.
                (unsafe { &*s.name() == MTLCommonCounterSetTimestamp })
                    && s.counters().iter().any(|c| {
                        // SAFETY: as above.
                        unsafe { &*c.name() == MTLCommonCounterTimestamp }
                    })
            })
            ?;
        Some(PassTimer { dev: dev.clone(), set, top, points })
    }

    pub(crate) fn points(&self) -> Points {
        self.points
    }

    /// A sample buffer for one command buffer of `passes` passes: two samples a pass (its
    /// encoder's start and end), plus two for the blit pass a refused pass's fills are
    /// re-encoded into.
    pub(crate) fn begin(&self, passes: usize) -> Option<PassTiming> {
        let d = MTLCounterSampleBufferDescriptor::new();
        d.setCounterSet(Some(&*self.set));
        d.setLabel(&NSString::from_str("victory pass timing"));
        d.setStorageMode(MTLStorageMode::Shared);
        // SAFETY: the count is a plain number; the samples stay below it (see `begin_pass`).
        unsafe { d.setSampleCount(2 * passes + 2) };
        let buf = self.dev.newCounterSampleBufferWithDescriptor_error(&d).ok()?;
        Some(PassTiming { buf, next: 0, passes: Vec::with_capacity(passes), points: self.points })
    }
}

/// The timing of one command buffer in flight: its sample buffer, the next sample index (two a
/// pass), and one label per timed pass, in encode order.
pub(crate) struct PassTiming {
    buf: Retained<ProtocolObject<dyn MTLCounterSampleBuffer>>,
    next: usize,
    passes: Vec<PassLabel>,
    points: Points,
}

// SAFETY: Metal's objects may be used from any thread (as `Compiler` in metal.rs); the buffer is
// only sampled from the encoders of the one command buffer it was made for, and resolved from
// that buffer's completion handler.
unsafe impl Send for PassTiming {}

impl PassTiming {
    /// Whether the device samples `kind`'s encoders at their boundary.
    pub(crate) fn kind_ok(&self, kind: &str) -> bool {
        match kind {
            "compute" => self.points.compute,
            "render" => self.points.render,
            _ => self.points.blit,
        }
    }

    /// Reserves the sample pair for the next pass and returns its start index, or `None` when
    /// the device does not sample this kind (no label, no samples: the pairs stay aligned).
    pub(crate) fn begin_pass(&mut self, kind: &'static str, pipe: usize, work: u64) -> Option<usize> {
        if !self.kind_ok(kind) {
            return None;
        }
        let at = self.next;
        self.next += 2;
        self.passes.push(PassLabel { kind, pipe, work });
        Some(at)
    }

    pub(crate) fn sample_count(&self) -> usize {
        self.next
    }

    pub(crate) fn labels(&self) -> &[PassLabel] {
        &self.passes
    }

    pub(crate) fn buffer(&self) -> &ProtocolObject<dyn MTLCounterSampleBuffer> {
        &self.buf
    }
}

/// Samples the encoder at `idx` of `t`'s buffer, behind a barrier so the timestamp follows the
/// work encoded before it. Works on any encoder protocol: each declares the method itself.
macro_rules! sample {
    ($e:expr, $t:expr, $idx:expr) => {
        // SAFETY: `$idx` is below the sample buffer's count, reserved by `begin_pass`.
        unsafe { $e.sampleCountersInBuffer_atSampleIndex_withBarrier($t.buffer(), $idx, true) }
    };
}
pub(crate) use sample;

/// Timed command buffers by their submit serial, until they complete.
static PENDING: Mutex<Option<HashMap<u64, PassTiming>>> = Mutex::new(None);
/// Passes timed since the last frame boundary.
static FRAME: Mutex<Vec<PassSample>> = Mutex::new(Vec::new());
static FRAME_NO: AtomicU64 = AtomicU64::new(0);

pub(crate) fn register(serial: u64, timing: PassTiming) {
    PENDING
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .get_or_insert_with(HashMap::new)
        .insert(serial, timing);
}

/// The command buffer `serial` completed: resolve its samples into the frame's list. A sample
/// pair that errored (a pass refused after its start sample, say) is left out.
pub(crate) fn completed(serial: u64) {
    let Some(timing) = PENDING
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .get_or_insert_with(HashMap::new)
        .remove(&serial)
    else {
        return;
    };
    let n = timing.sample_count();
    if n == 0 {
        return;
    }
    // SAFETY: the range is the buffer's own count.
    let Some(data) = (unsafe { timing.buffer().resolveCounterRange(NSRange { location: 0, length: n }) }) else {
        return;
    };
    let len = data.length();
    if len < n * 8 {
        return;
    }
    // Each sample is one `MTLCounterResultTimestamp` (u64).
    let mut raw = vec![0u8; len];
    // SAFETY: `raw` holds `len` bytes, as asked.
    unsafe { data.getBytes_length(NonNull::new(raw.as_mut_ptr().cast()).unwrap(), len) };
    let mut frame = FRAME.lock().unwrap_or_else(|e| e.into_inner());
    for (i, label) in timing.labels().iter().enumerate() {
        // Pass `i`'s samples are at `2*i` and `2*i+1`.
        let (Some(a), Some(b)) = (raw.get(i * 16..i * 16 + 8), raw.get(i * 16 + 8..i * 16 + 16)) else {
            continue;
        };
        let (a, b) = (
            u64::from_le_bytes(a.try_into().unwrap()),
            u64::from_le_bytes(b.try_into().unwrap()),
        );
        if a == ERROR_VALUE || b == ERROR_VALUE {
            continue;
        }
        frame.push(PassSample { label: label.clone(), nanos: b.saturating_sub(a) });
    }
}

/// At a frame boundary (a present): log the frame's `top` most expensive passes, by pipeline
/// (named through `name`) and work, most expensive first, and the frame's total. Returns what
/// was logged (the tests read it).
pub(crate) fn frame_boundary(top: usize, name: impl Fn(usize) -> Option<String>) -> Vec<PassSample> {
    let mut frame = FRAME.lock().unwrap_or_else(|e| e.into_inner());
    if frame.is_empty() {
        return Vec::new();
    }
    frame.sort_by_key(|s| std::cmp::Reverse(s.nanos));
    let no = FRAME_NO.fetch_add(1, Ordering::Relaxed) + 1;
    let total: u64 = frame.iter().map(|s| s.nanos).sum();
    let logged: Vec<PassSample> = frame.iter().take(top).cloned().collect();
    for (rank, s) in logged.iter().enumerate() {
        let pipe = name(s.label.pipe).unwrap_or_else(|| format!("{:#x}", s.label.pipe));
        tracing::info!(
            frame = no,
            rank = rank + 1,
            kind = s.label.kind,
            pipe,
            work = s.label.work,
            ms = s.nanos as f64 / 1e6,
            "pass time"
        );
    }
    tracing::info!(frame = no, passes = frame.len(), total_ms = total as f64 / 1e6, "pass time: frame total");
    std::mem::take(&mut *frame)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::ComputeBackend;
    use crate::mem::{Region, page_size};
    use crate::metal::Metal;
    use crate::resources::{BlitOp, ColorTarget, Format, GraphicsBackend, Pass, PassSubmission, RenderPass, ResourceBackend, TextureDesc};

    /// Two passes of a kind this Mac samples are each sampled at their encoder's start and end,
    /// and a frame boundary reports both, while the work itself still lands.
    #[test]
    fn each_pass_of_a_command_buffer_is_sampled_and_a_frame_reports_them() {
        let m = Metal::new().unwrap().with_pass_timing(true);
        let Some(timer) = &m.pass_timing else {
            eprintln!("skipped: this device samples no counter boundary");
            return;
        };
        let points = timer.points();
        let q = m.create_queue().unwrap();
        let fence = m.create_fence(0).unwrap();
        let region = Region::new(page_size(), None).unwrap();
        let buf = m.wrap_title_memory(region.ptr, region.len).unwrap();
        let mut td = TextureDesc::d2(Format::Rgba8Unorm, 64, 64);
        td.render_target = true;
        let tex = m.create_texture(&td).unwrap();
        // Blit fills need no shader; a render pass of zero draws over a colour target needs no
        // pipeline either. Take whichever kind this GPU samples.
        // SAFETY: inside the region; the GPU is idle.
        let at = |o: usize| unsafe { region.ptr.add(o) as *const u8 };
        let (passes, kind): (Vec<Pass<'_, Metal>>, &'static str) = if points.blit {
            (
                vec![
                    Pass::Blit(vec![BlitOp::Fill { dst: (&buf, at(0)), len: 8, value: 0x11 }]),
                    Pass::Blit(vec![BlitOp::Fill { dst: (&buf, at(8)), len: 8, value: 0x22 }]),
                ],
                "blit",
            )
        } else if points.render {
            let rp = |clear: [f32; 4]| {
                Pass::Render(RenderPass {
                    color: vec![ColorTarget { texture: &tex, clear: Some(clear) }],
                    depth: None,
                    draws: vec![],
                })
            };
            (vec![rp([0.0, 0.0, 0.0, 1.0]), rp([1.0, 0.0, 0.0, 1.0])], "render")
        } else {
            eprintln!("skipped: this device samples neither blit nor render boundaries");
            return;
        };
        m.submit_passes(&q, &PassSubmission { waits: &[], passes: &passes, signals: &[(&fence, 1)] }).unwrap();
        assert!(m.wait(&fence, 1, 5000));
        // The completed handler runs when the buffer completes, which may be just after the
        // signal: wait for the two passes' samples to arrive.
        let mut samples = Vec::new();
        for _ in 0..100 {
            samples = frame_boundary(10, |_| None);
            if samples.len() >= 2 {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(20));
        }
        assert_eq!(samples.len(), 2, "one timed sample per pass");
        assert!(samples.iter().all(|s| s.label.kind == kind));
        assert!(
            samples.iter().all(|s| s.nanos < 60 * 1_000_000_000),
            "a tiny pass does not take a minute: {:?}",
            samples.iter().map(|s| s.nanos).collect::<Vec<_>>()
        );
    }
}
