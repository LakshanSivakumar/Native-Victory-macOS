//! MetalFX temporal scaler spike (#375): build an `MTLFXTemporalScaler` over
//! synthetic textures, encode two frames through `encodeToCommandBuffer`, and
//! check the output moved. No title, no shaders, no Metal Shader Converter.
//!
//! Also the synthetic calibration the plan's step D calls for: a square
//! pans across known ground-truth motion and the tests hold the scaler to
//! it (position tracking, drift, ghosting) — the motion-convention record
//! the M2 dispatch needs before it feeds real title vectors in.

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLCommandBuffer, MTLCommandQueue as _, MTLCreateSystemDefaultDevice, MTLDevice, MTLPixelFormat, MTLRegion, MTLStorageMode,
    MTLTexture, MTLTextureDescriptor, MTLTextureUsage, MTLOrigin, MTLSize,
};
use objc2_metal_fx::{
    MTLFXFrameInterpolatableScaler as _, MTLFXTemporalScaler, MTLFXTemporalScalerBase as _,
    MTLFXTemporalScalerDescriptor,
};

const IN_W: usize = 640;
const IN_H: usize = 360;
const OUT_W: usize = 1280;
const OUT_H: usize = 720;
const SQUARE_SIZE: usize = 64; // input px; 128 output px at this suite's 2x scale

fn region(w: usize, h: usize) -> MTLRegion {
    MTLRegion {
        origin: MTLOrigin { x: 0, y: 0, z: 0 },
        size: MTLSize {
            width: w,
            height: h,
            depth: 1,
        },
    }
}

fn texture(
    device: &ProtocolObject<dyn MTLDevice>,
    w: usize,
    h: usize,
    format: MTLPixelFormat,
    render_target: bool,
) -> Result<Retained<ProtocolObject<dyn MTLTexture>>, String> {
    let desc = unsafe { MTLTextureDescriptor::new() };
    unsafe {
        desc.setPixelFormat(format);
        desc.setWidth(w);
        desc.setHeight(h);
        desc.setStorageMode(MTLStorageMode::Shared);
        let usage = if render_target {
            MTLTextureUsage::ShaderRead | MTLTextureUsage::ShaderWrite
        } else {
            MTLTextureUsage::ShaderRead
        };
        desc.setUsage(usage);
    }
    unsafe { device.newTextureWithDescriptor(&desc) }.ok_or_else(|| "texture refused".into())
}

/// Builds a temporal scaler, feeds it one synthetic frame, and reports what
/// the output looked like. The string is for the test's assert message.
pub fn temporal_scaler_smoke() -> Result<String, String> {
    let Some(device) = (unsafe { MTLCreateSystemDefaultDevice() }) else {
        return Err("no default Metal device".into());
    };

    let color = texture(&device, IN_W, IN_H, MTLPixelFormat::RGBA16Float, false)?;
    let depth = texture(&device, IN_W, IN_H, MTLPixelFormat::Depth32Float, false)?;
    let motion = texture(&device, IN_W, IN_H, MTLPixelFormat::RG16Float, false)?;
    let output = texture(&device, OUT_W, OUT_H, MTLPixelFormat::RGBA16Float, true)?;

    // Paint a horizontal gradient into the colour target; half-float pixels.
    let mut pixels = vec![0u16; IN_W * IN_H * 4];
    for y in 0..IN_H {
        for x in 0..IN_W {
            let i = (y * IN_W + x) * 4;
            pixels[i] = ((x * 0x7fff) / IN_W) as u16;
            pixels[i + 1] = ((y * 0x7fff) / IN_H) as u16;
            pixels[i + 3] = 0x3c00; // alpha 1.0
        }
    }
    unsafe {
        color.replaceRegion_mipmapLevel_withBytes_bytesPerRow(
            region(IN_W, IN_H),
            0,
            std::ptr::NonNull::new(pixels.as_mut_ptr().cast()).unwrap(),
            IN_W * 8,
        );
    }
    // Depth 1.0 everywhere, zero motion: a still camera, so the upscaler only
    // has the gradient itself to move.
    let mut depth_px = vec![0x3f80u16; IN_W * IN_H * 2];
    unsafe {
        depth.replaceRegion_mipmapLevel_withBytes_bytesPerRow(
            region(IN_W, IN_H),
            0,
            std::ptr::NonNull::new(depth_px.as_mut_ptr().cast()).unwrap(),
            IN_W * 4,
        );
    }

    let desc = unsafe {
        let d = MTLFXTemporalScalerDescriptor::new();
        d.setInputWidth(IN_W);
        d.setInputHeight(IN_H);
        d.setOutputWidth(OUT_W);
        d.setOutputHeight(OUT_H);
        d.setColorTextureFormat(MTLPixelFormat::RGBA16Float);
        d.setDepthTextureFormat(MTLPixelFormat::Depth32Float);
        d.setMotionTextureFormat(MTLPixelFormat::RG16Float);
        d.setOutputTextureFormat(MTLPixelFormat::RGBA16Float);
        d
    };
    let Some(scaler) = (unsafe { desc.newTemporalScalerWithDevice(&device) }) else {
        return Err("the device refused a temporal scaler".into());
    };

    unsafe {
        scaler.setColorTexture(Some(&color));
        scaler.setDepthTexture(Some(&depth));
        scaler.setMotionTexture(Some(&motion));
        scaler.setOutputTexture(Some(&output));
        scaler.setInputContentWidth(IN_W);
        scaler.setInputContentHeight(IN_H);
        scaler.setJitterOffsetX(0.0);
        scaler.setJitterOffsetY(0.0);
        scaler.setMotionVectorScaleX(1.0);
        scaler.setMotionVectorScaleY(1.0);
        scaler.setReset(true);
    }

    let Some(queue) = (unsafe { device.newCommandQueue() }) else {
        return Err("no command queue".into());
    };
    let Some(cb) = (unsafe { queue.commandBuffer() }) else {
        return Err("no command buffer".into());
    };
    unsafe { scaler.encodeToCommandBuffer(&cb) };
    unsafe { cb.commit() };
    unsafe { cb.waitUntilCompleted() };

    let mut out = vec![0u16; OUT_W * OUT_H * 4];
    unsafe {
        output.getBytes_bytesPerRow_fromRegion_mipmapLevel(
            std::ptr::NonNull::new(out.as_mut_ptr().cast()).unwrap(),
            OUT_W * 8,
            region(OUT_W, OUT_H),
            0,
        );
    }
    let distinct: std::collections::HashSet<u16> =
        out.iter().step_by(64).copied().collect();
    if distinct.len() < 4 {
        return Err(format!(
            "output looks flat: {} distinct sampled values",
            distinct.len()
        ));
    }
    Ok(format!(
        "frame encoded; {} distinct output values sampled",
        distinct.len()
    ))
}

// ===== Synthetic calibration (step D of the #375 plan) =====
//
// Ground truth the tests hold MetalFX to: a bright square on black whose
// CONTENT steps by (+3, +2) input pixels per frame — (+6, +4) output pixels
// at this suite's 2x scale — while the motion texture is fed independently,
// so the convention can be checked against the same content. Apple's
// documented units (output-space pixels, previous frame to current) are the
// pass condition; a wrong sign or unit shows up as ghosting at the square's
// trailing edge, which the tests measure directly. If a test fails on a new
// host, its printout is the calibration record.

const SQUARE_STEP_X: f64 = 3.0; // input px per frame
const SQUARE_STEP_Y: f64 = 2.0;

/// f32 -> f16 bits. Truncating; exact for the small integers these tests
/// use (0, 1, 2, 4, 6), which is all they need.
#[cfg(test)]
fn f16_of(v: f32) -> u16 {
    let s = if v.is_sign_negative() { 0x8000u16 } else { 0 };
    let a = v.abs();
    if a == 0.0 || a < 2f32.powi(-14) {
        return s;
    }
    if a >= 65504.0 {
        return s | 0x7c00;
    }
    let e = a.log2().floor().max(-14.0) as i32;
    let mant = ((a / 2f32.powi(e) - 1.0) * 1024.0) as u16;
    s | (((e + 15) as u16) << 10) | mant
}

/// f16 bits -> f32.
#[cfg(test)]
fn f32_of(h: u16) -> f32 {
    let sign = if h & 0x8000 != 0 { -1.0f32 } else { 1.0 };
    let exp = ((h >> 10) & 0x1f) as i32;
    let mant = (h & 0x03ff) as i32;
    let v = if exp == 0 {
        (mant as f32) * 2f32.powi(-24)
    } else if exp == 31 {
        f32::INFINITY
    } else {
        (1.0 + mant as f32 / 1024.0) * 2f32.powi(exp - 15)
    };
    sign * v
}

/// Frame `k`'s colour: the square at `origin + k * (step_x, step_y)`.
#[cfg(test)]
fn square_color_bits(k: usize, step_x: f64, step_y: f64) -> Vec<u16> {
    let mut px = vec![0u16; IN_W * IN_H * 4];
    let ox = (32.0 + step_x * k as f64).round() as usize;
    let oy = (32.0 + step_y * k as f64).round() as usize;
    for y in oy..oy + SQUARE_SIZE {
        for x in ox..ox + SQUARE_SIZE {
            let i = (y * IN_W + x) * 4;
            px[i] = 0x3c00;
            px[i + 1] = 0x3c00;
            px[i + 2] = 0x3c00;
            px[i + 3] = 0x3c00;
        }
    }
    px
}

/// A uniform motion field: every pixel moved by `(dx, dy)` output pixels.
#[cfg(test)]
fn motion_bits(dx: f32, dy: f32) -> Vec<u16> {
    let (mx, my) = (f16_of(dx), f16_of(dy));
    let mut px = Vec::with_capacity(IN_W * IN_H * 2);
    for _ in 0..IN_W * IN_H {
        px.push(mx);
        px.push(my);
    }
    px
}

#[cfg(test)]
fn depth_bits() -> Vec<u16> {
    vec![0x3f80; IN_W * IN_H * 2] // 1.0
}

/// Uploads `bits` into `tex`.
fn upload(tex: &ProtocolObject<dyn MTLTexture>, bits: &mut [u16], bytes_per_row: usize, w: usize, h: usize) {
    unsafe {
        tex.replaceRegion_mipmapLevel_withBytes_bytesPerRow(
            region(w, h),
            0,
            std::ptr::NonNull::new(bits.as_mut_ptr().cast()).unwrap(),
            bytes_per_row,
        );
    }
}

/// Runs `frames` frames of the scene through one scaler and returns each
/// frame's output bits. The content steps by `(step_x, step_y)` input pixels
/// per frame while the motion texture claims `(mv_dx, mv_dy)` output pixels,
/// so the two can disagree on purpose. `reset` on the first frame only; the
/// scaler keeps its history across the rest, as the title would.
#[cfg(test)]
#[allow(clippy::too_many_arguments)]
fn run_scene(
    device: &ProtocolObject<dyn MTLDevice>,
    step_x: f64,
    step_y: f64,
    mv_dx: f32,
    mv_dy: f32,
    frames: usize,
) -> Result<Vec<Vec<u16>>, String> {
    let color = texture(device, IN_W, IN_H, MTLPixelFormat::RGBA16Float, false)?;
    let depth = texture(device, IN_W, IN_H, MTLPixelFormat::Depth32Float, false)?;
    let motion = texture(device, IN_W, IN_H, MTLPixelFormat::RG16Float, false)?;
    let output = texture(device, OUT_W, OUT_H, MTLPixelFormat::RGBA16Float, true)?;

    let desc = unsafe {
        let d = MTLFXTemporalScalerDescriptor::new();
        d.setInputWidth(IN_W);
        d.setInputHeight(IN_H);
        d.setOutputWidth(OUT_W);
        d.setOutputHeight(OUT_H);
        d.setColorTextureFormat(MTLPixelFormat::RGBA16Float);
        d.setDepthTextureFormat(MTLPixelFormat::Depth32Float);
        d.setMotionTextureFormat(MTLPixelFormat::RG16Float);
        d.setOutputTextureFormat(MTLPixelFormat::RGBA16Float);
        d
    };
    let Some(scaler) = (unsafe { desc.newTemporalScalerWithDevice(device) }) else {
        return Err("the device refused a temporal scaler".into());
    };
    unsafe {
        scaler.setColorTexture(Some(&color));
        scaler.setDepthTexture(Some(&depth));
        scaler.setMotionTexture(Some(&motion));
        scaler.setOutputTexture(Some(&output));
        scaler.setInputContentWidth(IN_W);
        scaler.setInputContentHeight(IN_H);
        scaler.setMotionVectorScaleX(1.0);
        scaler.setMotionVectorScaleY(1.0);
    }

    let Some(queue) = (unsafe { device.newCommandQueue() }) else {
        return Err("no command queue".into());
    };
    let mut depth_px = depth_bits();
    upload(&depth, &mut depth_px, IN_W * 4, IN_W, IN_H);

    let mut outs = Vec::with_capacity(frames);
    for k in 0..frames {
        let mut color_px = square_color_bits(k, step_x, step_y);
        upload(&color, &mut color_px, IN_W * 8, IN_W, IN_H);
        let mut motion_px = motion_bits(mv_dx, mv_dy);
        upload(&motion, &mut motion_px, IN_W * 4, IN_W, IN_H);

        unsafe {
            scaler.setJitterOffsetX(0.0);
            scaler.setJitterOffsetY(0.0);
            scaler.setReset(k == 0);
        }
        let Some(cb) = (unsafe { queue.commandBuffer() }) else {
            return Err("no command buffer".into());
        };
        unsafe { scaler.encodeToCommandBuffer(&cb) };
        unsafe { cb.commit() };
        unsafe { cb.waitUntilCompleted() };

        let mut out = vec![0u16; OUT_W * OUT_H * 4];
        unsafe {
            output.getBytes_bytesPerRow_fromRegion_mipmapLevel(
                std::ptr::NonNull::new(out.as_mut_ptr().cast()).unwrap(),
                OUT_W * 8,
                region(OUT_W, OUT_H),
                0,
            );
        }
        outs.push(out);
    }
    Ok(outs)
}

/// Per-pixel luminance (R+G+B) of an RGBA16Float output frame.
#[cfg(test)]
fn luma(bits: &[u16]) -> Vec<f32> {
    bits.chunks_exact(4)
        .map(|c| f32_of(c[0]) + f32_of(c[1]) + f32_of(c[2]))
        .collect()
}

/// Centre of mass of pixels brighter than `thresh`; `None` when nothing is.
#[cfg(test)]
fn centroid(l: &[f32], w: usize, h: usize, thresh: f32) -> Option<(f64, f64)> {
    let (mut sx, mut sy, mut s) = (0.0f64, 0.0f64, 0.0f64);
    for y in 0..h {
        for x in 0..w {
            if l[y * w + x] > thresh {
                sx += x as f64;
                sy += y as f64;
                s += 1.0;
            }
        }
    }
    (s > 0.0).then(|| (sx / s, sy / s))
}

/// Mean luminance inside a rectangle (clamped to the frame).
#[cfg(test)]
fn rect_mean(l: &[f32], w: usize, h: usize, x0: usize, y0: usize, rw: usize, rh: usize) -> f64 {
    let (mut sum, mut n) = (0.0f64, 0.0f64);
    for y in y0.min(h)..(y0 + rh).min(h) {
        for x in x0.min(w)..(x0 + rw).min(w) {
            sum += l[y * w + x] as f64;
            n += 1.0;
        }
    }
    if n > 0.0 { sum / n } else { 0.0 }
}

/// The square's output-space origin at frame `k`.
#[cfg(test)]
fn square_out_origin(k: usize, step_x: f64, step_y: f64) -> (usize, usize) {
    (
        ((32.0 + step_x * k as f64).round() as usize) * 2,
        ((32.0 + step_y * k as f64).round() as usize) * 2,
    )
}

/// Mean luminance in the background strip just behind the square's trailing
/// edge (the side the square vacated) at frame `k` — where ghosting from a
/// wrong motion convention shows up first.
#[cfg(test)]
fn trailing_bleed(l: &[f32], k: usize, step_x: f64, step_y: f64) -> f64 {
    let (ox, oy) = square_out_origin(k, step_x, step_y);
    // Motion is +x,+y, so the vacated side is the square's left/top.
    // A 24-px-wide strip 4 px behind the square's edge, spanning the
    // square's inner height, keeps clear of the square's own soft edge.
    let strip_x0 = ox.saturating_sub(28);
    let strip_w = 20;
    rect_mean(l, OUT_W, OUT_H, strip_x0, oy + 16, strip_w, 96)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn temporal_scaler_scales_synthetic_input() {
        match super::temporal_scaler_smoke() {
            Ok(what) => println!("{what}"),
            Err(e) => panic!("{e}"),
        }
    }

    /// Static content, zero motion, zero jitter: the output must hold the
    /// square where it was, frame after frame. Catches a scaler that
    /// hallucinates motion or drifts its history.
    #[test]
    fn still_scene_holds_position() {
        let device = unsafe { MTLCreateSystemDefaultDevice() }.expect("a Metal device");
        let outs = run_scene(&device, 0.0, 0.0, 0.0, 0.0, 3).expect("scene ran");
        let l0 = luma(&outs[0]);
        let l2 = luma(&outs[2]);
        let c0 = centroid(&l0, OUT_W, OUT_H, 0.3).expect("a bright square in frame 0");
        let c2 = centroid(&l2, OUT_W, OUT_H, 0.3).expect("a bright square in frame 2");
        let (dx, dy) = (c2.0 - c0.0, c2.1 - c0.1);
        println!("still scene: frame-0 centroid {c0:?}, frame-2 centroid {c2:?}, drift ({dx:.2}, {dy:.2})");
        assert!(
            dx.abs() < 1.5 && dy.abs() < 1.5,
            "a still scene drifted by ({dx:.2}, {dy:.2}) output pixels"
        );
        let (x0, y0) = square_out_origin(0, 0.0, 0.0);
        let interior = rect_mean(&l2, OUT_W, OUT_H, x0 + 16, y0 + 16, 96, 96);
        assert!(
            interior > 2.5,
            "the square's interior faded to {interior:.3} of 3.0"
        );
    }

    /// Content steps (+6, +4) output pixels per frame and the motion texture
    /// says the same, in Apple's documented units. The output must track the
    /// ground truth, keep the square's interior bright, and leave the
    /// vacated background dark.
    #[test]
    fn panning_scene_tracks_documented_motion() {
        let device = unsafe { MTLCreateSystemDefaultDevice() }.expect("a Metal device");
        let frames = run_scene(&device, SQUARE_STEP_X, SQUARE_STEP_Y, 6.0, 4.0, 4).expect("scene ran");
        let luma_per_frame: Vec<Vec<f32>> = frames.iter().map(|o| luma(o)).collect();
        let centroids: Vec<(f64, f64)> = luma_per_frame
            .iter()
            .map(|l| centroid(l, OUT_W, OUT_H, 0.3).expect("a bright square"))
            .collect();
        println!("panning centroids: {centroids:?}");
        for k in 1..frames.len() {
            let (dx, dy) = (
                centroids[k].0 - centroids[k - 1].0,
                centroids[k].1 - centroids[k - 1].1,
            );
            assert!(
                (dx - 6.0).abs() < 1.5 && (dy - 4.0).abs() < 1.5,
                "frame {k} moved ({dx:.2}, {dy:.2}); ground truth (+6, +4)"
            );
        }
        let last = luma_per_frame.last().unwrap();
        let cur = {
            let (x0, y0) = square_out_origin(3, SQUARE_STEP_X, SQUARE_STEP_Y);
            rect_mean(last, OUT_W, OUT_H, x0 + 16, y0 + 16, 96, 96)
        };
        let bleed = trailing_bleed(last, 3, SQUARE_STEP_X, SQUARE_STEP_Y);
        println!("panning: square interior {cur:.3} (of 3.0), trailing bleed {bleed:.3}");
        assert!(
            cur > 2.5,
            "the square's interior washed out to {cur:.3} of 3.0"
        );
        assert!(
            bleed < 0.3,
            "the vacated background holds {bleed:.3} of 3.0 — ghosting,              which reads as a motion-convention mismatch (sign or units)"
        );
    }

    /// Calibration differential (informational): the same panning scene with
    /// the motion sign flipped. Apple's documented convention should bleed
    /// less than the flipped one; if it does not, the convention this suite
    /// encodes is wrong and M2's dispatch must translate the title's vectors
    /// the other way. Prints both; asserts nothing.
    #[test]
    fn panning_convention_differential_report() {
        let device = unsafe { MTLCreateSystemDefaultDevice() }.expect("a Metal device");
        let documented = run_scene(&device, SQUARE_STEP_X, SQUARE_STEP_Y, 6.0, 4.0, 4).expect("scene ran");
        let flipped = run_scene(&device, SQUARE_STEP_X, SQUARE_STEP_Y, -6.0, -4.0, 4).expect("scene ran");
        let bleed_of = |outs: &Vec<Vec<u16>>| {
            let l = luma(outs.last().unwrap());
            let (x0, y0) = square_out_origin(3, SQUARE_STEP_X, SQUARE_STEP_Y);
            let interior = rect_mean(&l, OUT_W, OUT_H, x0 + 16, y0 + 16, 96, 96);
            (interior, trailing_bleed(&l, 3, SQUARE_STEP_X, SQUARE_STEP_Y))
        };
        let (d_int, d_bleed) = bleed_of(&documented);
        let (f_int, f_bleed) = bleed_of(&flipped);
        println!(
            "convention differential (interior of 3.0, trailing bleed): \
             documented (+6, +4): ({d_int:.3}, {d_bleed:.3}); \
             flipped (-6, -4): ({f_int:.3}, {f_bleed:.3})"
        );
    }
}
