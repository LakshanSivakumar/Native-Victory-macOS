//! The MetalFX upscale, end to end, without a title: synthetic gradients land in
//! host textures through buffer copies, one `metalfx_upscale` command goes through a
//! [`NativeList`] and the [`Host`], and the output is read back through another buffer
//! copy and checked against the input. Unlike the games this needs no shader
//! translation, no title thread and no window: the whole dispatch → record → pass →
//! scaler chain runs in the test process.

use super::Host;
use super::PoolObserver;
use super::NativeList;
use victory_backend::resources::{Format, TextureDesc};
use victory_metal::metal::Metal;

/// A f16 bit pattern that grows with `v` over 0..1 (f16's bit order is its value
/// order for positives, which is all the gradient checks need).
fn f16_bits(v: u16) -> u16 {
    v
}

struct Pool {
    mem: victory_metal::mem::Alias,
}

impl Pool {
    fn new(h: &Host, pages: usize) -> Pool {
        let mem =
            victory_metal::mem::Alias::new(pages * victory_metal::mem::page_size().max(16384))
                .unwrap();
        // The copies name the pool's alias address as title memory: register it the
        // way the title's own pools register, or the host drops the work unmapped.
        let info = crate::native::memory::PoolInfo {
            id: 1,
            base: mem.primary as usize,
            len: mem.len,
            memory_entry: None,
        };
        h.memory().pool_created(&info);
        h.memory().mapped(mem.alias as usize, mem.len, &info, 0);
        Pool { mem }
    }
    fn va(&self, at: usize) -> u64 {
        self.mem.alias as u64 + at as u64
    }
    fn write<T: Copy>(&self, at: usize, v: &[T]) {
        // SAFETY: inside the mapping; the GPU is idle while the test writes.
        unsafe {
            std::ptr::copy_nonoverlapping(v.as_ptr(), self.mem.alias.add(at).cast(), v.len())
        };
    }
    fn read<T: Copy>(&self, at: usize, n: usize) -> Vec<T> {
        // SAFETY: inside the mapping; read after the fence wait.
        unsafe { std::slice::from_raw_parts(self.mem.alias.add(at).cast::<T>(), n).to_vec() }
    }
}

/// The one 2× upscale: a horizontal grey gradient in, a taller gradient out. Asserts
/// the output is not flat and spans the input's range — the checks the synthetic
/// suite in `victory-metal` makes against the scaler alone, now through the host's
/// recording and submission path.
#[test]
fn metalfx_upscale_runs_a_synthetic_frame_through_the_host() {
    // Skip without a Metal device (CI without a GPU).
    let Ok(m) = Metal::new() else {
        eprintln!("skipped: no Metal device");
        return;
    };
    let h = Host::new(m);
    let _ = tracing_subscriber::fmt().with_max_level(tracing::Level::DEBUG).try_init();
    let pool = Pool::new(&h, 32);
    let (w_in, h_in, w_out, h_out) = (128usize, 64, 256, 128);

    // MetalFX wants read-write on everything it touches (the copies and its own
    // encoder both write).
    let rw = |f: Format, w, hh| TextureDesc {
        render_target: true,
        unordered_access: true,
        ..TextureDesc::d2(f, w, hh)
    };
    let colour = h.texture(&rw(Format::Rgba16Float, w_in, h_in)).unwrap();
    let motion = h.texture(&rw(Format::Rg16Float, w_in, h_in)).unwrap();
    let output = h.texture(&rw(Format::Rgba16Float, w_out, h_out)).unwrap();

    // Input pixels: a horizontal gradient with alpha 1, zero motion, into the pool.
    let in_rgba: Vec<u16> = (0..w_in * h_in)
        .flat_map(|i| {
            let x = f16_bits((((i % w_in) as u32 * 0x3bff as u32 / (w_in as u32 - 1)) as u16));
            [x, x, x, 0x3c00]
        })
        .collect();
    let in_motion: Vec<u16> = vec![0; w_in * h_in * 2];
    let (in_at, motion_at) = (0, in_rgba.len() * 2);
    let out_at = motion_at + in_motion.len() * 2;
    let out_fill = vec![0xEEEEu16; w_out * h_out * 4];
    pool.write(in_at, &in_rgba);
    pool.write(motion_at, &in_motion);
    pool.write(out_at, &out_fill);

    let mut l = NativeList::default();
    l.copy_buffer_to_texture(
        pool.va(in_at),
        in_rgba.len() * 2,
        w_in * 8,
        w_in * h_in * 8,
        [w_in, h_in, 1],
        &colour,
        0,
        0,
        [0, 0, 0],
    );
    l.copy_buffer_to_texture(
        pool.va(motion_at),
        in_motion.len() * 2,
        w_in * 4,
        w_in * h_in * 4,
        [w_in, h_in, 1],
        &motion,
        0,
        0,
        [0, 0, 0],
    );
    // One upscale through the host's own recording path (the context handle is
    // ours here; the hle layer's ABI tests cover the title-facing side).
    l.metalfx_upscale(
        1,
        Some(colour.clone()),
        None,
        Some(motion.clone()),
        None,
        output.clone(),
        true,
        [0.0, 0.0],
    );
    l.copy_texture_to_buffer(
        &output,
        0,
        0,
        [0, 0, 0],
        [w_out, h_out, 1],
        pool.va(out_at),
        w_out * h_out * 8,
        w_out * 8,
        w_out * h_out * 8,
    );
    super::run(&h, &[&l]);

    let out: Vec<u16> = pool.read(out_at, w_out * h_out * 4);
    assert!(
        out.iter().any(|v| *v != 0xEEEE),
        "the output copy never landed"
    );
    let distinct: std::collections::HashSet<u16> = out.iter().step_by(16).copied().collect();
    assert!(
        distinct.len() >= 4,
        "the upscaled output looks flat: {distinct:?}; first {:?}",
        &out[..8]
    );
}
