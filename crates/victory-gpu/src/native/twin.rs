//! Depth textures the title also writes through unordered access, on the native host: the
//! design of `crate::twin` (the D3D12 host's; decision 0007, "One host texture per texture
//! memory"). The console allows both uses of one texture (the title's material depth, which a
//! compute shader writes and the tiled lighting draws then test EQUAL against; its shadow-map
//! copies; its upscaled depth). Metal does not: a depth texture takes no shader writes. Such a
//! texture gets a writable twin, a colour texture per plane (depth, then stencil), which its
//! unordered access views show instead (`NativeHost::twin_plane`), and the newer contents move
//! between the two where the use changes (`NativeHost::execute`): blits through a host buffer,
//! in a pass of their own before the command that needs the other side.
//!
//! Where this differs from the D3D12 host's twin:
//! - The twin's depth plane has the host depth plane's own texel (32-bit floats for the 32-bit
//!   formats and for the console's 16-bit depth with stencil, which the host keeps as 32-bit
//!   float depth; 16-bit normalised values for 16-bit depth without stencil), so the copies are
//!   plain blits. The D3D12 host keeps the console's 16 bits in the twin and converts in a
//!   compute pass of its own (`depth = float(u16) * 0x37800080`). shortcut: a value a shader
//!   stores through a 16-bit view of a float plane is not rounded to 16 bits. The title's
//!   material depth stores `float(id) * 0x37800080` itself, which is what the conversion gives
//!   and what its draws rasterise, so EQUAL holds; upgrade (a rounding pass) if a title stores
//!   other values there and compares them.
//! - Uses are found per command at submission, from the entries of the tables it binds that
//!   show the texture or its twin (the shader's declared extent of each table, which may be
//!   more than it reads) and from its depth target. As on the D3D12 host, shaders that index
//!   the heap themselves move nothing.
//!
//! shortcut: level 0 of slice 0 only (what the title's uses name).

use std::cell::Cell;
use std::rc::Rc;
use victory_backend::resources::*;

/// Which side holds the newer contents.
pub(super) const BOTH: u8 = 0;
pub(super) const DEPTH: u8 = 1;
pub(super) const TWIN: u8 = 2;

pub(super) struct Plane<B: GraphicsBackend> {
    /// The colour texture unordered access views of this plane show.
    pub texture: Rc<B::Texture>,
    /// Bytes a texel, and the plane's offset in the scratch buffer.
    bytes: usize,
    at: usize,
}

pub(super) struct Twin<B: GraphicsBackend> {
    pub depth: Rc<B::Texture>,
    /// The depth plane, then the stencil plane if the format has one.
    pub planes: Vec<Plane<B>>,
    scratch: B::Buffer,
    base: *mut u8,
    size: [usize; 2],
    /// Which side is newer as of the commands planned so far.
    pub newer: Cell<u8>,
}

/// The twin's plane formats for a host depth format, with their texel sizes.
pub(super) fn plane_formats(depth: Format) -> Option<&'static [(Format, usize)]> {
    match depth {
        Format::D32FloatS8 => Some(&[(Format::R32Float, 4), (Format::R8Uint, 1)]),
        Format::D32Float => Some(&[(Format::R32Float, 4)]),
        Format::D16Unorm => Some(&[(Format::R16Unorm, 2)]),
        _ => None,
    }
}

impl<B: GraphicsBackend> Twin<B> {
    /// The twin of depth texture `depth`, which is as `desc` says; an error if the host makes
    /// none.
    pub(super) fn new(backend: &B, depth: &Rc<B::Texture>, desc: &TextureDesc) -> Result<Twin<B>, String> {
        let formats = plane_formats(desc.format).ok_or("not a depth format with a writable twin")?;
        if desc.dim == TexDim::D3 {
            return Err("a 3D depth texture".into());
        }
        let (w, h) = (desc.width, desc.height);
        let mut planes = Vec::new();
        let mut at = 0;
        for &(format, bytes) in formats {
            let texture = backend.create_texture(&TextureDesc {
                dim: TexDim::D2,
                format,
                width: w,
                height: h,
                depth_or_layers: 1,
                mips: 1,
                // Attachable: an unordered access clear is a render pass's load action.
                render_target: true,
                unordered_access: true,
            })?;
            planes.push(Plane { texture: Rc::new(texture), bytes, at });
            at = (at + w * h * bytes).next_multiple_of(256);
        }
        let (scratch, base) = backend.host_buffer(at.max(256))?;
        Ok(Twin { depth: depth.clone(), planes, scratch, base, size: [w, h], newer: Cell::new(DEPTH) })
    }

    /// Width and height.
    pub(super) fn size(&self) -> [usize; 2] {
        self.size
    }

    /// The blits that bring one side up to date from the other: each plane through the scratch
    /// buffer.
    pub(super) fn copy(&self, to_depth: bool) -> Vec<BlitOp<'_, B>> {
        let [w, h] = self.size;
        let mut ops = Vec::new();
        for (i, p) in self.planes.iter().enumerate() {
            // SAFETY: the plane's rows lie in the scratch buffer (`new` sized it for them).
            let buffer = (&self.scratch, unsafe { self.base.add(p.at) }.cast_const());
            let (row_pitch, len) = (w * p.bytes, w * h * p.bytes);
            let depth = BlitOp::CopyDepthPlane { texture: &*self.depth, stencil: i == 1, to_texture: to_depth, buffer, size: [w, h], row_pitch };
            if to_depth {
                ops.push(BlitOp::CopyTextureToBuffer {
                    src: &*p.texture,
                    src_mip: 0,
                    src_slice: 0,
                    src_origin: [0; 3],
                    size: [w, h, 1],
                    dst: buffer,
                    len,
                    row_pitch,
                    slice_pitch: len,
                });
                ops.push(depth);
            } else {
                ops.push(depth);
                ops.push(BlitOp::CopyBufferToTexture {
                    src: buffer,
                    len,
                    row_pitch,
                    slice_pitch: len,
                    size: [w, h, 1],
                    dst: &*p.texture,
                    dst_mip: 0,
                    dst_slice: 0,
                    dst_origin: [0; 3],
                });
            }
        }
        ops
    }

    /// Before a use that needs `side` current: whether a copy must come first (true: into the
    /// depth texture). Then both sides are current.
    pub(super) fn need(&self, side: u8) -> Option<bool> {
        let newer = self.newer.get();
        (newer != BOTH && newer != side).then(|| {
            self.newer.set(BOTH);
            side == DEPTH
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_host_depth_format_has_twin_planes_of_its_own_texels() {
        assert_eq!(plane_formats(Format::D32FloatS8).unwrap(), &[(Format::R32Float, 4), (Format::R8Uint, 1)]);
        assert_eq!(plane_formats(Format::D32Float).unwrap(), &[(Format::R32Float, 4)]);
        assert_eq!(plane_formats(Format::D16Unorm).unwrap(), &[(Format::R16Unorm, 2)]);
        assert!(plane_formats(Format::R32Float).is_none());
    }
}
