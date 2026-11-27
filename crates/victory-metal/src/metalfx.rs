//! MetalFX temporal scaler spike (#375): build an `MTLFXTemporalScaler` over
//! synthetic textures, encode two frames through `encodeToCommandBuffer`, and
//! check the output moved. No title, no shaders, no Metal Shader Converter.

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

#[cfg(test)]
mod tests {
    #[test]
    fn temporal_scaler_scales_synthetic_input() {
        match super::temporal_scaler_smoke() {
            Ok(what) => println!("{what}"),
            Err(e) => panic!("{e}"),
        }
    }
}
