//! Pins for the platform-neutral value types of [`crate::resources`]:
//! [`Format`]'s block/integer/depth/row rules, [`TextureDesc::d2`]'s
//! defaults, [`VertexFormat::byte_size`], [`Blend::second_source`] and the
//! two `Default` impls.

use crate::resources::{
    Blend, BlendFactor, BlendOp, ColorState, Cull, Format, TexDim, TextureDesc, VertexFormat,
};

#[test]
fn format_block_is_one_texel_or_a_4x4_block() {
    assert_eq!(Format::Rgba8Unorm.block(), (1, 4));
    assert_eq!(Format::D32FloatS8.block(), (1, 8));
    assert_eq!(Format::Bc1.block(), (4, 8));
    assert_eq!(Format::Bc4.block(), (4, 8));
    assert_eq!(Format::Bc2.block(), (4, 16));
    assert_eq!(Format::Bc7.block(), (4, 16));
    assert_eq!(Format::Bc7Srgb.block(), (4, 16));
    assert_eq!(Format::Bc6hSigned.block(), (4, 16));
    assert_eq!(Format::R8Unorm.block(), (1, 1));
    assert_eq!(Format::A8Unorm.block(), (1, 1));
    assert_eq!(Format::R16Unorm.block(), (1, 2));
    assert_eq!(Format::Bgr5a1Unorm.block(), (1, 2));
    assert_eq!(Format::Rgba16Float.block(), (1, 8));
    assert_eq!(Format::Rgba32Float.block(), (1, 16));
    assert_eq!(Format::Rgba32Uint.block(), (1, 16));
    assert_eq!(Format::Rgb9e5Float.block(), (1, 4));
}

#[test]
fn format_integer_and_depth_flags() {
    for f in [
        Format::Rgba8Uint,
        Format::R32Uint,
        Format::R8Uint,
        Format::Rgba32Uint,
        Format::Rg32Uint,
        Format::Rg16Uint,
        Format::Rg8Uint,
        Format::R16Uint,
        Format::Rgba16Uint,
        Format::Rgb10a2Uint,
    ] {
        assert!(f.is_integer(), "{f:?} is integer");
    }
    for f in [
        Format::Rgba8Unorm,
        Format::D32Float,
        Format::Bc7,
        Format::R16Unorm,
    ] {
        assert!(!f.is_integer(), "{f:?} is not integer");
    }
    for f in [Format::D32Float, Format::D32FloatS8, Format::D16Unorm] {
        assert!(f.is_depth(), "{f:?} is depth");
    }
    for f in [Format::Rgba8Unorm, Format::R32Float, Format::Bc1] {
        assert!(!f.is_depth(), "{f:?} is not depth");
    }
}

#[test]
fn format_row_bytes_counts_whole_blocks() {
    assert_eq!(Format::Rgba8Unorm.row_bytes(3), 12);
    assert_eq!(Format::Rgba8Unorm.row_bytes(0), 0);
    assert_eq!(Format::R8Unorm.row_bytes(5), 5);
    assert_eq!(Format::D32Float.row_bytes(2), 8);
    assert_eq!(Format::Bc1.row_bytes(4), 8);
    assert_eq!(Format::Bc1.row_bytes(5), 16);
    assert_eq!(Format::Bc7.row_bytes(8), 32);
    assert_eq!(Format::Bc7.row_bytes(1), 16);
}

#[test]
fn texture_desc_d2_has_plain_defaults() {
    let d = TextureDesc::d2(Format::Bgra8Unorm, 64, 32);
    assert_eq!(d.dim, TexDim::D2);
    assert_eq!(d.format, Format::Bgra8Unorm);
    assert_eq!(d.width, 64);
    assert_eq!(d.height, 32);
    assert_eq!(d.depth_or_layers, 1);
    assert_eq!(d.mips, 1);
    assert!(!d.render_target);
    assert!(!d.unordered_access);
}

#[test]
fn vertex_format_byte_size_matches_the_d3d12_element_sizes() {
    for f in [
        VertexFormat::Unorm8,
        VertexFormat::Snorm8,
        VertexFormat::Uint8,
        VertexFormat::Sint8,
    ] {
        assert_eq!(f.byte_size(), 1, "{f:?}");
    }
    for f in [
        VertexFormat::Half,
        VertexFormat::Unorm16,
        VertexFormat::Uint16,
        VertexFormat::Unorm8x2,
        VertexFormat::Uint8x2,
    ] {
        assert_eq!(f.byte_size(), 2, "{f:?}");
    }
    for f in [
        VertexFormat::Float,
        VertexFormat::Uint,
        VertexFormat::Sint,
        VertexFormat::Unorm8x4,
        VertexFormat::Uint8x4,
        VertexFormat::Unorm1010102,
        VertexFormat::FloatRG11B10,
    ] {
        assert_eq!(f.byte_size(), 4, "{f:?}");
    }
    for f in [
        VertexFormat::Float2,
        VertexFormat::Half4,
        VertexFormat::Unorm16x4,
        VertexFormat::Uint16x4,
    ] {
        assert_eq!(f.byte_size(), 8, "{f:?}");
    }
    assert_eq!(VertexFormat::Float3.byte_size(), 12);
    assert_eq!(VertexFormat::Float4.byte_size(), 16);
    assert_eq!(VertexFormat::Uint4.byte_size(), 16);
}

#[test]
fn blend_second_source_reads_only_the_src1_factors() {
    let plain = Blend {
        src: BlendFactor::SrcColor,
        dst: BlendFactor::One,
        op: BlendOp::Add,
        src_alpha: BlendFactor::SrcAlpha,
        dst_alpha: BlendFactor::One,
        op_alpha: BlendOp::Add,
    };
    assert!(!plain.second_source());
    assert!(
        Blend {
            src: BlendFactor::Src1Color,
            ..plain
        }
        .second_source()
    );
    assert!(
        Blend {
            dst: BlendFactor::InvSrc1Color,
            ..plain
        }
        .second_source()
    );
    assert!(
        Blend {
            src_alpha: BlendFactor::Src1Alpha,
            ..plain
        }
        .second_source()
    );
    assert!(
        Blend {
            dst_alpha: BlendFactor::InvSrc1Alpha,
            ..plain
        }
        .second_source()
    );
    assert!(
        !Blend {
            src: BlendFactor::SrcAlphaSat,
            ..plain
        }
        .second_source()
    );
}

#[test]
fn default_color_state_blends_nothing_and_writes_everything() {
    let c = ColorState::default();
    assert_eq!(c.blend, None);
    assert_eq!(c.write_mask, 0xf);
    assert_eq!(Cull::default(), Cull::Back);
}
