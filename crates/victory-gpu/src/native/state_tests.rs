//! Graphics pipeline state on the native host: blend, write mask, stencil, culling and winding,
//! viewport and scissor, as a title's pipeline and list calls set them, drawn with our own HLSL
//! (`victory-metal/tests/fixtures/tri_*`) into render targets and read back.

use super::{Pool, fixture, host, run};
use crate::native::Host;
use crate::native::list::NativeList;
use std::rc::Rc;
use victory_backend::resources::*;
use victory_metal::metal::Metal;

type Vertex = [f32; 8];
const W: usize = 16;

fn vertex(x: f32, y: f32, c: [f32; 4]) -> Vertex {
    [x, y, 0.5, 0.5, c[0], c[1], c[2], c[3]]
}

fn triangle(c: [f32; 4]) -> [Vertex; 3] {
    [vertex(-0.5, -0.5, c), vertex(0.5, -0.5, c), vertex(0.0, 0.5, c)]
}

const ATTRS: [VertexAttr; 3] = [
    VertexAttr {
        format: VertexFormat::Float2,
        slot: 0,
        offset: 0,
    },
    VertexAttr {
        format: VertexFormat::Float2,
        slot: 0,
        offset: 8,
    },
    VertexAttr {
        format: VertexFormat::Float4,
        slot: 0,
        offset: 16,
    },
];

const STREAMS: [VertexStream; 1] = [VertexStream {
    slot: 0,
    stride: 32,
    per_instance: false,
}];

/// A host, a white texture and a sampler in heaps, and a way to make pipelines of the triangle
/// shaders with a given state.
struct Rig {
    h: Host,
    pool: Pool,
    vheap: Rc<crate::native::list::NativeHeap<Metal>>,
    sheap: Rc<crate::native::list::NativeHeap<Metal>>,
    _slots: (Vec<u32>, Vec<u32>),
}

impl Rig {
    fn new() -> Option<Rig> {
        let h = host()?;
        let pool = Pool::new(&h, 2);
        let white = h
            .texture(&TextureDesc::d2(Format::Rgba8Unorm, 1, 1))
            .unwrap();
        h.backend.upload_texture(&white, 0, 0, &[255; 4], 4, 0);
        let key = 0x57A7E;
        h.register_texture(key, &white);
        let views = vec![0u32; 8 * 4];
        let smp = vec![0u32; 4 * 4];
        let (vstart, sstart) = (views.as_ptr() as u64, smp.as_ptr() as u64);
        let vheap = h.heap(HeapKind::Resource, vstart, 4).unwrap();
        let sheap = h.heap(HeapKind::Sampler, sstart, 4).unwrap();
        let mut srv = [0u32; 10];
        (srv[0], srv[1], srv[5]) = (28, 4, 1);
        crate::descriptor::create_srv(std::ptr::null(), key as _, &srv, vstart);
        let mut sw = [0u32; 27];
        (sw[1], sw[2], sw[6], sw[7], sw[8]) = (1, 1, 3, 3, 3);
        crate::descriptor::create_sampler_x(std::ptr::null(), &sw, sstart);
        assert!(h.mirror(vstart).unwrap() && h.mirror(sstart).unwrap());
        Some(Rig {
            h,
            pool,
            vheap,
            sheap,
            _slots: (views, smp),
        })
    }

    fn pipeline(
        &self,
        depth: Option<Format>,
        color_state: &[ColorState],
        configure: impl FnOnce(&mut RenderPipelineDesc<'_>),
    ) -> Rc<<Metal as GraphicsBackend>::RenderPipeline> {
        let (vs, ps) = (fixture("tri_vs"), fixture("tri_ps"));
        let mut desc = RenderPipelineDesc {
            vs: &vs,
            ps: Some(&ps),
            vs_layout: None,
            ps_layout: None,
            attributes: &ATTRS,
            streams: &STREAMS,
            color: &[Format::Rgba8Unorm],
            depth,
            cull: Cull::None,
            color_state,
            ..Default::default()
        };
        configure(&mut desc);
        self.h.render_pipeline(&desc).unwrap()
    }

    /// The triangle's vertex shader with pixel shader `ps` (translated DXIL) and `color_state`.
    fn pipeline_with_pixel(
        &self,
        ps: &victory_shader::emit::HostShader,
        color_state: &[ColorState],
        background: bool,
    ) -> Result<Rc<<Metal as GraphicsBackend>::RenderPipeline>, String> {
        let vs = fixture("tri_vs");
        let ps_layout = crate::native::list::Binding::of(ps).layout();
        let desc = RenderPipelineDesc {
            vs: &vs,
            ps: Some(&ps.dxil),
            ps_layout: ps_layout.as_ref(),
            attributes: &ATTRS,
            streams: &STREAMS,
            color: &[Format::Rgba8Unorm],
            cull: Cull::None,
            color_state,
            ..Default::default()
        };
        assert!(!desc.dual_source, "exercise inference from blend state");
        if background {
            self.h.backend.create_render_pipeline_within(&desc, std::time::Duration::from_secs(30))?
                .map(Rc::new).ok_or_else(|| "render compile timed out".into())
        } else {
            self.h.render_pipeline(&desc)
        }
    }

    fn target(&self, format: Format, depth: bool) -> Rc<<Metal as ResourceBackend>::Texture> {
        let mut d = TextureDesc::d2(format, W, W);
        d.render_target = true;
        let _ = depth;
        self.h.texture(&d).unwrap()
    }

    /// A list that clears `color` to `clear` and draws each vertex buffer (at pool offset) with
    /// `pso`, with whatever `state` sets first.
    fn list(
        &self,
        color: &Rc<<Metal as ResourceBackend>::Texture>,
        depth: Option<&Rc<<Metal as ResourceBackend>::Texture>>,
        clear: [f32; 4],
    ) -> NativeList<Metal> {
        let mut l = NativeList::default();
        l.set_render_targets(std::slice::from_ref(color), depth);
        l.clear_render_target(color, clear);
        if let Some(d) = depth {
            l.clear_depth_stencil(d, Some(1.0), Some(0));
        }
        l.set_heaps(Some(&self.vheap), Some(&self.sheap));
        l.set_graphics_root_constants(0, &[0.5f32.to_bits(), 0, 0, 0]);
        l.set_graphics_root_table(1, &self.vheap, 0);
        l.set_graphics_root_table(2, &self.sheap, 0);
        l
    }

    fn draw(
        &self,
        l: &mut NativeList<Metal>,
        pso: &Rc<<Metal as GraphicsBackend>::RenderPipeline>,
        at: usize,
        verts: &[Vertex],
    ) {
        self.pool.write(at, verts);
        l.set_graphics_pipeline(pso, Topology::Triangles);
        l.set_vertex_buffer(0, self.pool.va(at));
        l.draw_instanced(verts.len() as u32, 1, 0, 0).unwrap();
    }

    fn read(&self, color: &Rc<<Metal as ResourceBackend>::Texture>) -> Vec<u8> {
        let mut out = vec![0u8; W * W * 4];
        self.h.backend.read_texture(color, 0, 0, &mut out, W * 4, 0);
        out
    }
}

fn px(out: &[u8], x: usize, y: usize) -> [u8; 4] {
    <[u8; 4]>::try_from(&out[(y * W + x) * 4..][..4]).unwrap()
}

fn near(a: [u8; 4], b: [u8; 4]) -> bool {
    a.iter().zip(b).all(|(x, y)| x.abs_diff(y) <= 2)
}

/// Blending: half-transparent red over blue with (src alpha, 1 - src alpha) gives purple; the
/// write mask keeps a channel out; the blend constant works as a factor.
#[test]
fn blend_and_write_mask_are_applied() {
    let Some(r) = Rig::new() else { return };
    let color = r.target(Format::Rgba8Unorm, false);
    let alpha_blend = Blend {
        src: BlendFactor::SrcAlpha,
        dst: BlendFactor::InvSrcAlpha,
        op: BlendOp::Add,
        src_alpha: BlendFactor::One,
        dst_alpha: BlendFactor::InvSrcAlpha,
        op_alpha: BlendOp::Add,
    };
    let state = [ColorState {
        blend: Some(alpha_blend),
        write_mask: 0xf,
    }];
    let pso = r.pipeline(None, &state, |_| {});
    let mut l = r.list(&color, None, [0.0, 0.0, 1.0, 1.0]);
    r.draw(&mut l, &pso, 0, &triangle([1.0, 0.0, 0.0, 0.5]));
    run(&r.h, &[&l]);
    let out = r.read(&color);
    assert!(near(px(&out, 8, 8), [128, 0, 128, 255]), "{:?}", px(&out, 8, 8));
    assert_eq!(px(&out, 0, 0), [0, 0, 255, 255]);

    // Without blending but with the green and blue channels masked, only red lands.
    let masked = [ColorState {
        blend: None,
        write_mask: 0b0001,
    }];
    let pso = r.pipeline(None, &masked, |_| {});
    let mut l = r.list(&color, None, [0.0, 0.0, 1.0, 1.0]);
    r.draw(&mut l, &pso, 0, &triangle([1.0, 1.0, 1.0, 1.0]));
    run(&r.h, &[&l]);
    assert_eq!(px(&r.read(&color), 8, 8), [255, 0, 255, 255]);

    // The blend constant: factor (constant, 1 - constant) with 0.25 gives a quarter white.
    let constant = [ColorState {
        blend: Some(Blend {
            src: BlendFactor::Constant,
            dst: BlendFactor::InvConstant,
            op: BlendOp::Add,
            src_alpha: BlendFactor::One,
            dst_alpha: BlendFactor::Zero,
            op_alpha: BlendOp::Add,
        }),
        write_mask: 0xf,
    }];
    let pso = r.pipeline(None, &constant, |_| {});
    let mut l = r.list(&color, None, [0.0, 0.0, 0.0, 1.0]);
    l.set_blend_factor([0.25; 4]);
    r.draw(&mut l, &pso, 0, &triangle([1.0, 1.0, 1.0, 1.0]));
    run(&r.h, &[&l]);
    assert!(near(px(&r.read(&color), 8, 8), [64, 64, 64, 255]));
}

/// Blending with a second source colour (`Src1` factors): the pixel program exports mrt0
/// (1, 0, 0, 1) and mrt1 (0.5, 0.5, 0.5, 0.5), the console's second source. With factors
/// (source 1, zero) the target takes mrt0 x mrt1. Before, the pixel function was converted with
/// mrt1 as a second attachment's colour and Metal refused the pipeline ("Fragment shader does
/// not write to render target color(0), index(1) that is required for blending").
#[test]
fn a_second_source_colour_blends() {
    let Some(r) = Rig::new() else { return };
    let words: [u32; 8] = [
        0x7E00_02F2, // v_mov_b32 v0, 1.0
        0x7E02_02F0, // v_mov_b32 v1, 0.5
        0x7E04_0280, // v_mov_b32 v2, 0
        0xF800_000F, 0x0002_0200, // exp mrt0 v0, v2, v2, v0
        0xF800_181F, 0x0101_0101, // exp mrt1 v1, v1, v1, v1 done vm
        0xBF81_0000, // s_endpgm
    ];
    let bytes: Vec<_> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    let mut meta = [0; 0x10];
    meta[12] = 0x40;
    let c = victory_shader::emit::graphics::container(victory_shader::native::HwStage::Ps, &bytes, &meta).unwrap();
    let ps = crate::native::shaders::translate(&c).unwrap();
    let state = [ColorState {
        blend: Some(Blend {
            src: BlendFactor::Src1Color,
            dst: BlendFactor::Zero,
            op: BlendOp::Add,
            src_alpha: BlendFactor::Src1Alpha,
            dst_alpha: BlendFactor::Zero,
            op_alpha: BlendOp::Add,
        }),
        write_mask: 0xf,
    }];
    for background in [false, true] {
        let pso = r.pipeline_with_pixel(&ps, &state, background).unwrap();
        let color = r.target(Format::Rgba8Unorm, false);
        let mut l = r.list(&color, None, [0.0, 0.0, 1.0, 1.0]);
        r.draw(&mut l, &pso, 0, &triangle([1.0; 4]));
        run(&r.h, &[&l]);
        let out = r.read(&color);
        assert!(near(px(&out, 8, 8), [128, 0, 0, 128]), "{:?}", px(&out, 8, 8));
        assert_eq!(px(&out, 0, 0), [0, 0, 255, 255]);
        // Without a second-source factor the same program is a plain one: mrt0 lands as it is.
        let pso = r.pipeline_with_pixel(&ps, &[], background).unwrap();
        let mut l = r.list(&color, None, [0.0, 0.0, 1.0, 1.0]);
        r.draw(&mut l, &pso, 0, &triangle([1.0; 4]));
        run(&r.h, &[&l]);
        assert_eq!(px(&r.read(&color), 8, 8), [255, 0, 0, 255]);
    }
}

/// Viewport and scissor: the viewport maps the triangle into the left half; the scissor cuts the
/// right of the target away; an empty scissor draws nothing.
#[test]
fn viewport_and_scissor_are_applied() {
    let Some(r) = Rig::new() else { return };
    let color = r.target(Format::Rgba8Unorm, false);
    let pso = r.pipeline(None, &[], |_| {});
    let red = triangle([1.0, 0.0, 0.0, 1.0]);

    let mut l = r.list(&color, None, [0.0, 0.0, 1.0, 1.0]);
    l.set_viewport(Some(Viewport {
        x: 0.0,
        y: 0.0,
        width: 8.0,
        height: 16.0,
        min_depth: 0.0,
        max_depth: 1.0,
    }));
    r.draw(&mut l, &pso, 0, &red);
    run(&r.h, &[&l]);
    let out = r.read(&color);
    assert_eq!(px(&out, 4, 8), [255, 0, 0, 255], "inside the viewport");
    assert_eq!(px(&out, 12, 8), [0, 0, 255, 255], "right of the viewport");

    let mut l = r.list(&color, None, [0.0, 0.0, 1.0, 1.0]);
    l.set_scissor(Some([0, 0, 8, 16]));
    r.draw(&mut l, &pso, 0, &red);
    run(&r.h, &[&l]);
    let out = r.read(&color);
    assert_eq!(px(&out, 6, 8), [255, 0, 0, 255], "inside the scissor");
    assert_eq!(px(&out, 10, 8), [0, 0, 255, 255], "right of it");

    // A rectangle outside the target is clipped to it, an empty one draws nothing.
    let mut l = r.list(&color, None, [0.0, 0.0, 1.0, 1.0]);
    l.set_scissor(Some([-100, -100, 100, 100]));
    r.draw(&mut l, &pso, 0, &red);
    l.set_scissor(Some([4, 4, 4, 12]));
    r.draw(&mut l, &pso, 4096, &triangle([0.0, 1.0, 0.0, 1.0]));
    run(&r.h, &[&l]);
    assert_eq!(px(&r.read(&color), 8, 8), [255, 0, 0, 255]);
}

/// Stencil: a triangle writes 1 where it covers (replace, reference 1); a second, larger one
/// passes only where the stencil equals 1, so it paints over the first and nowhere else.
#[test]
fn stencil_test_and_reference_are_applied() {
    let Some(r) = Rig::new() else { return };
    let color = r.target(Format::Rgba8Unorm, false);
    let depth = r.target(Format::D32FloatS8, true);
    let face = |compare, pass| StencilFace {
        fail: StencilOp::Keep,
        depth_fail: StencilOp::Keep,
        pass,
        compare,
    };
    let stencil = |compare, pass| StencilState {
        read_mask: 0xff,
        write_mask: 0xff,
        front: face(compare, pass),
        back: face(compare, pass),
    };
    let write = r.pipeline(Some(Format::D32FloatS8), &[], |d| {
        d.stencil = Some(stencil(Compare::Always, StencilOp::Replace))
    });
    let test = r.pipeline(Some(Format::D32FloatS8), &[], |d| {
        d.stencil = Some(stencil(Compare::Equal, StencilOp::Keep))
    });
    let mut l = r.list(&color, Some(&depth), [0.0, 0.0, 1.0, 1.0]);
    l.set_stencil_ref(1);
    r.draw(&mut l, &write, 0, &triangle([1.0, 0.0, 0.0, 1.0]));
    let green = [0.0, 1.0, 0.0, 1.0];
    let big = [vertex(-3.0, -1.0, green), vertex(3.0, -1.0, green), vertex(0.0, 5.0, green)];
    r.draw(&mut l, &test, 4096, &big);
    run(&r.h, &[&l]);
    let out = r.read(&color);
    assert_eq!(px(&out, 8, 8), [0, 255, 0, 255], "stencil equal: painted");
    assert_eq!(px(&out, 1, 8), [0, 0, 255, 255], "stencil zero: left alone");
    assert_eq!(px(&out, 14, 14), [0, 0, 255, 255], "stencil zero: left alone");
}

/// A colour target smaller than its depth target (a 1920x1080 target over a 3840x2160 depth
/// buffer in GTA VI's story load) draws into the area both cover, as the console does, with a
/// scissor of the whole depth target. The pass used to be refused, and the whole piece with it.
#[test]
fn a_depth_target_larger_than_the_colour_target_still_draws() {
    let Some(r) = Rig::new() else { return };
    let color = r.target(Format::Rgba8Unorm, false);
    let mut d = TextureDesc::d2(Format::D32FloatS8, W * 2, W * 2);
    d.render_target = true;
    let depth = r.h.texture(&d).unwrap();
    let pso = r.pipeline(Some(Format::D32FloatS8), &[], |d| d.depth_test = Some((true, Compare::Always)));
    let mut l = r.list(&color, Some(&depth), [0.0, 0.0, 1.0, 1.0]);
    l.set_scissor(Some([0, 0, 2 * W as i32, 2 * W as i32]));
    r.draw(&mut l, &pso, 0, &triangle([1.0, 0.0, 0.0, 1.0]));
    run(&r.h, &[&l]);
    let out = r.read(&color);
    assert_eq!(px(&out, 8, 8), [255, 0, 0, 255], "inside the triangle");
    assert_eq!(px(&out, 0, 0), [0, 0, 255, 255], "cleared corner");
}

/// Clearing depth must preserve stencil written by an earlier draw.
#[test]
fn depth_only_clear_preserves_stencil() {
    let Some(r) = Rig::new() else { return };
    let color = r.target(Format::Rgba8Unorm, false);
    let depth = r.target(Format::D32FloatS8, true);
    let stencil = |compare, pass| {
        let face = StencilFace {
            fail: StencilOp::Keep,
            depth_fail: StencilOp::Keep,
            pass,
            compare,
        };
        StencilState {
            read_mask: 0xff,
            write_mask: 0xff,
            front: face,
            back: face,
        }
    };
    let write = r.pipeline(Some(Format::D32FloatS8), &[], |d| {
        d.stencil = Some(stencil(Compare::Always, StencilOp::Replace));
        d.depth_test = Some((true, Compare::Always));
    });
    let test = r.pipeline(Some(Format::D32FloatS8), &[], |d| {
        d.stencil = Some(stencil(Compare::Equal, StencilOp::Keep));
        d.depth_test = Some((false, Compare::Less));
    });
    let mut l = r.list(&color, Some(&depth), [0.0, 0.0, 1.0, 1.0]);
    l.set_stencil_ref(37);
    r.draw(&mut l, &write, 0, &triangle([1.0, 0.0, 0.0, 1.0]));
    l.clear_depth(&depth, 1.0);
    r.draw(&mut l, &test, 4096, &triangle([0.0, 1.0, 0.0, 1.0]));
    run(&r.h, &[&l]);
    assert_eq!(
        px(&r.read(&color), 8, 8),
        [0, 255, 0, 255],
        "depth-only clear erased stencil"
    );
}

/// Nonzero and zero clears, either stencil alone or both planes. The draw checks the exact
/// depth and stencil together, then mismatching draws must be rejected by each test in turn.
#[test]
fn stencil_clears_select_the_requested_planes_and_value() {
    let Some(r) = Rig::new() else { return };
    let color = r.target(Format::Rgba8Unorm, false);
    let depth = r.target(Format::D32FloatS8, true);
    let pso = r.pipeline(Some(Format::D32FloatS8), &[], |d| {
        let face = StencilFace {
            fail: StencilOp::Keep,
            depth_fail: StencilOp::Keep,
            pass: StencilOp::Keep,
            compare: Compare::Equal,
        };
        d.stencil = Some(StencilState {
            read_mask: 0xff,
            write_mask: 0xff,
            front: face,
            back: face,
        });
        d.depth_test = Some((false, Compare::Equal));
    });
    for clear_depth in [None, Some(0.75f32)] {
        for value in [0u8, 173, 255] {
            let mut l = r.list(&color, Some(&depth), [0.0, 0.0, 1.0, 1.0]);
            l.clear_depth_stencil(&depth, Some(0.25), Some(!value));
            l.clear_depth_stencil(&depth, clear_depth, Some(value));
            let expected_depth = clear_depth.unwrap_or(0.25);
            l.set_graphics_root_constants(0, &[expected_depth.to_bits(), 0, 0, 0]);
            l.set_stencil_ref(u32::from(value));
            r.draw(&mut l, &pso, 0, &triangle([0.0, 1.0, 0.0, 1.0]));
            l.set_stencil_ref(u32::from(!value));
            r.draw(&mut l, &pso, 4096, &triangle([1.0, 0.0, 0.0, 1.0]));
            l.set_stencil_ref(u32::from(value));
            l.set_graphics_root_constants(0, &[0.5f32.to_bits(), 0, 0, 0]);
            r.draw(&mut l, &pso, 8192, &triangle([0.0, 0.0, 1.0, 1.0]));
            // Submission snapshots copy commands, not the mutable recording state.
            let snapshot = l.snapshot();
            run(&r.h, &[&snapshot]);
            assert_eq!(
                px(&r.read(&color), 8, 8),
                [0, 255, 0, 255],
                "depth={clear_depth:?}, stencil={value}"
            );
        }
    }
}

/// Culling and winding: the triangle's vertices run counter-clockwise on screen; back-face
/// culling with clockwise fronts (D3D's default) removes it, with counter-clockwise fronts keeps
/// it; front-face culling does the opposite.
#[test]
fn culling_follows_the_winding_and_its_front_face() {
    let Some(r) = Rig::new() else { return };
    let color = r.target(Format::Rgba8Unorm, false);
    let red = [1.0, 0.0, 0.0, 1.0];
    // Counter-clockwise as seen on screen (y up in clip space).
    let ccw = [vertex(-0.5, -0.5, red), vertex(0.5, -0.5, red), vertex(0.0, 0.5, red)];
    let drawn = |cull, front_ccw| {
        let pso = r.pipeline(None, &[], |d| {
            d.cull = cull;
            d.raster.front_ccw = front_ccw;
        });
        let mut l = r.list(&color, None, [0.0, 0.0, 1.0, 1.0]);
        r.draw(&mut l, &pso, 0, &ccw);
        run(&r.h, &[&l]);
        px(&r.read(&color), 8, 8) == [255, 0, 0, 255]
    };
    assert!(!drawn(Cull::Back, false), "back culled: it is a back face when fronts are clockwise");
    assert!(drawn(Cull::Back, true));
    assert!(drawn(Cull::Front, false));
    assert!(!drawn(Cull::Front, true));
    assert!(drawn(Cull::None, false));
}

/// Depth bias applies to a draw and resets to zero on the following draw in the same render pass.
#[test]
fn depth_bias_resets_between_draws_in_the_same_pass() {
    let Some(r) = Rig::new() else { return };
    let color = r.target(Format::Rgba8Unorm, false);
    let depth = r.target(Format::D32Float, true);
    let biased = r.pipeline(Some(Format::D32Float), &[], |d| {
        d.depth_test = Some((true, Compare::Always));
        d.raster.depth_bias = (10000.0, 0.0, 0.0);
    });
    let unbiased = r.pipeline(Some(Format::D32Float), &[], |d| {
        d.depth_test = Some((false, Compare::Less));
    });

    let mut l = r.list(&color, Some(&depth), [0.0; 4]);
    // Draw 1 pushes depth back with positive bias; Draw 2 at the same Z with zero bias must pass Less.
    r.draw(&mut l, &biased, 0, &triangle([1.0, 0.0, 0.0, 1.0]));
    r.draw(&mut l, &unbiased, 256, &triangle([0.0, 1.0, 0.0, 1.0]));
    run(&r.h, &[&l]);
    let out = r.read(&color);
    assert_eq!(
        px(&out, 8, 8),
        [0, 255, 0, 255],
        "zero depth bias must not inherit the previous draw's bias"
    );
}

/// A depth texture that a compute shader writes through unordered access and draws then test
/// EQUAL against (the title's material depth: `tiled_lighting` writes a material number per
/// texel as depth and draws each material's tiles at that depth). Metal's depth textures take
/// no shader writes, so the views show a writable twin (`native/twin.rs`) and the contents move
/// between the two: the cleared depth and stencil into the twin before the dispatch, what the
/// dispatch left back into the depth texture before the draws. The fill is our own HLSL
/// (`tex_fill`: the texel at (x, y) gets x / 255), run over the top half only.
#[test]
fn depth_a_compute_shader_writes_is_what_draws_test_equal_against() {
    let Some(r) = Rig::new() else { return };
    let color = r.target(Format::Rgba8Unorm, false);
    let mut desc = TextureDesc::d2(Format::D32FloatS8, W, W);
    (desc.render_target, desc.unordered_access) = (true, true);
    let depth = r.h.texture(&desc).unwrap();
    let key = 0xDE97_0000;
    r.h.register_texture(key, &depth);
    r.h.register_depth(key, &depth, &desc);
    // The title's views of the two planes, as it makes them for its 16-bit depth with stencil:
    // R16_UNORM (56) of plane 0, R8_UINT (62) of plane 1; 2D, level 0.
    let slots = vec![0u32; 8 * 2];
    let start = slots.as_ptr() as u64;
    let heap = r.h.heap(HeapKind::Resource, start, 2).unwrap();
    for (i, (format, plane)) in [(56u32, 0u32), (62, 1)].into_iter().enumerate() {
        let uav = [format, 4, 0, plane, 0, 0, 0, 0, 0, 0];
        let slot = start + 32 * i as u64;
        crate::descriptor::create_uav(std::ptr::null(), key as _, std::ptr::null(), &uav, slot);
        assert!(r.h.mirror(slot).unwrap());
    }
    let mut l = NativeList::default();
    l.clear_depth_stencil(&depth, Some(0.25), Some(173));
    let fill = r.h.compute_pipeline(&fixture("tex_fill"), [8, 8, 1]).unwrap();
    l.set_heaps(Some(&heap), None);
    l.set_compute_pipeline(&fill);
    l.set_compute_root_table(0, &heap, 0);
    l.dispatch(W as u32 / 8, 1, 1).unwrap();
    run(&r.h, &[&l]);
    assert_eq!(r.h.twin_copies(), 1, "the cleared depth goes into the twin before the dispatch");
    // The twin's depth plane: the fill in the top half, the clear below it.
    let plane = r.h.twin_plane_of(key, 0).expect("a twin");
    let mut raw = vec![0u8; W * W * 4];
    r.h.backend.read_texture(&plane, 0, 0, &mut raw, W * 4, 0);
    let at = |x: usize, y: usize| f32::from_le_bytes(raw[(y * W + x) * 4..][..4].try_into().unwrap());
    assert_eq!(at(3, 12), 0.25);
    assert!((at(5, 2) - 5.0 / 255.0).abs() < 1e-6 && (at(6, 2) - 6.0 / 255.0).abs() < 1e-6, "{} {}", at(5, 2), at(6, 2));
    // Draws over the whole target that pass only where depth and stencil are EQUAL: green at
    // the cleared depth, red at the depth the fill gave column 5.
    let face = StencilFace { fail: StencilOp::Keep, depth_fail: StencilOp::Keep, pass: StencilOp::Keep, compare: Compare::Equal };
    let pso = r.pipeline(Some(Format::D32FloatS8), &[], |d| {
        d.depth_test = Some((false, Compare::Equal));
        d.stencil = Some(StencilState { read_mask: 0xff, write_mask: 0, front: face, back: face });
    });
    let whole = |c| [vertex(-1.0, -1.0, c), vertex(3.0, -1.0, c), vertex(-1.0, 3.0, c)];
    let mut l = r.list(&color, None, [0.0, 0.0, 1.0, 1.0]);
    l.set_render_targets(std::slice::from_ref(&color), Some(&depth));
    l.set_stencil_ref(173);
    l.set_graphics_root_constants(0, &[0.25f32.to_bits(), 0, 0, 0]);
    r.draw(&mut l, &pso, 0, &whole([0.0, 1.0, 0.0, 1.0]));
    l.set_graphics_root_constants(0, &[at(5, 2).to_bits(), 0, 0, 0]);
    r.draw(&mut l, &pso, 4096, &whole([1.0, 0.0, 0.0, 1.0]));
    run(&r.h, &[&l]);
    assert_eq!(r.h.twin_copies(), 2, "what the dispatch wrote goes into the depth texture before the draws");
    let out = r.read(&color);
    let (blue, green, red) = ([0, 0, 255, 255], [0, 255, 0, 255], [255, 0, 0, 255]);
    for y in 0..W {
        for x in 0..W {
            let want = if y >= 8 { green } else if x == 5 { red } else { blue };
            assert_eq!(px(&out, x, y), want, "at ({x}, {y})");
        }
    }
    // A stencil reference the cleared plane does not hold: nothing passes (the stencil plane
    // went through the twin and back too).
    let mut l = r.list(&color, None, [0.0, 0.0, 1.0, 1.0]);
    l.set_render_targets(std::slice::from_ref(&color), Some(&depth));
    l.set_stencil_ref(172);
    l.set_graphics_root_constants(0, &[0.25f32.to_bits(), 0, 0, 0]);
    r.draw(&mut l, &pso, 0, &whole([0.0, 1.0, 0.0, 1.0]));
    run(&r.h, &[&l]);
    assert!(r.read(&color).chunks_exact(4).all(|p| p == blue));
    drop(slots);
}

/// `ClearUnorderedAccessViewFloat` / `...Uint` (the title clears its sun shadow mask to white
/// this way before the tiled shadow passes darken it): a texture view's level is cleared to the
/// values, a depth texture's on its twin's plane, a buffer view is filled.
#[test]
fn unordered_access_clears_write_the_view() {
    use crate::native::host::ClearTarget;
    let Some(r) = Rig::new() else { return };
    let slots = vec![0u32; 8 * 4];
    let start = slots.as_ptr() as u64;
    let view_at = |i: u64, key: usize, words: [u32; 10]| {
        crate::descriptor::create_uav(std::ptr::null(), key as _, std::ptr::null(), &words, start + 32 * i);
        crate::descriptor::view(start + 32 * i).expect("a view")
    };
    let texture = |view: &crate::descriptor::View| match r.h.unordered_clear_target(view) {
        Some((ClearTarget::Texture(t), ..)) => t,
        _ => panic!("a texture to clear"),
    };
    // An 8-bit mask the title made a render target and unordered access, cleared to 1.0.
    let mut desc = TextureDesc::d2(Format::R8Unorm, W, W);
    (desc.render_target, desc.unordered_access) = (true, true);
    let mask = r.h.texture(&desc).unwrap();
    r.h.register_texture(0xC1EA_0001, &mask);
    r.h.register_attachable(0xC1EA_0001);
    // A 32-bit integer texture, cleared to 7.
    let mut desc = TextureDesc::d2(Format::R32Uint, W, W);
    (desc.render_target, desc.unordered_access) = (true, true);
    let ids = r.h.texture(&desc).unwrap();
    r.h.register_texture(0xC1EA_0002, &ids);
    r.h.register_attachable(0xC1EA_0002);
    // One that is unordered access alone cannot be an attachment: not cleared (a shortcut).
    let mut desc = TextureDesc::d2(Format::R8Unorm, W, W);
    desc.unordered_access = true;
    r.h.register_texture(0xC1EA_0005, &r.h.texture(&desc).unwrap());
    crate::descriptor::create_uav(std::ptr::null(), 0xC1EA_0005usize as _, std::ptr::null(), &[61, 4, 0, 0, 0, 0, 0, 0, 0, 0], start);
    assert!(r.h.unordered_clear_target(&crate::descriptor::view(start).unwrap()).is_none());
    // A depth texture: plane 0 of its twin, cleared to 0.5.
    let mut desc = TextureDesc::d2(Format::D32FloatS8, W, W);
    (desc.render_target, desc.unordered_access) = (true, true);
    let depth = r.h.texture(&desc).unwrap();
    r.h.register_texture(0xC1EA_0003, &depth);
    r.h.register_depth(0xC1EA_0003, &depth, &desc);
    // A buffer of 64 dwords, filled with 0xffffffff.
    r.pool.write(0, &[0x1234_5678u32; 80]);
    r.h.register_buffer(0xC1EA_0004, r.pool.va(0), 256);
    let mut buffer = [0u32; 10];
    (buffer[1], buffer[4], buffer[8]) = (1, 64, 1);
    let views = [
        view_at(0, 0xC1EA_0001, [61, 4, 0, 0, 0, 0, 0, 0, 0, 0]),
        view_at(1, 0xC1EA_0002, [42, 4, 0, 0, 0, 0, 0, 0, 0, 0]),
        view_at(2, 0xC1EA_0003, [56, 4, 0, 0, 0, 0, 0, 0, 0, 0]),
        view_at(3, 0xC1EA_0004, buffer),
    ];
    let mut l = NativeList::default();
    l.clear_render_target(&texture(&views[0]), [1.0, 0.0, 0.0, 0.0]);
    l.clear_render_target(&texture(&views[1]), [7.0, 0.0, 0.0, 0.0]);
    let plane = texture(&views[2]);
    l.clear_render_target(&plane, [0.5, 0.0, 0.0, 0.0]);
    match r.h.unordered_clear_target(&views[3]) {
        Some((ClearTarget::Buffer(at, len), ..)) => l.fill(at, len, 0xff),
        _ => panic!("a buffer to clear"),
    }
    // Then a draw at depth 0.5 with the test EQUAL: the cleared plane reaches the depth texture.
    let color = r.target(Format::Rgba8Unorm, false);
    let pso = r.pipeline(Some(Format::D32FloatS8), &[], |d| d.depth_test = Some((false, Compare::Equal)));
    l.set_render_targets(std::slice::from_ref(&color), Some(&depth));
    l.clear_render_target(&color, [0.0, 0.0, 1.0, 1.0]);
    l.set_heaps(Some(&r.vheap), Some(&r.sheap));
    l.set_graphics_root_constants(0, &[0.5f32.to_bits(), 0, 0, 0]);
    l.set_graphics_root_table(1, &r.vheap, 0);
    l.set_graphics_root_table(2, &r.sheap, 0);
    r.draw(&mut l, &pso, 4096, &triangle([0.0, 1.0, 0.0, 1.0]));
    run(&r.h, &[&l]);
    let mut out = vec![0u8; W * W];
    r.h.backend.read_texture(&mask, 0, 0, &mut out, W, 0);
    assert!(out.iter().all(|&b| b == 255), "{:?}", &out[..8]);
    let mut out = vec![0u8; W * W * 4];
    r.h.backend.read_texture(&ids, 0, 0, &mut out, W * 4, 0);
    assert!(out.chunks_exact(4).all(|p| p == [7, 0, 0, 0]), "{:?}", &out[..8]);
    r.h.backend.read_texture(&plane, 0, 0, &mut out, W * 4, 0);
    assert!(out.chunks_exact(4).all(|p| p == 0.5f32.to_le_bytes()), "{:?}", &out[..8]);
    assert_eq!(r.pool.read::<u32>(0, 80), [vec![0xffff_ffff; 64], vec![0x1234_5678; 16]].concat());
    assert_eq!(px(&r.read(&color), 8, 8), [0, 255, 0, 255], "the draw passes where the twin's cleared depth arrived");
    drop(slots);
}

mod rect_program {
    include!("../../../victory-shader/tests/programs/rect.rs");
}

// (console draw arguments, complete tiles, instance rows). The converter subtracts the
// native vertex/instance bases before the shader sees either ID. Explicit expectations
// keep both tests independent of the host's rectangle count/start rewrite.
const RECT_DRAW_CASES: [([u32; 4], usize, usize); 5] = [
    ([6, 1, 0, 0], 2, 1), // IDs 0..5, instance 0.
    ([6, 1, 0, 1], 2, 1), // Base instance does not change the shader's IDs.
    ([4, 2, 3, 0], 1, 2), // IDs 0..2 complete tile 0; the trailing vertex is discarded.
    ([4, 2, 3, 1], 1, 2), // Same tile and rows with both bases nonzero.
    ([5, 1, 0, 1], 1, 1), // The last two vertices must not paint tile 1.
];

/// Console v0/v3 read SV_VertexID/SV_InstanceID directly, without adding draw bases.
/// Anchor rectangle coverage to ordinary triangles for every rectangle argument set.
#[test]
fn ordinary_vertex_and_instance_ids_exclude_draw_bases() {
    let Some(r) = Rig::new() else { return };
    let vs = crate::native::shaders::translate(&rect_program::vertex_container()).unwrap();
    let ps = crate::native::shaders::translate(&rect_program::pixel_container()).unwrap();
    let color = r.target(Format::Rgba8Unorm, false);
    let vs_layout = crate::native::list::Binding::of(&vs).layout();
    let ps_layout = crate::native::list::Binding::of(&ps).layout();
    let pso = r.h.render_pipeline(&RenderPipelineDesc {
        vs: &vs.dxil, ps: Some(&ps.dxil), vs_layout: vs_layout.as_ref(), ps_layout: ps_layout.as_ref(),
        color: &[Format::Rgba8Unorm], cull: Cull::None, ..Default::default()
    }).unwrap();
    r.pool.write(64, &[0u16, 1, 2, 3, 4, 5]);
    r.pool.write(32, &[1u32]);
    for (args, tiles, rows) in RECT_DRAW_CASES {
        for indexed in [false, true] {
            for indirect in 0..3 {
                // The indexed control uses indices 0..5 and the same base vertex. Metal
                // adds it to each index, and the converter subtracts it from SV_VertexID.
                if indexed {
                    r.pool.write(0, &[args[0], args[1], 0, args[2], args[3]]);
                } else {
                    r.pool.write(0, &args);
                }
                let mut l = NativeList::default();
                l.set_render_targets(std::slice::from_ref(&color), None);
                l.clear_render_target(&color, [0.0; 4]);
                l.set_graphics_pipeline(&pso, Topology::Triangles);
                if indexed { l.set_index_buffer(r.pool.va(64), false); }
                match (indirect, indexed) {
                    (0, false) => l.draw_instanced(args[0], args[1], args[2], args[3]).unwrap(),
                    (0, true) => l.draw_indexed_instanced(args[0], args[1], 0, args[2] as i32, args[3]).unwrap(),
                    (1, _) => l.draw_indirect(r.pool.va(0), indexed).unwrap(),
                    _ => l.draw_indirect_counted(r.pool.va(0), indexed, r.pool.va(32), 0).unwrap(),
                }
                run(&r.h, &[&l]);
                let out = r.read(&color);
                // Interior pixels of both halves, away from diagonal edges: ordinary
                // triangles leave the complementary half empty, even with a partial tail.
                for row in 0..2 {
                    for tile in 0..2 {
                        for (u, v, supplied) in [(1, 1, true), (5, 5, false)] {
                            let (x, y) = (tile * 8 + u, row * 8 + v);
                            let want = if supplied && tile < tiles && row < rows {
                                [21, 21, 64, 255]
                            } else { [0; 4] };
                            assert!(near(px(&out, x, y), want), "{args:?} indexed={indexed} indirect={indirect} pixel ({x},{y}): {:?}, expected {want:?}", px(&out, x, y));
                        }
                    }
                }
            }
        }
    }
}

/// Two tiles per instance, with positions and UVs derived only from id / 3 and id % 3.
/// Direct and GPU-written indirect records must cover both halves, with the same winding.
#[test]
fn vertex_id_rectangle_lists_cover_both_triangles_direct_and_indirect() {
    let Some(r) = Rig::new() else { return };
    let vs = crate::native::shaders::translate_rect(&rect_program::vertex_container()).unwrap();
    let ps = crate::native::shaders::translate(&rect_program::pixel_container()).unwrap();
    let color = r.target(Format::Rgba8Unorm, false);
    let vs_layout = crate::native::list::Binding::of(&vs).layout();
    let ps_layout = crate::native::list::Binding::of(&ps).layout();
    for (cull, front_ccw, visible) in [(Cull::Back, true, true), (Cull::Front, false, true), (Cull::Back, false, false)] {
        let pso = r.h.render_pipeline(&RenderPipelineDesc {
            vs: &vs.dxil, ps: Some(&ps.dxil), vs_layout: vs_layout.as_ref(), ps_layout: ps_layout.as_ref(),
            color: &[Format::Rgba8Unorm], cull,
            raster: Raster { front_ccw, ..Default::default() }, ..Default::default()
        }).unwrap();
        // Whole two-rectangle lists, nonzero base instance, and a nonzero first vertex with
        // two instances. Five vertices from zero leave the incomplete second tile on-screen,
        // so scaling count * 2 instead of count / 3 * 6 would paint an unwanted triangle.
        for (args, tiles, rows) in RECT_DRAW_CASES {
            for indirect in 0..3 {
                let mut l = NativeList::default();
                let count = 1u32;
                // Two instances cover both rows, so a failed gate paints pixels outside every
                // allowed draw above. Changing only base instance would redraw the same row.
                let denied = [6u32, 2, 0, 1];
                r.pool.write(512, &args);
                r.pool.write(528, &[count]);
                r.pool.write(544, &denied);
                r.pool.write(0, &[0u32; 16]);
                if indirect != 0 {
                    l.copy_buffer_region(r.pool.va(0), r.pool.va(512), 16);
                    l.copy_buffer_region(r.pool.va(16), r.pool.va(528), 4);
                    l.copy_buffer_region(r.pool.va(32), r.pool.va(544), 16);
                }
                l.set_render_targets(std::slice::from_ref(&color), None);
                l.clear_render_target(&color, [0.0; 4]);
                l.set_graphics_pipeline(&pso, Topology::Rectangles);
                match indirect {
                    0 => l.draw_instanced(args[0], args[1], args[2], args[3]).unwrap(),
                    1 => l.draw_indirect(r.pool.va(0), false).unwrap(),
                    _ => {
                        l.draw_indirect_counted(r.pool.va(0), false, r.pool.va(16), 0).unwrap();
                        l.draw_indirect_counted(r.pool.va(32), false, r.pool.va(16), 1).unwrap();
                    }
                }
                run(&r.h, &[&l]);
                let out = r.read(&color);
                // Every pixel, including all four quadrants and the missing corner's UVs.
                for y in 0..W {
                    for x in 0..W {
                        let (tile, instance) = (x / 8, y / 8);
                        let (u, v) = (x % 8, y % 8);
                        // As for ordinary draws, both converted IDs exclude their bases.
                        // For first=3, native IDs 6..11 become 0..5, replaying console IDs
                        // 0,1,2 for tile 0. Neither first vertex nor base instance shifts it.
                        let inside = visible && (1..7).contains(&u) && (1..7).contains(&v)
                            && tile < tiles && instance < rows;
                        let want = if inside {
                            [(((u as f32 - 0.5) / 6.0) * 255.0).round() as u8,
                             (((v as f32 - 0.5) / 6.0) * 255.0).round() as u8, 64, 255]
                        } else { [0; 4] };
                        assert!(near(px(&out, x, y), want), "{args:?} indirect={indirect} cull={cull:?} front_ccw={front_ccw} pixel ({x},{y}): {:?}, expected {want:?}", px(&out, x, y));
                    }
                }
                if indirect != 0 {
                    assert_eq!(r.pool.read::<u32>(0, 4), args, "the title's record is unchanged");
                    assert_eq!(r.pool.read::<u32>(16, 1), [count]);
                    assert_eq!(r.pool.read::<u32>(32, 4), denied, "the denied record is unchanged");
                }
            }
        }
    }
}
