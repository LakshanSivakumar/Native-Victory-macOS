//! The shader path of the native host: translation settings, the library cache, pipelines from
//! translated shaders and graphics state, and refusals that are counted instead of failing a
//! submission. Our own HLSL (`victory-metal/tests/fixtures`) stands in for translator output: a
//! `HostShader` is plain data. Skipped without the Metal Shader Converter.

use crate::native::memory::PoolTable;
use crate::native::shaders::{self, ShaderStats};
use crate::native::state::{self, Graphics};
use crate::native::title::{Pipe, Title};
use std::sync::Arc;
use victory_metal::metal::Metal;
use victory_shader::HwStage;
use victory_shader::emit::HostShader;

fn read(path: &str) -> Vec<u8> {
    let p = format!("{}/../victory-metal/{path}", env!("CARGO_MANIFEST_DIR"));
    std::fs::read(&p).unwrap_or_else(|e| panic!("{p}: {e}"))
}

fn title() -> Option<Title<Metal>> {
    if !std::path::Path::new(victory_metal::msc::DEFAULT_DYLIB).exists()
        && std::env::var_os("VICTORY_MSC_DYLIB").is_none()
    {
        eprintln!("skipped: Metal Shader Converter not installed");
        return None;
    }
    Some(Title::new(Metal::new().unwrap(), Arc::new(PoolTable::default())))
}

/// A translated shader as plain data: our own DXIL, no title arguments.
fn host_shader(stage: HwStage, dxil: Vec<u8>) -> HostShader {
    HostShader {
        stage,
        dxil,
        user_data: vec![],
        samplers: vec![],
        static_samplers: vec![],
        root_cbvs: vec![],
        root_srvs: vec![],
        root_uavs: vec![],
        memory: false,
        memory_writes: false,
        view_addresses: false,
        heap: false,
        heap_writes: false,
        view_marks: false,
        splits: vec![],
        table_entries: vec![],
    }
}

#[test]
fn the_host_translates_for_32_lane_waves() {
    let o = shaders::options();
    assert!(o.wave32_host);
    // Everything else is what D3D12 gets by default, except the workarounds `wave32_host` implies.
    assert!(!o.view_marks && !o.atomic_locks && !o.bias_as_gradients);
    assert!(shaders::translate(b"not a container").is_err());
}

#[test]
fn group_sizes_come_from_the_converted_container() {
    assert_eq!(shaders::threads_of(&read("tests/affine.dxil")), Some([64, 1, 1]));
    assert_eq!(shaders::threads_of(b"junk"), None);
}

/// One conversion per distinct DXIL: a second pipeline of the same shader (another title pipeline
/// object with the same container) is served from the library cache.
#[test]
fn compute_pipelines_are_cached_by_the_content_of_their_dxil() {
    let Some(mut t) = title() else { return };
    let cs = host_shader(HwStage::Cs, read("tests/affine.dxil"));
    let first = t.compute_pipe(&cs, "a");
    let again = t.compute_pipe(&cs.clone(), "b");
    let (Pipe::Compute(a), Pipe::Compute(b)) = (first, again) else {
        panic!("not made: {:?}", t.shaders)
    };
    assert!(std::rc::Rc::ptr_eq(&a, &b));
    assert_eq!((t.shaders.pipelines, t.shaders.cached), (1, 1));
    assert_eq!(a.threads, [64, 1, 1]);
    let other = host_shader(HwStage::Cs, read("tests/fixtures/heap_rw.dxil"));
    assert!(matches!(t.compute_pipe(&other, "c"), Pipe::Compute(_)));
    assert_eq!(t.shaders.pipelines, 2);
}

/// A compute pipeline Metal has not compiled within the frame's wait is put off, not refused: a
/// later use picks it up, and each miss is counted.
#[test]
fn a_compute_pipeline_put_off_is_made_by_a_later_use() {
    let Some(mut t) = title() else { return };
    let cs = host_shader(HwStage::Cs, read("tests/affine.dxil"));
    t.compile_wait = std::time::Duration::ZERO;
    let mut asks = 0;
    while !matches!(t.compute_pipe(&cs, "a"), Pipe::Compute(_)) {
        asks += 1;
        assert!(asks < 3000, "never made: {:?}", t.shaders);
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
    assert_eq!((t.shaders.put_off, t.shaders.pipelines, t.shaders.refusals()), (asks, 1, 0));
}

/// What the host cannot give a shader is refused, counted by reason, and logged once; nothing
/// fails.
#[test]
fn shaders_the_host_cannot_run_are_refused_and_counted() {
    let Some(mut t) = title() else { return };
    let dxil = read("tests/affine.dxil");
    let mut paged = host_shader(HwStage::Cs, dxil.clone());
    paged.memory_writes = true;
    let garbage = host_shader(HwStage::Cs, b"DXBC but not really".to_vec());
    for s in [&paged, &paged, &garbage] {
        // The same refusal twice is two refusals of one reason.
        assert!(matches!(t.compute_pipe(s, "x"), Pipe::Refused));
    }
    let ShaderStats { refused, pipelines, .. } = t.shaders.clone();
    assert_eq!(pipelines, 0);
    assert_eq!(refused.len(), 2, "{refused:?}");
    assert!(refused.iter().any(|(why, _)| why.contains("marks the pages")));
    assert!(refused.iter().any(|(why, _)| why.contains("thread group") || why.contains("conversion")));
    // Compute shaders get the page table; a graphics pipeline does not yet.
    let mut vs = host_shader(HwStage::Vs, read("tests/fixtures/tri_vs.dxil"));
    vs.memory = true;
    let g = Graphics::default();
    assert!(matches!(t.render_pipe(&vs, None, &g, "vs"), Pipe::Refused));
    assert!(t.shaders.refused.iter().any(|(why, _)| why.contains("graphics pipeline")));
    t.refused_translation("y", "no DXIL part".into());
    assert_eq!(t.shaders.refusals(), 5, "{:?}", t.shaders.refused);
}

/// A graphics pipeline from a vertex and pixel shader and a pipeline-state stream's state:
/// the input layout in the shader's signature order, formats, blend, depth, stencil and raster.
#[test]
fn render_pipelines_come_from_the_translated_shaders_and_the_streams_state() {
    let Some(mut t) = title() else { return };
    let vs = host_shader(HwStage::Vs, read("tests/fixtures/tri_vs.dxil"));
    let ps = host_shader(HwStage::Ps, read("tests/fixtures/tri_ps.dxil"));
    // The vertex shader reads POSITION, TEXCOORD, COLOR, in the signature's order.
    let reads = state::vertex_inputs(&vs.dxil).unwrap();
    assert_eq!(
        reads,
        [("POSITION".to_owned(), 0), ("TEXCOORD".to_owned(), 0), ("COLOR".to_owned(), 0)]
    );

    // A stream with the layout, a target and blending on.
    let names = [c"POSITION", c"TEXCOORD", c"COLOR"];
    let formats = [16u32, 16, 2];
    let mut elements = Vec::new();
    for (i, n) in names.iter().enumerate() {
        let mut e = [0u8; 32];
        e[..8].copy_from_slice(&(n.as_ptr() as u64).to_le_bytes());
        for (k, w) in [0, formats[i], 0, u32::MAX, 0, 0].into_iter().enumerate() {
            e[8 + 4 * k..12 + 4 * k].copy_from_slice(&w.to_le_bytes());
        }
        elements.extend(e);
    }
    let mut layout = 12u32.to_le_bytes().to_vec();
    layout.extend([0; 4]);
    layout.extend((elements.as_ptr() as u64).to_le_bytes());
    layout.extend(3u32.to_le_bytes());
    layout.extend([0; 4]);
    let mut rts = 15u32.to_le_bytes().to_vec();
    for f in [28u32, 0, 0, 0, 0, 0, 0, 0, 1] {
        rts.extend(f.to_le_bytes());
    }
    rts.extend([0; 4]);
    let mut g = Graphics::default();
    for (kind, s) in [(12u32, &layout), (15, &rts)] {
        // SAFETY: whole subobjects over live elements and names.
        unsafe { g.read(kind, s.as_ptr()) };
    }
    // With no time to wait it is put off while Metal compiles it, not refused: nothing is cached
    // and a later use picks it up.
    t.compile_wait = std::time::Duration::ZERO;
    let mut asks = 0;
    let pipe = loop {
        let pipe = t.render_pipe(&vs, Some(&ps), &g, "tri");
        if matches!(pipe, Pipe::Render(..)) {
            break pipe;
        }
        asks += 1;
        assert!(asks < 3000, "never made: {:?}", t.shaders);
        std::thread::sleep(std::time::Duration::from_millis(10));
    };
    assert!(asks > 0);
    assert_eq!((t.shaders.put_off, t.shaders.refusals()), (asks, 0));
    t.compile_wait = std::time::Duration::MAX;
    assert!(matches!(t.render_pipe(&vs, Some(&ps), &g, "tri again"), Pipe::Render(..)));
    assert_eq!((t.shaders.pipelines, t.shaders.cached), (1, 1));

    // A different state is a different pipeline; a layout the shader's input is missing from is
    // refused, not drawn wrong.
    let mut no_layout = Graphics::default();
    // SAFETY: as above, only the formats.
    unsafe { no_layout.read(15, rts.as_ptr()) };
    assert!(matches!(t.render_pipe(&vs, Some(&ps), &no_layout, "bare"), Pipe::Refused));
    // A pixel stage with arguments of its own is converted with the host's layout of them: this
    // fixture (plain HLSL with a texture at t0, not translator output) is then refused, as its
    // texture is in no host layout.
    let mut args = ps.clone();
    args.user_data.push(victory_shader::emit::UserData {
        index: 0,
        slot: 0,
        root: victory_shader::RootType::Constants,
        dword: 0,
    });
    assert!(matches!(t.render_pipe(&vs, Some(&args), &g, "args"), Pipe::Refused));
    assert!(t.shaders.refused.iter().any(|(why, _)| why.starts_with("args") && why.contains("root signature")));
    assert!(t.shaders.refused.iter().any(|(why, _)| why.contains("layout lacks")));
}

#[test]
fn a_2d_compute_sample_selects_the_mip_from_both_quad_gradients() {
    check_compute_gradient_mips([4, 4, 2], false);
    check_compute_gradient_mips([8, 4, 1], false);
    check_compute_gradient_mips([8, 8, 1], true);
}

fn check_compute_gradient_mips(threads: [u32; 3], wave64: bool) {
    use super::addr_tests::{Fixture, VA_A, halves};
    use crate::native::list::NativeList;
    use victory_backend::resources::{HeapKind, ResourceBackend};
    use victory_shader::native::{ComputeIo, NativeProgram, ResourceType, RootParam, RootType, StageIo, UsedResource};
    use victory_shader::{Container, io, ir, resource, structure};

    let Some(mut t) = title() else { return };
    let f = Fixture::new(&t);
    const OUTPUT: usize = 0x6_0000; // Fixture's CPU offset for VA_A + 0x4_0000.
    let vop1 = |op: u32, dst: u32, src: u32| 0x7e00_0000 | dst << 17 | op << 9 | src;
    let vop2 = |op: u32, dst: u32, src0: u32, src1: u32| op << 25 | dst << 17 | src1 << 9 | src0;
    let object = crate::resource::test_texture_at(
        VA_A,
        (64, 64),
        3,
        41,
        victory_formats::texture::Swizzle::LINEAR.0,
        0,
    );
    let key = object as usize;
    t.texture_created(key, crate::resource::texture_info(object).unwrap());
    let tex = t.texture_of(key).unwrap();
    let mip_values = [0.25f32, 0.5, 0.75];
    for (level, value) in mip_values.into_iter().enumerate() {
        let width = 64 >> level;
        t.host.backend.upload_texture(
            &tex,
            level,
            0,
            &value.to_le_bytes().repeat(width * width),
            width * 4,
            0,
        );
    }

    let code = [
        0xbe8a_0304, 0xbe8b_0380, // resource table at s[10:11]
        0xf408_0405, 0xfa00_0000, // output buffer descriptor at s[16:19]
        0xf40c_0605, 0xfa00_0020, // input texture descriptor at s[24:31]
        0xbe8c_0305, 0xbe8d_0380, // sampler table at s[12:13]
        0xf408_0806, 0xfa00_0000, // sampler at s[32:35]
        0xbf8c_c07f, // s_waitcnt lgkmcnt(0)
        vop2(0x1a, 8, 0x82, 2),  // v8 = z * 4 (z is zero for the 2D groups)
        vop2(0x25, 8, 0x101, 8), // v8 = z * 4 + y
        vop2(0x1a, 8, 0x80 + threads[0].ilog2(), 8),
        vop2(0x25, 8, 0x100, 8), // v8 = (z * 4 + y) * width + x
        vop2(0x1a, 8, 0x84, 8),  // byte offset for float4
        vop1(6, 9, 0x100),       // float(x)
        vop1(6, 10, 0x101),      // float(y)
        vop2(8, 9, 0, 9),        // x * s0
        vop2(8, 10, 1, 10),      // y * s1
        0xf080_0108, 0x0106_0409, // image_sample v4, v[9:10], s[24:31], s[32:35]
        0xbf8c_3f70, // s_waitcnt vmcnt(0)
        0xe070_1000, 0x8004_0408, // buffer_store_dword v4, v8, s[16:19], 0 offen
        0xbf81_0000,
    ];
    let param = |root, user_data, sgprs, slot| RootParam {
        root, table_offset_dwords: 0, sgprs, base_register: 0, user_data, slot,
    };
    let used = |kind, table_offset_dwords, dwords, user_data, slot| UsedResource {
        kind, root: RootType::Table, table_offset_dwords, lower_bound: 0, stride: 0,
        dwords, user_data, slot,
    };
    let native = NativeProgram {
        stage: HwStage::Cs,
        wave64,
        vgprs: 16,
        layout: vec![
            param(RootType::Constants, 0, 4, 0),
            param(RootType::Table, 4, 1, 1),
            param(RootType::Table, 5, 1, 2),
        ],
        used: vec![
            used(ResourceType::Uav, 0, 4, 4, 1),
            used(ResourceType::Srv, 8, 8, 4, 1),
            used(ResourceType::Sampler, 0, 4, 5, 2),
        ],
        io: StageIo::Cs(ComputeIo { rsrc2: 6 << 1 | 2 << 11, threads }),
        code: code.into_iter().flat_map(u32::to_le_bytes).collect(),
    };
    let ir = ir::build(&native.code, wave64, native.vgprs).unwrap();
    let structured = structure::structure(&ir).unwrap();
    let resources = resource::track(&ir, &native.layout, native.stage, Some(&structured)).unwrap();
    let io = io::raise(&ir, &native).unwrap();
    let container = Container { digest: [0; 16], version: (1, 0), parts: vec![] };
    let shader = victory_shader::emit::emit_with(
        &container, &native, &ir, &structured, &io, &resources, shaders::options(),
    ).unwrap();

    let slots = [0u32; 16];
    let handle = slots.as_ptr() as u64;
    let heap = t.host.heap(HeapKind::Resource, handle, 2).unwrap();
    let mut buffer = [0u32; 14];
    buffer[0] = 1;
    let mut uav = [0u32; 16];
    (uav[1], uav[4], uav[8]) = (1, 64 * 4, 1);
    [uav[10], uav[11]] = halves(VA_A + 0x4_0000);
    crate::descriptor::create_placed_uav(std::ptr::null(), &buffer, &uav, handle);
    let srv = [41, 4, 5768, 0, 0, 3, 0, 0, 0, 0];
    crate::descriptor::create_srv(std::ptr::null(), object, &srv, handle + 32);
    assert!(t.host.mirror(handle).unwrap());
    assert!(t.host.mirror(handle + 32).unwrap());

    let samplers = [0u32; 4];
    let sampler_handle = samplers.as_ptr() as u64;
    let sampler_heap = t.host.heap(HeapKind::Sampler, sampler_handle, 1).unwrap();
    let mut sampler = [0u32; 27];
    sampler[3] = 1; // point mip filter
    sampler[6..9].fill(3); // clamp xyz
    sampler[10] = 2.0f32.to_bits();
    crate::descriptor::create_sampler_x(std::ptr::null(), &sampler, sampler_handle);
    assert!(t.host.mirror(sampler_handle).unwrap());

    let Pipe::Compute(pso) = t.compute_pipe(&shader, "compute gradient mips") else {
        panic!("{:?}", t.shaders)
    };
    let lanes = threads.into_iter().product::<u32>() as usize;
    for (x, y, mip) in [(1.0f32, 1.0f32, 0), (1.0, 2.0, 1), (4.0, 1.0, 2), (1.0, 4.0, 2)] {
        f.put(OUTPUT, &[0u32; 64 * 4]);
        let mut list = NativeList::default();
        list.set_heaps(Some(&heap), Some(&sampler_heap));
        list.set_compute_pipeline(&pso);
        list.set_compute_root_constants(0, &[x / 64.0, y / 64.0, 0.0, 0.0].map(f32::to_bits));
        list.set_compute_root_table(1, &heap, 0);
        list.set_compute_root_table(2, &sampler_heap, 0);
        list.dispatch(1, 1, 1).unwrap();
        super::run(&t.host, &[&list]);
        let want = mip_values[mip];
        for (lane, bits) in f.get(OUTPUT, lanes * 4).into_iter().step_by(4).enumerate() {
            assert_eq!(f32::from_bits(bits), want, "lane {lane}, scales {x}, {y}");
        }
    }
}
