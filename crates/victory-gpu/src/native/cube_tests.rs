//! End-to-end Metal GPU verification for `D3D12_SRV_DIMENSION_TEXTURECUBE` (`ViewDim::Cube`) over
//! a 6-slice `TexDim::D2Array` backing texture paired with a 13-dword `D3D12_SAMPLER_DESC`
//! (`View::Sampler`).

use super::addr_tests::{Fixture, VA_A, halves, title};
use crate::native::list::NativeList;
use crate::native::shaders;
use crate::native::title::Pipe;
use victory_backend::resources::{Format, HeapKind, ResourceBackend, TexDim, TextureDesc};
use victory_shader::emit::emit_with;
use victory_shader::native::{
    ComputeIo, HwStage, NativeProgram, ResourceType, RootParam, RootType, StageIo, UsedResource,
};
use victory_shader::{Container, io, ir, resource, structure};

/// Six compute lanes (`v0 = 0..5`) sample the centre `(s = 1.5, t = 1.5)` of cube face `v0`
/// through `s[8:15]` (table slot 0 in `s2`) and sampler `s[16:19]` (table slot 0 in `s3`),
/// storing the RGBA floats to `s[0:1] + 16 * v0`.
const SAMPLE_CUBE_FACES: &[u32] = &[
    0xBE84_0302, // s_mov_b32 s4, s2
    0xBE85_0380, // s_mov_b32 s5, 0
    0xBE86_0303, // s_mov_b32 s6, s3
    0xBE87_0380, // s_mov_b32 s7, 0
    0xF40C_0202,
    0xFA00_0000, // s_load_dwordx8 s[8:15], s[4:5], 0x0
    0xF408_0403,
    0xFA00_0000, // s_load_dwordx4 s[16:19], s[6:7], 0x0
    0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
    0x7E02_02FF,
    0x3FC0_0000, // v_mov_b32 v1, 1.5
    0x7E04_02FF,
    0x3FC0_0000, // v_mov_b32 v2, 1.5
    0x7E06_0D00, // v_cvt_f32_u32 v3, v0
    0xF09C_0F18,
    0x0082_0401, // image_sample_lz v[4:7], v[1:3], s[8:15], s[16:19] dmask:0xf dim:CUBE
    0xBF8C_3F70, // s_waitcnt vmcnt(0)
    0x3410_0084, // v_lshlrev_b32 v8, 4, v0
    0xDC70_8000,
    0x0000_0408, // global_store_dword v8, v4, s[0:1]
    0xDC70_8004,
    0x0000_0508, // global_store_dword v8, v5, s[0:1] offset:4
    0xDC70_8008,
    0x0000_0608, // global_store_dword v8, v6, s[0:1] offset:8
    0xDC70_800C,
    0x0000_0708, // global_store_dword v8, v7, s[0:1] offset:12
    0xBF81_0000, // s_endpgm
];

#[test]
fn cube_srv_and_d3d12_sampler_sample_all_six_faces_on_metal() {
    let Some(mut t) = title() else { return };
    let f = Fixture::new(&t);

    let mut desc = TextureDesc::d2(Format::Rgba8Unorm, 4, 4);
    desc.dim = TexDim::D2Array;
    desc.depth_or_layers = 6;
    let cube = t.host.texture(&desc).expect("6-slice cubemap backing texture");
    let faces: [[u8; 4]; 6] = [
        [255, 0, 0, 255],
        [0, 255, 0, 255],
        [0, 0, 255, 255],
        [255, 255, 0, 255],
        [255, 0, 255, 255],
        [0, 255, 255, 255],
    ];
    for (slice, rgba) in faces.iter().enumerate() {
        t.host
            .backend
            .upload_texture(&cube, 0, slice, &rgba.repeat(16), 16, 0);
    }
    let key = std::rc::Rc::as_ptr(&cube) as usize;
    t.host.register_texture(key, &cube);

    let vslots = vec![0u32; 8 * 4];
    let vbase = vslots.as_ptr() as u64;
    let vheap = t.host.heap(HeapKind::Resource, vbase, 4).unwrap();
    let mut srv = [0u32; 10];
    (srv[0], srv[1], srv[2], srv[5]) = (28, 9, 5768, 1); // RGBA8_UNORM, TEXTURECUBE, 1 mip
    crate::descriptor::create_srv(std::ptr::null_mut(), key as _, &srv, vbase);
    assert!(t.host.mirror(vbase).unwrap());

    let sslots = vec![0u32; 8 * 4];
    let sbase = sslots.as_ptr() as u64;
    let sheap = t.host.heap(HeapKind::Sampler, sbase, 4).unwrap();
    let smp = [0u32, 3, 3, 3, 0, 1, 0, 0, 0, 0, 0, 0, 0x7F7F_FFFF]; // MIN_MAG_MIP_POINT, CLAMP
    crate::descriptor::create_sampler(std::ptr::null_mut(), &smp, sbase);
    assert!(t.host.mirror(sbase).unwrap());

    let native = NativeProgram {
        stage: HwStage::Cs,
        wave64: false,
        vgprs: 16,
        layout: vec![
            RootParam {
                root: RootType::Constants,
                table_offset_dwords: 0,
                sgprs: 2,
                base_register: 0,
                user_data: 0,
                slot: 0,
            },
            RootParam {
                root: RootType::Table,
                table_offset_dwords: 0,
                sgprs: 1,
                base_register: 0,
                user_data: 2,
                slot: 1,
            },
            RootParam {
                root: RootType::Table,
                table_offset_dwords: 0,
                sgprs: 1,
                base_register: 0,
                user_data: 3,
                slot: 2,
            },
        ],
        used: vec![
            UsedResource {
                kind: ResourceType::Srv,
                root: RootType::Table,
                table_offset_dwords: 0,
                lower_bound: 0,
                stride: 0,
                dwords: 8,
                user_data: 2,
                slot: 1,
            },
            UsedResource {
                kind: ResourceType::Sampler,
                root: RootType::Table,
                table_offset_dwords: 0,
                lower_bound: 0,
                stride: 0,
                dwords: 4,
                user_data: 3,
                slot: 2,
            },
        ],
        io: StageIo::Cs(ComputeIo {
            rsrc2: 4 << 1,
            threads: [6, 1, 1],
        }),
        code: SAMPLE_CUBE_FACES.iter().flat_map(|w| w.to_le_bytes()).collect(),
    };
    let mut p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
    let opts = shaders::options();
    p.thread_counts = opts.thread_counts;
    let st = structure::structure(&p).expect("structure");
    let res = resource::track(&p, &native.layout, native.stage, Some(&st)).expect("track");
    let io = io::raise(&p, &native).expect("io::raise");
    let mut empty = b"DXBC".to_vec();
    empty.extend([0; 16]);
    empty.extend([1, 0, 0, 0]);
    empty.extend(32u32.to_le_bytes());
    empty.extend(0u32.to_le_bytes());
    let c = Container::parse(&empty).expect("container");
    let cs = emit_with(&c, &native, &p, &st, &io, &res, opts).expect("emit");
    let Pipe::Compute(pso) = t.compute_pipe(&cs, "cube") else {
        panic!("cube pipeline not made: {:?}", t.shaders);
    };

    let dst_va = VA_A + 0x4_0000;
    let mut l = NativeList::default();
    l.set_heaps(Some(&vheap), Some(&sheap));
    l.set_compute_pipeline(&pso);
    l.set_compute_root_constants(0, &halves(dst_va));
    l.set_compute_root_table(1, &vheap, 0);
    l.set_compute_root_table(2, &sheap, 0);
    l.dispatch(1, 1, 1).unwrap();
    super::run(&t.host, &[&l]);

    let words = f.get(0x6_0000, 24);
    for (face, rgba) in faces.iter().enumerate() {
        let got: [f32; 4] = std::array::from_fn(|c| f32::from_bits(words[face * 4 + c]));
        let want = rgba.map(|b| f32::from(b) / 255.0);
        assert_eq!(got, want, "cube face {face}");
    }
    assert_eq!(t.shaders.refusals(), 0, "{:?}", t.shaders.refused);
    crate::descriptor::forget(key);
}

/// Two compute lanes (`v0 = 0..1`) run `image_sample_c_lz` (`v1` = reference `0.5`,
/// `v2 = (2 * v0 + 1) * 0.25`, `v3 = 0.5`) against a `2x1` `D32Float` shadow map (`[0.75, 0.25]`).
const SAMPLE_SHADOW_CMP: &[u32] = &[
    0xBE84_0302, // s_mov_b32 s4, s2
    0xBE85_0380, // s_mov_b32 s5, 0
    0xBE86_0303, // s_mov_b32 s6, s3
    0xBE87_0380, // s_mov_b32 s7, 0
    0xF40C_0202,
    0xFA00_0000, // s_load_dwordx8 s[8:15], s[4:5], 0x0
    0xF408_0403,
    0xFA00_0000, // s_load_dwordx4 s[16:19], s[6:7], 0x0
    0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
    0x7E02_02F0, // v_mov_b32 v1, 0.5 (z-compare reference)
    0x3404_0081, // v_lshlrev_b32 v2, 1, v0
    0x4A04_0481, // v_add_nc_u32 v2, 1, v2
    0x7E04_0D02, // v_cvt_f32_u32 v2, v2
    0x1004_04FF,
    0x3E80_0000, // v_mul_f32 v2, 0.25, v2
    0x7E06_02F0, // v_mov_b32 v3, 0.5
    0xF0BC_0108,
    0x0082_0401, // image_sample_c_lz v4, v[1:3], s[8:15], s[16:19] dmask:0x1 dim:2D
    0xBF8C_3F70, // s_waitcnt vmcnt(0)
    0x3410_0082, // v_lshlrev_b32 v8, 2, v0
    0xDC70_8000,
    0x0000_0408, // global_store_dword v8, v4, s[0:1]
    0xBF81_0000, // s_endpgm
];

#[test]
fn shadow_comparison_samplers_evaluate_depth_comparisons_on_metal() {
    let Some(mut t) = title() else { return };
    let f = Fixture::new(&t);

    let mut desc = TextureDesc::d2(Format::D32Float, 2, 1);
    desc.render_target = true;
    let shadow = t.host.texture(&desc).expect("2x1 D32Float shadow map");
    let depths: [u8; 8] = [0.75f32.to_le_bytes(), 0.25f32.to_le_bytes()].concat().try_into().unwrap();
    t.host.backend.upload_texture(&shadow, 0, 0, &depths, 8, 0);
    let key = std::rc::Rc::as_ptr(&shadow) as usize;
    t.host.register_texture(key, &shadow);

    let vslots = vec![0u32; 8 * 2];
    let vbase = vslots.as_ptr() as u64;
    let vheap = t.host.heap(HeapKind::Resource, vbase, 2).unwrap();
    let mut srv = [0u32; 10];
    (srv[0], srv[1], srv[2], srv[5]) = (41, 4, 5768, 1); // R32_FLOAT over D32_FLOAT, TEXTURE2D
    crate::descriptor::create_srv(std::ptr::null_mut(), key as _, &srv, vbase);
    assert!(t.host.mirror(vbase).unwrap());

    let sslots = vec![0u32; 8 * 2];
    let sbase = sslots.as_ptr() as u64;
    let sheap = t.host.heap(HeapKind::Sampler, sbase, 2).unwrap();
    // Slot 0: console shadow sampler (`CreateSamplerX`, LESS_EQUAL = 4).
    let mut console_smp = [0u32; 27];
    (console_smp[6], console_smp[7], console_smp[8], console_smp[16]) = (3, 3, 3, 4);
    crate::descriptor::create_sampler_x(std::ptr::null_mut(), &console_smp, sbase);
    // Slot 1: D3D12 comparison sampler (`CreateSampler`, COMPARISON_MIN_MAG_MIP_POINT = 0x80, LESS_EQUAL = 4).
    let d3d_smp = [0x80u32, 3, 3, 3, 0, 1, 4, 0, 0, 0, 0, 0, 0x7F7F_FFFF];
    crate::descriptor::create_sampler(std::ptr::null_mut(), &d3d_smp, sbase + 16);
    assert!(t.host.mirror(sbase).unwrap() && t.host.mirror(sbase + 16).unwrap());

    let native = NativeProgram {
        stage: HwStage::Cs,
        wave64: false,
        vgprs: 16,
        layout: vec![
            RootParam {
                root: RootType::Constants,
                table_offset_dwords: 0,
                sgprs: 2,
                base_register: 0,
                user_data: 0,
                slot: 0,
            },
            RootParam {
                root: RootType::Table,
                table_offset_dwords: 0,
                sgprs: 1,
                base_register: 0,
                user_data: 2,
                slot: 1,
            },
            RootParam {
                root: RootType::Table,
                table_offset_dwords: 0,
                sgprs: 1,
                base_register: 0,
                user_data: 3,
                slot: 2,
            },
        ],
        used: vec![
            UsedResource {
                kind: ResourceType::Srv,
                root: RootType::Table,
                table_offset_dwords: 0,
                lower_bound: 0,
                stride: 0,
                dwords: 8,
                user_data: 2,
                slot: 1,
            },
            UsedResource {
                kind: ResourceType::Sampler,
                root: RootType::Table,
                table_offset_dwords: 0,
                lower_bound: 0,
                stride: 0,
                dwords: 4,
                user_data: 3,
                slot: 2,
            },
        ],
        io: StageIo::Cs(ComputeIo {
            rsrc2: 4 << 1,
            threads: [2, 1, 1],
        }),
        code: SAMPLE_SHADOW_CMP.iter().flat_map(|w| w.to_le_bytes()).collect(),
    };
    let mut p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
    let opts = shaders::options();
    p.thread_counts = opts.thread_counts;
    let st = structure::structure(&p).expect("structure");
    let res = resource::track(&p, &native.layout, native.stage, Some(&st)).expect("track");
    let io = io::raise(&p, &native).expect("io::raise");
    let mut empty = b"DXBC".to_vec();
    empty.extend([0; 16]);
    empty.extend([1, 0, 0, 0]);
    empty.extend(32u32.to_le_bytes());
    empty.extend(0u32.to_le_bytes());
    let c = Container::parse(&empty).expect("container");
    let cs = emit_with(&c, &native, &p, &st, &io, &res, opts).expect("emit");
    let Pipe::Compute(pso) = t.compute_pipe(&cs, "shadow_cmp") else {
        panic!("shadow_cmp pipeline not made: {:?}", t.shaders);
    };

    for (slot, label) in [(0usize, "ConsoleSampler"), (1usize, "D3D12 Sampler")] {
        let dst_va = VA_A + 0x4_1000 + (slot as u64) * 64;
        let mut l = NativeList::default();
        l.set_heaps(Some(&vheap), Some(&sheap));
        l.set_compute_pipeline(&pso);
        l.set_compute_root_constants(0, &halves(dst_va));
        l.set_compute_root_table(1, &vheap, 0);
        l.set_compute_root_table(2, &sheap, slot);
        l.dispatch(1, 1, 1).unwrap();
        super::run(&t.host, &[&l]);

        let words = f.get(0x6_1000 + slot * 64, 2);
        let got = [f32::from_bits(words[0]), f32::from_bits(words[1])];
        assert_eq!(got, [1.0, 0.0], "{label} shadow comparison failed");
    }
    crate::descriptor::forget(key);
}
