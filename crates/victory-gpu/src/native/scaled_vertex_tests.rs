//! Synthetic GFX10 vertex fetches through translation, guessed input layouts and Metal readback.
use super::{Pool, host};
use crate::native::{
    list::NativeList,
    shaders, state,
    title::{Pipe, Title},
};
use victory_backend::backend::ComputeBackend;
use victory_backend::resources::{Format, ResourceBackend, TextureDesc, Topology};
use victory_shader::{emit::graphics, native::HwStage};

fn translate(stage: HwStage, words: &[u32], rect: bool) -> victory_shader::emit::HostShader {
    let bytes: Vec<_> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    let container = graphics::container(stage, &bytes, &[0; 0x10]).unwrap();
    shaders::translate_extra(&container, rect, 0, [0; 32]).unwrap()
}

// Assembled with llvm-mc -triple=amdgcn -mcpu=gfx1030 -show-encoding.
fn vertex(format: u32, count: u32, rect: bool) -> victory_shader::emit::HostShader {
    translate(
        HwStage::Vs,
        &[
            0xF408_0100,
            0xFA00_0000, // s_load_dwordx4 s[4:7], s[0:1], 0
            0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
            0xEA6B_2000,
            0x8001_0100, // tbuffer_load_format_xyzw v[1:4], ... FLOAT
            0x7E0A_02FF,
            0x4188_0000, // v5 = 17.0 (unloaded channels stay unchanged)
            0x7E0C_02FF,
            0x4188_0000,
            0x7E0E_02FF,
            0x4188_0000,
            0x7E10_02FF,
            0x4188_0000,
            0xE800_2010 | format << 19 | (count - 1) << 16,
            0x8001_0500,
            0xBF8C_3F70, // s_waitcnt vmcnt(0)
            0xF800_08CF,
            0x0403_0201, // exp pos0 v1, v2, v3, v4 done
            0xF800_020F,
            0x0807_0605, // exp param0 v5, v6, v7, v8
            0xBF81_0000,
        ],
        rect,
    )
}

#[test]
fn scaled_vertex_loads_reach_float_render_targets() {
    let Some(h) = host() else { return };
    let pool = Pool::new(&h, 1);
    let memory = h.memory();
    let mut t = Title::new(h.backend, memory);
    let ps = translate(
        HwStage::Ps,
        &[
            0xBEFC_0300, // s_mov_b32 m0, s0
            0xC806_0002,
            0xC80A_0102,
            0xC80E_0202,
            0xC812_0302, // flat attr0.xyzw
            0xF800_180F,
            0x0403_0201, // exp mrt0 v1, v2, v3, v4 done vm
            0xBF81_0000,
        ],
        false,
    );
    let mut desc = TextureDesc::d2(Format::Rgba32Float, 4, 4);
    desc.render_target = true;
    let color = t.host.texture(&desc).unwrap();
    let queue = t.host.backend.create_queue().unwrap();
    let fence = t.host.backend.create_fence(0).unwrap();
    let mut serial = 0;
    for (format, channels) in [(3, 1), (4, 1), (16, 2), (17, 2), (58, 4), (59, 4)] {
        for count in 1..=channels {
            let vs = vertex(format, count, false);
            let reads = state::vertex_inputs(&vs.dxil).unwrap();
            let g = state::Graphics::guess_base().for_draw(
                vec![Format::Rgba32Float],
                None,
                vec![20],
                &reads,
                false,
            );
            let Pipe::Render(pso, binding) = t.render_pipe(&vs, Some(&ps), &g, "scaled vertex")
            else {
                panic!("{:?}", t.shaders);
            };
            // Each channel visits all byte values, with distinct channel values to catch swaps.
            for value in 0..=255u8 {
                let data = [
                    value,
                    value.wrapping_add(67),
                    value.wrapping_add(131),
                    value.wrapping_add(193),
                ];
                for (i, xy) in [[-1.0f32, -1.0], [3.0, -1.0], [-1.0, 3.0]]
                    .iter()
                    .enumerate()
                {
                    pool.write(i * 20, &[xy[0], xy[1], 0.0, 1.0]);
                    pool.write(i * 20 + 16, &data);
                }
                let mut list = NativeList::default();
                list.set_render_targets(std::slice::from_ref(&color), None);
                list.clear_render_target(&color, [-999.0; 4]);
                list.set_graphics_pipeline(&pso, Topology::Triangles);
                list.set_graphics_binding(binding.clone());
                list.set_vertex_buffer(0, pool.va(0));
                list.draw_instanced(3, 1, 0, 0).unwrap();
                serial += 1;
                t.host
                    .execute(&queue, &[&list], &[], &[(&fence, serial)])
                    .unwrap();
                assert!(
                    t.host.backend.wait(&fence, serial, 10_000),
                    "GPU did not finish"
                );
                let mut pixels = [0u8; 4 * 4 * 16];
                t.host
                    .backend
                    .read_texture(&color, 0, 0, &mut pixels, 4 * 16, 0);
                for (channel, &byte) in data.iter().enumerate() {
                    let expected = if channel >= count as usize {
                        17.0f32
                    } else if matches!(format, 3 | 16 | 58) {
                        f32::from(byte)
                    } else {
                        f32::from(byte as i8)
                    };
                    let at = (2 * 4 + 2) * 16 + channel * 4;
                    let actual = u32::from_le_bytes(pixels[at..at + 4].try_into().unwrap());
                    assert_eq!(
                        actual,
                        expected.to_bits(),
                        "format={format} count={count} value={value} channel={channel}"
                    );
                }
            }
        }
    }
    eprintln!("verified 3584 Metal draws, all 256 byte values in each loaded channel");
}

#[test]
fn scaled_vertex_rectangles_match_explicit_float_corners() {
    let Some(h) = host() else { return };
    let pool = Pool::new(&h, 1);
    let memory = h.memory();
    let mut t = Title::new(h.backend, memory);
    let ps = translate(
        HwStage::Ps,
        &[
            0xBEFC_0300,
            0xC808_0000,
            0xC809_0001, // v_interp_p1/p2_f32 v2, v0/v1, attr0.x
            0xC80C_0100,
            0xC80D_0101,
            0xC810_0200,
            0xC811_0201,
            0xC814_0300,
            0xC815_0301,
            0xF800_180F,
            0x0504_0302,
            0xBF81_0000,
        ],
        false,
    );
    let mut desc = TextureDesc::d2(Format::Rgba32Float, 4, 4);
    desc.render_target = true;
    let color = t.host.texture(&desc).unwrap();
    for format in [58, 59] {
        let bytes = [[136u8, 156, 176, 196], [216, 236, 0, 20], [40, 60, 80, 100]];
        let values = bytes.map(|v| {
            v.map(|b| {
                if format == 58 {
                    f32::from(b)
                } else {
                    f32::from(b as i8)
                }
            })
        });
        let fourth: [f32; 4] = std::array::from_fn(|k| values[1][k] + values[2][k] - values[0][k]);
        // Three source corners: right angle, bottom right, top left.
        for (i, xy) in [[-1.0f32, -1.0], [1.0, -1.0], [-1.0, 1.0]]
            .iter()
            .enumerate()
        {
            pool.write(i * 20, &[xy[0], xy[1], 0.0, 1.0]);
            pool.write(i * 20 + 16, &bytes[i]);
        }
        // Independent reference: explicit four-corner strip, with already-converted floats.
        for (i, (xy, data)) in [
            ([1.0f32, -1.0], values[1]),
            ([-1.0, -1.0], values[0]),
            ([1.0, 1.0], fourth),
            ([-1.0, 1.0], values[2]),
        ]
        .iter()
        .enumerate()
        {
            pool.write(1024 + i * 32, &[xy[0], xy[1], 0.0, 1.0]);
            pool.write(1024 + i * 32 + 16, data);
        }
        let mut results = Vec::new();
        for rect in [false, true] {
            let vs = vertex(if rect { format } else { 77 }, 4, rect);
            let reads = state::vertex_inputs(&vs.dxil).unwrap();
            let g = state::Graphics::guess_base().for_draw(
                vec![Format::Rgba32Float],
                None,
                vec![if rect { 20 } else { 32 }],
                &reads,
                rect,
            );
            let Pipe::Render(pso, binding) = t.render_pipe(&vs, Some(&ps), &g, "scaled rectangle")
            else {
                panic!("{:?}", t.shaders);
            };
            let mut list = NativeList::default();
            list.set_render_targets(std::slice::from_ref(&color), None);
            list.clear_render_target(&color, [-999.0; 4]);
            list.set_graphics_pipeline(&pso, Topology::TriangleStrip);
            list.set_graphics_binding(binding);
            list.set_vertex_buffer(0, pool.va(if rect { 0 } else { 1024 }));
            list.draw_instanced(4, 1, 0, 0).unwrap();
            super::run(&t.host, &[&list]);
            let mut pixels = [0u8; 4 * 4 * 16];
            t.host
                .backend
                .read_texture(&color, 0, 0, &mut pixels, 4 * 16, 0);
            assert!(
                pixels
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .all(|v| *v != (-999.0f32).to_le_bytes()),
                "uncovered pixel"
            );
            results.push(pixels);
        }
        assert_eq!(
            results[0], results[1],
            "format={format}: rectangle differs from explicit quad"
        );
    }
    eprintln!("verified signed and unsigned rectangle pixels against explicit float quads");
}

#[test]
fn pixel_waterfall_and_exec_skip_mask_joins_run_on_the_gpu() {
    let Some(h) = host() else { return };
    let pool = Pool::new(&h, 1);
    let memory = h.memory();
    let mut t = Title::new(h.backend, memory);
    for (i, xy) in [[-1.0f32, -1.0], [3.0, -1.0], [-1.0, 3.0]]
        .iter()
        .enumerate()
    {
        pool.write(i * 16, &[xy[0], xy[1], 0.0, 1.0]);
    }
    let vs = translate(
        HwStage::Vs,
        &[
            0xF408_0100, 0xFA00_0000, // s_load_dwordx4 s[4:7], s[0:1], 0
            0xBF8C_C07F,              // s_waitcnt lgkmcnt(0)
            0xEA6B_2000, 0x8001_0100, // tbuffer_load_format_xyzw v[1:4], v0, s[4:7], 0 idxen format:77
            0xF800_08CF, 0x0403_0201, // exp pos0 v1, v2, v3, v4 done
            0xBF81_0000,              // s_endpgm
        ],
        false,
    );
    let ps32 = translate(
        HwStage::Ps,
        &[
            0x7E04_0F00,              // v_cvt_u32_f32 v2, v0 (x)
            0x7E06_0F01,              // v_cvt_u32_f32 v3, v1 (y)
            0x3406_0682,              // v_lshlrev_b32 v3, 2, v3
            0x4A04_0702,              // v_add_nc_u32 v2, v2, v3 (y * 4 + x)
            // Channel R: wave32 waterfall loop (`s_ff1_i32_b32` + `v_readlane_b32` + `s_bitset0_b32`).
            0x7E08_0280,              // v_mov_b32 v4, 0
            0x8704_C17E,              // s_and_b32 s4, exec_lo, -1
            0xBF84_000B,              // s_cbranch_scc0 after_loop
            // loop:
            0xBE85_1304,              // s_ff1_i32_b32 s5, s4
            0xD760_0006, 0x0000_0B02, // v_readlane_b32 s6, v2, s5
            0xD482_0007, 0x0002_0406, // v_cmp_eq_i32 s7, s6, v2
            0x7E0A_0C06,              // v_cvt_f32_u32 v5, s6
            0xD501_0004, 0x001E_0B04, // v_cndmask_b32 v4, v4, v5, s7
            0xBE84_1B05,              // s_bitset0_b32 s4, s5
            0xBF07_8004,              // s_cmp_lg_u32 s4, 0
            0xBF85_FFF5,              // s_cbranch_scc1 loop
            // after_loop — Channel G: 1-bit `s_bfe_i32` uniform joins (bit 0 = 1, bit 1 = 0 of 5).
            0x9408_FF85, 0x0001_0000, // s_bfe_i32 s8, 5, 0x00010000
            0xBF06_8008,              // s_cmp_eq_u32 s8, 0
            0xBF85_0001,              // s_cbranch_scc1 bfe_else
            0xBF82_0001,              // s_branch bfe_join
            0xBE88_0380,              // bfe_else: s_mov_b32 s8, 0
            // bfe_join:
            0x7E0A_02FF, 0x4120_0000, // v_mov_b32 v5, 10.0
            0x7E10_02FF, 0x41C8_0000, // v_mov_b32 v8, 25.0
            0xD501_0005, 0x0022_1105, // v_cndmask_b32 v5, v5, v8, s8
            0x9409_FF85, 0x0001_0001, // s_bfe_i32 s9, 5, 0x00010001
            0xBF06_8009,              // s_cmp_eq_u32 s9, 0
            0xBF84_0001,              // s_cbranch_scc0 bfe2_then
            0xBF82_0001,              // s_branch bfe2_join
            0xBE89_03C1,              // bfe2_then: s_mov_b32 s9, -1
            // bfe2_join:
            0xD501_0005, 0x0026_0905, // v_cndmask_b32 v5, v5, v4, s9
            // Channels B & A: `s_cbranch_execz` skip joins read by `v_cndmask_b32` and `s_and_saveexec_b32`.
            0x7E0C_02FF, 0x42C8_0000, // v_mov_b32 v6, 100.0
            0x7E0E_02FF, 0x4348_0000, // v_mov_b32 v7, 200.0
            0x7E10_02FF, 0x4396_0000, // v_mov_b32 v8, 300.0
            0x7E12_02FF, 0x43C8_0000, // v_mov_b32 v9, 400.0
            0x7E14_0502,              // v_readfirstlane_b32 s10, v2
            0x7E16_0502,              // v_readfirstlane_b32 s11, v2
            0xD404_0001, 0x0001_E900, // v_cmp_gt_f32 s1, v0, 2.0
            0xD401_0003, 0x0001_E901, // v_cmp_lt_f32 s3, v1, 2.0
            0xBE82_3C01,              // s_and_saveexec_b32 s2, s1
            0xBF88_0002,              // s_cbranch_execz join_ba
            0x880A_8003,              // s_or_b32 s10, s3, 0
            0x880B_8003,              // s_or_b32 s11, s3, 0
            // join_ba:
            0xD501_0006, 0x002A_1106, // v_cndmask_b32 v6, v6, v8, s10
            0xBE8C_3C0B,              // s_and_saveexec_b32 s12, s11
            0x7E0E_0309,              // v_mov_b32 v7, v9
            0xBEFE_0302,              // s_mov_b32 exec_lo, s2
            0xF800_180F, 0x0706_0504, // exp mrt0 v4, v5, v6, v7 done vm
            0xBF81_0000,              // s_endpgm
        ],
        false,
    );
    let ps64 = translate(
        HwStage::Ps,
        &[
            0x7E04_0F00,              // v_cvt_u32_f32 v2, v0 (x)
            0x7E06_0F01,              // v_cvt_u32_f32 v3, v1 (y)
            0x3406_0682,              // v_lshlrev_b32 v3, 2, v3
            0x4A04_0702,              // v_add_nc_u32 v2, v2, v3 (y * 4 + x)
            // Channel R: wave64 waterfall loop (`s_ff1_i32_b64` + `v_readlane_b32` + `s_bitset0_b64`).
            0x7E08_0280,              // v_mov_b32 v4, 0
            0x8784_C17E,              // s_and_b64 s[4:5], exec, -1
            0xBF84_000B,              // s_cbranch_scc0 after_loop
            // loop:
            0xBE86_1404,              // s_ff1_i32_b64 s6, s[4:5]
            0xD760_0007, 0x0000_0D02, // v_readlane_b32 s7, v2, s6
            0xD482_0008, 0x0002_0407, // v_cmp_eq_i32 s[8:9], s7, v2
            0x7E0A_0C07,              // v_cvt_f32_u32 v5, s7
            0xD501_0004, 0x0022_0B04, // v_cndmask_b32 v4, v4, v5, s[8:9]
            0xBE84_1C06,              // s_bitset0_b64 s[4:5], s6
            0xBF13_8004,              // s_cmp_lg_u64 s[4:5], 0
            0xBF85_FFF5,              // s_cbranch_scc1 loop
            // after_loop — Channel G: 1-bit `s_bfe_i64` (bit 35 = 1, bit 34 = 0) + `s_bitreplicate_b64_b32`.
            0xBE8A_0380,              // s_mov_b32 s10, 0
            0xBE8B_0388,              // s_mov_b32 s11, 8 (1 << 35 in s[10:11])
            0x950C_FF0A, 0x0001_0023, // s_bfe_i64 s[12:13], s[10:11], 0x00010023 (bit 35)
            0xBF12_800C,              // s_cmp_eq_u64 s[12:13], 0
            0xBF85_0001,              // s_cbranch_scc1 bfe64_else
            0xBF82_0001,              // s_branch bfe64_join
            0xBE8C_0480,              // bfe64_else: s_mov_b64 s[12:13], 0
            // bfe64_join:
            0x7E0A_02FF, 0x4120_0000, // v_mov_b32 v5, 10.0
            0x7E10_02FF, 0x420C_0000, // v_mov_b32 v8, 35.0
            0xD501_0005, 0x0032_1105, // v_cndmask_b32 v5, v5, v8, s[12:13]
            0x950E_FF0A, 0x0001_0022, // s_bfe_i64 s[14:15], s[10:11], 0x00010022 (bit 34)
            0xBF12_800E,              // s_cmp_eq_u64 s[14:15], 0
            0xBF84_0001,              // s_cbranch_scc0 bfe64_2_then
            0xBF82_0002,              // s_branch bfe64_2_join
            0xBE8E_3BFF, 0xFFFF_FFFF, // bfe64_2_then: s_bitreplicate_b64_b32 s[14:15], -1
            // bfe64_2_join:
            0xD501_0005, 0x003A_0905, // v_cndmask_b32 v5, v5, v4, s[14:15]
            // Channels B & A: `s_cbranch_execz` skip joins read by `v_cndmask_b32` and `s_and_saveexec_b64`.
            0x7E0C_02FF, 0x42C8_0000, // v_mov_b32 v6, 100.0
            0x7E0E_02FF, 0x4348_0000, // v_mov_b32 v7, 200.0
            0x7E10_02FF, 0x4396_0000, // v_mov_b32 v8, 300.0
            0x7E12_02FF, 0x43C8_0000, // v_mov_b32 v9, 400.0
            0x7E20_0502,              // v_readfirstlane_b32 s16, v2
            0x7E24_0502,              // v_readfirstlane_b32 s18, v2
            0xD404_0002, 0x0001_E900, // v_cmp_gt_f32 s[2:3], v0, 2.0
            0xD401_0004, 0x0001_E901, // v_cmp_lt_f32 s[4:5], v1, 2.0
            0xBE86_2402,              // s_and_saveexec_b64 s[6:7], s[2:3]
            0xBF88_0002,              // s_cbranch_execz join_ba
            0x8890_8004,              // s_or_b64 s[16:17], s[4:5], 0
            0x8892_8004,              // s_or_b64 s[18:19], s[4:5], 0
            // join_ba:
            0xD501_0006, 0x0042_1106, // v_cndmask_b32 v6, v6, v8, s[16:17]
            0xBE94_2412,              // s_and_saveexec_b64 s[20:21], s[18:19]
            0x7E0E_0309,              // v_mov_b32 v7, v9
            0xBEFE_0406,              // s_mov_b64 exec, s[6:7]
            0xF800_180F, 0x0706_0504, // exp mrt0 v4, v5, v6, v7 done vm
            0xBF81_0000,              // s_endpgm
        ],
        false,
    );

    let reads = state::vertex_inputs(&vs.dxil).unwrap();
    let g = state::Graphics::guess_base().for_draw(
        vec![Format::Rgba32Float],
        None,
        vec![16],
        &reads,
        false,
    );
    let mut desc = TextureDesc::d2(Format::Rgba32Float, 4, 4);
    desc.render_target = true;
    let color = t.host.texture(&desc).unwrap();

    for (label, ps, expected_g) in [("wave32", &ps32, 25.0f32), ("wave64", &ps64, 35.0f32)] {
        let Pipe::Render(pso, binding) = t.render_pipe(&vs, Some(ps), &g, label) else {
            panic!("{label}: {:?}", t.shaders);
        };
        let mut list = NativeList::default();
        list.set_render_targets(std::slice::from_ref(&color), None);
        list.clear_render_target(&color, [-1.0; 4]);
        list.set_graphics_pipeline(&pso, Topology::Triangles);
        list.set_graphics_binding(binding);
        list.set_vertex_buffer(0, pool.va(0));
        list.draw_instanced(3, 1, 0, 0).unwrap();
        super::run(&t.host, &[&list]);

        let mut pixels = [0u8; 4 * 4 * 16];
        t.host
            .backend
            .read_texture(&color, 0, 0, &mut pixels, 4 * 16, 0);
        for y in 0..4usize {
            for x in 0..4usize {
                let at = (y * 4 + x) * 16;
                let rgba: [f32; 4] = std::array::from_fn(|c| {
                    f32::from_le_bytes(pixels[at + c * 4..at + c * 4 + 4].try_into().unwrap())
                });
                let top_right = x >= 2 && y < 2;
                let want = [
                    (y * 4 + x) as f32,
                    expected_g,
                    if top_right { 300.0 } else { 100.0 },
                    if top_right { 400.0 } else { 200.0 },
                ];
                assert_eq!(rgba, want, "{label} pixel ({x}, {y})");
            }
        }
    }
}
