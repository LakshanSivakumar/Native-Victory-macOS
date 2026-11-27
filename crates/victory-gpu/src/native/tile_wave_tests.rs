//! The cross-lane work of a tile classifier, on the GPU, in dispatches of many groups: the
//! title's `tiled_lighting` compute shaders run 8 x 8 threads a group (one console wave of 64
//! lanes, two host waves) over 54 x 30 groups and gather each tile's material bits through LDS
//! with no barrier, pick lanes by prefix counts, read named lanes across the halves and switch on
//! a per-lane number by narrowing EXEC. Programs of our own (assembled with `llvm-mc -arch=amdgcn
//! -mcpu=gfx1030 -mattr=+wavefrontsize64`) in the same instruction forms, with inputs that differ
//! by group and by lane; every lane's result is checked against what the console's lanes give
//! (RDNA2 ISA: 5.5 EXEC and lane masks, 6.3 `v_readlane` / `v_readfirstlane` ignore or follow
//! EXEC, 6.4 DPP, 10 LDS).
//!
//! Like those shaders, the programs reach no title memory by address (they store through a
//! buffer view of a descriptor table) and read a first lane, so they are lowered with the console
//! flow in halves (`victory_shader`'s `lower/console.rs`); `VICTORY_NO_CONSOLE_FLOW=1` runs them
//! through upstream's lowering instead, where a program it refuses is skipped. Skipped without
//! the Metal Shader Converter.

use super::addr_tests::{CANARY, Fixture, VA_A, halves, title};
use crate::native::list::NativeList;
use crate::native::shaders;
use crate::native::title::Pipe;
use victory_backend::resources::HeapKind;
use victory_shader::emit::emit_with;
use victory_shader::native::{
    ComputeIo, HwStage, NativeProgram, ResourceType, RootParam, RootType, StageIo, UsedResource,
};
use victory_shader::{Container, io, ir, resource, structure};

/// Groups in x and y of every dispatch here.
const GROUPS: [u32; 2] = [4, 3];
/// Pool offset of the stored dwords, at `VA_A + 0x4_0000`.
const DST: usize = 0x6_0000;

/// What every program starts with. `s4` is the groups in x, `s5` the descriptor table, `s6` and
/// `s7` the group's id, `v0` and `v1` the thread's id in the group: `v2` is the lane
/// (`8 * v1 + v0`, the thread's index in the group), `s7` the group's number `g`, `v3` the byte
/// offset of the thread's four dwords and `s[20:23]` the buffer they are stored in. The first
/// lane read is not used.
const START: [u32; 13] = [
    0xD746_0002,
    0x0401_0701, // v_lshl_add_u32 v2, v1, 3, v0
    0x9308_0407, // s_mul_i32 s8, s7, s4
    0x8107_0608, // s_add_i32 s7, s8, s6
    0x8F08_8607, // s_lshl_b32 s8, s7, 6
    0x4A06_0408, // v_add_nc_u32_e32 v3, s8, v2
    0x3406_0684, // v_lshlrev_b32_e32 v3, 4, v3
    0xBE98_0305, // s_mov_b32 s24, s5
    0xBE99_0380, // s_mov_b32 s25, 0
    0xF408_050C,
    0xFA00_0000, // s_load_dwordx4 s[20:23], s[24:25], 0x0
    0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
    0x7E34_0502, // v_readfirstlane_b32 s26, v2
];

/// `words` as a compute program of 8 x 8 threads in 64-lane waves with 512 bytes of LDS, the
/// group's id in two SGPRs and the thread's in two VGPRs; its root signature is five constants
/// and a table of one buffer view for unordered access.
fn translate(words: &[u32]) -> Result<victory_shader::emit::HostShader, victory_shader::ShaderError> {
    let param = |root, user_data, sgprs, slot| RootParam {
        root,
        table_offset_dwords: 0,
        sgprs,
        base_register: 0,
        user_data,
        slot,
    };
    let native = NativeProgram {
        stage: HwStage::Cs,
        wave64: true,
        vgprs: 16,
        layout: vec![param(RootType::Constants, 0, 5, 0), param(RootType::Table, 5, 1, 1)],
        used: vec![UsedResource {
            kind: ResourceType::Uav,
            root: RootType::Table,
            table_offset_dwords: 0,
            lower_bound: 0,
            stride: 0,
            dwords: 4,
            user_data: 5,
            slot: 1,
        }],
        // Six user SGPRs, the group's id in x and y (bits 7, 8), two thread id VGPRs (bit 11),
        // one granule of LDS (bit 15).
        io: StageIo::Cs(ComputeIo {
            rsrc2: 1 << 15 | 1 << 11 | 1 << 8 | 1 << 7 | 6 << 1,
            threads: [8, 8, 1],
        }),
        code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
    };
    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
    let st = structure::structure(&p).expect("structure");
    let res = resource::track(&p, &native.layout, native.stage, Some(&st)).expect("track");
    let io = io::raise(&p, &native).expect("io::raise");
    // A container of no parts: `DXBC`, a digest, version 1.0, its size, no parts.
    let mut empty = b"DXBC".to_vec();
    empty.extend([0; 16]);
    empty.extend([1, 0, 0, 0]);
    empty.extend(32u32.to_le_bytes());
    empty.extend(0u32.to_le_bytes());
    let c = Container::parse(&empty).expect("container");
    emit_with(&c, &native, &p, &st, &io, &res, shaders::options())
}

/// `body` after [`START`], run over [`GROUPS`]: the four dwords of each thread, by group number
/// and lane. `None` where the GPU cannot be used, or the lowering asked for refuses the program.
fn run(body: &[u32]) -> Option<Vec<[[u32; 4]; 64]>> {
    let mut t = title()?;
    let f = Fixture::new(&t);
    let cs = match translate(&[&START[..], body].concat()) {
        Ok(cs) => cs,
        Err(e) if std::env::var_os("VICTORY_NO_CONSOLE_FLOW").is_some() => {
            eprintln!("skipped: refused without the console flow: {e}");
            return None;
        }
        Err(e) => panic!("refused: {e}"),
    };
    let dwords = (GROUPS[0] * GROUPS[1]) as usize * 64 * 4;
    // The title's heap of one entry: a raw buffer view for unordered access over the dwords.
    let records = vec![0u32; 8];
    let slot = records.as_ptr() as u64;
    let heap = t.host.heap(HeapKind::Resource, slot, 1).unwrap();
    let mut resource = [0u32; 14];
    resource[0] = 1;
    let mut view = [0u32; 16];
    (view[1], view[4], view[8]) = (1, dwords as u32, 1);
    [view[10], view[11]] = halves(VA_A + 0x4_0000);
    crate::descriptor::create_placed_uav(std::ptr::null_mut(), &resource, &view, slot);
    assert!(t.host.mirror(slot).unwrap());
    let Pipe::Compute(pso) = t.compute_pipe(&cs, "tiles") else {
        panic!("not made: {:?}", t.shaders)
    };
    assert_eq!(pso.threads, [8, 8, 1]);
    let mut l = NativeList::default();
    l.set_heaps(Some(&heap), None);
    l.set_compute_pipeline(&pso);
    l.set_compute_root_constants(0, &[0, 0, 0, 0, GROUPS[0]]);
    l.set_compute_root_table(1, &heap, 0);
    l.dispatch(GROUPS[0], GROUPS[1], 1).unwrap();
    super::run(&t.host, &[&l]);
    assert_eq!(t.shaders.refusals(), 0, "{:?}", t.shaders.refused);
    Some(
        f.get(DST, dwords)
            .chunks(256)
            .map(|g| std::array::from_fn(|lane| std::array::from_fn(|k| g[lane * 4 + k])))
            .collect(),
    )
}

/// What `body` stores against `want` by group and lane, a few times over (a race shows in some
/// runs only).
fn check(body: &[u32], want: impl Fn(u32, u32) -> [u32; 4]) {
    for round in 0..3 {
        let Some(got) = run(body) else { return };
        let mut wrong = Vec::new();
        for (g, lanes) in got.iter().enumerate() {
            for (lane, &have) in lanes.iter().enumerate() {
                let w = want(g as u32, lane as u32);
                if have != w {
                    wrong.push(format!("group {g} lane {lane}: {have:x?}, not {w:x?}"));
                }
            }
        }
        assert!(
            wrong.is_empty(),
            "round {round}: {} of {} lanes wrong, the first: {:#?}",
            wrong.len(),
            got.len() * 64,
            &wrong[..wrong.len().min(12)]
        );
    }
}

const C: u32 = CANARY;

/// Lanes of group `g` a mask of the tests holds: `(lane + g) & 5 != 0` from lane `40 - 3g` on.
fn picked(g: u32, lane: u32) -> bool {
    (lane + g) & 5 != 0 && lane >= 40 - 3 * g
}

#[test]
fn prefix_counts_and_the_first_lane_of_a_mask_in_each_group() {
    // `v_mbcnt` of a mask (the count of its lanes below this one) and of all lanes (the lane's
    // index), and the first lane of the mask picked as the title picks one lane to append a
    // record: `s_ff1_i32_b64` of EXEC compared with the lane's index. The mask starts in the
    // upper half in the first groups and in the lower in the last.
    let body = [
        0x9309_8307, // s_mul_i32 s9, s7, 3
        0x8189_09A8, // s_sub_i32 s9, 40, s9
        0x4A08_0407, // v_add_nc_u32_e32 v4, s7, v2
        0x3608_0885, // v_and_b32_e32 v4, 5, v4
        0x7D8A_0880, // v_cmp_ne_u32_e32 vcc, 0, v4
        0xD4C3_000A,
        0x0002_0409, // v_cmp_le_u32_e64 s[10:11], s9, v2
        0x87EA_0A6A, // s_and_b64 vcc, vcc, s[10:11]
        0xD765_0005,
        0x0001_006A, // v_mbcnt_lo_u32_b32 v5, vcc_lo, 0
        0xD766_0005,
        0x0002_0A6B, // v_mbcnt_hi_u32_b32 v5, vcc_hi, v5
        0xD765_0006,
        0x0001_00C1, // v_mbcnt_lo_u32_b32 v6, -1, 0
        0xD766_0006,
        0x0002_0CC1, // v_mbcnt_hi_u32_b32 v6, -1, v6
        0xD76F_0005,
        0x0419_1105, // v_lshl_or_b32 v5, v5, 8, v6
        0xE070_1000,
        0x8005_0503, // buffer_store_dword v5, v3, s[20:23], 0 offen
        0xBE8A_246A, // s_and_saveexec_b64 s[10:11], vcc
        0xBE89_147E, // s_ff1_i32_b64 s9, exec
        0x7D24_0C09, // v_cmpx_eq_i32_e32 s9, v6
        0x7E0E_0209, // v_mov_b32_e32 v7, s9
        0xE070_1004,
        0x8005_0703, // buffer_store_dword v7, v3, s[20:23], 0 offen offset:4
        0xBF81_0000, // s_endpgm
    ];
    check(&body, |g, lane| {
        let below = (0..lane).filter(|&k| picked(g, k)).count() as u32;
        let first = (0..64).find(|&k| picked(g, k)).unwrap();
        [below << 8 | lane, if lane == first { first } else { C }, C, C]
    });
}

#[test]
fn named_and_first_lane_reads_under_part_of_exec_in_each_group() {
    // Lanes from `40 - 3g` on stay. `v_readfirstlane_b32` gives the first of them (RDNA2 ISA
    // 6.3: the first lane EXEC has, lane 0 only when EXEC is empty); they then add 1000 to their
    // own `v5`, and `v_readlane_b32` of lanes 0, 31, 32 and 63 reads each lane's register as it
    // is, in EXEC or not, in this half or the other. The lanes that left then read the first of
    // themselves: lane 0.
    if std::env::var_os("VICTORY_NO_CONSOLE_FLOW").is_some() {
        // Upstream's lowering gives each half its own first lane whatever EXEC holds (the fault
        // `wave32_tests::a_branch_on_a_first_lane_read_is_not_uniform_across_the_halves` of
        // `victory_shader` keeps): 469 of the 768 lanes here. Only the console flow is held to it.
        eprintln!("skipped: upstream's lowering reads each half's own first lane");
        return;
    }
    let body = [
        0x7E08_0207, // v_mov_b32_e32 v4, s7
        0xD569_0005,
        0x0002_0902, // v_mul_lo_u32 v5, v2, v4
        0x4A0A_0A87, // v_add_nc_u32_e32 v5, 7, v5
        0x9309_8307, // s_mul_i32 s9, s7, 3
        0x8189_09A8, // s_sub_i32 s9, 40, s9
        0x7D86_0409, // v_cmp_le_u32_e32 vcc, s9, v2
        0xBE8A_246A, // s_and_saveexec_b64 s[10:11], vcc
        0x7E18_0505, // v_readfirstlane_b32 s12, v5
        0x4A0A_0AFF,
        0x0000_03E8, // v_add_nc_u32_e32 v5, 0x3e8, v5
        0xD760_000D,
        0x0001_0105, // v_readlane_b32 s13, v5, 0
        0xD760_000E,
        0x0001_4105, // v_readlane_b32 s14, v5, 32
        0xD760_000F,
        0x0001_7F05, // v_readlane_b32 s15, v5, 63
        0xD760_0010,
        0x0001_3F05, // v_readlane_b32 s16, v5, 31
        0x7E0C_020C, // v_mov_b32_e32 v6, s12
        0x7E0E_020D, // v_mov_b32_e32 v7, s13
        0x7E10_020E, // v_mov_b32_e32 v8, s14
        0x7E12_020F, // v_mov_b32_e32 v9, s15
        0x7E14_0210, // v_mov_b32_e32 v10, s16
        0xD76F_0007,
        0x041D_210A, // v_lshl_or_b32 v7, v10, 16, v7
        0xE070_1000,
        0x8005_0603, // buffer_store_dword v6, v3, s[20:23], 0 offen
        0xE070_1004,
        0x8005_0703, // buffer_store_dword v7, v3, s[20:23], 0 offen offset:4
        0xE070_1008,
        0x8005_0803, // buffer_store_dword v8, v3, s[20:23], 0 offen offset:8
        0xE070_100C,
        0x8005_0903, // buffer_store_dword v9, v3, s[20:23], 0 offen offset:12
        0x8AFE_7E0A, // s_andn2_b64 exec, s[10:11], exec
        0x7E18_0505, // v_readfirstlane_b32 s12, v5
        0x7E0C_020C, // v_mov_b32_e32 v6, s12
        0xE070_1000,
        0x8005_0603, // buffer_store_dword v6, v3, s[20:23], 0 offen
        0xBF81_0000, // s_endpgm
    ];
    check(&body, |g, lane| {
        let t = 40 - 3 * g;
        let v5 = |l: u32| l * g + 7 + if l >= t { 1000 } else { 0 };
        if lane >= t {
            [t * g + 7, v5(31) << 16 | v5(0), v5(32), v5(63)]
        } else {
            [7, C, C, C]
        }
    });
}

#[test]
fn a_whole_wave_sum_and_a_prefix_sum_of_some_lanes_in_each_group() {
    // The sum and the exclusive prefix sum of `lane + g` over the lanes with `(lane + g) & 3`
    // not 0, in the steps the console compiler makes of them: DPP adds within rows of 16
    // (`dpp8`, `row_mirror`, `row_shr`), `v_permlanex16_b32` between the rows of a half (with
    // lanes fetched that EXEC lacks), `v_readlane_b32` of lanes 32, 0 and 31 between the halves,
    // and EXEC written a half at a time.
    let body = [
        0x4A0A_0407, // v_add_nc_u32_e32 v5, s7, v2
        0x360C_0A83, // v_and_b32_e32 v6, 3, v5
        0x7D8A_0C80, // v_cmp_ne_u32_e32 vcc, 0, v6
        0xBE8A_246A, // s_and_saveexec_b64 s[10:11], vcc
        0xBE8C_25C1, // s_or_saveexec_b64 s[12:13], -1
        0xD501_0006,
        0x0032_0A80, // v_cndmask_b32_e64 v6, 0, v5, s[12:13]
        0x4A0C_0CE9,
        0xDE54_C106, // v_add_nc_u32_dpp v6, v6, v6 dpp8:[1,0,3,2,5,4,7,6]
        0x4A0C_0CE9,
        0x9360_1206, // v_add_nc_u32_dpp v6, v6, v6 dpp8:[2,2,0,0,6,6,4,4]
        0x4A0C_0CE9,
        0x0009_2406, // v_add_nc_u32_dpp v6, v6, v6 dpp8:[4,4,4,4,0,0,0,0]
        0x4A0C_0CFA,
        0xFF09_4006, // v_add_nc_u32_dpp v6, v6, v6 row_mirror row_mask:0xf bank_mask:0xf bound_ctrl:1
        0xD778_0007,
        0x0201_0106, // v_permlanex16_b32 v7, v6, 0, 0
        0x4A0C_0D07, // v_add_nc_u32_e32 v6, v7, v6
        0xD760_000E,
        0x0001_4106, // v_readlane_b32 s14, v6, 32
        0x4A0C_0C0E, // v_add_nc_u32_e32 v6, s14, v6
        0xD760_000F,
        0x0001_0106, // v_readlane_b32 s15, v6, 0
        0xD501_0008,
        0x0032_0A80, // v_cndmask_b32_e64 v8, 0, v5, s[12:13]
        0x7E12_0308, // v_mov_b32_e32 v9, v8
        0x4A12_12FA,
        0xFF09_1109, // v_add_nc_u32_dpp v9, v9, v9 row_shr:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
        0x4A12_12FA,
        0xFF01_1209, // v_add_nc_u32_dpp v9, v9, v9 row_shr:2 row_mask:0xf bank_mask:0xf
        0x4A12_12FA,
        0xFF01_1409, // v_add_nc_u32_dpp v9, v9, v9 row_shr:4 row_mask:0xf bank_mask:0xf
        0x4A12_12FA,
        0xFF01_1809, // v_add_nc_u32_dpp v9, v9, v9 row_shr:8 row_mask:0xf bank_mask:0xf
        0xBEFE_03FF,
        0xFFFF_0000, // s_mov_b32 exec_lo, 0xffff0000
        0xBEFF_03FF,
        0xFFFF_0000, // s_mov_b32 exec_hi, 0xffff0000
        0xD778_080A,
        0x0305_8309, // v_permlanex16_b32 v10, v9, -1, -1 op_sel:[1,0]
        0x4A12_130A, // v_add_nc_u32_e32 v9, v10, v9
        0xD760_0010,
        0x0001_3F09, // v_readlane_b32 s16, v9, 31
        0xBEFE_0380, // s_mov_b32 exec_lo, 0
        0xBEFF_03C1, // s_mov_b32 exec_hi, -1
        0x4A12_1210, // v_add_nc_u32_e32 v9, s16, v9
        0xBEFE_03C1, // s_mov_b32 exec_lo, -1
        0xBEFF_03C1, // s_mov_b32 exec_hi, -1
        0x4C12_1109, // v_sub_nc_u32_e32 v9, v9, v8
        0xBEFE_040C, // s_mov_b64 exec, s[12:13]
        0x7E16_020F, // v_mov_b32_e32 v11, s15
        0xE070_1000,
        0x8005_0B03, // buffer_store_dword v11, v3, s[20:23], 0 offen
        0xE070_1004,
        0x8005_0903, // buffer_store_dword v9, v3, s[20:23], 0 offen offset:4
        0xBF81_0000, // s_endpgm
    ];
    check(&body, |g, lane| {
        let on = |l: u32| (l + g) & 3 != 0;
        let sum = |to: u32| (0..to).filter(|&l| on(l)).map(|l| l + g).sum::<u32>();
        if on(lane) { [sum(64), sum(lane), C, C] } else { [C; 4] }
    });
}

#[test]
fn four_words_of_lds_gather_bits_and_sums_of_the_lanes_in_each_group() {
    // The thread at (0, 0) clears four words of LDS; lanes 20 and up OR a bit into the fourth
    // and add to the third and second, the odd ones among them OR into the first; the thread at
    // (0, 0) reads the four back. No barrier anywhere: the console runs the wave in lockstep.
    let body = [
        0x3808_0101, // v_or_b32_e32 v4, v1, v0
        0xD4C2_000A,
        0x0002_0880, // v_cmp_eq_u32_e64 s[10:11], 0, v4
        0xBE8C_240A, // s_and_saveexec_b64 s[12:13], s[10:11]
        0x7E08_0280, // v_mov_b32_e32 v4, 0
        0xD838_0302,
        0x0004_0404, // ds_write2_b32 v4, v4, v4 offset0:2 offset1:3
        0xD838_0100,
        0x0004_0404, // ds_write2_b32 v4, v4, v4 offset1:1
        0xBEFE_040C, // s_mov_b64 exec, s[12:13]
        0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
        0x2C0A_0481, // v_lshrrev_b32_e32 v5, 1, v2
        0x4A0A_0A07, // v_add_nc_u32_e32 v5, s7, v5
        0x360A_0A9F, // v_and_b32_e32 v5, 31, v5
        0xD51A_0005,
        0x0001_0305, // v_lshlrev_b32_e64 v5, v5, 1
        0x4A0C_0407, // v_add_nc_u32_e32 v6, s7, v2
        0x360E_0481, // v_and_b32_e32 v7, 1, v2
        0xBE8E_047E, // s_mov_b64 s[14:15], exec
        0x7DA6_0494, // v_cmpx_le_u32_e32 20, v2
        0x7E10_0280, // v_mov_b32_e32 v8, 0
        0xD828_000C,
        0x0000_0508, // ds_or_b32 v8, v5 offset:12
        0xD800_0008,
        0x0000_0608, // ds_add_u32 v8, v6 offset:8
        0xD800_0004,
        0x0000_0708, // ds_add_u32 v8, v7 offset:4
        0x7DAA_0E80, // v_cmpx_ne_u32_e32 0, v7
        0x7E12_0280, // v_mov_b32_e32 v9, 0
        0x7E14_0207, // v_mov_b32_e32 v10, s7
        0x4A14_1481, // v_add_nc_u32_e32 v10, 1, v10
        0xD828_0000,
        0x0000_0A09, // ds_or_b32 v9, v10
        0xBEFE_040E, // s_mov_b64 exec, s[14:15]
        0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
        0xBE8C_240A, // s_and_saveexec_b64 s[12:13], s[10:11]
        0xBF88_000E, // s_cbranch_execz end
        0x7E08_0280, // v_mov_b32_e32 v4, 0
        0xD8DC_0100,
        0x0B00_0004, // ds_read2_b32 v[11:12], v4 offset1:1
        0xD8DC_0302,
        0x0D00_0004, // ds_read2_b32 v[13:14], v4 offset0:2 offset1:3
        0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
        0xE070_1000,
        0x8005_0B03, // buffer_store_dword v11, v3, s[20:23], 0 offen
        0xE070_1004,
        0x8005_0C03, // buffer_store_dword v12, v3, s[20:23], 0 offen offset:4
        0xE070_1008,
        0x8005_0D03, // buffer_store_dword v13, v3, s[20:23], 0 offen offset:8
        0xE070_100C,
        0x8005_0E03, // buffer_store_dword v14, v3, s[20:23], 0 offen offset:12
        // end:
        0xBF81_0000, // s_endpgm
    ];
    check(&body, |g, lane| {
        if lane != 0 {
            return [C; 4];
        }
        let bits = (20..64).fold(0, |m, l| m | 1 << (((l >> 1) + g) & 31));
        [g + 1, 22, (20..64).map(|l| l + g).sum(), bits]
    });
}

#[test]
fn lds_bits_maxima_and_minima_of_each_quarter_of_a_group() {
    // The four 4 x 4 quarters of the group each keep three words of LDS: the quarter's first
    // thread (lanes 0, 4, 32 and 36) writes 0, 0xffff and 0, every lane ORs a bit, and takes the
    // maximum and the minimum with its number, the first thread reads them and the others end
    // (`ds_or_b32`, `ds_max_u32`, `ds_min_u32`).
    let body = [
        0x3808_0101, // v_or_b32_e32 v4, v1, v0
        0x3608_0883, // v_and_b32_e32 v4, 3, v4
        0x7D84_0880, // v_cmp_eq_u32_e32 vcc, 0, v4
        0x2C0A_0081, // v_lshrrev_b32_e32 v5, 1, v0
        0x360A_0A82, // v_and_b32_e32 v5, 2, v5
        0x2C0C_0282, // v_lshrrev_b32_e32 v6, 2, v1
        0x4A0A_0B06, // v_add_nc_u32_e32 v5, v6, v5
        0x340A_0A82, // v_lshlrev_b32_e32 v5, 2, v5
        0xBE8A_246A, // s_and_saveexec_b64 s[10:11], vcc
        0x7E0C_0280, // v_mov_b32_e32 v6, 0
        0x7E0E_02FF,
        0x0000_FFFF, // v_mov_b32_e32 v7, 0xffff
        0xD838_0C08,
        0x0006_0705, // ds_write2_b32 v5, v7, v6 offset0:8 offset1:12
        0xD834_0010,
        0x0000_0605, // ds_write_b32 v5, v6 offset:16
        0xBEFE_040A, // s_mov_b64 exec, s[10:11]
        0x9309_8B07, // s_mul_i32 s9, s7, 11
        0x8109_A509, // s_add_i32 s9, s9, 37
        0x7E10_0209, // v_mov_b32_e32 v8, s9
        0xD569_0009,
        0x0002_1102, // v_mul_lo_u32 v9, v2, v8
        0x3612_12FF,
        0x0000_03FF, // v_and_b32_e32 v9, 0x3ff, v9
        0x3614_129F, // v_and_b32_e32 v10, 31, v9
        0xD51A_000A,
        0x0001_030A, // v_lshlrev_b32_e64 v10, v10, 1
        0xD828_0030,
        0x0000_0A05, // ds_or_b32 v5, v10 offset:48
        0xD820_0010,
        0x0000_0905, // ds_max_u32 v5, v9 offset:16
        0xD81C_0020,
        0x0000_0905, // ds_min_u32 v5, v9 offset:32
        0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
        0xBE8A_246A, // s_and_saveexec_b64 s[10:11], vcc
        0xBF88_0001, // s_cbranch_execz end
        0xBF82_0001, // s_branch cont
        // end:
        0xBF81_0000, // s_endpgm
        // cont:
        0xD8D8_0030,
        0x0B00_0005, // ds_read_b32 v11, v5 offset:48
        0xD8D8_0010,
        0x0C00_0005, // ds_read_b32 v12, v5 offset:16
        0xD8D8_0020,
        0x0D00_0005, // ds_read_b32 v13, v5 offset:32
        0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
        0xE070_1000,
        0x8005_0B03, // buffer_store_dword v11, v3, s[20:23], 0 offen
        0xE070_1004,
        0x8005_0C03, // buffer_store_dword v12, v3, s[20:23], 0 offen offset:4
        0xE070_1008,
        0x8005_0D03, // buffer_store_dword v13, v3, s[20:23], 0 offen offset:8
        0xBF81_0000, // s_endpgm
    ];
    check(&body, |g, lane| {
        let quarter = |l: u32| (l % 8 / 4) * 2 + l / 32;
        if lane % 4 != 0 || lane / 8 % 4 != 0 {
            return [C; 4];
        }
        let numbers: Vec<u32> = (0..64)
            .filter(|&l| quarter(l) == quarter(lane))
            .map(|l| l * (11 * g + 37) & 0x3ff)
            .collect();
        [
            numbers.iter().fold(0, |m, n| m | 1 << (n & 31)),
            *numbers.iter().max().unwrap(),
            (*numbers.iter().min().unwrap()).min(0xffff),
            C,
        ]
    });
}

#[test]
fn a_switch_by_narrowing_exec_then_a_guarded_rewrite_gathered_through_lds() {
    // Each lane has a number from 0 to 7 (the same in all lanes in some groups, by row or by
    // lane in others). The lanes with x below 7 switch on it the way the console compiler writes
    // a switch: `v_cmpx_ne_u32` takes the lanes of each case out of EXEC after a `v_mov_b32` set
    // their flags, and `s_cbranch_execz` leaves for the join once no lane is left. Lanes whose
    // flags have bit 0 or 2 then replace their number by 5 or 0, every lane of the switch ORs
    // the bit of its number into a word of LDS lane 0 cleared, and lane 0 reads the word.
    let body = [
        0x8709_8307, // s_and_b32 s9, s7, 3
        0x8F09_8109, // s_lshl_b32 s9, s9, 1
        0x2C08_0409, // v_lshrrev_b32_e32 v4, s9, v2
        0x4A08_0807, // v_add_nc_u32_e32 v4, s7, v4
        0x3608_0887, // v_and_b32_e32 v4, 7, v4
        0xD4C2_000A,
        0x0002_0480, // v_cmp_eq_u32_e64 s[10:11], 0, v2
        0xBE8C_240A, // s_and_saveexec_b64 s[12:13], s[10:11]
        0x7E0A_0280, // v_mov_b32_e32 v5, 0
        0xD834_0000,
        0x0000_0505, // ds_write_b32 v5, v5
        0xBEFE_040C, // s_mov_b64 exec, s[12:13]
        0x7D8A_0087, // v_cmp_ne_u32_e32 vcc, 7, v0
        0x7E0C_02FF,
        0x0000_00EE, // v_mov_b32_e32 v6, 0xee
        0x7E10_02FF,
        0x0000_0210, // v_mov_b32_e32 v8, 0x210
        0xBE8E_246A, // s_and_saveexec_b64 s[14:15], vcc
        0xBF88_001E, // s_cbranch_execz out
        0x7E0C_0280, // v_mov_b32_e32 v6, 0
        0xBE90_047E, // s_mov_b64 s[16:17], exec
        0x7DAA_0880, // v_cmpx_ne_u32_e32 0, v4
        0xBF88_000B, // s_cbranch_execz join
        0x7E0C_0281, // v_mov_b32_e32 v6, 1
        0x7DAA_0881, // v_cmpx_ne_u32_e32 1, v4
        0xBF88_0008, // s_cbranch_execz join
        0x7E0C_0284, // v_mov_b32_e32 v6, 4
        0x7DAA_0882, // v_cmpx_ne_u32_e32 2, v4
        0xBF88_0005, // s_cbranch_execz join
        0x7E0C_0282, // v_mov_b32_e32 v6, 2
        0x7DAA_0883, // v_cmpx_ne_u32_e32 3, v4
        0x7E0C_02C0, // v_mov_b32_e32 v6, 64
        0x7D84_0885, // v_cmp_eq_u32_e32 vcc, 5, v4
        0x020C_1106, // v_cndmask_b32_e32 v6, v6, v8, vcc
        // join:
        0xBEFE_0410, // s_mov_b64 exec, s[16:17]
        0xBE92_047E, // s_mov_b64 s[18:19], exec
        0x3612_0C85, // v_and_b32_e32 v9, 5, v6
        0x7DAA_1280, // v_cmpx_ne_u32_e32 0, v9
        0xBF88_0004, // s_cbranch_execz skip
        0x3612_0C81, // v_and_b32_e32 v9, 1, v6
        0x7D84_1281, // v_cmp_eq_u32_e32 vcc, 1, v9
        0xD501_0004,
        0x01A9_0A80, // v_cndmask_b32_e64 v4, 0, 5, vcc
        // skip:
        0xBEFE_0412, // s_mov_b64 exec, s[18:19]
        0xD51A_0007,
        0x0001_0304, // v_lshlrev_b32_e64 v7, v4, 1
        0x7E0A_0280, // v_mov_b32_e32 v5, 0
        0xD828_0000,
        0x0000_0705, // ds_or_b32 v5, v7
        // out:
        0xBEFE_040E, // s_mov_b64 exec, s[14:15]
        0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
        0xD76F_000A,
        0x0411_1106, // v_lshl_or_b32 v10, v6, 8, v4
        0xE070_1000,
        0x8005_0A03, // buffer_store_dword v10, v3, s[20:23], 0 offen
        0xBE8C_240A, // s_and_saveexec_b64 s[12:13], s[10:11]
        0xBF88_0006, // s_cbranch_execz end
        0x7E0A_0280, // v_mov_b32_e32 v5, 0
        0xD8D8_0000,
        0x0B00_0005, // ds_read_b32 v11, v5
        0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
        0xE070_1004,
        0x8005_0B03, // buffer_store_dword v11, v3, s[20:23], 0 offen offset:4
        // end:
        0xBF81_0000, // s_endpgm
    ];
    let lane_of = |g: u32, lane: u32| {
        let number = ((lane >> ((g & 3) * 2)) + g) & 7;
        if lane % 8 == 7 {
            return (0xee, number, false);
        }
        let flags = match number {
            0 => 0,
            1 => 1,
            2 => 4,
            3 => 2,
            5 => 0x210,
            _ => 64,
        };
        let number = match flags & 5 {
            0 => number,
            f => (f & 1) * 5,
        };
        (flags, number, true)
    };
    check(&body, |g, lane| {
        let (flags, number, _) = lane_of(g, lane);
        let word = (0..64)
            .map(|l| lane_of(g, l))
            .filter(|l| l.2)
            .fold(0, |m, l| m | 1 << l.1);
        [flags << 8 | number, if lane == 0 { word } else { C }, C, C]
    });
}
