//! Translation-only #439 regression programs. No GPU or validator is opened.
use super::lower::{self, Lowered, Parts};
use crate::native::{ComputeIo, HwStage, NativeProgram, RootParam, RootType, StageIo};
use crate::{ShaderError, io, ir, resource, structure};

fn lower_body(body: &[u32], threads: u32, lds: bool) -> Result<Lowered, ShaderError> {
    let mut words = vec![
        0xBE82_0300,
        0xBE83_0380,
        0xF408_0101,
        0xFA00_0000,
        0x9308_FF01,
        threads * 4,
        0xBF8C_C07F,
    ];
    words.extend_from_slice(body);
    words.extend_from_slice(&[
        0x3406_0082,
        0x4A06_0608,
        0xE070_1000,
        0x8001_0203,
        0xBF81_0000,
    ]);
    let native = NativeProgram {
        stage: HwStage::Cs,
        wave64: true,
        vgprs: 256,
        layout: vec![RootParam {
            root: RootType::Table,
            table_offset_dwords: 0,
            sgprs: 1,
            base_register: 0,
            user_data: 0,
            slot: 0,
        }],
        used: vec![],
        io: StageIo::Cs(ComputeIo {
            rsrc2: 1 << 1 | 1 << 7 | u32::from(lds) << 15,
            threads: [threads, 1, 1],
        }),
        code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
    };
    let p = ir::build(&native.code, true, 256)?;
    let st = structure::structure(&p)?;
    let res = resource::track(&p, &native.layout, native.stage, Some(&st))?;
    let io = io::raise(&p, &native)?;
    lower::lower(&Parts {
        native: &native,
        p: &p,
        st: &st,
        io: &io,
        res: &res,
        inputs: vec![],
        outputs: vec![],
        prims: vec![],
        mesh_stride: 0,
        view_marks: false,
        atomic_locks: false,
        bias_as_gradients: false,
        narrow_waves: false,
        wave32_host: true,
        zero_copy_memory: true,
        rect_list: false,
        copy_params: [0; 32],
        bvh_no_hit: false,
    })
}

#[test]
fn exchanges_inside_a_branch_one_half_skips() {
    let body = [
        0x7E04_0281, // v_mov_b32 v2, 1
        0x7D88_0094, // v_cmp_gt_u32 vcc, 20, v0
        0xBE8A_246A, // s_and_saveexec_b64 s[10:11], vcc
        0xBF88_0003, // s_cbranch_execz +3
        0xD760_0009,
        0x0001_1500, // v_readlane_b32 s9, v0, 10
        0x7E04_0209, // v_mov_b32 v2, s9
        0x88FE_0A7E, // s_or_b64 exec, exec, s[10:11]
    ];
    let l = lower_body(&body, 64, false).unwrap();
    assert_eq!(l.wave_size, Some(32));
    assert!(l.exchanges);
}

#[test]
fn loops_run_while_any_lane_of_the_console_wave_does() {
    let body = [
        0xBE89_0380, // s_mov_b32 s9, 0
        0x7E04_0280, // v_mov_b32 v2, 0
        0x7D8A_0080, // v_cmp_ne_u32 vcc, 0, v0
        0xBE8A_246A, // s_and_saveexec_b64 s[10:11], vcc
        0xBF88_0006, // s_cbranch_execz end
        // loop:
        0xD760_0009,
        0x0001_5100, // v_readlane_b32 s9, v0, 40
        0x4A04_0481, // v_add_nc_u32 v2, 1, v2
        0x7D82_0102, // v_cmp_lt_u32 vcc, v2, v0
        0x87FE_6A7E, // s_and_b64 exec, exec, vcc
        0xBF89_FFFA, // s_cbranch_execnz loop
        // end:
        0x88FE_0A7E, // s_or_b64 exec, exec, s[10:11]
        0x4A04_0409, // v_add_nc_u32 v2, s9, v2
    ];
    let l = lower_body(&body, 64, false).unwrap();
    assert_eq!(l.wave_size, Some(32));
    assert!(l.exchanges);
}

#[test]
fn first_lane_reads_are_the_console_waves() {
    let body = [
        0x7E06_02FA,
        0xFF00_E400, // v_mov_b32_dpp v3, v0 quad_perm:[0,1,2,3]
        0x7E12_0500, // v_readfirstlane_b32 s9, v0
        0x7E04_0209, // v_mov_b32 v2, s9
    ];
    let l = lower_body(&body, 64, false).unwrap();
    assert_eq!(l.wave_size, Some(32));
    assert!(l.exchanges);
}

#[test]
fn lane_writes_reach_lanes_outside_the_branch() {
    let body = [
        0x7E04_0281, // v_mov_b32 v2, 1
        0x7D88_0094, // v_cmp_gt_u32 vcc, 20, v0
        0xBE8A_246A, // s_and_saveexec_b64 s[10:11], vcc
        0xBF88_0004, // s_cbranch_execz +4
        0xBE89_03FF,
        0x0000_004D, // s_mov_b32 s9, 77
        0xD761_0002,
        0x0001_5009, // v_writelane_b32 v2, s9, 40
        0x88FE_0A7E, // s_or_b64 exec, exec, s[10:11]
    ];
    let l = lower_body(&body, 64, false).unwrap();
    assert_eq!(l.wave_size, Some(32));
    assert!(l.exchanges);
}

#[test]
fn loops_with_an_exit_mask_grown_from_zero() {
    let body = [
        0x4A02_0081, // v_add_nc_u32 v1, 1, v0
        0xBE8A_0480, // s_mov_b64 s[10:11], 0
        0xBE89_0380, // s_mov_b32 s9, 0
        // loop:
        0xD760_0009,
        0x0001_5100, // v_readlane_b32 s9, v0, 40
        0x4E02_0281, // v_subrev_nc_u32 v1, 1, v1
        0x7D84_0280, // v_cmp_eq_u32 vcc, 0, v1
        0x888A_0A6A, // s_or_b64 s[10:11], vcc, s[10:11]
        0x8AFE_0A7E, // s_andn2_b64 exec, exec, s[10:11]
        0xBF89_FFF9, // s_cbranch_execnz loop
        0x88FE_0A7E, // s_or_b64 exec, exec, s[10:11]
        0xBE8C_100A, // s_bcnt1_i32_b64 s12, s[10:11]
        0x7E04_0209, // v_mov_b32 v2, s9
        0x4A04_040C, // v_add_nc_u32 v2, s12, v2
    ];
    let l = lower_body(&body, 64, false).unwrap();
    assert_eq!(l.wave_size, Some(32));
    assert!(l.exchanges);
}

#[test]
fn saveexec_scc_tests_new_exec_not_saved_exec() {
    let body = [
        0xBE89_03AA, // s_mov_b32 s9, 42
        0x7D88_0080, // v_cmp_gt_u32 vcc, 0, v0 (false in every lane)
        0xBE8C_246A, // s_and_saveexec_b64 s[12:13], vcc (old != 0, new == 0 -> SCC = 0)
        0xBF84_0003, // s_cbranch_scc0 +3 (must skip the body)
        0xD760_000E,
        0x0001_1500, // v_readlane_b32 s14, v0, 10 (takes the console flow)
        0xBE89_03BC, // s_mov_b32 s9, 60 (must not run)
        0x88FE_0C7E, // s_or_b64 exec, exec, s[12:13]
        0x7E04_0209, // v_mov_b32 v2, s9
    ];
    let l = lower_body(&body, 64, false).unwrap();
    assert_eq!(l.wave_size, Some(32));
    assert!(l.exchanges);
}

#[test]
fn loops_leave_on_the_new_exec_of_saveexec() {
    let body = [
        0xD760_000E,
        0x0001_1500, // v_readlane_b32 s14, v0, 10 (takes the console flow)
        0x4A02_0081, // v_add_nc_u32 v1, 1, v0
        0xBE8A_047E, // s_mov_b64 s[10:11], exec
        0xBE89_0380, // s_mov_b32 s9, 0
        // loop:
        0x8009_8109, // s_add_u32 s9, s9, 1
        0x4E02_0281, // v_subrev_nc_u32 v1, 1, v1
        0x7D82_0280, // v_cmp_lt_u32 vcc, 0, v1
        0xBE8C_246A, // s_and_saveexec_b64 s[12:13], vcc
        0xBF85_FFFB, // s_cbranch_scc1 loop
        0x88FE_0A7E, // s_or_b64 exec, exec, s[10:11]
        0x7E04_0209, // v_mov_b32 v2, s9
    ];
    let l = lower_body(&body, 64, false).unwrap();
    assert_eq!(l.wave_size, Some(32));
    assert!(l.exchanges);
}

#[test]
fn copied_masks_terminate_without_recursing() {
    let body = [
        0x7D88_0094, // v_cmp_gt_u32 vcc, 20, v0
        0xBE8A_047E, // s_mov_b64 s[10:11], exec
        0x878C_6A7E, // s_and_b64 s[12:13], exec, vcc
        0xBEFE_030C, // s_mov_b32 exec_lo, s12
        0xBEFF_030D, // s_mov_b32 exec_hi, s13
        0xD760_0009,
        0x0001_1500, // v_readlane_b32 s9, v0, 10
        0x7E04_0209, // v_mov_b32 v2, s9
        0xBEFE_040A, // s_mov_b64 exec, s[10:11]
    ];
    let _ = lower_body(&body, 64, false);
}

#[test]
fn several_console_waves_cannot_barrier_in_divergent_flow() {
    let body = [
        0x7E04_0281, // v2 = 1
        0x7D88_0094, // v_cmp_gt_u32 vcc, 20, v0
        0xBE8A_246A, // s_and_saveexec_b64 s[10:11], vcc
        0xBF88_0003, // skip readlane and move
        0xD760_0009,
        0x0001_5100, // readlane s9, v0, 40
        0x7E04_0209, // v2 = s9
        0x88FE_0A7E, // restore EXEC
    ];
    assert!(lower_body(&body, 64, false).unwrap().console_flow);
    assert!(lower_body(&body, 128, false).is_err());
}

#[test]
fn title_memory_with_lds_does_not_enter_console_flow() {
    let branch = [
        0x7E04_0281,
        0x7D88_0094,
        0xBE8A_246A,
        0xBF88_0003,
        0xD760_0009,
        0x0001_5100,
        0x7E04_0209,
        0x88FE_0A7E,
    ];
    assert!(lower_body(&branch, 64, true).unwrap().console_flow);
    let mut body = vec![
        0xBE8C_03FF,
        0x0000_1000, // s12 = an arbitrary address
        0xBE8D_0380, // s13 = 0
        0xF400_0386,
        0xFA00_0000, // s_load_dword s14, s[12:13], null
    ];
    body.extend(branch);
    body.push(0x4A04_040E); // consume the title-memory load
    let err = lower_body(&body, 64, true)
        .err()
        .expect("conservative #439 exclusion");
    assert!(
        err.to_string()
            .contains("cross-lane operation between host waves"),
        "{err}"
    );
}

#[test]
fn a_whole_wave_section_outside_the_sum_simulator_uses_console_flow() {
    // LDEXP is not a reduction step the section simulator recognizes.
    // Use the result in a store so silently dropping it cannot pass the test.
    let body = [
        0xBE8A_25C1, // s_or_saveexec_b64 s[10:11], -1
        0x7E04_0300, // v_mov_b32 v2, v0
        0xD762_0002,
        0x0001_0302, // v_ldexp_f32 v2, v2, 1
        0xD760_000C,
        0x0001_5102, // v_readlane_b32 s12, v2, 40
        0xBEFE_040A, // s_mov_b64 exec, s[10:11]
        0x7E04_020C, // v_mov_b32 v2, s12
    ];
    let l = lower_body(&body, 64, false).unwrap();
    assert!(l.console_flow && l.exchanges);
    assert_eq!(l.wave_size, Some(32));
}
