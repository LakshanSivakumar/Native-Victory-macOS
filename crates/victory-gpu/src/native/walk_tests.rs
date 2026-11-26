//! Wave64 programs whose two host waves meet only inside per-lane branches the whole console wave
//! takes (`victory_shader` `emit::widened`, `Options::no_wave_walk` off), on the GPU: synthetic
//! GFX10 programs of our own (those of `emit/walk_tests.rs` in victory-shader, assembled there),
//! translated with the host's options, converted by Apple's Metal Shader Converter and run. Every
//! lane of both halves must store what the console's lockstep wave of 64 lanes gives it. Each
//! program is refused without the walk. Skipped without the Metal Shader Converter.

use crate::native::list::NativeList;
use crate::native::memory::{PoolInfo, PoolObserver};
use crate::native::shaders;
use crate::native::title::{Pipe, Title};
use std::sync::Arc;
use victory_metal::mem::Alias;
use victory_metal::metal::Metal;
use victory_shader::emit::{HostShader, Options, emit_with};
use victory_shader::native::{ComputeIo, HwStage, NativeProgram, RootParam, RootType, StageIo};
use victory_shader::{Container, ShaderError, io, ir, resource, structure};

/// Root constants in `s0..s7`; the output's title address in `s[6:7]`, each lane's dword at
/// `v1` = 4 × its id.
const SGPRS: u32 = 8;

/// A count of the even lanes below 40 inside `if (id % 2 == 0)`: 20 in each even lane.
const COUNT_IN_BRANCH: &[u32] = &[
    0x3402_0082, // v_lshlrev_b32 v1, 2, v0
    0x3604_0081, // v_and_b32 v2, 1, v0
    0x7D84_0480, // v_cmp_eq_u32 vcc, 0, v2
    0xBE94_246A, // s_and_saveexec_b64 s[20:21], vcc
    0xBF88_0006, // s_cbranch_execz label_002C
    0xD4C4_000E,
    0x0002_00A8, // v_cmp_gt_u32 s[14:15], 40, v0
    0xBE90_100E, // s_bcnt1_i32_b64 s16, s[14:15]
    0x7E04_0210, // v_mov_b32 v2, s16
    0xDC70_8000,
    0x0006_0201, // global_store_dword v1, v2, s[6:7]
    0xBEFE_0414, // label_002C: s_mov_b64 exec, s[20:21]
    0xBF81_0000, // s_endpgm
];

/// A loop each lane goes round `id % 4 + 1` times (its exit per lane, by EXEC), adding the lanes
/// still in it to its sum and 1 to a scalar count: sums 64, 112, 144, 160 by `id % 4`, the count 4
/// (stored at +256).
const LOOP_BY_LANE: &[u32] = &[
    0x3402_0082, // v_lshlrev_b32 v1, 2, v0
    0x3606_0083, // v_and_b32 v3, 3, v0
    0x4A06_0681, // v_add_nc_u32 v3, 1, v3
    0x7E08_0280, // v_mov_b32 v4, 0
    0xBE91_0380, // s_mov_b32 s17, 0
    0xBE8C_047E, // s_mov_b64 s[12:13], exec
    0xBE90_107E, // label_0018: s_bcnt1_i32_b64 s16, exec
    0x4A08_0810, // v_add_nc_u32 v4, s16, v4
    0x8111_8111, // s_add_i32 s17, s17, 1
    0x4E06_0681, // v_subrev_nc_u32 v3, 1, v3
    0x7D8A_0680, // v_cmp_ne_u32 vcc, 0, v3
    0x87FE_6A7E, // s_and_b64 exec, exec, vcc
    0xBF89_FFF9, // s_cbranch_execnz label_0018
    0xBEFE_040C, // s_mov_b64 exec, s[12:13]
    0x7E0A_0211, // v_mov_b32 v5, s17
    0xDC70_8000,
    0x0006_0401, // global_store_dword v1, v4, s[6:7]
    0xDC70_8100,
    0x0006_0501, // global_store_dword v1, v5, s[6:7] offset:256
    0xBF81_0000, // s_endpgm
];

/// Inside `if (id % 2 == 0)`: each lane writes its id to LDS and reads lane id ^ 32's, which the
/// other half wrote (no barrier: the console's wave runs in lockstep). Even lanes store id ^ 32.
const LDS_IN_BRANCH: &[u32] = &[
    0x3402_0082, // v_lshlrev_b32 v1, 2, v0
    0x3604_0081, // v_and_b32 v2, 1, v0
    0x7D84_0480, // v_cmp_eq_u32 vcc, 0, v2
    0xBE94_246A, // s_and_saveexec_b64 s[20:21], vcc
    0xBF88_0009, // s_cbranch_execz label_0038
    0xD834_0000,
    0x0000_0001, // ds_write_b32 v1, v0
    0xBE92_03FF,
    0x0000_0080, // s_mov_b32 s18, 0x80
    0x3A0C_0212, // v_xor_b32 v6, s18, v1
    0xD8D8_0000,
    0x0700_0006, // ds_read_b32 v7, v6
    0xDC70_8000,
    0x0006_0701, // global_store_dword v1, v7, s[6:7]
    0xBEFE_0414, // label_0038: s_mov_b64 exec, s[20:21]
    0xBF81_0000, // s_endpgm
];

/// `WaveActiveSum(id)` inside `if (id < 50)`, as the console compiler expands it: 1225 in each
/// such lane.
const SUM_IN_BRANCH: &[u32] = &[
    0x3402_0082, // v_lshlrev_b32 v1, 2, v0
    0x7D88_00B2, // v_cmp_gt_u32 vcc, 50, v0
    0xBE94_246A, // s_and_saveexec_b64 s[20:21], vcc
    0xBF88_0018, // s_cbranch_execz label_0070
    0x7E20_0300, // v_mov_b32 v16, v0
    0xBE80_25C1, // s_or_saveexec_b64 s[0:1], -1
    0xD501_0002,
    0x0002_2080, // v_cndmask_b32 v2, 0, v16, s[0:1]
    0x4A04_04E9,
    0xDE54_C102, // v_add_nc_u32 v2, v2, v2 dpp8:[1,0,3,2,5,4,7,6]
    0x4A04_04E9,
    0x9360_1202, // v_add_nc_u32 v2, v2, v2 dpp8:[2,2,0,0,6,6,4,4]
    0x4A04_04E9,
    0x0009_2402, // v_add_nc_u32 v2, v2, v2 dpp8:[4,4,4,4,0,0,0,0]
    0x4A04_04FA,
    0xFF09_4002, // v_add_nc_u32 v2, v2, v2 row_mirror bound_ctrl:1
    0xD778_0003,
    0x0201_0102, // v_permlanex16_b32 v3, v2, 0, 0
    0x4A04_0503, // v_add_nc_u32 v2, v3, v2
    0xD760_0002,
    0x0001_4102, // v_readlane_b32 s2, v2, 32
    0x4A04_0402, // v_add_nc_u32 v2, s2, v2
    0xD760_000C,
    0x0001_0102, // v_readlane_b32 s12, v2, 0
    0xBEFE_0400, // s_mov_b64 exec, s[0:1]
    0x7E04_020C, // v_mov_b32 v2, s12
    0xDC70_8000,
    0x0006_0201, // global_store_dword v1, v2, s[6:7]
    0xBEFE_0414, // label_0070: s_mov_b64 exec, s[20:21]
    0xBF81_0000, // s_endpgm
];

/// `if (id >= 56) return;`, then [`COUNT_IN_BRANCH`]'s count: 20 in each even lane below 56.
const COUNT_AFTER_RETURN: &[u32] = &[
    0x3402_0082, // v_lshlrev_b32 v1, 2, v0
    0x7DA8_00B8, // v_cmpx_gt_u32 exec, 56, v0
    0xBF88_000B, // s_cbranch_execz label_0038
    0x3604_0081, // v_and_b32 v2, 1, v0
    0x7D84_0480, // v_cmp_eq_u32 vcc, 0, v2
    0xBE94_246A, // s_and_saveexec_b64 s[20:21], vcc
    0xBF88_0006, // s_cbranch_execz label_0034
    0xD4C4_000E,
    0x0002_00A8, // v_cmp_gt_u32 s[14:15], 40, v0
    0xBE90_100E, // s_bcnt1_i32_b64 s16, s[14:15]
    0x7E04_0210, // v_mov_b32 v2, s16
    0xDC70_8000,
    0x0006_0201, // global_store_dword v1, v2, s[6:7]
    0xBEFE_0414, // label_0034: s_mov_b64 exec, s[20:21]
    0xBF81_0000, // label_0038: s_endpgm
];

/// [`LOOP_BY_LANE`]'s loop with `if (id % 2 == 0)` inside, counting the lanes there: each even
/// lane adds the even lanes still in the loop on each of its passes, 32 where id % 4 is 0, 64 where
/// it is 2; odd lanes add nothing.
const BRANCH_IN_LOOP_BY_LANE: &[u32] = &[
    0x3402_0082, // v_lshlrev_b32 v1, 2, v0
    0x3606_0083, // v_and_b32 v3, 3, v0
    0x4A06_0681, // v_add_nc_u32 v3, 1, v3
    0x7E08_0280, // v_mov_b32 v4, 0
    0x360A_0081, // v_and_b32 v5, 1, v0
    0xBE8C_047E, // s_mov_b64 s[12:13], exec
    0x7D84_0A80, // label_0018: v_cmp_eq_u32 vcc, 0, v5
    0xBE94_246A, // s_and_saveexec_b64 s[20:21], vcc
    0xBF88_0004, // s_cbranch_execz label_0034
    0xD4C4_000E,
    0x0002_00C0, // v_cmp_gt_u32 s[14:15], 64, v0
    0xBE90_100E, // s_bcnt1_i32_b64 s16, s[14:15]
    0x4A08_0810, // v_add_nc_u32 v4, s16, v4
    0xBEFE_0414, // label_0034: s_mov_b64 exec, s[20:21]
    0x4E06_0681, // v_subrev_nc_u32 v3, 1, v3
    0x7D8A_0680, // v_cmp_ne_u32 vcc, 0, v3
    0x87FE_6A7E, // s_and_b64 exec, exec, vcc
    0xBF89_FFF4, // s_cbranch_execnz label_0018
    0xBEFE_040C, // s_mov_b64 exec, s[12:13]
    0xDC70_8000,
    0x0006_0401, // global_store_dword v1, v4, s[6:7]
    0xBF81_0000, // s_endpgm
];

/// What lane `id` stores at dword `n` of the output (`None`: nothing).
type Want = fn(u32, u32) -> Option<u32>;

/// Name, program, LDS bytes, the dwords to read back, and what each holds.
fn programs() -> [(&'static str, &'static [u32], u32, usize, Want); 6] {
    [
        ("a count in a branch", COUNT_IN_BRANCH, 0, 64, |id, n| {
            (n == id && id % 2 == 0).then_some(20)
        }),
        ("a loop by lane", LOOP_BY_LANE, 0, 128, |id, n| {
            if n == id {
                Some([64, 112, 144, 160][(id % 4) as usize])
            } else {
                (n == id + 64).then_some(4)
            }
        }),
        ("LDS in a branch", LDS_IN_BRANCH, 512, 64, |id, n| {
            (n == id && id % 2 == 0).then_some(id ^ 32)
        }),
        ("a sum in a branch", SUM_IN_BRANCH, 0, 64, |id, n| {
            (n == id && id < 50).then_some(1225)
        }),
        (
            "a count after a return",
            COUNT_AFTER_RETURN,
            0,
            64,
            |id, n| (n == id && id % 2 == 0 && id < 56).then_some(20),
        ),
        (
            "a branch in a loop by lane",
            BRANCH_IN_LOOP_BY_LANE,
            0,
            64,
            |id, n| (n == id).then_some([32, 0, 64, 0][(id % 4) as usize]),
        ),
    ]
}

/// A 64-lane compute program (one console wave) translated with `options`, from machine code of
/// our own in a container of no parts.
fn translate(words: &[u32], lds_bytes: u32, options: Options) -> Result<HostShader, ShaderError> {
    let native = NativeProgram {
        stage: HwStage::Cs,
        wave64: true,
        vgprs: 32,
        layout: vec![RootParam {
            root: RootType::Constants,
            table_offset_dwords: 0,
            sgprs: SGPRS,
            base_register: 0,
            user_data: 0,
            slot: 0,
        }],
        used: vec![],
        io: StageIo::Cs(ComputeIo {
            rsrc2: (lds_bytes / 512) << 15 | SGPRS << 1,
            threads: [64, 1, 1],
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
    emit_with(&c, &native, &p, &st, &io, &res, options)
}

/// Title address of the output, which no CPU pointer is at.
const VA: u64 = 0x12_3450_0000;
const POOL: usize = 0x1_0000;
const CANARY: u32 = 0xEEEE_EEEE;

#[test]
fn walked_programs_give_the_console_wave_s_results_on_the_gpu() {
    if !std::path::Path::new(victory_metal::msc::DEFAULT_DYLIB).exists()
        && std::env::var_os("VICTORY_MSC_DYLIB").is_none()
    {
        eprintln!("skipped: Metal Shader Converter not installed");
        return;
    }
    let mut t = Title::new(
        Metal::new().unwrap(),
        Arc::new(crate::native::memory::PoolTable::default()),
    );
    let mem = Alias::new(POOL).unwrap();
    let info = PoolInfo {
        id: 1,
        base: mem.primary as usize,
        len: POOL,
        memory_entry: None,
    };
    let m = t.host.memory();
    m.pool_created(&info);
    m.mapped(VA as usize, POOL, &info, 0);
    let (lo, hi) = (VA as u32, (VA >> 32) as u32);
    for (name, words, lds, dwords, want) in programs() {
        assert!(
            translate(
                words,
                lds,
                Options {
                    no_wave_walk: true,
                    ..shaders::options()
                }
            )
            .is_err(),
            "{name}: made without the walk"
        );
        let shader =
            translate(words, lds, shaders::options()).unwrap_or_else(|e| panic!("{name}: {e}"));
        let Pipe::Compute(pso) = t.compute_pipe(&shader, "walk") else {
            panic!("{name}: not made: {:?}", t.shaders)
        };
        // Several groups at once, so that host waves of several groups run side by side; each
        // stores the same.
        for groups in [1, 8] {
            // SAFETY: the pool's own mapping, POOL bytes; the GPU is idle.
            unsafe { std::slice::from_raw_parts_mut(mem.primary.cast::<u32>(), POOL / 4) }
                .fill(CANARY);
            let mut l = NativeList::default();
            l.set_compute_pipeline(&pso);
            l.set_compute_root_constants(0, &[0, 0, 0, 0, 0, 0, lo, hi]);
            l.dispatch(groups, 1, 1).unwrap();
            super::run(&t.host, &[&l]);
            // SAFETY: read after the fence wait.
            let out = unsafe { std::slice::from_raw_parts(mem.primary.cast::<u32>(), dwords) };
            for n in 0..dwords as u32 {
                let expect = (0..64).find_map(|id| want(id, n)).unwrap_or(CANARY);
                assert_eq!(
                    out[n as usize], expect,
                    "{name}, {groups} groups, dword {n}"
                );
            }
        }
    }
    assert_eq!(t.shaders.refusals(), 0, "{:?}", t.shaders.refused);
}
