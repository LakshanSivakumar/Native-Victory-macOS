//! #256 on Metal: a guarded VGPR write keeps the old value in the lanes outside its guard. When
//! that old value has no DXIL form (here the `s_memtime` timestamp) and a lane outside the guard
//! reads it, the program is refused; on main it translated and those lanes read 0. When every lane
//! that kept it overwrites it before the read, the program translates and runs with the lanes'
//! own values. Skipped without the Metal Shader Converter.

use super::addr_tests::{Fixture, VA_A, dispatch, halves, title, translate_io, try_translate_wave};
use crate::native::shaders;

/// Pool offset of `dst`, at `VA_A + 0x4_0000`.
const DST: usize = 0x6_0000;

/// `if (v0 < 40) v2 = timestamp; else v2 = 0;` with EXEC restored after it. The else arm's write
/// keeps the timestamp in lanes 0..39.
const IF_TIMESTAMP_ELSE_0: [u32; 9] = [
    0xF490_0180,
    0x0000_0000, // s_memtime s[6:7]
    0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
    0x7D88_00A8, // v_cmp_gt_u32 vcc, 40, v0
    0xBE84_246A, // s_and_saveexec_b64 s[4:5], vcc
    0x7E04_0206, // v_mov_b32 v2, s6
    0x8AFE_7E04, // s_andn2_b64 exec, s[4:5], exec
    0x7E04_0280, // v_mov_b32 v2, 0
    0xBEFE_0404, // s_mov_b64 exec, s[4:5]
];

/// Every lane stores `v2` at `dst[lane]`.
const STORE_V2: [u32; 4] = [
    0x3402_0082, // v_lshlrev_b32 v1, 2, v0
    0xDC70_8000,
    0x0002_0201, // global_store_dword v1, v2, s[2:3]
    0xBF81_0000, // s_endpgm
];

fn program(between: &[u32]) -> Vec<u32> {
    [&IF_TIMESTAMP_ELSE_0[..], between, &STORE_V2].concat()
}

/// A wave64 group of 64; `dst[0..64]`.
fn run(words: &[u32]) -> Vec<u32> {
    let Some(mut t) = title() else { return Vec::new() };
    let f = Fixture::new(&t);
    let cs = translate_io(words, 5, 0, 64, shaders::options());
    let [c, d] = halves(VA_A + 0x4_0000);
    dispatch(&mut t, &cs, &[0, 0, c, d, 0], false);
    assert_eq!(t.shaders.refusals(), 0, "{:?}", t.shaders.refused);
    f.get(DST, 64)
}

#[test]
fn a_kept_value_without_a_dxil_form_read_outside_the_guard_is_refused() {
    // Lanes 0..39 read the timestamp the else arm kept. On main this translated and those lanes
    // stored 0 (all 64 lanes read 0 on Metal).
    let Err(why) = try_translate_wave(&program(&[]), 5, 0, 64, true, shaders::options()) else {
        panic!("translated");
    };
    assert!(why.to_string().contains("a scalar memory read as data"), "{why}");
}

#[test]
fn a_kept_value_its_lanes_overwrite_before_the_read_runs_with_their_values() {
    // Lanes 0..39 (the same `vcc`) overwrite the kept timestamp with 5; lanes 40..63 keep the 0.
    let got = run(&program(&[
        0xBE84_246A, // s_and_saveexec_b64 s[4:5], vcc
        0x7E04_0285, // v_mov_b32 v2, 5
        0xBEFE_0404, // s_mov_b64 exec, s[4:5]
    ]));
    if got.is_empty() {
        return;
    }
    assert_eq!(got[..40], [5; 40]);
    assert_eq!(got[40..], [0; 24]);

    // An unguarded write over it (a7fc47af's own case).
    let got = run(&program(&[0x7E04_0287])); // v_mov_b32 v2, 7
    assert_eq!(got, [7; 64]);
}
