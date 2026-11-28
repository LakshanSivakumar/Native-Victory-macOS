//! #531 on Metal: `v_cvt_f32_f64`, a double rounded to a float. Apple GPUs run no doubles, so the
//! translation rounds on the double's bits; these check it against Rust's own conversion (round
//! to nearest even, denormal results kept) on special values and on many random doubles, with
//! the source's neg and abs modifiers and the result's clamp. Skipped without the Metal Shader
//! Converter.

use super::addr_tests::{Fixture, VA_A, dispatch, halves, title, translate_io};
use crate::native::shaders;

/// Pool offset of `dst`, at `VA_A + 0x4_0000`: four rows of 64 results, then the 64 inputs at
/// byte 1024.
const DST: usize = 0x6_0000;
const INPUTS: usize = DST + 1024;

/// Each lane converts its own double from `INPUTS`, plain, negated, absolute and clamped (each
/// result in a row of `dst`); assembled with llvm-mc (gfx1030).
const PROGRAM: [u32; 22] = [
    0x3402_0083, // v_lshlrev_b32 v1, 3, v0
    0xDC34_8400,
    0x0202_0001, // global_load_dwordx2 v[2:3], v1, s[2:3] offset:1024
    0xBF8C_3F70, // s_waitcnt vmcnt(0)
    0x7E08_1F02, // v_cvt_f32_f64 v4, v[2:3]
    0xD58F_0005,
    0x2201_0102, // v_cvt_f32_f64_e64 v5, -v[2:3]
    0xD58F_0106,
    0x0201_0102, // v_cvt_f32_f64_e64 v6, |v[2:3]|
    0xD58F_8007,
    0x0201_0102, // v_cvt_f32_f64_e64 v7, v[2:3] clamp
    0x3402_0082, // v_lshlrev_b32 v1, 2, v0
    0xDC70_8000,
    0x0002_0401, // global_store_dword v1, v4, s[2:3]
    0xDC70_8100,
    0x0002_0501, // global_store_dword v1, v5, s[2:3] offset:256
    0xDC70_8200,
    0x0002_0601, // global_store_dword v1, v6, s[2:3] offset:512
    0xDC70_8300,
    0x0002_0701, // global_store_dword v1, v7, s[2:3] offset:768
    0xBF81_0000, // s_endpgm
    0xBF81_0000, // (padding)
];

/// What `v_cvt_f32_f64` gives for the double with bits `x`: Rust's conversion (IEEE round to
/// nearest even, denormals kept, too large an infinity), except that a NaN keeps the top 22 bits
/// of its payload and its sign, quieted (as RDNA2 converts NaNs).
fn want(x: u64) -> u32 {
    let d = f64::from_bits(x);
    if d.is_nan() {
        return (x >> 32) as u32 & 0x8000_0000 | 0x7FC0_0000 | (x >> 29) as u32 & 0x3F_FFFF;
    }
    (d as f32).to_bits()
}

/// `v_cvt_f32_f64 ... clamp`: the result clamped to [0, 1], NaN to 0 (RDNA2's clamp). A denormal
/// result and -0 clamp to +0: the clamp every float result shares (`write_f`, the host's
/// `saturate`) flushes the one and drops the other's sign, whatever the conversion gave.
fn want_clamped(x: u64) -> u32 {
    let f = f32::from_bits(want(x));
    if f.is_nan() || f.is_subnormal() || f <= 0.0 { 0 } else { f.min(1.0).to_bits() }
}

/// Doubles that sit on every edge of the conversion: zeros, the smallest and largest floats and
/// one bit either side, halfway cases that round to even either way, float denormals and their
/// halfway points, values that round up into the next binade or into infinity, infinities, NaNs.
fn edges() -> Vec<u64> {
    let mut v: Vec<u64> = vec![0, 1, (1 << 52) - 1, 1 << 52, 0x7FF0_0000_0000_0000, 0x7FF8_0000_0000_0000];
    v.push(0x7FF0_0000_0000_0001); // a signalling NaN with the smallest payload
    v.push(0x7FF4_5678_9ABC_DEF0); // a signalling NaN with a payload
    for f in [
        1.0f32,
        f32::MIN_POSITIVE,
        f32::MAX,
        f32::from_bits(1),
        f32::from_bits(0x7F_FFFF),
        f32::from_bits(0x3F80_0001),
        f32::from_bits(0x4B7F_FFFF),
    ] {
        let x = f64::from(f).to_bits();
        // The float itself, one double ulp either side, and the halfway points to its neighbours.
        v.extend([x, x - 1, x + 1]);
        let next = f64::from(f32::from_bits(f.to_bits() + 1)).to_bits();
        let mid = x + (next - x) / 2;
        v.extend([mid, mid - 1, mid + 1]);
    }
    // Halfway between f32::MAX and the next power of two: the first double that rounds to infinity.
    let max = f64::from(f32::MAX).to_bits();
    v.extend([max + (1 << 28), max + (1 << 28) - 1, (2.0f64.powi(128)).to_bits(), f64::MAX.to_bits()]);
    // Below half the smallest denormal: zero; at it: a tie to even (zero); just above it: 2^-149.
    let tiny = 2.0f64.powi(-150).to_bits();
    v.extend([tiny, tiny - 1, tiny + 1, 2.0f64.powi(-151).to_bits(), 1e-300f64.to_bits()]);
    v.extend([2.5f64.to_bits(), 1e10f64.to_bits(), std::f64::consts::PI.to_bits(), 0.1f64.to_bits()]);
    let negatives: Vec<u64> = v.iter().map(|x| x | 1 << 63).collect();
    v.extend(negatives);
    v
}

/// Doubles of every exponent a float can take and a little past (and some anywhere), with random
/// mantissas, from a fixed xorshift seed.
fn randoms(n: usize) -> Vec<u64> {
    let mut s = 0x9E37_79B9_7F4A_7C15u64;
    let mut next = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        s
    };
    (0..n)
        .map(|i| {
            let r = next();
            if i % 4 == 3 {
                return r; // any double at all
            }
            let e = 1023 - 160 + r % 300; // from below the denormals to past the largest float
            r & (1 << 63 | (1 << 52) - 1) | e << 52
        })
        .collect()
}

#[test]
fn a_double_rounds_to_the_nearest_float_with_its_modifiers() {
    let Some(mut t) = title() else { return };
    let f = Fixture::new(&t);
    let cs = translate_io(&PROGRAM, 5, 0, 64, shaders::options());
    let [c, d] = halves(VA_A + 0x4_0000);
    let mut inputs = edges();
    inputs.extend(randoms(64 * 64));
    for batch in inputs.chunks(64) {
        let mut xs = batch.to_vec();
        xs.resize(64, 0);
        f.put(INPUTS, &xs);
        dispatch(&mut t, &cs, &[0, 0, c, d, 0], false);
        assert_eq!(t.shaders.refusals(), 0, "{:?}", t.shaders.refused);
        let got = f.get(DST, 4 * 64);
        for (i, &x) in xs.iter().enumerate() {
            let rows = [want(x), want(x ^ 1 << 63), want(x & !(1 << 63)), want_clamped(x)];
            for (row, (name, w)) in ["plain", "neg", "abs", "clamp"].iter().zip(rows).enumerate() {
                assert_eq!(got[row * 64 + i], w, "{name} of {x:#018x} ({:e})", f64::from_bits(x));
            }
        }
    }
}
