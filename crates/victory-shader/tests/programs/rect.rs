// Hand-assembled test programs, shared by translator and native host tests. No game code.
use victory_shader::dxbc::{self, FourCC};

fn vop1(op: u32, dst: u32, src: u32) -> u32 {
    0x7e00_0000 | dst << 17 | op << 9 | src
}
fn vop2(op: u32, dst: u32, a: u32, b: u32) -> u32 {
    op << 25 | dst << 17 | b << 9 | a
}
fn vop3(op: u32, dst: u32, a: u32, b: u32, c: u32) -> [u32; 2] {
    [0xd400_0000 | op << 16 | dst, a | b << 9 | c << 18]
}

/// No vertex buffers. r = id / 3, c = id % 3, using the exact integer reciprocal.
/// p0 = (r - .875, .875 - (instance & 1)), p1 = p0 + (0, -.75), p2 = p0 + (.75, 0).
/// TEXCOORD0 is (c == 2, c == 1, .25, 1). ID 3 is another tile, never p3.
pub fn vertex_words() -> Vec<u32> {
    let mut w = Vec::new();
    w.extend(vop3(0x16a, 4, 0x100, 255, 0)); // v_mul_hi_u32 v4, v0, literal
    w.push(0xaaaa_aaab);
    w.push(vop2(0x16, 4, 129, 4)); // v_lshrrev_b32 v4, 1, v4: rectangle
    w.extend(vop3(0x169, 5, 131, 0x104, 0)); // v_mul_lo_u32 v5, 3, v4
    w.push(vop2(0x26, 5, 0x100, 5)); // v_sub_nc_u32 v5, v0, v5: corner
    w.extend(vop3(0x0c2, 0, 130, 0x105, 0)); // v_cmp_eq_u32 s[0:1], 2, v5
    w.extend(vop3(0x101, 6, 128, 242, 0)); // v_cndmask_b32 v6, 0, 1.0, s[0:1]
    w.extend(vop3(0x0c2, 0, 129, 0x105, 0)); // v_cmp_eq_u32 s[0:1], 1, v5
    w.extend(vop3(0x101, 7, 128, 242, 0)); // v_cndmask_b32 v7, 0, 1.0, s[0:1]
    w.push(vop1(6, 4, 0x104)); // v_cvt_f32_u32 v4, v4
    w.extend([vop2(3, 8, 255, 4), (-0.875f32).to_bits()]); // x0
    w.extend([vop2(8, 9, 255, 6), 0.75f32.to_bits()]); // dx = .75 * u
    w.push(vop2(3, 8, 0x108, 9)); // x = x0 + dx
    w.push(vop2(0x1b, 10, 129, 3)); // v_and_b32 v10, 1, v3: instance
    w.push(vop1(6, 10, 0x10a));
    w.extend([vop2(8, 9, 255, 7), 0.75f32.to_bits()]); // dy = .75 * v
    w.push(vop2(3, 9, 0x10a, 9)); // instance + dy
    w.extend([vop2(4, 9, 255, 9), 0.875f32.to_bits()]); // y = .875 - (instance + dy)
    w.push(vop1(1, 10, 240)); // z = .5
    w.push(vop1(1, 11, 242)); // w = 1
    w.extend([vop1(1, 12, 255), 0.25f32.to_bits()]);
    w.extend([0xf800_020f, 6 | 7 << 8 | 12 << 16 | 11 << 24]); // exp param0
    w.extend([0xf800_08cf, 8 | 9 << 8 | 10 << 16 | 11 << 24]); // exp pos0 done
    w.push(0xbf81_0000);
    w
}

fn signature(elements: &[(&str, u32, u32)]) -> Vec<u8> {
    let mut out = vec![0; 8 + 32 * elements.len()];
    out[..4].copy_from_slice(&(elements.len() as u32).to_le_bytes());
    out[4..8].copy_from_slice(&8u32.to_le_bytes());
    for (i, &(name, system, row)) in elements.iter().enumerate() {
        let name_at = out.len() as u32;
        out.extend(name.as_bytes());
        out.push(0);
        let at = 8 + 32 * i;
        for (off, value) in [(4, name_at), (12, system), (16, 3), (20, row)] {
            out[at + off..at + off + 4].copy_from_slice(&value.to_le_bytes());
        }
        out[at + 24] = 15;
    }
    out
}

/// A described (FSSM, not guessed FSPG) shader with an empty input layout.
pub fn vertex_container() -> Vec<u8> {
    let code: Vec<_> = vertex_words()
        .iter()
        .flat_map(|w| w.to_le_bytes())
        .collect();
    let mut payload = vec![0; 0x68];
    payload[..8].copy_from_slice(&6u64.to_le_bytes());
    for at in [8, 16, 24, 32, 40, 48, 72, 96] {
        payload[at..at + 8].copy_from_slice(&0xdead_beef_dead_beefu64.to_le_bytes());
    }
    let record = payload.len();
    payload[40..48].copy_from_slice(&(record as u64).to_le_bytes());
    payload.resize(record + 0x54 + 0x18c, 0);
    let code_at = payload.len() as u64;
    payload[record..record + 8].copy_from_slice(&code_at.to_le_bytes());
    payload[record + 8..record + 16].copy_from_slice(&(code.len() as u64).to_le_bytes());
    payload[record + 0x50..record + 0x54].copy_from_slice(&16u32.to_le_bytes());
    let data = record + 0x54;
    payload[data + 0xc8..data + 0xce].copy_from_slice(&[6, 0, 0, 0, 0, 0]);
    payload[data + 0x188..data + 0x18c].copy_from_slice(&1u32.to_le_bytes());
    payload.extend(code);
    let mut fssm = b"FSSM".to_vec();
    fssm.extend(0x101u32.to_le_bytes());
    fssm.extend((payload.len() as u64).to_le_bytes());
    fssm.extend(payload);
    let packed = miniz_oxide::deflate::compress_to_vec_zlib(&fssm, 6);
    let mut xbch = b"XBCH".to_vec();
    xbch.extend(3u32.to_le_bytes());
    xbch.extend((packed.len() as u64).to_le_bytes());
    xbch.extend((fssm.len() as u64).to_le_bytes());
    xbch.extend(packed);
    let mut part = vec![0; 24];
    part.extend(b"XDAT");
    for w in [1, 0x24, 0x0d, 48, xbch.len() as u32] {
        part.extend(w.to_le_bytes());
    }
    part.extend(xbch);
    dxbc::build(&[
        (FourCC::DXIL, &part),
        (FourCC::ISG1, &signature(&[])),
        (
            FourCC::OSG1,
            &signature(&[("TEXCOORD", 0, 0), ("SV_Position", 1, 1)]),
        ),
    ])
}

#[allow(dead_code)]
pub fn pixel_container() -> Vec<u8> {
    let words = [
        0xc808_0000u32,
        0xc809_0001, // v_interp_p1/p2_f32 v2, attr0.x
        0xc80c_0100,
        0xc80d_0101, // v_interp_p1/p2_f32 v3, attr0.y
        0x7e08_02ff,
        0x3e80_0000, // v_mov_b32 v4, .25
        0x7e0a_02f2, // v_mov_b32 v5, 1.0
        0xf800_180f,
        0x0504_0302, // exp mrt0 v2, v3, v4, v5 done vm
        0xbf81_0000,
    ];
    let bytes: Vec<_> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    let mut meta = [0; 16];
    meta[12] = 0x40;
    victory_shader::emit::graphics::container(victory_shader::HwStage::Ps, &bytes, &meta).unwrap()
}
