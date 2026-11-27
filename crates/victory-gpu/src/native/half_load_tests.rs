//! 16-bit loads at a dword's last byte on Metal: `global_load_ushort`/`sshort` through title
//! memory and `buffer_load_ushort`/`sshort` through a raw buffer view, at byte offsets 0 to 3 of
//! a known pattern. A half at byte 3 has its low byte in one dword and its high byte in the next.
//! Skipped without the Metal Shader Converter.

use super::addr_tests::{Fixture, VA_A, dispatch, halves, title, translate_io, try_translate_native};
use crate::native::list::NativeList;
use crate::native::shaders;
use crate::native::title::Pipe;
use victory_backend::resources::HeapKind;
use victory_shader::native::{
    ComputeIo, HwStage, NativeProgram, ResourceType, RootParam, RootType, StageIo, UsedResource,
};

/// Pool offset of the destination `dst`, at `VA_A + 0x4_0000`.
const DST: usize = 0x6_0000;

/// The bytes 11 22 33 44 85 66 77 88 ..., as dwords.
const PATTERN: [u32; 4] = [0x4433_2211, 0x8877_6685, 0xAABB_CCDD, 0x1122_3344];

/// What the loads at byte offsets 0, 1, 2 and 3 (unsigned) and 3 (signed) must give: the two
/// bytes from that offset, little-endian.
const WANT: [u32; 5] = [0x2211, 0x3322, 0x4433, 0x8544, 0xFFFF_8544];

fn check(rows: &[u32]) {
    for (row, want) in WANT.iter().enumerate() {
        assert_eq!(
            rows[64 * row..64 * (row + 1)],
            [*want; 64],
            "{} at byte offset {}",
            if row == 4 { "sshort" } else { "ushort" },
            if row == 4 { 3 } else { row }
        );
    }
}

/// Case A: `global_load_ushort` at byte offsets 0 to 3 of `src` (`s[0:1]`) and `global_load_sshort`
/// at 3, each result stored to its own 64-dword row of `dst` (`s[2:3]`).
#[test]
fn a_half_at_a_dwords_last_byte_reads_the_next_dword_in_a_global_load_on_the_gpu() {
    let w = [
        0x3406_0082, // v_lshlrev_b32_e32 v3, 2, v0
        0x7E02_0280, // v_mov_b32_e32 v1, 0
        0xDC28_8000, 0x0400_0001, // global_load_ushort v4, v1, s[0:1]
        0xDC28_8001, 0x0500_0001, // global_load_ushort v5, v1, s[0:1] offset:1
        0xDC28_8002, 0x0600_0001, // global_load_ushort v6, v1, s[0:1] offset:2
        0xDC28_8003, 0x0700_0001, // global_load_ushort v7, v1, s[0:1] offset:3
        0xDC2C_8003, 0x0800_0001, // global_load_sshort v8, v1, s[0:1] offset:3
        0xBF8C_3F70, // s_waitcnt vmcnt(0)
        0xDC70_8000, 0x0002_0403, // global_store_dword v3, v4, s[2:3]
        0xDC70_8100, 0x0002_0503, // global_store_dword v3, v5, s[2:3] offset:256
        0xDC70_8200, 0x0002_0603, // global_store_dword v3, v6, s[2:3] offset:512
        0xDC70_8300, 0x0002_0703, // global_store_dword v3, v7, s[2:3] offset:768
        0xDC70_8400, 0x0002_0803, // global_store_dword v3, v8, s[2:3] offset:1024
        0xBF81_0000, // s_endpgm
    ];
    let Some(mut t) = title() else { return };
    let f = Fixture::new(&t);
    f.put(0x2_0000, &PATTERN);
    let cs = translate_io(&w, 4, 0, 64, shaders::options());
    let [a, b] = halves(VA_A);
    let [c, d] = halves(VA_A + 0x4_0000);
    dispatch(&mut t, &cs, &[a, b, c, d], false);
    assert_eq!(t.shaders.refusals(), 0, "{:?}", t.shaders.refused);
    check(&f.get(DST, 5 * 64));
}

/// Case B: `buffer_load_ushort` at byte offsets 0 to 3 (`offen` with a VGPR of 0, the rest in
/// the instruction's offset) and `buffer_load_sshort` at 3, through a raw UAV the program loads
/// the descriptor of from the title's table, each result stored to its own 64-dword row of a
/// second raw UAV.
#[test]
fn a_half_at_a_dwords_last_byte_reads_the_next_dword_in_a_buffer_load_on_the_gpu() {
    const BUFFER: &[u32] = &[
        0x3402_0082, // v_lshlrev_b32_e32 v1, 2, v0
        0xBE85_0380, // s_mov_b32 s5, 0
        0xF408_0202, 0xFA00_0000, // s_load_dwordx4 s[8:11], s[4:5], 0x0
        0xF408_0302, 0xFA00_0020, // s_load_dwordx4 s[12:15], s[4:5], 0x20
        0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
        0x7E04_0280, // v_mov_b32_e32 v2, 0
        0xE028_1000, 0x8002_0302, // buffer_load_ushort v3, v2, s[8:11], 0 offen
        0xE028_1001, 0x8002_0402, // buffer_load_ushort v4, v2, s[8:11], 0 offen offset:1
        0xE028_1002, 0x8002_0502, // buffer_load_ushort v5, v2, s[8:11], 0 offen offset:2
        0xE028_1003, 0x8002_0602, // buffer_load_ushort v6, v2, s[8:11], 0 offen offset:3
        0xE02C_1003, 0x8002_0702, // buffer_load_sshort v7, v2, s[8:11], 0 offen offset:3
        0xBF8C_3F70, // s_waitcnt vmcnt(0)
        0xE070_1000, 0x8003_0301, // buffer_store_dword v3, v1, s[12:15], 0 offen
        0xE070_1100, 0x8003_0401, // buffer_store_dword v4, v1, s[12:15], 0 offen offset:256
        0xE070_1200, 0x8003_0501, // buffer_store_dword v5, v1, s[12:15], 0 offen offset:512
        0xE070_1300, 0x8003_0601, // buffer_store_dword v6, v1, s[12:15], 0 offen offset:768
        0xE070_1400, 0x8003_0701, // buffer_store_dword v7, v1, s[12:15], 0 offen offset:1024
        0xBF81_0000, // s_endpgm
    ];
    let Some(mut t) = title() else { return };
    let f = Fixture::new(&t);
    f.put(0x2_1000, &PATTERN);
    f.put(0x2_2000, &[0u32; 5 * 64]);
    // The title's heap: two raw UAVs of 4 KiB, the source and the destination.
    let records = [0u32; 8 * 2];
    let start = records.as_ptr() as u64;
    let heap = t.host.heap(HeapKind::Resource, start, 2).unwrap();
    let (rs, rd) = (0x57D_0011usize, 0x57D_0012usize);
    t.host.register_buffer(rs, VA_A + 0x1000, 4096);
    t.host.register_buffer(rd, VA_A + 0x2000, 4096);
    let mut uav = [0u32; 10];
    (uav[1], uav[4], uav[8]) = (1, 1024, 1);
    crate::descriptor::create_uav(std::ptr::null(), rs as _, std::ptr::null(), &uav, start);
    crate::descriptor::create_uav(std::ptr::null(), rd as _, std::ptr::null(), &uav, start + 32);
    assert!(t.host.mirror(start).unwrap() && t.host.mirror(start + 32).unwrap());
    let used = |table_offset_dwords| UsedResource {
        kind: ResourceType::Uav,
        root: RootType::Table,
        table_offset_dwords,
        lower_bound: 0,
        stride: 0,
        dwords: 4,
        user_data: 4,
        slot: 0,
    };
    let native = NativeProgram {
        stage: HwStage::Cs,
        wave64: false,
        vgprs: 16,
        layout: vec![RootParam {
            root: RootType::Table,
            table_offset_dwords: 0,
            sgprs: 1,
            base_register: 0,
            user_data: 4,
            slot: 0,
        }],
        used: vec![used(0), used(8)],
        io: StageIo::Cs(ComputeIo { rsrc2: 5 << 1, threads: [32, 1, 1] }),
        code: BUFFER.iter().flat_map(|w| w.to_le_bytes()).collect(),
    };
    let cs = try_translate_native(&native, shaders::options()).expect("emit");
    let Pipe::Compute(pso) = t.compute_pipe(&cs, "half") else {
        panic!("not made: {:?}", t.shaders)
    };
    let mut l = NativeList::default();
    l.set_heaps(Some(&heap), None);
    l.set_compute_pipeline(&pso);
    l.set_compute_root_table(0, &heap, 0);
    l.dispatch(1, 1, 1).unwrap();
    super::run(&t.host, &[&l]);
    assert_eq!(t.shaders.refusals(), 0, "{:?}", t.shaders.refused);
    // 32 lanes ran (a wave of 32 in a group of 32): rows of 64 dwords, the lanes' first 32.
    let got = f.get(0x2_2000, 5 * 64);
    for (row, want) in WANT.iter().enumerate() {
        assert_eq!(
            got[64 * row..64 * row + 32],
            [*want; 32],
            "{} at byte offset {}",
            if row == 4 { "sshort" } else { "ushort" },
            if row == 4 { 3 } else { row }
        );
    }
}
