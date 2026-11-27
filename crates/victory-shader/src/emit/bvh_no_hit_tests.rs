//! CPU-only translation checks: no BVH node reads when the native builder is suppressed.
use super::{Options, translate_with};
use crate::native::tests::{fspx, fspx_resource};
use crate::{FourCC, dxbc};

fn program(wide: bool, node: u32) -> Vec<u8> {
    let mut code = vec![
        0xBE85_0380, // s5 = 0; output UAV from the table at s4
        0xF408_0302,
        0xFA00_0000, // s_load_dwordx4 s[12:15], s[4:5], 0
        0xBE88_0380, // BVH descriptor s[8:11], base zero, sorted, no bound, barycentrics
        0xBE89_03FF,
        0x8000_0000,
        0xBE8A_03C1,
        0xBE8B_03FF,
        0x8900_03FF,
    ];
    for reg in 0u32..12 {
        code.extend([0x7E00_02FF | reg << 17, if reg == 0 { node } else { 0 }]);
    }
    let rest = 20 << 8 | (8 >> 2) << 16; // result v[20:23], descriptor s[8:11]
    if wide {
        code.extend([0xF19C_9F07, rest, 0x0403_0201, 0x0807_0605, 0x000B_0A09]);
    } else {
        code.extend([0xF198_9F01, rest]);
    }
    code.extend([
        0xE078_0000,
        0x8003_1400, // buffer_store_dwordx4 v[20:23], off, s[12:15], 0
        0xBF81_0000,
    ]);
    let code: Vec<u8> = code.iter().flat_map(|w| w.to_le_bytes()).collect();
    let table = [fspx_resource(3, 3, 0, 0, 4, 0)];
    let part = fspx(&code, 5 << 1, [32, 1, 1], true, &table);
    dxbc::build(&[(FourCC::ACCC, &part)])
}

#[test]
fn both_bvh_instructions_emit_no_node_reads_in_no_hit_mode() {
    for wide in [false, true] {
        for node_type in 0..8 {
            let bytes = program(wide, 0xffff_ff00 | node_type);
            let options = Options {
                wave32_host: true,
                zero_copy_memory: true,
                ..Default::default()
            };
            let on = translate_with(&bytes, options).unwrap();
            let off = translate_with(
                &bytes,
                Options {
                    bvh_no_hit: true,
                    ..options
                },
            )
            .unwrap();
            assert!(on.memory, "full intersection must read nodes");
            assert!(!off.memory, "wide={wide} type={node_type}: no node reads");
            assert!(!off.memory_writes);
            assert_ne!(off.dxil, on.dxil);
            assert!(!off.dxil.is_empty());
        }
    }
}
