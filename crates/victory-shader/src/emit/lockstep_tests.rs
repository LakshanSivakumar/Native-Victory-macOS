//! Pins for the lockstep rewrites of `lockstep.rs`, on hand-assembled
//! programs: `branches` (a mask branch becomes a 64-bit compare with 0 and
//! an SCC branch of the same distance) and `lds_barriers` (a group barrier
//! around each LDS write, none around reads).

use super::lockstep::{branches, lds_barriers};
use crate::ir::{self, Exit, Reg};

const S_ENDPGM: u32 = 0xBF81_0000;

fn build_words(words: &[u32]) -> ir::Program {
    let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    ir::build(&code, true, 256).unwrap()
}

#[test]
fn an_execz_branch_becomes_a_compare_with_zero_and_an_scc_branch() {
    // s_cbranch_execz +1: while EXEC is zero, skip one instruction.
    let p = build_words(&[0xBF88_0001, 0xBF80_0000, S_ENDPGM]);
    let q = branches(&p);
    let block = q
        .blocks
        .iter()
        .find(|b| matches!(b.exit, Exit::Branch { .. }))
        .expect("a branch block");
    let names: Vec<_> = block.insts.iter().map(|i| i.name()).collect();
    assert_eq!(names, ["s_cmp_eq_u64", "s_cbranch_scc1"], "{names:?}");
    let (cmp, br) = (&block.insts[0], &block.insts[1]);
    // The compare reads EXEC as a pair and writes SCC; the branch reads that SCC
    // and keeps the original distance.
    assert_eq!(
        cmp.reads,
        [
            (Reg::S(ir::EXEC), cmp.reads[0].1),
            (Reg::S(ir::EXEC + 1), cmp.reads[1].1)
        ]
    );
    assert_eq!(cmp.writes.len(), 1);
    assert_eq!(cmp.writes[0].0, Reg::Scc);
    assert_eq!(br.reads, [(Reg::Scc, cmp.writes[0].1)]);
    assert_eq!(br.machine.words[0] & 0xFFFF, 1, "the branch's distance");
    assert_eq!(br.machine.words[0] >> 16 & 0x7F, 0x05, "s_cbranch_scc1");
}

#[test]
fn execnz_vccz_and_vccnz_use_the_matching_compare() {
    // s_cbranch_execnz -> s_cmp_lg_u64 exec, 0; s_cbranch_vccz -> s_cmp_eq_u64 vcc, 0;
    // s_cbranch_vccnz -> s_cmp_lg_u64 vcc, 0. (SOPP opcodes 0x009, 0x006, 0x007.)
    for (word, cmp_op, reg) in [
        (0xBF89_0000u32, 0x13u32, ir::EXEC),
        (0xBF86_0000, 0x12, ir::VCC),
        (0xBF87_0000, 0x13, ir::VCC),
    ] {
        let p = build_words(&[word | 1, 0xBF80_0000, S_ENDPGM]); // branch over the nop
        let q = branches(&p);
        let block = q
            .blocks
            .iter()
            .find(|b| matches!(b.exit, Exit::Branch { .. }))
            .expect("a branch block");
        assert_eq!(
            block.insts[0].name(),
            if cmp_op == 0x12 {
                "s_cmp_eq_u64"
            } else {
                "s_cmp_lg_u64"
            }
        );
        assert_eq!((block.insts[0].machine.words[0] >> 16) & 0x7F, cmp_op);
        assert_eq!((block.insts[0].machine.words[0]) & 0xFF, u32::from(reg));
        assert_eq!(block.insts[1].name(), "s_cbranch_scc1");
    }
}

#[test]
fn a_branch_on_scc_or_a_non_mask_branch_is_untouched() {
    // s_cbranch_scc1 branches on SCC, not on a mask: kept as it is.
    let p = build_words(&[0xBF85_0001, 0xBF80_0000, S_ENDPGM]);
    let q = branches(&p);
    let block = q
        .blocks
        .iter()
        .find(|b| matches!(b.exit, Exit::Branch { .. }))
        .expect("a branch block");
    assert_eq!(block.insts.len(), 1);
    assert_eq!(block.insts[0].name(), "s_cbranch_scc1");
    assert_eq!(block.insts[0].machine.words[0], 0xBF85_0001);
}

#[test]
fn lds_barriers_wrap_each_write_and_no_read() {
    // ds_write_b32 v1, v2 offset 0x40; ds_read_b32 v3, v1 offset 0x10.
    // DS encoding: 0xD8.. | op<<18 | gds<<17 | off1<<8 | off0; w1 = addr | data<<8 | dst<<24.
    let write = [0xD800_0000u32 | 0x00D << 18 | 0x40, 1 | 2 << 8];
    let read = [0xD800_0000 | 0x036 << 18 | 0x10, 1 | 3 << 24];
    let p = build_words(&[write[0], write[1], read[0], read[1], S_ENDPGM]);
    let at = lds_barriers(&p);
    let block = 0;
    assert!(at.contains(&(block, 0)), "a barrier before the write");
    assert!(at.contains(&(block, 1)), "a barrier after the write");
    assert!(!at.contains(&(block, 2)), "none around the read");
    assert!(!at.contains(&(block, 3)), "none around the read");
    assert_eq!(at.len(), 2);
}
