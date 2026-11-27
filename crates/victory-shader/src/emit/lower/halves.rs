//! Wave64 on a 32-lane host ([`Options::wave32_host`](crate::emit::Options)):
//! Apple's Metal Shader Converter takes `WaveSize(32)` only, and Apple GPUs
//! run 32-lane SIMD groups.
//!
//! One thread stays one console lane. A console wave of 64 lanes is two host
//! waves of the group (flat ids `64n..64n+31` and `64n+32..64n+63`), and a
//! thread's console lane is its group index modulo 64 (`console_flat`, which
//! also makes the host lane the console's). What stays inside 32 lanes is
//! the host's own: DPP rows and quads, `ds_swizzle` (its masks keep bit 5),
//! `ds_bpermute_b32`, `v_mbcnt`, `v_writelane`, lane-bit tests of masks held as numbers and
//! `v_readfirstlane` of a value the same in every lane of the console wave
//! ([`Lower::wave_uniform`]). Any other `v_readfirstlane` reads the first lane of
//! the console's 64-lane EXEC across the halves, or is refused (#254,
//! [`Lower::split_first_lane`]).
//!
//! What reaches across the two halves goes through group-shared memory (the
//! `xchg` array, one dword per thread) between two group barriers: a ballot
//! of a mask read as data, `v_readlane_b32`, `v_readfirstlane_b32` of a value
//! that differs between lanes, and the reductions and prefixes
//! of whole-wave sections. A ballot is each
//! half's 32 bits, joined; a lane read is a load at the source lane's slot;
//! a reduction is each half's result, combined; a prefix combines the lower
//! half's total with each upper lane's prefix. Lanes the group lacks (a group
//! whose lanes are not a multiple of 64) write no slot: their ballot bits are
//! 0, a missing upper half adds nothing to a reduction, and a lane read of
//! one gives 0.
//!
//! Group barriers need every thread of the group. Such an operation is
//! therefore lowered only in a block each thread runs exactly once
//! ([`once_blocks`](super::once_blocks)), or, in a group that is one console
//! wave, in one reached by uniform (SCC) branches alone and outside any loop
//! ([`wave_blocks`](super::wave_blocks)): both halves then arrive together.
//! A block outside any loop that holds one of the title's own `s_barrier`s
//! counts too ([`barrier_blocks`](super::barrier_blocks)): the console's
//! barrier waits for every wave that has not ended, so the title brings every
//! live wave there together, and the host's group barrier for it already
//! relies on that (not past a branch on an SCC from a `v_readfirstlane`,
//! unless of an LDS word read alike in a barrier phase: `barrier_reads`).
//! An SCC computed from a `v_readfirstlane` is not uniform here: each half
//! reads its own first lane. Elsewhere (in a branch of EXEC, a
//! per-lane loop) the translation is refused. Cost: two barriers and a
//! shared-memory round trip per such operation, 4 bytes per thread of
//! group-shared memory in all, and `WaveSize(32)` in place of 64.
//!
//! LDS that lanes of one console wave share without a barrier (the console
//! runs the wave in lockstep) gets the group barriers `narrow_waves` adds
//! ([`split_wave_barriers`](super::split_wave_barriers)), or is refused.

use super::{Lower, Map, Set, Why};
use crate::emit::bitcode::{Attr, Bin, V, pred};
use crate::emit::shader::flags;
use crate::emit::wave::Op;

impl Lower<'_> {
    /// A group barrier with a shared-memory fence in DXIL block `at`.
    fn xchg_barrier(&mut self, at: usize) {
        let mode = self.s.m.i32(1 | 8);
        let void = self.s.t.void;
        self.s
            .op(at, "barrier", Attr::NoDuplicate, void, 80, &[mode]);
    }

    /// Whether every thread of the group (both halves of a console wave)
    /// reaches IR block `ir` together.
    fn exchange_site(&self, ir: usize) -> Result<(), Why> {
        // In lockstep every thread reaches every block together (`lockstep.rs`).
        if self.once[ir] || self.x.st.lockstep {
            return Ok(());
        }
        Err(super::DIVERGENT_EXCHANGE)
    }

    /// The group-shared slot `index`.
    fn xchg_slot(&mut self, at: usize, index: V) -> V {
        let g = self.s.xchg();
        self.s.m.element(at, g, index)
    }

    /// Whether exchange slot `index`, which holds lanes `index << shift` on, is a thread's, where
    /// the group's lanes are not a multiple of 64 (its last console wave lacks lanes, or a whole
    /// host wave, and no thread writes their slots). `None`: every slot is a thread's.
    fn xchg_written(&mut self, at: usize, index: V, shift: u32) -> Option<V> {
        let lanes: u32 = self.s.threads.map_or(0, |t| t.iter().product());
        if lanes.is_multiple_of(64) {
            return None;
        }
        let k = self.s.m.i32(shift);
        let first = self.s.m.bin(at, Bin::Shl, index, k);
        let n = self.s.m.i32(lanes);
        Some(self.s.m.cmp(at, pred::IULT, first, n))
    }

    /// The low and high words of the current host wave's active ballot.
    pub(super) fn host_ballot(&mut self, at: usize, on: V) -> (V, V) {
        let i32t = self.s.t.i32;
        let four = self.s.m.named_struct("dx.types.fouri32", vec![i32t; 4]);
        let ballot = self
            .s
            .op(at, "waveActiveBallot", Attr::NoUnwind, four, 116, &[on]);
        self.s.flags |= flags::WAVE_OPS;
        (
            self.s.m.extract(at, ballot, 0),
            self.s.m.extract(at, ballot, 1),
        )
    }

    /// The console's ballot of `on` in DXIL block `at` (IR block `ir`): the
    /// low and high dwords.
    pub(super) fn wide_ballot(&mut self, at: usize, ir: usize, on: V) -> Result<(V, V), Why> {
        let (x, y) = self.host_ballot(at, on);
        if !self.halves {
            return Ok((x, y));
        }
        self.exchange_site(ir)?;
        let f = self.console_flat();
        let (five, six, one) = (self.s.m.i32(5), self.s.m.i32(6), self.s.m.i32(1));
        let mine = self.s.m.bin(at, Bin::LShr, f, five);
        let slot = self.xchg_slot(at, mine);
        self.s.m.store(at, slot, x);
        self.xchg_barrier(at);
        let pair = self.s.m.bin(at, Bin::LShr, f, six);
        let lo_at = self.s.m.bin(at, Bin::Shl, pair, one);
        let hi_at = self.s.m.bin(at, Bin::Add, lo_at, one);
        let (lo_p, hi_p) = (self.xchg_slot(at, lo_at), self.xchg_slot(at, hi_at));
        let (lo, hi) = (self.s.m.load(at, lo_p), self.s.m.load(at, hi_p));
        self.xchg_barrier(at);
        // A missing upper half has no lanes on.
        let hi = match self.xchg_written(at, hi_at, 5) {
            Some(there) => {
                let z = self.s.m.i32(0);
                self.s.m.select(at, there, hi, z)
            }
            None => hi,
        };
        Ok((lo, hi))
    }

    /// `x` of console lane `lane` (the low six bits count).
    pub(super) fn wide_read_lane(&mut self, x: V, lane: V) -> Result<V, Why> {
        if !self.host32 {
            return Ok(self.read_lane(x, lane));
        }
        let c = self.cur;
        let (k31, k32, z) = (self.s.m.i32(31), self.s.m.i32(32), self.s.m.i32(0));
        if !self.halves {
            // A group of 32 lanes or fewer: no thread holds a lane above 31.
            self.s.flags |= flags::WAVE_OPS;
            let low = self.s.m.bin(c, Bin::And, lane, k31);
            let r = self.read_lane(x, low);
            let there = self.s.m.cmp(c, pred::IULT, lane, k32);
            return Ok(self.s.m.select(c, there, r, z));
        }
        self.exchange_site(self.cur_ir)?;
        self.s.flags |= flags::WAVE_OPS;
        let f = self.console_flat();
        let slot = self.xchg_slot(c, f);
        self.s.m.store(c, slot, x);
        self.xchg_barrier(c);
        let (k63, not63) = (self.s.m.i32(63), self.s.m.i32(!63));
        let wave = self.s.m.bin(c, Bin::And, f, not63);
        let lane = self.s.m.bin(c, Bin::And, lane, k63);
        let src = self.s.m.bin(c, Bin::Or, wave, lane);
        let p = self.xchg_slot(c, src);
        let r = self.s.m.load(c, p);
        self.xchg_barrier(c);
        // Guess: a lane the group lacks reads as 0 (the console reads whatever its VGPR holds).
        Ok(match self.xchg_written(c, src, 0) {
            Some(there) => self.s.m.select(c, there, r, z),
            None => r,
        })
    }

    /// `v_readfirstlane_b32` of a console wave split into two host waves (#254). The console reads
    /// the first lane of its 64-lane EXEC, and every lane sees that value; `waveReadLaneFirst`
    /// reads the first lane of each host wave, its own half.
    ///
    /// - A value the same in every active lane of the console wave ([`Self::wave_uniform`]) is
    ///   read per half as before: each half's first lane holds it (0 barriers).
    /// - Otherwise, where both halves arrive together ([`Self::exchange_site`]), each half finds
    ///   its own first active lane via a single host ballot + `waveReadLaneAt`, then exchanges
    ///   `(val, has_any)` across the two halves in **one** LDS round-trip (2 barriers, no
    ///   `mask_dword(EXEC)`). When both halves have empty EXEC, the lower half's lane 0 is
    ///   returned (matching RDNA 2 ISA). If `EXEC` is untracked, falls back to
    ///   [`Self::lockstep_first_lane`] + [`Self::wide_read_lane`].
    /// - Otherwise the program is refused; a group of one console wave is then tried in lockstep,
    ///   which reads it exactly.
    pub(super) fn split_first_lane(
        &mut self,
        inst: &crate::ir::Inst,
        a: V,
    ) -> Result<Option<V>, Why> {
        use crate::gfx10::{Fields, Operand};
        use crate::ir::{EXEC, Reg};
        if !self.host32 || !self.halves {
            return Ok(None);
        }
        let source = match &inst.fields {
            Fields::Valu(f) => match f.src.first() {
                Some(&Operand::Vector { reg, .. }) => {
                    inst.read(Reg::V(reg)).map(|v| (Reg::V(reg), v))
                }
                Some(&Operand::Scalar { code, .. }) => {
                    inst.read(Reg::S(code)).map(|v| (Reg::S(code), v))
                }
                // A constant is the same in every lane.
                Some(Operand::Const(_) | Operand::Literal(_)) => return Ok(None),
                _ => None,
            },
            _ => None,
        };
        if let Some((reg, v)) = source
            && self.wave_uniform(inst, reg, v)
        {
            return Ok(None);
        }
        if self.exchange_site(self.cur_ir).is_err() {
            return Err(
                "a first lane read across the two host waves of a console wave, in divergent code",
            );
        }
        // Fast path: when EXEC is a tracked lane function, compute each host half's first active
        // lane locally and exchange (val, has_any) across the two halves in a single 2-barrier
        // round-trip (avoiding `mask_dword(EXEC)` and its extra 2-barrier ballot).
        let exec_bool = inst
            .read(Reg::S(EXEC))
            .and_then(|v| self.x.st.masks.get(&v).or_else(|| self.x.st.other.get(&v)).copied());
        if let Some(m) = exec_bool {
            let c = self.cur;
            let on = self.boolean(m, c)?;
            let my_mask = self.host_ballot(c, on).0;
            let i32t = self.s.t.i32;
            let (z, one, two, three, five, six, not63) = (
                self.s.m.i32(0),
                self.s.m.i32(1),
                self.s.m.i32(2),
                self.s.m.i32(3),
                self.s.m.i32(5),
                self.s.m.i32(6),
                self.s.m.i32(!63),
            );
            let has_any = self.s.m.cmp(c, pred::NE, my_mask, z);
            let first = self.s.op(c, "unaryBits.i32", Attr::ReadNone, i32t, 32, &[my_mask]);
            let my_lane = self.s.m.select(c, has_any, first, z);
            let my_val = self.read_lane(a, my_lane);
            let my_any = self.s.m.select(c, has_any, one, z);
            let f = self.console_flat();
            let wave_base = self.s.m.bin(c, Bin::And, f, not63);
            let half_raw = self.s.m.bin(c, Bin::LShr, f, five);
            let half = self.s.m.bin(c, Bin::And, half_raw, one);
            // Each console wave owns 64 dwords in `xchg` (`s.xchg_dwords = lanes.next_multiple_of(64)`):
            // +0/+1 hold lower/upper half's first-lane value, +2/+3 hold whether that half is non-empty.
            let val_idx = self.s.m.bin(c, Bin::Or, wave_base, half);
            let any_off = self.s.m.bin(c, Bin::Add, half, two);
            let any_idx = self.s.m.bin(c, Bin::Or, wave_base, any_off);
            let val_slot = self.xchg_slot(c, val_idx);
            let any_slot = self.xchg_slot(c, any_idx);
            self.s.m.store(c, val_slot, my_val);
            self.s.m.store(c, any_slot, my_any);
            self.xchg_barrier(c);
            let lo_val_slot = self.xchg_slot(c, wave_base);
            let lo_val = self.s.m.load(c, lo_val_slot);
            let hi_val_idx = self.s.m.bin(c, Bin::Or, wave_base, one);
            let hi_val_slot = self.xchg_slot(c, hi_val_idx);
            let hi_val = self.s.m.load(c, hi_val_slot);
            let lo_any_idx = self.s.m.bin(c, Bin::Or, wave_base, two);
            let lo_any_slot = self.xchg_slot(c, lo_any_idx);
            let lo_any = self.s.m.load(c, lo_any_slot);
            let hi_any_idx = self.s.m.bin(c, Bin::Or, wave_base, three);
            let hi_any_slot = self.xchg_slot(c, hi_any_idx);
            let hi_any = self.s.m.load(c, hi_any_slot);
            self.xchg_barrier(c);
            let pair = self.s.m.bin(c, Bin::LShr, f, six);
            let lo_half = self.s.m.bin(c, Bin::Shl, pair, one);
            let hi_half = self.s.m.bin(c, Bin::Add, lo_half, one);
            let hi_any = match self.xchg_written(c, hi_half, 5) {
                Some(there) => self.s.m.select(c, there, hi_any, z),
                None => hi_any,
            };
            let lo_empty = self.s.m.cmp(c, pred::EQ, lo_any, z);
            let hi_nonempty = self.s.m.cmp(c, pred::NE, hi_any, z);
            let use_hi = self.s.m.bin(c, Bin::And, lo_empty, hi_nonempty);
            return Ok(Some(self.s.m.select(c, use_hi, hi_val, lo_val)));
        }
        let lane = self.lockstep_first_lane(inst)?;
        Ok(Some(self.wide_read_lane(a, lane)?))
    }

    /// Whether `v`, what `reg` holds where `at` reads it, is provably the same in every lane that
    /// is on in `at`'s EXEC, in both halves of the console wave. Conservative and memoized: what
    /// it cannot prove is not uniform.
    ///
    /// Uniform are: constants; program inputs in SGPRs (user data); scalar instructions of uniform
    /// values (not reading a lane mask as per-lane data); scalar phis at uniform joins whose
    /// incoming values are all uniform; non-waterfall `v_readfirstlane_b32` results; values an
    /// earlier whole-wave section left uniform; and VALU instructions of uniform values with no
    /// DPP and no lane-mask operand, made under an EXEC that covers every lane active at `at`
    /// (`at_exec => def_exec`) and not carried out of a divergent loop.
    pub(super) fn wave_uniform(
        &self,
        at: &crate::ir::Inst,
        reg: crate::ir::Reg,
        v: crate::ir::Value,
    ) -> bool {
        use crate::ir::{EXEC, Reg};
        let execs = (at.read(Reg::S(EXEC)), at.read(Reg::S(EXEC + 1)));
        let at_mask = execs
            .0
            .and_then(|e| self.x.st.masks.get(&e).or_else(|| self.x.st.other.get(&e)).copied());
        let mut memo = Map::default();
        let mut visiting = Set::default();
        let mut hit_cycle = false;
        self.uniform_in(execs, at_mask, self.cur_ir, reg, v, &mut memo, &mut visiting, &mut hit_cycle)
    }

    #[allow(clippy::too_many_arguments)]
    fn uniform_in(
        &self,
        execs: (Option<crate::ir::Value>, Option<crate::ir::Value>),
        at_mask: Option<crate::structure::Bool>,
        at_block: usize,
        reg: crate::ir::Reg,
        v: crate::ir::Value,
        memo: &mut Map<crate::ir::Value, bool>,
        visiting: &mut Set<crate::ir::Value>,
        hit_cycle: &mut bool,
    ) -> bool {
        use crate::gfx10::{Fields, Operand};
        use crate::ir::{EXEC, Reg, VCC};
        if let Some(&cached) = memo.get(&v) {
            return cached;
        }
        if !visiting.insert(v) {
            // A cycle (loop phi): optimistically return true (greatest fixed point) so we still
            // inspect every non-cyclic input to the cycle; mark `hit_cycle` so intermediate nodes
            // inside an open cycle do not cache a tentative `true` in `memo`.
            *hit_cycle = true;
            return true;
        }
        let mut child_cycle = false;
        let res = (|| {
            if self.section_uniform(v) {
                return true;
            }
            let mask =
                |r: Reg| matches!(r, Reg::S(c) if c == EXEC || c == EXEC + 1 || c == VCC || c == VCC + 1);
            if mask(reg) {
                // A lane mask read as a number: with the halves on it is the console's 64 bits in
                // both (a ballot through `xchg`), so the same in every lane.
                return self.halves;
            }
            let p = self.x.p;
            let Some(inst) = self.def_inst(v) else {
                let Some(&b) = self.def_block.get(&v) else {
                    // A program input: user data in SGPRs is the wave's; VGPR inputs are per lane.
                    return matches!(reg, Reg::S(_));
                };
                let Some(phi) = p.blocks[b].phis.iter().find(|ph| ph.value == v) else {
                    return false;
                };
                let args: Vec<_> = phi.args.iter().copied().filter(|&x| x != v).collect();
                let Some(&first) = args.first() else {
                    return false;
                };
                if matches!(phi.reg, Reg::S(_)) {
                    // A scalar phi is wave-uniform only at a uniform join (where the whole console
                    // wave enters along the same edge; rejects values carried out of divergent
                    // loops or divergent if-joins).
                    if !self.uniform_join(b, |_| true) {
                        return false;
                    }
                    return args.iter().all(|&x| {
                        self.uniform_in(
                            execs,
                            at_mask,
                            at_block,
                            phi.reg,
                            x,
                            memo,
                            visiting,
                            &mut child_cycle,
                        )
                    });
                }
                // A vector join merges lanes from different paths: uniform only when every
                // non-self edge brings the exact same uniform SSA value.
                return args.iter().all(|&x| x == first)
                    && self.uniform_in(
                        execs,
                        at_mask,
                        at_block,
                        phi.reg,
                        first,
                        memo,
                        visiting,
                        &mut child_cycle,
                    );
            };
            let name = inst.name();
            let dst = match &inst.fields {
                Fields::Valu(f) => match f.dst {
                    Some(Operand::Vector { reg, dwords }) => Some((reg, dwords.max(1))),
                    _ => None,
                },
                _ => None,
            };
            let old = |r: Reg| {
                matches!((r, dst), (Reg::V(x), Some((d, n))) if x >= d && x < d.saturating_add(n))
            };
            match &inst.fields {
                // A first-lane read is the console wave's first lane since #254 (exact, or
                // refused); a waterfall loop's read is the lane's own.
                Fields::Valu(_) if name == "v_readfirstlane_b32" => {
                    !inst.writes.iter().any(|w| self.waterfall_reads.contains(&w.1))
                }
                Fields::Valu(f) => {
                    if f.dpp.is_some()
                        || name.starts_with("v_mbcnt")
                        || name.starts_with("v_cmp")
                        || name.starts_with("v_readlane")
                        || name.starts_with("v_writelane")
                        || name.starts_with("v_interp")
                        || f.src.iter().any(|o| matches!(o, Operand::Attr { .. }))
                    {
                        return false;
                    }
                    // Made where at least `at`'s active lanes were on: identical EXEC, entry EXEC,
                    // or `at_mask => def_mask` (every lane active at `at` was active at `def`).
                    // Reject if `def` is in a different block that could be inside a divergent
                    // loop `at_block` is outside (not proven same-iteration).
                    let def = (inst.read(Reg::S(EXEC)), inst.read(Reg::S(EXEC + 1)));
                    let entry = |x: Option<crate::ir::Value>| {
                        x.is_some_and(|x| self.def_inst(x).is_none() && !self.def_block.contains_key(&x))
                    };
                    let exec_covers = def == execs
                        || (entry(def.0) && entry(def.1))
                        || matches!(
                            (
                                at_mask,
                                def.0.and_then(|e| {
                                    self.x.st.masks.get(&e).or_else(|| self.x.st.other.get(&e)).copied()
                                }),
                            ),
                            (Some(am), Some(dm)) if !self.x.st.bools.meet(&[(am, true), (dm, false)])
                        );
                    if !exec_covers {
                        return false;
                    }
                    // Guard against a VALU defined inside a divergent loop and read outside it
                    // without an intervening phi (possible when the loop has a single exit).
                    if let Some(&db) = self.def_block.get(&v)
                        && db != at_block
                        && self.rpo[db] > 0
                        && !self.once[db]
                        && self.in_cycle_not_containing(db, at_block)
                    {
                        return false;
                    }
                    inst.reads.iter().all(|&(r, x)| {
                        if matches!(r, Reg::S(c) if c == EXEC || c == EXEC + 1) || old(r) {
                            return true;
                        }
                        !mask(r)
                            && self.uniform_in(
                                execs,
                                at_mask,
                                at_block,
                                r,
                                x,
                                memo,
                                visiting,
                                &mut child_cycle,
                            )
                    })
                }
                Fields::Sop { .. } | Fields::Smem { .. } => inst.reads.iter().all(|&(r, x)| {
                    self.uniform_in(
                        execs,
                        at_mask,
                        at_block,
                        r,
                        x,
                        memo,
                        visiting,
                        &mut child_cycle,
                    )
                }),
                _ => false,
            }
        })();
        visiting.remove(&v);
        if !res || !child_cycle || visiting.is_empty() {
            memo.insert(v, res);
        }
        if child_cycle && !visiting.is_empty() {
            *hit_cycle = true;
        }
        res
    }

    /// Whether IR block `db` can reach itself without passing through `at_block` (i.e. `db` is in
    /// a loop that `at_block` is outside).
    fn in_cycle_not_containing(&self, db: usize, at_block: usize) -> bool {
        let p = self.x.p;
        let mut seen = vec![false; p.blocks.len()];
        let mut stack: Vec<usize> = p.blocks[db].exit.successors();
        while let Some(x) = stack.pop() {
            if x == db {
                return true;
            }
            if x == at_block || std::mem::replace(&mut seen[x], true) {
                continue;
            }
            stack.extend(p.blocks[x].exit.successors());
        }
        false
    }

    /// The console lane `v_readfirstlane_b32` reads in lockstep (`lockstep.rs`): the lowest bit of
    /// EXEC as `inst` reads it, as a number (0 when none is set, as the console reads lane 0).
    pub(super) fn lockstep_first_lane(&mut self, inst: &crate::ir::Inst) -> Result<V, Why> {
        use crate::ir::{EXEC, Reg};
        let c = self.cur;
        let word = |l: &mut Self, r: u8| -> Result<V, Why> {
            let v = inst.read(Reg::S(r)).ok_or("EXEC of a first lane read")?;
            let was = std::mem::replace(&mut l.mask_ok, true);
            let x = l.i32_of(v);
            l.mask_ok = was;
            x
        };
        let (lo, hi) = (word(self, EXEC)?, word(self, EXEC + 1)?);
        let i32t = self.s.t.i32;
        let flo = self.s.op(c, "unaryBits.i32", Attr::ReadNone, i32t, 32, &[lo]);
        let fhi = self.s.op(c, "unaryBits.i32", Attr::ReadNone, i32t, 32, &[hi]);
        let (z, k32, k63) = (self.s.m.i32(0), self.s.m.i32(32), self.s.m.i32(63));
        let fhi = self.s.m.bin(c, Bin::Add, fhi, k32);
        // An empty EXEC reads lane 0 (RDNA 2 ISA): `FirstbitLo(0)` is all ones, which plus 32
        // would wrap to lane 31 (found in review of #254 by evan-kolberg).
        let (low, high) = (
            self.s.m.cmp(c, pred::NE, lo, z),
            self.s.m.cmp(c, pred::NE, hi, z),
        );
        let upper = self.s.m.select(c, high, fhi, z);
        let r = self.s.m.select(c, low, flo, upper);
        Ok(self.s.m.bin(c, Bin::And, r, k63))
    }

    /// Extend a host wave's result over the console's 64 lanes. A prefix
    /// also supplies its combining operation and the whole host wave's total.
    pub(super) fn halves_reduce(
        &mut self,
        op: Op,
        r: V,
        prefix: Option<(Bin, V)>,
    ) -> Result<V, Why> {
        let c = self.cur;
        self.exchange_site(self.cur_ir)?;
        let float = matches!(op, Op::AddF | Op::MulF | Op::MinF | Op::MaxF);
        let mine = prefix.map_or(r, |(_, total)| total);
        let bits = if float { self.as_int(mine) } else { mine };
        let f = self.console_flat();
        let (five, six, one) = (self.s.m.i32(5), self.s.m.i32(6), self.s.m.i32(1));
        let half = self.s.m.bin(c, Bin::LShr, f, five);
        let slot = self.xchg_slot(c, half);
        self.s.m.store(c, slot, bits);
        self.xchg_barrier(c);
        let pair = self.s.m.bin(c, Bin::LShr, f, six);
        let lo_at = self.s.m.bin(c, Bin::Shl, pair, one);
        let hi_at = self.s.m.bin(c, Bin::Add, lo_at, one);
        let (lo_p, hi_p) = (self.xchg_slot(c, lo_at), self.xchg_slot(c, hi_at));
        let (lo, hi) = (self.s.m.load(c, lo_p), self.s.m.load(c, hi_p));
        self.xchg_barrier(c);
        let (lo, hi) = if float {
            (self.as_float(lo), self.as_float(hi))
        } else {
            (lo, hi)
        };
        if let Some((bin, _)) = prefix {
            // The upper half's lanes also have every lane of the lower below.
            let upper = self.s.m.bin(c, Bin::And, half, one);
            let z = self.s.m.i32(0);
            let is_upper = self.s.m.cmp(c, pred::NE, upper, z);
            let combined = self.s.m.bin(c, bin, lo, r);
            return Ok(self.s.m.select(c, is_upper, combined, r));
        }
        let both = match op {
            Op::AddU | Op::AddF => self.s.m.bin(c, Bin::Add, lo, hi),
            Op::MulF => self.s.m.bin(c, Bin::Mul, lo, hi),
            Op::Or => self.s.m.bin(c, Bin::Or, lo, hi),
            Op::And => self.s.m.bin(c, Bin::And, lo, hi),
            Op::Xor => self.s.m.bin(c, Bin::Xor, lo, hi),
            Op::MinF | Op::MaxF => self.max_min_flushed(op == Op::MaxF, lo, hi),
            Op::MinU | Op::MaxU | Op::MinI | Op::MaxI => {
                let p = match op {
                    Op::MinU => pred::IULT,
                    Op::MaxU => pred::IUGT,
                    Op::MinI => pred::SLT,
                    _ => pred::SGT,
                };
                let t = self.s.m.cmp(c, p, lo, hi);
                self.s.m.select(c, t, lo, hi)
            }
            Op::AddH => return Err("a 16-bit wave reduction across the host's waves"),
        };
        // A missing upper half adds nothing: the lower half's result alone.
        Ok(match self.xchg_written(c, hi_at, 5) {
            Some(there) => self.s.m.select(c, there, both, lo),
            None => both,
        })
    }
}
