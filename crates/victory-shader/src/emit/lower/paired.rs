//! Lockstep's paired form (`emit/lockstep.rs`, #376): a console wave of 64 lanes runs in one host
//! wave of 32, each host thread holding two console lanes, `n` (the lower) and `n + 32` (the
//! upper) of host lane `n`. The design follows TaynanGT/Vice-Tay 5e588e8 ("paired lanes", after
//! ByFxbian's work on fxceless/victory-fork); this is our own lowering of it.
//!
//! Why: on a 32-lane host a console wave is otherwise two host waves, and what crosses them needs
//! both at a barrier. In a group of several console waves, waves that branch differently never meet
//! at the same group barrier, and a barrier of two SIMD groups alone, waited for by spinning, is not
//! sound on Apple GPUs (#438: right in small tests, hangs once the GPU is full). In the paired form
//! nothing crosses host waves: a console wave's 64 lanes are one host wave's 32 threads, the
//! structuriser's lockstep reading keeps every branch uniform over that host wave, and the title's
//! own `s_barrier` is the group barrier every console wave reaches as on the console.
//!
//! How: the lowering keeps a second copy of its per-lane state (`Twin`), and
//! [`Lower::swap_lanes`] exchanges the two, so the lane being lowered is always the one in
//! `Lower`'s own fields. A vector instruction is lowered once for each lane; a scalar instruction
//! once, its results copied to the other lane ([`Lower::mirror`]); the instructions that read other
//! lanes take both lanes' values at once ([`Lower::paired_cross`]). A lane mask's number is the
//! host ballot of the lower lanes' bits (bits 0-31) and of the upper lanes' (bits 32-63).

use super::*;

/// The per-lane state of the lane not being lowered (see the module).
#[derive(Default)]
pub(super) struct Twin {
    lane_fail: Map<Value, Why>,
    vals: Map<Value, Val>,
    floats: Map<Value, V>,
    low16: Map<Value, V>,
    high16: Map<Value, V>,
    bools: Map<(usize, Bool), V>,
    lane_bits: Map<Value, V>,
    mask_phis: Map<Value, V>,
    pending_masks: Vec<(V, usize, usize)>,
    data_phis: Vec<(V, usize, usize)>,
    spill_phis: Map<(Value, u32), V>,
    pending_spills: Vec<(V, usize, usize, u32)>,
    loaded_inputs: Map<(usize, u8), Val>,
    handles: Map<(usize, Site, u8, bool, u32), V>,
    console_flat: Option<V>,
}

impl Lower<'_> {
    /// Makes the other console lane of the thread the one being lowered.
    pub(super) fn swap_lanes(&mut self) {
        let Some(t) = self.twin.as_deref_mut() else {
            return;
        };
        std::mem::swap(&mut self.lane_fail, &mut t.lane_fail);
        std::mem::swap(&mut self.vals, &mut t.vals);
        std::mem::swap(&mut self.floats, &mut t.floats);
        std::mem::swap(&mut self.low16, &mut t.low16);
        std::mem::swap(&mut self.high16, &mut t.high16);
        std::mem::swap(&mut self.bools, &mut t.bools);
        std::mem::swap(&mut self.lane_bits, &mut t.lane_bits);
        std::mem::swap(&mut self.mask_phis, &mut t.mask_phis);
        std::mem::swap(&mut self.pending_masks, &mut t.pending_masks);
        std::mem::swap(&mut self.data_phis, &mut t.data_phis);
        std::mem::swap(&mut self.spill_phis, &mut t.spill_phis);
        std::mem::swap(&mut self.pending_spills, &mut t.pending_spills);
        std::mem::swap(&mut self.loaded_inputs, &mut t.loaded_inputs);
        std::mem::swap(&mut self.handles, &mut t.handles);
        std::mem::swap(&mut self.console_flat, &mut t.console_flat);
        self.upper = !self.upper;
    }

    /// `f` for the lower lane, then for the upper one; the lane being lowered is the same after.
    pub(super) fn both<T>(&mut self, mut f: impl FnMut(&mut Self) -> Result<T, Why>) -> Result<[T; 2], Why> {
        let was_upper = self.upper;
        if was_upper {
            self.swap_lanes();
        }
        let lo = f(self);
        self.swap_lanes();
        let hi = f(self);
        if !was_upper {
            self.swap_lanes();
        }
        Ok([lo?, hi?])
    }

    /// `f` for each lane in turn, the lane being lowered first; it stays the one being lowered.
    pub(super) fn each_lane(&mut self, mut f: impl FnMut(&mut Self) -> Result<(), Stop>) -> Result<(), Stop> {
        f(self)?;
        if self.paired {
            self.swap_lanes();
            let r = f(self);
            self.swap_lanes();
            r?;
        }
        Ok(())
    }

    /// The paired form's console lane ids: the host wave's first flat id, rounded down to 32, is
    /// its console wave's 32 threads; the lower lane is `64 w + n`, the upper `64 w + n + 32`.
    pub(super) fn paired_console_flat(&mut self) -> V {
        if let Some(f) = self.console_flat {
            return f;
        }
        let e = self.s.entry;
        let i32t = self.s.t.i32;
        let flat = self.s.op(e, "flattenedThreadIdInGroup.i32", Attr::ReadNone, i32t, 96, &[]);
        let first = self.s.op(e, "waveReadLaneFirst.i32", Attr::NoUnwind, i32t, 118, &[flat]);
        let (not31, one, k32) = (self.s.m.i32(!31), self.s.m.i32(1), self.s.m.i32(32));
        let base = self.s.m.bin(e, Bin::And, first, not31);
        let base = self.s.m.bin(e, Bin::Shl, base, one);
        let lane = self.s.op(e, "waveGetLaneIndex", Attr::ReadOnly, i32t, 111, &[]);
        let lower = self.s.m.bin(e, Bin::Add, base, lane);
        let upper = self.s.m.bin(e, Bin::Add, lower, k32);
        self.s.flags |= flags::WAVE_OPS;
        let (mine, other) = if self.upper { (upper, lower) } else { (lower, upper) };
        self.console_flat = Some(mine);
        if let Some(t) = self.twin.as_deref_mut() {
            t.console_flat = Some(other);
        }
        mine
    }

    /// A lane mask's number in the paired form: the host ballot of `bit` in the lower lanes (the
    /// console's bits 0-31) and in the upper lanes (bits 32-63). `bit` is made for each lane in
    /// DXIL block `at`.
    pub(super) fn paired_ballot(&mut self, at: usize, bit: impl Fn(&mut Self) -> Result<V, Why>) -> Result<(V, V), Why> {
        let [lo, hi] = self.both(|l| bit(l))?;
        let (x, _) = self.host_ballot(at, lo);
        let (y, _) = self.host_ballot(at, hi);
        Ok((x, y))
    }

    /// What `inst` wrote in the lane being lowered, for the other lane too: a scalar instruction's
    /// results, the same in both lanes.
    pub(super) fn mirror(&mut self, inst: &Inst) {
        let Some(t) = self.twin.as_deref_mut() else {
            return;
        };
        for &(_, w) in &inst.writes {
            if let Some(&v) = self.vals.get(&w) {
                t.vals.insert(w, v);
            }
            if let Some(&v) = self.floats.get(&w) {
                t.floats.insert(w, v);
            }
            if let Some(&v) = self.low16.get(&w) {
                t.low16.insert(w, v);
            }
            if let Some(&v) = self.high16.get(&w) {
                t.high16.insert(w, v);
            }
            if let Some(&why) = self.lane_fail.get(&w) {
                t.lane_fail.insert(w, why);
            }
        }
    }

    /// Instruction `i` of IR block `b` in the paired form. Scalar instructions run once (the wave's
    /// scalar unit runs them once), their results copied to the other lane; lane reads take both
    /// lanes at once; everything else is lowered for the lane being lowered and then for the other,
    /// so each lane does what its console lane does. An LDS access gets a group-shared memory fence
    /// first after a write, so what one lane wrote is seen by the other lanes' later accesses (all
    /// of a console wave's lanes are one host wave, which runs them in order).
    pub(super) fn paired_inst(&mut self, b: usize, i: usize) -> Result<(), Stop> {
        let inst: &Inst = &self.x.p.blocks[b].insts[i];
        let offset = inst.machine.offset;
        if matches!(inst.fields, Fields::Sop { .. } | Fields::Sopp { .. } | Fields::Smem { .. }) {
            self.inst(b, i)?;
            self.mirror(inst);
            return Ok(());
        }
        let name = inst.name();
        let waterfall = inst.writes.first().is_some_and(|w| self.waterfall_reads.contains(&w.1));
        if name == "v_readlane_b32" || name == "v_readfirstlane_b32" && !waterfall {
            self.paired_cross(inst).map_err(|w| unsupported(offset, w))?;
            self.mirror(inst);
            return Ok(());
        }
        if matches!(inst.fields, Fields::Ds { .. }) && !name.starts_with("ds_swizzle") && !name.contains("permute") {
            let write = !name.starts_with("ds_read");
            if write || self.lds_last == Some(true) {
                self.lds_fence();
            }
            self.lds_last = Some(write);
        }
        self.inst(b, i)?;
        self.swap_lanes();
        let r = self.inst(b, i);
        self.swap_lanes();
        r
    }

    /// A fence on group-shared memory, without a group barrier (the other host waves are other
    /// console waves, which the title orders by its own `s_barrier`).
    fn lds_fence(&mut self) {
        let mode = self.s.m.i32(8);
        let void = self.s.t.void;
        self.s.op(self.cur, "barrier", Attr::NoDuplicate, void, 80, &[mode]);
    }

    /// `v_readlane_b32` and `v_readfirstlane_b32` in the paired form: console lane `k`'s value is
    /// host lane `k & 31`'s lower value below 32 and its upper value from 32 on. The first lane
    /// read is lockstep's: the lowest lane of the wide EXEC ([`Lower::lockstep_first_lane`]).
    fn paired_cross(&mut self, inst: &Inst) -> Result<(), Why> {
        let lane = if inst.name() == "v_readlane_b32" {
            self.src(inst, 1)?
        } else {
            self.lockstep_first_lane(inst)?
        };
        let [lo, hi] = self.both(|l| l.src(inst, 0))?;
        let c = self.cur;
        let (k31, k32) = (self.s.m.i32(31), self.s.m.i32(32));
        let k63 = self.s.m.i32(63);
        let lane = self.s.m.bin(c, Bin::And, lane, k63);
        let at = self.s.m.bin(c, Bin::And, lane, k31);
        let (x, y) = (self.read_lane(lo, at), self.read_lane(hi, at));
        let below = self.s.m.cmp(c, pred::IULT, lane, k32);
        let r = self.s.m.select(c, below, x, y);
        self.s.flags |= flags::WAVE_OPS;
        self.write_i(inst, r, None)
    }
}
