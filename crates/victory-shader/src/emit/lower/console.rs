//! Console control flow for one wave64 split across two host SIMD groups.
//! Port of the #439 approach: every thread follows wave-wide branches, and
//! EXEC guards vector work. Only then can LDS exchanges use group barriers.

use super::*;

pub(super) fn retry_reason(why: &str) -> bool {
    // Replacing control flow cannot give a scalar mask instruction the other wave width.
    // Preserve the precise upstream refusal instead of hiding it behind an opaque-mask retry.
    if why.ends_with("instruction of the other width") {
        return false;
    }
    // Upstream appends the specific unsupported step to these refusal families.
    let why = why.split_once(": ").map_or(why, |(reason, _)| reason);
    matches!(
        why,
        "a cross-lane operation between host waves in divergent code"
            | "LDS shared in a wave the host splits, in divergent code"
            | "a lane-mask join of an untracked mask"
            | "a lane mask holding lanes elsewhere read as data"
            | "a whole-wave section step the simulation does not follow"
            | "an opaque lane mask"
            | "a constant lane mask"
    )
}

pub(super) fn eligible(x: &Parts) -> bool {
    let StageIo::Cs(cs) = x.native.io else {
        return false;
    };
    if !x.wave32_host || !x.native.wave64 || cs.threads.iter().product::<u32>() > 64 {
        return false;
    }
    let insts = || x.p.blocks.iter().flat_map(|b| &b.insts);
    // Preserve #439's title-memory exclusion: enabling only part of the
    // ray-structure builder was reported to hang a GPU. Its cause is unknown.
    let memory = x
        .res
        .scalar
        .iter()
        .any(|s| matches!(s.1.param, MEMORY | USER_DATA))
        || insts().any(|i| i.name().starts_with("image_bvh"));
    let sections = super::super::wave::has_sections(x.p);
    !(insts().any(|i| i.name().contains("permute") && i.name() != "ds_bpermute_b32")
        || memory
            && (cs.lds_bytes() > 0 || sections || insts().any(|i| i.name().contains("permute"))))
}

/// Lane reads/writes ignore EXEC and must see registers of lanes that do
/// not take the branch. Per-lane control flow cannot represent that.
pub(super) fn preferred(x: &Parts) -> bool {
    x.p.blocks.iter().enumerate().any(|(b, block)| {
        let sections: Vec<_> = super::super::wave::section_ranges(x.p, b).collect();
        block.insts.iter().enumerate().any(|(i, inst)| {
            matches!(
                inst.name(),
                "v_readlane_b32" | "v_writelane_b32" | "v_readfirstlane_b32"
            ) && !sections
                .iter()
                .any(|range| range.contains(&i))
        })
    })
}

impl Lower<'_> {
    pub(super) fn console_branch(&mut self, b: usize) -> Result<V, Why> {
        let inst = self.x.p.blocks[b].insts.last().ok_or("a branch")?;
        let name = inst.name();
        let c = self.cur;
        let (any, set) = match name {
            "s_cbranch_scc0" | "s_cbranch_scc1" => {
                let v = inst.read(Reg::Scc).ok_or("a branch's SCC")?;
                let scc = self.scc_bit(v)?;
                (scc, name == "s_cbranch_scc1")
            }
            "s_cbranch_vccz" | "s_cbranch_vccnz" | "s_cbranch_execz" | "s_cbranch_execnz" => {
                let reg = if name.contains("vcc") {
                    crate::ir::VCC
                } else {
                    EXEC
                };
                let lo = inst.read(Reg::S(reg)).ok_or("a branch's mask")?;
                let &m = self
                    .x
                    .st
                    .masks
                    .get(&lo)
                    .ok_or("a branch on an untracked mask")?;
                (self.any_lane(m)?, name.ends_with("nz"))
            }
            _ => return Err("a branch the console flow does not model"),
        };
        Ok(if set {
            any
        } else {
            let t = self.s.m.bool(true);
            self.s.m.bin(c, Bin::Xor, any, t)
        })
    }

    /// Console flow: an SCC value as a bit. A mask operation's SCC (result
    /// not zero) is whether some lane of the console wave has its bit.
    pub(super) fn scc_bit(&mut self, v: Value) -> Result<V, Why> {
        match self.vals.get(&v) {
            Some(&Val::Bit(x)) => return Ok(x),
            Some(&Val::Missing(why)) if why != LANE_MASK => return Err(why),
            _ => {}
        }
        let b = *self
            .def_block
            .get(&v)
            .ok_or("an SCC from the program's start")?;
        let inst = self.x.p.blocks[b]
            .insts
            .iter()
            .find(|i| i.writes.iter().any(|w| w.1 == v))
            .ok_or("an SCC without its instruction")?;
        // SCC tests the new EXEC. `s_*_saveexec` also writes a scalar, the
        // EXEC it replaces, and that write comes first (`s_*_wrexec` writes
        // the new EXEC to both).
        let mask = inst
            .write(Reg::S(EXEC))
            .and_then(|w| self.x.st.masks.get(&w).copied())
            .or_else(|| {
                inst.writes
                    .iter()
                    .filter(|w| matches!(w.0, Reg::S(_)))
                    .find_map(|w| self.x.st.masks.get(&w.1).copied())
            })
            .ok_or("an SCC the emitter did not compute")?;
        self.any_lane(mask)
    }

    /// Console flow: whether `m` holds in some lane of the console wave.
    fn any_lane(&mut self, m: Bool) -> Result<V, Why> {
        let c = self.cur;
        let bit = self.boolean(m, c)?;
        let (lo, hi) = self.wide_ballot(c, self.cur_ir, bit)?;
        let or = self.s.m.bin(c, Bin::Or, lo, hi);
        let z = self.s.m.i32(0);
        Ok(self.s.m.cmp(c, pred::NE, or, z))
    }
}
