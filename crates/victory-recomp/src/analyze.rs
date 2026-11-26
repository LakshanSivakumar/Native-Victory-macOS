//! Whole-image analysis: disassemble every `.pdata` function and gather the
//! statistics that drive the recompiler design (extensions used, indirect
//! control flow, jump tables, JIT / self-modifying-code signs).

use std::collections::{BTreeMap, HashMap, HashSet};
use std::fmt::Write as _;

use anyhow::Result;
use iced_x86::{
    Code, CpuidFeature, Decoder, DecoderOptions, FlowControl, Instruction, InstructionInfoFactory, Mnemonic,
    OpAccess, OpKind, Register,
};

use crate::image::{Func, Image, SCN_EXEC, SCN_WRITE};

/// A jump table recovered from a `jmp reg` dispatch sequence.
#[derive(Debug, Clone)]
pub struct JumpTable {
    /// RVA of the `jmp reg` instruction.
    pub jmp_rva: u32,
    pub table_rva: u32,
    /// Target RVAs, all inside the owning function.
    pub targets: Vec<u32>,
    /// The size came from a range check on the index (else the table ran as far as its entries stayed in code).
    pub bounded: bool,
}

/// Decodes one function. Returns instructions and whether decoding hit garbage.
///
/// Linear, then resynchronised: data inside the range (a table of case bytes after a `jmp`, constants) throws the linear
/// decode off until it happens to realign, so every branch target and jump-table case that is not an instruction start
/// is decoded from there on (the junk that overlaps it is dropped), until nothing new turns up.
pub fn decode_function(img: &Image, f: &Func) -> (Vec<Instruction>, bool) {
    let Some(code) = img.bytes(f.begin, (f.end - f.begin) as usize) else {
        return (vec![], true);
    };
    let base = img.info.image_base;
    let ip0 = base + f.begin as u64;
    let end_ip = base + f.end as u64;
    let mut bad = false;
    let mut by_ip: std::collections::BTreeMap<u64, Instruction> = std::collections::BTreeMap::new();
    {
        let mut dec = Decoder::with_ip(64, code, ip0, DecoderOptions::NONE);
        while dec.can_decode() {
            let i = dec.decode();
            if i.is_invalid() {
                bad = true;
                break;
            }
            by_ip.insert(i.ip(), i);
        }
    }
    let mut tried: std::collections::HashSet<u64> = std::collections::HashSet::new();
    for _ in 0..8 {
        let mut wanted: Vec<u64> = by_ip
            .values()
            .filter(|i| {
                matches!(
                    i.flow_control(),
                    iced_x86::FlowControl::ConditionalBranch | iced_x86::FlowControl::UnconditionalBranch
                ) && i.op0_kind() == OpKind::NearBranch64
            })
            .map(|i| i.near_branch_target())
            .collect();
        let ins_now: Vec<Instruction> = by_ip.values().copied().collect();
        wanted.extend(
            find_jump_tables(img, f, &ins_now)
                .iter()
                .filter(|t| t.bounded)
                .flat_map(|t| t.targets.iter().map(|&r| base + r as u64)),
        );
        wanted.retain(|t| (ip0..end_ip).contains(t) && !by_ip.contains_key(t) && tried.insert(*t));
        if wanted.is_empty() {
            break;
        }
        for t in wanted {
            if by_ip.contains_key(&t) {
                continue;
            }
            let off = (t - ip0) as usize;
            let mut dec = Decoder::with_ip(64, &code[off..], t, DecoderOptions::NONE);
            // The old instruction spanning `t` is junk.
            if let Some((&k, old)) = by_ip.range(..t).next_back()
                && k + old.len() as u64 > t
            {
                by_ip.remove(&k);
            }
            while dec.can_decode() {
                let i = dec.decode();
                if i.is_invalid() {
                    break;
                }
                if i.ip() != t && by_ip.contains_key(&i.ip()) {
                    break; // realigned with the stream already decoded
                }
                // Drop old instructions this one overlaps.
                let stale: Vec<u64> = by_ip
                    .range(i.ip()..i.ip() + i.len() as u64)
                    .map(|(&k, _)| k)
                    .collect();
                for k in stale {
                    by_ip.remove(&k);
                }
                by_ip.insert(i.ip(), i);
            }
        }
    }
    (by_ip.into_values().collect(), bad)
}

fn is_cmov_free_load(code: Code) -> bool {
    matches!(
        code,
        Code::Movsxd_r64_rm32
            | Code::Mov_r32_rm32
            | Code::Mov_r64_rm64
            | Code::Movzx_r32_rm16
            | Code::Movzx_r64_rm16
    )
}

/// Recognises both table styles: clang/gcc (`lea base,[rip+table]` then
/// `movsxd r,[base+idx*4]`) and MSVC (`lea base,[__ImageBase]` then
/// `mov r32,[base+idx*4+table_rva]`), each followed by `add r,base; jmp r`.
pub fn find_jump_tables(img: &Image, f: &Func, ins: &[Instruction]) -> Vec<JumpTable> {
    find_jump_tables_ext(img.info.image_base, &|rva| img.read_u32(rva), &|rva| img.is_exec(rva), f, ins)
}

/// Same, over any RVA-addressed `u32` reader (used by tests on raw blobs). A case outside the function ends the table.
pub fn find_jump_tables_with(
    base: u64,
    read_u32: &dyn Fn(u32) -> Option<u32>,
    f: &Func,
    ins: &[Instruction],
) -> Vec<JumpTable> {
    find_jump_tables_ext(base, read_u32, &|_| false, f, ins)
}

/// [`find_jump_tables_with`] where a case that lies outside the function but in code (`is_code`) is kept: a switch
/// whose case is a tail call into another function. The table ends at the first case that is not code.
pub fn find_jump_tables_ext(
    base: u64,
    read_u32: &dyn Fn(u32) -> Option<u32>,
    is_code: &dyn Fn(u32) -> bool,
    f: &Func,
    ins: &[Instruction],
) -> Vec<JumpTable> {
    let mut out = vec![];
    let mut found = vec![];
    // Where other paths join: the targets of the function's direct branches.
    let labels: HashSet<u64> = ins
        .iter()
        .filter(|i| {
            matches!(i.flow_control(), FlowControl::ConditionalBranch | FlowControl::UnconditionalBranch)
                && i.op0_kind() == OpKind::NearBranch64
        })
        .map(|i| i.near_branch_target())
        .collect();
    let mut facts = InstructionInfoFactory::new();
    for (n, j) in ins.iter().enumerate() {
        if j.code() == Code::Jmp_rm64 && j.op0_kind() == OpKind::Memory {
            // `jmp qword ptr [base + 8*idx]`: a table of 64-bit addresses (an interpreter's handlers, each ending in
            // the same dispatch: rva 0x3de2612, 530 of them), its base an `lea base, [rip + table]` of the function,
            // the last one before (else the first after: the interpreter sets the register once, at its top).
            let b = j.memory_base();
            if j.memory_index_scale() == 8 && j.memory_index() != Register::None && b != Register::None && !j.is_ip_rel_memory_operand() {
                let is_lea = |i: &&Instruction| i.code() == Code::Lea_r64_m && i.is_ip_rel_memory_operand() && i.op0_register() == b;
                if let Some(lea) = ins[..n].iter().rev().find(is_lea).or_else(|| ins[n..].iter().find(is_lea)) {
                    let lea_rva = (lea.ip_rel_memory_address().wrapping_sub(base)) as u32;
                    let table_rva = lea_rva.wrapping_add(j.memory_displacement64() as u32);
                    found.push(((j.ip() - base) as u32, table_rva, lea_rva, Entry::Abs64, None));
                }
            }
            continue;
        }
        if j.code() != Code::Jmp_rm64 || j.op0_kind() != OpKind::Register {
            continue;
        }
        let win = &ins[n.saturating_sub(40)..n];
        // The scaled load from the table, then the rip-relative `lea` of its base register: close by in compiler
        // switches, but MSVC's memmove loads `__ImageBase` once at the entry for all its tables.
        let Some((ld, lea)) = win.iter().rev().find_map(|ld| {
            if !(is_cmov_free_load(ld.code())
                && ld.op1_kind() == OpKind::Memory
                && ld.memory_index_scale() == 4)
            {
                return None;
            }
            let base_reg = ld.memory_base();
            let ld_at = ins[..n].iter().rposition(|i| std::ptr::eq(i, ld) || i.ip() == ld.ip())?;
            let lea_at = ins[..ld_at]
                .iter()
                .rposition(|i| i.code() == Code::Lea_r64_m && i.is_ip_rel_memory_operand() && i.op0_register() == base_reg)?;
            let base_va = ins[lea_at].ip_rel_memory_address();
            feeds_jump(ins, ld_at, n, base_reg, base_va, &labels, &mut InstructionInfoFactory::new()).then(|| (ld, &ins[lea_at]))
        }) else {
            continue;
        };
        let lea_rva = (lea.ip_rel_memory_address().wrapping_sub(base)) as u32;
        let table_rva = lea_rva.wrapping_add(ld.memory_displacement64() as u32);
        // The size from the range check `cmp idx, N` / `ja default` on the value the load takes as its index, or,
        // for MSVC's two-level switch (`movzx eax, byte [base+idx+bytes]` then `mov ecx, [base+4*rax+table]`), from
        // the largest of the byte table's checked entries. With none, the table runs as far as its entries land inside
        // the function (a compare of something else would cut a real table short, and a missing entry is a trap at
        // run time, where an extra one is only a label).
        let ld_at = ins[..n].iter().rposition(|i| i.ip() == ld.ip()).unwrap_or(n);
        let bound = match index_bound(ins, ld_at.saturating_sub(40), ld_at, &labels, &mut facts) {
            Some(Bound::Entries(b)) => Some(b),
            Some(Bound::Bytes { at, entries }) => {
                let bytes = (0..entries).map(|k| read_u32(lea_rva.wrapping_add(at).wrapping_add(k as u32)).map(|w| w & 0xff));
                bytes.collect::<Option<Vec<u32>>>().and_then(|b| b.into_iter().max()).map(|m| m as usize + 1)
            }
            None => None,
        }
        .filter(|b| (1..=65536).contains(b));
        // `movsxd`: offsets from the base; `mov`: RVAs (the base is `__ImageBase`).
        let entry = if matches!(ld.code(), Code::Mov_r32_rm32 | Code::Mov_r64_rm64) { Entry::Rva32 } else { Entry::Rel32 };
        found.push(((j.ip() - base) as u32, table_rva, lea_rva, entry, bound));
    }
    // A table ends where the function's next table starts: tables sit side by side in .rdata (and MSVC's byte table
    // right after its offsets), and reading on would take the next one's entries for cases. A range check that
    // reaches past it is not this table's. Up to there every entry is the table's, so one in other code is a case
    // that is a tail call, as with a range check; with neither, the table ends at the first entry outside the
    // function.
    let mut starts: Vec<u32> = found.iter().map(|t| t.1).collect();
    starts.sort_unstable();
    for (jmp_rva, table_rva, lea_rva, entry, bound) in found {
        let size = if entry == Entry::Abs64 { 8 } else { 4 };
        let room = starts.iter().find(|&&s| s > table_rva).map(|&next| ((next - table_rva) / size) as usize);
        let sized = bound.is_some() || room.is_some();
        let mut targets = vec![];
        for k in 0..bound.unwrap_or(4096).min(room.unwrap_or(4096)) {
            let at = table_rva.wrapping_add(size * k as u32);
            let Some(e) = read_u32(at) else {
                break;
            };
            let t = match entry {
                Entry::Rel32 => lea_rva.wrapping_add(e as i32 as u32),
                Entry::Rva32 => lea_rva.wrapping_add(e),
                // The address at the image's preferred base (the loader's relocations move the table with the image).
                Entry::Abs64 => {
                    let Some(hi) = read_u32(at.wrapping_add(4)) else { break };
                    match (u64::from(hi) << 32 | u64::from(e)).checked_sub(base) {
                        Some(r) if r <= u64::from(u32::MAX) => r as u32,
                        _ => break,
                    }
                }
            };
            if (t < f.begin || t >= f.end) && !(sized && is_code(t)) {
                break;
            }
            targets.push(t);
        }
        if targets.len() >= 2 || (bound.is_some() && !targets.is_empty()) {
            out.push(JumpTable { jmp_rva, table_rva, targets, bounded: bound.is_some() });
        }
    }
    out
}

/// Whether the table load `ins[ld_at]` makes the target of the `jmp reg` at `ins[n]`: the loaded offset is added to
/// the table's base (`add off, base` into the jump's register, or `add base, off` when the jump takes the base; the
/// base the load's own base register, or one an `lea` of the same address set since: MSVC's Bink loads the offset
/// over its base, `mov edx, [rdx + 4*rcx + table]`, then `lea rcx, [__ImageBase]` / `add rdx, rcx`), and on the way
/// from the load to the jump nothing writes the offset (before the add) or the target (after it) otherwise, breaks
/// the flow (the code after an unconditional jump or a return is not reached from the load) or joins another path
/// (a branch target, the jump included). Else the jump is another one through a register: a virtual call in tail
/// position right after a switch (rva 0x1ffa572, `mov rax, [rcx]` / `mov rax, [rax + 0x78]` / `jmp rax` behind the
/// switch's own `jmp rax`) was taken for the switch, and its target trapped as a case out of the table.
fn feeds_jump(
    ins: &[Instruction],
    ld_at: usize,
    n: usize,
    base_reg: Register,
    base_va: u64,
    labels: &HashSet<u64>,
    facts: &mut InstructionInfoFactory,
) -> bool {
    let (off, jr) = (ins[ld_at].op0_register().full_register(), ins[n].op0_register().full_register());
    // The registers that hold the base.
    let mut bases: Vec<Register> = vec![base_reg.full_register()];
    bases.retain(|&b| b != off);
    let mut added = false;
    for i in &ins[ld_at + 1..=n] {
        if labels.contains(&i.ip()) {
            return false;
        }
        if i.ip() == ins[n].ip() {
            break;
        }
        if matches!(
            i.flow_control(),
            FlowControl::UnconditionalBranch | FlowControl::IndirectBranch | FlowControl::Return | FlowControl::Interrupt
        ) {
            return false;
        }
        let add = !added
            && i.mnemonic() == Mnemonic::Add
            && i.op0_kind() == OpKind::Register
            && i.op1_kind() == OpKind::Register
            && {
                let (d, s) = (i.op0_register().full_register(), i.op1_register().full_register());
                (d == off && bases.contains(&s) && jr == off) || (bases.contains(&d) && s == off && jr == d)
            };
        if add {
            added = true;
            continue;
        }
        let same_base = i.code() == Code::Lea_r64_m && i.is_ip_rel_memory_operand() && i.ip_rel_memory_address() == base_va;
        for u in facts.info(i).used_registers() {
            if !matches!(u.access(), OpAccess::Write | OpAccess::CondWrite | OpAccess::ReadWrite | OpAccess::ReadCondWrite) {
                continue;
            }
            let w = u.register().full_register();
            if (!added && w == off) || (added && w == jr) {
                return false;
            }
            bases.retain(|&b| b != w);
            if same_base {
                bases.push(w);
            }
        }
    }
    added
}

/// How a jump table's entries name their cases.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Entry {
    /// 32-bit offsets from the table's base (clang: `movsxd` then `add`).
    Rel32,
    /// 32-bit RVAs (MSVC: `mov r32` from `[__ImageBase + 4*idx + table]`, then `add` of the base).
    Rva32,
    /// 64-bit addresses (`jmp qword ptr [base + 8*idx]`).
    Abs64,
}

fn is_cmp_imm(i: &Instruction) -> bool {
    matches!(
        i.code(),
        Code::Cmp_rm32_imm8
            | Code::Cmp_rm32_imm32
            | Code::Cmp_rm64_imm8
            | Code::Cmp_rm64_imm32
            | Code::Cmp_rm8_imm8
            | Code::Cmp_rm16_imm8
            | Code::Cmp_rm16_imm16
            | Code::Cmp_AL_imm8
            | Code::Cmp_AX_imm16
            | Code::Cmp_EAX_imm32
            | Code::Cmp_RAX_imm32
    )
}

/// What a range check says of a jump table's size.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Bound {
    /// The table's index is below this.
    Entries(usize),
    /// The index is an entry of a byte table (MSVC's two-level switch) at `at` from the table load's base, whose own
    /// index is below `entries`.
    Bytes { at: u32, entries: usize },
}

/// The range check on the index of the table load `ins[ld_at]`, read forward from `ins[from]`. Each register holds
/// a value number: a plain copy (`mov eax, r13d`, `movzx r10d, dl`, `movsxd rax, ecx`, `mov r8d, r8d`) passes it on,
/// any other write gives a new one, and `cmp reg, N` followed by `ja` / `jae` bounds the value it compares (by the
/// least of its checks). So a check of a value changed after it (`add r8d, 2` after `cmp r8d, 7`) or of another one
/// does not bound the index, and one of the value before a copy made ahead of it (`movzx r10d, dl` ... `cmp dl, 7`,
/// the index `r10`) does. A branch target forgets everything before it: another path joins there, unchecked.
fn index_bound(
    ins: &[Instruction],
    from: usize,
    ld_at: usize,
    labels: &HashSet<u64>,
    facts: &mut InstructionInfoFactory,
) -> Option<Bound> {
    use iced_x86::Mnemonic as M;
    let ld = &ins[ld_at];
    let (index, table_base) = (ld.memory_index().full_register(), ld.memory_base().full_register());
    let mut value: HashMap<Register, u32> = HashMap::new();
    let mut bounds: HashMap<u32, usize> = HashMap::new();
    // Values loaded from a byte table: the table's displacement and the value of its index.
    let mut byte_of: HashMap<u32, (u32, u32)> = HashMap::new();
    let mut fresh = 0u32;
    let mut num = |value: &mut HashMap<Register, u32>, r: Register| -> u32 {
        *value.entry(r).or_insert_with(|| {
            fresh += 1;
            fresh
        })
    };
    for p in from..=ld_at {
        let i = &ins[p];
        if p > from && labels.contains(&i.ip()) {
            value.clear();
            bounds.clear();
            byte_of.clear();
        }
        if p == ld_at {
            let v = num(&mut value, index);
            if let Some(&b) = bounds.get(&v) {
                return Some(Bound::Entries(b));
            }
            return byte_of
                .get(&v)
                .and_then(|&(at, of)| bounds.get(&of).map(|&entries| Bound::Bytes { at, entries }));
        }
        let jcc = ins.get(p + 1).map(|j| j.code());
        if is_cmp_imm(i)
            && i.op0_kind() == OpKind::Register
            && matches!(jcc, Some(Code::Ja_rel8_64 | Code::Ja_rel32_64 | Code::Jae_rel8_64 | Code::Jae_rel32_64))
        {
            let v = match i.code() {
                Code::Cmp_AL_imm8 | Code::Cmp_rm8_imm8 => i.immediate(1) as u8 as usize,
                Code::Cmp_rm16_imm8 | Code::Cmp_rm16_imm16 | Code::Cmp_AX_imm16 => i.immediate(1) as u16 as usize,
                _ => i.immediate(1) as usize,
            };
            let b = v.saturating_add(usize::from(!matches!(jcc, Some(Code::Jae_rel8_64 | Code::Jae_rel32_64))));
            let id = num(&mut value, i.op0_register().full_register());
            bounds.entry(id).and_modify(|x| *x = (*x).min(b)).or_insert(b);
            continue;
        }
        if matches!(i.mnemonic(), M::Mov | M::Movzx | M::Movsx | M::Movsxd)
            && i.op0_kind() == OpKind::Register
            && i.op1_kind() == OpKind::Register
        {
            // A write of 32 bits or more sets the whole register; a narrower one keeps the value only where
            // the register already held it.
            let s = num(&mut value, i.op1_register().full_register());
            let d = i.op0_register();
            if d.size() >= 4 || value.get(&d.full_register()) == Some(&s) {
                value.insert(d.full_register(), s);
            } else {
                let v = num(&mut HashMap::new(), d);
                value.insert(d.full_register(), v);
            }
            continue;
        }
        if matches!(i.code(), Code::Cdqe | Code::Cwde) {
            // `cdqe` after `cmp eax, N` (MSVC, ahead of the byte table): the checked value, widened.
            continue;
        }
        if matches!(i.flow_control(), FlowControl::Call | FlowControl::IndirectCall) {
            // The callee may change the registers the Windows x64 convention leaves to it.
            for r in [Register::RAX, Register::RCX, Register::RDX, Register::R8, Register::R9, Register::R10, Register::R11] {
                let v = num(&mut HashMap::new(), r);
                value.insert(r, v);
            }
        }
        let byte_load = i.code() == Code::Movzx_r32_rm8
            && i.op1_kind() == OpKind::Memory
            && i.memory_base().full_register() == table_base
            && i.memory_index_scale() == 1
            && i.memory_index() != Register::None;
        let of = byte_load.then(|| num(&mut value, i.memory_index().full_register()));
        for u in facts.info(i).used_registers() {
            if matches!(u.access(), OpAccess::Write | OpAccess::CondWrite | OpAccess::ReadWrite | OpAccess::ReadCondWrite) {
                let v = num(&mut HashMap::new(), u.register());
                value.insert(u.register().full_register(), v);
                if let (Some(of), true) = (of, u.register().full_register() == i.op0_register().full_register()) {
                    byte_of.insert(v, (i.memory_displacement64() as u32, of));
                }
            }
        }
    }
    None
}

#[derive(Default, Clone)]
pub struct Stats {
    pub functions: u64,
    pub chained: u64,
    pub decode_failed_funcs: u64,
    pub code_bytes: u64,
    pub instructions: u64,
    pub max_func_instrs: u64,
    pub func_sizes: Vec<u32>,
    pub features: BTreeMap<String, u64>,
    pub mnemonics: HashMap<String, u64>,
    pub call_direct: u64,
    pub call_indirect_reg: u64,
    pub call_indirect_mem_rip: u64,
    pub call_indirect_mem_other: u64,
    pub jmp_direct: u64,
    pub jmp_cond: u64,
    pub jmp_indirect_reg: u64,
    pub jmp_indirect_mem: u64,
    pub tail_jmp_out_of_func: u64,
    pub rets: u64,
    pub jump_tables: u64,
    pub jump_table_entries: u64,
    pub jmp_reg_without_table: u64,
    pub lock_prefix: u64,
    pub rep_string: u64,
    pub gs_access: u64,
    pub fs_access: u64,
    pub syscall_like: u64,
    pub cpuid_rdtsc: u64,
    pub store_into_exec: u64,
    pub x87: u64,
    pub xchg_mem: u64,
    pub xmm_ins: u64,
    pub ymm_ins: u64,
    pub zmm_ins: u64,
    pub call_targets_non_pdata: u64,
    pub call_targets_into_exec_gaps: u64,
    /// Distinct non-.pdata direct call / out-of-function jmp targets (leaf functions, thunks).
    pub gap_targets: HashSet<u32>,
}

impl Stats {
    fn merge(&mut self, o: Stats) {
        macro_rules! add { ($($f:ident),*) => { $( self.$f += o.$f; )* } }
        add!(
            functions,
            chained,
            decode_failed_funcs,
            code_bytes,
            instructions,
            call_direct,
            call_indirect_reg,
            call_indirect_mem_rip,
            call_indirect_mem_other,
            jmp_direct,
            jmp_cond,
            jmp_indirect_reg,
            jmp_indirect_mem,
            tail_jmp_out_of_func,
            rets,
            jump_tables,
            jump_table_entries,
            jmp_reg_without_table,
            lock_prefix,
            rep_string,
            gs_access,
            fs_access,
            syscall_like,
            cpuid_rdtsc,
            store_into_exec,
            x87,
            xchg_mem,
            xmm_ins,
            ymm_ins,
            zmm_ins,
            call_targets_non_pdata,
            call_targets_into_exec_gaps
        );
        self.gap_targets.extend(o.gap_targets);
        self.max_func_instrs = self.max_func_instrs.max(o.max_func_instrs);
        self.func_sizes.extend(o.func_sizes);
        for (k, v) in o.features {
            *self.features.entry(k).or_default() += v;
        }
        for (k, v) in o.mnemonics {
            *self.mnemonics.entry(k).or_default() += v;
        }
    }
}

/// Image-level facts that are not per-instruction.
#[derive(Default)]
pub struct ImageFacts {
    pub image_bytes_in_exec_sections: u64,
    pub exec_sections: Vec<(String, u32, u32, bool)>,
    pub wx_sections: Vec<String>,
    pub pdata_covered_bytes: u64,
    pub exports: usize,
    pub exports_not_in_pdata: usize,
    pub entry_in_pdata: bool,
    pub tls_callbacks: usize,
    pub tls_callbacks_not_in_pdata: usize,
    pub relocs_dir64: u64,
    pub relocs_into_exec: u64,
    pub relocs_to_func_start: u64,
    pub relocs_into_exec_not_func_start: u64,
    pub suspicious_imports: Vec<String>,
    pub reloc_gap_targets: usize,
    pub gap_targets_union: usize,
}

pub struct Report {
    pub stats: Stats,
    pub facts: ImageFacts,
}

fn analyze_chunk(
    img: &Image,
    funcs: &[Func],
    starts: &HashSet<u32>,
    exec_ranges: &[(u32, u32)],
) -> Stats {
    let base = img.info.image_base;
    let mut st = Stats::default();
    let in_exec = |rva: u32| exec_ranges.iter().any(|&(a, b)| rva >= a && rva < b);
    for f in funcs {
        st.functions += 1;
        st.chained += f.chained as u64;
        st.code_bytes += (f.end - f.begin) as u64;
        st.func_sizes.push(f.end - f.begin);
        let (ins, bad) = decode_function(img, f);
        st.decode_failed_funcs += bad as u64;
        st.instructions += ins.len() as u64;
        st.max_func_instrs = st.max_func_instrs.max(ins.len() as u64);
        let tables = find_jump_tables(img, f, &ins);
        st.jump_tables += tables.len() as u64;
        st.jump_table_entries += tables.iter().map(|t| t.targets.len() as u64).sum::<u64>();
        let table_jmps: HashSet<u32> = tables.iter().map(|t| t.jmp_rva).collect();
        for i in &ins {
            *st.mnemonics
                .entry(format!("{:?}", i.mnemonic()))
                .or_default() += 1;
            for &ft in i.cpuid_features() {
                *st.features.entry(format!("{ft:?}")).or_default() += 1;
            }
            if i.cpuid_features().contains(&CpuidFeature::FPU) {
                st.x87 += 1;
            }
            if i.has_lock_prefix() {
                st.lock_prefix += 1;
            }
            let rep = i.has_rep_prefix() || i.has_repe_prefix() || i.has_repne_prefix();
            if rep
                && matches!(
                    i.mnemonic(),
                    Mnemonic::Movsb
                        | Mnemonic::Movsw
                        | Mnemonic::Movsd
                        | Mnemonic::Movsq
                        | Mnemonic::Stosb
                        | Mnemonic::Stosw
                        | Mnemonic::Stosd
                        | Mnemonic::Stosq
                        | Mnemonic::Cmpsb
                        | Mnemonic::Scasb
                        | Mnemonic::Cmpsq
                        | Mnemonic::Scasq
                        | Mnemonic::Lodsb
                )
            {
                st.rep_string += 1;
            }
            match i.segment_prefix() {
                Register::GS => st.gs_access += 1,
                Register::FS => st.fs_access += 1,
                _ => {}
            }
            match i.mnemonic() {
                Mnemonic::Syscall | Mnemonic::Sysenter | Mnemonic::Int | Mnemonic::Int1 => {
                    st.syscall_like += 1
                }
                Mnemonic::Cpuid | Mnemonic::Rdtsc | Mnemonic::Rdtscp | Mnemonic::Xgetbv => {
                    st.cpuid_rdtsc += 1
                }
                Mnemonic::Xchg
                    if i.op0_kind() == OpKind::Memory || i.op1_kind() == OpKind::Memory =>
                {
                    st.xchg_mem += 1
                }
                _ => {}
            }
            let (mut x, mut y, mut z) = (false, false, false);
            for k in 0..i.op_count() {
                if i.op_kind(k) == OpKind::Register {
                    let r = i.op_register(k);
                    x |= r.is_xmm();
                    y |= r.is_ymm();
                    z |= r.is_zmm();
                }
            }
            st.xmm_ins += (x && !y && !z) as u64;
            st.ymm_ins += y as u64;
            st.zmm_ins += z as u64;
            // Memory stores into executable sections: self-modifying-code sign.
            if i.is_ip_rel_memory_operand()
                && i.op0_kind() == OpKind::Memory
                && !matches!(
                    i.mnemonic(),
                    Mnemonic::Cmp
                        | Mnemonic::Test
                        | Mnemonic::Lea
                        | Mnemonic::Jmp
                        | Mnemonic::Call
                        | Mnemonic::Nop
                        | Mnemonic::Prefetcht0
                        | Mnemonic::Prefetchw
                )
                && in_exec(i.ip_rel_memory_address().wrapping_sub(base) as u32)
            {
                st.store_into_exec += 1;
            }
            match i.flow_control() {
                FlowControl::Call => {
                    st.call_direct += 1;
                    let t = (i.near_branch_target().wrapping_sub(base)) as u32;
                    if !starts.contains(&t) {
                        st.call_targets_non_pdata += 1;
                        if in_exec(t) {
                            st.call_targets_into_exec_gaps += 1;
                            st.gap_targets.insert(t);
                        }
                    }
                }
                FlowControl::IndirectCall => match i.op0_kind() {
                    OpKind::Register => st.call_indirect_reg += 1,
                    _ if i.is_ip_rel_memory_operand() => st.call_indirect_mem_rip += 1,
                    _ => st.call_indirect_mem_other += 1,
                },
                FlowControl::UnconditionalBranch => {
                    st.jmp_direct += 1;
                    let t = (i.near_branch_target().wrapping_sub(base)) as u32;
                    if t < f.begin || t >= f.end {
                        st.tail_jmp_out_of_func += 1;
                        if !starts.contains(&t) && in_exec(t) {
                            st.gap_targets.insert(t);
                        }
                    }
                }
                FlowControl::ConditionalBranch => st.jmp_cond += 1,
                FlowControl::IndirectBranch => {
                    if i.op0_kind() == OpKind::Register {
                        st.jmp_indirect_reg += 1;
                        if !table_jmps.contains(&((i.ip() - base) as u32)) {
                            st.jmp_reg_without_table += 1;
                        }
                    } else {
                        st.jmp_indirect_mem += 1;
                    }
                }
                FlowControl::Return => st.rets += 1,
                _ => {}
            }
        }
    }
    st
}

pub fn analyze(img: &Image) -> Result<Report> {
    let funcs = img.functions()?;
    let starts: HashSet<u32> = funcs.iter().map(|f| f.begin).collect();
    let exec_ranges: Vec<(u32, u32)> = img
        .info
        .sections
        .iter()
        .filter(|s| s.characteristics & SCN_EXEC != 0)
        .map(|s| (s.rva, s.rva + s.virtual_size.max(s.file_size)))
        .collect();
    let threads = std::thread::available_parallelism().map_or(4, |n| n.get());
    let chunk = funcs.len().div_ceil(threads).max(1);
    let mut stats = Stats::default();
    std::thread::scope(|sc| {
        let hs: Vec<_> = funcs
            .chunks(chunk)
            .map(|c| sc.spawn(|| analyze_chunk(img, c, &starts, &exec_ranges)))
            .collect();
        for h in hs {
            stats.merge(h.join().unwrap());
        }
    });

    let mut facts = ImageFacts::default();
    for s in &img.info.sections {
        if s.characteristics & SCN_EXEC != 0 {
            facts.image_bytes_in_exec_sections += s.virtual_size as u64;
            facts.exec_sections.push((
                s.name.clone(),
                s.rva,
                s.virtual_size,
                s.characteristics & SCN_WRITE != 0,
            ));
            if s.characteristics & SCN_WRITE != 0 {
                facts.wx_sections.push(s.name.clone());
            }
        }
    }
    facts.pdata_covered_bytes = stats.code_bytes;
    facts.exports = img.info.exports.len();
    facts.exports_not_in_pdata = img
        .info
        .exports
        .iter()
        .filter(|e| {
            e.rva
                .is_some_and(|r| img.is_exec(r) && !starts.contains(&r))
        })
        .count();
    facts.entry_in_pdata = starts.contains(&img.info.entry_rva);
    if let Some(t) = &img.info.tls {
        facts.tls_callbacks = t.callback_rvas.len();
        facts.tls_callbacks_not_in_pdata = t
            .callback_rvas
            .iter()
            .filter(|r| !starts.contains(r))
            .count();
    }
    let relocs = img.dir64_relocs();
    facts.relocs_dir64 = relocs.len() as u64;
    let mut reloc_gap: HashSet<u32> = HashSet::new();
    for r in relocs {
        let Some(b) = img.bytes(r, 8) else { continue };
        let v = u64::from_le_bytes(b.try_into().unwrap());
        let t = v.wrapping_sub(img.info.image_base);
        if t > u32::MAX as u64 {
            continue;
        }
        let t = t as u32;
        if img.is_exec(t) {
            facts.relocs_into_exec += 1;
            if starts.contains(&t) {
                facts.relocs_to_func_start += 1;
            } else {
                facts.relocs_into_exec_not_func_start += 1;
                reloc_gap.insert(t);
            }
        }
    }
    facts.reloc_gap_targets = reloc_gap.len();
    facts.gap_targets_union = reloc_gap.union(&stats.gap_targets).count();
    const SUS: &[&str] = &[
        "VirtualProtect",
        "VirtualAlloc",
        "VirtualAllocEx",
        "FlushInstructionCache",
        "WriteProcessMemory",
        "NtProtectVirtualMemory",
        "VirtualProtectFromApp",
        "VirtualAllocFromApp",
        "MapViewOfFile3",
        "MapViewOfFileFromApp",
        "CreateFileMappingFromApp",
    ];
    for m in &img.info.imports {
        for e in &m.entries {
            let n = e.symbol.to_string();
            if SUS.iter().any(|s| n == *s) {
                facts.suspicious_imports.push(format!("{}!{}", m.dll, n));
            }
        }
    }
    Ok(Report { stats, facts })
}

impl Report {
    pub fn render(&self) -> String {
        let (s, f) = (&self.stats, &self.facts);
        let mut o = String::new();
        let pct = |a: u64, b: u64| {
            if b == 0 {
                0.0
            } else {
                a as f64 * 100.0 / b as f64
            }
        };
        let mut sizes = s.func_sizes.clone();
        sizes.sort_unstable();
        let med = sizes.get(sizes.len() / 2).copied().unwrap_or(0);
        let p99 = sizes.get(sizes.len() * 99 / 100).copied().unwrap_or(0);
        let _ = writeln!(
            o,
            "functions (.pdata entries): {}  (chained/cold parts: {})",
            s.functions, s.chained
        );
        let _ = writeln!(
            o,
            "functions with a decode failure: {}",
            s.decode_failed_funcs
        );
        let _ = writeln!(
            o,
            "code bytes covered by .pdata: {}  (exec section virtual bytes: {}, {:.1}% covered)",
            s.code_bytes,
            f.image_bytes_in_exec_sections,
            pct(s.code_bytes, f.image_bytes_in_exec_sections)
        );
        let _ = writeln!(
            o,
            "instructions: {}  (avg {:.2} bytes)",
            s.instructions,
            s.code_bytes as f64 / s.instructions.max(1) as f64
        );
        let _ = writeln!(
            o,
            "function size bytes: median {med}, p99 {p99}, max {}; largest function {} instructions",
            sizes.last().copied().unwrap_or(0),
            s.max_func_instrs
        );
        let _ = writeln!(o, "\ncontrol flow");
        let _ = writeln!(
            o,
            "  direct calls {}  (to non-.pdata targets: {}, of which inside exec sections: {})",
            s.call_direct, s.call_targets_non_pdata, s.call_targets_into_exec_gaps
        );
        let _ = writeln!(
            o,
            "  indirect calls: reg {}  mem[rip] {}  mem[other] {}",
            s.call_indirect_reg, s.call_indirect_mem_rip, s.call_indirect_mem_other
        );
        let _ = writeln!(
            o,
            "  direct jmp {}  (leaving the function: {})  conditional {}",
            s.jmp_direct, s.tail_jmp_out_of_func, s.jmp_cond
        );
        let _ = writeln!(
            o,
            "  indirect jmp: reg {}  mem {}  ret {}",
            s.jmp_indirect_reg, s.jmp_indirect_mem, s.rets
        );
        let _ = writeln!(
            o,
            "  recovered jump tables {} ({} entries); jmp reg with no recovered table: {}",
            s.jump_tables, s.jump_table_entries, s.jmp_reg_without_table
        );
        let _ = writeln!(
            o,
            "\nextensions (instruction counts by CPUID feature; an instruction can list several)"
        );
        let mut feats: Vec<_> = s.features.iter().collect();
        feats.sort_by(|a, b| b.1.cmp(a.1));
        for (k, v) in feats {
            let _ = writeln!(o, "  {k:<16} {v}");
        }
        let _ = writeln!(o, "  x87 (FPU) instructions: {}", s.x87);
        let _ = writeln!(
            o,
            "  touching xmm only: {}  ymm: {}  zmm: {}",
            s.xmm_ins, s.ymm_ins, s.zmm_ins
        );
        let _ = writeln!(o, "\nother");
        let _ = writeln!(
            o,
            "  lock prefix {}  xchg-mem {}  rep string ops {}",
            s.lock_prefix, s.xchg_mem, s.rep_string
        );
        let _ = writeln!(
            o,
            "  gs-prefixed {}  fs-prefixed {}  syscall/int {}  cpuid/rdtsc/xgetbv {}",
            s.gs_access, s.fs_access, s.syscall_like, s.cpuid_rdtsc
        );
        let _ = writeln!(o, "\nimage / discovery");
        for (n, rva, sz, w) in &f.exec_sections {
            let _ = writeln!(
                o,
                "  exec section {n} rva {rva:#x} size {sz:#x}{}",
                if *w { " WRITABLE" } else { "" }
            );
        }
        let _ = writeln!(
            o,
            "  exports {} (not in .pdata: {})  entry in .pdata: {}",
            f.exports, f.exports_not_in_pdata, f.entry_in_pdata
        );
        let _ = writeln!(
            o,
            "  tls callbacks {} (not in .pdata: {})",
            f.tls_callbacks, f.tls_callbacks_not_in_pdata
        );
        let _ = writeln!(
            o,
            "  DIR64 relocations {}  into exec {} (at .pdata function starts {}, elsewhere {})",
            f.relocs_dir64,
            f.relocs_into_exec,
            f.relocs_to_func_start,
            f.relocs_into_exec_not_func_start
        );
        let _ = writeln!(
            o,
            "  distinct non-.pdata targets: from calls/tail jmps {}, from relocs {}, union {} (leaf functions and thunks the .pdata seed misses)",
            s.gap_targets.len(),
            f.reloc_gap_targets,
            f.gap_targets_union
        );
        let _ = writeln!(o, "\nself-modifying / JIT signs");
        let _ = writeln!(o, "  W+X sections: {:?}", f.wx_sections);
        let _ = writeln!(
            o,
            "  rip-relative stores into exec sections: {}",
            s.store_into_exec
        );
        let _ = writeln!(o, "  suspicious imports: {:?}", f.suspicious_imports);
        let _ = writeln!(o, "\ntop mnemonics");
        let mut m: Vec<_> = s.mnemonics.iter().collect();
        m.sort_by(|a, b| b.1.cmp(a.1).then(a.0.cmp(b.0)));
        for (k, v) in m.iter().take(40) {
            let _ = writeln!(o, "  {k:<12} {v}  ({:.2}%)", pct(**v, s.instructions));
        }
        o
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use iced_x86::code_asm::*;
    use iced_x86::BlockEncoderOptions;

    const BASE: u64 = 0x1_4000_0000;
    const CODE: u64 = BASE + 0x1000;

    fn decode(bytes: &[u8]) -> Vec<Instruction> {
        let mut d = Decoder::with_ip(64, bytes, CODE, DecoderOptions::NONE);
        let mut v = vec![];
        while d.can_decode() {
            v.push(d.decode());
        }
        v
    }

    fn labels(ins: &[Instruction]) -> HashSet<u64> {
        ins.iter()
            .filter(|i| {
                matches!(i.flow_control(), FlowControl::ConditionalBranch | FlowControl::UnconditionalBranch)
                    && i.op0_kind() == OpKind::NearBranch64
            })
            .map(|i| i.near_branch_target())
            .collect()
    }

    /// The bound `index_bound` finds for the last scaled-by-4 load of the code `build` writes (ahead of 64 bytes of
    /// table, after the code).
    fn bound(build: impl FnOnce(&mut CodeAssembler, CodeLabel) -> Result<(), IcedError>) -> Option<Bound> {
        let mut a = CodeAssembler::new(64).unwrap();
        let mut table = a.create_label();
        build(&mut a, table).unwrap();
        a.set_label(&mut table).unwrap();
        a.dd(&[0; 16]).unwrap();
        let r = a.assemble_options(CODE, BlockEncoderOptions::RETURN_NEW_INSTRUCTION_OFFSETS).unwrap();
        let end = (r.label_ip(&table).unwrap() - CODE) as usize;
        let ins = decode(&r.inner.code_buffer[..end]);
        let ld_at = ins
            .iter()
            .rposition(|i| i.op1_kind() == OpKind::Memory && i.memory_index_scale() == 4)
            .unwrap();
        index_bound(&ins, 0, ld_at, &labels(&ins), &mut InstructionInfoFactory::new())
    }

    #[test]
    fn a_check_of_the_index_bounds_it() {
        let b = bound(|a, t| {
            let mut def = a.create_label();
            a.cmp(ecx, 5)?;
            a.ja(def)?;
            a.movsxd(rax, ecx)?;
            a.lea(rdx, ptr(t))?;
            a.movsxd(rax, dword_ptr(rdx + rax * 4))?;
            a.add(rax, rdx)?;
            a.jmp(rax)?;
            a.set_label(&mut def)?;
            a.ret()
        });
        assert_eq!(b, Some(Bound::Entries(6)));
        // `jae` and an 8-bit compare.
        let b = bound(|a, t| {
            let mut def = a.create_label();
            a.cmp(al, 9)?;
            a.jae(def)?;
            a.movzx(eax, al)?;
            a.lea(rdx, ptr(t))?;
            a.movsxd(rax, dword_ptr(rdx + rax * 4))?;
            a.add(rax, rdx)?;
            a.jmp(rax)?;
            a.set_label(&mut def)?;
            a.ret()
        });
        assert_eq!(b, Some(Bound::Entries(9)));
    }

    #[test]
    fn a_check_of_a_value_changed_after_it_does_not() {
        // The a3 trap (rva 0x53bf9ef): `cmp r8d, 7` / `jae`, then `add r8d, 2`; the table had 8 entries, the check
        // gave 7.
        let b = bound(|a, t| {
            let mut out = a.create_label();
            a.cmp(r8d, 7)?;
            a.jae(out)?;
            a.add(r8d, 2)?;
            a.lea(r9, ptr(t))?;
            a.movsxd(r8, dword_ptr(r9 + r8 * 4))?;
            a.add(r8, r9)?;
            a.jmp(r8)?;
            a.set_label(&mut out)?;
            a.ret()
        });
        assert_eq!(b, None);
        // Nor a check of another value: the index is a loop counter copied after the check.
        let b = bound(|a, t| {
            let mut out = a.create_label();
            a.cmp(ecx, 3)?;
            a.ja(out)?;
            a.mov(ecx, r12d)?;
            a.lea(rsi, ptr(t))?;
            a.movsxd(rcx, dword_ptr(rsi + rcx * 4))?;
            a.add(rcx, rsi)?;
            a.jmp(rcx)?;
            a.set_label(&mut out)?;
            a.ret()
        });
        assert_eq!(b, None);
    }

    #[test]
    fn a_path_that_joins_after_the_check_is_unchecked() {
        let b = bound(|a, t| {
            let (mut def, mut join, mut other) = (a.create_label(), a.create_label(), a.create_label());
            a.cmp(ecx, 5)?;
            a.ja(def)?;
            a.set_label(&mut join)?;
            a.lea(rdx, ptr(t))?;
            a.movsxd(rax, dword_ptr(rdx + rcx * 4))?;
            a.add(rax, rdx)?;
            a.jmp(rax)?;
            a.set_label(&mut other)?;
            a.mov(ecx, 9)?;
            a.jmp(join)?;
            a.set_label(&mut def)?;
            a.ret()
        });
        assert_eq!(b, None);
    }

    #[test]
    fn a_copy_made_ahead_of_the_check_is_checked_with_it() {
        // `movzx r10d, dl` ... `cmp dl, 7`, the index r10 (rva 0x22e39e3); an outer, looser check first.
        let b = bound(|a, t| {
            let (mut def, mut high) = (a.create_label(), a.create_label());
            a.cmp(dl, 0x28)?;
            a.ja(def)?;
            a.movzx(r10d, dl)?;
            a.cmp(dl, 7)?;
            a.ja(high)?;
            a.lea(rax, ptr(t))?;
            a.movsxd(rdx, dword_ptr(rax + r10 * 4))?;
            a.add(rdx, rax)?;
            a.jmp(rdx)?;
            a.set_label(&mut high)?;
            a.nop()?;
            a.set_label(&mut def)?;
            a.ret()
        });
        assert_eq!(b, Some(Bound::Entries(8)));
        // Two copies of one argument, the check on one, the index the other (rva 0xbdc05c).
        let b = bound(|a, t| {
            let mut def = a.create_label();
            a.mov(ebp, r9d)?;
            a.mov(r12d, r9d)?;
            a.cmp(ebp, 0x10)?;
            a.ja(def)?;
            a.lea(rax, ptr(t))?;
            a.movsxd(rcx, dword_ptr(rax + r12 * 4))?;
            a.add(rcx, rax)?;
            a.jmp(rcx)?;
            a.set_label(&mut def)?;
            a.ret()
        });
        assert_eq!(b, Some(Bound::Entries(17)));
    }

    #[test]
    fn a_call_ends_what_volatile_registers_held() {
        let switch = |checked: AsmRegister32, index: AsmRegister64| {
            bound(move |a, t| {
                let (mut def, mut callee) = (a.create_label(), a.create_label());
                a.cmp(checked, 5)?;
                a.ja(def)?;
                a.call(callee)?;
                a.lea(rdx, ptr(t))?;
                a.movsxd(rax, dword_ptr(rdx + index * 4))?;
                a.add(rax, rdx)?;
                a.jmp(rax)?;
                a.set_label(&mut callee)?;
                a.ret()?;
                a.set_label(&mut def)?;
                a.ret()
            })
        };
        assert_eq!(switch(ecx, rcx), None);
        assert_eq!(switch(ebx, rbx), Some(Bound::Entries(6)));
    }

    #[test]
    fn msvc_two_level_switches_are_bounded_by_the_byte_table() {
        // `cmp eax, 0x1c` / `ja`, `cdqe`, the byte table indexed by the checked value, the offsets by its byte.
        let b = bound(|a, t| {
            let mut def = a.create_label();
            a.cmp(eax, 0x1c)?;
            a.ja(def)?;
            a.lea(r8, ptr(t))?;
            a.cdqe()?;
            a.movzx(eax, byte_ptr(r8 + rax + 0x50))?;
            a.mov(ecx, dword_ptr(r8 + rax * 4 + 0x48))?;
            a.add(rcx, r8)?;
            a.jmp(rcx)?;
            a.set_label(&mut def)?;
            a.ret()
        });
        assert_eq!(b, Some(Bound::Bytes { at: 0x50, entries: 0x1d }));
    }

    /// Two switches over tables side by side: the first reached only by a jump (no check reaches it), the second
    /// checked to 8 cases.
    fn side_by_side() -> (Vec<u8>, Func, Vec<u32>) {
        let mut entries: Vec<u32> = vec![0; 16];
        for pass in 0..2 {
            let mut a = CodeAssembler::new(64).unwrap();
            let mut cases: Vec<CodeLabel> = (0..8).map(|_| a.create_label()).collect();
            let (mut first, mut second, mut join, mut def) =
                (a.create_label(), a.create_label(), a.create_label(), a.create_label());
            a.jmp(join).unwrap();
            a.set_label(&mut join).unwrap();
            a.lea(r9, ptr(first)).unwrap();
            a.movsxd(r8, dword_ptr(r9 + r8 * 4)).unwrap();
            a.add(r8, r9).unwrap();
            a.jmp(r8).unwrap();
            a.cmp(ecx, 7).unwrap();
            a.ja(def).unwrap();
            a.lea(rdx, ptr(second)).unwrap();
            a.movsxd(rax, dword_ptr(rdx + rcx * 4)).unwrap();
            a.add(rax, rdx).unwrap();
            a.jmp(rax).unwrap();
            for c in cases.iter_mut() {
                a.set_label(c).unwrap();
                a.mov(eax, 1).unwrap();
                a.ret().unwrap();
            }
            a.set_label(&mut def).unwrap();
            a.ret().unwrap();
            a.set_label(&mut first).unwrap();
            a.dd(&entries[..8]).unwrap();
            a.set_label(&mut second).unwrap();
            a.dd(&entries[8..]).unwrap();
            let r = a.assemble_options(CODE, BlockEncoderOptions::RETURN_NEW_INSTRUCTION_OFFSETS).unwrap();
            let ip = |l: &CodeLabel| r.label_ip(l).unwrap();
            let rva = |x: u64| (x - BASE) as u32;
            if pass == 0 {
                let (t1, t2) = (ip(&first), ip(&second));
                // The second table's cases in the reverse order, so read from the first table's base they land
                // elsewhere in the function.
                for k in 0..8 {
                    entries[k] = ip(&cases[k]).wrapping_sub(t1) as u32;
                    entries[8 + k] = ip(&cases[7 - k]).wrapping_sub(t2) as u32;
                }
            } else {
                let f = Func { begin: rva(CODE), end: rva(ip(&first)), unwind: 0, chained: false };
                let cases = cases.iter().map(|c| rva(ip(c))).collect();
                return (r.inner.code_buffer, f, cases);
            }
        }
        unreachable!()
    }

    /// An interpreter's dispatch (rva 0x3de2612): `jmp qword ptr [r12 + 8*rbx]` through a table of the handlers'
    /// addresses, its base set once at the top. (`entries`, the table's 64-bit words, is filled from the code's layout.)
    fn interpreter() -> (Vec<u8>, Func, Vec<u32>) {
        let mut words: Vec<u64> = vec![0; 3];
        for pass in 0..2 {
            let mut a = CodeAssembler::new(64).unwrap();
            let (mut table, mut dispatch, mut h0, mut h1, mut h2) =
                (a.create_label(), a.create_label(), a.create_label(), a.create_label(), a.create_label());
            a.lea(r12, ptr(table)).unwrap();
            a.set_label(&mut dispatch).unwrap();
            a.movzx(ebx, byte_ptr(rsi)).unwrap();
            a.inc(rsi).unwrap();
            a.jmp(qword_ptr(r12 + rbx * 8)).unwrap();
            for (h, n) in [(&mut h0, 1), (&mut h1, 2)] {
                a.set_label(h).unwrap();
                a.add(eax, n).unwrap();
                a.jmp(dispatch).unwrap();
            }
            a.set_label(&mut h2).unwrap();
            a.ret().unwrap();
            a.set_label(&mut table).unwrap();
            a.dq(&words).unwrap();
            let r = a.assemble_options(CODE, BlockEncoderOptions::RETURN_NEW_INSTRUCTION_OFFSETS).unwrap();
            let ip = |l: &CodeLabel| r.label_ip(l).unwrap();
            if pass == 0 {
                words = vec![ip(&h0), ip(&h1), ip(&h2)];
            } else {
                let rva = |x: u64| (x - BASE) as u32;
                let f = Func { begin: rva(CODE), end: rva(ip(&table)), unwind: 0, chained: false };
                let handlers = vec![rva(ip(&h0)), rva(ip(&h1)), rva(ip(&h2))];
                return (r.inner.code_buffer, f, handlers);
            }
        }
        unreachable!()
    }

    #[test]
    fn a_table_of_addresses_gives_the_dispatch_its_handlers() {
        let (bytes, f, handlers) = interpreter();
        let read = |rva: u32| {
            let o = (rva - f.begin) as usize;
            bytes.get(o..o + 4).map(|b| u32::from_le_bytes(b.try_into().unwrap()))
        };
        let ins = decode(&bytes[..(f.end - f.begin) as usize]);
        let tables = find_jump_tables_with(BASE, &read, &f, &ins);
        assert_eq!(tables.len(), 1);
        assert_eq!(tables[0].targets, handlers);
        assert!(!tables[0].bounded);
    }

    #[test]
    fn a_jump_through_a_register_after_a_switch_is_not_the_switch() {
        // rva 0x1ffa540: the switch's `jmp rax`, then in a case a virtual call in tail position, `jmp rax` again
        // within 40 instructions of the switch's load and add.
        let mut words: Vec<u32> = vec![0; 2];
        let (bytes, f, switch, case0) = loop {
            let mut a = CodeAssembler::new(64).unwrap();
            let (mut table, mut def, mut c0, mut c1) = (a.create_label(), a.create_label(), a.create_label(), a.create_label());
            a.cmp(rax, 1).unwrap();
            a.ja(def).unwrap();
            a.lea(r8, ptr(table)).unwrap();
            a.movsxd(rax, dword_ptr(r8 + rax * 4)).unwrap();
            a.add(rax, r8).unwrap();
            a.jmp(rax).unwrap();
            a.set_label(&mut c0).unwrap();
            a.mov(rax, qword_ptr(rcx)).unwrap();
            a.mov(rax, qword_ptr(rax + 0x78)).unwrap();
            a.jmp(rax).unwrap();
            a.set_label(&mut c1).unwrap();
            a.nop().unwrap();
            a.set_label(&mut def).unwrap();
            a.ret().unwrap();
            a.set_label(&mut table).unwrap();
            a.dd(&words).unwrap();
            let r = a.assemble_options(CODE, BlockEncoderOptions::RETURN_NEW_INSTRUCTION_OFFSETS).unwrap();
            let ip = |l: &CodeLabel| r.label_ip(l).unwrap();
            let t = ip(&table);
            let want = vec![ip(&c0).wrapping_sub(t) as u32, ip(&c1).wrapping_sub(t) as u32];
            if want == words {
                let rva = |x: u64| (x - BASE) as u32;
                let f = Func { begin: rva(CODE), end: rva(t), unwind: 0, chained: false };
                let (switch, case0) = (rva(ip(&c0)) - 2, rva(ip(&c0)));
                break (r.inner.code_buffer, f, switch, case0);
            }
            words = want;
        };
        let read = |rva: u32| {
            let o = (rva - f.begin) as usize;
            bytes.get(o..o + 4).map(|b| u32::from_le_bytes(b.try_into().unwrap()))
        };
        let ins = decode(&bytes[..(f.end - f.begin) as usize]);
        let tables = find_jump_tables_with(BASE, &read, &f, &ins);
        assert_eq!(tables.len(), 1, "{tables:x?}");
        assert_eq!(tables[0].jmp_rva, switch);
        assert_eq!(tables[0].targets[0], case0);
    }

    #[test]
    fn the_base_may_come_back_in_another_register() {
        // Bink (rva 0x1751d of its DLL): the offset is loaded over the base register, and the base set again in
        // another one for the add: `mov edx, [rdx + 4*rcx + table]` / `lea rcx, [__ImageBase]` / `add rdx, rcx`.
        let ins = decode(
            &CodeAssembler::new(64)
                .and_then(|mut a| {
                    let mut image = a.create_label();
                    a.set_label(&mut image)?;
                    a.lea(rdx, ptr(image))?;
                    a.mov(edx, dword_ptr(rdx + rcx * 4 + 0x40))?;
                    a.lea(rcx, ptr(image))?;
                    a.add(rdx, rcx)?;
                    a.jmp(rdx)?;
                    a.assemble(CODE)
                })
                .unwrap(),
        );
        let labels = HashSet::new();
        let base_va = ins[0].ip_rel_memory_address();
        assert!(feeds_jump(&ins, 1, 4, Register::RDX, base_va, &labels, &mut InstructionInfoFactory::new()));
        // A base the add takes from a register that no longer holds it does not.
        let ins = decode(
            &CodeAssembler::new(64)
                .and_then(|mut a| {
                    let mut image = a.create_label();
                    a.set_label(&mut image)?;
                    a.lea(rdx, ptr(image))?;
                    a.mov(edx, dword_ptr(rdx + rcx * 4 + 0x40))?;
                    a.mov(rcx, qword_ptr(rsp + 8))?;
                    a.add(rdx, rcx)?;
                    a.jmp(rdx)?;
                    a.assemble(CODE)
                })
                .unwrap(),
        );
        assert!(!feeds_jump(&ins, 1, 4, Register::RDX, base_va, &labels, &mut InstructionInfoFactory::new()));
    }

    #[test]
    fn an_unchecked_table_ends_where_the_next_one_starts() {
        let (bytes, f, cases) = side_by_side();
        let read = |rva: u32| {
            let o = (rva - f.begin) as usize;
            bytes.get(o..o + 4).map(|b| u32::from_le_bytes(b.try_into().unwrap()))
        };
        let ins = decode(&bytes[..(f.end - f.begin) as usize]);
        let tables = find_jump_tables_with(BASE, &read, &f, &ins);
        assert_eq!(tables.len(), 2);
        assert!(!tables[0].bounded);
        assert_eq!(tables[0].targets, cases, "the first table's own 8 cases, none of the second's");
        assert!(tables[1].bounded);
        assert_eq!(tables[1].targets, cases.iter().rev().copied().collect::<Vec<_>>());
    }
}
