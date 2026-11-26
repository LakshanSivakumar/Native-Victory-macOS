//! x86-64 -> C lifter. Each guest function becomes `void f_<rva>(Cpu *c)`:
//! registers live in C locals (so clang can allocate them freely), EFLAGS are
//! lazy (`Fl`), and memory is host memory addressed 1:1. See `docs/recompiler.md`.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as _;

use iced_x86::{ConditionCode, Instruction, Mnemonic, OpKind, Register};

pub const RT_HEADER: &str = concat!(include_str!("rt.h"), include_str!("rt_vec.h"));

mod bits;
mod legacy;
mod vex;

pub struct FuncIn<'a> {
    pub begin_va: u64,
    pub end_va: u64,
    pub ins: &'a [Instruction],
    /// Jump-table dispatches: `jmp reg` address -> target addresses (all VAs).
    pub tables: BTreeMap<u64, Vec<u64>>,
    /// Chained `.pdata` parts (cold blocks, funclets) merged into this function: `[begin, end)` VAs.
    /// Their instructions follow the parent's in `ins`, so a jump into them is a `goto`.
    pub extra: Vec<(u64, u64)>,
    /// Addresses an unwinder resumes at (the `__except` blocks of `__try`): VAs inside the function. Such a
    /// function gets a frame record and an entry at each, so the host can drop the frames below it and
    /// continue there with the registers of the unwound context (`docs/recompiler.md`, "Exceptions").
    pub landing_pads: Vec<u64>,
    /// Addresses inside the function that code outside it jumps or calls to directly (hand-written CRT routines
    /// share their tails): VAs of its instructions. Each gets a function of its own, `f_<rva>`, the same body
    /// entered there. (Not with landing pads: such functions have none.)
    pub entries: Vec<u64>,
}

impl FuncIn<'_> {
    pub fn contains(&self, va: u64) -> bool {
        (va >= self.begin_va && va < self.end_va) || self.extra.iter().any(|&(b, e)| va >= b && va < e)
    }
}

pub struct Ctx<'a> {
    pub image_base: u64,
    /// Is this VA the start of a function in the unit (direct call target)?
    pub is_func: &'a (dyn Fn(u64) -> bool + Sync),
}

#[derive(Default)]
pub struct Unit {
    pub c: String,
    pub lifted: Vec<u64>,
    pub failed: Vec<(u64, String)>,
}

/// One translation unit of a split build: bodies only, `#include "rt.h"`, declarations just for the
/// functions it calls. The dispatch table is emitted once by `table_unit`.
pub fn lift_part(ctx: &Ctx, funcs: &[FuncIn]) -> Unit {
    let mut u = Unit::default();
    let mut body = String::new();
    for f in funcs {
        match lift_function(ctx, f) {
            Ok(s) => {
                body.push_str(&s);
                u.lifted.push(f.begin_va);
            }
            Err(e) => {
                for va in std::iter::once(f.begin_va).chain(f.entries.iter().copied()) {
                    let _ = writeln!(
                        body,
                        "void {}(Cpu *c) {{ (void)c; rt_trap(\"unlifted function {:#x}: {}\"); }}",
                        name(va, ctx.image_base),
                        f.begin_va - ctx.image_base,
                        e.replace('"', "'")
                    );
                }
                body.push_str(&pad_defs(ctx, f));
                u.failed.push((f.begin_va, e));
            }
        }
    }
    let mut called = BTreeSet::new();
    let b = body.as_bytes();
    let mut k = 0;
    while k + 3 < b.len() {
        if b[k] == b'f' && b[k + 1] == b'_' && (k == 0 || !(b[k - 1].is_ascii_alphanumeric() || b[k - 1] == b'_')) {
            let mut e = k + 2;
            while e < b.len() && b[e].is_ascii_hexdigit() {
                e += 1;
            }
            if e > k + 2 && body[e..].starts_with("(c)") {
                called.insert(body[k..e].to_string());
            }
            k = e;
        } else {
            k += 1;
        }
    }
    let _ = writeln!(u.c, "#define IMAGE_BASE {:#x}ull\n#include \"rt.h\"", ctx.image_base);
    for n in &called {
        let _ = writeln!(u.c, "void {n}(Cpu *c);");
    }
    u.c.push_str(&body);
    u
}

/// The dispatch table for a split build: every function declared, sorted by RVA.
pub fn table_unit(image_base: u64, starts: &[u64], pads: &[(u64, u64)]) -> String {
    let mut c = format!("#define IMAGE_BASE {image_base:#x}ull\n#include \"rt.h\"\n");
    let mut sorted = starts.to_vec();
    sorted.sort_unstable();
    for va in &sorted {
        let _ = writeln!(c, "void {}(Cpu *c);", name(*va, image_base));
    }
    let mut pads = pads.to_vec();
    pads.sort_unstable_by_key(|p| p.1);
    pads.dedup_by_key(|p| p.1);
    for &(f, p) in &pads {
        let _ = writeln!(c, "void {}(Cpu *c);", pad_name(f, p, image_base));
    }
    let mut entries: Vec<(u64, String, u32)> = sorted.iter().map(|&va| (va, name(va, image_base), 0)).collect();
    entries.extend(pads.iter().map(|&(f, p)| (p, pad_name(f, p, image_base), 1)));
    entries.sort_unstable_by_key(|e| e.0);
    let _ = writeln!(c, "const RtEntry rt_table[] = {{");
    for (va, n, pad) in &entries {
        let _ = writeln!(c, "  {{ {:#x}, {n}, {pad} }},", va - image_base);
    }
    let _ = writeln!(c, "}};\nconst unsigned rt_table_n = {};", entries.len());
    c
}

fn pad_name(func: u64, pad: u64, base: u64) -> String {
    format!("pad_{:x}_{:x}", func - base, pad - base)
}

/// What a jump to a landing pad of `f` does: resume the function's frame there, with the registers the jumper has
/// (`rt_resume`, with no return address: it is not a call). One per pad, whether or not the body could be lifted.
fn pad_defs(ctx: &Ctx, f: &FuncIn) -> String {
    let mut o = String::new();
    let mut pads = f.landing_pads.clone();
    pads.sort_unstable();
    pads.dedup();
    for p in pads {
        let _ = writeln!(
            o,
            "void {}(Cpu *c) {{ rt_resume(c->r, c->x, (u64)H({p:#x}ull), 0); }}",
            pad_name(f.begin_va, p, ctx.image_base)
        );
    }
    o
}

fn name(va: u64, base: u64) -> String {
    format!("f_{:x}", va - base)
}

/// Lifts a set of functions into one C translation unit (runtime header
/// included, jump-table dispatch table emitted).
pub fn lift_unit(ctx: &Ctx, funcs: &[FuncIn]) -> Unit {
    let mut u = Unit::default();
    let _ = writeln!(
        u.c,
        "#define IMAGE_BASE {:#x}ull\n{RT_HEADER}",
        ctx.image_base
    );
    for f in funcs {
        let _ = writeln!(u.c, "void {}(Cpu *c);", name(f.begin_va, ctx.image_base));
    }
    for f in funcs {
        match lift_function(ctx, f) {
            Ok(s) => {
                u.c.push_str(&s);
                u.lifted.push(f.begin_va);
            }
            Err(e) => {
                let _ = writeln!(
                    u.c,
                    "void {}(Cpu *c) {{ (void)c; rt_trap(\"unlifted function {:#x}: {}\"); }}",
                    name(f.begin_va, ctx.image_base),
                    f.begin_va - ctx.image_base,
                    e.replace('"', "'")
                );
                u.c.push_str(&pad_defs(ctx, f));
                u.failed.push((f.begin_va, e));
            }
        }
    }
    let starts: Vec<u64> = funcs.iter().map(|f| f.begin_va).collect();
    let pads: Vec<(u64, u64)> = funcs.iter().flat_map(|f| f.landing_pads.iter().map(|&p| (f.begin_va, p))).collect();
    // The table (and `IMG`) are the same text a split build puts in table.c.
    let table = table_unit(ctx.image_base, &starts, &pads);
    u.c.push_str(table.split_once("#include \"rt.h\"\n").map_or("", |t| t.1));
    u
}

struct L<'a> {
    ctx: &'a Ctx<'a>,
    f: &'a FuncIn<'a>,
    o: String,
    /// The function touches vector registers, so ymm upper halves travel with it.
    y: bool,
    /// Addresses of this function's instructions: the only valid `goto` targets.
    ips: std::collections::HashSet<u64>,
    /// Stack reads that are shared accesses all the same ([`spin_reads`]).
    spin: std::collections::HashSet<u64>,
    /// The cases of the function's tables of addresses, which its `JT` switch takes.
    jt: BTreeSet<u64>,
}

type R<T> = Result<T, String>;

/// Reads of a fixed stack slot (`[rsp + d]`) in a polling loop: a short loop that only reads memory and tests what
/// it read (loads, compares, branches; no call, no store, no register arithmetic, so nothing of its own changes from
/// one turn to the next). Only another thread can end it: a waiter spins on a counter it handed out by address
/// (rva 0x9dd0da: workers `lock inc` it, the waiter does `mov eax, [rsp + 0x48]` / `cmp eax, 8` / `jne` back), and
/// as a plain access clang takes the read as loop-invariant and the loop as endless (`b .`). These are emitted as
/// shared accesses, which it must repeat (and which order what follows them, as the spin's exit must). (A loop
/// with `pause` keeps its reads in any case: `pause` is a compiler barrier.)
fn spin_reads(ins: &[Instruction]) -> std::collections::HashSet<u64> {
    use iced_x86::FlowControl as FC;
    let at: std::collections::HashMap<u64, usize> = ins.iter().enumerate().map(|(k, i)| (i.ip(), k)).collect();
    let mut out = std::collections::HashSet::new();
    for (b, j) in ins.iter().enumerate() {
        if !matches!(j.flow_control(), FC::ConditionalBranch | FC::UnconditionalBranch) || j.op0_kind() != OpKind::NearBranch64 {
            continue;
        }
        let t = j.near_branch_target();
        let Some(&s) = at.get(&t).filter(|_| t <= j.ip()) else { continue };
        let body = &ins[s..=b];
        let polls = body.len() <= 16
            && body.iter().all(|i| {
                let load = matches!(i.mnemonic(), Mnemonic::Mov | Mnemonic::Movzx | Mnemonic::Movsx | Mnemonic::Movsxd)
                    && i.op0_kind() == OpKind::Register
                    && i.op1_kind() == OpKind::Memory;
                load || matches!(i.mnemonic(), Mnemonic::Cmp | Mnemonic::Test | Mnemonic::Nop)
                    || matches!(i.flow_control(), FC::ConditionalBranch | FC::UnconditionalBranch)
                        && i.op0_kind() == OpKind::NearBranch64
            });
        if polls {
            out.extend(
                body.iter()
                    .filter(|i| i.memory_base() == Register::RSP && i.memory_index() == Register::None)
                    .filter(|i| (0..i.op_count()).any(|n| i.op_kind(n) == OpKind::Memory))
                    .map(|i| i.ip()),
            );
        }
    }
    out
}

fn gpr(r: Register) -> R<(usize, u32, bool)> {
    if !r.is_gpr() {
        return Err(format!("register {r:?}"));
    }
    let full = r.full_register();
    let idx = full as usize - Register::RAX as usize;
    let high = matches!(r, Register::AH | Register::CH | Register::DH | Register::BH);
    Ok((idx, r.size() as u32 * 8, high))
}

fn mask(bits: u32) -> String {
    if bits == 64 {
        "~0ull".into()
    } else {
        format!("{:#x}ull", (1u64 << bits) - 1)
    }
}

impl L<'_> {
    fn sp(&self) -> &'static str {
        if self.y { "SPILLY()" } else { "SPILL()" }
    }
    fn rl(&self) -> &'static str {
        if self.y { "RELOADY()" } else { "RELOAD()" }
    }

    fn emit(&mut self, s: &str) {
        self.o.push_str(s);
        self.o.push('\n');
    }

    fn addr(&self, i: &Instruction) -> R<String> {
        if i.is_ip_rel_memory_operand() {
            let a = format!("H({:#x}ull)", i.ip_rel_memory_address());
            return Ok(if i.segment_prefix() == Register::GS {
                format!("(u8*)((u64){a})")
            } else {
                a
            });
        }
        let mut p = vec![];
        if i.memory_base() != Register::None {
            if !i.memory_base().is_gpr64() {
                return Err("32-bit address size".into());
            }
            p.push(format!("r{}", gpr(i.memory_base())?.0));
        }
        if i.memory_index() != Register::None {
            if !i.memory_index().is_gpr64() {
                return Err("32-bit address size".into());
            }
            p.push(format!(
                "r{}*{}",
                gpr(i.memory_index())?.0,
                i.memory_index_scale()
            ));
        }
        p.push(format!("(u64)(s64){}ll", i.memory_displacement64() as i64));
        let sum = p.join(" + ");
        Ok(match i.segment_prefix() {
            Register::GS => format!("(u8*)(c->gs_base + {sum})"),
            Register::None | Register::DS | Register::ES | Register::SS | Register::CS => {
                format!("(u8*)({sum})")
            }
            s => return Err(format!("segment {s:?}")),
        })
    }

    fn mem_bits(i: &Instruction) -> u32 {
        i.memory_size().size() as u32 * 8
    }

    /// Operand `n` as a zero-extended u64 expression. `bits` sizes immediates.
    fn rd(&self, i: &Instruction, n: u32, bits: u32) -> R<String> {
        Ok(match i.op_kind(n) {
            OpKind::Register => {
                let (idx, b, high) = gpr(i.op_register(n))?;
                match (b, high) {
                    (64, _) => format!("r{idx}"),
                    (_, true) => format!("(u64)(u8)(r{idx}>>8)"),
                    _ => format!("(u64)(u{b})r{idx}"),
                }
            }
            OpKind::Memory => {
                let a = self.addr(i)?;
                match Self::mem_bits(i) {
                    8 => format!("(u64)ld8({a})"),
                    16 => format!("(u64)ld16({a})"),
                    32 => format!("(u64)ld32({a})"),
                    64 => format!("ld64({a})"),
                    b => return Err(format!("memory size {b}")),
                }
            }
            OpKind::Immediate8
            | OpKind::Immediate16
            | OpKind::Immediate32
            | OpKind::Immediate64
            | OpKind::Immediate8to16
            | OpKind::Immediate8to32
            | OpKind::Immediate8to64
            | OpKind::Immediate32to64 => {
                let v = i.immediate(n);
                let v = if bits == 64 {
                    v
                } else {
                    v & ((1u64 << bits) - 1)
                };
                format!("{v:#x}ull")
            }
            k => return Err(format!("operand kind {k:?}")),
        })
    }

    fn wr(&self, i: &Instruction, n: u32, v: &str) -> R<String> {
        Ok(match i.op_kind(n) {
            OpKind::Register => {
                let (idx, b, high) = gpr(i.op_register(n))?;
                match (b, high) {
                    (64, _) => format!("r{idx} = {v};"),
                    (32, _) => format!("r{idx} = (u32)({v});"),
                    (_, true) => format!("r{idx} = (r{idx} & ~0xff00ull) | ((u64)(u8)({v}) << 8);"),
                    (b, _) => format!("r{idx} = (r{idx} & ~{}) | (u{b})({v});", mask(b)),
                }
            }
            OpKind::Memory => {
                let a = self.addr(i)?;
                match Self::mem_bits(i) {
                    8 => format!("st8({a}, {v});"),
                    16 => format!("st16({a}, {v});"),
                    32 => format!("st32({a}, {v});"),
                    64 => format!("st64({a}, {v});"),
                    b => return Err(format!("memory size {b}")),
                }
            }
            k => return Err(format!("write operand kind {k:?}")),
        })
    }

    fn bits0(i: &Instruction) -> R<u32> {
        match i.op0_kind() {
            OpKind::Register => Ok(gpr(i.op0_register())?.1),
            OpKind::Memory => Ok(Self::mem_bits(i)),
            k => Err(format!("operand kind {k:?}")),
        }
    }

    fn cond(i: &Instruction) -> R<String> {
        let (cf, zf, sf, of, pf) = (
            "fl_cf(&F)",
            "fl_zf(&F)",
            "fl_sf(&F)",
            "fl_of(&F)",
            "fl_pf(&F)",
        );
        Ok(match i.condition_code() {
            ConditionCode::o => of.into(),
            ConditionCode::no => format!("!{of}"),
            ConditionCode::b => cf.into(),
            ConditionCode::ae => format!("!{cf}"),
            ConditionCode::e => zf.into(),
            ConditionCode::ne => format!("!{zf}"),
            ConditionCode::be => format!("({cf} || {zf})"),
            ConditionCode::a => format!("(!{cf} && !{zf})"),
            ConditionCode::s => sf.into(),
            ConditionCode::ns => format!("!{sf}"),
            ConditionCode::p => pf.into(),
            ConditionCode::np => format!("!{pf}"),
            ConditionCode::l => format!("({sf} != {of})"),
            ConditionCode::ge => format!("({sf} == {of})"),
            ConditionCode::le => format!("({zf} || {sf} != {of})"),
            ConditionCode::g => format!("(!{zf} && {sf} == {of})"),
            ConditionCode::None => return Err("no condition".into()),
        })
    }

    /// Statement that transfers control to guest address `t`.
    /// Whether `t` is inside this function (its range or a chained part).
    fn inside(&self, t: u64) -> bool {
        (self.f.begin_va..self.f.end_va).contains(&t) || self.f.extra.iter().any(|&(b, e)| (b..e).contains(&t))
    }

    fn jump_to(&self, t: u64, _labels: &BTreeSet<u64>) -> String {
        if self.ips.contains(&t) {
            format!("goto L_{t:x};")
        } else if (self.ctx.is_func)(t) {
            format!(
                "{{ {}; {}(c); return; }}",
                self.sp(),
                name(t, self.ctx.image_base)
            )
        } else {
            format!("{{ {}; rt_dispatch(c, (u64)H({t:#x}ull)); return; }}", self.sp())
        }
    }

    fn call(&mut self, i: &Instruction) -> R<()> {
        let ret = i.next_ip();
        let push = format!("r4 -= 8; st64((u8*)r4, (u64)H({ret:#x}ull));");
        match i.op0_kind() {
            OpKind::NearBranch64 => {
                let t = i.near_branch_target();
                let callee = if (self.ctx.is_func)(t) {
                    format!("{}(c);", name(t, self.ctx.image_base))
                } else {
                    format!("rt_dispatch(c, (u64)H({t:#x}ull));")
                };
                self.emit(&format!("{{ {push} {}; {callee} {}; }}", self.sp(), self.rl()));
            }
            _ => {
                let t = self.rd(i, 0, 64)?;
                self.emit(&format!(
                    "{{ u64 t = {t}; {push} {}; rt_dispatch(c, t); {}; }}",
                    self.sp(),
                    self.rl()
                ));
            }
        }
        Ok(())
    }

    fn alu(&mut self, i: &Instruction, kind: &str, expr: &str, write: bool, carry: bool) -> R<()> {
        let bits = Self::bits0(i)?;
        let (a, b) = (self.rd(i, 0, bits)?, self.rd(i, 1, bits)?);
        let w = if write {
            self.wr(i, 0, "r")?
        } else {
            String::new()
        };
        let cin = if carry {
            "u64 cin = fl_cf(&F); "
        } else {
            "u64 cin = 0; "
        };
        self.emit(&format!(
            "{{ u64 a = {a}, b = {b}; {cin}(void)cin; u64 r = ({expr}) & {m}; {w} F.k = {kind}; F.sz = {bits}; F.a = a; F.b = b; F.r = r; F.c = cin; }}",
            m = mask(bits)
        ));
        Ok(())
    }

    fn shift(&mut self, i: &Instruction, op: &str) -> R<()> {
        let bits = Self::bits0(i)?;
        let a = self.rd(i, 0, bits)?;
        let cnt = if i.op_count() > 1 {
            format!(
                "({} & {})",
                self.rd(i, 1, 8)?,
                if bits == 64 { 63 } else { 31 }
            )
        } else {
            "1".into()
        };
        let w = self.wr(i, 0, "r")?;
        let m = mask(bits);
        let body = match op {
            "shl" => format!(
                "r = (a << cnt) & {m}; int cf = cnt <= {bits} ? (a >> ({bits} - cnt)) & 1 : 0; int of = (int)(((r >> {s}) & 1) ^ cf);",
                s = bits - 1
            ),
            "shr" => format!(
                "r = a >> cnt; int cf = (a >> (cnt - 1)) & 1; int of = (int)((a >> {s}) & 1);",
                s = bits - 1
            ),
            "sar" => format!(
                "s64 sa = (s64)sx(a, {bits}); r = (u64)(sa >> (cnt > 63 ? 63 : cnt)) & {m}; int cf = (int)((sa >> (cnt > 64 ? 63 : cnt - 1)) & 1); int of = 0;"
            ),
            _ => unreachable!(),
        };
        self.emit(&format!("{{ u64 a = {a}; u64 cnt = {cnt}; u64 r = a; if (cnt) {{ {body} fl_set_raw(&F, {bits}, r, cf, of); }} {w} }}"));
        Ok(())
    }

    fn rotate(&mut self, i: &Instruction, left: bool) -> R<()> {
        let bits = Self::bits0(i)?;
        let a = self.rd(i, 0, bits)?;
        let cnt = if i.op_count() > 1 {
            format!(
                "({} & {})",
                self.rd(i, 1, 8)?,
                if bits == 64 { 63 } else { 31 }
            )
        } else {
            "1".into()
        };
        let w = self.wr(i, 0, "r")?;
        let m = mask(bits);
        let s = bits - 1;
        let body = if left {
            format!(
                "u64 k = cnt % {bits}; r = k ? ((a << k) | (a >> ({bits} - k))) & {m} : a; int cf = r & 1; int of = (int)(((r >> {s}) & 1) ^ cf);"
            )
        } else {
            format!(
                "u64 k = cnt % {bits}; r = k ? ((a >> k) | (a << ({bits} - k))) & {m} : a; int cf = (int)((r >> {s}) & 1); int of = (int)(((r >> {s}) ^ (r >> {})) & 1);",
                bits.saturating_sub(2)
            )
        };
        self.emit(&format!("{{ u64 a = {a}; u64 cnt = {cnt}; u64 r = a; if (cnt) {{ {body} fl_set_cfof(&F, cf, of); }} {w} }}"));
        Ok(())
    }

    // ---- SSE helpers ----
    fn xreg(i: &Instruction, n: u32) -> R<String> {
        let r = i.op_register(n);
        if i.op_kind(n) == OpKind::Register && r.is_xmm() {
            Ok(format!("x{}", r.number()))
        } else {
            Err(format!("xmm operand {r:?}"))
        }
    }
    /// Source operand as 128-bit value.
    fn x128(&self, i: &Instruction, n: u32) -> R<String> {
        match i.op_kind(n) {
            OpKind::Register => Self::xreg(i, n),
            OpKind::Memory => Ok(format!("ld128({})", self.addr(i)?)),
            k => Err(format!("operand kind {k:?}")),
        }
    }
    /// Scalar source: `f32`/`f64` lane 0 or memory.
    fn xscalar(&self, i: &Instruction, n: u32, dbl: bool) -> R<String> {
        match i.op_kind(n) {
            OpKind::Register => Ok(format!(
                "{}.{}[0]",
                Self::xreg(i, n)?,
                if dbl { "g" } else { "f" }
            )),
            OpKind::Memory => Ok(format!(
                "{}({})",
                if dbl { "ldg" } else { "ldf" },
                self.addr(i)?
            )),
            k => Err(format!("operand kind {k:?}")),
        }
    }

    fn sse_arith(&mut self, i: &Instruction, op: &str, dbl: bool, packed: bool) -> R<()> {
        let d = Self::xreg(i, 0)?;
        let (lane, n) = if dbl { ("g", 2) } else { ("f", 4) };
        if packed {
            let s = self.x128(i, 1)?;
            self.emit(&format!("{{ X128 s = {s}; for (int k = 0; k < {n}; k++) {d}.{lane}[k] = nanfix_{lane2}({d}.{lane}[k] {op} s.{lane}[k], {d}.{lane}[k], s.{lane}[k]); }}", lane2 = if dbl { "d" } else { "f" }));
        } else {
            let s = self.xscalar(i, 1, dbl)?;
            let t = if dbl { "d" } else { "f" };
            self.emit(&format!("{{ {ty} s = {s}; {d}.{lane}[0] = nanfix_{t}({d}.{lane}[0] {op} s, {d}.{lane}[0], s); }}", ty = if dbl { "double" } else { "float" }));
        }
        Ok(())
    }

    fn sse(&mut self, i: &Instruction) -> R<bool> {
        use Mnemonic as M;
        let m = i.mnemonic();
        match m {
            M::Addss => self.sse_arith(i, "+", false, false)?,
            M::Subss => self.sse_arith(i, "-", false, false)?,
            M::Mulss => self.sse_arith(i, "*", false, false)?,
            M::Divss => self.sse_arith(i, "/", false, false)?,
            M::Addsd => self.sse_arith(i, "+", true, false)?,
            M::Subsd => self.sse_arith(i, "-", true, false)?,
            M::Mulsd => self.sse_arith(i, "*", true, false)?,
            M::Divsd => self.sse_arith(i, "/", true, false)?,
            M::Addps => self.sse_arith(i, "+", false, true)?,
            M::Subps => self.sse_arith(i, "-", false, true)?,
            M::Mulps => self.sse_arith(i, "*", false, true)?,
            M::Divps => self.sse_arith(i, "/", false, true)?,
            M::Addpd => self.sse_arith(i, "+", true, true)?,
            M::Subpd => self.sse_arith(i, "-", true, true)?,
            M::Mulpd => self.sse_arith(i, "*", true, true)?,
            M::Divpd => self.sse_arith(i, "/", true, true)?,
            M::Sqrtss | M::Sqrtsd => {
                let dbl = m == M::Sqrtsd;
                let (d, s) = (Self::xreg(i, 0)?, self.xscalar(i, 1, dbl)?);
                let (l, f) = if dbl { ("g", "sqrt") } else { ("f", "sqrtf") };
                let (ty, t) = if dbl { ("double", "d") } else { ("float", "f") };
                self.emit(&format!(
                    "{{ {ty} s = {s}; {d}.{l}[0] = nanfix_{t}({f}(s), s, s); }}"
                ));
            }
            M::Minss | M::Maxss | M::Minsd | M::Maxsd => {
                let dbl = matches!(m, M::Minsd | M::Maxsd);
                let cmp = if matches!(m, M::Minss | M::Minsd) {
                    "<"
                } else {
                    ">"
                };
                let l = if dbl { "g" } else { "f" };
                let (d, s) = (Self::xreg(i, 0)?, self.xscalar(i, 1, dbl)?);
                // x86 returns the second operand when either is NaN or both are equal.
                self.emit(&format!(
                    "{{ {ty} s = {s}; {d}.{l}[0] = ({d}.{l}[0] {cmp} s) ? {d}.{l}[0] : s; }}",
                    ty = if dbl { "double" } else { "float" }
                ));
            }
            M::Cvtsi2ss | M::Cvtsi2sd => {
                let bits = if i.op1_kind() == OpKind::Register {
                    gpr(i.op1_register())?.1
                } else {
                    Self::mem_bits(i)
                };
                let v = self.rd(i, 1, bits)?;
                let d = Self::xreg(i, 0)?;
                let (l, t) = if m == M::Cvtsi2sd {
                    ("g", "double")
                } else {
                    ("f", "float")
                };
                self.emit(&format!("{d}.{l}[0] = ({t})(s64)sx({v}, {bits});"));
            }
            M::Cvttss2si | M::Cvttsd2si => {
                let dbl = m == M::Cvttsd2si;
                let bits = gpr(i.op0_register())?.1;
                let s = self.xscalar(i, 1, dbl)?;
                let w = self.wr(i, 0, &format!("(u64)cvtt{bits}((double){s})"))?;
                self.emit(&w);
            }
            M::Cvtss2sd => {
                let (d, s) = (Self::xreg(i, 0)?, self.xscalar(i, 1, false)?);
                self.emit(&format!("{d}.g[0] = (double){s};"));
            }
            M::Cvtsd2ss => {
                let (d, s) = (Self::xreg(i, 0)?, self.xscalar(i, 1, true)?);
                self.emit(&format!("{d}.f[0] = (float){s};"));
            }
            M::Ucomiss | M::Comiss | M::Ucomisd | M::Comisd => {
                let dbl = matches!(m, M::Ucomisd | M::Comisd);
                let (a, b) = (self.xscalar(i, 0, dbl)?, self.xscalar(i, 1, dbl)?);
                self.emit(&format!("fl_set_ucomi(&F, (double){a}, (double){b});"));
            }
            M::Movss | M::Movsd
                if i.op0_kind() == OpKind::Register && i.op0_register().is_xmm()
                    || i.op1_kind() == OpKind::Register && i.op1_register().is_xmm() =>
            {
                let dbl = m == M::Movsd;
                let (l, sz) = if dbl { ("g", 64) } else { ("f", 32) };
                if i.op0_kind() == OpKind::Memory {
                    let (a, s) = (self.addr(i)?, Self::xreg(i, 1)?);
                    self.emit(&format!(
                        "st{}({a}, {s}.{l}[0]);",
                        if dbl { "g" } else { "f" }
                    ));
                } else {
                    let d = Self::xreg(i, 0)?;
                    if i.op1_kind() == OpKind::Memory {
                        let a = self.addr(i)?;
                        self.emit(&format!(
                            "{d}.q[0] = 0; {d}.q[1] = 0; {d}.{l}[0] = ld{}({a});",
                            if dbl { "g" } else { "f" }
                        ));
                    } else {
                        let s = Self::xreg(i, 1)?;
                        self.emit(&format!("{d}.{l}[0] = {s}.{l}[0];"));
                    }
                }
                let _ = sz;
            }
            M::Movntps | M::Movntpd | M::Movntdq => {
                let (a, s) = (self.addr(i)?, Self::xreg(i, 1)?);
                self.emit(&format!("st128({a}, {s});"));
            }
            M::Movntdqa => {
                let (d, s) = (Self::xreg(i, 0)?, self.x128(i, 1)?);
                self.emit(&format!("{d} = {s};"));
            }
            M::Movnti => {
                let bits = Self::bits0(i)?;
                let v = self.rd(i, 1, bits)?;
                let w = self.wr(i, 0, &v)?;
                self.emit(&w);
            }
            M::Movaps | M::Movups | M::Movapd | M::Movupd | M::Movdqa | M::Movdqu => {
                if i.op0_kind() == OpKind::Memory {
                    let (a, s) = (self.addr(i)?, Self::xreg(i, 1)?);
                    self.emit(&format!("st128({a}, {s});"));
                } else {
                    let (d, s) = (Self::xreg(i, 0)?, self.x128(i, 1)?);
                    self.emit(&format!("{d} = {s};"));
                }
            }
            M::Movd | M::Movq if i.op0_register().is_xmm() || i.op1_register().is_xmm() => {
                let to_xmm = i.op0_kind() == OpKind::Register && i.op0_register().is_xmm();
                let bits = if m == M::Movd { 32 } else { 64 };
                if to_xmm {
                    let d = Self::xreg(i, 0)?;
                    let v = if i.op1_kind() == OpKind::Register && i.op1_register().is_xmm() {
                        format!("{}.q[0]", Self::xreg(i, 1)?)
                    } else {
                        self.rd(i, 1, bits)?
                    };
                    self.emit(&format!("{d}.q[0] = {v} & {}; {d}.q[1] = 0;", mask(bits)));
                } else {
                    let s = Self::xreg(i, 1)?;
                    let w = self.wr(i, 0, &format!("{s}.q[0] & {}", mask(bits)))?;
                    self.emit(&w);
                }
            }
            M::Pxor
            | M::Xorps
            | M::Xorpd
            | M::Pand
            | M::Andps
            | M::Andpd
            | M::Por
            | M::Orps
            | M::Orpd
            | M::Pandn
            | M::Andnps
            | M::Andnpd => {
                let d = Self::xreg(i, 0)?;
                let s = self.x128(i, 1)?;
                let e = |op: &str, inv: bool| {
                    format!(
                        "{{ X128 s = {s}; for (int k = 0; k < 2; k++) {d}.q[k] = {}{d}.q[k] {op} s.q[k]; }}",
                        if inv { "~" } else { "" }
                    )
                };
                let st = match m {
                    M::Pxor | M::Xorps | M::Xorpd => e("^", false),
                    M::Pand | M::Andps | M::Andpd => e("&", false),
                    M::Por | M::Orps | M::Orpd => e("|", false),
                    _ => e("&", true),
                };
                self.emit(&st);
            }
            _ => return Ok(false),
        }
        Ok(true)
    }

    /// Lifts one instruction. Memory accesses that are not provably thread-local (not based on rsp, not
    /// `gs:`) are emitted as `sld*`/`sst*`, which the C header turns into acquire/release accesses under
    /// `-DRT_TSO`; the rest stay plain `ld*`/`st*`.
    fn insn(&mut self, i: &Instruction, labels: &BTreeSet<u64>) -> R<()> {
        let start = self.o.len();
        self.insn_inner(i, labels)?;
        let local = (i.segment_prefix() == Register::GS || i.memory_base() == Register::RSP) && !self.spin.contains(&i.ip());
        if !local && (0..i.op_count()).any(|n| i.op_kind(n) == OpKind::Memory) {
            let tail = shared_accesses(&self.o[start..]);
            self.o.truncate(start);
            self.o.push_str(&tail);
        }
        Ok(())
    }

    fn insn_inner(&mut self, i: &Instruction, labels: &BTreeSet<u64>) -> R<()> {
        use Mnemonic as M;
        if i.has_lock_prefix()
            || i.has_rep_prefix()
            || i.has_repe_prefix()
            || i.has_repne_prefix()
            // A string instruction on its own (one element, direction flag clear).
            || matches!(
                i.mnemonic(),
                M::Scasb | M::Scasw | M::Scasd | M::Scasq | M::Stosb | M::Stosw | M::Stosq | M::Lodsb | M::Lodsw
                    | M::Lodsd | M::Lodsq | M::Movsb | M::Movsw | M::Movsq | M::Cmpsb | M::Cmpsw | M::Cmpsq
            )
        {
            return self.prefixed(i);
        }
        let m = i.mnemonic();
        if self.sse(i)? || self.vex(i)? || self.bits(i)? || self.legacy(i)? {
            return Ok(());
        }
        match m {
            M::Nop => {}
            // A spin-wait hint: ARM's own, and a compiler barrier, so a waiting loop reads memory again on every turn
            // (clang would otherwise keep a plain read of a loop that writes nothing in a register, and spin forever).
            M::Pause => self.emit("__asm__ volatile(\"yield\" ::: \"memory\");"),
            M::Mfence | M::Sfence | M::Lfence => self.emit("__atomic_thread_fence(__ATOMIC_SEQ_CST);"),
            M::Int3 | M::Ud2 => self.emit("rt_trap(\"int3/ud2\");"),
            M::Mov => {
                let bits = Self::bits0(i)?;
                let v = self.rd(i, 1, bits)?;
                let w = self.wr(i, 0, &v)?;
                self.emit(&w);
            }
            M::Movzx | M::Movsx | M::Movsxd => {
                let sb = if i.op1_kind() == OpKind::Register {
                    gpr(i.op1_register())?.1
                } else {
                    Self::mem_bits(i)
                };
                let v = self.rd(i, 1, sb)?;
                let v = if m == M::Movzx {
                    v
                } else {
                    format!("sx({v}, {sb})")
                };
                let dbits = Self::bits0(i)?;
                let w = self.wr(i, 0, &format!("({v}) & {}", mask(dbits)))?;
                self.emit(&w);
            }
            M::Lea => {
                let a = self.addr(i)?;
                let w = self.wr(i, 0, &format!("(u64)({a})"))?;
                self.emit(&w);
            }
            M::Xchg if i.op0_kind() == OpKind::Register && i.op1_kind() == OpKind::Register => {
                let (a, b) = (self.rd(i, 0, 64)?, self.rd(i, 1, 64)?);
                let (w0, w1) = (self.wr(i, 0, "tb")?, self.wr(i, 1, "ta")?);
                self.emit(&format!("{{ u64 ta = {a}, tb = {b}; {w0} {w1} }}"));
            }
            M::Add => self.alu(i, "FK_ADD", "a + b", true, false)?,
            M::Adc => self.alu(i, "FK_ADC", "a + b + cin", true, true)?,
            M::Sub => self.alu(i, "FK_SUB", "a - b", true, false)?,
            M::Sbb => self.alu(i, "FK_SBB", "a - b - cin", true, true)?,
            M::Cmp => self.alu(i, "FK_SUB", "a - b", false, false)?,
            M::And => self.alu(i, "FK_LOGIC", "a & b", true, false)?,
            M::Or => self.alu(i, "FK_LOGIC", "a | b", true, false)?,
            M::Xor => self.alu(i, "FK_LOGIC", "a ^ b", true, false)?,
            M::Test => self.alu(i, "FK_LOGIC", "a & b", false, false)?,
            M::Inc | M::Dec | M::Neg | M::Not => {
                let bits = Self::bits0(i)?;
                let a = self.rd(i, 0, bits)?;
                let w = self.wr(i, 0, "r")?;
                let m_ = mask(bits);
                let st = match m {
                    M::Inc => format!(
                        "u64 r = (a + 1) & {m_}; F.c = (u64)fl_cf(&F); F.k = FK_INC; F.sz = {bits}; F.r = r;"
                    ),
                    M::Dec => format!(
                        "u64 r = (a - 1) & {m_}; F.c = (u64)fl_cf(&F); F.k = FK_DEC; F.sz = {bits}; F.r = r;"
                    ),
                    M::Neg => format!(
                        "u64 r = (0 - a) & {m_}; F.k = FK_SUB; F.sz = {bits}; F.a = 0; F.b = a; F.r = r; F.c = 0;"
                    ),
                    _ => format!("u64 r = ~a & {m_};"),
                };
                // The write must come after reading flags but flag fields are independent of it.
                self.emit(&format!("{{ u64 a = {a}; {st} {w} }}"));
            }
            M::Shl | M::Sal => self.shift(i, "shl")?,
            M::Shr => self.shift(i, "shr")?,
            M::Sar => self.shift(i, "sar")?,
            M::Rol => self.rotate(i, true)?,
            M::Ror => self.rotate(i, false)?,
            M::Imul if i.op_count() >= 2 => {
                let bits = Self::bits0(i)?;
                let (a, b) = if i.op_count() == 3 {
                    (self.rd(i, 1, bits)?, self.rd(i, 2, bits)?)
                } else {
                    (self.rd(i, 0, bits)?, self.rd(i, 1, bits)?)
                };
                let w = self.wr(i, 0, "r")?;
                self.emit(&format!(
                    "{{ s64 sa = (s64)sx({a}, {bits}), sb = (s64)sx({b}, {bits}); s64 p; int ov = __builtin_mul_overflow(sa, sb, &p); u64 r = (u64)p & {m}; if ({bits} != 64) ov = (s64)sx(r, {bits}) != p; {w} fl_set_raw(&F, {bits}, r, ov, ov); }}",
                    m = mask(bits)
                ));
            }
            M::Imul | M::Mul => {
                let bits = Self::bits0(i)?;
                let s = self.rd(i, 0, bits)?;
                let sgn = m == M::Imul;
                let (ty, wide) = if sgn {
                    ("__int128", "(__int128)(s64)")
                } else {
                    ("unsigned __int128", "(unsigned __int128)")
                };
                let (lo, hi) = match bits {
                    8 => ("r0 = (r0 & ~0xffffull) | (u16)p".to_string(), String::new()),
                    16 => (
                        "r0 = (r0 & ~0xffffull) | (u16)p".into(),
                        "r2 = (r2 & ~0xffffull) | (u16)(p >> 16)".into(),
                    ),
                    32 => ("r0 = (u32)p".into(), "r2 = (u32)(p >> 32)".into()),
                    _ => ("r0 = (u64)p".into(), "r2 = (u64)(p >> 64)".into()),
                };
                let (av, sv) = if sgn {
                    (
                        format!("sx(r0 & {m}, {bits})", m = mask(bits)),
                        format!("sx({s}, {bits})"),
                    )
                } else {
                    (format!("(r0 & {m})", m = mask(bits)), s.clone())
                };
                let ovf = if sgn {
                    format!("p != ({ty})(s64)sx((u64)p & {m}, {bits})", m = mask(bits))
                } else {
                    format!("(p >> {bits}) != 0")
                };
                self.emit(&format!("{{ {ty} p = {wide}({av}) * {wide}({sv}); {lo}; {hi}; int ov = {ovf}; fl_set_raw(&F, {bits}, (u64)p, ov, ov); }}"));
            }
            M::Div | M::Idiv => {
                let bits = Self::bits0(i)?;
                let ip = i.ip();
                let s = self.rd(i, 0, bits)?;
                let sgn = m == M::Idiv;
                let body = match (bits, sgn) {
                    (8, _) | (16, _) => return Err("8/16-bit div".into()),
                    (32, false) => format!(
                        "u64 n = (r2 << 32) | (u32)r0; u64 d = {s}; if (!d || n / d > 0xffffffffull) RT_DE({ip:#x}ull, d != 0); r0 = (u32)(n / d); r2 = (u32)(n % d);"
                    ),
                    (32, true) => format!(
                        "s64 n = (s64)(((u64)(u32)r2 << 32) | (u32)r0); s64 d = (s32){s}; if (!d || n / d > INT32_MAX || n / d < INT32_MIN) RT_DE({ip:#x}ull, d != 0); r0 = (u32)(s32)(n / d); r2 = (u32)(s32)(n % d);"
                    ),
                    (_, false) => format!(
                        "unsigned __int128 n = ((unsigned __int128)r2 << 64) | r0; u64 d = {s}; if (!d || n / d > ~0ull) RT_DE({ip:#x}ull, d != 0); r0 = (u64)(n / d); r2 = (u64)(n % d);"
                    ),
                    (_, true) => format!(
                        "__int128 n = (__int128)(((unsigned __int128)r2 << 64) | r0); s64 d = (s64){s}; if (!d) RT_DE({ip:#x}ull, d != 0); __int128 q = n / d; if (q > INT64_MAX || q < INT64_MIN) RT_DE({ip:#x}ull, d != 0); r0 = (u64)(s64)q; r2 = (u64)(s64)(n % d);"
                    ),
                };
                self.emit(&format!("{{ {body} }}"));
            }
            M::Cdq => self.emit("r2 = (u32)((s32)r0 >> 31);"),
            M::Cqo => self.emit("r2 = (u64)((s64)r0 >> 63);"),
            M::Cdqe => self.emit("r0 = (u64)(s64)(s32)r0;"),
            M::Cwde => self.emit("r0 = (u32)(s32)(s16)r0;"),
            M::Bsf | M::Bsr | M::Popcnt => {
                let bits = Self::bits0(i)?;
                let s = self.rd(i, 1, bits)?;
                let cur = self.rd(i, 0, bits)?;
                let (expr, undef) = match m {
                    M::Bsf => ("(u64)__builtin_ctzll(s)", true),
                    M::Bsr => ("(u64)(63 - __builtin_clzll(s))", true),
                    _ => ("(u64)__builtin_popcountll(s)", false),
                };
                let w = self.wr(i, 0, "res")?;
                // bsf/bsr leave the whole destination register untouched on a zero source (even for
                // 32-bit forms); popcnt always writes.
                let guard = if undef { "if (s) " } else { "" };
                let _ = cur;
                self.emit(&format!("{{ u64 s = {s}; u64 res = s ? {expr} : 0; {guard}{{ {w} }} F.k = FK_RAW; F.c = (s == 0) << 1; }}"));
            }
            M::Push => {
                let v = self.rd(i, 0, 64)?;
                self.emit(&format!("{{ u64 v = {v}; r4 -= 8; st64((u8*)r4, v); }}"));
            }
            M::Pop => {
                let w = self.wr(i, 0, "v")?;
                self.emit(&format!("{{ u64 v = ld64((u8*)r4); r4 += 8; {w} }}"));
            }
            M::Setb
            | M::Setae
            | M::Sete
            | M::Setne
            | M::Setbe
            | M::Seta
            | M::Sets
            | M::Setns
            | M::Setp
            | M::Setnp
            | M::Setl
            | M::Setge
            | M::Setle
            | M::Setg
            | M::Seto
            | M::Setno => {
                let c = Self::cond(i)?;
                let w = self.wr(i, 0, &format!("(u64)({c})"))?;
                self.emit(&w);
            }
            M::Cmovb
            | M::Cmovae
            | M::Cmove
            | M::Cmovne
            | M::Cmovbe
            | M::Cmova
            | M::Cmovs
            | M::Cmovns
            | M::Cmovp
            | M::Cmovnp
            | M::Cmovl
            | M::Cmovge
            | M::Cmovle
            | M::Cmovg
            | M::Cmovo
            | M::Cmovno => {
                let bits = Self::bits0(i)?;
                let (c, s, cur) = (Self::cond(i)?, self.rd(i, 1, bits)?, self.rd(i, 0, bits)?);
                // A 32-bit cmov zero-extends even when the condition is false.
                let w = self.wr(i, 0, &format!("({c}) ? {s} : {cur}"))?;
                self.emit(&w);
            }
            M::Call => self.call(i)?,
            M::Ret => {
                let extra = if i.op_count() == 1 { i.immediate(0) } else { 0 };
                self.emit(&format!("r4 += {}; {}; return;", 8 + extra, self.sp()));
            }
            M::Jmp => match i.op0_kind() {
                OpKind::NearBranch64 => {
                    let s = self.jump_to(i.near_branch_target(), labels);
                    self.emit(&s);
                }
                OpKind::Register => {
                    let r = self.rd(i, 0, 64)?;
                    // No recovered table: a tail call through a register (or a switch we could not
                    // decode); hand it to the dispatcher, which logs it if it is not a function start.
                    // Cases that fall inside the function but on no instruction (the table ran on into other data) are left
                    // out; one that is hit anyway traps.
                    let usable: Option<Vec<u64>> = self.f.tables.get(&i.ip()).map(|ts| {
                        ts.iter().copied().filter(|t| self.ips.contains(t) || !self.inside(*t)).collect()
                    });
                    let Some(ts) = usable.filter(|ts| !ts.is_empty()) else {
                        self.emit(&format!("{{ u64 t = {r}; {}; rt_dispatch(c, t); return; }}", self.sp()));
                        return Ok(());
                    };
                    let mut sw = format!("switch ({r} - (u64)IMG + IMAGE_BASE) {{ ");
                    for t in ts.iter().collect::<BTreeSet<_>>() {
                        if self.ips.contains(t) {
                            let _ = write!(sw, "case {t:#x}ull: goto L_{t:x}; ");
                        } else {
                            // A case that is another function: a tail call.
                            let _ = write!(sw, "case {t:#x}ull: {{ {}; rt_dispatch(c, {r}); return; }} ", self.sp());
                        }
                    }
                    sw.push_str("default: rt_trap(\"jump table index out of range\"); }");
                    self.emit(&sw);
                }
                _ => {
                    let t = self.rd(i, 0, 64)?;
                    let cases: Vec<u64> = self
                        .f
                        .tables
                        .get(&i.ip())
                        .map(|ts| ts.iter().copied().filter(|t| self.ips.contains(t)).collect())
                        .unwrap_or_default();
                    if cases.is_empty() {
                        self.emit(&format!(
                            "{{ u64 t = {t}; {}; rt_dispatch(c, t); return; }}",
                            self.sp()
                        ));
                    } else {
                        // `jmp [base + 8*idx]` through a table of addresses: an interpreter's dispatch, at the end of
                        // each of its handlers. One switch for the function, after its body (`JT`, [`lift_core`]).
                        self.jt.extend(cases);
                        self.emit(&format!("{{ jt = {t}; goto JT; }}"));
                    }
                }
            },
            _ if i.flow_control() == iced_x86::FlowControl::ConditionalBranch => {
                let c = Self::cond(i)?;
                let j = self.jump_to(i.near_branch_target(), labels);
                self.emit(&format!("if ({c}) {j}"));
            }
            _ => return Err(format!("{m:?}")),
        }
        Ok(())
    }
}

/// Accessors whose shared (possibly ordered) form is the same name with an `s` in front.
const ACCESSORS: [&str; 18] = [
    "ld8", "st8", "ld16", "ld32", "ld64", "st16", "st32", "st64", "ld128", "st128", "ldf", "ldg", "stf", "stg",
    "ldx8", "ldx16", "ldx32", "ldx64",
];

/// Renames the accessor calls in `code` to their shared forms, except the implicit stack accesses
/// (`ld64((u8*)r4)` and friends: push, pop, call, ret).
fn shared_accesses(code: &str) -> String {
    let b = code.as_bytes();
    let mut out = String::with_capacity(code.len() + 16);
    let mut k = 0;
    while k < b.len() {
        let word_start = b[k].is_ascii_alphabetic() && (k == 0 || !(b[k - 1].is_ascii_alphanumeric() || b[k - 1] == b'_'));
        if word_start {
            let mut e = k;
            while e < b.len() && (b[e].is_ascii_alphanumeric() || b[e] == b'_') {
                e += 1;
            }
            let w = &code[k..e];
            let stack = code[e..].starts_with("((u8*)r4");
            if ACCESSORS.contains(&w) && code[e..].starts_with('(') && !stack {
                out.push('s');
            }
            out.push_str(w);
            k = e;
        } else {
            out.push(b[k] as char);
            k += 1;
        }
    }
    out
}

/// VEX instruction that names an xmm/ymm register, or vzeroupper/vzeroall.
fn uses_vector(i: &Instruction) -> bool {
    use iced_x86::EncodingKind;
    i.code().encoding() == EncodingKind::VEX
        && (matches!(i.mnemonic(), Mnemonic::Vzeroupper | Mnemonic::Vzeroall)
            || (0..i.op_count()).any(|n| {
                i.op_kind(n) == OpKind::Register
                    && (i.op_register(n).is_xmm() || i.op_register(n).is_ymm())
            }))
}

pub fn lift_function(ctx: &Ctx, f: &FuncIn) -> Result<String, String> {
    let (o, errs) = lift_core(ctx, f, false);
    match errs.into_iter().next() {
        Some(e) => {
            // Data inside the function's range (a jump table or constants after the `ret`) decodes as junk the lifter
            // cannot take; if it is not reachable from the entry, leave it out and lift the code.
            let live = reachable(f);
            if live.len() == f.ins.len() {
                return Err(e);
            }
            let ins: Vec<Instruction> = f.ins.iter().filter(|i| live.contains(&i.ip())).copied().collect();
            let pruned = FuncIn {
                begin_va: f.begin_va,
                end_va: f.end_va,
                ins: &ins,
                tables: f.tables.clone(),
                extra: f.extra.clone(),
                landing_pads: f.landing_pads.clone(),
                entries: f.entries.clone(),
            };
            let (o, errs) = lift_core(ctx, &pruned, false);
            match errs.into_iter().next() {
                Some(_) => Err(e),
                None => Ok(o),
            }
        }
        None => Ok(o),
    }
}

/// The instructions reachable from the entry, the landing pads and the chained parts by falling through, branching and
/// the recovered jump tables.
fn reachable(f: &FuncIn) -> std::collections::HashSet<u64> {
    use iced_x86::FlowControl as FC;
    let at: std::collections::HashMap<u64, usize> = f.ins.iter().enumerate().map(|(k, i)| (i.ip(), k)).collect();
    let mut seen = std::collections::HashSet::new();
    let mut work: Vec<u64> = vec![f.begin_va];
    work.extend(f.landing_pads.iter().copied());
    work.extend(f.entries.iter().copied());
    work.extend(f.extra.iter().map(|&(b, _)| b));
    while let Some(ip) = work.pop() {
        let Some(&k) = at.get(&ip) else { continue };
        for i in &f.ins[k..] {
            if !seen.insert(i.ip()) {
                break;
            }
            match i.flow_control() {
                FC::ConditionalBranch => {
                    work.push(i.near_branch_target());
                }
                FC::UnconditionalBranch => {
                    if i.op0_kind() == OpKind::NearBranch64 {
                        work.push(i.near_branch_target());
                    } else if let Some(ts) = f.tables.get(&i.ip()) {
                        work.extend(ts.iter().copied());
                    }
                    break;
                }
                FC::IndirectBranch => {
                    if let Some(ts) = f.tables.get(&i.ip()) {
                        work.extend(ts.iter().copied());
                    }
                    break;
                }
                FC::Return => break,
                FC::Exception if matches!(i.mnemonic(), Mnemonic::Ud2) => break,
                _ => {}
            }
        }
    }
    seen
}

/// Every unsupported-instruction error in `f` (not just the first), for ranking what to lift next.
pub fn unsupported(ctx: &Ctx, f: &FuncIn) -> Vec<String> {
    lift_core(ctx, f, true).1
}

fn lift_core(ctx: &Ctx, f: &FuncIn, all: bool) -> (String, Vec<String>) {
    // Leaders: branch targets inside the function and jump-table targets.
    let mut labels = BTreeSet::new();
    for i in f.ins {
        use iced_x86::FlowControl as FC;
        if matches!(
            i.flow_control(),
            FC::ConditionalBranch | FC::UnconditionalBranch
        ) && i.op0_kind() == OpKind::NearBranch64
        {
            labels.insert(i.near_branch_target());
        }
    }
    let ipset: std::collections::HashSet<u64> = f.ins.iter().map(|i| i.ip()).collect();
    let inside = |t: &u64| (f.begin_va..f.end_va).contains(t) || f.extra.iter().any(|&(b, e)| (b..e).contains(t));
    let _ = inside;
    for ts in f.tables.values() {
        labels.extend(ts.iter().copied().filter(|t| ipset.contains(t)));
    }
    let pads: Vec<u64> = f.landing_pads.iter().copied().filter(|p| ipset.contains(p)).collect();
    labels.extend(pads.iter().copied());
    // Every entry must get its function (the dispatch table names it): one that is no instruction here traps.
    let entries: Vec<u64> = if pads.is_empty() { f.entries.clone() } else { vec![] };
    labels.extend(entries.iter().copied().filter(|e| ipset.contains(e)));
    // In a function with landing pads, a fault in the guarded code is recorded with every register (`c->rip`), and the
    // instruction is a resume point too: an exception handler that continues execution re-runs it with the registers it
    // chose. (EFLAGS are not carried: the lifted flags start clean, which an instruction reading them would notice.)
    let marked = |i: &Instruction| {
        !pads.is_empty()
            && (matches!(i.mnemonic(), Mnemonic::Div | Mnemonic::Idiv)
                || (!matches!(i.mnemonic(), Mnemonic::Lea | Mnemonic::Nop)
                    && (0..i.op_count()).any(|k| i.op_kind(k) == OpKind::Memory)))
    };
    let resume_points: BTreeSet<u64> = pads.iter().copied().chain(f.ins.iter().filter(|i| marked(i)).map(|i| i.ip())).collect();
    labels.extend(resume_points.iter().copied());
    let y = f.ins.iter().any(uses_vector);
    let mut l = L {
        ctx,
        f,
        o: String::new(),
        y,
        ips: f.ins.iter().map(|i| i.ip()).collect(),
        spin: spin_reads(f.ins),
        jt: BTreeSet::new(),
    };
    let mut errs = vec![];
    let mut prologue = String::from(if y { "FUNC_PROLOGUE_Y()" } else { "FUNC_PROLOGUE()" });
    // A dispatch through a table of addresses (`jmp [base + 8*idx]`) passes the address to the `JT` switch in `jt`.
    if f.ins.iter().any(|i| {
        i.code() == iced_x86::Code::Jmp_rm64
            && i.op0_kind() == OpKind::Memory
            && f.tables.get(&i.ip()).is_some_and(|ts| ts.iter().any(|t| ipset.contains(t)))
    }) {
        prologue.push_str("; u64 jt = 0; (void)jt");
    }
    let mut body_at = 0;
    if pads.is_empty() {
        let _ = writeln!(l.o, "void {}(Cpu *c) {{ {prologue};", name(f.begin_va, ctx.image_base));
        body_at = l.o.len();
    } else {
        // The body takes `resume`: set when the host unwound to this frame, with the landing pad in
        // `rt_resume_va` and the registers in `c`, which the prologue loads as always.
        let _ = writeln!(
            l.o,
            "static void fb_{:x}(Cpu *c, int resume) {{ {prologue};\nif (resume) switch (rt_resume_va - (u64)IMG + IMAGE_BASE) {{",
            f.begin_va - ctx.image_base
        );
        for p in &resume_points {
            let _ = writeln!(l.o, "case {p:#x}ull: goto L_{p:x};");
        }
        l.o.push_str("default: rt_trap(\"resume at an address this function has no entry for (continuing after a fault needs landing pads in the function)\"); }\n");
    }
    let mut expect = f.begin_va;
    for i in f.ins {
        if i.ip() != expect {
            // Start of a merged part: nothing falls into it.
            l.o.push_str("rt_trap(\"fell into a merged part\");\n");
        }
        expect = i.next_ip();
        if labels.contains(&i.ip()) {
            let _ = writeln!(l.o, "L_{:x}:;", i.ip());
        }
        // In a function with landing pads a fault may be in the guarded code: record where, with every register
        // (an access violation has only the last spill otherwise).
        if marked(i) {
            // The barrier keeps the compiler from moving the stores after the access that may fault.
            let _ = writeln!(l.o, "{}; c->rip = (u64)H({:#x}ull); __asm__ volatile(\"\" ::: \"memory\");", l.sp(), i.ip());
        }
        if let Err(e) = l.insn(i, &labels) {
            errs.push(format!("{:#x}: {e}", i.ip() - ctx.image_base));
            if !all {
                break;
            }
        }
    }
    // Falling off the end of a function is only legal after a non-returning call.
    if l.jt.is_empty() {
        l.o.push_str("rt_trap(\"fell off end of function\"); }\n");
    } else {
        // The function's dispatch through its tables of addresses (`jt`, the address read); one not among the
        // cases (no instruction here) goes to the dispatcher, as an untabled jump would.
        let mut sw = String::from("rt_trap(\"fell off end of function\");\nJT:; switch (jt - (u64)IMG + IMAGE_BASE) { ");
        for t in &l.jt {
            let _ = write!(sw, "case {t:#x}ull: goto L_{t:x}; ");
        }
        let _ = writeln!(sw, "default: {{ {}; rt_dispatch(c, jt); return; }} }}\n}}", l.sp());
        l.o.push_str(&sw);
    }
    if !entries.is_empty() {
        // The same body for each inner entry, entered at its label (the prologue loads every register as at the
        // start; nothing of the body before the label has run, as for the guest).
        let body = l.o[body_at..].to_string();
        for &e in &entries {
            if ipset.contains(&e) {
                let _ = write!(l.o, "void {}(Cpu *c) {{ {prologue};\ngoto L_{e:x};\n{body}", name(e, ctx.image_base));
            } else {
                let _ = writeln!(
                    l.o,
                    "void {}(Cpu *c) {{ (void)c; rt_trap(\"entry {:#x} is inside an instruction of {:#x}\"); }}",
                    name(e, ctx.image_base),
                    e - ctx.image_base,
                    f.begin_va - ctx.image_base
                );
            }
        }
    }
    if !pads.is_empty() {
        for &e in &f.entries {
            let _ = writeln!(
                l.o,
                "void {}(Cpu *c) {{ (void)c; rt_trap(\"entry {:#x} into {:#x}, a function with landing pads\"); }}",
                name(e, ctx.image_base),
                e - ctx.image_base,
                f.begin_va - ctx.image_base
            );
        }
        let (n, rva) = (name(f.begin_va, ctx.image_base), f.begin_va - ctx.image_base);
        let _ = writeln!(
            l.o,
            "void {n}(Cpu *c) {{ RtFrame fr; rt_frame_enter(&fr, c); if (_setjmp(fr.jb)) {{ if (rt_resume_call) rt_resume_enter(c); fb_{rva:x}(c, 1); }} else fb_{rva:x}(c, 0); rt_frame_leave(&fr); }}"
        );
    }
    l.o.push_str(&pad_defs(ctx, f));
    (l.o, errs)
}

#[cfg(test)]
mod tests {
    use super::*;
    use iced_x86::code_asm::*;
    use iced_x86::{BlockEncoderOptions, Decoder, DecoderOptions};

    #[test]
    fn an_inner_entry_is_the_body_entered_at_its_label() {
        const BASE: u64 = 0x1_4000_0000;
        let at = BASE + 0x1000;
        let mut a = CodeAssembler::new(64).unwrap();
        let mut tail = a.create_label();
        a.mov(eax, 1).unwrap();
        a.set_label(&mut tail).unwrap();
        a.add(eax, 2).unwrap();
        a.ret().unwrap();
        let r = a.assemble_options(at, BlockEncoderOptions::RETURN_NEW_INSTRUCTION_OFFSETS).unwrap();
        let entry = r.label_ip(&tail).unwrap();
        let bytes = r.inner.code_buffer;
        let ins: Vec<Instruction> = Decoder::with_ip(64, &bytes, at, DecoderOptions::NONE).iter().collect();
        let is_func = move |va: u64| va == at || va == entry;
        let ctx = Ctx { image_base: BASE, is_func: &is_func };
        let f = FuncIn {
            begin_va: at,
            end_va: at + bytes.len() as u64,
            ins: &ins,
            tables: BTreeMap::new(),
            extra: vec![],
            landing_pads: vec![],
            entries: vec![entry],
        };
        let c = lift_function(&ctx, &f).unwrap();
        assert!(c.contains(&format!("void f_{:x}(Cpu *c) {{", at - BASE)), "{c}");
        assert!(c.contains(&format!("void f_{:x}(Cpu *c) {{", entry - BASE)), "{c}");
        assert!(c.contains(&format!("goto L_{entry:x};")), "{c}");
        // The label in both bodies.
        assert_eq!(c.matches(&format!("L_{entry:x}:;")).count(), 2, "{c}");
        // An entry that is no instruction start still gets its function, one that traps.
        let f = FuncIn { entries: vec![entry + 1], ..f };
        let c = lift_function(&ctx, &f).unwrap();
        assert!(c.contains(&format!("void f_{:x}(Cpu *c) {{ (void)c; rt_trap(", entry + 1 - BASE)), "{c}");
    }

    #[test]
    fn a_dispatch_through_a_table_of_addresses_goes_to_the_function_switch() {
        const BASE: u64 = 0x1_4000_0000;
        let at = BASE + 0x1000;
        let mut a = CodeAssembler::new(64).unwrap();
        let (mut dispatch, mut h0, mut h1) = (a.create_label(), a.create_label(), a.create_label());
        a.set_label(&mut dispatch).unwrap();
        a.movzx(ebx, byte_ptr(rsi)).unwrap();
        a.jmp(qword_ptr(r12 + rbx * 8)).unwrap();
        a.set_label(&mut h0).unwrap();
        a.jmp(dispatch).unwrap();
        a.set_label(&mut h1).unwrap();
        a.ret().unwrap();
        let r = a.assemble_options(at, BlockEncoderOptions::RETURN_NEW_INSTRUCTION_OFFSETS).unwrap();
        let (h0, h1) = (r.label_ip(&h0).unwrap(), r.label_ip(&h1).unwrap());
        let bytes = r.inner.code_buffer;
        let ins: Vec<Instruction> = Decoder::with_ip(64, &bytes, at, DecoderOptions::NONE).iter().collect();
        let jmp = ins[1].ip();
        let is_func = move |va: u64| va == at;
        let ctx = Ctx { image_base: BASE, is_func: &is_func };
        let f = FuncIn {
            begin_va: at,
            end_va: at + bytes.len() as u64,
            ins: &ins,
            tables: BTreeMap::from([(jmp, vec![h0, h1, BASE + 0x9000])]),
            extra: vec![],
            landing_pads: vec![],
            entries: vec![],
        };
        let c = lift_function(&ctx, &f).unwrap();
        assert!(c.contains("u64 jt = 0;"), "{c}");
        assert!(c.contains("goto JT;"), "{c}");
        assert!(c.contains(&format!("JT:; switch (jt - (u64)IMG + IMAGE_BASE) {{ case {h0:#x}ull: goto L_{h0:x}; case {h1:#x}ull: goto L_{h1:x}; default:")), "{c}");
    }

    fn lifted(build: impl FnOnce(&mut CodeAssembler) -> Result<(), IcedError>) -> String {
        const BASE: u64 = 0x1_4000_0000;
        let at = BASE + 0x1000;
        let mut a = CodeAssembler::new(64).unwrap();
        build(&mut a).unwrap();
        let bytes = a.assemble(at).unwrap();
        let ins: Vec<Instruction> = Decoder::with_ip(64, &bytes, at, DecoderOptions::NONE).iter().collect();
        let is_func = move |va: u64| va == at;
        let ctx = Ctx { image_base: BASE, is_func: &is_func };
        let f = FuncIn {
            begin_va: at,
            end_va: at + bytes.len() as u64,
            ins: &ins,
            tables: BTreeMap::new(),
            extra: vec![],
            landing_pads: vec![],
            entries: vec![],
        };
        lift_function(&ctx, &f).unwrap()
    }

    #[test]
    fn a_stack_slot_a_loop_spins_on_is_read_as_shared() {
        // rva 0x9dd0da: the workers' counter, handed out by address, on the waiter's stack.
        let c = lifted(|a| {
            let mut spin = a.create_label();
            a.mov(dword_ptr(rsp + 0x48), 0)?;
            a.set_label(&mut spin)?;
            a.mov(eax, dword_ptr(rsp + 0x48))?;
            a.cmp(eax, 8)?;
            a.jne(spin)?;
            a.mov(ecx, dword_ptr(rsp + 0x50))?;
            a.ret()
        });
        assert!(c.contains("r0 = (u32)((u64)sld32((u8*)(r4 + (u64)(s64)72ll)));"), "{c}");
        // Outside the loop, and in one that writes the slot or calls, the stack stays plain.
        assert!(c.contains("ld32((u8*)(r4 + (u64)(s64)80ll))") && !c.contains("sld32((u8*)(r4 + (u64)(s64)80ll))"), "{c}");
        let c = lifted(|a| {
            let mut again = a.create_label();
            a.set_label(&mut again)?;
            a.mov(eax, dword_ptr(rsp + 0x48))?;
            a.inc(dword_ptr(rsp + 0x48))?;
            a.cmp(eax, 8)?;
            a.jne(again)?;
            a.ret()
        });
        assert!(!c.contains("sld32"), "{c}");
        let c = lifted(|a| {
            let (mut again, mut callee) = (a.create_label(), a.create_label());
            a.set_label(&mut again)?;
            a.call(callee)?;
            a.mov(eax, dword_ptr(rsp + 0x48))?;
            a.cmp(eax, 8)?;
            a.jne(again)?;
            a.set_label(&mut callee)?;
            a.ret()
        });
        assert!(!c.contains("sld32"), "{c}");
        // A search: the loop counts (its own state changes), its reads are its business.
        let c = lifted(|a| {
            let mut again = a.create_label();
            a.set_label(&mut again)?;
            a.mov(rdx, qword_ptr(rsp + 0x28))?;
            a.cmp(dword_ptr(rdx + rcx * 4), eax)?;
            a.inc(rcx)?;
            a.cmp(rcx, qword_ptr(rsp + 0x40))?;
            a.jb(again)?;
            a.ret()
        });
        assert!(!c.contains("sld64((u8*)(r4"), "{c}");
        // `pause` is a compiler barrier.
        let c = lifted(|a| {
            a.pause()?;
            a.ret()
        });
        assert!(c.contains("__asm__ volatile(\"yield\" ::: \"memory\");"), "{c}");
    }
}
