//! Static x86-64 -> ARM64 recompiler foundation. See `docs/recompiler.md`.

pub mod analyze;
pub mod discover;
pub mod image;
pub mod lift;

/// Files a host needs next to a lifted unit: the runtime header and prototype host.
pub const RT_HOST_C: &str = include_str!("rt_host.c");
/// For an image built as a dylib of its own (several images in one process): its `IMG` and unit record.
pub const RT_UNIT_C: &str = include_str!("rt_unit.c");
pub const HARNESS_H: &str = include_str!("harness.h");

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};

use anyhow::Result;
use image::{Func, Image};

type Tables = BTreeMap<u64, Vec<u64>>;

fn tables_for(img: &Image, f: &Func, ins: &[iced_x86::Instruction]) -> Tables {
    let base = img.info.image_base;
    analyze::find_jump_tables(img, f, ins)
        .iter()
        .map(|t| {
            (
                base + t.jmp_rva as u64,
                t.targets.iter().map(|&r| base + r as u64).collect(),
            )
        })
        .collect()
}

/// Entry-point functions with their chained `.pdata` parts attached (cold blocks and funclets
/// continue a parent through the unwind info's chain pointer). A part whose parent is unknown is dropped.
pub fn group_chained(img: &Image, all: Vec<Func>) -> Vec<(Func, Vec<Func>)> {
    let mut parts: std::collections::HashMap<u32, Vec<Func>> = Default::default();
    let by_begin: std::collections::HashMap<u32, Func> = all.iter().map(|f| (f.begin, *f)).collect();
    for f in all.iter().filter(|f| f.chained) {
        let mut root = *f;
        for _ in 0..8 {
            match img.chain_parent(&root).and_then(|p| by_begin.get(&p)) {
                Some(p) => root = *p,
                None => break,
            }
            if !root.chained {
                break;
            }
        }
        if !root.chained {
            parts.entry(root.begin).or_default().push(*f);
        }
    }
    all.into_iter()
        .filter(|f| !f.chained)
        .map(|f| {
            let mut p = parts.remove(&f.begin).unwrap_or_default();
            p.sort_by_key(|x| x.begin);
            (f, p)
        })
        .collect()
}

/// Decodes a function and its merged parts into one instruction list (parent first).
fn decode_group(img: &Image, f: &Func, parts: &[Func]) -> (Vec<iced_x86::Instruction>, bool) {
    let (mut ins, mut bad) = analyze::decode_function(img, f);
    for p in parts {
        let (i2, b2) = analyze::decode_function(img, p);
        ins.extend(i2);
        bad |= b2;
    }
    (ins, bad)
}

fn extra_ranges(img: &Image, parts: &[Func]) -> Vec<(u64, u64)> {
    let base = img.info.image_base;
    parts.iter().map(|p| (base + p.begin as u64, base + p.end as u64)).collect()
}

fn landing_pads(img: &Image, f: &Func) -> Vec<u64> {
    img.landing_pads(f).iter().map(|&r| img.info.image_base + r as u64).collect()
}

/// The direct call, jump and branch targets (and executable `lea [rip + x]` addresses) in each function that land
/// inside another one, not at its start: hand-written CRT routines share their tails, and a jump into the middle of
/// one went to the dispatcher, which has no function there. Keyed by the begin of the function they are in (VAs),
/// for those of its instructions; not for functions with landing pads.
pub fn inner_entries(img: &Image, groups: &[(Func, Vec<Func>)]) -> HashMap<u32, Vec<u64>> {
    let base = img.info.image_base;
    // Every extent (a function and its chained parts) with the group it belongs to, by begin.
    let mut extents: Vec<(u32, u32, usize)> = groups
        .iter()
        .enumerate()
        .flat_map(|(g, (f, parts))| std::iter::once((f.begin, f.end, g)).chain(parts.iter().map(move |p| (p.begin, p.end, g))))
        .collect();
    extents.sort_unstable();
    let group_of = |t: u32| {
        let k = extents.partition_point(|e| e.0 <= t);
        (k > 0 && t < extents[k - 1].1).then(|| extents[k - 1].2)
    };
    let starts: HashSet<u32> = groups.iter().map(|(f, _)| f.begin).collect();
    let mut wanted: BTreeMap<usize, BTreeSet<u32>> = BTreeMap::new();
    for (g, (f, parts)) in groups.iter().enumerate() {
        for e in std::iter::once(f).chain(parts.iter()) {
            for t in discover::sweep(img, e.begin, e.end) {
                if starts.contains(&t) {
                    continue;
                }
                match group_of(t) {
                    Some(h) if h != g => {
                        wanted.entry(h).or_default().insert(t);
                    }
                    _ => {}
                }
            }
        }
    }
    let mut out = HashMap::new();
    let (mut padded, mut split) = (vec![], vec![]);
    for (g, ts) in wanted {
        let (f, parts) = &groups[g];
        if !landing_pads(img, f).is_empty() {
            padded.extend(ts);
            continue;
        }
        let (ins, _) = decode_group(img, f, parts);
        let ips: HashSet<u64> = ins.iter().map(|i| i.ip()).collect();
        let (at, off): (Vec<u64>, Vec<u64>) = ts.into_iter().map(|t| base + t as u64).partition(|va| ips.contains(va));
        split.extend(off.iter().map(|va| (va - base) as u32));
        if !at.is_empty() {
            out.insert(f.begin, at);
        }
    }
    // Left to the dispatcher (a jump there still finds no function): say which.
    for t in padded.iter().take(20) {
        eprintln!("not an entry (its function has landing pads): {t:#x}");
    }
    for t in split.iter().take(20) {
        eprintln!("not an entry (inside an instruction of its function): {t:#x}");
    }
    out
}

/// Lifts every non-chained `.pdata` function of `img` into one C unit.
pub fn lift_image(img: &Image) -> Result<lift::Unit> {
    let funcs = group_chained(img, discover::discover(img)?);
    let base = img.info.image_base;
    let starts: HashSet<u64> = funcs.iter().map(|(f, _)| base + f.begin as u64).collect();
    let decoded: Vec<_> = funcs
        .iter()
        .map(|(f, parts)| {
            let (ins, _) = decode_group(img, f, parts);
            let tables = tables_for(img, f, &ins);
            (f, parts, ins, tables)
        })
        .collect();
    let fins: Vec<lift::FuncIn> = decoded
        .iter()
        .map(|(f, parts, ins, tables)| lift::FuncIn {
            begin_va: base + f.begin as u64,
            end_va: base + f.end as u64,
            ins,
            tables: tables.clone(),
            extra: extra_ranges(img, parts),
            landing_pads: landing_pads(img, f),
            entries: vec![],
        })
        .collect();
    let is_func = |va: u64| starts.contains(&va);
    Ok(lift::lift_unit(
        &lift::Ctx {
            image_base: base,
            is_func: &is_func,
        },
        &fins,
    ))
}

fn unit_name(ext: bool, n_old: &mut usize, n_new: &mut usize) -> String {
    let n = if ext { n_new } else { n_old };
    *n += 1;
    if ext { format!("tu_x{:04}", *n - 1) } else { format!("tu_{:05}", *n - 1) }
}

/// Writes `data` to `path` unless the file already holds exactly that, so an unchanged unit keeps its modification time
/// and a build tool (make) skips recompiling it: a lifter fix rebuilds only the units whose generated C changed.
fn write_if_changed(path: &std::path::Path, data: impl AsRef<[u8]>) -> std::io::Result<()> {
    if std::fs::read(path).is_ok_and(|old| old == data.as_ref()) {
        return Ok(());
    }
    std::fs::write(path, data)
}

/// Largest amount of x86 code (bytes) one split unit may hold.
pub const UNIT_BYTES: usize = 600_000;

/// Writes the whole image as `tu_*.c` files of `per_unit` functions, plus `table.c`, `rt.h` and
/// `rt_host.c`, into `dir`. Returns (functions, lifted, stubbed, units).
pub fn lift_image_split(
    img: &Image,
    dir: &std::path::Path,
    per_unit: usize,
    extra: &[u32],
) -> Result<(usize, usize, usize, usize)> {
    use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};
    let funcs = group_chained(img, discover::discover_with(img, extra)?);
    // Functions only the extra seeds found go into units of their own (`tu_x*.c`), so adding seeds to a built title
    // does not move the others between units.
    let plain: HashSet<u32> = if extra.is_empty() {
        HashSet::new()
    } else {
        group_chained(img, discover::discover(img)?).iter().map(|(f, _)| f.begin).collect()
    };
    let base = img.info.image_base;
    let entries = inner_entries(img, &funcs);
    eprintln!(
        "{} entries inside {} functions (jumps into the middle of them)",
        entries.values().map(Vec::len).sum::<usize>(),
        entries.len()
    );
    let starts: HashSet<u64> = funcs
        .iter()
        .map(|(f, _)| base + f.begin as u64)
        .chain(entries.values().flatten().copied())
        .collect();
    let is_func = |va: u64| starts.contains(&va);
    let ctx = lift::Ctx {
        image_base: base,
        is_func: &is_func,
    };
    std::fs::create_dir_all(dir)?;
    write_if_changed(&dir.join("rt.h"), lift::RT_HEADER)?;
    write_if_changed(&dir.join("rt_host.c"), RT_HOST_C)?;
    write_if_changed(&dir.join("rt_unit.c"), RT_UNIT_C)?;
    let mut all: Vec<u64> = starts.iter().copied().collect();
    all.sort_unstable();
    let pads: Vec<(u64, u64)> = funcs
        .iter()
        .flat_map(|(f, _)| landing_pads(img, f).into_iter().map(move |p| (base + f.begin as u64, p)))
        .collect();
    write_if_changed(&dir.join("table.c"), lift::table_unit(base, &all, &pads))?;
    // Units are `per_unit` functions, further cut so no unit holds more than UNIT_BYTES of x86 code
    // (a few giant functions would otherwise make one unit take far longer than the rest).
    let (old, new): (Vec<_>, Vec<_>) = funcs.iter().cloned().partition(|(f, _)| extra.is_empty() || plain.contains(&f.begin));
    let mut chunks: Vec<(String, &[(Func, Vec<Func>)])> = vec![];
    let (mut n_old, mut n_new) = (0, 0);
    for (list, ext) in [(&old, false), (&new, true)] {
    for part in list.chunks(per_unit) {
        let (mut from, mut bytes) = (0, 0usize);
        for (i, (f, parts)) in part.iter().enumerate() {
            let sz = (f.end - f.begin) as usize + parts.iter().map(|p| (p.end - p.begin) as usize).sum::<usize>();
            if i > from && bytes + sz > UNIT_BYTES {
                chunks.push((unit_name(ext, &mut n_old, &mut n_new), &part[from..i]));
                (from, bytes) = (i, 0);
            }
            bytes += sz;
        }
        chunks.push((unit_name(ext, &mut n_old, &mut n_new), &part[from..]));
    }
    }
    let threads = std::thread::available_parallelism().map_or(4, |n| n.get());
    let (next, lifted, stubbed) = (AtomicUsize::new(0), AtomicUsize::new(0), AtomicUsize::new(0));
    std::thread::scope(|sc| -> Result<()> {
        let hs: Vec<_> = (0..threads)
            .map(|_| {
                sc.spawn(|| -> Result<()> {
                    loop {
                        let n = next.fetch_add(1, Relaxed);
                        let Some((name, part)) = chunks.get(n) else {
                            return Ok(());
                        };
                        let decoded: Vec<_> = part
                            .iter()
                            .map(|(f, parts)| {
                                let (ins, _) = decode_group(img, f, parts);
                                let tables = tables_for(img, f, &ins);
                                (f, parts, ins, tables)
                            })
                            .collect();
                        let fins: Vec<lift::FuncIn> = decoded
                            .iter()
                            .map(|(f, parts, ins, tables)| lift::FuncIn {
                                begin_va: base + f.begin as u64,
                                end_va: base + f.end as u64,
                                ins,
                                tables: tables.clone(),
                                extra: extra_ranges(img, parts),
                                landing_pads: landing_pads(img, f),
                                entries: entries.get(&f.begin).cloned().unwrap_or_default(),
                            })
                            .collect();
                        let u = lift::lift_part(&ctx, &fins);
                        lifted.fetch_add(u.lifted.len(), Relaxed);
                        stubbed.fetch_add(u.failed.len(), Relaxed);
                        write_if_changed(&dir.join(format!("{name}.c")), u.c)?;
                    }
                })
            })
            .collect();
        for h in hs {
            h.join().map_err(|_| anyhow::anyhow!("lift thread panicked"))??;
        }
        Ok(())
    })?;
    Ok((funcs.len(), lifted.into_inner(), stubbed.into_inner(), chunks.len()))
}

/// How much of an image the current lifter handles, by first blocking reason.
#[derive(Default)]
pub struct Coverage {
    pub total: u64,
    pub lifted: u64,
    pub instructions_in_lifted: u64,
    /// `jmp reg` sites with no recovered table (lifted as dispatcher tail calls) and the functions holding them.
    pub untabled_jmps: u64,
    pub funcs_with_untabled_jmp: u64,
    /// Chained `.pdata` parts merged into their parents.
    pub merged_parts: u64,
    pub blockers: BTreeMap<String, u64>,
    /// Every unsupported instruction reason: (instructions, functions containing it).
    pub unsupported: BTreeMap<String, (u64, u64)>,
}

pub fn coverage(img: &Image) -> Result<Coverage> {
    let funcs = group_chained(img, discover::discover(img)?);
    let base = img.info.image_base;
    let starts: HashSet<u64> = funcs.iter().map(|(f, _)| base + f.begin as u64).collect();
    let is_func = |va: u64| starts.contains(&va);
    let ctx = lift::Ctx {
        image_base: base,
        is_func: &is_func,
    };
    let threads = std::thread::available_parallelism().map_or(4, |n| n.get());
    let chunk = funcs.len().div_ceil(threads).max(1);
    let mut cov = Coverage::default();
    std::thread::scope(|sc| {
        let hs: Vec<_> = funcs
            .chunks(chunk)
            .map(|part| {
                let ctx = &ctx;
                sc.spawn(move || {
                    let mut c = Coverage::default();
                    for (f, parts) in part {
                        let (ins, bad) = decode_group(img, f, parts);
                        c.total += 1;
                        let tables = tables_for(img, f, &ins);
                        let untabled = ins
                            .iter()
                            .filter(|i| {
                                i.code() == iced_x86::Code::Jmp_rm64
                                    && i.op0_kind() == iced_x86::OpKind::Register
                                    && !tables.contains_key(&i.ip())
                            })
                            .count() as u64;
                        c.untabled_jmps += untabled;
                        c.funcs_with_untabled_jmp += (untabled > 0) as u64;
                        c.merged_parts += parts.len() as u64;
                        let fi = lift::FuncIn {
                            begin_va: base + f.begin as u64,
                            end_va: base + f.end as u64,
                            ins: &ins,
                            tables,
                            extra: extra_ranges(img, parts),
                            landing_pads: landing_pads(img, f),
                            entries: vec![],
                        };
                        match lift::lift_function(ctx, &fi) {
                            Ok(_) if !bad => {
                                c.lifted += 1;
                                c.instructions_in_lifted += ins.len() as u64;
                            }
                            Ok(_) => *c.blockers.entry("decode failure".into()).or_default() += 1,
                            Err(e) => {
                                let mut seen = HashSet::new();
                                for u in lift::unsupported(ctx, &fi) {
                                    let r = u.split_once(": ").map_or(u.as_str(), |x| x.1);
                                    let e = c.unsupported.entry(r.to_string()).or_default();
                                    e.0 += 1;
                                    if seen.insert(r.to_string()) {
                                        e.1 += 1;
                                    }
                                }
                                // "<rva>: <reason>"; bucket by reason.
                                let r = e.split_once(": ").map_or(e.as_str(), |x| x.1);
                                *c.blockers.entry(r.to_string()).or_default() += 1;
                            }
                        }
                    }
                    c
                })
            })
            .collect();
        for h in hs {
            let c = h.join().unwrap();
            cov.total += c.total;
            cov.lifted += c.lifted;
            cov.instructions_in_lifted += c.instructions_in_lifted;
            cov.untabled_jmps += c.untabled_jmps;
            cov.funcs_with_untabled_jmp += c.funcs_with_untabled_jmp;
            cov.merged_parts += c.merged_parts;
            for (k, v) in c.blockers {
                *cov.blockers.entry(k).or_default() += v;
            }
            for (k, v) in c.unsupported {
                let e = cov.unsupported.entry(k).or_default();
                e.0 += v.0;
                e.1 += v.1;
            }
        }
    });
    Ok(cov)
}

/// Shapes of the instruction windows before every `jmp reg` that has no recovered table
/// (mnemonic + first operand kind of the previous 6 instructions), most common first.
pub fn unresolved_jmp_shapes(img: &Image) -> Result<Vec<(String, u64)>> {
    use iced_x86::Code;
    let funcs: Vec<Func> = discover::discover(img)?.into_iter().filter(|f| !f.chained).collect();
    let mut m: BTreeMap<String, u64> = BTreeMap::new();
    for f in &funcs {
        let (ins, _) = analyze::decode_function(img, f);
        let tables = tables_for(img, f, &ins);
        for (n, j) in ins.iter().enumerate() {
            if j.code() != Code::Jmp_rm64 || j.op0_kind() != iced_x86::OpKind::Register {
                continue;
            }
            if tables.contains_key(&j.ip()) {
                continue;
            }
            let sig: Vec<String> = ins[n.saturating_sub(6)..n]
                .iter()
                .map(|i| format!("{:?}", i.mnemonic()))
                .collect();
            *m.entry(sig.join(" ")).or_default() += 1;
        }
    }
    let mut v: Vec<_> = m.into_iter().collect();
    v.sort_by_key(|a| std::cmp::Reverse(a.1));
    Ok(v)
}

/// What the static checks of [`check_image`] found.
#[derive(Default, Debug)]
pub struct CheckReport {
    pub functions: usize,
    /// Pairs of functions whose extents overlap (a hard error: some code would be lifted twice).
    pub overlaps: Vec<(u32, u32)>,
    /// Functions whose decode hit an invalid instruction (a hard error).
    pub undecodable: Vec<u32>,
    /// Jump tables recovered, and those with a case that is not an instruction start of the function.
    pub tables: usize,
    pub tables_with_bad_cases: Vec<(u32, usize)>,
    /// Tables whose size came from a range check (the rest ran as far as their entries stayed in code).
    pub bounded_tables: usize,
    /// `jmp reg` with no recovered table (a tail call or an unrecovered switch: routed through the dispatcher).
    pub untabled_jumps: usize,
    /// Direct calls and jumps whose target is in no function and not a label of the caller (routed through the dispatcher).
    pub unresolved_direct: usize,
    /// Functions whose last instruction does not end the flow: they fall into whatever follows.
    pub falls_through: Vec<u32>,
}

impl CheckReport {
    /// Errors that must stop a lift.
    pub fn hard_errors(&self) -> usize {
        self.overlaps.len() + self.undecodable.len()
    }

    pub fn render(&self) -> String {
        use std::fmt::Write as _;
        let mut s = String::new();
        let _ = writeln!(s, "functions                         {}", self.functions);
        let _ = writeln!(s, "overlapping extents (hard)        {}", self.overlaps.len());
        let _ = writeln!(s, "undecodable functions (hard)      {}", self.undecodable.len());
        let _ = writeln!(s, "jump tables                       {} ({} bounded by a range check)", self.tables, self.bounded_tables);
        let _ = writeln!(s, "  with a case on no instruction   {}", self.tables_with_bad_cases.len());
        let _ = writeln!(s, "jmp reg with no table             {}", self.untabled_jumps);
        let _ = writeln!(s, "direct targets in no function     {}", self.unresolved_direct);
        let _ = writeln!(s, "functions that fall through       {}", self.falls_through.len());
        let mut worst = self.tables_with_bad_cases.clone();
        worst.sort_by_key(|x| std::cmp::Reverse(x.1));
        for (f, n) in worst.iter().take(5) {
            let _ = writeln!(s, "  worst table with bad cases: function {f:#x}: {n}");
        }
        for (a, b) in self.overlaps.iter().take(5) {
            let _ = writeln!(s, "  overlap: {a:#x} and {b:#x}");
        }
        for f in self.undecodable.iter().take(5) {
            let _ = writeln!(s, "  undecodable: {f:#x}");
        }
        for f in self.falls_through.iter().take(5) {
            let _ = writeln!(s, "  falls through: {f:#x}");
        }
        s
    }
}

/// Static checks over every function of an image, as the lifter will see them: extents that overlap, code that does not
/// decode, jump-table cases that land on no instruction, indirect and direct transfers the dispatcher has to resolve,
/// and functions that fall off their end.
pub fn check_image(img: &Image) -> Result<CheckReport> {
    use iced_x86::{FlowControl as FC, Mnemonic};
    let all = discover::discover(img)?;
    let funcs: Vec<Func> = all.iter().filter(|f| !f.chained).copied().collect();
    let mut r = CheckReport { functions: funcs.len(), ..Default::default() };
    let base = img.info.image_base;
    let starts: HashSet<u64> = funcs.iter().map(|f| base + f.begin as u64).collect();
    for w in funcs.windows(2) {
        if w[0].end > w[1].begin {
            r.overlaps.push((w[0].begin, w[1].begin));
        }
    }
    for f in &funcs {
        let (ins, bad) = analyze::decode_function(img, f);
        if bad {
            r.undecodable.push(f.begin);
        }
        let ips: HashSet<u64> = ins.iter().map(|i| i.ip()).collect();
        let tables = analyze::find_jump_tables(img, f, &ins);
        let tabled: HashSet<u64> = tables.iter().map(|t| base + t.jmp_rva as u64).collect();
        r.tables += tables.len();
        r.bounded_tables += tables.iter().filter(|t| t.bounded).count();
        let bad_cases: usize = tables
            .iter()
            .map(|t| t.targets.iter().filter(|&&x| (f.begin..f.end).contains(&x) && !ips.contains(&(base + x as u64))).count())
            .sum();
        if bad_cases > 0 {
            r.tables_with_bad_cases.push((f.begin, bad_cases));
        }
        for i in &ins {
            match i.flow_control() {
                FC::IndirectBranch if i.op0_kind() == iced_x86::OpKind::Register && !tabled.contains(&i.ip()) => r.untabled_jumps += 1,
                FC::Call | FC::UnconditionalBranch | FC::ConditionalBranch if i.op0_kind() == iced_x86::OpKind::NearBranch64 => {
                    let t = i.near_branch_target();
                    if !starts.contains(&t) && !ips.contains(&t) {
                        r.unresolved_direct += 1;
                    }
                }
                _ => {}
            }
        }
        if let Some(last) = ins.last() {
            let ends = matches!(last.flow_control(), FC::Return | FC::UnconditionalBranch | FC::IndirectBranch)
                || matches!(last.mnemonic(), Mnemonic::Int3 | Mnemonic::Ud2 | Mnemonic::Hlt)
                // A call that never returns (`__fastfail`, a noreturn helper) ends functions as well.
                || last.flow_control() == FC::Call;
            if !ends {
                r.falls_through.push(f.begin);
            }
        }
    }
    Ok(r)
}
