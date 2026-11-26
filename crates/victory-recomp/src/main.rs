use anyhow::{Context, Result, bail};
use victory_recomp::{analyze, coverage, image::Image, lift_image};

const USAGE: &str = "usage: victory-recomp analyze <pe> | coverage <pe> | unsupported <pe> | lift <pe> <out.c> | lift-split <pe> <outdir> <funcs-per-unit> | flatten <pe> <out.bin> | exports <pe>";

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let path = args.get(2).context(USAGE)?;
    let data = std::fs::read(path)?;
    let img = Image::parse(&data)?;
    match args[1].as_str() {
        "analyze" => print!("{}", analyze::analyze(&img)?.render()),
        "coverage" => {
            let c = coverage(&img)?;
            println!(
                "functions {}  fully lifted {} ({:.1}%)",
                c.total,
                c.lifted,
                c.lifted as f64 * 100.0 / c.total.max(1) as f64
            );
            let mut b: Vec<_> = c.blockers.iter().collect();
            b.sort_by(|a, b| b.1.cmp(a.1));
            println!(
                "jmp reg with no recovered table (dispatcher tail call): {} sites in {} functions; chained parts merged: {}",
                c.untabled_jmps, c.funcs_with_untabled_jmp, c.merged_parts
            );
            println!("first blocker per unlifted function (top 40):");
            for (k, v) in b.iter().take(40) {
                println!("  {v:>8}  {k}");
            }
        }
        "unsupported" => {
            // Every unsupported instruction (not only the first per function), by functions blocked.
            let c = coverage(&img)?;
            let mut b: Vec<_> = c.unsupported.iter().collect();
            b.sort_by_key(|a| std::cmp::Reverse(a.1.1));
            println!("unsupported reason: functions containing it, instructions (top 150)");
            for (k, v) in b.iter().take(150) {
                println!("  {:>8} {:>9}  {k}", v.1, v.0);
            }
        }
        "lift-split" => {
            let dir = std::path::PathBuf::from(args.get(3).context(USAGE)?);
            let per: usize = args.get(4).context(USAGE)?.parse()?;
            // Optional file of extra function starts, one RVA per line (hex with 0x).
            let extra: Vec<u32> = match args.get(5) {
                Some(f) => std::fs::read_to_string(f)?
                    .lines()
                    .filter_map(|l| u32::from_str_radix(l.trim().trim_start_matches("0x"), 16).ok())
                    .collect(),
                None => vec![],
            };
            let checks = victory_recomp::check_image(&img)?;
            eprint!("{}", checks.render());
            // Not yet fatal (step 2 is unfinished: the 4 undecodable functions of the title are known); `check` exits non-zero.
            if checks.hard_errors() > 0 {
                eprintln!("warning: {} hard errors in the static checks", checks.hard_errors());
            }
            let t = std::time::Instant::now();
            let (n, ok, stub, units) = victory_recomp::lift_image_split(&img, &dir, per, &extra)?;
            eprintln!(
                "{n} functions, {ok} lifted, {stub} stubbed, {units} units in {:.1}s",
                t.elapsed().as_secs_f64()
            );
        }
        "forms" => {
            // Every distinct instruction form (iced `Code`: mnemonic, operand kinds, prefixes) in the image with its
            // count, most used first: names and counts only, no bytes (input for `VICTORY_FORMS_LIST` of the form test).
            let mut counts: std::collections::HashMap<iced_x86::Code, u64> = Default::default();
            for f in victory_recomp::discover::discover(&img)? {
                for i in victory_recomp::analyze::decode_function(&img, &f).0 {
                    *counts.entry(i.code()).or_default() += 1;
                }
            }
            let mut v: Vec<_> = counts.into_iter().collect();
            v.sort_by_key(|&(c, n)| (std::cmp::Reverse(n), c as u32));
            for (c, n) in v {
                println!("{c:?} {n}");
            }
        }
        "check" => {
            // Static checks over every function; exits non-zero on hard errors.
            let r = victory_recomp::check_image(&img)?;
            print!("{}", r.render());
            if r.hard_errors() > 0 {
                bail!("{} hard errors", r.hard_errors());
            }
        }
        "tables" => {
            // The function holding an RVA, and the jump tables recovered in it (a debugging aid for a trap in a switch).
            let rva = u32::from_str_radix(args.get(3).context(USAGE)?.trim_start_matches("0x"), 16)?;
            let all = victory_recomp::discover::discover(&img)?;
            let f = all.iter().find(|f| f.begin <= rva && rva < f.end).context("no function holds it")?;
            println!("function {:#x}..{:#x}", f.begin, f.end);
            let (ins, _) = victory_recomp::analyze::decode_function(&img, f);
            let ips: std::collections::HashSet<u64> = ins.iter().map(|i| i.ip()).collect();
            for t in victory_recomp::analyze::find_jump_tables(&img, f, &ins) {
                let bad: Vec<String> = t
                    .targets
                    .iter()
                    .filter(|&&x| !ips.contains(&(img.info.image_base + x as u64)))
                    .map(|x| format!("{x:#x}"))
                    .collect();
                println!("jmp at {:#x}: table {:#x}, {} targets, not instruction starts: {bad:?}", t.jmp_rva, t.table_rva, t.targets.len());
            }
        }
        "tablelist" => {
            // Every jump table of every function, one per line (`jmp table cases bounded`), to compare two builds of
            // the recovery. `tablelist <pe> [seeds]`.
            let extra: Vec<u32> = match args.get(3) {
                Some(f) => std::fs::read_to_string(f)?
                    .lines()
                    .filter_map(|l| u32::from_str_radix(l.trim().trim_start_matches("0x"), 16).ok())
                    .collect(),
                None => vec![],
            };
            let funcs = victory_recomp::discover::discover_with(&img, &extra)?;
            for f in funcs.iter().filter(|f| !f.chained) {
                let (ins, _) = victory_recomp::analyze::decode_function(&img, f);
                for t in victory_recomp::analyze::find_jump_tables(&img, f, &ins) {
                    println!("{:#x} {:#x} {} {}", t.jmp_rva, t.table_rva, t.targets.len(), u8::from(t.bounded));
                }
            }
        }
        "escapes" => {
            // Starts to feed back as extra seeds (`lift-split ... <seeds>`): direct targets that leave their function
            // and land in none. `escapes <pe> [seeds]`, the seeds file the build uses so far.
            let extra: Vec<u32> = match args.get(3) {
                Some(f) => std::fs::read_to_string(f)?
                    .lines()
                    .filter_map(|l| u32::from_str_radix(l.trim().trim_start_matches("0x"), 16).ok())
                    .collect(),
                None => vec![],
            };
            let funcs = victory_recomp::discover::discover_with(&img, &extra)?;
            let (free, inside) = victory_recomp::discover::escaping_targets(&img, &funcs);
            eprintln!("{} targets in no function, {} into the middle of one", free.len(), inside.len());
            for t in free {
                println!("{t:#x}");
            }
            for t in inside {
                eprintln!("into a function: {t:#x}");
            }
        }
        "jmpshapes" => {
            for (k, v) in victory_recomp::unresolved_jmp_shapes(&img)?.iter().take(25) {
                println!("{v:>6}  {k}");
            }
        }
        "lift" => {
            let out = args.get(3).context(USAGE)?;
            let u = lift_image(&img)?;
            std::fs::write(out, &u.c)?;
            eprintln!(
                "lifted {} functions, {} stubbed",
                u.lifted.len(),
                u.failed.len()
            );
            for (va, e) in &u.failed {
                eprintln!("  stub {:#x}: {e}", va - img.info.image_base);
            }
        }
        "flatten" => {
            // Section data at RVA offsets: the layout the lifted code addresses (no relocations applied).
            let out = args.get(3).context(USAGE)?;
            let mut flat = vec![0u8; img.info.size_of_image as usize];
            for s in &img.info.sections {
                let raw = &data[s.file_offset as usize..(s.file_offset + s.file_size) as usize];
                let n = raw.len().min(s.virtual_size.max(s.file_size) as usize);
                flat[s.rva as usize..s.rva as usize + n].copy_from_slice(&raw[..n]);
            }
            std::fs::write(out, flat)?;
        }
        "exports" => {
            for e in &img.info.exports {
                if let (Some(n), Some(r)) = (&e.name, e.rva) {
                    println!("{n} {r:#x}");
                }
            }
        }
        _ => bail!(USAGE),
    }
    Ok(())
}
