//! Survey shader containers: bitcode, console additions and native code;
//! and list shader packs, naming each stage's container where one of the
//! given files holds it.

use std::borrow::Cow;
use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};

use anyhow::Result;
use victory_game::{sgd5, shaderpack};
use victory_shader::gfx10::{self, Instruction};
use victory_shader::{Container, DxilProgram, FourCC, HwStage, ShaderError, native_programs};

/// Files above this size are skipped (read whole into memory for now).
pub(crate) const MAX_FILE: u64 = 2 << 30;

/// A shader container in a file: as is, or packed in an SGD5 record.
pub(crate) enum Source<'a> {
    Raw { offset: usize, data: &'a [u8] },
    Packed(sgd5::Record<'a>),
}

impl Source<'_> {
    /// The container bytes; `None` if a packed record is corrupt.
    pub(crate) fn bytes(&self) -> Option<Cow<'_, [u8]>> {
        match self {
            Source::Raw { data, .. } => Some(Cow::Borrowed(data)),
            Source::Packed(r) => r.unpack().map(Cow::Owned),
        }
    }

    pub(crate) fn label(&self) -> String {
        match self {
            Source::Raw { offset, .. } => format!("@{offset:#x}"),
            Source::Packed(r) => format!("@{:#x} {}", r.offset, r.name.unwrap_or("?")),
        }
    }
}

/// The packed records of SGD5 data, and every container stored as is (the
/// executable keeps some records uncompressed), found by its magic.
pub(crate) fn sources(data: &[u8]) -> Vec<Source<'_>> {
    let mut out: Vec<Source<'_>> = Vec::new();
    if sgd5::is_sgd5(data) {
        out.extend(sgd5::records(data).into_iter().map(Source::Packed));
    }
    let raw = memchr::memmem::find_iter(data, b"DXBC")
        .filter(|&offset| Container::parse(&data[offset..]).is_ok())
        .map(|offset| Source::Raw {
            offset,
            data: &data[offset..],
        });
    out.extend(raw);
    out
}

#[derive(Default)]
struct Totals {
    containers: usize,
    with_dxil_part: usize,
    with_bitcode: usize,
    standard_layout: usize,
    xdat: usize,
    xbop: usize,
    kinds: BTreeMap<String, usize>,
    with_native: usize,
    native_stages: BTreeMap<HwStage, usize>,
    native_errors: BTreeMap<String, usize>,
    decoded: usize,
    instructions: usize,
    unnamed: BTreeMap<String, usize>,
    decode_errors: BTreeMap<&'static str, usize>,
    files_with_shaders: usize,
    /// A native program's hash instructions -> its container's BLAKE3
    /// (first 8 bytes, hex, as the GPU side logs it) and name.
    programs: HashMap<[u8; 24], String>,
    /// The same, by the hash a shader database stores for the container.
    engine: HashMap<u64, String>,
}

pub fn run(paths: Vec<PathBuf>, verbose: bool, disasm: bool, effects: bool) -> Result<()> {
    let paths = crate::default_paths(paths)?;
    let mut t = Totals::default();
    let mut packs = Vec::new();
    for root in &paths {
        for entry in walkdir::WalkDir::new(root)
            .into_iter()
            .filter_map(Result::ok)
        {
            if !entry.file_type().is_file() {
                continue;
            }
            let path = entry.path();
            if entry.metadata().map(|m| m.len()).unwrap_or(0) > MAX_FILE {
                eprintln!("skipping large file {}", path.display());
                continue;
            }
            let Ok(data) = std::fs::read(path) else {
                continue;
            };
            if effects {
                list_effects(path, &data);
                continue;
            }
            if data.starts_with(shaderpack::MAGIC) {
                packs.push((path.to_owned(), data));
                continue;
            }
            let mut found_here = 0;
            let engine: HashMap<usize, u64> = sgd5::databases(&data)
                .iter()
                .flat_map(|db| db.shaders.iter().flatten())
                .map(|s| (s.record.offset, s.hash))
                .collect();
            for source in sources(&data) {
                let Some(bytes) = source.bytes() else {
                    eprintln!("corrupt record {}{}", path.display(), source.label());
                    continue;
                };
                let Ok(container) = Container::parse(&bytes) else {
                    continue;
                };
                found_here += 1;
                t.containers += 1;
                let dxil = container.part(FourCC::DXIL);
                t.with_dxil_part += usize::from(dxil.is_some());
                let program = dxil.and_then(|p| DxilProgram::parse(p.data).ok());
                let mut listing = Vec::new();
                // Independent of the bitcode: titles may strip it.
                let code = dxil.or(container.part(FourCC::ACCC));
                let native = match code.map(|p| native_programs(p.data)) {
                    Some(Ok(programs)) if !programs.is_empty() => {
                        t.with_native += 1;
                        let mut list = Vec::new();
                        // `parse` checked the size field against the data.
                        let size = u32::from_le_bytes(bytes[0x18..0x1c].try_into()?) as usize;
                        let name = match &source {
                            Source::Raw { offset, .. } => {
                                let file = path.file_name().unwrap_or_default();
                                format!("{}@{offset:#x}", file.to_string_lossy())
                            }
                            Source::Packed(r) => r.name.unwrap_or("?").to_owned(),
                        };
                        let id = format!("{}:{name}", &blake3::hash(&bytes[..size]).to_hex()[..16]);
                        if let Source::Packed(r) = &source
                            && let Some(&hash) = engine.get(&r.offset)
                        {
                            t.engine.entry(hash).or_insert_with(|| id.clone());
                        }
                        for n in &programs {
                            if let Some(key) = shaderpack::program_id(&n.code) {
                                t.programs.entry(key).or_insert_with(|| id.clone());
                            }
                            *t.native_stages.entry(n.stage).or_default() += 1;
                            list.push(format!("{:?} {} B", n.stage, n.code.len()));
                            if disasm {
                                listing.push(format!("    {:?} stage I/O: {:?}", n.stage, n.io));
                                listing.extend(n.layout.iter().map(|p| format!("    {p:?}")));
                                listing.extend(n.used.iter().map(|u| format!("    {u:?}")));
                            }
                            match gfx10::decode(&n.code) {
                                Ok(insts) => {
                                    t.decoded += 1;
                                    t.instructions += insts.len();
                                    for i in insts.iter().filter(|i| i.name().is_none()) {
                                        let key = format!("{:?} {:#x}", i.encoding, i.opcode);
                                        *t.unnamed.entry(key).or_default() += 1;
                                    }
                                    if disasm {
                                        listing.extend(
                                            insts.iter().map(|i| format_instruction(i, n.wave64)),
                                        );
                                    }
                                }
                                Err(e) => {
                                    let reason = match e {
                                        ShaderError::Isa { reason, .. } => reason,
                                        _ => "other",
                                    };
                                    *t.decode_errors.entry(reason).or_default() += 1;
                                    list.push(format!("({e})"));
                                }
                            }
                        }
                        format!(", native {}", list.join(" "))
                    }
                    Some(Err(e)) => {
                        *t.native_errors.entry(e.to_string()).or_default() += 1;
                        format!(", native: {e}")
                    }
                    _ => String::new(),
                };
                let line = match &program {
                    Some(p) => {
                        t.with_bitcode += 1;
                        let x = p.xbox_extras();
                        t.standard_layout += usize::from(p.is_standard_layout());
                        t.xdat += usize::from(x.xdat);
                        t.xbop += usize::from(x.xbop);
                        *t.kinds.entry(format!("{:?}", p.kind)).or_default() += 1;
                        format!(
                            "{:?} sm{}.{} bitcode {} B, xdat {}, xbop {}",
                            p.kind,
                            p.shader_model.0,
                            p.shader_model.1,
                            p.bitcode.len(),
                            x.xdat,
                            x.xbop
                        )
                    }
                    None => "no parsable DXIL program".to_owned(),
                } + &native;
                if verbose {
                    println!("{}{}: {line}", path.display(), source.label());
                    for l in &listing {
                        println!("{l}");
                    }
                }
            }
            t.files_with_shaders += usize::from(found_here > 0);
        }
    }
    println!(
        "containers found       {} (in {} files)",
        t.containers, t.files_with_shaders
    );
    println!("  with DXIL part       {}", t.with_dxil_part);
    println!("  with LLVM bitcode    {}", t.with_bitcode);
    println!("  PC-standard layout   {}", t.standard_layout);
    println!("  XDAT prefix          {}", t.xdat);
    println!("  xbop trailer         {}", t.xbop);
    for (k, n) in &t.kinds {
        println!("  {k:<20} {n}");
    }
    println!("  with native code     {}", t.with_native);
    for (stage, n) in &t.native_stages {
        println!("    {:<18} {n}", format!("{stage:?}"));
    }
    for (e, n) in &t.native_errors {
        println!("  native unreadable    {n}: {e}");
    }
    if t.with_native > 0 {
        println!(
            "  decoded programs     {} ({} instructions)",
            t.decoded, t.instructions
        );
        for (e, n) in &t.decode_errors {
            println!("  decode failed        {n}: {e}");
        }
        let mut unnamed: Vec<_> = t.unnamed.iter().collect();
        unnamed.sort_by_key(|&(_, n)| std::cmp::Reverse(*n));
        for (op, n) in unnamed.iter().take(10) {
            println!("  opcode not in table  {op} x{n}");
        }
    }
    if t.containers == 0 && packs.is_empty() {
        println!("(none found: shaders may sit in compressed or encrypted archives)");
    }
    for (path, data) in &packs {
        list_pack(path, data, &t, verbose);
    }
    Ok(())
}

/// A shader pack's pipelines; with `verbose` one line each, naming every
/// stage's container (BLAKE3 prefix and name) found in the other files.
fn list_pack(path: &Path, data: &[u8], t: &Totals, verbose: bool) {
    let pack = match shaderpack::Pack::parse(data) {
        Ok(pack) => pack,
        Err(e) => return println!("{}: {e}", path.display()),
    };
    let name = |code: &[u8]| {
        let id = shaderpack::program_id(code).and_then(|id| t.programs.get(&id));
        id.map_or("?", String::as_str)
    };
    let hex = |h: u64| format!("{h:016x}");
    let (mut graphics, mut compute, mut stages, mut named, mut keyed) = (0, 0, 0, 0, 0);
    for g in pack.graphics() {
        graphics += 1;
        keyed += usize::from(t.engine.contains_key(&g.shader_hash));
        let mut line = format!(
            "graphics key={} shader={} pipeline={} depth-stencil={} rasterizer={} blend={}",
            hex(g.key),
            hex(g.shader_hash),
            hex(g.pipeline_hash),
            hex(g.depth_stencil_hash),
            hex(g.rasterizer_hash),
            hex(g.blend_hash)
        );
        if let (Some(f), Some((depth, count, topology)), Some((mask, samples, quality))) =
            (g.target_formats(), g.depth_targets_topology(), g.samples())
        {
            let f: Vec<_> = f.iter().map(u8::to_string).collect();
            line += &format!(
                " targets={count}:{} depth={depth} topology={topology} samples={samples}.{quality} mask={mask:#x}",
                f.join(",")
            );
        }
        if let Some(inputs) = g.input_layout() {
            let e: Vec<_> = (inputs.iter())
                .map(|e| {
                    let (s, i, f, slot, o) = (e.semantic, e.index, e.format, e.slot, e.offset);
                    format!("{s}{i}:{f}:{slot}:{o}:{}:{}", e.class, e.step_rate)
                })
                .collect();
            line += &format!(" inputs={}", e.join(","));
        }
        for (stage, s) in shaderpack::STAGES.iter().zip(&g.stages) {
            let Some(s) = s else { continue };
            stages += 1;
            let n = name(s.program);
            named += usize::from(n != "?");
            line += &format!(" {stage}={n}");
        }
        if verbose {
            println!("{line}");
        }
    }
    let mut compute_keyed = 0;
    for c in pack.compute() {
        compute += 1;
        compute_keyed += usize::from(t.engine.contains_key(&c.shader_hash));
        let n = name(c.stage.program);
        named += usize::from(n != "?");
        stages += 1;
        if verbose {
            println!("compute shader={} CS={n}", hex(c.shader_hash));
        }
    }
    println!(
        "{}: shader pack ({}, compiler {}, database {:016x}): {graphics} graphics and {compute} compute pipelines",
        path.display(),
        pack.driver(),
        pack.compiler(),
        pack.fingerprint()
    );
    println!("  stages with their container found   {named} of {stages}");
    println!("  graphics shader hash in a database  {keyed} of {graphics}");
    println!("  compute shader hash in a database   {compute_keyed} of {compute}");
}

/// Effect -> technique -> pass -> shaders (name, container hash, the
/// engine's hash) and the state each pass sets, for every database in `data`.
fn list_effects(path: &std::path::Path, data: &[u8]) {
    for db in sgd5::databases(data) {
        let passes: usize = (db.effects.iter())
            .flat_map(|e| &e.techniques)
            .map(|t| t.passes.len())
            .sum();
        println!(
            "{}@{:#x}: {} effects, {passes} passes",
            path.display(),
            db.start,
            db.effects.len()
        );
        let mut containers = HashMap::new();
        for e in &db.effects {
            println!("effect {}", e.name);
            for t in &e.techniques {
                println!("  technique {}", t.name);
                for (i, p) in t.passes.iter().enumerate() {
                    let mut parts = Vec::new();
                    for (slot, index) in p.shaders.iter().enumerate() {
                        let Some(index) = *index else { continue };
                        let s = &db.shaders[slot][index];
                        let container = containers.entry((slot, index)).or_insert_with(|| {
                            s.record.unpack().map_or("corrupt".to_owned(), |c| {
                                blake3::hash(&c).to_hex()[..16].to_owned()
                            })
                        });
                        let name = s.record.name.unwrap_or("?");
                        let stage = sgd5::STAGES[slot];
                        parts.push(format!("{stage} {name} {container} {:016x}", s.hash));
                    }
                    let states = [
                        ("rasterizer", p.rasterizer),
                        ("depth-stencil", p.depth_stencil),
                        ("blend", p.blend),
                    ];
                    for (what, bytes) in states {
                        if let Some(b) = bytes {
                            let hex: String = b.iter().map(|x| format!("{x:02x}")).collect();
                            parts.push(format!("{what} {hex}"));
                        }
                    }
                    if let Some(r) = p.stencil_ref {
                        parts.push(format!("stencil ref {r:#x}"));
                    }
                    println!("    pass {i}: {}", parts.join(", "));
                }
            }
        }
    }
}

fn format_instruction(i: &Instruction, wave64: bool) -> String {
    let words: Vec<_> = i.words.iter().map(|w| format!("{w:08X}")).collect();
    let text = match i.name() {
        Some(_) => i.text(wave64),
        None => format!("<{:?} {:#x}>", i.encoding, i.opcode),
    };
    format!("    {:06x}: {:<44} {text}", i.offset, words.join(" "))
}

#[cfg(test)]
mod sources_tests {
    use super::{Container, Source, sources};

    /// A minimal PC container: the 0x20-byte header, no parts. The digest is
    /// not checked on parse.
    fn container() -> Vec<u8> {
        let mut out = b"DXBC".to_vec();
        out.extend_from_slice(&[0; 16]);
        out.extend_from_slice(&1u16.to_le_bytes()); // version 1.0
        out.extend_from_slice(&0u16.to_le_bytes());
        out.extend_from_slice(&0x20u32.to_le_bytes()); // total size
        out.extend_from_slice(&0u32.to_le_bytes()); // part count
        out
    }

    /// A stored (uncompressed) deflate block holding `payload`: BFINAL=1,
    /// BTYPE=00, LEN, its one's complement, the bytes.
    fn stored_deflate(payload: &[u8]) -> Vec<u8> {
        let len = payload.len() as u16;
        let nlen = !len;
        let mut out = vec![0x01, len as u8, (len >> 8) as u8, nlen as u8, (nlen >> 8) as u8];
        out.extend_from_slice(payload);
        out
    }

    /// An SGD5 record as the marker scan reads it: a length-prefixed name,
    /// the kind byte, `n`, the 0x000DDCE7 marker, `n` again, the unpacked
    /// size, then the packed stream.
    fn sgd5_record(name: &str, kind: u8, payload: &[u8]) -> Vec<u8> {
        let packed = stored_deflate(payload);
        let n = (packed.len() + 12) as u32;
        let mut out = vec![0xF4, 0x01, 0, 0, name.len() as u8 + 1];
        out.extend_from_slice(name.as_bytes());
        out.extend_from_slice(&[0, kind]);
        for v in [n, 0x000D_DCE7, n, payload.len() as u32] {
            out.extend_from_slice(&v.to_le_bytes());
        }
        out.extend_from_slice(&packed);
        out
    }

    #[test]
    fn input_without_shaders_yields_nothing() {
        assert!(sources(&[]).is_empty());
        assert!(sources(b"title bytes without magic").is_empty());
        // The SGD5 magic alone is no record, and a "DXBC" with no header
        // behind it is no container.
        assert!(sources(b"SGD5\x00\x00\x00\x00").is_empty());
        assert!(sources(b"DXBC").is_empty());
        // A "DXBC" whose size field disagrees with the data is not a container…
        let mut fake = container();
        fake.truncate(0x18);
        fake.extend_from_slice(&0xFFFFu32.to_le_bytes()); // total > data.len()
        fake.extend_from_slice(&0u32.to_le_bytes()); // part count
        assert!(sources(&fake).is_empty());
        // …but a valid container after a fake magic is still found.
        let at = fake.len();
        fake.extend_from_slice(&container());
        let found = sources(&fake);
        assert_eq!(found.len(), 1);
        assert!(matches!(&found[0], Source::Raw { offset, .. } if *offset == at));
    }

    #[test]
    fn every_raw_container_in_a_file_is_found_in_order() {
        let c = container();
        let mut file = c.clone();
        file.extend_from_slice(b"middle");
        let second = file.len();
        file.extend_from_slice(&c);
        let found = sources(&file);
        assert_eq!(found.len(), 2);
        assert!(matches!(&found[0], Source::Raw { offset: 0, .. }));
        assert!(matches!(&found[1], Source::Raw { offset, .. } if *offset == second));
    }

    #[test]
    fn a_raw_source_keeps_the_whole_tail_and_labels_its_offset() {
        let c = container();
        let mut file = b"some prefix".to_vec();
        let at = file.len();
        file.extend_from_slice(&c);
        file.extend_from_slice(b"and a suffix");
        let found = sources(&file);
        assert_eq!(found.len(), 1);
        match &found[0] {
            // `bytes` is the tail from the magic; `Container::parse` bounds it
            // by the container's own size field.
            Source::Raw { offset, data } => {
                assert_eq!(*offset, at);
                assert_eq!(*data, &file[at..]);
            }
            Source::Packed(_) => panic!("expected a raw source"),
        }
        assert_eq!(found[0].label(), format!("@{at:#x}"));
        let bytes = found[0].bytes().unwrap();
        assert_eq!(&bytes[..], &file[at..]);
        assert!(Container::parse(&bytes).is_ok());
    }

    #[test]
    fn an_sgd5_record_is_listed_and_unpacks_to_its_container() {
        let c = container();
        let mut file = b"SGD5".to_vec();
        file.extend_from_slice(&[1, 2, 3, 4]); // no database table parses: the marker scan runs
        file.extend_from_slice(&sgd5_record("PS_x", 0x32, &c));
        let marker = memchr::memmem::find(&file, &0x000D_DCE7u32.to_le_bytes()).unwrap();
        let inner = memchr::memmem::find(&file, b"DXBC").unwrap();

        let found = sources(&file);
        // The packed record first, then the raw scan, which also finds the
        // container inside the stored-deflate payload.
        assert_eq!(found.len(), 2);
        match &found[0] {
            Source::Packed(r) => {
                assert_eq!(r.offset, marker);
                assert_eq!(r.name, Some("PS_x"));
            }
            Source::Raw { .. } => panic!("expected the packed source first"),
        }
        assert_eq!(found[0].label(), format!("@{marker:#x} PS_x"));
        assert_eq!(found[0].bytes().as_deref(), Some(&c[..]));
        assert!(Container::parse(&found[0].bytes().unwrap()).is_ok());
        match &found[1] {
            Source::Raw { offset, .. } => assert_eq!(*offset, inner),
            Source::Packed(_) => panic!("expected the raw source second"),
        }
    }

    #[test]
    fn a_corrupt_record_is_listed_but_yields_no_bytes() {
        let payload = b"not deflate at all";
        let mut file = b"SGD5".to_vec();
        file.extend_from_slice(&[1, 2, 3, 4]);
        let mut rec = sgd5_record("CS_y", 0x3c, payload);
        let packed_at = rec.len() - stored_deflate(payload).len();
        rec[packed_at] = 0xFF; // a reserved deflate block type: the stream fails
        file.extend_from_slice(&rec);

        let found = sources(&file);
        assert_eq!(found.len(), 1);
        assert!(matches!(&found[0], Source::Packed(_)));
        assert_eq!(found[0].bytes(), None);
        assert!(found[0].label().contains("CS_y"));
    }
}
