//! Disassembles a dumped code piece: `VICTORY_PIECE=<file> [VICTORY_WAVE32=1] cargo run --example disasm`;
//! `VICTORY_CONTAINER=<file>` takes the code of a guessed graphics container (`FSPG` part) instead,
//! or of a title container's programs (as `VICTORY_DUMP_REFUSED` writes them; the patched code,
//! each program under a line with its stage, wave size and threads).
//! An `ACCC` part holding a compute container (`FSPX`) is read at its own wave size.

fn main() {
    let mut wave64 = std::env::var_os("VICTORY_WAVE32").is_none();
    let code = if let Ok(path) = std::env::var("VICTORY_CONTAINER") {
        let bytes = std::fs::read(path).expect("readable container");
        let c = victory_shader::Container::parse(&bytes).expect("a container");
        if let Some(dxil) = c.part(victory_shader::FourCC::DXIL) {
            for p in victory_shader::native_programs(dxil.data).expect("native programs") {
                println!("; {:?} wave64={} vgprs={} io={:?}", p.stage, p.wave64, p.vgprs, p.io);
                match victory_shader::gfx10::decode(&p.patched_code()) {
                    Ok(list) => list.iter().for_each(|i| {
                        let words: Vec<String> = i.words.iter().map(|w| format!("{w:08x}")).collect();
                        println!("{:04x}: {:<60} ; {}", i.offset, i.text(p.wave64), words.join(" "))
                    }),
                    Err(e) => println!("decode failed: {e}"),
                }
            }
            return;
        }
        let part = c.part(victory_shader::FourCC::ACCC).expect("an ACCC part");
        if part.data.starts_with(b"FSPX") {
            let p = victory_shader::native_programs(part.data).expect("an FSPX program").remove(0);
            wave64 = p.wave64;
            p.code
        } else {
            let word = |at: usize| u32::from_le_bytes(part.data[at..at + 4].try_into().unwrap()) as usize;
            part.data[16..16 + word(8)].to_vec()
        }
    } else {
        let path = std::env::var("VICTORY_PIECE").expect("VICTORY_PIECE names a dumped code piece");
        std::fs::read(path).expect("readable piece")
    };
    match victory_shader::gfx10::decode(&code) {
        Ok(list) => {
            for i in list {
                println!("{:04x}: {}", i.offset, i.text(wave64));
            }
        }
        Err(e) => println!("decode failed: {e}"),
    }
}
