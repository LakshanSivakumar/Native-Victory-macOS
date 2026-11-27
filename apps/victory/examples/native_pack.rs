//! Offline reproduction of the native guessed graphics path from a shipped shader pack.
//! Usage: cargo run --release -p victory --example native_pack -- <shaders.rpf> [hash,hash,...]
//! Reports identities and outcomes only; no game code is executed or dumped.

use std::collections::{BTreeMap, HashSet};
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};
use victory_game::shaderpack;
use victory_shader::{HwStage, emit};

fn main() -> anyhow::Result<()> {
    let path = std::env::args_os()
        .nth(1)
        .ok_or_else(|| anyhow::anyhow!("expected shader pack/archive"))?;
    let data = std::fs::read(path)?;
    let data = shaderpack::find(&data).ok_or_else(|| anyhow::anyhow!("no shader pack"))?;
    let pack = shaderpack::Pack::parse(&data).map_err(anyhow::Error::msg)?;
    let filter = std::env::args().nth(2);
    let mut seen = HashSet::new();
    let mut jobs = Vec::new();
    for g in pack.graphics() {
        // Same stage selection as pipeline::graphics_containers.
        for (stage, piece) in [
            (HwStage::Vs, g.stages[1].or(g.stages[2])),
            (HwStage::Ps, g.stages[0]),
        ] {
            let Some(piece) = piece else { continue };
            let Some(meta) = piece.metadata else { continue };
            let identity = (stage, blake3::hash(piece.program), blake3::hash(meta));
            if seen.insert(identity) {
                jobs.push((stage, piece));
            }
        }
    }
    let next = AtomicUsize::new(0);
    let outcomes = Mutex::new(Vec::new());
    std::thread::scope(|scope| {
        for _ in 0..4 {
            scope.spawn(|| {
                while let Some(&(stage, piece)) = jobs.get(next.fetch_add(1, Ordering::Relaxed)) {
                    let container =
                        emit::graphics::container(stage, piece.program, piece.metadata.unwrap());
                    let hash = container
                        .as_ref()
                        .map(|c| blake3::hash(c).to_hex()[..16].to_owned())
                        .unwrap_or_else(|_| blake3::hash(piece.program).to_hex()[..16].to_owned());
                    if let Some(wanted) = &filter
                        && !wanted.split(',').any(|w| w == hash)
                    {
                        continue;
                    }
                    let result = container.and_then(|c| {
                        emit::translate_with(
                            &c,
                            emit::Options {
                                wave32_host: true,
                                zero_copy_memory: true,
                                ..Default::default()
                            },
                        )
                    });
                    let reason = match result {
                        Ok(_) => "translated".to_owned(),
                        Err(victory_shader::ShaderError::Unsupported { what, .. }) => {
                            format!("refused: {what}")
                        }
                        Err(e) => format!("refused: {e}"),
                    };
                    outcomes.lock().unwrap().push((hash, stage, reason));
                }
            });
        }
    });
    let mut outcomes = outcomes.into_inner().unwrap();
    outcomes.sort();
    let mut counts = BTreeMap::<_, usize>::new();
    for (hash, stage, reason) in &outcomes {
        println!("{hash} {stage:?} {reason}");
        *counts.entry(reason).or_default() += 1;
    }
    println!("unique stage pieces {}", outcomes.len());
    for (reason, count) in counts {
        println!("{count:6} {reason}");
    }
    Ok(())
}
