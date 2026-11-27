//! Offline native translation. No device, pipeline or title is created. With `cache`, each
//! translation goes into the native shader cache `run` reads (the warm-up before a run), under
//! the options the environment selects: run it with the `VICTORY_*` variables of the run.

use std::collections::{BTreeMap, HashSet};
use std::path::PathBuf;
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

use anyhow::Result;
use victory_shader::emit;
use victory_shader::{Container, ShaderError};

use crate::shader::{MAX_FILE, sources};

pub fn run(paths: Vec<PathBuf>, list: bool, jobs: Option<usize>, cache: bool) -> Result<()> {
    let options = victory_gpu::native::shaders::options();
    println!("native options {options:?}");
    if cache {
        let cfg = victory_core::Config::load()?;
        let dir = victory_gpu::shader_cache::use_native_cache(cfg.cache_subdir("native-shaders")?);
        println!("native shader cache {dir:?}");
    }
    let jobs = jobs.unwrap_or(4).max(1);
    let seen = Mutex::new(HashSet::new());
    let outcomes = Mutex::new(Vec::new());
    for root in crate::default_paths(paths)? {
        for entry in walkdir::WalkDir::new(root)
            .into_iter()
            .filter_map(Result::ok)
        {
            if !entry.file_type().is_file() || entry.metadata()?.len() > MAX_FILE {
                continue;
            }
            let data = std::fs::read(entry.path())?;
            let sources = sources(&data);
            let next = AtomicUsize::new(0);
            std::thread::scope(|scope| {
                for _ in 0..jobs.min(sources.len()) {
                    scope.spawn(|| {
                        while let Some(source) = sources.get(next.fetch_add(1, Ordering::Relaxed)) {
                            let Some(bytes) = source.bytes() else {
                                continue;
                            };
                            if Container::parse(&bytes).is_err() {
                                continue;
                            }
                            // Raw sources extend to EOF. Hash only the checked container size,
                            // matching the runtime's logged shader identities.
                            let size =
                                u32::from_le_bytes(bytes[24..28].try_into().unwrap()) as usize;
                            let bytes = &bytes[..size];
                            let hash = blake3::hash(bytes);
                            if !seen.lock().unwrap().insert(hash) {
                                continue;
                            }
                            let reason = if cache {
                                // The run's own entry point: translated once, kept on disk.
                                match victory_gpu::native::shaders::translate(bytes) {
                                    Ok(shader) => format!("translated {:?}", shader.stage),
                                    Err(e) => format!("refused: {e}"),
                                }
                            } else {
                                match emit::translate_with(bytes, options) {
                                    Ok(shader) => format!("translated {:?}", shader.stage),
                                    Err(ShaderError::Unsupported { what, .. }) => {
                                        format!("refused: {what}")
                                    }
                                    Err(e) => format!("refused: {e}"),
                                }
                            };
                            outcomes.lock().unwrap().push((
                                format!(
                                    "{} {}{}",
                                    &hash.to_hex()[..16],
                                    entry.path().display(),
                                    source.label()
                                ),
                                reason,
                            ));
                        }
                    });
                }
            });
        }
    }
    let mut outcomes = outcomes.into_inner().unwrap();
    outcomes.sort();
    let mut counts = BTreeMap::<String, usize>::new();
    for (label, reason) in &outcomes {
        *counts.entry(reason.clone()).or_default() += 1;
        if list {
            println!("{label}  {reason}");
        }
    }
    println!("unique containers {}", outcomes.len());
    for (reason, count) in counts {
        println!("{count:6}  {reason}");
    }
    Ok(())
}
