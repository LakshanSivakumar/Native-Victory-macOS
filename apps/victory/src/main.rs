//! `victory`: the command-line front end.
//!
//! Every command reads the user's own files and writes only to stdout or to
//! the user's cache directory (outside the repository).

mod census;
mod inspect;
#[cfg(all(target_os = "macos", target_arch = "x86_64"))]
mod native_run;
#[cfg(windows)]
mod prebuild;
mod run;
mod shader;
#[cfg(windows)]
mod translate;
#[cfg(target_os = "macos")]
mod warm;
#[cfg(all(target_os = "macos", target_arch = "x86_64"))]
mod window;

use std::path::PathBuf;

use anyhow::Result;
use clap::{Parser, Subcommand, ValueEnum};

#[derive(Parser)]
#[command(
    name = "victory",
    version,
    about = "Run an Xbox Series title natively on Windows"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Summarise one PE image: headers, sections, imports, TLS, build id.
    Inspect {
        file: PathBuf,
        /// List every imported function.
        #[arg(long)]
        imports: bool,
        /// List every exported function.
        #[arg(long)]
        exports: bool,
    },
    /// Aggregate the imports of every PE image under the given paths and
    /// show which ones our runtime covers. Defaults to the configured
    /// game and test-title directories.
    Census {
        paths: Vec<PathBuf>,
        /// Do not count host pass-through as covered.
        #[arg(long)]
        strict: bool,
        /// List every missing `dll!function`, most used first.
        #[arg(long)]
        missing: bool,
    },
    /// List the `dll!export` pairs an import library describes.
    LibExports {
        lib: PathBuf,
        /// List every export, not just per-DLL counts.
        #[arg(long)]
        list: bool,
    },
    /// Find shader containers in files or directories (also inside engine
    /// archives, if uncompressed) and report their layout. Shader packs
    /// among the paths are listed last, their stages named from the
    /// containers found.
    ShaderInfo {
        paths: Vec<PathBuf>,
        /// Print one line per container found (and per pack pipeline).
        #[arg(long)]
        verbose: bool,
        /// Also list the decoded native instructions (implies --verbose).
        #[arg(long)]
        disasm: bool,
        /// Instead list the effects of shader databases: technique, pass,
        /// shaders (name, container hash, engine hash) and pass state.
        #[arg(long)]
        effects: bool,
    },
    /// Run every shader found (containers, SGD5 files) through the
    /// translator and report what it cannot handle yet.
    Translate {
        paths: Vec<PathBuf>,
        /// Check emitted DXIL with dxil.dll's validator (`VICTORY_DXIL`,
        /// else the newest Windows SDK's).
        #[arg(long)]
        validate: bool,
        /// Also create a host D3D12 pipeline for each emitted shader. With
        /// `--cache`: then make every pipeline of the title's shader pack
        /// into the host pipeline library `run` reads (on this GPU and
        /// driver; a second time only what is not there yet).
        #[arg(long)]
        pipelines: bool,
        /// Print one line per program and outcome.
        #[arg(long)]
        list: bool,
        /// Worker threads (default: every logical core).
        #[arg(long)]
        jobs: Option<usize>,
        /// Only translate into the shader cache `run` reads (the warm-up):
        /// every container found, once, nothing else checked or listed
        /// (with `--pipelines`, then the shader pack's pipelines).
        #[arg(long, conflicts_with_all = ["validate", "list"])]
        cache: bool,
    },
    /// Prepare, map and bind a title executable; with --execute, run it.
    Run {
        exe: PathBuf,
        /// What unimplemented imports do when called.
        #[arg(long, value_enum, default_value_t = Unimplemented::Trap)]
        unimplemented: Unimplemented,
        /// Disable host pass-through (everything must be implemented by us).
        #[arg(long)]
        strict: bool,
        /// Actually start the title. Without it, stop after binding.
        #[arg(long)]
        execute: bool,
        /// macOS native host: load and bind, print every import as served
        /// or stubbed, and exit without running the entry point.
        #[arg(long, conflicts_with = "execute")]
        list_imports: bool,
        /// The console model the title sees (device type, memory).
        #[arg(long, value_enum, default_value_t = Console::SeriesX)]
        console: Console,
        /// macOS native host: no window; frames are presented to an offscreen layer.
        #[arg(long)]
        headless: bool,
        /// Arguments for the title, after `--` (it reads the whole command
        /// line; a title that relaunches itself passes some).
        #[arg(last = true)]
        title_args: Vec<String>,
    },
    /// Show the resolved configuration and check it.
    Config,
}

#[derive(Clone, Copy, ValueEnum)]
enum Unimplemented {
    Trap,
    Zero,
}

#[derive(Clone, Copy, ValueEnum)]
enum Console {
    SeriesX,
    SeriesS,
}

/// SIGTERM, SIGINT and SIGHUP stop the process after its GPU work is woken and drained
/// (`victory_gpu::stop_gpu`), not in the middle of it: a process killed while its command buffers
/// sleep on events can leave the GPU memory they hold allocated until a reboot. The signals are
/// blocked here, before any other thread starts (threads inherit the mask), and taken by a thread
/// of their own with `sigwait`, so the stop runs as ordinary code, not in a signal handler.
#[cfg(target_os = "macos")]
fn stop_gpu_on_signals() {
    unsafe extern "C" {
        fn pthread_sigmask(how: i32, set: *const u32, old: *mut u32) -> i32;
        fn sigwait(set: *const u32, sig: *mut i32) -> i32;
    }
    const SIG_BLOCK: i32 = 1;
    const SIG_UNBLOCK: i32 = 2;
    let set: u32 = [1u32 /* SIGHUP */, 2 /* SIGINT */, 15 /* SIGTERM */]
        .iter()
        .fold(0, |set, sig| set | 1 << (sig - 1));
    // SAFETY: libSystem's documented calls with a valid set (`sigset_t` is 32 bits on macOS).
    if unsafe { pthread_sigmask(SIG_BLOCK, &set, std::ptr::null_mut()) } != 0 {
        return;
    }
    let spawned = std::thread::Builder::new().name("stop-on-signal".into()).spawn(move || {
        let mut sig = 0;
        // SAFETY: as above; `sig` receives the signal taken.
        if unsafe { sigwait(&set, &mut sig) } != 0 {
            return;
        }
        // A second signal stops the process at once, should the drain or the exit hang.
        // SAFETY: as above.
        unsafe { pthread_sigmask(SIG_UNBLOCK, &set, std::ptr::null_mut()) };
        tracing::warn!(signal = sig, "stopping: waking and draining the GPU's queued work first");
        // A command buffer still running on the GPU gets up to 20 s in all: run-native.sh's
        // `timeout` gives a run 30 s past its --seconds stop.
        let (in_flight, left) =
            victory_gpu::stop_gpu(std::time::Duration::from_secs(3), std::time::Duration::from_secs(20));
        tracing::warn!(in_flight, left, "stopped the GPU's work; exiting");
        std::process::exit(128 + sig);
    });
    if spawned.is_err() {
        // No thread to take them: leave the signals their default action.
        // SAFETY: as above; no other thread has started yet.
        unsafe { pthread_sigmask(SIG_UNBLOCK, &set, std::ptr::null_mut()) };
    }
}

/// A fatal signal (SIGSEGV, SIGBUS, SIGILL, SIGFPE) is logged to stderr before the process dies
/// by it: nothing else says anything when native code faults, so a run that ends in a segfault
/// leaves the log without a line naming the fault (#392: a 122 s run died at `Segmentation fault:
/// 11` with no crash line, after GPU errors). The handler writes one line, with the signal, its
/// code and the faulting address, through async-signal-safe calls only — no allocation, no
/// tracing — then resets the disposition to `SIG_DFL` itself and re-raises, so the process dies
/// by the signal as before, with the same exit code and the same crash report. The explicit
/// reset (not just `SA_RESETHAND`) matters when the loader's `seh::install()` has since replaced
/// this handler and chains to it (`seh::signal::chain`): then the kernel entered the loader's
/// handler, `SA_RESETHAND` never applied, and a bare re-raise would loop through that handler
/// instead of ending the process.
#[cfg(unix)]
fn log_fatal_signals() {
    unsafe extern "C" fn fatal(sig: libc::c_int, info: *mut libc::siginfo_t, _: *mut libc::c_void) {
        /// `n` hex digits of `v` at `at`, async-signal-safe.
        fn hex(line: &mut [u8], at: usize, v: usize, n: usize) {
            for k in 0..n {
                line[at + k] = b"0123456789abcdef"[(v >> (4 * (n - 1 - k))) & 15];
            }
        }
        /// Two decimal digits of `v` at `at`, async-signal-safe.
        fn dec(line: &mut [u8], at: usize, v: usize) {
            line[at] = b'0' + (v / 10 % 10) as u8;
            line[at + 1] = b'0' + (v % 10) as u8;
        }
        const LINE: &[u8; 65] =
            b"victory: fatal signal 00 (code 00) at address 0x0000000000000000\n";
        let sig_at = LINE.windows(7).position(|w| w == b"signal ").unwrap() + 7;
        let code_at = LINE.windows(6).position(|w| w == b"(code ").unwrap() + 6;
        let addr_at = LINE.windows(2).position(|w| w == b"0x").unwrap() + 2;
        // SAFETY: the kernel passed a valid siginfo for this fault.
        let (addr, code) = unsafe { ((*info).si_addr() as usize, (*info).si_code as usize) };
        let mut line = *LINE;
        dec(&mut line, sig_at, sig.unsigned_abs() as usize);
        hex(&mut line, code_at, code, 2);
        hex(&mut line, addr_at, addr, 16);
        // SAFETY: a fixed line on the stack; write is async-signal-safe.
        unsafe { libc::write(2, line.as_ptr().cast(), line.len()) };
        // Reset the disposition ourselves before re-raising: sigaction is
        // async-signal-safe, and a zeroed action with SIG_DFL is the default on every unix
        // libc models. This covers the chained path (seh::signal::chain calls us from the
        // loader's handler, where SA_RESETHAND never applied): the pending raise — or the
        // faulting instruction re-run after that handler returns — now ends the process.
        let dfl = libc::sigaction {
            sa_sigaction: libc::SIG_DFL,
            ..unsafe { std::mem::zeroed() }
        };
        // SAFETY: dfl is a valid action; the previous one is not wanted.
        unsafe { libc::sigaction(sig, &dfl, std::ptr::null_mut()) };
        // SAFETY: the disposition is SIG_DFL again; raise terminates.
        unsafe { libc::raise(sig) };
    }
    // SAFETY: sa_mask left empty (an empty set); a zeroed struct on every unix libc models.
    let act = libc::sigaction {
        sa_sigaction: fatal as *const () as libc::sighandler_t,
        sa_flags: libc::SA_SIGINFO | libc::SA_RESETHAND,
        ..unsafe { std::mem::zeroed() }
    };
    for sig in [libc::SIGSEGV, libc::SIGBUS, libc::SIGILL, libc::SIGFPE] {
        // SAFETY: act outlives the calls; the old actions are not wanted.
        unsafe { libc::sigaction(sig, &act, std::ptr::null_mut()) };
    }
}

#[cfg(all(test, unix))]
mod fatal_signal_tests {
    use super::log_fatal_signals;
    use std::os::fd::FromRawFd;

    /// A child that installs the logger and faults: the line naming the fault comes out on
    /// stderr, and the child still dies by the signal, with the same status a run without the
    /// logger leaves (#392).
    #[test]
    fn a_fatal_signal_is_logged_before_the_process_dies_by_it() {
        let mut pipe = [0; 2];
        // SAFETY: a two-word buffer for two fds.
        assert_eq!(unsafe { libc::pipe(pipe.as_mut_ptr()) }, 0);
        // SAFETY: the test's own process; the child takes the calls below.
        let child = unsafe { libc::fork() };
        assert!(child >= 0);
        if child == 0 {
            // SAFETY: the child only takes async-signal-safe calls until it faults.
            unsafe {
                libc::close(pipe[0]);
                libc::dup2(pipe[1], 2);
            }
            log_fatal_signals();
            // SAFETY: the handler logs this fault and re-raises it.
            unsafe { libc::raise(libc::SIGSEGV) };
            // SAFETY: unreachable; the fault kills the child.
            unsafe { libc::_exit(0) };
        }
        // SAFETY: the write end is the child's now.
        unsafe { libc::close(pipe[1]) };
        // SAFETY: the read end is ours alone; the File closes it.
        let read = unsafe { std::fs::File::from_raw_fd(pipe[0]) };
        let mut line = Vec::new();
        std::io::Read::read_to_end(&mut std::io::BufReader::new(read), &mut line).unwrap();
        let mut status = 0;
        // SAFETY: a one-word status for this child.
        assert_eq!(unsafe { libc::waitpid(child, &mut status, 0) }, child);
        assert!(libc::WIFSIGNALED(status) && libc::WTERMSIG(status) == libc::SIGSEGV);
        assert!(
            line.starts_with(b"victory: fatal signal 11 (code "),
            "{:?}",
            String::from_utf8_lossy(&line)
        );
    }

    /// The action a second handler saved when it was installed over the logger; the second
    /// handler forwards to it, the way `seh::signal::chain` calls the handler `seh::install`
    /// replaced.
    static mut SAVED: Option<libc::sigaction> = None;

    /// Stands in for the loader's `on_signal`: forwards to the saved previous handler.
    unsafe extern "C" fn forwarder(
        sig: libc::c_int,
        info: *mut libc::siginfo_t,
        uc: *mut libc::c_void,
    ) {
        // SAFETY: written in the child before the signal below could arrive; read here only.
        let saved = unsafe { (&raw const SAVED).read() };
        if let Some(old) = saved {
            if old.sa_flags & libc::SA_SIGINFO != 0
                && old.sa_sigaction != libc::SIG_DFL
                && old.sa_sigaction != libc::SIG_IGN
            {
                // SAFETY: a SA_SIGINFO handler, called as the kernel would call it.
                let f: unsafe extern "C" fn(
                    libc::c_int,
                    *mut libc::siginfo_t,
                    *mut libc::c_void,
                ) = unsafe { std::mem::transmute(old.sa_sigaction) };
                unsafe { f(sig, info, uc) };
            }
        }
    }

    /// When a later handler chains to the logger (as `seh::install()` + `seh::signal::chain`
    /// do in a real run, where `SA_RESETHAND` never applies because the kernel entered the
    /// later handler), the line must still come out exactly once and the child must still
    /// die by the signal — a bare re-raise would loop through the chain instead.
    #[test]
    fn a_chained_fatal_signal_is_logged_once_and_still_kills() {
        let mut pipe = [0; 2];
        // SAFETY: a two-word buffer for two fds.
        assert_eq!(unsafe { libc::pipe(pipe.as_mut_ptr()) }, 0);
        // SAFETY: the test's own process; the child takes the calls below.
        let child = unsafe { libc::fork() };
        assert!(child >= 0);
        if child == 0 {
            // SAFETY: the child only takes async-signal-safe calls until it faults.
            unsafe {
                libc::close(pipe[0]);
                libc::dup2(pipe[1], 2);
            }
            log_fatal_signals();
            // A later handler that chains to the logger, as the loader's does.
            let act = libc::sigaction {
                sa_sigaction: forwarder as *const () as libc::sighandler_t,
                sa_flags: libc::SA_SIGINFO,
                ..unsafe { std::mem::zeroed() }
            };
            let mut old = std::mem::MaybeUninit::<libc::sigaction>::uninit();
            // SAFETY: old receives the logger's action; the write below precedes any signal.
            unsafe {
                libc::sigaction(libc::SIGSEGV, &act, old.as_mut_ptr());
                (&raw mut SAVED).write(Some(old.assume_init()));
                libc::raise(libc::SIGSEGV);
                libc::_exit(0);
            }
        }
        // SAFETY: the write end is the child's now.
        unsafe { libc::close(pipe[1]) };
        // SAFETY: the read end is ours alone; the File closes it.
        let read = unsafe { std::fs::File::from_raw_fd(pipe[0]) };
        let mut line = Vec::new();
        std::io::Read::read_to_end(&mut std::io::BufReader::new(read), &mut line).unwrap();
        let mut status = 0;
        // SAFETY: a one-word status for this child.
        assert_eq!(unsafe { libc::waitpid(child, &mut status, 0) }, child);
        assert!(libc::WIFSIGNALED(status) && libc::WTERMSIG(status) == libc::SIGSEGV);
        assert!(
            line.starts_with(b"victory: fatal signal 11 (code "),
            "{:?}",
            String::from_utf8_lossy(&line)
        );
        assert_eq!(
            line.iter().filter(|&&b| b == b'\n').count(),
            1,
            "exactly one line, not a loop: {:?}",
            String::from_utf8_lossy(&line)
        );
    }
}

/// Diagnostics: the heap bytes Rust code holds, for the statistics.
#[cfg(all(windows, feature = "diagnostics"))]
#[global_allocator]
static ALLOC: victory_gpu::CountingAlloc = victory_gpu::CountingAlloc;

fn main() -> Result<()> {
    victory_core::logging::init("info");
    #[cfg(target_os = "macos")]
    stop_gpu_on_signals();
    #[cfg(unix)]
    log_fatal_signals();
    // The audio recorder `run` starts for `VICTORY_AUDIO_CAPTURE`.
    #[cfg(all(windows, feature = "diagnostics"))]
    if victory_hle::modules::audio::capture::child() {
        return Ok(());
    }
    let cli = Cli::parse();
    match cli.command {
        Command::Inspect {
            file,
            imports,
            exports,
        } => inspect::run(&file, imports, exports),
        Command::Census {
            paths,
            strict,
            missing,
        } => census::run(paths, strict, missing),
        Command::LibExports { lib, list } => inspect::lib_exports(&lib, list),
        Command::ShaderInfo {
            paths,
            verbose,
            disasm,
            effects,
        } => shader::run(paths, verbose || disasm, disasm, effects),
        Command::Translate {
            paths,
            jobs,
            pipelines,
            cache: true,
            ..
        } => {
            #[cfg(windows)]
            {
                translate::warm(paths, jobs, pipelines)
            }
            // The native host's warm-up goes from the containers earlier runs
            // kept, and its Metal libraries and pipelines are made in the run.
            #[cfg(target_os = "macos")]
            {
                anyhow::ensure!(!pipelines, "--pipelines needs a Windows host for now");
                anyhow::ensure!(
                    paths.is_empty(),
                    "on a Mac the warm-up translates the containers earlier runs kept, not files: give no paths"
                );
                warm::native(jobs)
            }
            #[cfg(not(any(windows, target_os = "macos")))]
            {
                let _ = (paths, jobs, pipelines);
                anyhow::bail!("translate needs a Windows host for now")
            }
        }
        Command::Translate {
            paths,
            validate,
            pipelines,
            list,
            jobs,
            cache: false,
        } => {
            #[cfg(windows)]
            {
                translate::run(paths, validate, pipelines, list, jobs)
            }
            // Translation checks shaders with the Windows DXC host, which
            // other hosts do not have yet.
            #[cfg(not(windows))]
            {
                let _ = (paths, validate, pipelines, list, jobs);
                anyhow::bail!("translate needs a Windows host for now")
            }
        }
        Command::Run {
            exe,
            unimplemented,
            strict,
            execute,
            list_imports,
            console,
            headless,
            title_args,
        } => {
            let code = run::run(
                &exe,
                unimplemented,
                strict,
                execute,
                list_imports,
                console,
                &title_args,
                headless,
            )?;
            if code != 0 {
                // Preserve all 32 bits of the Windows process status.
                std::process::exit(code as i32);
            }
            Ok(())
        }
        Command::Config => show_config(),
    }
}

fn show_config() -> Result<()> {
    let cfg = victory_core::Config::load()?;
    let status = |p: &Option<PathBuf>| match p {
        Some(p) if p.exists() => format!("{}", p.display()),
        Some(p) => format!("{}  (does not exist)", p.display()),
        None => "(not set)".to_owned(),
    };
    println!(
        "config file : {}",
        cfg.source
            .as_ref()
            .map_or("(none)".into(), |p| p.display().to_string())
    );
    println!("game_dir    : {}", status(&cfg.game_dir));
    println!("cache_dir   : {}", cfg.cache_dir.display());
    println!("data_dir    : {}", cfg.data_dir.display());
    println!("fidelityfx  : {}", status(&cfg.fidelityfx_dir));
    for t in &cfg.test_titles {
        println!("test_title  : {}", status(&Some(t.clone())));
    }
    for p in cfg
        .game_dir
        .iter()
        .chain(&cfg.test_titles)
        .chain([&cfg.cache_dir, &cfg.data_dir])
    {
        if let Err(e) = victory_core::ensure_outside_repo(p) {
            println!("POLICY VIOLATION: {e}");
        }
    }
    Ok(())
}

/// Paths given on the command line, or the configured game + test titles.
pub(crate) fn default_paths(paths: Vec<PathBuf>) -> Result<Vec<PathBuf>> {
    if !paths.is_empty() {
        return Ok(paths);
    }
    let cfg = victory_core::Config::load()?;
    let all: Vec<PathBuf> = cfg.game_dir.into_iter().chain(cfg.test_titles).collect();
    anyhow::ensure!(
        !all.is_empty(),
        "no paths given and none configured (see victory.example.toml)"
    );
    Ok(all)
}
