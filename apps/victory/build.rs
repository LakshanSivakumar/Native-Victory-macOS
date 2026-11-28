mod build_git;

// The macOS host runs an x86-64 title under Rosetta. The default `__PAGEZERO`
// of an x86-64 executable is 4 GiB: the whole range below it is then
// unmappable, yet the title's 32-bit `XMemAlloc` / `XMemVirtualAlloc`
// addresses (and anything it keeps in a `u32`) must come from there. Shrink
// it to one page so the low 4 GiB can be reserved. Null still faults.
fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=build_git.rs");
    let manifest_dir = std::env::var_os("CARGO_MANIFEST_DIR")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| std::path::PathBuf::from("."));
    for path in build_git::watch_paths(&manifest_dir) {
        println!("cargo:rerun-if-changed={}", path.display());
    }
    println!(
        "cargo:rustc-env=VICTORY_GIT_COMMIT={}",
        build_git::short_commit(&manifest_dir)
    );

    let os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    let arch = std::env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();
    if os == "macos" && arch == "x86_64" {
        println!("cargo:rustc-link-arg-bins=-Wl,-pagezero_size,0x1000");
    }
}
