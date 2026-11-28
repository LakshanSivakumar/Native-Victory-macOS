//! Git `HEAD` tracking for `apps/victory/build.rs`: resolves both the worktree
//! `gitdir` and shared `commondir` so Cargo reruns `build.rs` whenever a commit
//! moves the checked-out branch ref, `packed-refs` updates, or `HEAD` changes (#533).

use std::path::{Path, PathBuf};

/// Resolved git directories `(git_dir, common_dir)` for the checkout containing `start`.
pub fn git_dirs(start: &Path) -> Option<(PathBuf, PathBuf)> {
    let dot_git = start
        .ancestors()
        .map(|d| d.join(".git"))
        .find(|p| p.exists())?;
    if dot_git.is_dir() {
        return Some((dot_git.clone(), dot_git));
    }
    let text = std::fs::read_to_string(&dot_git).ok()?;
    let rel = text.trim().strip_prefix("gitdir:")?.trim();
    let parent = dot_git.parent()?;
    let git_dir = resolve_against(parent, Path::new(rel));
    let common_dir = match std::fs::read_to_string(git_dir.join("commondir")) {
        Ok(common) if !common.trim().is_empty() => {
            resolve_against(&git_dir, Path::new(common.trim()))
        }
        _ => git_dir.clone(),
    };
    Some((git_dir, common_dir))
}

fn resolve_against(base: &Path, path: &Path) -> PathBuf {
    if path.is_absolute() {
        path.to_path_buf()
    } else {
        base.join(path)
    }
}

fn push_unique_existing(out: &mut Vec<PathBuf>, path: PathBuf) {
    if path.exists() && !out.contains(&path) {
        out.push(path);
    }
}

/// Existing filesystem paths Cargo must watch so `VICTORY_GIT_COMMIT` stays
/// current across commits, branch switches, packed refs, and linked worktrees.
pub fn watch_paths(start: &Path) -> Vec<PathBuf> {
    let Some((git_dir, common_dir)) = git_dirs(start) else {
        return Vec::new();
    };
    let mut out = Vec::new();
    let head_path = git_dir.join("HEAD");
    push_unique_existing(&mut out, head_path.clone());
    let Ok(head) = std::fs::read_to_string(&head_path) else {
        return out;
    };
    if let Some(ref_rel) = head.trim().strip_prefix("ref:") {
        let ref_rel = ref_rel.trim();
        let worktree_ref = git_dir.join(ref_rel);
        let common_ref = common_dir.join(ref_rel);
        if worktree_ref.exists() {
            push_unique_existing(&mut out, worktree_ref);
        } else if common_ref.exists() {
            push_unique_existing(&mut out, common_ref);
        } else if let Some(dir) = common_ref.ancestors().skip(1).find(|p| p.exists()) {
            // Watch the nearest existing ref directory so creating the loose ref
            // on the next commit triggers a rerun without watching a missing path.
            push_unique_existing(&mut out, dir.to_path_buf());
        }
        push_unique_existing(&mut out, common_dir.join("packed-refs"));
    }
    out
}

/// Resolves the short commit hash directly from `.git` files (`HEAD`, loose ref,
/// or `packed-refs`), used when `git` is unavailable or in unit tests.
pub fn commit_from_files(start: &Path) -> Option<String> {
    let (git_dir, common_dir) = git_dirs(start)?;
    let head = std::fs::read_to_string(git_dir.join("HEAD")).ok()?;
    let trimmed = head.trim();
    let full = if let Some(ref_rel) = trimmed.strip_prefix("ref:") {
        let ref_rel = ref_rel.trim();
        read_hex(&git_dir.join(ref_rel))
            .or_else(|| read_hex(&common_dir.join(ref_rel)))
            .or_else(|| packed_ref(&common_dir.join("packed-refs"), ref_rel))?
    } else if is_hex_oid(trimmed) {
        trimmed.to_owned()
    } else {
        return None;
    };
    Some(full[..full.len().min(7)].to_ascii_lowercase())
}

fn read_hex(path: &Path) -> Option<String> {
    let s = std::fs::read_to_string(path).ok()?;
    let trimmed = s.trim();
    is_hex_oid(trimmed).then(|| trimmed.to_owned())
}

fn packed_ref(packed_refs: &Path, target_ref: &str) -> Option<String> {
    let content = std::fs::read_to_string(packed_refs).ok()?;
    for line in content.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') || line.starts_with('^') {
            continue;
        }
        let mut parts = line.split_whitespace();
        let (Some(oid), Some(name)) = (parts.next(), parts.next()) else {
            continue;
        };
        if name == target_ref && is_hex_oid(oid) {
            return Some(oid.to_owned());
        }
    }
    None
}

fn is_hex_oid(s: &str) -> bool {
    s.len() >= 7 && s.bytes().all(|b| b.is_ascii_hexdigit())
}

/// Short commit hash for `VICTORY_GIT_COMMIT`: asks `git rev-parse --short HEAD`
/// first, then falls back to `.git` files, else `"unknown"`.
pub fn short_commit(start: &Path) -> String {
    if let Ok(out) = std::process::Command::new("git")
        .args(["rev-parse", "--short", "HEAD"])
        .current_dir(start)
        .output()
        && out.status.success()
    {
        let s = String::from_utf8_lossy(&out.stdout);
        let trimmed = s.trim();
        if is_hex_oid(trimmed) {
            return trimmed.to_ascii_lowercase();
        }
    }
    commit_from_files(start).unwrap_or_else(|| "unknown".to_owned())
}
