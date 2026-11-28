#!/bin/bash
# native-binary.sh's find_rustup: rustup on PATH, rustup only in CARGO_HOME/bin or
# Homebrew's keg-only opt dir (PATH gains the folder, rustup_found names it), and no
# rustup anywhere (returns 1). Run: /bin/bash tools/macos/tests/rustup-path.sh
set -euo pipefail
source "$(cd "$(dirname "$0")/.." && pwd)/native-binary.sh"
root=$(mktemp -d "${TMPDIR:-/tmp}/victory rustup.XXXXXX")
saved_path="$PATH"
cleanup() { export PATH="$saved_path"; rm -rf "$root"; }
trap cleanup EXIT

mkdir -p "$root/empty"
# A stub rustup that does nothing; only its executability matters.
stub() { mkdir -p "$1"; printf '#!/bin/sh\n' > "$1/rustup"; chmod +x "$1/rustup"; }

# Already on PATH: found, nothing added.
stub "$root/on/bin"
PATH="$root/on/bin:$root/empty:/usr/bin:/bin" CARGO_HOME="$root/none" HOMEBREW_PREFIX="$root/none"
if ! find_rustup || [ -n "$rustup_found" ]; then
  echo "FAIL: on PATH: rustup_found='$rustup_found'" >&2
  exit 1
fi

# Only in CARGO_HOME/bin: PATH gains it, rustup_found names the folder.
stub "$root/cargo/bin"
PATH="$root/empty:/usr/bin:/bin" CARGO_HOME="$root/cargo" HOMEBREW_PREFIX="$root/none"
if ! find_rustup || [ "$rustup_found" != "$root/cargo/bin" ] \
  || [[ ":$PATH:" != *":$root/cargo/bin:"* ]]; then
  echo "FAIL: CARGO_HOME: rustup_found='$rustup_found' PATH=$PATH" >&2
  exit 1
fi

# Only in Homebrew's keg-only rustup formula: same via HOMEBREW_PREFIX/opt/rustup/bin.
stub "$root/hb/opt/rustup/bin"
PATH="$root/empty:/usr/bin:/bin" CARGO_HOME="$root/none" HOMEBREW_PREFIX="$root/hb"
if ! find_rustup || [ "$rustup_found" != "$root/hb/opt/rustup/bin" ] \
  || [[ ":$PATH:" != *":$root/hb/opt/rustup/bin:"* ]]; then
  echo "FAIL: HOMEBREW_PREFIX: rustup_found='$rustup_found' PATH=$PATH" >&2
  exit 1
fi

# Nowhere: returns 1 and leaves rustup_found empty.
PATH="$root/empty:/usr/bin:/bin" CARGO_HOME="$root/none" HOMEBREW_PREFIX="$root/none"
if find_rustup; then
  echo "FAIL: no rustup anywhere, but find_rustup returned 0" >&2
  exit 1
fi
if [ -n "$rustup_found" ]; then
  echo "FAIL: rustup_found='$rustup_found' after a failed find" >&2
  exit 1
fi
echo "rustup path: ok"
