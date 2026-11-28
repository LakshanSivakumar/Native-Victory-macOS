# Shared by native-setup.sh and run-native.sh, from the workspace root.

# Puts rustup on PATH when it is installed but not on PATH yet. That happens
# right after the rustup.rs installer (until a new terminal opens) and with
# Homebrew's rustup formula, which is keg-only. Sets rustup_found to the
# folder it added, or leaves it empty when rustup was already on PATH.
# Returns 1 when there is no rustup.
find_rustup() {
  rustup_found=""
  command -v rustup >/dev/null 2>&1 && return 0
  local dir
  for dir in "${CARGO_HOME:-$HOME/.cargo}/bin" "${HOMEBREW_PREFIX:-/opt/homebrew}/opt/rustup/bin"; do
    if [[ -x "$dir/rustup" ]]; then
      export PATH="$dir:$PATH"
      rustup_found="$dir"
      return 0
    fi
  done
  return 1
}

# The console model run-native.sh reports when --console is not given, from the
# Mac's RAM in whole GiB. Series X declares 13.5 GiB of title memory: more than
# Metal's working set on 16 GB Macs (#8), and it can still run out of GPU
# memory at the world load on 24 GB (#79, #110). Series S declares 7.6 GiB.
default_console() {
  if (( $1 <= 24 )); then echo series-s; else echo series-x; fi
}

# Cargo resolves environment overrides and local/global build.target-dir config.
# plutil is part of macOS and decodes JSON strings without losing path quoting.
native_binary() {
  local target_dir
  if ! command -v cargo >/dev/null 2>&1 && ! find_rustup; then
    echo "Cargo is not on your PATH. Install rustup from https://rustup.rs, then run tools/macos/native-setup.sh." >&2
    return 1
  fi
  if ! target_dir=$(set -o pipefail
    cargo metadata --format-version 1 --no-deps --offline |
      /usr/bin/plutil -extract target_directory raw -expect string -o - -
  ); then
    echo "Cannot resolve Cargo's target directory. Check the Cargo error above and your Cargo configuration/toolchain." >&2
    return 1
  fi
  if [[ -z "$target_dir" || "$target_dir" != /* ]]; then
    echo "Cargo metadata did not provide an absolute target directory." >&2
    return 1
  fi
  printf '%s/x86_64-apple-darwin/release/victory\n' "$target_dir"
}

# Wraps the built host executable in a minimal .app bundle beside it so macOS
# sees an Info.plist with LSApplicationCategoryType=public.app-category.games
# and LSSupportsGameMode=true, allowing Game Mode to engage in full screen (#533).
# Prints the bundled executable path.
stage_native_bundle() {
  local bin="$1"
  local bundle="$(dirname "$bin")/Victory.app"
  local contents="$bundle/Contents"
  local macos="$contents/MacOS"
  local app_bin="$macos/victory"
  local plist="$contents/Info.plist"
  local changed=0
  mkdir -p "$macos" || return 1
  cat > "$plist.tmp" <<'PLIST' || return 1
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleDevelopmentRegion</key>
  <string>en</string>
  <key>CFBundleDisplayName</key>
  <string>ScarlettMac Native</string>
  <key>CFBundleExecutable</key>
  <string>victory</string>
  <key>CFBundleIdentifier</key>
  <string>dev.victory.native</string>
  <key>CFBundleInfoDictionaryVersion</key>
  <string>6.0</string>
  <key>CFBundleName</key>
  <string>ScarlettMac Native</string>
  <key>CFBundlePackageType</key>
  <string>APPL</string>
  <key>CFBundleShortVersionString</key>
  <string>0.1.0</string>
  <key>CFBundleVersion</key>
  <string>1</string>
  <key>LSApplicationCategoryType</key>
  <string>public.app-category.games</string>
  <key>LSSupportsGameMode</key>
  <true/>
  <key>NSHighResolutionCapable</key>
  <true/>
</dict>
</plist>
PLIST
  if [[ ! -f "$plist" ]] || ! cmp -s "$plist.tmp" "$plist"; then
    mv -f "$plist.tmp" "$plist" || return 1
    changed=1
  else
    rm -f "$plist.tmp"
  fi
  if [[ ! -x "$app_bin" ]] || ! cmp -s "$bin" "$app_bin"; then
    # Remove before copying so Apple Silicon never sees an in-place Mach-O
    # overwrite against a cached vnode code signature.
    rm -f "$app_bin"
    cp "$bin" "$app_bin" || return 1
    chmod +x "$app_bin" || return 1
    changed=1
  fi
  if (( changed )) && [[ -x /usr/bin/codesign ]]; then
    /usr/bin/codesign --force --sign - "$bundle" >/dev/null 2>&1 || true
  fi
  printf '%s\n' "$app_bin"
}
