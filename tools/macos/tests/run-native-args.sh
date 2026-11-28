#!/bin/bash
# run-native.sh's argument checks and its missing-game-file paths, without a game and without
# building: every case here stops before the game would be launched.
# Run: /bin/bash tools/macos/tests/run-native-args.sh
set -uo pipefail
script="$(cd "$(dirname "$0")/.." && pwd)/run-native.sh"
dir=$(mktemp -d "${TMPDIR:-/tmp}/run-native-args.XXXXXX")
trap 'rm -rf "$dir"' EXIT
out=""
rc=0
ok=1

# Runs the script, keeping its output and status.
try() {
  out=$("$script" "$@" 2>&1)
  rc=$?
}

fail() {
  echo "FAIL: $*" >&2
  printf 'exit %d, output:\n%s\n' "$rc" "$out" >&2
  ok=0
}

# A path that does not exist: the message names it as given, and the run stops there.
not_a_file() {
  try "$1"
  [ "$rc" = 2 ] || fail "$*: exit $rc, expected 2"
  case "$out" in *"Not a file: $1"*) ;; *) fail "$*: expected \"Not a file: $1\"" ;; esac
}

# Usage is printed (and nothing runs) when the arguments do not name exactly one target.
usage() {
  try "$@"
  [ "$rc" = 2 ] || fail "$*: exit $rc, expected 2"
  case "$out" in *"Usage: tools/macos/run-native.sh"*) ;; *) fail "$*: expected the usage line" ;; esac
}

# One message, one status: the argument checks refuse before anything is launched.
refused() {
  local want="$1"
  shift
  try "$@"
  [ "$rc" = 2 ] || fail "$*: exit $rc, expected 2"
  case "$out" in *"$want"*) ;; *) fail "$*: expected \"$want\"" ;; esac
}

# --help is not an error, and it prints the switches.
try --help
[ "$rc" = 0 ] || fail "--help: exit $rc, expected 0"
case "$out" in *"--headless"*) ;; *) fail "--help: expected the switches" ;; esac

usage
usage "$dir" "$dir"
refused "--seconds requires a non-negative decimal integer" --seconds
refused "--seconds requires a non-negative decimal integer" --seconds abc "$dir"
refused "--seconds requires a non-negative decimal integer" "--seconds=-1" "$dir"
refused "--seconds exceeds the timer range" --seconds 99999999999999999999 "$dir"
refused "--console requires a model" --console
refused "--console requires a model" "--console=-h" "$dir"
not_a_file "$dir/not-here.exe"
not_a_file "not-here-either.exe"

# A folder that exists but holds no game_*master.exe: the partial download and the wrong folder
# both land here, and the message says which folder was searched.
try "$dir"
[ "$rc" = 2 ] || fail "empty folder: exit $rc, expected 2"
case "$out" in *"No game_*master.exe found in $dir"*) ;; *) fail "empty folder: expected \"No game_*master.exe found in $dir\"" ;; esac

# Nor does a file of another name count, even in the right folder.
: > "$dir/game_scarlett_master.txt"
try "$dir"
[ "$rc" = 2 ] || fail "wrong file name: exit $rc, expected 2"
case "$out" in *"No game_*master.exe found in $dir"*) ;; *) fail "wrong file name: expected the folder message" ;; esac

[ "$ok" = 1 ] || exit 1
echo "run-native arguments: ok"
