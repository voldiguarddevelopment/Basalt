#!/usr/bin/env bash
#
# run_diff.sh — differential harness: the x86-64 oracle runs every kernel and records
# outputs; every other backend runs the same kernel and its output is diffed against the
# oracle (integers bit-exact, floats within a stated ULP tolerance).
#
# Seed form: until the oracle backend lands, there are no goldens to diff.
# The harness discovers kernel/golden pairs under tests/diff/golden and exits 0 when there
# are none, so the gate is honestly green on the pre-backend tree. Landing the oracle fills
# in the oracle run + golden capture; later backends add their compare lanes here.

set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
golden_dir="$root/tests/diff/golden"

shopt -s nullglob
goldens=("$golden_dir"/*)
if [ "${#goldens[@]}" -eq 0 ]; then
  echo "run_diff: no goldens yet (oracle backend not landed) — nothing to diff"
  exit 0
fi

echo "run_diff: found ${#goldens[@]} golden(s) but no backend compare lanes are wired yet" >&2
echo "run_diff: this is a stub — the oracle run + diff are not implemented yet" >&2
exit 1
