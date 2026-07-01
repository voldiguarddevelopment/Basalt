#!/usr/bin/env bash
#
# run_diff.sh — differential harness: the x86-64 oracle runs every kernel and records
# outputs; every other backend runs the same kernel and its output is diffed against the
# oracle (integers bit-exact, floats within a stated ULP tolerance). For now the oracle is
# the only backend that exists, so this harness's job is: run the oracle on every registered
# kernel via the real `basalt --cpu` path, link the result against its host driver, execute
# it, and compare the outcome against a stored golden. A future backend adds its own compare
# lane per kernel here rather than replacing this one.
#
# Kernel/driver pairs are listed in KERNELS below as "kernel.cu:driver.c" — add a line there
# to bring a new kernel into the harness. A kernel with no entry here (e.g.
# deliberate_errors.cu, which is a sema-error fixture and was never meant to run) is simply
# not exercised; nothing needs to special-case it.
#
# Golden files live in tests/diff/golden/<name>.txt, one per kernel, holding the driver's
# exit code and stdout from the last known-good run. A kernel with no golden yet gets one
# written from its current (just-verified) run; a kernel with an existing golden is compared
# against it and this script fails loudly on any mismatch.

set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
golden_dir="$root/tests/diff/golden"
kernel_dir="$root/tests/kernels"

# "kernel.cu:driver.c" — driver.c is relative to the repo root.
KERNELS=(
  "vector_add.cu:examples/cpu_launch_vadd.c"
)

if [ "${#KERNELS[@]}" -eq 0 ]; then
  echo "run_diff: no kernels registered — nothing to diff"
  exit 0
fi

if ! command -v cc >/dev/null 2>&1; then
  echo "run_diff: 'cc' not found, cannot link oracle output — skipping" >&2
  exit 0
fi

cargo build --locked --quiet --bin basalt
basalt="$root/target/debug/basalt"

tmpdir="$(mktemp -d)"
trap 'rm -rf "$tmpdir"' EXIT

fail=0

for pair in "${KERNELS[@]}"; do
  kernel="${pair%%:*}"
  driver="${pair#*:}"
  name="${kernel%.cu}"

  kernel_path="$kernel_dir/$kernel"
  driver_path="$root/$driver"
  obj="$tmpdir/$name.o"
  shim_o="$tmpdir/$name-shim.o"
  exe="$tmpdir/$name-exe"

  echo "run_diff: $name"

  if ! "$basalt" --cpu "$kernel_path" -o "$obj" 2>"$tmpdir/$name.stderr"; then
    echo "  FAIL: basalt --cpu $kernel did not exit 0:" >&2
    sed 's/^/    /' "$tmpdir/$name.stderr" >&2
    fail=1
    continue
  fi

  if ! cc -c "$driver_path" -o "$shim_o" 2>"$tmpdir/$name.cc1.log"; then
    echo "  FAIL: compiling $driver failed:" >&2
    sed 's/^/    /' "$tmpdir/$name.cc1.log" >&2
    fail=1
    continue
  fi

  if ! cc "$shim_o" "$obj" -o "$exe" 2>"$tmpdir/$name.cc2.log"; then
    echo "  FAIL: linking $name failed:" >&2
    sed 's/^/    /' "$tmpdir/$name.cc2.log" >&2
    fail=1
    continue
  fi

  set +e
  stdout="$("$exe")"
  code=$?
  set -e
  actual="exit=$code
$stdout"

  golden="$golden_dir/$name.txt"
  if [ ! -f "$golden" ]; then
    printf '%s\n' "$actual" >"$golden"
    echo "  stored golden: $golden"
    continue
  fi

  expected="$(cat "$golden")"
  if [ "$expected" != "$actual" ]; then
    echo "  FAIL: $name does not match its golden" >&2
    diff <(printf '%s\n' "$expected") <(printf '%s\n' "$actual") >&2 || true
    fail=1
    continue
  fi

  echo "  matched golden: $golden"
done

if [ "$fail" -ne 0 ]; then
  echo "run_diff: FAIL — see above" >&2
  exit 1
fi

echo "run_diff: all kernels linked, ran, and matched their goldens"
