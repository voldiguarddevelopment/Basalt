#!/usr/bin/env bash
#
# run_diff.sh — differential harness: the x86-64 oracle runs every kernel and records
# outputs; every other backend runs the same kernel and its output is diffed against the
# oracle (integers bit-exact, floats within a stated ULP tolerance). This harness's job per
# kernel is two-fold: run the oracle via the real `basalt --cpu` path, link the result
# against its host driver, execute it, and compare the outcome against a stored golden; then
# run the same kernel through every other registered backend and compare its live output
# directly against the oracle's own live output from the same run, not just against the
# golden. Right now the x86-64 regalloc backend (`basalt --cpu-regalloc`) is the only other
# backend that exists, so it gets its own compare lane below; a future backend adds its own
# lane here rather than replacing this one.
#
# Kernel/driver pairs are listed in KERNELS below as "kernel.cu:driver.c" — add a line there
# to bring a new kernel into the harness. A kernel with no entry here (e.g.
# deliberate_errors.cu, which is a sema-error fixture and was never meant to run) is simply
# not exercised; nothing needs to special-case it.
#
# Golden files live in tests/diff/golden/<name>.txt, one per kernel, holding the oracle
# driver's exit code and stdout from the last known-good run. A kernel with no golden yet
# gets one written from its current (just-verified) oracle run; a kernel with an existing
# golden is compared against it and this script fails loudly on any mismatch. The
# regalloc-vs-oracle comparison is separate from the golden and runs every time, golden or
# not.

set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
golden_dir="$root/tests/diff/golden"
kernel_dir="$root/tests/kernels"

# "kernel.cu:driver.c" — driver.c is relative to the repo root.
KERNELS=(
  "vector_add.cu:examples/cpu_launch_vadd.c"
  "stress.cu:examples/cpu_launch_stress.c"
  "mymathhomework.cu:examples/cpu_launch_mymathhomework.c"
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
  else
    expected="$(cat "$golden")"
    if [ "$expected" != "$actual" ]; then
      echo "  FAIL: $name does not match its golden" >&2
      diff <(printf '%s\n' "$expected") <(printf '%s\n' "$actual") >&2 || true
      fail=1
      continue
    fi
    echo "  matched golden: $golden"
  fi

  # Cross-backend diff: the regalloc backend must reproduce the oracle's own live output for
  # this exact run, not just the golden — this is the real cross-backend correctness check.
  obj_ra="$tmpdir/$name-ra.o"
  exe_ra="$tmpdir/$name-ra-exe"

  if ! "$basalt" --cpu-regalloc "$kernel_path" -o "$obj_ra" 2>"$tmpdir/$name.ra.stderr"; then
    echo "  FAIL: basalt --cpu-regalloc $kernel did not exit 0:" >&2
    sed 's/^/    /' "$tmpdir/$name.ra.stderr" >&2
    fail=1
    continue
  fi

  if ! cc "$shim_o" "$obj_ra" -o "$exe_ra" 2>"$tmpdir/$name.cc3.log"; then
    echo "  FAIL: linking $name (regalloc) failed:" >&2
    sed 's/^/    /' "$tmpdir/$name.cc3.log" >&2
    fail=1
    continue
  fi

  set +e
  stdout_ra="$("$exe_ra")"
  code_ra=$?
  set -e
  actual_ra="exit=$code_ra
$stdout_ra"

  if [ "$actual" != "$actual_ra" ]; then
    echo "  FAIL: $name diverges between the oracle and regalloc backends" >&2
    echo "    oracle (live):" >&2
    sed 's/^/      /' <<<"$actual" >&2
    echo "    regalloc (live):" >&2
    sed 's/^/      /' <<<"$actual_ra" >&2
    fail=1
    continue
  fi

  echo "  oracle and regalloc agree: $name"
done

if [ "$fail" -ne 0 ]; then
  echo "run_diff: FAIL — see above" >&2
  exit 1
fi

echo "run_diff: all kernels linked, ran, and matched their goldens"
