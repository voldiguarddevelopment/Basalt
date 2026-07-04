#!/usr/bin/env bash
#
# run_kernel.sh — links a basalt-rv-emitted RV32IM object against its kernel's own existing
# host-side C driver (examples/cpu_launch_*.c — the same source the x86 oracle lane already
# runs, cross-compiled instead of host-compiled) plus this directory's bare-metal start.S/
# virt32.ld, boots the result under qemu-system-riscv32's "-M virt" board, and prints
# whatever the guest wrote to its semihosting console.
#
# Why qemu-system-riscv32 (a full machine emulator) and not qemu-riscv64 (user-mode Linux
# emulation): basalt-rv targets bare-metal RV32IM (`-march=rv32im -mabi=ilp32`, see
# crates/basalt-rv/src/lower.rs's own header) for control-core-class targets, not a Linux
# userspace ABI — and no RV32 (as opposed to RV64) glibc/dynamic-linking toolchain exists to
# feed qemu-riscv64's user-mode emulation in the first place. riscv64-elf-gcc is a bare-metal/
# newlib multilib cross-compiler (despite the "riscv64" name) that really does support an
# `rv32im/ilp32` target, matching this backend's own ABI exactly; qemu-system-riscv32 is the
# matching bare-metal machine simulator.
#
# Real, load-bearing limitation, not a bug in this script: 32-bit RISC-V/ARM semihosting's
# plain SYS_EXIT call (as opposed to the 64-bit-only SYS_EXIT_EXTENDED) carries a fixed
# "application exited" reason code, never the guest's actual `exit()`/`return` status — this
# was confirmed by disassembling libsemihost.a's own `_exit`, whose `a1` operand is a hardcoded
# constant, with the real status argument register never read at all. qemu-system-riscv32
# itself always exits 0 on that call regardless of what the guest returned. So this harness
# cannot use a numeric exit code as its pass/fail signal (unlike the x86 oracle lane) — every
# examples/cpu_launch_*.c driver already prints a "PASS: ..." / "FAIL: ..." line either way,
# so the caller compares that printed text against the oracle's own live stdout instead. A
# qemu hang (a real guest fault with no working console output at all) still surfaces as this
# script's own timeout-triggered failure below.
#
# Exit codes: 0 (guest console output printed to stdout) · 1 (a real failure: compile/link
# error, or the guest never returned control inside the timeout) · 77 (skip — the RV32
# bare-metal toolchain or qemu-system-riscv32 isn't installed here; a caller should treat this
# as "this lane doesn't run on this machine," not as a failure).

set -uo pipefail

SKIP=77

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
gcc_bin="${RV32_SIM_GCC:-riscv64-elf-gcc}"
qemu_bin="${RV32_SIM_QEMU:-qemu-system-riscv32}"
timeout_s="${RV32_SIM_TIMEOUT:-10}"

if [ "$#" -ne 2 ]; then
  echo "usage: run_kernel.sh <kernel.o> <driver.c>" >&2
  exit 1
fi
kernel_obj="$1"
driver_c="$2"

if ! command -v "$gcc_bin" >/dev/null 2>&1; then
  echo "$gcc_bin not found — RV32 bare-metal toolchain not installed here" >&2
  exit "$SKIP"
fi
if ! command -v "$qemu_bin" >/dev/null 2>&1; then
  echo "$qemu_bin not found — qemu-system-riscv32 not installed here" >&2
  exit "$SKIP"
fi
if ! "$gcc_bin" -march=rv32im -mabi=ilp32 -print-multi-lib 2>/dev/null | grep -q '^rv32im/ilp32;'; then
  echo "$gcc_bin has no rv32im/ilp32 multilib — bare-metal newlib not installed here" >&2
  exit "$SKIP"
fi

tmpdir="$(mktemp -d)"
trap 'rm -rf "$tmpdir"' EXIT

cflags=(-march=rv32im -mabi=ilp32)

if ! "$gcc_bin" "${cflags[@]}" -c "$here/start.S" -o "$tmpdir/start.o" 2>"$tmpdir/start.log"; then
  echo "rv32-sim: FAIL assembling start.S:" >&2
  sed 's/^/  /' "$tmpdir/start.log" >&2
  exit 1
fi

if ! "$gcc_bin" "${cflags[@]}" -c "$driver_c" -o "$tmpdir/driver.o" 2>"$tmpdir/driver.log"; then
  echo "rv32-sim: FAIL cross-compiling $driver_c:" >&2
  sed 's/^/  /' "$tmpdir/driver.log" >&2
  exit 1
fi

if ! "$gcc_bin" "${cflags[@]}" --specs=semihost.specs -nostartfiles -T "$here/virt32.ld" \
    "$tmpdir/start.o" "$tmpdir/driver.o" "$kernel_obj" -o "$tmpdir/out.elf" 2>"$tmpdir/link.log"; then
  echo "rv32-sim: FAIL linking bare-metal image:" >&2
  sed 's/^/  /' "$tmpdir/link.log" >&2
  exit 1
fi

set +e
guest_out="$(timeout "$timeout_s" "$qemu_bin" -M virt -bios none -nographic -semihosting \
  -kernel "$tmpdir/out.elf" 2>&1)"
qemu_rc=$?
set -e

if [ "$qemu_rc" -eq 124 ]; then
  echo "rv32-sim: FAIL — guest did not reach exit() within ${timeout_s}s (hang or trap):" >&2
  printf '%s\n' "$guest_out" | sed 's/^/  /' >&2
  exit 1
fi

printf '%s\n' "$guest_out"
