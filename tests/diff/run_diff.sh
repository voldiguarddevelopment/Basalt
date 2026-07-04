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
#
# TRITON_KERNELS below is a second, separate array/loop rather than more "kernel.cu:driver.c"
# lines in KERNELS: a Triton kernel is parsed via a different frontend entirely
# (`basalt_frontend_triton::parse`, selected by the `--triton` modifier — see `basalt-cli`'s
# own module header), so every `$basalt` invocation in this second loop carries that extra
# flag alongside `--cpu`/`--cpu-regalloc`; the golden/cross-backend comparison logic itself is
# unchanged from the main loop above.

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

  # MLIR/NVPTX lane, vector_add only: `basalt-mlir`'s own gpu/arith/memref/cf-through-mlir-opt
  # NVPTX path (crates/basalt-mlir/src/emit.rs) is a second, independent way to reach real PTX
  # for this exact kernel, JIT-loaded and launched on real hardware via basalt-runtime's CUDA
  # driver loader (crates/basalt-mlir/tests/nvptx_gpu_proof.rs), exactly as basalt-ptx's own
  # hand-rolled path already is in ptx_gpu_proof.rs. Unlike stress's single-float comparison
  # below, vector_add's own oracle check (examples/cpu_launch_vadd.c, captured above in
  # $actual) compares 1024 elements one at a time, which is not practical to re-diff at the
  # shell level here — nvptx_gpu_proof.rs instead re-derives the identical reference (the same
  # a[i]=i / b[i]=i*2 generator and the same a[i]+b[i] equality, bit-exact, no ULP tolerance)
  # and checks it itself; this lane's job is only to confirm that check actually ran and
  # passed, or cleanly self-skipped, never to fail the default run when the toolchain or
  # hardware is absent. This lane needs `mlir-opt` on PATH (installed only alongside a real
  # LLVM/MLIR 22.1.6 toolchain build) plus `--features mlir` buildable plus a working CUDA
  # driver and device; any of those missing is a clean skip. Note: an earlier draft of this
  # lane assumed `mlir-translate`/`llc` would also be needed, mirroring basalt-llvm's own
  # TargetMachine finish; `emit.rs`'s own module header documents why that assumption did not
  # hold on the real toolchain (`-gpu-module-to-binary` reaches the identical NVPTX backend
  # from inside `mlir-opt` itself), so only `mlir-opt`'s presence is checked below.
  if [ "$name" = "vector_add" ]; then
    if ! command -v mlir-opt >/dev/null 2>&1; then
      echo "  skip: mlir-nvptx (no mlir-opt — --features mlir's toolchain is not installed here)"
    elif ! cargo build --locked --quiet --features mlir -p basalt-mlir 2>"$tmpdir/$name.mlirbuild.log"; then
      echo "  skip: mlir-nvptx (--features mlir build failed)"
      sed 's/^/    /' "$tmpdir/$name.mlirbuild.log"
    else
      mlir_test_out="$tmpdir/$name.mlir-nvptx.out"
      if cargo test --locked --quiet --features mlir -p basalt-mlir --test nvptx_gpu_proof \
        -- --nocapture >"$mlir_test_out" 2>&1; then
        if grep -q "^PASS: vector_add via basalt-mlir's NVPTX lane" "$mlir_test_out"; then
          echo "  mlir-nvptx matches oracle: $name ($(grep '^PASS:' "$mlir_test_out"))"
        elif grep -q "skipping vector_add_via_mlir_nvptx_runs_on_real_hardware" "$mlir_test_out"; then
          skip_reason="$(grep -o 'skipping vector_add_via_mlir_nvptx_runs_on_real_hardware:.*' "$mlir_test_out" | head -1)"
          echo "  skip: mlir-nvptx ($skip_reason)"
        else
          echo "  FAIL: mlir-nvptx test exited 0 but produced neither a PASS nor a recognized skip line:" >&2
          sed 's/^/    /' "$mlir_test_out" >&2
          fail=1
        fi
      else
        echo "  FAIL: mlir-nvptx test (cargo test -p basalt-mlir --test nvptx_gpu_proof) failed:" >&2
        sed 's/^/    /' "$mlir_test_out" >&2
        fail=1
      fi
    fi
  fi

  # AMDGCN-via-emulator lanes, stress only: two independent backends can turn this kernel into
  # a real RDNA3 artifact — the LLVM backend's AMDGCN object-emission path (`--llvm
  # --amdgpu-bin`) and the hand-rolled `basalt-amdgpu` backend (plain `--amdgpu-bin`, the "no
  # LLVM" flagship) — and each gets its own lane below rather than one replacing the other.
  # Actually running either needs RDNA3 silicon (none of this project's machines have one) or
  # an instruction-level emulator — tests/diff/rdna3_sim/run_kernel.py drives tinygrad's
  # maintained one (DEV=MOCK+AMD) against the real HSACO bytes. The LLVM lane additionally
  # needs the `llvm` feature buildable; the hand-rolled lane needs nothing but tinygrad, since
  # `basalt-amdgpu` is always built. Both lanes skip (never fail the default run) when their
  # own prerequisite is missing.
  if [ "$name" = "stress" ]; then
    rdna3_harness="$root/tests/diff/rdna3_sim/run_kernel.py"
    rdna3_python="${RDNA3_SIM_PYTHON:-python3}"
    if ! command -v llvm-config-18 >/dev/null 2>&1; then
      echo "  skip: rdna3-sim (no llvm-config-18 — --features llvm cannot be built here)"
    elif ! command -v "$rdna3_python" >/dev/null 2>&1; then
      echo "  skip: rdna3-sim (no $rdna3_python — set RDNA3_SIM_PYTHON to an interpreter with tinygrad's mockgpu)"
    else
      export LLVM_SYS_180_PREFIX="${LLVM_SYS_180_PREFIX:-$(llvm-config-18 --prefix)}"
      if ! cargo build --locked --quiet --features llvm --bin basalt 2>"$tmpdir/$name.llvmbuild.log"; then
        echo "  skip: rdna3-sim (--features llvm build failed)"
        sed 's/^/    /' "$tmpdir/$name.llvmbuild.log"
      else
        llvm_obj="$tmpdir/$name.hsaco"
        if ! "$basalt" --llvm --amdgpu-bin "$kernel_path" -o "$llvm_obj" 2>"$tmpdir/$name.amdgpu.log"; then
          echo "  FAIL: basalt --llvm --amdgpu-bin $kernel did not exit 0:" >&2
          sed 's/^/    /' "$tmpdir/$name.amdgpu.log" >&2
          fail=1
        else
          # Same eighteen-temporary fold, same a[i] = (i+1)*0.5 - 3.0 generator as
          # examples/cpu_launch_stress.c, so the emulated run and the oracle's own live run
          # (captured above in $actual) are exercising identical inputs.
          expected_val="$(grep -oE '[0-9]+\.[0-9]+' <<<"$actual" | tail -1)"
          set +e
          rdna3_out="$("$rdna3_python" "$rdna3_harness" --hsaco "$llvm_obj" --kernel stress \
            --buf in:f32:-2.5,-2.0,-1.5,-1.0,-0.5,0.0,0.5,1.0,1.5,2.0,2.5,3.0,3.5,4.0,4.5,5.0,5.5,6.0,6.5,7.0 \
            --buf out:f32:1 --scalar i32:1 --global 1,1,1 --local 1,1,1 2>"$tmpdir/$name.rdna3.log")"
          rdna3_code=$?
          set -e
          if [ "$rdna3_code" -eq 77 ]; then
            echo "  skip: rdna3-sim ($(tail -1 "$tmpdir/$name.rdna3.log"))"
          elif [ "$rdna3_code" -ne 0 ]; then
            echo "  FAIL: rdna3-sim harness did not exit 0:" >&2
            sed 's/^/    /' "$tmpdir/$name.rdna3.log" >&2
            fail=1
          else
            rdna3_val="$(tail -1 <<<"$rdna3_out")"
            if awk -v a="$expected_val" -v b="$rdna3_val" 'BEGIN { d = a - b; if (d < 0) d = -d; exit !(d < 0.001) }'; then
              echo "  rdna3-sim matches oracle: $name ($rdna3_val)"
            else
              echo "  FAIL: $name diverges between the oracle ($expected_val) and rdna3-sim ($rdna3_val)" >&2
              fail=1
            fi
          fi
        fi
      fi
    fi

    # Hand-rolled AMDGCN-via-emulator lane, stress only: `basalt-amdgpu` (`--amdgpu-bin`, no
    # `--llvm`) needs no LLVM anywhere — this is the "no LLVM" flagship's own real proof, run
    # against the default (no-feature) `$basalt` already built above. Only a tinygrad checkout
    # is needed (no llvm-config-18 requirement, unlike the LLVM lane above), so this lane skips
    # (never fails the default run) only when that is missing.
    if ! command -v "$rdna3_python" >/dev/null 2>&1; then
      echo "  skip: rdna3-sim hand-rolled (no $rdna3_python — set RDNA3_SIM_PYTHON to an interpreter with tinygrad's mockgpu)"
    else
      handrolled_obj="$tmpdir/$name.handrolled.hsaco"
      if ! "$basalt" --amdgpu-bin "$kernel_path" -o "$handrolled_obj" 2>"$tmpdir/$name.amdgpu-hr.log"; then
        echo "  FAIL: basalt --amdgpu-bin $kernel did not exit 0:" >&2
        sed 's/^/    /' "$tmpdir/$name.amdgpu-hr.log" >&2
        fail=1
      else
        expected_val="$(grep -oE '[0-9]+\.[0-9]+' <<<"$actual" | tail -1)"
        set +e
        rdna3_out_hr="$("$rdna3_python" "$rdna3_harness" --hsaco "$handrolled_obj" --kernel stress \
          --buf in:f32:-2.5,-2.0,-1.5,-1.0,-0.5,0.0,0.5,1.0,1.5,2.0,2.5,3.0,3.5,4.0,4.5,5.0,5.5,6.0,6.5,7.0 \
          --buf out:f32:1 --scalar i32:1 --global 1,1,1 --local 1,1,1 2>"$tmpdir/$name.rdna3-hr.log")"
        rdna3_code_hr=$?
        set -e
        if [ "$rdna3_code_hr" -eq 77 ]; then
          echo "  skip: rdna3-sim hand-rolled ($(tail -1 "$tmpdir/$name.rdna3-hr.log"))"
        elif [ "$rdna3_code_hr" -ne 0 ]; then
          echo "  FAIL: rdna3-sim harness (hand-rolled) did not exit 0:" >&2
          sed 's/^/    /' "$tmpdir/$name.rdna3-hr.log" >&2
          fail=1
        else
          rdna3_val_hr="$(tail -1 <<<"$rdna3_out_hr")"
          if awk -v a="$expected_val" -v b="$rdna3_val_hr" 'BEGIN { d = a - b; if (d < 0) d = -d; exit !(d < 0.001) }'; then
            echo "  rdna3-sim hand-rolled matches oracle: $name ($rdna3_val_hr)"
          else
            echo "  FAIL: $name diverges between the oracle ($expected_val) and rdna3-sim hand-rolled ($rdna3_val_hr)" >&2
            fail=1
          fi
        fi
      fi
    fi
  fi
done

# "kernel.py:driver.c" — same pairing convention as KERNELS above, driver.c relative to the
# repo root. Routed through `--triton --cpu`/`--triton --cpu-regalloc` instead of the plain
# `--cpu`/`--cpu-regalloc` the main loop uses (see this file's header).
TRITON_KERNELS=(
  "tri_vadd.py:examples/cpu_launch_triton_vadd.c"
  "tri_matmul.py:examples/cpu_launch_triton_matmul.c"
)

for pair in "${TRITON_KERNELS[@]}"; do
  kernel="${pair%%:*}"
  driver="${pair#*:}"
  name="${kernel%.py}"

  kernel_path="$kernel_dir/$kernel"
  driver_path="$root/$driver"
  obj="$tmpdir/$name.o"
  shim_o="$tmpdir/$name-shim.o"
  exe="$tmpdir/$name-exe"

  echo "run_diff: $name (triton)"

  if ! "$basalt" --triton --cpu "$kernel_path" -o "$obj" 2>"$tmpdir/$name.stderr"; then
    echo "  FAIL: basalt --triton --cpu $kernel did not exit 0:" >&2
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
  # this exact run, not just the golden — same real cross-backend correctness check the main
  # loop runs, over BIR the Triton frontend produced instead of the CUDA-C one.
  obj_ra="$tmpdir/$name-ra.o"
  exe_ra="$tmpdir/$name-ra-exe"

  if ! "$basalt" --triton --cpu-regalloc "$kernel_path" -o "$obj_ra" 2>"$tmpdir/$name.ra.stderr"; then
    echo "  FAIL: basalt --triton --cpu-regalloc $kernel did not exit 0:" >&2
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

  # MLIR/NVPTX lane, tri_vadd only: basalt-mlir's own gpu/arith/memref/cf-through-mlir-opt
  # NVPTX path (crates/basalt-mlir/src/emit.rs) is a second, independent way to reach real PTX
  # for this exact Triton kernel, JIT-loaded and launched on real hardware via basalt-runtime's
  # CUDA driver loader (crates/basalt-mlir/tests/triton_nvptx_gpu_proof.rs), exactly as
  # basalt-ptx's own hand-rolled path already is in triton_ptx_gpu_proof.rs — mirrors the
  # vector_add-only mlir-nvptx lane above, keyed off the Triton kernel instead. This lane needs
  # `mlir-opt` on PATH plus `--features mlir` buildable plus a working CUDA driver and device;
  # any of those missing is a clean skip, never a failure of the default run.
  if [ "$name" = "tri_vadd" ]; then
    if ! command -v mlir-opt >/dev/null 2>&1; then
      echo "  skip: mlir-nvptx (no mlir-opt — --features mlir's toolchain is not installed here)"
    elif ! cargo build --locked --quiet --features mlir -p basalt-mlir 2>"$tmpdir/$name.mlirbuild.log"; then
      echo "  skip: mlir-nvptx (--features mlir build failed)"
      sed 's/^/    /' "$tmpdir/$name.mlirbuild.log"
    else
      mlir_test_out="$tmpdir/$name.mlir-nvptx.out"
      if cargo test --locked --quiet --features mlir -p basalt-mlir --test triton_nvptx_gpu_proof \
        -- --nocapture >"$mlir_test_out" 2>&1; then
        if grep -q "^PASS: masked triton vector_add via basalt-mlir's NVPTX lane" "$mlir_test_out"; then
          echo "  mlir-nvptx matches oracle: $name ($(grep '^PASS:' "$mlir_test_out"))"
        elif grep -q "skipping masked_triton_vector_add_via_mlir_nvptx_runs_on_real_gpu_hardware" "$mlir_test_out"; then
          skip_reason="$(grep -o 'skipping masked_triton_vector_add_via_mlir_nvptx_runs_on_real_gpu_hardware:.*' "$mlir_test_out" | head -1)"
          echo "  skip: mlir-nvptx ($skip_reason)"
        else
          echo "  FAIL: mlir-nvptx test exited 0 but produced neither a PASS nor a recognized skip line:" >&2
          sed 's/^/    /' "$mlir_test_out" >&2
          fail=1
        fi
      else
        echo "  FAIL: mlir-nvptx test (cargo test -p basalt-mlir --test triton_nvptx_gpu_proof) failed:" >&2
        sed 's/^/    /' "$mlir_test_out" >&2
        fail=1
      fi
    fi
  fi
done

if [ "$fail" -ne 0 ]; then
  echo "run_diff: FAIL — see above" >&2
  exit 1
fi

echo "run_diff: all kernels linked, ran, and matched their goldens"
