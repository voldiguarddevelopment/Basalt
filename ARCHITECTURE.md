# Basalt — Architecture

> A multi-architecture, multi-language kernel compiler in Rust. Eats CUDA-C (subset),
> HIP, and Triton; emits x86-64, RV32IM/RV64, NVIDIA PTX, AMD RDNA/CDNA, SPIR-V, and
> Tenstorrent Metalium — with its **own** IR, encoders, and register allocator.
> LLVM and MLIR are **optional** supporting backends, never a hard dependency.

*Working name — rename freely. Spiritual successor to Zane Hambly's BarraCUDA; not a fork.*

---

## 0. Design goals (in priority order)

1. **Correct-first, fast-later.** Every backend is validated against a trivially-correct
   CPU oracle by differential testing before it is allowed to be clever.
2. **Hand-rolled core, no LLVM in the default build.** `cargo build` with no features
   produces a working compiler that does its own instruction encoding. This is the
   identity of the project.
3. **LLVM/MLIR as opt-in accelerators.** Behind cargo features `llvm` / `mlir`. They buy
   you matrix intrinsics and mature codegen for the paths that are murder to hand-roll
   (AMDGCN, `wgmma`), but the project must build and pass its core suite without them.
4. **Target-independent IR.** BIR knows nothing about any backend. Adding a target is a
   new `Backend` impl, nothing else.
5. **Close the gaps BarraCUDA left open.** GPU matmul (MFMA/WMMA/`mma.sync`), a real CPU
   register allocator, host codegen, multi-TU linking, textures, dynamic parallelism.

## 1. The dual-identity principle (the crux)

There is exactly one abstraction boundary that makes the whole thing coherent: the
`Backend` trait. Everything above it (frontends, sema, BIR, mid-end passes) is shared.
Everything below is a pluggable emitter.

```
                     ┌─────────── shared, always compiled ───────────┐
 source ─▶ frontend ─▶ AST ─▶ sema ─▶ BIR ─▶ mid-end passes ─▶ BIR' ──┤
                     └───────────────────────────────────────────────┘
                                                                      │
                                        ┌─────── Backend trait ───────┘
                                        ▼
   ┌──────────────── hand-rolled (core identity, default) ───────────────┐
   │ x86-64 oracle │ x86-64 regalloc │ RV32IM/RV64 │ PTX │ AMDGCN │ SPIRV │
   └─────────────────────────────────────────────────────────────────────┘
   ┌──────────── optional supporting backends (feature-gated) ───────────┐
   │  llvm:  BIR → LLVM IR  → {NVPTX, AMDGCN, x86, …}   (feature = "llvm") │
   │  mlir:  BIR → dialects → Triton-faithful lowering  (feature = "mlir") │
   │  clif:  BIR → Cranelift IR → x86/aarch64/riscv     (feature = "clif") │
   └─────────────────────────────────────────────────────────────────────┘
```

**Consequence for BIR:** it must sit at roughly LLVM-IR abstraction level — typed SSA,
explicit control flow, GPU ops (thread/block indices, barriers, shared mem, warp
shuffles, **matrix ops as first-class**) — so that a single BIR module can lower *either*
to a hand-rolled encoder *or* to LLVM IR / an MLIR dialect without loss. If a hand-rolled
op has no clean LLVM/MLIR mapping, that's a BIR design smell, not a backend problem.

## 2. Workspace layout

```
basalt/
├── Cargo.toml                # workspace; features: llvm, mlir, clif, ptx, amdgpu, spirv, tensix
├── crates/
│   ├── basalt-cli/           # the `basalt` binary; flag parsing mirrors BarraCUDA UX
│   ├── basalt-diag/          # diagnostics: E-codes, --lang tables, ABEND/SNAP/SYSPRINT
│   ├── basalt-frontend-c/    # CUDA-C / HIP subset (C++-subset recursive descent)
│   ├── basalt-frontend-triton/  # @triton.jit Python → AST (ruff_python_parser)
│   ├── basalt-sema/          # type check, shape inference (Triton tiles), lowering to BIR
│   ├── basalt-bir/           # BIR types, textual printer + parser (round-trippable)
│   ├── basalt-passes/        # target-independent: SSA, constfold, DCE, LICM, divergence
│   ├── basalt-backend/       # the Backend trait + shared codegen utilities (ELF via `object`)
│   ├── basalt-x86/           # hand-rolled x86-64: oracle emitter + regalloc emitter
│   ├── basalt-rv/            # hand-rolled RV32IM / RV64IMFD + soft-float runtime
│   ├── basalt-ptx/           # hand-rolled NVIDIA PTX text emitter
│   ├── basalt-amdgpu/        # hand-rolled RDNA/CDNA binary encoder + HSACO writer
│   ├── basalt-spirv/         # SPIR-V via rspirv (Intel Arc / Vulkan compute)
│   ├── basalt-tensix/        # Tenstorrent Metalium C++ + TDF (Tile DataFlow) layer
│   ├── basalt-llvm/          # OPTIONAL (feature=llvm): BIR → LLVM IR via inkwell
│   ├── basalt-mlir/          # OPTIONAL (feature=mlir): BIR → dialects via melior
│   ├── basalt-clif/          # OPTIONAL (feature=clif): BIR → Cranelift IR
│   └── basalt-runtime/       # HSA loader (dlopen), CUDA Driver API loader, launch ABI
└── tests/
    ├── diff/                 # differential harness: oracle vs every backend
    └── kernels/              # vector_add.cu, sgemm, reductions, triton vadd, …
```

`basalt-runtime` deliberately `dlopen`s `libhsa-runtime64.so` and the CUDA driver — zero
compile-time dependency on ROCm/CUDA, same as BarraCUDA.

## 3. BIR — Basalt IR

Typed SSA, in memory as an arena of instructions per function; a textual form that
round-trips (`--ir` prints it, and the parser reads it back — this is a test invariant).

Core categories:

- **Scalar/vector ops** — arithmetic, compare, select, bitops, casts; vector types
  (`float2..4`, `int2..4`) as first-class, not lowered early.
- **Memory** — `load`/`store` with address spaces: `global`, `shared` (LDS), `constant`,
  `local`, `param`. Alignment + volatility carried on the op.
- **Control flow** — basic blocks, `br`/`condbr`, `switch`, structured `phi`. Reducible by
  construction where possible; irreducible handled by the SSA pass.
- **GPU intrinsics** — `tid.{x,y,z}`, `bid.*`, `bdim.*`, `gdim.*`, `barrier`, warp
  `shuffle{,_up,_down,_xor}`, `ballot`/`any`/`all`, `atomic.*`.
- **Matrix ops (first-class)** — `mma` op with (M,N,K), input/accumulator dtypes, and a
  layout attribute. Hand-rolled backends lower to MFMA/WMMA/`mma.sync`; the CPU oracle
  lowers to a triple loop; LLVM lowers to the matching intrinsic. **This is the single most
  important BIR decision** — matmul is an op, not a pattern to be re-recognized per backend.
- **Tile ops (Triton)** — rank-0/1/2 tiles with shape metadata, broadcast, `[:,None]`
  reshape, `tl.dot`, masked `tl.load`/`tl.store`. Lowered to scalar+loop for CPU, to
  vector/tile forms for GPU.

Metadata carried on the module: launch bounds, shared-mem bytes, target dtype set, source
locations (for diagnostics + `--ir` line mapping).

## 4. The `Backend` trait

```rust
pub trait Backend {
    /// Stable identifier used by --<name> flags and the diff harness.
    fn name(&self) -> &'static str;

    /// Does this backend claim to implement every op in the module?
    /// Missing matrix codegen, e.g., returns Unsupported(E099) instead of wrong code.
    fn supports(&self, module: &BirModule) -> Support;

    /// Lower a validated BIR module to an artifact (object bytes, PTX text, C++, …).
    fn emit(&self, module: &BirModule, opts: &EmitOpts) -> Result<Artifact, Diag>;
}
```

Invariants every backend must honor:

- **No silently-wrong code.** If an op is unimplemented, refuse with a stable E-code
  (BarraCUDA's E099 convention for "rank-2 tile on GPU without matrix path"). Refusal is a
  first-class, tested behavior.
- **Deterministic output.** Same BIR in → byte-identical artifact out. Non-negotiable for
  reproducible builds and for the diff harness.
- **Oracle-validatable.** The backend's results on the kernel suite must match the x86-64
  oracle bit-for-bit (integer) / within ULP tolerance (float).

The `llvm`/`mlir`/`clif` backends implement exactly this trait. They are not special —
they're just impls that happen to shell out to a library instead of an encoder table.

## 5. Backends

### Hand-rolled (core, default build)

| Backend | Crate | Emits | Notes |
|---|---|---|---|
| x86-64 oracle | `basalt-x86` | SysV object | Stack-everything, correct-first, **the truth source** |
| x86-64 regalloc | `basalt-x86` | SysV object | SSA-based linear-scan; the CPU perf path (milestone) |
| RV32IM / RV64 | `basalt-rv` | ELF object | RV32IM for Tensix baby cores; soft-float runtime wired in |
| NVIDIA PTX | `basalt-ptx` | PTX text | Text ISA — no LLVM needed; JIT via CUDA Driver API |
| AMD RDNA/CDNA | `basalt-amdgpu` | HSACO (ELF) | Hardest hand-rolled path; GFX10/11/12 + GFX942 |
| SPIR-V | `basalt-spirv` | SPIR-V module | via `rspirv`; Intel Arc (Level Zero) / Vulkan |
| Tensix Metalium | `basalt-tensix` | C++ + TDF | Dataflow: regions/channels/NoC arcs above BIR |

### Optional supporting (feature-gated, opt-in)

| Feature | Crate | Path | When to reach for it |
|---|---|---|---|
| `llvm` | `basalt-llvm` | BIR → LLVM IR (inkwell) → NVPTX/AMDGCN/x86 | Fastest route to correct matrix codegen; the AMDGCN path you don't want to hand-roll on day one |
| `mlir` | `basalt-mlir` | BIR → dialects (melior) | Triton-faithful lowering; reuse `gpu`/`nvgpu`/`amdgpu`/`vector`/`linalg` |
| `clif` | `basalt-clif` | BIR → Cranelift IR | Free regalloc CPU/aarch64/riscv without hand-rolling; useful as a *second* oracle |

Rule: CI runs the **default (no-feature) build** as the gate. Feature builds are additive
lanes. If `llvm` produces different results than the oracle, LLVM lowering is wrong (or
BIR semantics are underspecified) — the oracle wins.

## 6. Frontends

- **CUDA-C / HIP** — a **C++-subset** recursive-descent parser (`lang-c` gets you C11 but
  not templates/namespaces/operator overloading, which real CUDA needs). Scope matches
  BarraCUDA: structs/enums/typedefs/namespaces, basic template instantiation, operator
  overloading, full C control flow + preprocessor. HIP mode predefines `__HIPCC__` and
  platform macros, then shares the pipeline.
- **Triton** — parse `@triton.jit` functions with `ruff_python_parser` (fast, maintained;
  `rustpython-parser` is the fallback). Sema does tile shape inference: rank-0/1/2 shape on
  every expression, `constexpr` propagation (`BLOCK: tl.constexpr = 256` → `vec[256]`),
  numpy broadcasting, `[:,None]`/`[None,:]` reshape, `tl.dot` → BIR `mma`.

All three frontends converge on the same sema → BIR path.

## 7. Mid-end passes (shared, target-independent)

Dominators (Cooper–Harvey–Kennedy) · SSA construction & destruction (Braun–Hack spilling
lineage) · constant folding · DCE · LICM · **divergence analysis** (Sampaio–Souza–Collange–
Pereira) feeding a **divergence-aware register allocator** on the GPU backends. Occupancy
tuning driven by register pressure. Passes operate on BIR only and are backend-agnostic.

## 8. Differential testing — the safety net

The oracle (x86-64 stack-everything) runs every kernel and records outputs. Every other
backend — hand-rolled *and* LLVM/MLIR/Cranelift — runs the same kernel; the harness diffs.
A disagreement points at the non-oracle backend (or, occasionally, at underspecified BIR
semantics, which is itself a finding). `tests/diff/run_diff.sh` is genuine cross-backend
(x86 vs RISC-V vs PTX-under-driver), no GPU required for the CPU/RV lanes.

## 9. Diagnostics layer (the soul — kept from BarraCUDA)

Not a gimmick; the debugging affordances that make a from-scratch compiler survivable.

- **ABEND dumps** — GPU faults get IBM-style completion codes (G0Cx), faulting address
  correlated against tracked allocations, dispatch snapshot. Wired to the HSA event callback.
- **SNAP (`--snap`)** — per-kernel-parameter register values dumped to a host-visible buffer
  on entry. Read the evidence instead of the disassembly.
- **SYSPRINT** — class-tagged structured kernel output; host registers sinks by pattern
  (`STEP1.*`, `*.ERROR`, `*`); `drain` walks the buffer post-kernel.
- **Multilingual diagnostics** — language-neutral E-codes + `--lang <file>` tables. E-codes
  are the stable contract (tests assert on codes, not messages).

## 10. Non-negotiable invariants (for humans and agents)

1. Default build is LLVM-free and passes the core suite.
2. BIR round-trips: `parse(print(m)) == m`.
3. No backend emits wrong code for an unsupported op — it refuses with an E-code.
4. Every backend is bit/ULP-validated against the oracle before optimization work lands.
5. Backends are deterministic.
6. Adding a target touches only its own crate + a registration line — never BIR or sema.
