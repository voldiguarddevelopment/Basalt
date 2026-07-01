# Basalt

A from-scratch, multi-target GPU/CPU **kernel compiler** written in Rust. It eats a
CUDA-C subset (and HIP) and Triton; it emits x86-64, RV32IM/RV64, NVIDIA PTX, AMD
RDNA/CDNA, SPIR-V, and Tenstorrent Metalium — with its **own** IR, its own instruction
encoders, and its own register allocator.

LLVM and MLIR are **optional** supporting backends, never a hard dependency. `cargo build`
with no features produces a working compiler that does its own instruction encoding — that
is the identity of the project.

## Design goals

1. **Correct-first, fast-later.** Every backend is validated against a trivially-correct
   x86-64 CPU oracle by differential testing before it is allowed to be clever.
2. **Hand-rolled core, no LLVM in the default build.** The default build encodes machine
   code itself. LLVM/MLIR are opt-in accelerators behind cargo features.
3. **Target-independent IR.** BIR knows nothing about any backend; adding a target is a new
   `Backend` implementation and one registration line.
4. **No silently-wrong codegen.** A backend that cannot lower an op refuses with a stable
   error code rather than emitting wrong output.
5. **Determinism.** Same IR in, byte-identical artifact out, on every backend.

## Architecture at a glance

```
 source ─▶ frontend ─▶ AST ─▶ sema ─▶ BIR ─▶ mid-end passes ─▶ BIR' ──▶ Backend ──▶ artifact
```

Everything above the `Backend` trait is shared: the CUDA-C and Triton frontends, semantic
analysis, BIR (a typed-SSA intermediate representation with a round-trippable textual form),
and target-independent optimization passes. Everything below it is a pluggable emitter.

See [ARCHITECTURE.md](ARCHITECTURE.md) for the full design.

## Backends

| Target | Emits | Notes |
|---|---|---|
| x86-64 (oracle) | SysV object | Stack-everything, correct-first — the differential-testing truth source |
| x86-64 (regalloc) | SysV object | SSA-based linear-scan; the CPU performance path |
| RV32IM / RV64 | ELF object | RISC-V with a soft-float runtime |
| NVIDIA PTX | PTX text | Text ISA, JIT via the CUDA Driver API — no LLVM needed |
| AMD RDNA/CDNA | HSACO | Hand-rolled instruction encoder; GFX10/11/12 + CDNA3 |
| SPIR-V | SPIR-V module | Vulkan / Level Zero compute |
| Tenstorrent Tensix | Metalium C++ | Dataflow tiles above BIR |

Optional feature-gated backends: `llvm` (BIR → LLVM IR), `mlir` (BIR → MLIR dialects),
`clif` (BIR → Cranelift IR). These are additive; the default build never requires them.

## Building

```
cargo build              # default, LLVM-free
cargo test               # core test suite
cargo build --features llvm    # opt-in LLVM lane
```

The BIR textual form round-trips (`parse(print(module)) == module`), which the test suite
enforces. You can pretty-print and validate a hand-written BIR module directly:

```
basalt --ir path/to/module.bir
```

## Status

Early. The compiler frontend, BIR, the diagnostics layer, and the driver are in place; the
CPU oracle and the hand-rolled backends are landing target by target, each validated against
the oracle before any optimization work on it begins.

## License

TBD.
