// BIR -> MLIR dialect lowering: `gpu`/`arith`/`memref`/`cf` cover a kernel's structure,
// index intrinsics, arithmetic, memory, and control flow; `vector`/`linalg` are named in
// this task's own scope but are not reached by the one kernel this lane bootstraps
// against (`tests/kernels/vector_add.cu`) — see "Scope" below for exactly why, and what a
// later task still owes. This builds a real, in-memory `melior::ir::Module` and prints it
// to MLIR's textual form; it does not lower further to any target (PTX/AMDGCN/SPIR-V/...,
// or even a `TargetMachine`-shaped in-memory form) — that is a separate, later task, the
// same way `basalt-llvm`'s `lower::lower_module` stops at an in-memory LLVM module and
// leaves `TargetMachine`-based object/assembly emission to `emit.rs`.
//
// # Why this is not a `Backend` impl
//
// `basalt_backend::Backend::emit` returns an `Artifact` — a final target payload (object
// bytes, PTX text, a SPIR-V module) meant to be run or assembled further by something
// outside this compiler. Dialect-level MLIR text is not that: it is connective tissue
// between BIR and a real target-specific lowering pipeline (`gpu-to-nvvm`, `gpu-to-rocdl`,
// ...) that a later task (P11-T2) owns, exactly the way `basalt-llvm`'s own
// `lower::lower_module` (BIR -> in-memory `inkwell::Module`) is a plain function, not a
// `Backend` impl — only `emit::LlvmAmdgcn`, which goes all the way to a `TargetMachine`
// and object bytes, implements the trait. This crate follows the identical split:
// `lower_module`/`lower_to_text` here are plain functions; a `Backend` impl (if one ever
// makes sense for this crate rather than for whichever target crate consumes its output)
// is P11-T2's decision to make, not this task's.
//
// # Text, not an in-memory handle, as the public surface
//
// `lower_module` builds a real `melior::ir::Module` (so `Operation::verify()` — real MLIR
// C-API verification, not a guess — can run inside this crate's own tests, no external
// process required) and `lower_to_text` is the convenience entry point that owns a fresh
// `Context`, calls `lower_module`, and prints the result. Textual MLIR is what P11-T2, an
// `mlir-opt`-based test, or a future CLI flag would all actually want to consume, mirroring
// `basalt-ptx`'s own "emit text, not a library handle" stance for a target with a stable
// textual form.
//
// # Dialect mapping — what maps where, and why
//
// - **`gpu`**: kernel/module structure and the thread/block index surface. A BIR `Module`
//   becomes one `gpu.module` (fixed `sym_name = "basalt"`, mirroring `basalt-llvm`'s own
//   fixed `create_module("basalt")`); each BIR `Function` becomes one `gpu.func ... kernel`
//   inside it. `TidX/Y/Z`, `BidX/Y/Z`, `BdimX/Y/Z`, `GdimX/Y/Z` become
//   `gpu.thread_id`/`gpu.block_id`/`gpu.block_dim`/`gpu.grid_dim`, each immediately narrowed
//   from MLIR's `index` result via `arith.index_cast` to whichever plain integer scalar type
//   the instruction's own BIR `Ty` declares (real Triton and MLIR GPU lowering pipelines do
//   the same narrowing at this exact boundary — `index` is a target-width-agnostic
//   abstraction the dialect uses on purpose). `vector_add.cu`'s own lowering always declares
//   these `i32` (matching the 32-bit register `basalt-ptx`/`basalt-llvm` already use for
//   them), but that is a choice this file's own CUDA-C-only bring-up made, not a BIR-wide
//   contract — found empirically proving out `tri_vadd.py`: `basalt-sema::triton_lower`
//   declares `Op::BidX` (`tl.program_id`) at `i64` instead (see that module's own header),
//   and every other backend that actually reads `inst.ty` generically here (`basalt-x86`'s
//   oracle `width_of(ty)`-driven `store_result`, plainly) already treats the width as
//   per-instruction, not fixed; a hardcoded `i32` cast was this lowering's own latent bug
//   once a second frontend disagreed, not a real BIR invariant. `Barrier` becomes
//   `gpu.barrier`. None of these five ops has a hand-written builder in
//   `melior` (`melior::dialect` only wraps `arith`/`cf`/`func`/`index`/`llvm`/`memref`/`scf`
//   by hand; `gpu` needs either its generic `OperationBuilder` path or the `ods-dialects`
//   feature, which pulls in `mlir-tblgen`/`bindgen` at build time for no benefit here) — this
//   lane uses the generic path throughout, the same mechanism those hand-written wrappers
//   use internally (confirmed by reading `melior`'s own real source on the one machine with
//   the matching toolchain, not assumed from memory). Every op name, attribute name, and
//   operand shape below was independently round-tripped through a real `mlir-opt` (LLVM/MLIR
//   22.1.6) before being written into this file, both in its pretty and its
//   `--mlir-print-op-generic` form, the same empirical-verification discipline
//   `basalt-amdgpu`'s `enc.rs` already modeled for a hand-decoded ISA.
// - **`arith`**: every scalar op BIR carries (`Bin`, `ICmp`/`FCmp`, `Select`, `Cast`,
//   `ConstInt`/`ConstFloat`) plus the `index_cast` connective tissue above. Not part of this
//   task's named five dialects, but unavoidable connective tissue the same way plain LLVM
//   instructions are unavoidable alongside `basalt-llvm`'s NVPTX/AMDGCN intrinsics — BIR's
//   scalar ops are not tile-shaped (`ARCHITECTURE §3`'s tile ops are a distinct, rank-0/1/2
//   construct this lane does not receive from `vector_add.cu`), so they map onto `arith`,
//   the dialect for plain scalar arithmetic, not `vector` (see below).
// - **`vector`**: deliberately **not used**. BIR's scalar ops (`Bin`/`ICmp`/`Select`/...)
//   operate on plain scalars here, never on a `vector<Nx...>`-shaped SSA value — `vector_add`
//   is a one-thread-one-element kernel; nothing in its compiled BIR is tile- or
//   vector-shaped. `ARCHITECTURE §3`'s "vector types (`float2..4`, `int2..4`) as first-class"
//   line refers to `Ty::Vec`, which this lane refuses outright (see below) since no kernel
//   in this task's bring-up path exercises it; when one does, `Ty::Vec` is the `vector`
//   dialect's real, natural target (`vector<Nxf32>` etc.), not `arith`.
// - **`memref`**: BIR's pointer model is flat and byte-addressed (`Ty::Ptr` carries no
//   pointee type; indexing is ordinary `Bin::Add`/`Mul` on an opaque address — see
//   `basalt-bir/src/ty.rs`'s own header), which has no representation under `memref`'s
//   element-indexed model at all (`memref.load`/`.store` take a per-dimension **element**
//   index, never a raw byte address). This lane recognizes the addressing shapes
//   `basalt-sema`'s indexed-access lowering emits (confirmed by reading
//   `basalt-sema/src/lower.rs`/`basalt-sema/src/triton_lower.rs` and by dumping this
//   project's own real post-`basalt_passes::optimize` BIR for both `vector_add.cu` and
//   `tests/kernels/tri_vadd.py` — see `lower::tests`) and reconstructs the original
//   **element** index from it, the same "recognize the shapes sema actually produces,
//   refuse everything else" discipline `basalt-spirv`'s `glcompute.rs` (`recognize_access`)
//   already established for the structurally identical problem (`Logical` addressing
//   there, `memref` here) — a fresh implementation here, not shared code, per backend
//   isolation. See `recognize_access` below for the exact shapes and what is refused.
//
//   A parameter's own `memref` element type is derived empirically, once, from every
//   recognized access that reads or writes through it (`analyze_memory_accesses`). Most
//   parameters settle on exactly one scalar type and get the simple, direct model: a real
//   `memref<Txsomething>` (`T` = that one scalar), indexed straight with `memref.load`/
//   `.store` — this is all `vector_add.cu` ever needs, and stays completely unchanged by
//   the byte-addressed fallback below (see `ParamElemModel::Typed`).
//
//   A parameter can genuinely be visited at more than one element type, though — a real,
//   load-bearing case found while proving out the Triton path, not a hypothetical:
//   `basalt-sema::triton_lower` materializes every tile a Triton kernel builds (`offsets`,
//   `mask`, `a`, `b`, ...) as a fixed byte range carved out of the kernel's own *last
//   pointer parameter* (see that module's own `Storage::Scratch` doc), which for
//   `tri_vadd.py` is `c_ptr` — the same parameter the kernel's real output also writes
//   through, at a genuinely different element type per tile (`i64` for `offsets`, `i1` for
//   `mask`, `f32` for the real output and the `a`/`b` tiles). A plain `memref<Txsomething>`
//   has no representation for that (`memref` is strongly typed: a `memref<i64>` cannot be
//   `memref.load`ed as `f32`), so such a parameter (`ParamElemModel::Bytes`) instead gets
//   modeled as `memref<?xi8>` — a dynamically-sized, byte-element memref — and every real
//   access through it reinterprets a single-element byte-offset slice of that buffer at the
//   real accessed type via `memref.view` (rank-1, statically-sized-1 result, dynamic byte
//   offset), then `memref.load`/`.store`s through *that* view. `memref.view`'s signature and
//   legality (source must be byte-typed with identity layout; result must be byte-offset,
//   not element-offset) were confirmed against a real, installed `mlir-opt` (LLVM/MLIR
//   22.1.6) before being written here, not assumed — see `byte_view` below. This is the
//   real answer to what an earlier pass over this file called "a byte-addressed escape
//   hatch defeating the entire point of choosing `memref`": it is not the default, general
//   model (a parameter with one consistent type never pays for it), only the fallback for
//   the one shape that genuinely needs it.
// - **`linalg`**: **not attempted.** `Op::Mma` is the natural `linalg.matmul` target (as
//   this task's own brief names), but `vector_add.cu` never emits one, and getting a real,
//   `mlir-opt`-verified `linalg.matmul` lowering right (operand layouts, accumulator
//   aliasing, `linalg`'s named-op operand-segment conventions) is its own real unit of work
//   this task's time budget does not reach. `Op::Mma` is refused with a stable E-code
//   (`E090`) rather than guessed at — a genuine deferral, not a silent gap; a real bonus
//   lane if one is ever added should hold itself to the same `mlir-opt`-round-tripped
//   standard as everything in this file.
// - **`nvgpu`/`amdgpu`**: **not needed and not touched.** Both are target-specific
//   intrinsic dialects (`nvgpu.mma.sync`, real AMDGPU-specific ops) that only matter once a
//   specific backend is being targeted — exactly P11-T2's job. `vector_add.cu` needs no
//   target-specific intrinsic at the dialect-emission stage, so this lane never references
//   either dialect.
//
// # Local/shared/constant/param storage: `memref.alloca`, not a refusal (P11-T3)
//
// `basalt-sema`'s lowering (both `lower.rs` for CUDA-C and `triton_lower.rs` for Triton) has
// no `alloca`: every local/parameter/shared/constant storage location is handed a small
// integer slot id and materialized, wherever BIR needs its address, as
// `const.i ptr.<space> (slot_id * SLOT_STRIDE)` — an opaque per-`(space, id)` key, not a real
// address (`basalt-x86`'s oracle and `basalt-llvm`'s own `build_local_slots` both document
// and handle the identical pattern). `build_local_slots` below builds one real, flat
// `memref.alloca` per distinct key actually used in a function, in that function's entry
// block ahead of its own instructions, so `Op::ConstInt`'s local-like-`Ty::Ptr` case has
// genuine backing storage to hand back instead of treating the opaque key as a literal
// index. Unlike `basalt-llvm`'s LLVM-side version (which allocates every slot uniformly as
// an 8-byte `i64`, since an opaque `alloca` can be loaded/stored at any bit-compatible type),
// `memref` is strongly typed — a `memref<i64>` cannot be `memref.load`ed as `f32` — so each
// slot's element type is derived empirically (`analyze_local_slots`): every `Load`/`Store`
// touching a given key must agree on one scalar type, mirroring `analyze_memory_accesses`'s
// identical derivation for a `Global` parameter's own element type. A slot found to hold a
// pointer value (`basalt-sema` can spill a previously-computed pointer through this same
// idiom) or used at more than one type is refused (`E092`) rather than guessed at — not
// exercised by either of this task's proof kernels (both promote every local/shared/
// constant/param slot to real SSA via `basalt_passes::construct_ssa` before this lowering
// ever sees them, so this pre-pass is inert for them in practice — see "Known gap" below for
// what its own real gate turned out to be — but the mechanism is real, tested on its own
// hand-built fixtures, and kept for parity with `basalt-llvm` and for whatever future kernel
// `construct_ssa`'s own safety checks decline to promote).
//
// # `Op::Phi`: a block argument, not a refusal (P11-T3)
//
// MLIR has no implicit-phi block instruction; `cf`'s own answer is a block argument on the
// merge block plus a matching operand on every predecessor's branch — real, existing
// mechanism, no new dialect needed. `basalt_passes::construct_ssa`'s own documented output
// order (a block's phis, if any, always come first, before its ordinary body — see that
// pass's header) is what makes this tractable without a topological fixup: `lower_function`
// gives every block one MLIR block argument per leading `Op::Phi` (in addition to the
// function's own parameters, for block 0), binds each phi's BIR value to that argument
// before lowering a single instruction, skips the phi instructions themselves during the
// main lowering walk (their value is already bound), and threads the right operand list
// through every `cf.br`/`cf.cond_br` by looking up, for the branch's own source block, which
// incoming value each target phi records for it (`branch_args`). A pointer- or vector-typed
// phi is refused (`E091`/`E092`) rather than guessed at — neither proof kernel's BIR ever
// produces one (every phi in both is `i64` or `f32`), so there is no real shape to confirm
// this lowering's block-argument mapping against yet.
//
// # Scope
//
// Implemented: kernel/module structure, `TidX/Y/Z`/`BidX/Y/Z`/`BdimX/Y/Z`/`GdimX/Y/Z`,
// `Barrier`, every `BinOp`, `ICmp`/`FCmp` (both predicate sets map onto `arith`'s own
// `Cmpi`/`CmpfPredicate` name-for-name), `Select`, every `CastOp`, `ConstInt`/`ConstFloat`
// (scalar, plus a local-like-`Ty::Ptr` slot constant — see above), `Load`/`Store` through a
// recognized `memref` addressing shape (either a `Global` parameter access or a local-like
// slot), scalar-typed `Op::Phi`, and `Br`/`CondBr`/`Ret(None)`.
//
// Refused with a stable E-code rather than guessed at: `Ty::Vec` (`E091`, the `vector`
// dialect's real target once a tile-shaped kernel bootstraps this lane further); a pointer
// value reaching `Load`/`Store` through any shape other than the ones `recognize_access`
// walks, including a bare pointer parameter used with no offset arithmetic at all (`E092`);
// a kernel parameter in a non-`Global` address space, or a `Global` pointer parameter never
// read or written through a recognized shape (`E092`, no element type to derive `memref`'s
// type from); a pointer-valued or multiply-typed local/shared/constant/param slot (`E092`,
// see above — unlike a `Global` parameter, a local-like slot has no byte-addressed fallback,
// since it is not exercised by either proof kernel and its own real address space is not a
// real buffer to begin with); a pointer- or vector-typed `Op::Phi` (`E091`/`E092`, see
// above); `Op::Switch` (`E090`, real and cheap to add via `melior::dialect::cf::switch` but
// not exercised by either proof kernel, so deliberately left for whenever a kernel actually
// needs it); `Op::Shuffle`/`Ballot`/`VoteAny`/`VoteAll`/`Atomic`/`AtomicCas` (`E093` — every
// one of these has a real mapping only once a target-specific dialect, `nvgpu`/`amdgpu`, is
// in the picture, exactly like `basalt-llvm`'s own per-dialect gaps for the harder
// warp-level ops; deferred to P11-T2, not guessed at here); `Op::Mma` (`E090`, see "linalg"
// above); a non-`Void`-returning top-level function (`E090` — every BIR function reaching
// this lowering is assumed a kernel entry point, the same assumption `basalt-llvm`'s own
// `amdgpu_kernel`-calling-convention code documents, and a real device function on a GPU
// target is out of scope until this project has one).
//
// # A `Global` parameter accessed at more than one element type (P11-T3c, `tri_vadd.py`)
//
// P11-T3's own brief expected the Triton gap to be local-slot storage and `Op::Phi`; both are
// real and are handled above (P11-T3a). What actually stopped `tri_vadd.py` short of a clean
// lowering, only visible once real Triton BIR was in hand, is different:
// `basalt-sema::triton_lower` carves every materialized tile's scratch storage out of the
// kernel's *last pointer parameter* (see that module's own `Storage::Scratch` doc) — for
// `tri_vadd.py` that parameter is `c_ptr`, which is *also* where the kernel's real output is
// written. The same `Ty::Ptr(Global)` parameter is therefore read/written at `i64` (the
// `offsets` tile), `i1` (`mask`), and `f32` (the `a`/`b` tiles *and* the real output) — three
// incompatible `memref` element types for one buffer, which a single `memref<Txsomething>`
// (`memref` is strongly typed) has no representation for at all. `analyze_memory_accesses`
// now recognizes this shape rather than refusing it (`ParamElemModel::Bytes`, see this file's
// `memref` section above), and `tri_vadd.py` lowers cleanly — see `lower::tests`'
// `tri_vadd_lowers_to_a_well_formed_module_via_the_real_pipeline`. This is not a
// `tri_vadd.py`-specific patch: `ParamElemModel` is decided per parameter, per module, purely
// from what `analyze_memory_accesses` actually observes, so it applies unchanged to any
// future kernel with the identical shape.
//
// `tri_matmul.py` (looked at, not this task's own bar — see `TASKS.md`) still refuses, on a
// distinct and more fundamental instance of the same underlying limit: it materializes
// `a_ptrs`/`b_ptrs`/`c_ptrs`/`out_ptrs` as tiles *of pointers* (`tl.dot`'s own operands are
// addressed through a pointer computed per element, not loaded data), so this lowering hits
// `Op::Store` of a `Ty::Ptr(Global)`-typed *value* — `E091`, "stored value must be a plain
// scalar" — a different problem (BIR has no `memref`-representable notion of "a buffer of
// addresses" at all) that a byte-addressed *element* reinterpretation does not touch.

use std::collections::{HashMap, HashSet};

use melior::dialect::arith::{self, CmpfPredicate, CmpiPredicate};
use melior::dialect::{cf, memref, DialectRegistry};
use melior::ir::attribute::{Attribute, FloatAttribute, IntegerAttribute, StringAttribute};
use melior::ir::operation::{OperationBuilder, OperationLike, OperationRef};
use melior::ir::r#type::{FunctionType, IntegerType, MemRefType};
use melior::ir::{
    Block, BlockLike, Identifier, Location, Module as MlirModule, Region, RegionLike, Type, Value,
};
use melior::utility::register_all_dialects;
use melior::Context;

use basalt_bir::{
    AddrSpace, BinOp, BlockId, CastOp, FCmpPred, Function, ICmpPred, Inst, InstId,
    Module as BirModule, Op, Scalar, Term, Ty, ValRef,
};
use basalt_diag::{Diag, ECode};

fn unsupported_op(detail: &'static str) -> Diag {
    Diag::new(ECode::UnsupportedOp).with_arg(detail)
}

fn unsupported_type(detail: &'static str) -> Diag {
    Diag::new(ECode::UnsupportedType).with_arg(detail)
}

fn unsupported_addr_space(detail: &'static str) -> Diag {
    Diag::new(ECode::UnsupportedAddressSpace).with_arg(detail)
}

fn unsupported_feature(detail: &'static str) -> Diag {
    Diag::new(ECode::UnsupportedFeature).with_arg(detail)
}

/// Lowers a BIR module to a real, in-memory MLIR module, then prints it to MLIR's own
/// textual form. Owns a fresh `Context` with every built-in dialect loaded (this lane only
/// ever emits `builtin`/`gpu`/`arith`/`memref`/`cf`, but loading the rest costs nothing and
/// keeps this entry point simple for callers that just want text).
pub fn lower_to_text(module: &BirModule) -> Result<String, Diag> {
    let context = Context::new();
    let registry = DialectRegistry::new();
    register_all_dialects(&registry);
    context.append_dialect_registry(&registry);
    context.load_all_available_dialects();

    let mlir_module = lower_module(module, &context)?;
    let op = mlir_module.as_operation();
    assert!(
        op.verify(),
        "basalt-mlir emitted a module MLIR's own verifier rejects: {op}"
    );
    Ok(op.to_string())
}

/// Lowers a BIR module into a real `melior::ir::Module` under a caller-owned `Context` (so a
/// caller that already has one, e.g. a later task driving further lowering passes, does not
/// pay for a second). Mirrors `basalt-llvm::lower::lower_module`'s own shape.
pub fn lower_module<'c>(module: &BirModule, context: &'c Context) -> Result<MlirModule<'c>, Diag> {
    let loc = Location::unknown(context);
    let mlir_module = MlirModule::new(loc);

    let gpu_module_block = Block::new(&[]);
    for f in &module.funcs {
        lower_function(context, &gpu_module_block, f)?;
    }
    let gpu_module_region = Region::new();
    gpu_module_region.append_block(gpu_module_block);

    let gpu_module_op = OperationBuilder::new("gpu.module", loc)
        .add_attributes(&[(
            Identifier::new(context, "sym_name"),
            StringAttribute::new(context, "basalt").into(),
        )])
        .add_regions([gpu_module_region])
        .build()
        .expect("valid gpu.module operation");
    mlir_module.body().append_operation(gpu_module_op);

    Ok(mlir_module)
}

fn scalar_byte_size(s: Scalar) -> i64 {
    match s {
        Scalar::I1 | Scalar::I8 => 1,
        Scalar::I16 | Scalar::F16 => 2,
        Scalar::I32 | Scalar::F32 => 4,
        Scalar::I64 | Scalar::F64 => 8,
    }
}

fn as_scalar(ty: Ty, detail: &'static str) -> Result<Scalar, Diag> {
    match ty {
        Ty::Scalar(s) => Ok(s),
        Ty::Vec(..) => Err(unsupported_type(detail)),
        Ty::Ptr(_) | Ty::Void => Err(unsupported_type(detail)),
    }
}

fn mlir_scalar_ty<'c>(context: &'c Context, s: Scalar) -> Type<'c> {
    match s {
        Scalar::I1 => IntegerType::new(context, 1).into(),
        Scalar::I8 => IntegerType::new(context, 8).into(),
        Scalar::I16 => IntegerType::new(context, 16).into(),
        Scalar::I32 => IntegerType::new(context, 32).into(),
        Scalar::I64 => IntegerType::new(context, 64).into(),
        Scalar::F16 => Type::float16(context),
        Scalar::F32 => Type::float32(context),
        Scalar::F64 => Type::float64(context),
    }
}

fn result_of<'c, 'a>(op: OperationRef<'c, 'a>) -> Value<'c, 'a> {
    op.result(0).expect("operation has a result").into()
}

const SHAPE_MISMATCH: &str =
    "load/store address is not one of the recognized element-index shapes this lowering walks";

/// Resolves the "base" operand of a recognized address to the kernel parameter it ultimately
/// reads/writes through. Two real shapes are known, both confirmed against this project's own
/// compiled BIR (`vector_add.cu`, and `tests/kernels/tri_vadd.py` via `basalt-sema`'s Triton
/// path):
///
///  - a bare `ValRef::Param(i)` — the whole address is `<param> + off` directly (every
///    `vector_add.cu`-shaped access, and a Triton kernel's own real-payload accesses, e.g.
///    `a_ptr + offsets[i]`);
///  - `add ptr.global <param>, <const>` — a real, once-computed pointer one fixed constant
///    byte offset into a parameter's own storage, reused as the base of further indexed
///    accesses. This is `basalt-sema::triton_lower`'s tile-scratch addressing (see that
///    module's own `Storage::Scratch` doc): every materialized Triton tile is a fixed byte
///    range carved out of the kernel's last pointer parameter, and each element access adds
///    an index-scaled offset on top of that fixed base.
///
/// Returns the resolved parameter index and, for the second shape, the constant byte offset
/// baked into the base (0 for the first shape) — the byte-addressed model (`byte_view`) needs
/// that constant added back in to recover the true byte offset from the parameter's own
/// origin, since it never gets to treat "param" and "param + tile's fixed byte range" as the
/// same address the way a real, direct `memref.load`/`.store` at the base's own resolved
/// parameter index implicitly does. Refuses (`E092`) at anything else.
fn resolve_base(f: &Function, base: ValRef) -> Result<(usize, i64), Diag> {
    match base {
        ValRef::Param(i) => Ok((i as usize, 0)),
        ValRef::Val(id) => {
            let inst = &f.insts[id.0 as usize];
            if inst.ty != Ty::Ptr(AddrSpace::Global) {
                return Err(unsupported_addr_space(SHAPE_MISMATCH));
            }
            match inst.op {
                Op::Bin(BinOp::Add, ValRef::Param(i), ValRef::Val(off_id)) => {
                    match f.insts[off_id.0 as usize].op {
                        Op::ConstInt(n) => Ok((i as usize, n)),
                        _ => Err(unsupported_addr_space(SHAPE_MISMATCH)),
                    }
                }
                _ => Err(unsupported_addr_space(SHAPE_MISMATCH)),
            }
        }
    }
}

/// Resolves the element index feeding a recognized address's `mul <index>, <stride>` offset.
/// Two real shapes are known:
///
///  - `sext i64 i32 <inner>` — CUDA-C's own shape (`vector_add.cu`): the index is computed at
///    `i32` and widened; this lowering casts the pre-widening `i32` value straight to `index`
///    rather than routing through the intermediate `i64` (either is a valid `arith.index_cast`
///    source, but the narrower value is what is semantically the "real" index);
///  - a bare `i64`-typed value with no `sext` at all — `basalt-sema::triton_lower`'s own shape:
///    that pass lowers every index computation natively at `i64` throughout (see that module's
///    own header), so there is never a narrower value to unwrap.
///
/// Refuses (`E092`) at anything else.
fn resolve_index(f: &Function, val: ValRef) -> Result<ValRef, Diag> {
    let id = match val {
        ValRef::Val(id) => id,
        ValRef::Param(_) => return Err(unsupported_addr_space(SHAPE_MISMATCH)),
    };
    let inst = &f.insts[id.0 as usize];
    if inst.ty != Ty::Scalar(Scalar::I64) {
        return Err(unsupported_addr_space(SHAPE_MISMATCH));
    }
    match inst.op {
        Op::Cast(CastOp::Sext, Ty::Scalar(Scalar::I32), inner) => Ok(inner),
        _ => Ok(val),
    }
}

/// One recognized `Load`/`Store` address, fully resolved. `elem_index` is what the direct,
/// single-type `memref` model (`ParamElemModel::Typed`) indexes with; `off_id`/
/// `base_extra_bytes` are what the byte-addressed fallback (`ParamElemModel::Bytes`,
/// `byte_view`) needs instead — the raw, already-in-bytes `Mul(index, esz)` node this shape
/// always computes, plus whatever constant byte offset was folded into the base pointer
/// itself (Triton's tile-scratch idiom; 0 for a bare parameter base). Every field but those
/// two mirrors what the pre-P11-T3c version of this function returned as a plain tuple.
struct RecognizedAccess {
    param_index: usize,
    elem_index: ValRef,
    off_id: InstId,
    base_extra_bytes: i64,
    add_id: InstId,
    base_id: Option<InstId>,
}

/// Walks a recognized pointer-arithmetic shape for an indexed memory access:
///
/// ```text
/// %off   = mul i64 <index>, <esz>      ; <esz> = the accessed scalar's byte size
/// %addr  = add ptr.global <base>, %off
/// ```
///
/// where `<base>` and `<index>` are each resolved by `resolve_base`/`resolve_index` above.
/// `add_id`/`base_id` (the inner "Param + const" base add's `InstId`, if `<base>` was that
/// shape rather than a bare parameter) exist purely so callers can mark those instructions as
/// consumed: neither is ever re-lowered as its own `memref`-incompatible pointer arithmetic
/// (see `analyze_memory_accesses`'s own use of them). Refuses (`E092`) at the first operand
/// that does not match, never guessing at what an unrecognized shape might mean — the same
/// discipline `basalt-spirv::glcompute`'s own `recognize_access` already established for the
/// identical "no raw address representation" problem under `Logical` addressing.
fn recognize_access(f: &Function, ptr: ValRef, elem_ty: Scalar) -> Result<RecognizedAccess, Diag> {
    let add_id = match ptr {
        ValRef::Val(id) => id,
        ValRef::Param(_) => return Err(unsupported_addr_space(SHAPE_MISMATCH)),
    };
    let add_inst = &f.insts[add_id.0 as usize];
    if add_inst.ty != Ty::Ptr(AddrSpace::Global) {
        return Err(unsupported_addr_space(SHAPE_MISMATCH));
    }
    let (base, off) = match add_inst.op {
        Op::Bin(BinOp::Add, base, off) => (base, off),
        _ => return Err(unsupported_addr_space(SHAPE_MISMATCH)),
    };
    let (param_index, base_extra_bytes) = resolve_base(f, base)?;
    let base_id = match base {
        ValRef::Val(id) => Some(id),
        ValRef::Param(_) => None,
    };
    let off_id = match off {
        ValRef::Val(id) => id,
        ValRef::Param(_) => return Err(unsupported_addr_space(SHAPE_MISMATCH)),
    };
    let (index_val, stride) = match f.insts[off_id.0 as usize].op {
        Op::Bin(BinOp::Mul, a, b) => (a, b),
        _ => return Err(unsupported_addr_space(SHAPE_MISMATCH)),
    };
    let stride_id = match stride {
        ValRef::Val(id) => id,
        ValRef::Param(_) => return Err(unsupported_addr_space(SHAPE_MISMATCH)),
    };
    let stride_n = match f.insts[stride_id.0 as usize].op {
        Op::ConstInt(n) => n,
        _ => return Err(unsupported_addr_space(SHAPE_MISMATCH)),
    };
    if stride_n != scalar_byte_size(elem_ty) {
        return Err(unsupported_addr_space(
            "load/store address's byte stride does not match the accessed type's size",
        ));
    }
    let elem_index = resolve_index(f, index_val)?;
    Ok(RecognizedAccess {
        param_index,
        elem_index,
        off_id,
        base_extra_bytes,
        add_id,
        base_id,
    })
}

/// How a `Ty::Ptr(Global)` parameter's `memref` is modeled, decided empirically by
/// `analyze_memory_accesses` from every recognized access that reads or writes through it.
/// `Typed` is the simple, direct case every kernel this lane bootstrapped against before
/// P11-T3c uses exclusively (`vector_add.cu` included) — a real `memref<Txsomething>`,
/// indexed straight. `Bytes` is the fallback for a parameter genuinely visited at more than
/// one element type (found proving out `tri_vadd.py` — see this file's own `memref` header
/// section) — a `memref<?xi8>`, with every real access going through `byte_view`'s
/// single-element `memref.view` reinterpretation instead of a direct `memref.load`/`.store`.
#[derive(Clone, Copy, PartialEq)]
enum ParamElemModel {
    Typed(Scalar),
    Bytes,
}

/// First pass over a function's `Global`-space instructions: derives each `Ty::Ptr(Global)`
/// parameter's `ParamElemModel` from the recognized accesses that use it (mirroring
/// `basalt-spirv::glcompute::resolve_binding_elem_tys`'s identical need, though that path
/// still refuses a multiply-typed binding outright — see this file's `memref` header section
/// for why `memref`'s `memref.view` gives this lane an escape hatch SPIR-V's `Logical`
/// addressing has no equivalent for), and collects the set of "outer add" instructions a
/// recognized access consumes so the second (codegen) pass skips them rather than attempting
/// arithmetic `memref` has no representation for. Local/shared/constant/param-space traffic is
/// a different storage model entirely (a synthesized per-slot key, not a real dereference) and
/// is left untouched here — see `analyze_local_slots`/`build_local_slots` below.
fn analyze_memory_accesses(
    f: &Function,
) -> Result<(Vec<Option<ParamElemModel>>, HashSet<u32>), Diag> {
    let mut model: Vec<Option<ParamElemModel>> = vec![None; f.params.len()];
    let mut skip = HashSet::new();

    let mut record = |ptr: ValRef, elem_ty: Scalar| -> Result<(), Diag> {
        let acc = recognize_access(f, ptr, elem_ty)?;
        model[acc.param_index] = Some(match model[acc.param_index] {
            None => ParamElemModel::Typed(elem_ty),
            Some(ParamElemModel::Typed(prev)) if prev == elem_ty => ParamElemModel::Typed(prev),
            // A second, disagreeing element type through the same parameter: no single
            // `memref<Txsomething>` can represent both, so this parameter falls back to the
            // byte-addressed model for every access, not just the one that triggered this.
            Some(ParamElemModel::Typed(_)) => ParamElemModel::Bytes,
            Some(ParamElemModel::Bytes) => ParamElemModel::Bytes,
        });
        skip.insert(acc.add_id.0);
        if let Some(id) = acc.base_id {
            skip.insert(id.0);
        }
        Ok(())
    };

    for inst in &f.insts {
        match &inst.op {
            Op::Load { ptr, space, .. } if *space == AddrSpace::Global => {
                let elem_ty = as_scalar(inst.ty, "loaded value must be a plain scalar")?;
                record(*ptr, elem_ty)?;
            }
            Op::Store { ptr, ty, space, .. } if *space == AddrSpace::Global => {
                let elem_ty = as_scalar(*ty, "stored value must be a plain scalar")?;
                record(*ptr, elem_ty)?;
            }
            _ => {}
        }
    }

    for (i, &ty) in f.params.iter().enumerate() {
        match ty {
            Ty::Ptr(AddrSpace::Global) if model[i].is_none() => {
                return Err(unsupported_addr_space(
                    "pointer parameter is never read or written through a recognized access, \
                     so no memref element type can be derived",
                ))
            }
            Ty::Ptr(AddrSpace::Global) => {}
            Ty::Ptr(_) => {
                return Err(unsupported_addr_space(
                    "kernel parameter in a non-global address space is out of scope",
                ))
            }
            _ => {}
        }
    }

    Ok((model, skip))
}

/// Tag `basalt-llvm`'s own `space_tag` also uses, for a `HashMap` key (`AddrSpace` has no
/// `Hash`/`Ord` impl of its own). `Global` is never a slot key — see `is_local_like`.
fn space_tag(space: AddrSpace) -> u8 {
    match space {
        AddrSpace::Global => 0,
        AddrSpace::Shared => 1,
        AddrSpace::Constant => 2,
        AddrSpace::Local => 3,
        AddrSpace::Param => 4,
    }
}

/// Whether `space` is one of `basalt-sema`'s synthesized address spaces, whose
/// `const.i ptr.<space> N` values are opaque per-function slot keys rather than real
/// addresses — see `build_local_slots`. `Global` pointers are real addresses from the moment
/// they arrive (a function parameter, or arithmetic on one) and never take this path.
fn is_local_like(space: AddrSpace) -> bool {
    matches!(
        space,
        AddrSpace::Local | AddrSpace::Param | AddrSpace::Shared | AddrSpace::Constant
    )
}

type SlotKey = (u8, i64);

/// Derives each local-like slot's element scalar type from every `Load`/`Store` that touches
/// it, the same "every use must agree" discipline `analyze_memory_accesses` already applies
/// per `Global` parameter. Unlike a `Global` pointer parameter (which BIR never lets hold
/// anything but a real dereferenceable address), `basalt-sema` can spill an arbitrary
/// previously-computed value — including a pointer — through this same slot idiom; a slot
/// found to carry a non-scalar value is refused (`E092`) rather than guessed at, since
/// `memref`'s element type must be a concrete scalar (or another shaped type this lowering
/// does not attempt) and there is no representation here for "whatever type happens to show
/// up."
fn analyze_local_slots(f: &Function) -> Result<HashMap<SlotKey, Scalar>, Diag> {
    let mut tys: HashMap<SlotKey, Scalar> = HashMap::new();

    let mut record = |space: AddrSpace, n: i64, ty: Ty| -> Result<(), Diag> {
        let elem_ty = as_scalar(
            ty,
            "local/shared/constant/param slot holds a non-scalar (pointer or vector) value, \
             which has no memref element-type representation in this lowering",
        )?;
        let key = (space_tag(space), n);
        match tys.get(&key) {
            Some(prev) if *prev != elem_ty => {
                return Err(unsupported_addr_space(
                    "local/shared/constant/param slot accessed at two different types",
                ))
            }
            _ => {
                tys.insert(key, elem_ty);
            }
        }
        Ok(())
    };

    let const_slot = |ptr: ValRef, space: AddrSpace| -> Option<i64> {
        let ValRef::Val(id) = ptr else { return None };
        let inst = &f.insts[id.0 as usize];
        match inst.op {
            Op::ConstInt(n) if inst.ty == Ty::Ptr(space) => Some(n),
            _ => None,
        }
    };

    for inst in &f.insts {
        match &inst.op {
            Op::Load { ptr, space, .. } if is_local_like(*space) => {
                if let Some(n) = const_slot(*ptr, *space) {
                    record(*space, n, inst.ty)?;
                }
            }
            Op::Store { ptr, ty, space, .. } if is_local_like(*space) => {
                if let Some(n) = const_slot(*ptr, *space) {
                    record(*space, n, *ty)?;
                }
            }
            _ => {}
        }
    }

    Ok(tys)
}

/// `basalt-sema`'s lowering has no `alloca`: every local/parameter/shared/constant storage
/// location is handed a small integer slot id and materialized, wherever BIR needs its
/// address, as `const.i ptr.<space> (slot_id * SLOT_STRIDE)` — an opaque per-`(space, id)`
/// key, not a real address (`basalt-x86`'s oracle and `basalt-llvm`'s own `build_local_slots`
/// both document and handle the identical pattern). This builds one real `memref.alloca` per
/// distinct key used in `f`, in `f`'s entry block ahead of its own instructions, so
/// `lower_inst`'s `Op::ConstInt` case has genuine backing storage to hand back instead of
/// treating the opaque key as a literal pointer value. Every slot is a flat, 0-d memref at
/// whatever scalar type `analyze_local_slots` derived for it — `memref` is strongly typed, so
/// (unlike `basalt-llvm`'s uniformly-`i64` `alloca`) this lowering cannot get away with one
/// fixed width for every slot regardless of what is actually stored there.
fn build_local_slots<'c, 'a>(
    context: &'c Context,
    entry: &'a Block<'c>,
    loc: Location<'c>,
    slot_tys: &HashMap<SlotKey, Scalar>,
) -> HashMap<SlotKey, Value<'c, 'a>> {
    let mut slots = HashMap::new();
    // `HashMap` iteration order is not deterministic; sort by key so this lowering's own
    // "same BIR in, byte-identical text out" invariant holds regardless of hash seed.
    let mut keys: Vec<&SlotKey> = slot_tys.keys().collect();
    keys.sort();
    for &key in &keys {
        let scalar = slot_tys[key];
        let mty = MemRefType::new(mlir_scalar_ty(context, scalar), &[], None, None);
        let built = memref::alloca(context, mty, &[], &[], None, loc);
        let val: Value<'c, 'a> = result_of(entry.append_operation(built));
        slots.insert(*key, val);
    }
    slots
}

fn mlir_param_ty<'c>(
    context: &'c Context,
    ty: Ty,
    param_model: &[Option<ParamElemModel>],
    index: usize,
) -> Result<Type<'c>, Diag> {
    match ty {
        Ty::Scalar(s) => Ok(mlir_scalar_ty(context, s)),
        Ty::Ptr(AddrSpace::Global) => {
            let elem_ty = match param_model[index].expect("validated by analyze_memory_accesses") {
                ParamElemModel::Typed(s) => mlir_scalar_ty(context, s),
                ParamElemModel::Bytes => IntegerType::new(context, 8).into(),
            };
            Ok(MemRefType::new(elem_ty, &[i64::MIN], None, None).into())
        }
        Ty::Ptr(_) => Err(unsupported_addr_space(
            "kernel parameter in a non-global address space is out of scope",
        )),
        Ty::Vec(..) => Err(unsupported_type(
            "vector-typed kernel parameter is out of scope",
        )),
        Ty::Void => Err(unsupported_type("void is not a valid parameter type")),
    }
}

/// Builds a single-element, real-type view of a byte-addressed (`ParamElemModel::Bytes`)
/// parameter's `memref<?xi8>` at the true byte offset `acc` resolved (`acc.off_id`'s own
/// value, already a real, ordinarily-lowered `i64` SSA value computing `index * esz`, plus
/// `acc.base_extra_bytes` if the recognized base folded in a constant offset — see
/// `RecognizedAccess`), reinterpreted as `elem_ty`. This is the byte-addressed model's
/// replacement for a direct `memref.load`/`.store` at the parameter's own (single, typed)
/// element type: `memref` gives no way to declare "this buffer holds more than one type", but
/// `memref.view` legally reinterprets a byte-offset slice of an `i8`-element, identity-layout
/// memref (this lowering's own `memref<?xi8>` parameter model, always identity-layout since it
/// is never given a strides/offset attribute of its own) as a memref of any other element type
/// — confirmed against a real, installed `mlir-opt` (LLVM/MLIR 22.1.6): a single-element result
/// (`memref<1xT>`, no dynamic `sizes` operands needed) sourced from a dynamically-shaped
/// `memref<?xi8>` at a dynamic `index`-typed byte offset round-trips with zero diagnostics for
/// every scalar type this lowering ever hands it (`f32`, `i64`, `i1` — the three real types
/// `tri_vadd.py`'s own scratch-sharing `c_ptr` parameter is visited at).
#[allow(clippy::too_many_arguments)]
fn byte_view<'c, 'a>(
    context: &'c Context,
    blk: &'a Block<'c>,
    buf: Value<'c, 'a>,
    params: &[Value<'c, 'a>],
    values: &[Option<Value<'c, 'a>>],
    acc: &RecognizedAccess,
    elem_ty: Scalar,
    loc: Location<'c>,
) -> Value<'c, 'a> {
    let off_val = get_val(params, values, ValRef::Val(acc.off_id));
    let total_bytes = if acc.base_extra_bytes != 0 {
        let base_const_attr =
            IntegerAttribute::new(mlir_scalar_ty(context, Scalar::I64), acc.base_extra_bytes);
        let base_const =
            result_of(blk.append_operation(arith::constant(context, base_const_attr.into(), loc)));
        result_of(blk.append_operation(arith::addi(base_const, off_val, loc)))
    } else {
        off_val
    };
    let byte_idx =
        result_of(blk.append_operation(arith::index_cast(total_bytes, Type::index(context), loc)));
    let result_ty = MemRefType::new(mlir_scalar_ty(context, elem_ty), &[1], None, None);
    let view = memref::view(context, buf, byte_idx, &[], result_ty, loc);
    result_of(blk.append_operation(view))
}

/// The single, always-zero element index every `byte_view` result (a statically-1-element
/// memref) is loaded/stored at.
fn zero_index<'c, 'a>(
    context: &'c Context,
    blk: &'a Block<'c>,
    loc: Location<'c>,
) -> Value<'c, 'a> {
    let attr = IntegerAttribute::new(Type::index(context), 0);
    result_of(blk.append_operation(arith::constant(context, attr.into(), loc)))
}

fn icmp_pred(p: ICmpPred) -> CmpiPredicate {
    match p {
        ICmpPred::Eq => CmpiPredicate::Eq,
        ICmpPred::Ne => CmpiPredicate::Ne,
        ICmpPred::Slt => CmpiPredicate::Slt,
        ICmpPred::Sle => CmpiPredicate::Sle,
        ICmpPred::Sgt => CmpiPredicate::Sgt,
        ICmpPred::Sge => CmpiPredicate::Sge,
        ICmpPred::Ult => CmpiPredicate::Ult,
        ICmpPred::Ule => CmpiPredicate::Ule,
        ICmpPred::Ugt => CmpiPredicate::Ugt,
        ICmpPred::Uge => CmpiPredicate::Uge,
    }
}

fn fcmp_pred(p: FCmpPred) -> CmpfPredicate {
    match p {
        FCmpPred::Oeq => CmpfPredicate::Oeq,
        FCmpPred::One => CmpfPredicate::One,
        FCmpPred::Olt => CmpfPredicate::Olt,
        FCmpPred::Ole => CmpfPredicate::Ole,
        FCmpPred::Ogt => CmpfPredicate::Ogt,
        FCmpPred::Oge => CmpfPredicate::Oge,
        FCmpPred::Ord => CmpfPredicate::Ord,
        FCmpPred::Uno => CmpfPredicate::Uno,
    }
}

/// `gpu.{thread_id,block_id,block_dim,grid_dim}` have no hand-written `melior` builder (see
/// the module header), so this builds the op generically and immediately narrows its
/// `index` result to `result_ty` (whatever plain integer scalar the instruction's own BIR
/// `Ty` declares — see `gpu_index_result_ty`) via `arith.index_cast` — the cast is the value
/// actually threaded through the rest of the function; the raw `gpu.*` op's own `index`
/// result is never referenced again, matching real MLIR GPU lowering pipelines' own
/// convention of narrowing at this exact boundary.
fn gpu_index<'c, 'a>(
    context: &'c Context,
    blk: &'a Block<'c>,
    op_name: &str,
    dim: &str,
    result_ty: Type<'c>,
) -> Value<'c, 'a> {
    let loc = Location::unknown(context);
    let dim_attr = Attribute::parse(context, &format!("#gpu<dim {dim}>"))
        .expect("#gpu<dim ..> is a real, stable MLIR GPU-dialect attribute literal");
    let raw = OperationBuilder::new(op_name, loc)
        .add_attributes(&[(Identifier::new(context, "dimension"), dim_attr)])
        .add_results(&[Type::index(context)])
        .build()
        .expect("valid operation");
    let idx = result_of(blk.append_operation(raw));
    let cast = arith::index_cast(idx, result_ty, loc);
    result_of(blk.append_operation(cast))
}

/// The plain integer scalar type a `TidX/Y/Z`/`BidX/Y/Z`/`BdimX/Y/Z`/`GdimX/Y/Z` instruction's
/// own BIR `Ty` narrows `gpu_index`'s `index` result to. `arith.index_cast` only ever
/// converts between `index` and a signless integer type, so a float/vector/pointer-typed one
/// of these ops (never produced by either proof kernel's own frontend, but not a shape this
/// lowering is willing to guess a cast for) is refused rather than attempted.
fn gpu_index_result_ty<'c>(context: &'c Context, ty: Ty) -> Result<Type<'c>, Diag> {
    match ty {
        Ty::Scalar(s @ (Scalar::I8 | Scalar::I16 | Scalar::I32 | Scalar::I64)) => {
            Ok(mlir_scalar_ty(context, s))
        }
        _ => Err(unsupported_type(
            "a GPU index op (tid/bid/bdim/gdim) must produce a plain integer scalar",
        )),
    }
}

fn get_val<'c, 'a>(
    params: &[Value<'c, 'a>],
    values: &[Option<Value<'c, 'a>>],
    v: ValRef,
) -> Value<'c, 'a> {
    match v {
        ValRef::Param(i) => params[i as usize],
        ValRef::Val(id) => values[id.0 as usize]
            .expect("operand instruction not yet lowered (BIR dominance invariant violated)"),
    }
}

#[allow(clippy::too_many_arguments)]
fn lower_inst<'c, 'a>(
    context: &'c Context,
    blk: &'a Block<'c>,
    f: &Function,
    params: &[Value<'c, 'a>],
    values: &[Option<Value<'c, 'a>>],
    local_slots: &HashMap<SlotKey, Value<'c, 'a>>,
    param_model: &[Option<ParamElemModel>],
    inst: &Inst,
) -> Result<Option<Value<'c, 'a>>, Diag> {
    let loc = Location::unknown(context);
    let get = |v: ValRef| get_val(params, values, v);

    Ok(Some(match &inst.op {
        Op::ConstInt(n) => match inst.ty {
            Ty::Scalar(s) => {
                let ty = mlir_scalar_ty(context, s);
                let built = arith::constant(context, IntegerAttribute::new(ty, *n).into(), loc);
                result_of(blk.append_operation(built))
            }
            Ty::Ptr(space) if is_local_like(space) => *local_slots
                .get(&(space_tag(space), *n))
                .expect("build_local_slots pre-scans every local-like slot constant"),
            Ty::Ptr(_) => {
                return Err(unsupported_addr_space(
                    "a literal ptr.global constant has no memref representation in this \
                     lowering (a real global pointer always arrives as a kernel parameter, \
                     or arithmetic on one)",
                ))
            }
            Ty::Vec(..) | Ty::Void => return Err(unsupported_type("const.i of a non-scalar type")),
        },
        Op::ConstFloat(n) => {
            let s = as_scalar(inst.ty, "const.f of a non-scalar type")?;
            let ty = mlir_scalar_ty(context, s);
            let built = arith::constant(context, FloatAttribute::new(context, ty, *n).into(), loc);
            result_of(blk.append_operation(built))
        }
        Op::Bin(op, a, b) => {
            if matches!(inst.ty, Ty::Ptr(_)) {
                return Err(unsupported_addr_space(
                    "pointer arithmetic outside the one recognized load/store addressing \
                     shape has no memref representation",
                ));
            }
            let (av, bv) = (get(*a), get(*b));
            let built = match op {
                BinOp::Add => arith::addi(av, bv, loc),
                BinOp::Sub => arith::subi(av, bv, loc),
                BinOp::Mul => arith::muli(av, bv, loc),
                BinOp::Div => arith::divsi(av, bv, loc),
                BinOp::Rem => arith::remsi(av, bv, loc),
                BinOp::FAdd => arith::addf(av, bv, loc),
                BinOp::FSub => arith::subf(av, bv, loc),
                BinOp::FMul => arith::mulf(av, bv, loc),
                BinOp::FDiv => arith::divf(av, bv, loc),
                BinOp::FRem => arith::remf(av, bv, loc),
                BinOp::And => arith::andi(av, bv, loc),
                BinOp::Or => arith::ori(av, bv, loc),
                BinOp::Xor => arith::xori(av, bv, loc),
                BinOp::Shl => arith::shli(av, bv, loc),
                BinOp::Lshr => arith::shrui(av, bv, loc),
                BinOp::Ashr => arith::shrsi(av, bv, loc),
            };
            result_of(blk.append_operation(built))
        }
        Op::ICmp(pred, _oty, a, b) => {
            let built = arith::cmpi(context, icmp_pred(*pred), get(*a), get(*b), loc);
            result_of(blk.append_operation(built))
        }
        Op::FCmp(pred, _oty, a, b) => {
            let built = arith::cmpf(context, fcmp_pred(*pred), get(*a), get(*b), loc);
            result_of(blk.append_operation(built))
        }
        Op::Select(c, t, e) => {
            let built = arith::select(get(*c), get(*t), get(*e), loc);
            result_of(blk.append_operation(built))
        }
        Op::Cast(cop, _src_ty, v) => {
            let target = match inst.ty {
                Ty::Scalar(s) => mlir_scalar_ty(context, s),
                _ => return Err(unsupported_type("cast to a non-scalar type")),
            };
            let vv = get(*v);
            let built = match cop {
                CastOp::Trunc => arith::trunci(vv, target, loc),
                CastOp::Zext => arith::extui(vv, target, loc),
                CastOp::Sext => arith::extsi(vv, target, loc),
                // `arith.truncf`'s target width cannot be inferred from its operand alone
                // (unlike `arith.negf`, which shares its op-generator macro in `melior`), so
                // this is built explicitly rather than through melior's own
                // `enable_result_type_inference()`-based helper.
                CastOp::FpTrunc => OperationBuilder::new("arith.truncf", loc)
                    .add_operands(&[vv])
                    .add_results(&[target])
                    .build()
                    .expect("valid operation"),
                CastOp::FpExt => arith::extf(vv, target, loc),
                CastOp::FpToSi => arith::fptosi(vv, target, loc),
                CastOp::FpToUi => arith::fptoui(vv, target, loc),
                CastOp::SiToFp => arith::sitofp(vv, target, loc),
                CastOp::UiToFp => arith::uitofp(vv, target, loc),
                CastOp::Bitcast => arith::bitcast(vv, target, loc),
            };
            result_of(blk.append_operation(built))
        }
        Op::Load { ptr, space, .. } if is_local_like(*space) => {
            let _ = as_scalar(inst.ty, "loaded value must be a plain scalar")?;
            let built = memref::load(get(*ptr), &[], loc);
            result_of(blk.append_operation(built))
        }
        Op::Load { ptr, .. } => {
            let elem_ty = as_scalar(inst.ty, "loaded value must be a plain scalar")?;
            let acc = recognize_access(f, *ptr, elem_ty)?;
            match param_model[acc.param_index].expect("validated by analyze_memory_accesses") {
                ParamElemModel::Typed(_) => {
                    let idx_int = get(acc.elem_index);
                    let idx = result_of(blk.append_operation(arith::index_cast(
                        idx_int,
                        Type::index(context),
                        loc,
                    )));
                    let built = memref::load(params[acc.param_index], &[idx], loc);
                    result_of(blk.append_operation(built))
                }
                ParamElemModel::Bytes => {
                    let view = byte_view(
                        context,
                        blk,
                        params[acc.param_index],
                        params,
                        values,
                        &acc,
                        elem_ty,
                        loc,
                    );
                    let zero = zero_index(context, blk, loc);
                    let built = memref::load(view, &[zero], loc);
                    result_of(blk.append_operation(built))
                }
            }
        }
        Op::Store {
            ptr, val, space, ..
        } if is_local_like(*space) => {
            let built = memref::store(get(*val), get(*ptr), &[], loc);
            blk.append_operation(built);
            return Ok(None);
        }
        Op::Store { ptr, val, ty, .. } => {
            let elem_ty = as_scalar(*ty, "stored value must be a plain scalar")?;
            let acc = recognize_access(f, *ptr, elem_ty)?;
            match param_model[acc.param_index].expect("validated by analyze_memory_accesses") {
                ParamElemModel::Typed(_) => {
                    let idx_int = get(acc.elem_index);
                    let idx = result_of(blk.append_operation(arith::index_cast(
                        idx_int,
                        Type::index(context),
                        loc,
                    )));
                    let built = memref::store(get(*val), params[acc.param_index], &[idx], loc);
                    blk.append_operation(built);
                }
                ParamElemModel::Bytes => {
                    let view = byte_view(
                        context,
                        blk,
                        params[acc.param_index],
                        params,
                        values,
                        &acc,
                        elem_ty,
                        loc,
                    );
                    let zero = zero_index(context, blk, loc);
                    let built = memref::store(get(*val), view, &[zero], loc);
                    blk.append_operation(built);
                }
            }
            return Ok(None);
        }
        Op::Phi(_) => unreachable!(
            "a phi's value is pre-bound to a block argument by lower_function before any \
             instruction in its block is lowered; it is never dispatched here"
        ),
        Op::TidX
        | Op::TidY
        | Op::TidZ
        | Op::BidX
        | Op::BidY
        | Op::BidZ
        | Op::BdimX
        | Op::BdimY
        | Op::BdimZ
        | Op::GdimX
        | Op::GdimY
        | Op::GdimZ => {
            let (op_name, dim) = match inst.op {
                Op::TidX => ("gpu.thread_id", "x"),
                Op::TidY => ("gpu.thread_id", "y"),
                Op::TidZ => ("gpu.thread_id", "z"),
                Op::BidX => ("gpu.block_id", "x"),
                Op::BidY => ("gpu.block_id", "y"),
                Op::BidZ => ("gpu.block_id", "z"),
                Op::BdimX => ("gpu.block_dim", "x"),
                Op::BdimY => ("gpu.block_dim", "y"),
                Op::BdimZ => ("gpu.block_dim", "z"),
                Op::GdimX => ("gpu.grid_dim", "x"),
                Op::GdimY => ("gpu.grid_dim", "y"),
                Op::GdimZ => ("gpu.grid_dim", "z"),
                _ => unreachable!("matched above"),
            };
            let result_ty = gpu_index_result_ty(context, inst.ty)?;
            gpu_index(context, blk, op_name, dim, result_ty)
        }
        Op::Barrier => {
            blk.append_operation(
                OperationBuilder::new("gpu.barrier", loc)
                    .build()
                    .expect("valid operation"),
            );
            return Ok(None);
        }
        Op::Shuffle(..) => {
            return Err(unsupported_feature(
                "warp shuffle has no settled gpu-dialect mapping without a target-specific \
                 dialect (nvgpu/amdgpu); deferred to a target-specific lowering task",
            ))
        }
        Op::Ballot(_) | Op::VoteAny(_) | Op::VoteAll(_) => {
            return Err(unsupported_feature(
                "ballot/vote has no settled gpu-dialect mapping without a target-specific \
                 dialect (nvgpu/amdgpu); deferred to a target-specific lowering task",
            ))
        }
        Op::Atomic(..) | Op::AtomicCas(..) => {
            return Err(unsupported_feature(
                "atomics need a target-specific memory-ordering mapping this dialect-only \
                 lowering does not attempt",
            ))
        }
        Op::Mma { .. } => {
            return Err(unsupported_op(
                "Op::Mma -> linalg.matmul is a real, deferred follow-up; not attempted by \
                 this task's vector_add-scoped minimum bar",
            ))
        }
    }))
}

/// The leading run of `Op::Phi` instructions in block `bi` — `basalt_passes::construct_ssa`'s
/// own documented output order (a block's phis, if any, always precede its ordinary body; see
/// that pass's header) is what makes reading this off as a plain prefix scan sound, with no
/// need to otherwise search the block's instruction list.
fn phi_ids(f: &Function, bi: usize) -> Vec<InstId> {
    f.blocks[bi]
        .insts
        .iter()
        .copied()
        .take_while(|id| matches!(f.insts[id.0 as usize].op, Op::Phi(_)))
        .collect()
}

/// The MLIR block-argument type a phi's own BIR type maps to. Only a scalar phi is attempted
/// — neither of this task's proof kernels ever produces a pointer- or vector-typed one (every
/// phi in both is `i64` or `f32`), so there is no real shape yet to confirm a `memref`/
/// `vector` block-argument mapping against.
fn phi_arg_ty<'c>(context: &'c Context, ty: Ty) -> Result<Type<'c>, Diag> {
    match ty {
        Ty::Scalar(s) => Ok(mlir_scalar_ty(context, s)),
        Ty::Vec(..) => Err(unsupported_type(
            "a vector-typed phi has no block-argument mapping attempted by this lowering",
        )),
        Ty::Ptr(_) | Ty::Void => Err(unsupported_addr_space(
            "a pointer-typed phi has no block-argument mapping attempted by this lowering",
        )),
    }
}

/// The branch operands a `cf.br`/`cf.cond_br` from block `pred` to block `target` must carry:
/// one value per `target`'s own leading phi, in order, each resolved to whichever incoming
/// value that phi records for `pred`.
fn branch_args<'c, 'a>(
    f: &Function,
    params: &[Value<'c, 'a>],
    values: &[Option<Value<'c, 'a>>],
    target: BlockId,
    pred: BlockId,
) -> Vec<Value<'c, 'a>> {
    phi_ids(f, target.0 as usize)
        .iter()
        .map(|&id| {
            let Op::Phi(incoming) = &f.insts[id.0 as usize].op else {
                unreachable!("phi_ids only ever returns Op::Phi instructions")
            };
            let v = incoming
                .iter()
                .find(|(b, _)| *b == pred)
                .unwrap_or_else(|| {
                    panic!(
                        "phi {id:?} at block {target:?} has no incoming value recorded for \
                         predecessor {pred:?} (BIR phi/predecessor invariant violated)"
                    )
                })
                .1;
            get_val(params, values, v)
        })
        .collect()
}

#[allow(clippy::too_many_arguments)]
fn lower_term<'c, 'a>(
    context: &'c Context,
    blk: &'a Block<'c>,
    mlir_blocks: &'a [Block<'c>],
    params: &[Value<'c, 'a>],
    values: &[Option<Value<'c, 'a>>],
    f: &Function,
    cur: BlockId,
    term: &Term,
) -> Result<(), Diag> {
    let loc = Location::unknown(context);
    match term {
        Term::Br(target) => {
            let args = branch_args(f, params, values, *target, cur);
            blk.append_operation(cf::br(&mlir_blocks[target.0 as usize], &args, loc));
        }
        Term::CondBr(cond, t, e) => {
            let c = get_val(params, values, *cond);
            let t_args = branch_args(f, params, values, *t, cur);
            let e_args = branch_args(f, params, values, *e, cur);
            blk.append_operation(cf::cond_br(
                context,
                c,
                &mlir_blocks[t.0 as usize],
                &mlir_blocks[e.0 as usize],
                &t_args,
                &e_args,
                loc,
            ));
        }
        Term::Switch(..) => {
            return Err(unsupported_op(
                "switch is real and cheap to add via melior's own cf::switch, but is not \
                 exercised by this task's one bring-up kernel and is deliberately left for \
                 whenever a kernel actually needs it",
            ))
        }
        Term::Ret(None) => {
            blk.append_operation(
                OperationBuilder::new("gpu.return", loc)
                    .build()
                    .expect("valid operation"),
            );
        }
        Term::Ret(Some(_)) => {
            return Err(unsupported_op(
                "a value-returning terminator has no meaning for a gpu.func kernel entry \
                 point; every BIR function reaching this lowering is assumed to be one",
            ))
        }
    }
    Ok(())
}

/// Pre-flight scan for ops/terminators this lowering refuses outright, run before any
/// codegen-shaped analysis (`analyze_memory_accesses` included). Without this, a function
/// that was never going to lower regardless (e.g. one using `Op::Mma`) could instead surface
/// a less specific, incidental refusal from a later pass (`Op::Mma`'s own pointer operands
/// are not routed through `Load`/`Store`, so `analyze_memory_accesses` would otherwise report
/// them as "never accessed" rather than naming the real, more useful reason). `lower_inst`/
/// `lower_term` re-check the identical conditions at the point they would otherwise mishandle
/// them — the same "pre-flight check plus a re-confirming check at the point of use" pattern
/// `Support`/`Backend::emit` already establishes project-wide, so the two can never drift
/// apart silently.
fn check_unsupported_ops(f: &Function) -> Result<(), Diag> {
    for inst in &f.insts {
        match &inst.op {
            Op::Phi(_) if matches!(inst.ty, Ty::Vec(..)) => {
                return Err(unsupported_type(
                    "a vector-typed phi has no block-argument mapping attempted by this \
                     lowering",
                ))
            }
            Op::Phi(_) if !matches!(inst.ty, Ty::Scalar(_)) => {
                return Err(unsupported_addr_space(
                    "a pointer-typed phi has no block-argument mapping attempted by this \
                     lowering",
                ))
            }
            Op::Shuffle(..) => {
                return Err(unsupported_feature(
                    "warp shuffle has no settled gpu-dialect mapping without a target-specific \
                     dialect (nvgpu/amdgpu); deferred to a target-specific lowering task",
                ))
            }
            Op::Ballot(_) | Op::VoteAny(_) | Op::VoteAll(_) => {
                return Err(unsupported_feature(
                    "ballot/vote has no settled gpu-dialect mapping without a target-specific \
                     dialect (nvgpu/amdgpu); deferred to a target-specific lowering task",
                ))
            }
            Op::Atomic(..) | Op::AtomicCas(..) => {
                return Err(unsupported_feature(
                    "atomics need a target-specific memory-ordering mapping this dialect-only \
                     lowering does not attempt",
                ))
            }
            Op::Mma { .. } => {
                return Err(unsupported_op(
                    "Op::Mma -> linalg.matmul is a real, deferred follow-up; not attempted by \
                     this task's vector_add-scoped minimum bar",
                ))
            }
            _ => {}
        }
    }
    for block in &f.blocks {
        match block.term {
            Term::Switch(..) => {
                return Err(unsupported_op(
                    "switch is real and cheap to add via melior's own cf::switch, but is not \
                     exercised by this task's one bring-up kernel and is deliberately left for \
                     whenever a kernel actually needs it",
                ))
            }
            Term::Ret(Some(_)) => {
                return Err(unsupported_op(
                    "a value-returning terminator has no meaning for a gpu.func kernel entry \
                     point; every BIR function reaching this lowering is assumed to be one",
                ))
            }
            _ => {}
        }
    }
    Ok(())
}

fn lower_function<'c>(
    context: &'c Context,
    gpu_module_block: &Block<'c>,
    f: &Function,
) -> Result<(), Diag> {
    if f.ret != Ty::Void {
        return Err(unsupported_op(
            "a non-void-returning top-level function has no meaning for a gpu.func kernel \
             entry point; every BIR function reaching this lowering is assumed to be one",
        ));
    }
    check_unsupported_ops(f)?;
    if !phi_ids(f, 0).is_empty() {
        // `basalt_passes::construct_ssa` only ever inserts a phi at a block with more than
        // one predecessor (see that pass's header); the entry block has none, so this never
        // fires on real optimizer output — kept as a real, precise refusal rather than an
        // assert, since a hand-built module could still reach here. A phi here would also
        // desync `gpu.func`'s own declared `function_type` (built from `param_tys` alone)
        // from its entry block's actual argument list.
        return Err(unsupported_op(
            "a phi in the function's entry block has no incoming value to bind (the entry \
             block has no predecessor) and is not attempted by this lowering",
        ));
    }

    let (param_model, skip) = analyze_memory_accesses(f)?;
    let slot_tys = analyze_local_slots(f)?;
    let param_tys: Vec<Type<'c>> = f
        .params
        .iter()
        .enumerate()
        .map(|(i, &ty)| mlir_param_ty(context, ty, &param_model, i))
        .collect::<Result<_, _>>()?;

    let loc = Location::unknown(context);
    // Every block gets one MLIR block argument per leading `Op::Phi` (see `phi_ids`), on top
    // of the function's own parameters for block 0 — see this file's own "Op::Phi" section
    // above for why this is sound without a topological fixup.
    let mlir_blocks: Vec<Block<'c>> = f
        .blocks
        .iter()
        .enumerate()
        .map(|(i, _)| {
            let phi_tys: Vec<Type<'c>> = phi_ids(f, i)
                .iter()
                .map(|&id| phi_arg_ty(context, f.insts[id.0 as usize].ty))
                .collect::<Result<_, _>>()?;
            let mut args: Vec<(Type<'c>, Location<'c>)> = Vec::new();
            if i == 0 {
                args.extend(param_tys.iter().map(|&ty| (ty, loc)));
            }
            args.extend(phi_tys.into_iter().map(|ty| (ty, loc)));
            Ok(Block::new(&args))
        })
        .collect::<Result<Vec<_>, Diag>>()?;

    let params: Vec<Value<'c, '_>> = (0..f.params.len())
        .map(|i| {
            mlir_blocks[0]
                .argument(i)
                .expect("declared parameter")
                .into()
        })
        .collect();

    // Real backing storage for every local/shared/constant/param slot this function's own
    // `Op::ConstInt`s reference — built into block 0 ahead of its own instructions (appended
    // here, before the main lowering walk below ever touches block 0), so every later slot
    // read/write dominates cleanly regardless of which block it is in.
    let local_slots = build_local_slots(context, &mlir_blocks[0], loc, &slot_tys);

    let mut values: Vec<Option<Value<'c, '_>>> = vec![None; f.insts.len()];
    // Bind every phi's BIR value to its own pre-built block argument before lowering a single
    // instruction — an operand referencing a phi (from within its own block, or from a block
    // this one dominates) must already resolve by the time the main walk below reaches it.
    for (bi, _) in f.blocks.iter().enumerate() {
        let base = if bi == 0 { param_tys.len() } else { 0 };
        for (k, id) in phi_ids(f, bi).iter().enumerate() {
            let arg = mlir_blocks[bi]
                .argument(base + k)
                .expect("declared phi block argument");
            values[id.0 as usize] = Some(arg.into());
        }
    }

    for (bi, bir_block) in f.blocks.iter().enumerate() {
        let blk = &mlir_blocks[bi];
        for &inst_id in &bir_block.insts {
            let inst = &f.insts[inst_id.0 as usize];
            // A phi's value is already bound above; it is never itself dispatched to
            // `lower_inst`.
            if matches!(inst.op, Op::Phi(_)) {
                continue;
            }
            if skip.contains(&inst_id.0) {
                continue;
            }
            values[inst_id.0 as usize] = lower_inst(
                context,
                blk,
                f,
                &params,
                &values,
                &local_slots,
                &param_model,
                inst,
            )?;
        }
        lower_term(
            context,
            blk,
            &mlir_blocks,
            &params,
            &values,
            f,
            BlockId(bi as u32),
            &bir_block.term,
        )?;
    }

    let region = Region::new();
    for blk in mlir_blocks {
        region.append_block(blk);
    }

    let function_type = FunctionType::new(context, &param_tys, &[]);
    let gpu_func = OperationBuilder::new("gpu.func", loc)
        .add_attributes(&[(
            Identifier::new(context, "function_type"),
            melior::ir::attribute::TypeAttribute::new(function_type.into()).into(),
        )])
        .add_attributes(&[
            (
                Identifier::new(context, "sym_name"),
                StringAttribute::new(context, &f.name).into(),
            ),
            (
                Identifier::new(context, "gpu.kernel"),
                Attribute::unit(context),
            ),
        ])
        .add_regions([region])
        .build()
        .expect("valid gpu.func operation");
    gpu_module_block.append_operation(gpu_func);

    Ok(())
}

#[cfg(test)]
mod tests;
