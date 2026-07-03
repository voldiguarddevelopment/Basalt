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
//   from MLIR's `index` result to BIR's fixed `i32` via `arith.index_cast` (real Triton and
//   MLIR GPU lowering pipelines do the same narrowing at this exact boundary — `index` is a
//   target-width-agnostic abstraction the dialect uses on purpose; BIR fixes these ops at
//   `i32`, matching the 32-bit register `basalt-ptx`/`basalt-llvm` already use for them).
//   `Barrier` becomes `gpu.barrier`. None of these five ops has a hand-written builder in
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
//   index, never a raw byte address). Rather than inventing a byte-addressed escape hatch
//   (there is none in `memref` short of `memref.reinterpret_cast`-ing to a byte-typed
//   1-D memref and computing the same flat address arithmetic SPIR-V's `Kernel` path
//   already does, which would defeat the entire point of choosing `memref` over a "hand
//   the raw pointer to `llvm.ptr`" dialect), this lane recognizes the *one* addressing shape
//   `basalt-sema`'s indexed-access lowering ever emits (confirmed by reading
//   `basalt-sema/src/lower.rs` and by dumping this project's own real post-optimize BIR for
//   `vector_add.cu` — see `lower::tests`) and reconstructs the original **element** index
//   from it, the same "recognize the one shape sema always produces, refuse everything
//   else" discipline `basalt-spirv`'s `glcompute.rs` (`recognize_access`) already
//   established for the structurally identical problem (`Logical` addressing there,
//   `memref` here) — a fresh implementation here, not shared code, per backend isolation.
//   See `recognize_access` below for the exact shape and what is refused.
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
// # Scope
//
// Implemented: kernel/module structure, `TidX/Y/Z`/`BidX/Y/Z`/`BdimX/Y/Z`/`GdimX/Y/Z`,
// `Barrier`, every `BinOp`, `ICmp`/`FCmp` (both predicate sets map onto `arith`'s own
// `Cmpi`/`CmpfPredicate` name-for-name), `Select`, every `CastOp`, `ConstInt`/`ConstFloat`
// (scalar only), `Load`/`Store` through the one recognized `memref` addressing shape, and
// `Br`/`CondBr`/`Ret(None)`.
//
// Refused with a stable E-code rather than guessed at: `Ty::Vec` (`E091`, the `vector`
// dialect's real target once a tile-shaped kernel bootstraps this lane further); a pointer
// value reaching `Load`/`Store` through any shape other than the one `recognize_access`
// walks, including a bare pointer parameter used with no offset arithmetic at all (`E092`);
// a kernel parameter in a non-`Global` address space, or a `Global` pointer parameter never
// read or written through the recognized shape (`E092`, no element type to derive `memref`'s
// type from); a local/shared/constant `ConstInt ptr.<space>` slot key, `basalt-sema`'s
// opaque-address-synthesis idiom that `basalt-llvm` backs with a real `alloca` — this lane
// has no such storage-synthesis pass yet, so it refuses rather than misreading the opaque
// key as a real index (`E092`); `Op::Phi` (`E090` — MLIR has no implicit-phi block
// instruction the way BIR does; a phi becomes a block argument plus a matching operand on
// every predecessor's branch, which needs a real pre-pass this lane does not yet have, see
// `basalt-llvm`'s own two-and-a-half-pass note for the mirror-image problem it solves for
// LLVM IR, which *does* have implicit phis); `Op::Switch` (`E090`, real and cheap to add via
// `melior::dialect::cf::switch` but not exercised by this task's one bring-up kernel, so
// deliberately left for whenever a kernel actually needs it); `Op::Shuffle`/`Ballot`/
// `VoteAny`/`VoteAll`/`Atomic`/`AtomicCas` (`E093` — every one of these has a real mapping
// only once a target-specific dialect, `nvgpu`/`amdgpu`, is in the picture, exactly like
// `basalt-llvm`'s own per-dialect gaps for the harder warp-level ops; deferred to P11-T2,
// not guessed at here); `Op::Mma` (`E090`, see "linalg" above); a non-`Void`-returning
// top-level function (`E090` — every BIR function reaching this lowering is assumed a
// kernel entry point, the same assumption `basalt-llvm`'s own `amdgpu_kernel`-calling-
// convention code documents, and a real device function on a GPU target is out of scope
// until this project has one).

use std::collections::HashSet;

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
    AddrSpace, BinOp, CastOp, FCmpPred, Function, ICmpPred, Inst, InstId, Module as BirModule, Op,
    Scalar, Term, Ty, ValRef,
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

/// Walks the one pointer-arithmetic shape `basalt-sema` ever emits for an indexed memory
/// access (confirmed against its real lowering and against this project's own compiled
/// `vector_add.cu` BIR):
///
/// ```text
/// %sext  = sext i64 i32 <inner>        ; <inner> is the pre-widened element index
/// %esz   = const.i i64 <esz>           ; <esz> = the accessed scalar's byte size
/// %off   = mul i64 %sext, %esz
/// %addr  = add ptr.global <param>, %off
/// ```
///
/// Returns `(param index, inner index ValRef, the outer Add's InstId)` on a match. Refuses
/// (`E092`) at the first operand that does not match, never guessing at what an
/// unrecognized shape might mean — the same discipline `basalt-spirv::glcompute`'s own
/// `recognize_access` already established for the identical "no raw address representation"
/// problem under `Logical` addressing.
fn recognize_access(
    f: &Function,
    ptr: ValRef,
    elem_ty: Scalar,
) -> Result<(usize, ValRef, InstId), Diag> {
    const SHAPE_MISMATCH: &str =
        "load/store address is not the one recognized element-index shape this lowering walks";

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
    let param_index = match base {
        ValRef::Param(i) => i as usize,
        _ => return Err(unsupported_addr_space(SHAPE_MISMATCH)),
    };
    let off_id = match off {
        ValRef::Val(id) => id,
        ValRef::Param(_) => return Err(unsupported_addr_space(SHAPE_MISMATCH)),
    };
    let (sext_val, stride) = match f.insts[off_id.0 as usize].op {
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
    let sext_id = match sext_val {
        ValRef::Val(id) => id,
        ValRef::Param(_) => return Err(unsupported_addr_space(SHAPE_MISMATCH)),
    };
    // `Op::Cast`'s own `Ty` field is the *source* type (`Inst::ty` carries the destination —
    // see `basalt-bir/src/ir.rs`'s own doc comment on `Op::Cast`), so both must be checked:
    // the instruction's result is `i64` and the cast narrows from `i32`.
    let sext_inst = &f.insts[sext_id.0 as usize];
    if sext_inst.ty != Ty::Scalar(Scalar::I64) {
        return Err(unsupported_addr_space(SHAPE_MISMATCH));
    }
    let inner = match sext_inst.op {
        Op::Cast(CastOp::Sext, Ty::Scalar(Scalar::I32), inner) => inner,
        _ => return Err(unsupported_addr_space(SHAPE_MISMATCH)),
    };
    Ok((param_index, inner, add_id))
}

/// First pass over a function's instructions: derives each `Ty::Ptr(Global)` parameter's
/// `memref` element type from the recognized accesses that use it (mirroring
/// `basalt-spirv::glcompute::resolve_binding_elem_tys`'s identical need), and collects the
/// set of "outer add" instructions a recognized access consumes so the second (codegen)
/// pass skips them rather than attempting arithmetic `memref` has no representation for.
fn analyze_memory_accesses(f: &Function) -> Result<(Vec<Option<Scalar>>, HashSet<u32>), Diag> {
    let mut elem_ty_of_param: Vec<Option<Scalar>> = vec![None; f.params.len()];
    let mut skip = HashSet::new();

    let mut record = |ptr: ValRef, elem_ty: Scalar| -> Result<(), Diag> {
        let (param_index, _inner, add_id) = recognize_access(f, ptr, elem_ty)?;
        match elem_ty_of_param[param_index] {
            Some(prev) if prev != elem_ty => {
                return Err(unsupported_addr_space(
                    "pointer parameter accessed at two different element types",
                ))
            }
            _ => elem_ty_of_param[param_index] = Some(elem_ty),
        }
        skip.insert(add_id.0);
        Ok(())
    };

    for inst in &f.insts {
        match &inst.op {
            Op::Load { ptr, space, .. } => {
                if *space != AddrSpace::Global {
                    return Err(unsupported_addr_space(
                        "load from a non-global address space is out of scope for this lowering",
                    ));
                }
                let elem_ty = as_scalar(inst.ty, "loaded value must be a plain scalar")?;
                record(*ptr, elem_ty)?;
            }
            Op::Store { ptr, ty, space, .. } => {
                if *space != AddrSpace::Global {
                    return Err(unsupported_addr_space(
                        "store to a non-global address space is out of scope for this lowering",
                    ));
                }
                let elem_ty = as_scalar(*ty, "stored value must be a plain scalar")?;
                record(*ptr, elem_ty)?;
            }
            _ => {}
        }
    }

    for (i, &ty) in f.params.iter().enumerate() {
        match ty {
            Ty::Ptr(AddrSpace::Global) if elem_ty_of_param[i].is_none() => {
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

    Ok((elem_ty_of_param, skip))
}

fn mlir_param_ty<'c>(
    context: &'c Context,
    ty: Ty,
    elem_ty_of_param: &[Option<Scalar>],
    index: usize,
) -> Result<Type<'c>, Diag> {
    match ty {
        Ty::Scalar(s) => Ok(mlir_scalar_ty(context, s)),
        Ty::Ptr(AddrSpace::Global) => {
            let elem = elem_ty_of_param[index].expect("validated by analyze_memory_accesses");
            Ok(MemRefType::new(mlir_scalar_ty(context, elem), &[i64::MIN], None, None).into())
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
/// `index` result to BIR's fixed `i32` via `arith.index_cast` — the cast is the value
/// actually threaded through the rest of the function; the raw `gpu.*` op's own `index`
/// result is never referenced again, matching real MLIR GPU lowering pipelines' own
/// convention of narrowing at this exact boundary.
fn gpu_index<'c, 'a>(
    context: &'c Context,
    blk: &'a Block<'c>,
    op_name: &str,
    dim: &str,
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
    let cast = arith::index_cast(idx, IntegerType::new(context, 32).into(), loc);
    result_of(blk.append_operation(cast))
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
            Ty::Ptr(_) => {
                return Err(unsupported_addr_space(
                    "opaque local/shared/constant slot constant has no backing storage in \
                     this lowering (basalt-sema's alloca-needing address-synthesis idiom is \
                     not yet implemented here)",
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
        Op::Load { ptr, .. } => {
            let elem_ty = as_scalar(inst.ty, "loaded value must be a plain scalar")?;
            let (param_index, inner, _) = recognize_access(f, *ptr, elem_ty)?;
            let idx_i32 = get(inner);
            let idx = result_of(blk.append_operation(arith::index_cast(
                idx_i32,
                Type::index(context),
                loc,
            )));
            let built = memref::load(params[param_index], &[idx], loc);
            result_of(blk.append_operation(built))
        }
        Op::Store { ptr, val, ty, .. } => {
            let elem_ty = as_scalar(*ty, "stored value must be a plain scalar")?;
            let (param_index, inner, _) = recognize_access(f, *ptr, elem_ty)?;
            let idx_i32 = get(inner);
            let idx = result_of(blk.append_operation(arith::index_cast(
                idx_i32,
                Type::index(context),
                loc,
            )));
            let built = memref::store(get(*val), params[param_index], &[idx], loc);
            blk.append_operation(built);
            return Ok(None);
        }
        Op::Phi(_) => {
            return Err(unsupported_op(
                "phi has no MLIR block-instruction equivalent; lowering it to a block \
                 argument plus per-predecessor branch operands is a real, deferred follow-up",
            ))
        }
        Op::TidX => gpu_index(context, blk, "gpu.thread_id", "x"),
        Op::TidY => gpu_index(context, blk, "gpu.thread_id", "y"),
        Op::TidZ => gpu_index(context, blk, "gpu.thread_id", "z"),
        Op::BidX => gpu_index(context, blk, "gpu.block_id", "x"),
        Op::BidY => gpu_index(context, blk, "gpu.block_id", "y"),
        Op::BidZ => gpu_index(context, blk, "gpu.block_id", "z"),
        Op::BdimX => gpu_index(context, blk, "gpu.block_dim", "x"),
        Op::BdimY => gpu_index(context, blk, "gpu.block_dim", "y"),
        Op::BdimZ => gpu_index(context, blk, "gpu.block_dim", "z"),
        Op::GdimX => gpu_index(context, blk, "gpu.grid_dim", "x"),
        Op::GdimY => gpu_index(context, blk, "gpu.grid_dim", "y"),
        Op::GdimZ => gpu_index(context, blk, "gpu.grid_dim", "z"),
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

fn lower_term<'c, 'a>(
    context: &'c Context,
    blk: &'a Block<'c>,
    mlir_blocks: &'a [Block<'c>],
    params: &[Value<'c, 'a>],
    values: &[Option<Value<'c, 'a>>],
    term: &Term,
) -> Result<(), Diag> {
    let loc = Location::unknown(context);
    match term {
        Term::Br(target) => {
            blk.append_operation(cf::br(&mlir_blocks[target.0 as usize], &[], loc));
        }
        Term::CondBr(cond, t, e) => {
            let c = get_val(params, values, *cond);
            blk.append_operation(cf::cond_br(
                context,
                c,
                &mlir_blocks[t.0 as usize],
                &mlir_blocks[e.0 as usize],
                &[],
                &[],
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
            Op::Phi(_) => {
                return Err(unsupported_op(
                    "phi has no MLIR block-instruction equivalent; lowering it to a block \
                     argument plus per-predecessor branch operands is a real, deferred \
                     follow-up",
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

    let (elem_ty_of_param, skip) = analyze_memory_accesses(f)?;
    let param_tys: Vec<Type<'c>> = f
        .params
        .iter()
        .enumerate()
        .map(|(i, &ty)| mlir_param_ty(context, ty, &elem_ty_of_param, i))
        .collect::<Result<_, _>>()?;

    let loc = Location::unknown(context);
    let mlir_blocks: Vec<Block<'c>> = f
        .blocks
        .iter()
        .enumerate()
        .map(|(i, _)| {
            if i == 0 {
                Block::new(&param_tys.iter().map(|&ty| (ty, loc)).collect::<Vec<_>>())
            } else {
                Block::new(&[])
            }
        })
        .collect();

    let params: Vec<Value<'c, '_>> = (0..f.params.len())
        .map(|i| {
            mlir_blocks[0]
                .argument(i)
                .expect("declared parameter")
                .into()
        })
        .collect();
    let mut values: Vec<Option<Value<'c, '_>>> = vec![None; f.insts.len()];

    for (bi, bir_block) in f.blocks.iter().enumerate() {
        let blk = &mlir_blocks[bi];
        for &inst_id in &bir_block.insts {
            if skip.contains(&inst_id.0) {
                continue;
            }
            let inst = &f.insts[inst_id.0 as usize];
            values[inst_id.0 as usize] = lower_inst(context, blk, f, &params, &values, inst)?;
        }
        lower_term(
            context,
            blk,
            &mlir_blocks,
            &params,
            &values,
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
