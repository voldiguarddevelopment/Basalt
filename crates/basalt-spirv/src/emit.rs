// Hand-rolled SPIR-V emitter, built on `rspirv` (see `Cargo.toml` for the version and why this
// dependency was picked). This backend targets the `Kernel` execution model with a `Physical64`
// addressing model and the `OpenCL` memory model — i.e. an OpenCL-C-shaped SPIR-V module, not a
// `GLCompute`/Vulkan-shaped one. That choice is deliberate, not incidental, and is the single
// most important design decision in this file; see "Why Kernel, not GLCompute" below.
//
// # Why Kernel, not GLCompute
//
// BIR's own pointer model (see `basalt-bir/src/ir.rs`) is a flat, byte-addressed one: a
// `Ty::Ptr(space)` value carries no pointee type, and pointer arithmetic is ordinary `Bin::Add`/
// `Sub` on an opaque address, exactly the model `basalt-ptx` and `basalt-x86` already lean on
// (see `basalt-ptx/src/emit.rs`'s own header). SPIR-V's `GLCompute` execution model runs under
// `Logical` addressing, which has no such thing as "a pointer plus a byte count" — pointers
// there are structured handles into descriptor-bound resources, reached only through
// `OpAccessChain`'s element/member indices, with no legal pointer-to-integer or
// integer-to-pointer conversion at all. Forcing BIR's raw-address model through that would mean
// inventing a resource-binding ABI (SSBO bindings, `ArrayStride`/`Offset` decorations, push
// constants for scalar args) that BIR does not carry any information for, and getting it right
// would be guessing, not lowering.
//
// The `Kernel` execution model (`Physical32`/`Physical64` addressing, `OpenCL` memory model) is
// the one SPIR-V has that matches BIR's own pointer model almost exactly: a kernel argument of
// pointer type is a real `OpFunctionParameter` of pointer type (no synthesized resource-binding
// ABI needed — this is precisely `basalt-bir::Function::params` already, one-to-one, the same
// simplicity `basalt-ptx`'s `.param` list already gets); and, with the `Addresses` capability
// (required by `Physical64` addressing; confirmed against `spirv.core.grammar.json`'s
// `AddressingModel` operand-kind table), `OpConvertPtrToU`/`OpConvertUToPtr` give a real,
// spec-legal way to treat a pointer as a raw 64-bit integer and back — exactly the two
// conversions this backend needs at its two boundaries (see "Pointer representation" below).
// This is not a lesser-used corner of SPIR-V: it is the same shape every real OpenCL-C-to-
// SPIR-V compiler (e.g. Clang's `-target spir64`) already emits for `__kernel` functions, so
// this is a well-trodden, verifiable target, not a novel one.
//
// The task that motivated this backend suggested `GLCompute` plus the `GlobalInvocationId`/
// `WorkgroupId`/`WorkgroupSize` builtins; per-op, the `Kernel` execution model uses the *same*
// `BuiltIn` enumerants for `WorkgroupId`/`LocalInvocationId`/`NumWorkgroups` (confirmed against
// `spirv.core.grammar.json`'s `BuiltIn` operand-kind table, none of which carry an execution-
// model-specific capability restriction) for the identical OpenCL `get_group_id`/`get_local_id`/
// `get_num_groups` intrinsics, so those three map the same way either way; only the addressing/
// pointer story changes, and that is exactly the part that needed to change for BIR to lower
// here without guessing. `WorkgroupSize` itself is *not* used by this backend — see "GPU index
// op -> BuiltIn mapping" below for why, and the real `spirv-val` finding that drove that call.
//
// # Pointer representation
//
// Every BIR `Ty::Ptr(Global)` value is represented, throughout arithmetic, as a plain 64-bit
// unsigned integer (the same `%ulong` type used for `Ty::Scalar(Scalar::I64)`) — the raw
// address, nothing more — exactly mirroring `basalt-ptx`'s own stance that a BIR pointer is
// "just another register" (`reg_class_of: Ty::Ptr(_) => RegClass::B64`). A real SPIR-V pointer
// object only exists momentarily at the two places the type system actually requires one:
//   - A `Ty::Ptr(Global)` function parameter is declared as a genuine
//     `OpTypePointer CrossWorkgroup %uint` (the pointee type is otherwise never observed — BIR
//     carries none — so `%uint` was picked only because it is already declared for other
//     reasons, not because it means anything); the very first instruction of the entry block
//     converts it to the working `%ulong` value via `OpConvertPtrToU`.
//   - Each `Load`/`Store`/(there are no atomics in this pass) reconstructs a pointer typed to
//     whatever scalar is actually being accessed via `OpConvertUToPtr` immediately before the
//     access, then never reuses that typed pointer again.
// `Bin::Add`/`Sub`/... on a `Ty::Ptr(Global)` result is therefore ordinary `%ulong` integer
// arithmetic (`OpIAdd`/`OpISub`/...), identical to `Ty::Scalar(Scalar::I64)` — this is exactly
// how BIR already expresses pointer arithmetic (see `basalt-sema/src/lower.rs`'s byte-offset
// convention): a `Bin::Add(ptr, i64_byte_offset)` needs no `OpPtrAccessChain`/element-index
// translation at all under this representation, since the "pointer" was never anything but an
// integer to begin with.
//
// # GPU index op -> BuiltIn mapping
//
// This backend always declares a fixed work-group size of `(1, 1, 1)` (`OpExecutionMode
// LocalSize 1 1 1` on every entry point, matching an OpenCL `reqd_work_group_size(1,1,1)`
// kernel) — a deliberate simplification, not a placeholder: it makes every one of the mappings
// below exact and unconditional, at the cost of one work-item per work-group (correct on any
// real device, just not occupancy-optimal; optimizing this is out of scope for a first pass,
// exactly the same "correct first, fast later" call every backend in this tree already makes).
// With that fixed:
//   - `Op::TidX/Y/Z`  (`threadIdx`) -> component of the `LocalInvocationId` `Input` builtin
//     variable (always `(0,0,0)` given the fixed work-group size, but read for real rather than
//     assumed, so this still lowers correctly if a later task changes the fixed size).
//   - `Op::BidX/Y/Z`  (`blockIdx`)  -> component of the `WorkgroupId` `Input` builtin variable
//     — a genuine per-dispatch hardware value, unlike `BdimX` below.
//   - `Op::GdimX/Y/Z` (`gridDim`)   -> component of the `NumWorkgroups` `Input` builtin variable
//     — also a genuine, host-dispatch-time value (the OpenCL analogue of PTX's `%nctaid`).
//   - `Op::BdimX/Y/Z` (`blockDim`)  -> the plain compile-time constant `1`, matching the fixed
//     `LocalSize` declared above — not a builtin-variable read at all. An earlier version of
//     this backend instead built a `BuiltIn WorkgroupSize`-decorated `OpConstantComposite` (the
//     spec does define that BuiltIn); `spirv-val` (present on this machine — see "Validation
//     tier" below) rejected it: `BuiltIn decoration on target <id> ... must be a variable`, i.e.
//     the real, current SPIRV-Tools validator requires `WorkgroupSize` to decorate a *variable*
//     regardless of what the spec text alone suggests, and no variable storage class this
//     backend has a capability for (`Private`/`Function`) is legal at module scope without
//     capabilities (`Shader`) this backend does not otherwise need. Rather than add an unneeded
//     capability to route around a single decoration, `BdimX`/`Y`/`Z` are represented as exactly
//     what they are under this backend's fixed-size convention: a compile-time-known value, not
//     a hardware read — no builtin machinery required to say that honestly.
//
// `Op::Barrier` lowers to a real `OpControlBarrier` (`Workgroup` execution+memory scope,
// `AcquireRelease | WorkgroupMemory` semantics) — genuinely load-bearing on real concurrent
// hardware, matching `basalt-ptx`'s `bar.sync` and unlike the CPU oracle's `nop`.
//
// # Control-flow scope: single-level structured if/if-else only
//
// SPIR-V under a `Shader`-family or `Kernel` module still requires *structured* control flow:
// every `OpBranchConditional` that is not a loop back-edge must be preceded, in the same block,
// by an `OpSelectionMerge` naming the block where its two arms reconverge (confirmed against
// `spirv.core.grammar.json`'s `OpSelectionMerge`/`OpBranchConditional` instruction entries, and
// well-established SPIR-V structured-control-flow doctrine). This backend supports exactly:
//   - straight-line code (any chain of `Br`s ending in `Ret`), and
//   - one level of `if`/`if-else`: a `CondBr` whose two arms are themselves straight-line `Br`
//     chains that reconverge at a common block (`find_merge_block`, used by both `supports()`
//     and codegen so the two can never drift apart).
// `Op::Phi` lowers to a genuine `OpPhi` (SPIR-V has real phis, unlike `basalt-ptx`'s per-
// predecessor-edge `mov` copies) referencing each predecessor's own label id and already-
// computed value id — always already known by the time a phi is reached, since only forward
// control flow is in scope (see below), so no forward-reference bookkeeping is needed.
// `Term::Switch` and any back-edge (a successor whose block index is `<=` its own block's
// index — i.e. any loop) are refused outright (`E093`): this is a real, current gap, not a
// guess at what a loop's merge/continue structure should be.
//
// # Refusal surface (everything else)
//
// - `i8`/`i16`/`f16` and every `Ty::Vec` (`E091`): only `i1`/`i32`/`i64`/`f32`/`f64` and
//   `Ptr(Global)` are given a representation. `i1`-typed `Load`/`Store` (`E091`) is refused
//   separately: SPIR-V's `Bool` type has no defined in-memory representation, and guessing one
//   (a byte? a bit?) is exactly the kind of silently-wrong codegen this project refuses to ship.
// - Any address space other than `Global` (`E092`): `Local`/`Shared`/`Constant`/`Param` are not
//   implemented — `tests/kernels/vector_add.cu` needs none of them once `construct_ssa` has run
//   (confirmed empirically: its post-optimize BIR contains zero non-`Global` `Load`/`Store`).
// - `Op::Atomic`/`Op::AtomicCas`, `Op::Shuffle`/`Op::Ballot`/`Op::VoteAny`/`Op::VoteAll`
//   (`E093`): no lowering in this pass yet.
// - `Op::Mma` (`E090`): no tensor-core-shaped path in this backend, matching
//   `basalt-ptx`'s identical stance and E-code choice for the identical situation.
// - Pointer-typed or `i1`-typed `Cast(Bitcast, ...)` (`E091`): a pointer is already represented
//   as a plain integer (see above), so a pointer bitcast has nothing distinct to do and is
//   refused rather than silently aliased; a bool has no defined bit pattern to reinterpret,
//   matching `basalt-ptx`'s identical refusal.
// - A `Bin` whose result type is `i1` with any op other than `And`/`Or`/`Xor` (`E090`): matches
//   `basalt-ptx`'s identical restriction (only logical ops are defined on a boolean `Bin`).
//
// # Validation tier: real-validator-confirmed (`spirv-val`), not silicon/simulator
//
// `spirv-val` (SPIRV-Tools v2026.2, package `spirv-tools 1:1.4.350.1`) *is* installed on the
// machine this backend was developed on, and was actually run — this is not a substitute-checks-
// only claim. `spirv-val` (default target environment, matching this module's own declared
// SPIR-V 1.4) was run directly against this backend's real emitted bytes for: the real
// frontend/sema/passes/emit pipeline over `tests/kernels/vector_add.cu`; a hand-built if/else
// module exercising a genuine `OpPhi` merge; a hand-built module exercising all twelve GPU-index
// ops plus `OpControlBarrier`; and a hand-built module exercising `f32`/`f64`/`i64` arithmetic,
// float widen/narrow/`frem` casts, and pointer store. All four passed with exit code 0 and no
// diagnostics. (One real, load-bearing catch along the way: an earlier version of this backend
// decorated an `OpConstantComposite` with `BuiltIn WorkgroupSize`, which `spirv-val` correctly
// rejected — "`BuiltIn decoration on target <id> ... must be a variable`" — leading to the
// plain-constant `BdimX`/`Y`/`Z` lowering described above instead. That is exactly the kind of
// mistake this validation step exists to catch, and it caught it.)
//
// This is still not a blanket, permanent guarantee: `spirv-val`'s presence is a property of the
// machine this was developed on, not of the crate's own dependency graph (`basalt-spirv` has no
// build-time or runtime dependency on SPIRV-Tools; it cannot invoke `spirv-val` itself, and
// `./check.sh` does not shell out to it), so a future CI/dev environment lacking it would fall
// back to this backend's own test suite (`emit/tests.rs`): magic number, version/bound header
// fields, capability/memory-model/entry-point presence, and a round-trip through `rspirv`'s own
// binary parser (`dr::load_bytes`) confirming the emitted words parse back into the same
// structured module `rspirv` built. Read this backend as **`spirv-val`-confirmed on the
// specific modules listed above, on this development machine**, in addition to (not instead of)
// oracle-validated BIR semantics — it has not been run on real driver/hardware or a software
// Vulkan/Level-Zero runtime (that is P9-T2's job, named explicitly out of scope for this task).
//
// # A second path: `GLCompute` (`EmitOpts::target_variant == Some("glcompute")`)
//
// Everything above this point describes this backend's *default* behavior, unconditionally
// unchanged by the second path below: `target_variant == None` (or any string other than
// `"glcompute"`) always takes the `Kernel` path exactly as documented above. P9-T2
// (`basalt-runtime`'s Vulkan loader; see `crates/basalt-runtime/src/vulkan/mod.rs`) confirmed,
// against real `llvmpipe`, that `vkCreateComputePipelines` refuses this backend's `Kernel`-model
// output unconditionally — Vulkan's compute pipeline API requires `GLCompute` by spec, not by
// driver quirk. `glcompute.rs` (a child module of this one — see its own header for the full
// resource-binding ABI and pointer-arithmetic-to-`OpAccessChain` recognition it adds) is a real,
// second, opt-in emission path for exactly that model, selected only by an explicit
// `target_variant`, reusing this file's own arithmetic/cast/compare/control-flow/GPU-index
// lowering verbatim wherever the underlying SPIR-V shape is genuinely execution-model-agnostic
// (confirmed, not assumed — see `glcompute.rs`'s own validation-tier section).

use rspirv::binary::Assemble;
use rspirv::dr::{Builder, Operand};
use rspirv::spirv::{
    AddressingModel, BuiltIn, Capability, Decoration, ExecutionMode, ExecutionModel,
    FunctionControl, MemoryAccess, MemoryModel, MemorySemantics, Scope, SelectionControl,
    StorageClass, Word,
};

use basalt_backend::{Artifact, ArtifactKind, Backend, EmitOpts, Support};
use basalt_bir::{
    AddrSpace, BinOp, BlockId, CastOp, FCmpPred, Function, ICmpPred, Inst, Module, Op, Scalar,
    Term, Ty, ValRef,
};
use basalt_diag::{Diag, ECode};
use basalt_passes::construct_ssa;

// ---- refusal surface ------------------------------------------------------------------------

fn ty_in_scope(ty: Ty) -> Result<(), Diag> {
    match ty {
        Ty::Void => Ok(()),
        Ty::Scalar(Scalar::I1 | Scalar::I32 | Scalar::I64 | Scalar::F32 | Scalar::F64) => Ok(()),
        Ty::Scalar(Scalar::I8 | Scalar::I16 | Scalar::F16) => Err(Diag::new(
            ECode::UnsupportedType,
        )
        .with_arg("i8/i16/f16: only i1/i32/i64/f32/f64 are given a representation in this backend's first pass")),
        Ty::Ptr(AddrSpace::Global) => Ok(()),
        Ty::Ptr(_) => Err(Diag::new(ECode::UnsupportedAddressSpace)
            .with_arg("only AddrSpace::Global is implemented; local/shared/constant/param are not")),
        Ty::Vec(..) => Err(Diag::new(ECode::UnsupportedType)
            .with_arg("vector types are not implemented in this backend's first pass")),
    }
}

fn check_inst(inst: &Inst) -> Result<(), Diag> {
    ty_in_scope(inst.ty)?;
    match &inst.op {
        Op::Bin(op, ..) => {
            if matches!(inst.ty, Ty::Scalar(Scalar::I1))
                && !matches!(op, BinOp::And | BinOp::Or | BinOp::Xor)
            {
                return Err(Diag::new(ECode::UnsupportedOp)
                    .with_arg("only and/or/xor are defined on a bool-typed Bin"));
            }
        }
        Op::ICmp(_, cty, ..) | Op::FCmp(_, cty, ..) => ty_in_scope(*cty)?,
        Op::Cast(CastOp::Bitcast, sty, _) => {
            ty_in_scope(*sty)?;
            if matches!(inst.ty, Ty::Ptr(_)) || matches!(sty, Ty::Ptr(_)) {
                return Err(Diag::new(ECode::UnsupportedType).with_arg(
                    "pointer bitcast is not implemented; a pointer is already a plain integer here",
                ));
            }
            if matches!(inst.ty, Ty::Scalar(Scalar::I1)) || matches!(sty, Ty::Scalar(Scalar::I1)) {
                return Err(Diag::new(ECode::UnsupportedType)
                    .with_arg("bitcast on a bool-typed value has no defined bit pattern"));
            }
        }
        Op::Cast(_, sty, _) => ty_in_scope(*sty)?,
        Op::Load { space, .. } => {
            if *space != AddrSpace::Global {
                return Err(Diag::new(ECode::UnsupportedAddressSpace)
                    .with_arg("only AddrSpace::Global loads are implemented"));
            }
            if matches!(inst.ty, Ty::Scalar(Scalar::I1)) {
                return Err(Diag::new(ECode::UnsupportedType)
                    .with_arg("bool-typed load has no defined in-memory representation"));
            }
        }
        Op::Store { space, ty, .. } => {
            if *space != AddrSpace::Global {
                return Err(Diag::new(ECode::UnsupportedAddressSpace)
                    .with_arg("only AddrSpace::Global stores are implemented"));
            }
            ty_in_scope(*ty)?;
            if matches!(ty, Ty::Scalar(Scalar::I1)) {
                return Err(Diag::new(ECode::UnsupportedType)
                    .with_arg("bool-typed store has no defined in-memory representation"));
            }
        }
        Op::Atomic(..) | Op::AtomicCas(..) => {
            return Err(Diag::new(ECode::UnsupportedFeature)
                .with_arg("atomics are not implemented in this backend's first pass"));
        }
        Op::Shuffle(..) | Op::Ballot(..) | Op::VoteAny(..) | Op::VoteAll(..) => {
            return Err(Diag::new(ECode::UnsupportedFeature)
                .with_arg("warp-collective ops are not implemented in this backend's first pass"));
        }
        Op::Mma { .. } => {
            return Err(
                Diag::new(ECode::UnsupportedOp).with_arg("mma has no lowering in this backend yet")
            );
        }
        _ => {}
    }
    Ok(())
}

fn successors(term: &Term) -> Vec<BlockId> {
    match term {
        Term::Br(b) => vec![*b],
        Term::CondBr(_, t, f) => vec![*t, *f],
        Term::Switch(_, default, cases) => {
            let mut v = vec![*default];
            v.extend(cases.iter().map(|&(_, b)| b));
            v
        }
        Term::Ret(_) => vec![],
    }
}

/// Follows a chain of unconditional `Br`s starting at `start`, stopping at (and including) the
/// first block whose own terminator is not `Br`. Only ever called once forward-only control
/// flow has already been confirmed (see `check_cfg`), so this always terminates.
fn chase_br_chain(f: &Function, start: BlockId) -> Vec<BlockId> {
    let mut chain = vec![start];
    let mut cur = start;
    while let Term::Br(next) = &f.blocks[cur.0 as usize].term {
        chain.push(*next);
        cur = *next;
    }
    chain
}

/// The single-level if/if-else merge-block finder shared verbatim by `check_cfg` and codegen's
/// `lower_term`, so the two can never disagree about where a `CondBr`'s arms reconverge. See
/// the module header's "Control-flow scope" section.
fn find_merge_block(f: &Function, t: BlockId, fb: BlockId) -> Option<BlockId> {
    let chain_t = chase_br_chain(f, t);
    let chain_f = chase_br_chain(f, fb);
    chain_t.into_iter().find(|b| chain_f.contains(b))
}

fn check_cfg(f: &Function) -> Result<(), Diag> {
    for (bidx, block) in f.blocks.iter().enumerate() {
        if matches!(block.term, Term::Switch(..)) {
            return Err(Diag::new(ECode::UnsupportedFeature)
                .with_arg("switch is not implemented; only br/condbr/ret are lowered"));
        }
        for succ in successors(&block.term) {
            if succ.0 as usize <= bidx {
                return Err(Diag::new(ECode::UnsupportedFeature).with_arg(
                    "loops/back-edges are not implemented; only forward, structured control flow is lowered",
                ));
            }
        }
        if let Term::CondBr(_, t, fb) = &block.term {
            if find_merge_block(f, *t, *fb).is_none() {
                return Err(Diag::new(ECode::UnsupportedFeature).with_arg(
                    "branch arms do not reconverge at a common block; only single-level structured if/if-else is lowered",
                ));
            }
        }
    }
    Ok(())
}

/// Single source of truth for what this backend refuses, shared verbatim by `supports()` and
/// `emit()` — see `basalt-ptx/src/emit.rs`'s identically-named function for why this matters.
///
/// Every function in the module becomes its own `OpEntryPoint` (see `emit_module`), exactly
/// like `basalt-ptx`'s one-kernel-per-function stance — so a non-kernel function has the same
/// live gap `basalt-ptx`'s own header documents: nothing here yet distinguishes a real
/// `__global__` kernel from a host-side function that landed in the same module.
fn check_module(module: &Module) -> Result<(), Diag> {
    for f in &module.funcs {
        if !f.is_kernel {
            return Err(Diag::new(ECode::UnsupportedFeature)
                .with_arg("host/non-kernel function compilation is not yet implemented"));
        }
        for &pty in &f.params {
            ty_in_scope(pty)?;
        }
        for inst in &f.insts {
            check_inst(inst)?;
        }
        check_cfg(f)?;
    }
    Ok(())
}

// ---- module/builtin setup -------------------------------------------------------------------

/// Wraps the `rspirv` builder plus the three real `Input` builtin variables and the one
/// three real `Input` builtin variables every kernel in the module shares (see the module
/// header's "GPU index op" section for why these three and no others). Created once per
/// `emit()` call, before any function is begun — `rspirv::dr::Builder::variable` places an
/// `OpVariable` inside whatever block is currently selected, so these must exist before any
/// `begin_block` call, not lazily from within one.
struct Ctx {
    b: Builder,
    workgroup_id: Word,
    local_invocation_id: Word,
    num_workgroups: Word,
}

impl Ctx {
    fn ty_void(&mut self) -> Word {
        self.b.type_void()
    }
    fn ty_bool(&mut self) -> Word {
        self.b.type_bool()
    }
    fn ty_uint(&mut self) -> Word {
        self.b.type_int(32, 0)
    }
    fn ty_ulong(&mut self) -> Word {
        self.b.type_int(64, 0)
    }
    fn ty_float(&mut self) -> Word {
        self.b.type_float(32, None)
    }
    fn ty_double(&mut self) -> Word {
        self.b.type_float(64, None)
    }

    /// The one SPIR-V type used to represent every SSA value of BIR type `ty`. `rspirv`'s
    /// `type_*` builder methods dedupe non-aggregate types structurally (`dedup_insert_type`),
    /// so repeated calls are both cheap and, importantly, deterministic (a linear scan over an
    /// append-only `Vec`, never a `HashMap`'s iteration order).
    fn repr_ty(&mut self, ty: Ty) -> Word {
        match ty {
            Ty::Void => self.ty_void(),
            Ty::Scalar(Scalar::I1) => self.ty_bool(),
            Ty::Scalar(Scalar::I32) => self.ty_uint(),
            Ty::Scalar(Scalar::I64) => self.ty_ulong(),
            Ty::Scalar(Scalar::F32) => self.ty_float(),
            Ty::Scalar(Scalar::F64) => self.ty_double(),
            Ty::Ptr(AddrSpace::Global) => self.ty_ulong(),
            _ => unreachable!("check_module refused this type before codegen"),
        }
    }

    fn ptr_cross_workgroup(&mut self, elem_ty: Word) -> Word {
        self.b
            .type_pointer(None, StorageClass::CrossWorkgroup, elem_ty)
    }

    /// A width-correct integer constant of BIR type `dty` (`I32` or `I64` only).
    fn const_int(&mut self, dty: Ty, value: i64) -> Word {
        match dty {
            Ty::Scalar(Scalar::I32) => {
                let t = self.ty_uint();
                self.b.constant_bit32(t, value as u32)
            }
            Ty::Scalar(Scalar::I64) => {
                let t = self.ty_ulong();
                self.b.constant_bit64(t, value as u64)
            }
            _ => unreachable!("const_int only ever called for i32/i64 Zext/Sext targets"),
        }
    }
}

/// Declares the three real `Input` builtin variables every kernel in the module shares
/// (workgroup id, local invocation id, num workgroups) — identical under the `Kernel` and
/// `GLCompute` execution models alike (confirmed via `spirv-val` for both; see this file's
/// header and `glcompute.rs`'s own header), so both execution models' context constructors
/// share this one declaration site rather than each hand-rolling it.
fn declare_index_builtins(b: &mut Builder) -> (Word, Word, Word) {
    let uint_ty = b.type_int(32, 0);
    let uint3_ty = b.type_vector(uint_ty, 3);
    let ptr_input_uint3 = b.type_pointer(None, StorageClass::Input, uint3_ty);

    let workgroup_id = b.variable(ptr_input_uint3, None, StorageClass::Input, None);
    b.decorate(
        workgroup_id,
        Decoration::BuiltIn,
        vec![Operand::BuiltIn(BuiltIn::WorkgroupId)],
    );

    let local_invocation_id = b.variable(ptr_input_uint3, None, StorageClass::Input, None);
    b.decorate(
        local_invocation_id,
        Decoration::BuiltIn,
        vec![Operand::BuiltIn(BuiltIn::LocalInvocationId)],
    );

    let num_workgroups = b.variable(ptr_input_uint3, None, StorageClass::Input, None);
    b.decorate(
        num_workgroups,
        Decoration::BuiltIn,
        vec![Operand::BuiltIn(BuiltIn::NumWorkgroups)],
    );

    (workgroup_id, local_invocation_id, num_workgroups)
}

fn new_ctx() -> Ctx {
    let mut b = Builder::new();
    b.set_version(1, 4);
    b.capability(Capability::Kernel);
    b.capability(Capability::Addresses);
    b.capability(Capability::Int64);
    b.capability(Capability::Float64);
    b.memory_model(AddressingModel::Physical64, MemoryModel::OpenCL);

    let (workgroup_id, local_invocation_id, num_workgroups) = declare_index_builtins(&mut b);

    Ctx {
        b,
        workgroup_id,
        local_invocation_id,
        num_workgroups,
    }
}

// ---- per-op lowering ----------------------------------------------------------------------

fn resolve(v: ValRef, params: &[Word], done: &[Option<Word>]) -> Word {
    match v {
        ValRef::Param(i) => params[i as usize],
        ValRef::Val(id) => {
            done[id.0 as usize].expect("operand computed before use (forward-only CFG)")
        }
    }
}

fn lower_const_int(ctx: &mut Ctx, ty: Ty, n: i64) -> Word {
    match ty {
        Ty::Scalar(Scalar::I1) => {
            let t = ctx.ty_bool();
            if n != 0 {
                ctx.b.constant_true(t)
            } else {
                ctx.b.constant_false(t)
            }
        }
        Ty::Scalar(Scalar::I32) => {
            let t = ctx.ty_uint();
            ctx.b.constant_bit32(t, n as u32)
        }
        Ty::Scalar(Scalar::I64) | Ty::Ptr(AddrSpace::Global) => {
            let t = ctx.ty_ulong();
            ctx.b.constant_bit64(t, n as u64)
        }
        _ => unreachable!("check_module refused this ConstInt type"),
    }
}

fn lower_const_float(ctx: &mut Ctx, ty: Ty, v: f64) -> Word {
    match ty {
        Ty::Scalar(Scalar::F32) => {
            let t = ctx.ty_float();
            ctx.b.constant_bit32(t, (v as f32).to_bits())
        }
        Ty::Scalar(Scalar::F64) => {
            let t = ctx.ty_double();
            ctx.b.constant_bit64(t, v.to_bits())
        }
        _ => unreachable!("check_module refused this ConstFloat type"),
    }
}

fn lower_bin(ctx: &mut Ctx, op: BinOp, ty: Ty, a: Word, b: Word) -> Word {
    let rty = ctx.repr_ty(ty);
    match ty {
        Ty::Scalar(Scalar::I1) => match op {
            BinOp::And => ctx.b.logical_and(rty, None, a, b).unwrap(),
            BinOp::Or => ctx.b.logical_or(rty, None, a, b).unwrap(),
            BinOp::Xor => ctx.b.logical_not_equal(rty, None, a, b).unwrap(),
            _ => unreachable!("check_module allows only and/or/xor on a bool-typed Bin"),
        },
        Ty::Scalar(Scalar::F32 | Scalar::F64) => match op {
            BinOp::FAdd => ctx.b.f_add(rty, None, a, b).unwrap(),
            BinOp::FSub => ctx.b.f_sub(rty, None, a, b).unwrap(),
            BinOp::FMul => ctx.b.f_mul(rty, None, a, b).unwrap(),
            BinOp::FDiv => ctx.b.f_div(rty, None, a, b).unwrap(),
            // SPIR-V's `OpFRem` is a real single instruction with truncating-remainder
            // semantics ("the sign of a nonzero result matches the sign of Operand 1"),
            // exactly BIR's own `frem` intent — no CAS-retry-loop-style emulation needed here,
            // unlike `basalt-ptx`'s `lower_frem` (PTX has no native `frem`).
            BinOp::FRem => ctx.b.f_rem(rty, None, a, b).unwrap(),
            _ => unreachable!("integer BinOp on a float-typed Bin"),
        },
        Ty::Scalar(Scalar::I32 | Scalar::I64) | Ty::Ptr(AddrSpace::Global) => match op {
            BinOp::Add => ctx.b.i_add(rty, None, a, b).unwrap(),
            BinOp::Sub => ctx.b.i_sub(rty, None, a, b).unwrap(),
            BinOp::Mul => ctx.b.i_mul(rty, None, a, b).unwrap(),
            // Signed, matching the uniform convention `basalt-ptx`/`basalt-x86` already commit
            // to: BIR's `Bin` carries no signed/unsigned distinction for div/rem.
            BinOp::Div => ctx.b.s_div(rty, None, a, b).unwrap(),
            BinOp::Rem => ctx.b.s_rem(rty, None, a, b).unwrap(),
            BinOp::And => ctx.b.bitwise_and(rty, None, a, b).unwrap(),
            BinOp::Or => ctx.b.bitwise_or(rty, None, a, b).unwrap(),
            BinOp::Xor => ctx.b.bitwise_xor(rty, None, a, b).unwrap(),
            BinOp::Shl => ctx.b.shift_left_logical(rty, None, a, b).unwrap(),
            BinOp::Lshr => ctx.b.shift_right_logical(rty, None, a, b).unwrap(),
            BinOp::Ashr => ctx.b.shift_right_arithmetic(rty, None, a, b).unwrap(),
            _ => unreachable!("float BinOp on an integer/pointer-typed Bin"),
        },
        _ => unreachable!("check_module refused this Bin result type"),
    }
}

/// Address comparisons have no sign to begin with, matching `basalt-ptx`'s identical stance:
/// a signed predicate against a `Ptr(Global)` operand is treated as its unsigned counterpart.
fn force_unsigned(pred: ICmpPred) -> ICmpPred {
    match pred {
        ICmpPred::Slt => ICmpPred::Ult,
        ICmpPred::Sle => ICmpPred::Ule,
        ICmpPred::Sgt => ICmpPred::Ugt,
        ICmpPred::Sge => ICmpPred::Uge,
        other => other,
    }
}

fn apply_icmp(ctx: &mut Ctx, pred: ICmpPred, bool_ty: Word, a: Word, b: Word) -> Word {
    match pred {
        ICmpPred::Eq => ctx.b.i_equal(bool_ty, None, a, b).unwrap(),
        ICmpPred::Ne => ctx.b.i_not_equal(bool_ty, None, a, b).unwrap(),
        ICmpPred::Slt => ctx.b.s_less_than(bool_ty, None, a, b).unwrap(),
        ICmpPred::Sle => ctx.b.s_less_than_equal(bool_ty, None, a, b).unwrap(),
        ICmpPred::Sgt => ctx.b.s_greater_than(bool_ty, None, a, b).unwrap(),
        ICmpPred::Sge => ctx.b.s_greater_than_equal(bool_ty, None, a, b).unwrap(),
        ICmpPred::Ult => ctx.b.u_less_than(bool_ty, None, a, b).unwrap(),
        ICmpPred::Ule => ctx.b.u_less_than_equal(bool_ty, None, a, b).unwrap(),
        ICmpPred::Ugt => ctx.b.u_greater_than(bool_ty, None, a, b).unwrap(),
        ICmpPred::Uge => ctx.b.u_greater_than_equal(bool_ty, None, a, b).unwrap(),
    }
}

/// `OpIEqual`/`OpSLessThan`/... require scalar *integer* operands — a `Bool` is not one (per
/// `spirv.core.grammar.json`'s own operand typing for these instructions), so an `Eq`/`Ne`
/// comparison of two bools goes through `OpLogicalEqual`/`OpLogicalNotEqual` directly, and any
/// ordered comparison of two bools first widens each to `0u`/`1u` (`OpSelect`), matching
/// `basalt-ptx`'s identical `selp`-then-`setp` two-step for the same situation.
fn lower_icmp(ctx: &mut Ctx, pred: ICmpPred, cty: Ty, a: Word, b: Word) -> Word {
    let bool_ty = ctx.ty_bool();
    match cty {
        Ty::Scalar(Scalar::I1) => match pred {
            ICmpPred::Eq => ctx.b.logical_equal(bool_ty, None, a, b).unwrap(),
            ICmpPred::Ne => ctx.b.logical_not_equal(bool_ty, None, a, b).unwrap(),
            _ => {
                let uint_ty = ctx.ty_uint();
                let one = ctx.b.constant_bit32(uint_ty, 1);
                let zero = ctx.b.constant_bit32(uint_ty, 0);
                let ua = ctx.b.select(uint_ty, None, a, one, zero).unwrap();
                let ub = ctx.b.select(uint_ty, None, b, one, zero).unwrap();
                apply_icmp(ctx, pred, bool_ty, ua, ub)
            }
        },
        Ty::Scalar(Scalar::I32 | Scalar::I64) => apply_icmp(ctx, pred, bool_ty, a, b),
        Ty::Ptr(AddrSpace::Global) => apply_icmp(ctx, force_unsigned(pred), bool_ty, a, b),
        _ => unreachable!("check_module refused this ICmp operand type"),
    }
}

fn lower_fcmp(ctx: &mut Ctx, pred: FCmpPred, bool_ty: Word, a: Word, b: Word) -> Word {
    match pred {
        FCmpPred::Oeq => ctx.b.f_ord_equal(bool_ty, None, a, b).unwrap(),
        FCmpPred::One => ctx.b.f_ord_not_equal(bool_ty, None, a, b).unwrap(),
        FCmpPred::Olt => ctx.b.f_ord_less_than(bool_ty, None, a, b).unwrap(),
        FCmpPred::Ole => ctx.b.f_ord_less_than_equal(bool_ty, None, a, b).unwrap(),
        FCmpPred::Ogt => ctx.b.f_ord_greater_than(bool_ty, None, a, b).unwrap(),
        FCmpPred::Oge => ctx.b.f_ord_greater_than_equal(bool_ty, None, a, b).unwrap(),
        FCmpPred::Ord => ctx.b.ordered(bool_ty, None, a, b).unwrap(),
        FCmpPred::Uno => ctx.b.unordered(bool_ty, None, a, b).unwrap(),
    }
}

fn lower_select(ctx: &mut Ctx, ty: Ty, cond: Word, a: Word, b: Word) -> Word {
    let rty = ctx.repr_ty(ty);
    ctx.b.select(rty, None, cond, a, b).unwrap()
}

fn lower_cast(ctx: &mut Ctx, cop: CastOp, sty: Ty, dty: Ty, src: Word) -> Word {
    let dty_repr = ctx.repr_ty(dty);
    match cop {
        CastOp::Trunc => match (sty, dty) {
            (Ty::Scalar(Scalar::I64), Ty::Scalar(Scalar::I32)) => {
                ctx.b.u_convert(dty_repr, None, src).unwrap()
            }
            (Ty::Scalar(Scalar::I32), Ty::Scalar(Scalar::I1)) => {
                let uint_ty = ctx.ty_uint();
                let one = ctx.b.constant_bit32(uint_ty, 1);
                let zero = ctx.b.constant_bit32(uint_ty, 0);
                let bit = ctx.b.bitwise_and(uint_ty, None, src, one).unwrap();
                ctx.b.i_not_equal(dty_repr, None, bit, zero).unwrap()
            }
            _ => unreachable!("check_module scopes Trunc to i64->i32 or i32->i1"),
        },
        CastOp::Zext => match (sty, dty) {
            (Ty::Scalar(Scalar::I1), _) => {
                let one = ctx.const_int(dty, 1);
                let zero = ctx.const_int(dty, 0);
                ctx.b.select(dty_repr, None, src, one, zero).unwrap()
            }
            (Ty::Scalar(Scalar::I32), Ty::Scalar(Scalar::I64)) => {
                ctx.b.u_convert(dty_repr, None, src).unwrap()
            }
            _ => unreachable!("check_module scopes Zext to i1->i32/i64 or i32->i64"),
        },
        CastOp::Sext => match (sty, dty) {
            (Ty::Scalar(Scalar::I1), _) => {
                let neg1 = ctx.const_int(dty, -1);
                let zero = ctx.const_int(dty, 0);
                ctx.b.select(dty_repr, None, src, neg1, zero).unwrap()
            }
            (Ty::Scalar(Scalar::I32), Ty::Scalar(Scalar::I64)) => {
                ctx.b.s_convert(dty_repr, None, src).unwrap()
            }
            _ => unreachable!("check_module scopes Sext to i1->i32/i64 or i32->i64"),
        },
        // F64->F32 and F32->F64 both go through the one generic (possibly-narrowing or
        // possibly-widening) `OpFConvert`.
        CastOp::FpTrunc | CastOp::FpExt => ctx.b.f_convert(dty_repr, None, src).unwrap(),
        CastOp::FpToSi => ctx.b.convert_f_to_s(dty_repr, None, src).unwrap(),
        CastOp::FpToUi => ctx.b.convert_f_to_u(dty_repr, None, src).unwrap(),
        CastOp::SiToFp => ctx.b.convert_s_to_f(dty_repr, None, src).unwrap(),
        CastOp::UiToFp => ctx.b.convert_u_to_f(dty_repr, None, src).unwrap(),
        CastOp::Bitcast => ctx.b.bitcast(dty_repr, None, src).unwrap(),
    }
}

fn memory_access(volatile: bool) -> Option<MemoryAccess> {
    if volatile {
        Some(MemoryAccess::VOLATILE)
    } else {
        None
    }
}

fn lower_load(ctx: &mut Ctx, ty: Ty, raw_addr: Word, volatile: bool) -> Word {
    let elem_ty = ctx.repr_ty(ty);
    let ptr_ty = ctx.ptr_cross_workgroup(elem_ty);
    let typed_ptr = ctx.b.convert_u_to_ptr(ptr_ty, None, raw_addr).unwrap();
    ctx.b
        .load(elem_ty, None, typed_ptr, memory_access(volatile), vec![])
        .unwrap()
}

fn lower_store(ctx: &mut Ctx, ty: Ty, raw_addr: Word, val: Word, volatile: bool) {
    let elem_ty = ctx.repr_ty(ty);
    let ptr_ty = ctx.ptr_cross_workgroup(elem_ty);
    let typed_ptr = ctx.b.convert_u_to_ptr(ptr_ty, None, raw_addr).unwrap();
    ctx.b
        .store(typed_ptr, val, memory_access(volatile), vec![])
        .unwrap();
}

fn lower_barrier(ctx: &mut Ctx) {
    let uint_ty = ctx.ty_uint();
    let scope = ctx.b.constant_bit32(uint_ty, Scope::Workgroup as u32);
    let semantics = ctx.b.constant_bit32(
        uint_ty,
        (MemorySemantics::ACQUIRE_RELEASE | MemorySemantics::WORKGROUP_MEMORY).bits(),
    );
    ctx.b.control_barrier(scope, scope, semantics).unwrap();
}

/// Reads one component of a genuine `Input`-storage-class builtin (`WorkgroupId`/
/// `LocalInvocationId`/`NumWorkgroups`) — a fresh `OpLoad` of the whole `uvec3` plus an
/// `OpCompositeExtract`, every time (no caching: correct first, matching this project's
/// standing "ship a slow correct backend" priority order). Records `var` in `used` so the
/// caller's `OpEntryPoint` interface list includes it.
fn lower_gpu_index(ctx: &mut Ctx, var: Word, axis: u32, used: &mut Vec<Word>) -> Word {
    if !used.contains(&var) {
        used.push(var);
    }
    let uint_ty = ctx.ty_uint();
    let uint3_ty = ctx.b.type_vector(uint_ty, 3);
    let vec = ctx.b.load(uint3_ty, None, var, None, vec![]).unwrap();
    ctx.b
        .composite_extract(uint_ty, None, vec, vec![axis])
        .unwrap()
}

/// `blockDim` under this backend's fixed work-group size of `(1, 1, 1)` (see the module
/// header): a plain compile-time constant, not a hardware read — there is nothing to load, so
/// unlike `lower_gpu_index` this never touches a builtin variable at all. `axis` is accepted
/// (and ignored) only so every `Op::BdimX/Y/Z` call site looks like its `Op::TidX/Y/Z`/
/// `Op::BidX/Y/Z`/`Op::GdimX/Y/Z` siblings.
fn lower_bdim(ctx: &mut Ctx, _axis: u32) -> Word {
    ctx.const_int(Ty::Scalar(Scalar::I32), 1)
}

fn lower_inst(
    ctx: &mut Ctx,
    inst: &Inst,
    labels: &[Word],
    params: &[Word],
    done: &[Option<Word>],
    used_builtins: &mut Vec<Word>,
) -> Option<Word> {
    let ty = inst.ty;
    let r = |v: ValRef| resolve(v, params, done);
    match &inst.op {
        Op::ConstInt(n) => Some(lower_const_int(ctx, ty, *n)),
        Op::ConstFloat(v) => Some(lower_const_float(ctx, ty, *v)),
        Op::Bin(op, a, b) => Some(lower_bin(ctx, *op, ty, r(*a), r(*b))),
        Op::ICmp(pred, cty, a, b) => Some(lower_icmp(ctx, *pred, *cty, r(*a), r(*b))),
        Op::FCmp(pred, _cty, a, b) => {
            let bool_ty = ctx.ty_bool();
            Some(lower_fcmp(ctx, *pred, bool_ty, r(*a), r(*b)))
        }
        Op::Select(c, a, b) => Some(lower_select(ctx, ty, r(*c), r(*a), r(*b))),
        Op::Cast(cop, sty, v) => Some(lower_cast(ctx, *cop, *sty, ty, r(*v))),
        Op::Load { ptr, volatile, .. } => Some(lower_load(ctx, ty, r(*ptr), *volatile)),
        Op::Store {
            ptr,
            val,
            ty: sty,
            volatile,
            ..
        } => {
            lower_store(ctx, *sty, r(*ptr), r(*val), *volatile);
            None
        }
        Op::Phi(preds) => {
            let rty = ctx.repr_ty(ty);
            let operands: Vec<(Word, Word)> = preds
                .iter()
                .map(|&(bb, v)| (resolve(v, params, done), labels[bb.0 as usize]))
                .collect();
            Some(ctx.b.phi(rty, None, operands).unwrap())
        }
        Op::TidX => {
            let v = ctx.local_invocation_id;
            Some(lower_gpu_index(ctx, v, 0, used_builtins))
        }
        Op::TidY => {
            let v = ctx.local_invocation_id;
            Some(lower_gpu_index(ctx, v, 1, used_builtins))
        }
        Op::TidZ => {
            let v = ctx.local_invocation_id;
            Some(lower_gpu_index(ctx, v, 2, used_builtins))
        }
        Op::BidX => {
            let v = ctx.workgroup_id;
            Some(lower_gpu_index(ctx, v, 0, used_builtins))
        }
        Op::BidY => {
            let v = ctx.workgroup_id;
            Some(lower_gpu_index(ctx, v, 1, used_builtins))
        }
        Op::BidZ => {
            let v = ctx.workgroup_id;
            Some(lower_gpu_index(ctx, v, 2, used_builtins))
        }
        Op::GdimX => {
            let v = ctx.num_workgroups;
            Some(lower_gpu_index(ctx, v, 0, used_builtins))
        }
        Op::GdimY => {
            let v = ctx.num_workgroups;
            Some(lower_gpu_index(ctx, v, 1, used_builtins))
        }
        Op::GdimZ => {
            let v = ctx.num_workgroups;
            Some(lower_gpu_index(ctx, v, 2, used_builtins))
        }
        Op::BdimX => Some(lower_bdim(ctx, 0)),
        Op::BdimY => Some(lower_bdim(ctx, 1)),
        Op::BdimZ => Some(lower_bdim(ctx, 2)),
        Op::Barrier => {
            lower_barrier(ctx);
            None
        }
        Op::Atomic(..)
        | Op::AtomicCas(..)
        | Op::Shuffle(..)
        | Op::Ballot(..)
        | Op::VoteAny(..)
        | Op::VoteAll(..)
        | Op::Mma { .. } => {
            unreachable!("check_module refuses this op before codegen starts")
        }
    }
}

fn lower_term(
    ctx: &mut Ctx,
    f: &Function,
    bidx: usize,
    labels: &[Word],
    params: &[Word],
    done: &[Option<Word>],
) {
    match &f.blocks[bidx].term {
        Term::Br(target) => {
            ctx.b.branch(labels[target.0 as usize]).unwrap();
        }
        Term::CondBr(cond, t, fb) => {
            let merge = find_merge_block(f, *t, *fb).expect("checked by check_module's check_cfg");
            ctx.b
                .selection_merge(labels[merge.0 as usize], SelectionControl::NONE)
                .unwrap();
            let c = resolve(*cond, params, done);
            ctx.b
                .branch_conditional(c, labels[t.0 as usize], labels[fb.0 as usize], vec![])
                .unwrap();
        }
        Term::Switch(..) => unreachable!("check_module refuses switch before codegen starts"),
        Term::Ret(_) => {
            // A `Kernel`-model entry point returns nothing back to the host; any value a
            // `Ret` carries is dropped, matching `basalt-ptx`'s identical, documented stance
            // for the same reason (a kernel entry point has nowhere honest to hand it).
            ctx.b.ret().unwrap();
        }
    }
}

// ---- module/function assembly ------------------------------------------------------------

fn lower_function(ctx: &mut Ctx, f: &Function) {
    let param_tys: Vec<Word> = f
        .params
        .iter()
        .map(|&ty| match ty {
            Ty::Ptr(AddrSpace::Global) => {
                let u = ctx.ty_uint();
                ctx.ptr_cross_workgroup(u)
            }
            other => ctx.repr_ty(other),
        })
        .collect();

    let void_ty = ctx.ty_void();
    let fn_ty = ctx.b.type_function(void_ty, param_tys.clone());
    let fn_id = ctx
        .b
        .begin_function(void_ty, None, FunctionControl::NONE, fn_ty)
        .unwrap();
    ctx.b.name(fn_id, f.name.clone());

    let formal_params: Vec<Word> = param_tys
        .iter()
        .map(|&ty| ctx.b.function_parameter(ty).unwrap())
        .collect();

    // Pre-allocate every block's label id up front: a `CondBr`/`Br`/`Phi` in an earlier block
    // may need to name a later block's label before that block has been visited.
    let labels: Vec<Word> = (0..f.blocks.len()).map(|_| ctx.b.id()).collect();

    let mut used_builtins: Vec<Word> = Vec::new();
    let mut done: Vec<Option<Word>> = vec![None; f.insts.len()];
    let mut param_values: Vec<Word> = Vec::new();

    for bidx in 0..f.blocks.len() {
        ctx.b.begin_block(Some(labels[bidx])).unwrap();
        if bidx == 0 {
            // Convert every `Ptr(Global)` parameter to its working raw-address representation
            // once, at the top of the entry block — the one place a real SPIR-V pointer object
            // for a parameter is ever observed (see the module header's "Pointer representation").
            param_values = f
                .params
                .iter()
                .zip(formal_params.iter())
                .map(|(&ty, &formal)| match ty {
                    Ty::Ptr(AddrSpace::Global) => {
                        let ulong_ty = ctx.ty_ulong();
                        ctx.b.convert_ptr_to_u(ulong_ty, None, formal).unwrap()
                    }
                    _ => formal,
                })
                .collect();
        }
        for &inst_id in &f.blocks[bidx].insts {
            let inst = &f.insts[inst_id.0 as usize];
            let result = lower_inst(ctx, inst, &labels, &param_values, &done, &mut used_builtins);
            done[inst_id.0 as usize] = result;
        }
        lower_term(ctx, f, bidx, &labels, &param_values, &done);
    }

    ctx.b.end_function().unwrap();

    ctx.b
        .entry_point(ExecutionModel::Kernel, fn_id, f.name.clone(), used_builtins);
    ctx.b
        .execution_mode(fn_id, ExecutionMode::LocalSize, [1u32, 1, 1]);
}

fn emit_module(module: &Module) -> Vec<u8> {
    let mut ctx = new_ctx();
    for f in &module.funcs {
        lower_function(&mut ctx, f);
    }
    let spirv_module = ctx.b.module();
    let words = Assemble::assemble(&spirv_module);
    let mut bytes = Vec::with_capacity(words.len() * 4);
    for w in words {
        bytes.extend_from_slice(&w.to_le_bytes());
    }
    bytes
}

// ---- Backend impl -------------------------------------------------------------------------

/// The SPIR-V backend. `name()` returns `"spirv"`, matching the target id already declared in
/// `scripts/targets.tsv`. See the module header for the full design.
#[derive(Debug, Default, Clone, Copy)]
pub struct Spirv;

impl Backend for Spirv {
    fn name(&self) -> &'static str {
        "spirv"
    }

    fn supports(&self, module: &Module) -> Support {
        // `Support::supports` has no `EmitOpts` of its own to read (the trait signature is
        // shared by every backend, see `basalt-backend`), so it only ever checks this backend's
        // default (`Kernel`) coverage, matching `basalt-amdgpu`'s identical stance on its own
        // `EmitOpts::target_variant` — `glcompute`-specific coverage (the resource-binding ABI,
        // the recognized address-computation shape) is checked in `emit`, which does receive
        // `EmitOpts`, via `glcompute::check_module_glcompute`.
        match check_module(module) {
            Ok(()) => Support::Supported,
            Err(diag) => Support::Unsupported(diag.code),
        }
    }

    fn emit(&self, module: &Module, opts: &EmitOpts) -> Result<Artifact, Diag> {
        if glcompute::is_glcompute_variant(opts) {
            return glcompute::emit_glcompute(module);
        }
        check_module(module)?;
        let ssa_module = construct_ssa(module);
        let bytes = emit_module(&ssa_module);
        Ok(Artifact::bytes(ArtifactKind::SpirV, bytes))
    }
}

mod glcompute;

#[cfg(test)]
mod tests;
