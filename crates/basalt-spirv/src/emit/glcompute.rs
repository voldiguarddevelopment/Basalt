// The second, opt-in emission path this crate offers: `GLCompute` execution model, `Logical`
// addressing, `GLSL450` memory model — selected by `EmitOpts::target_variant == Some("glcompute")`
// (`None`, or any other string, keeps the parent module's `Kernel` path exactly as documented in
// its own header; see `super`'s "A second path" section). This module closes the two gaps that
// header, and `basalt-runtime`'s own `src/vulkan/mod.rs` finding, already named as the reason
// `basalt-spirv` targeted `Kernel` in the first place:
//
//   1. a real resource-binding ABI (which BIR pointer parameter becomes which descriptor-set
//      binding; how scalar parameters pack into a push-constant block), and
//   2. a real recognizer that turns BIR's byte-offset pointer arithmetic back into `Logical`
//      addressing's `OpAccessChain` element indices, for the one shape `basalt-sema` ever
//      produces for indexed access — not a general static-analysis problem, a mechanical one.
//
// # Resource-binding ABI
//
// Reuses, verbatim, the ABI `crates/basalt-runtime/tests/vulkan_gpu_proof.rs`'s hand-written
// `GLCompute` stand-in shader already defines and real-hardware-proves (see that file's own doc
// comment on `GLCOMPUTE_VECTOR_ADD_SPV`, and `basalt-runtime/src/vulkan/pipeline.rs`'s header):
//
//   - Every `Ty::Ptr(AddrSpace::Global)` function parameter becomes one
//     `VK_DESCRIPTOR_TYPE_STORAGE_BUFFER` binding at `set = 0`, bound in declared parameter
//     order starting at `binding = 0` — a `layout(set = 0, binding = N) buffer { T x[]; }` block,
//     one array-of-`T` member each, `T` being whatever scalar type this backend's own accesses to
//     that parameter turn out to use (see "Element type inference" below).
//   - Every non-pointer (scalar) function parameter is packed, in declared order, into a single
//     `layout(push_constant)` block, natural size/alignment (`round_up(cursor, size)`, 4 bytes
//     for `i32`/`f32`, 8 for `i64`/`f64`) — the same convention `basalt-amdgpu`'s own
//     `kernarg_layout` already documents and checks for its own kernarg-buffer packing.
//   - Every pointer parameter is required to precede every scalar parameter, matching
//     `basalt-sema`'s own lowering order and `kernarg_layout`'s identical, already-checked
//     assumption — `resource_layout` below enforces this the same way, refusing (`E093`)
//     otherwise rather than silently reordering.
//
// `resource_layout` is this ABI's one implementation, shared by `check_module_glcompute` (the
// pre-flight refusal) and `emit_glcompute` (the actual binding/push-constant construction) —
// the two can never disagree about which parameter is which binding.
//
// # Pointer-arithmetic-to-`OpAccessChain` recognition
//
// `basalt-sema/src/lower.rs`'s `lower_index_lvalue` (confirmed by reading it directly, and by
// dumping this project's own real post-optimize BIR for `tests/kernels/vector_add.cu` — see
// `emit/tests.rs`'s `vector_add_glcompute_*` tests) emits exactly one shape for every indexed
// memory access, never a different one:
//
// ```text
// %idx64  = sext i64 i32 %index        ; or: already i64, or a bare i64 parameter
// %stride = const.i i64 <esz>          ; <esz> = the accessed type's real byte size
// %off    = mul i64 %idx64, %stride
// %addr   = add ptr.global %base, %off ; %base is always the pointer parameter directly
// ```
//
// `recognize_access` walks exactly this shape from the outside in (the `Bin::Add`, then its
// `Bin::Mul` offset operand, then that `Mul`'s `ConstInt` stride operand) and returns the
// `(base parameter, element index)` pair an `OpAccessChain` needs — refusing (`E093`) at the
// first operand that does not match, never guessing at what an unrecognized shape might mean.
// `build_addr_map` runs this once per function, keyed by the outer `Bin::Add`'s own `InstId`
// (the *only* instruction this recognition ever "consumes" — see "What actually gets skipped"
// below), and additionally refuses (same `E093`) any `Ty::Ptr(Global)`-typed instruction that
// is *not* the outer `Add` of some recognized access (an address compared, selected between,
// merged by a `Phi`, or otherwise produced outside this one shape) and any `ICmp` comparing two
// raw addresses — `Logical` addressing has no representation for a raw pointer value at all
// outside this one recognized, immediately-consumed shape, so every other use is a clean
// refusal, not a guess. `check_module_glcompute` and `emit_glcompute`'s codegen both call
// `build_addr_map`/`resource_layout`/`resolve_binding_elem_tys` — the same functions, not
// parallel copies of the same pattern-matching — so the two can never drift apart, the same
// discipline `basalt-amdgpu`'s `find_merge_block`/`build_regalloc` already hold themselves to.
//
// ## What actually gets skipped
//
// Only the outer `Bin::Add` (the one instruction whose *result* is `Ty::Ptr(Global)`) is ever
// skipped during ordinary per-instruction codegen — it has no representation under `Logical`
// addressing (there is no raw address to convert it to, unlike the `Kernel` path's
// `OpConvertPtrToU`/`OpConvertUToPtr` pair), so the `Load`/`Store` that consumes it builds a real
// `OpAccessChain` directly in its place. The `Bin::Mul` and `ConstInt` stride instructions
// feeding that `Add` are ordinary `i64` values — this backend's shared `lower_bin`/
// `lower_const_int` (see `super`) lower them completely normally, producing a real (if, from
// this path's perspective, unused) byte-offset value; this is simpler and no less correct than
// hunting down and separately suppressing them, matching this project's standing "correct first"
// priority over minimizing instruction count.
//
// # Element type inference
//
// BIR's `Ty::Ptr` carries no pointee type (see `super`'s "Pointer representation" section), so
// the one scalar type each storage-buffer binding's `layout(...) buffer { T x[]; }` declares is
// derived, not assumed: `resolve_binding_elem_tys` collects, across every recognized access,
// which scalar type each pointer parameter was actually loaded/stored at, and refuses (`E093`)
// a pointer parameter accessed at more than one type (this backend cannot declare two element
// types for one binding) or never accessed at all in a recognized shape (nothing to derive from,
// so nothing is guessed).
//
// # Version, capabilities, memory model, and storage classes
//
// `SPIR-V 1.0`, capability `Shader` (plus `Int64`/`Float64`, needed by this backend's internal
// `i64` byte-offset arithmetic regardless of execution model — see `super`'s identical,
// unconditional declaration of the same two capabilities), `Logical` addressing, `GLSL450`
// memory model. Storage buffers use the `Uniform` storage class with a `BufferBlock`-decorated
// wrapping struct (the pre-1.3, SPIR-V-1.0-legal SSBO shape) rather than the newer
// `StorageBuffer` storage class + `Block` decoration — deliberately, because it is the exact
// shape `tests/vulkan_gpu_proof.rs`'s hand-written stand-in shader already used and
// real-hardware-proved on this project's own `llvmpipe` test machine (see that file's own
// `GLCOMPUTE_VECTOR_ADD_SPV` doc comment), not a novel, unverified choice. `OpEntryPoint`'s
// interface list therefore only needs the `Input`-class builtin variables (the pre-1.4 rule —
// see `super`'s own header for the identical, version-driven interface-list reasoning), exactly
// mirroring the `Kernel` path's `used_builtins` bookkeeping unchanged.
//
// An `OpAccessChain`'s element index is always truncated (`OpUConvert`) to 32 bits before use,
// regardless of the `i64` representation `recognize_access` confirms every recognized index
// already has: this matches the hand-proved stand-in shader's own `uint`-indexed convention
// exactly, rather than leaving an open question about whether a real Vulkan implementation
// accepts a 64-bit `OpAccessChain` index under the `Shader` capability profile.
//
// # Fixed work-group size, GPU-index-op mapping, and control flow: unchanged, shared verbatim
//
// This path declares the same fixed `(1, 1, 1)` `LocalSize` `super` documents and justifies (see
// its "GPU index op -> BuiltIn mapping" section) — `Op::TidX/Y/Z`, `Op::BidX/Y/Z`, `Op::GdimX/Y/Z`,
// and `Op::BdimX/Y/Z` all lower via the exact same `lower_gpu_index`/`lower_bdim` this file's
// parent module already defines, and `spirv-val` confirms the same three `Input` builtins
// (`WorkgroupId`/`LocalInvocationId`/`NumWorkgroups`) are legal, with the same semantics, under
// `GLCompute` as under `Kernel` — not assumed identical, checked. Single-level structured
// if/if-else (`find_merge_block`/`check_cfg`), `OpPhi`, and `OpControlBarrier` are unchanged and
// shared for the identical reason: `Logical` addressing changes nothing about how SPIR-V
// structures branches, merges phis, or barriers a workgroup.
//
// # Refusal surface (in addition to `super`'s, which this path inherits unchanged via the shared
// `check_inst`/`check_cfg`)
//
// - A function parameter appearing after a scalar parameter has already been seen (`E093`,
//   `resource_layout`) — the one ordering assumption this ABI, and `basalt-amdgpu`'s
//   `kernarg_layout`, both make and check.
// - A scalar function parameter whose type is not `i32`/`i64`/`f32`/`f64` (`E091`,
//   `resource_layout`) — includes `i1`: a push-constant-packed bool has no defined
//   representation this backend is willing to guess at, matching `super`'s identical stance on
//   `i1` loads/stores.
// - A pointer function parameter in any address space other than `Global` (`E092`,
//   `resource_layout`).
// - Any `Load`/`Store` address that does not match `recognize_access`'s one recognized shape, at
//   the first operand that fails to match (`E093`).
// - Any `Ty::Ptr(Global)`-typed instruction that is not the outer `Add` of a recognized access,
//   and any `ICmp` comparing two addresses (`E093`, `build_addr_map`) — see "Pointer-arithmetic-
//   to-`OpAccessChain` recognition" above.
// - A storage-buffer binding accessed at more than one element type, or never accessed in a
//   recognized shape at all (`E093`, `resolve_binding_elem_tys`).
//
// # Validation tier: real-validator-confirmed (`spirv-val`) plus real Vulkan dispatch
//
// `spirv-val` (the same SPIRV-Tools install `super`'s own header names) was run directly against
// this path's real emitted bytes for the real frontend/sema/passes/emit pipeline over
// `tests/kernels/vector_add.cu`, with no diagnostics, against both the generic and
// `--target-env vulkan1.0`/`vulkan1.1` environments. Beyond that — beyond what `super`'s `Kernel`
// path can claim, since P9-T2 found real Vulkan pipeline creation refuses that path outright —
// this path's `vector_add.cu` output was also actually dispatched through `basalt-runtime`'s real
// Vulkan compute runtime
// (`vector_add_dispatches_through_real_vulkan_runtime_via_real_glcompute_backend` in
// `crates/basalt-runtime/tests/vulkan_gpu_proof.rs`) against real `llvmpipe`, and its result
// compared bit-exact against the host-computed oracle result. See that test for the exact
// numbers.

use rspirv::binary::Assemble;
use rspirv::dr::{Builder, Operand};
use rspirv::spirv::{
    AddressingModel, Capability, Decoration, ExecutionMode, ExecutionModel, FunctionControl,
    MemoryModel, StorageClass, Word,
};

use basalt_backend::{Artifact, ArtifactKind, EmitOpts};
use basalt_bir::{AddrSpace, BinOp, Function, InstId, Module, Op, Scalar, Ty, ValRef};
use basalt_diag::{Diag, ECode};
use basalt_passes::construct_ssa;

use super::{
    check_cfg, check_inst, declare_index_builtins, lower_inst, lower_term, memory_access, resolve,
    Ctx,
};

/// The one string that selects this path; anything else (including `None`) keeps `super`'s
/// `Kernel` path exactly as documented there.
const VARIANT_NAME: &str = "glcompute";

pub(super) fn is_glcompute_variant(opts: &EmitOpts) -> bool {
    opts.target_variant.as_deref() == Some(VARIANT_NAME)
}

// ---- resource-binding ABI --------------------------------------------------------------------

/// One storage-buffer binding per pointer parameter (index into `Function::params`, in ascending
/// `binding` order — binding `N` is `ptr_params[N]`), plus `(param index, byte offset, byte
/// size)` for every scalar parameter packed into the single push-constant block, in declared
/// order. See this file's header, "Resource-binding ABI".
struct ResourceLayout {
    ptr_params: Vec<usize>,
    push_const: Vec<(usize, u32, u32)>,
}

/// The one real byte size this path is willing to give a scalar type — `None` for anything it
/// has no defined buffer-element or push-constant-member representation for (matches `super`'s
/// own `i8`/`i16`/`f16`/`i1`/vector refusals; a pointer is never a legal element type either,
/// since `Logical` addressing has nothing to store a raw address as — see "Pointer-arithmetic-
/// to-`OpAccessChain` recognition").
fn elem_byte_size(ty: Ty) -> Option<i64> {
    match ty {
        Ty::Scalar(Scalar::I32 | Scalar::F32) => Some(4),
        Ty::Scalar(Scalar::I64 | Scalar::F64) => Some(8),
        _ => None,
    }
}

fn e_addr(reason: impl Into<String>) -> Diag {
    Diag::new(ECode::UnsupportedFeature).with_arg(reason.into())
}

fn resource_layout(params: &[Ty]) -> Result<ResourceLayout, Diag> {
    let mut ptr_params = Vec::new();
    let mut push_const = Vec::new();
    let mut cursor: u32 = 0;
    let mut seen_scalar = false;
    for (i, &ty) in params.iter().enumerate() {
        match ty {
            Ty::Ptr(AddrSpace::Global) => {
                if seen_scalar {
                    return Err(e_addr(
                        "every pointer parameter must precede every scalar parameter, matching \
                         basalt-sema's own lowering order and basalt-amdgpu's kernarg_layout's \
                         identical, already-checked assumption",
                    ));
                }
                ptr_params.push(i);
            }
            Ty::Ptr(_) => {
                return Err(Diag::new(ECode::UnsupportedAddressSpace)
                    .with_arg("only AddrSpace::Global pointer parameters are implemented"));
            }
            _ => {
                let size = elem_byte_size(ty).ok_or_else(|| {
                    Diag::new(ECode::UnsupportedType).with_arg(
                        "only i32/i64/f32/f64 scalar parameters are packed into the \
                         push-constant block",
                    )
                })? as u32;
                seen_scalar = true;
                let offset = cursor.div_ceil(size) * size;
                push_const.push((i, offset, size));
                cursor = offset + size;
            }
        }
    }
    Ok(ResourceLayout {
        ptr_params,
        push_const,
    })
}

// ---- pointer-arithmetic-to-access-chain recognition ------------------------------------------

/// One recognized `Load`/`Store` address: which pointer parameter it indexes into, and the
/// element index to hand `OpAccessChain`. See this file's header for the exact shape this is
/// read back out of.
#[derive(Clone, Copy)]
struct RecognizedAccess {
    param: usize,
    index: ValRef,
    elem_ty: Ty,
}

/// Recognizes `addr` as `Bin::Add(ptr_param, Bin::Mul(index, ConstInt(stride)))` where `stride`
/// equals `accessed_ty`'s real byte size — the one shape `basalt-sema`'s `lower_index_lvalue`
/// ever produces (see this file's header) — refusing at the first operand that does not match.
/// Returns the outer `Add`'s own `InstId` alongside the recognized access so the caller can key
/// its consumed-instruction map by it.
fn recognize_access(
    f: &Function,
    addr: ValRef,
    accessed_ty: Ty,
) -> Result<(InstId, RecognizedAccess), Diag> {
    let stride = elem_byte_size(accessed_ty).ok_or_else(|| {
        e_addr("the accessed type has no defined storage-buffer-element representation")
    })?;
    let add_id = match addr {
        ValRef::Val(id) => id,
        ValRef::Param(_) => {
            return Err(e_addr(
                "address is a bare function parameter, not an index computation",
            ))
        }
    };
    let add_inst = &f.insts[add_id.0 as usize];
    if add_inst.ty != Ty::Ptr(AddrSpace::Global) {
        return Err(e_addr(
            "address instruction's own result type is not Ptr(Global)",
        ));
    }
    let (base, offset) = match &add_inst.op {
        Op::Bin(BinOp::Add, a, b) => (*a, *b),
        _ => return Err(e_addr("address is not a Bin::Add")),
    };
    let param = match base {
        ValRef::Param(p) if matches!(f.params[p as usize], Ty::Ptr(AddrSpace::Global)) => {
            p as usize
        }
        _ => {
            return Err(e_addr(
                "Bin::Add's base operand is not a Ty::Ptr(Global) function parameter",
            ))
        }
    };
    let mul_id = match offset {
        ValRef::Val(id) => id,
        ValRef::Param(_) => {
            return Err(e_addr(
                "Bin::Add's offset operand is not a computed Bin::Mul",
            ))
        }
    };
    let mul_inst = &f.insts[mul_id.0 as usize];
    if mul_inst.ty != Ty::Scalar(Scalar::I64) {
        return Err(e_addr("byte-offset instruction is not i64-typed"));
    }
    let (index, stride_ref) = match &mul_inst.op {
        Op::Bin(BinOp::Mul, a, b) => (*a, *b),
        _ => return Err(e_addr("Bin::Add's offset operand is not a Bin::Mul")),
    };
    let stride_val = match stride_ref {
        ValRef::Val(id) => {
            let inst = &f.insts[id.0 as usize];
            match inst.op {
                Op::ConstInt(n) if inst.ty == Ty::Scalar(Scalar::I64) => n,
                _ => return Err(e_addr("Bin::Mul's stride operand is not an i64 ConstInt")),
            }
        }
        ValRef::Param(_) => return Err(e_addr("Bin::Mul's stride operand is not an i64 ConstInt")),
    };
    if stride_val != stride {
        return Err(e_addr(
            "stride does not match the accessed type's real byte size",
        ));
    }
    let index_ty = match index {
        ValRef::Param(p) => f.params[p as usize],
        ValRef::Val(id) => f.insts[id.0 as usize].ty,
    };
    if index_ty != Ty::Scalar(Scalar::I64) {
        return Err(e_addr("index operand is not i64-typed"));
    }
    Ok((
        add_id,
        RecognizedAccess {
            param,
            index,
            elem_ty: accessed_ty,
        },
    ))
}

/// Runs `recognize_access` at every `Load`/`Store` in `f`, keyed by the outer `Add`'s own
/// `InstId` (see "What actually gets skipped" in this file's header), then refuses any
/// `Ty::Ptr(Global)`-typed instruction that is not one of those recognized `Add`s and any `ICmp`
/// comparing two addresses — the only other ways a raw pointer value could otherwise need a
/// representation this backend does not have. Shared, verbatim, by `check_module_glcompute` and
/// `emit_glcompute`'s codegen.
fn build_addr_map(f: &Function) -> Result<Vec<Option<RecognizedAccess>>, Diag> {
    let mut map: Vec<Option<RecognizedAccess>> = vec![None; f.insts.len()];
    for inst in &f.insts {
        match &inst.op {
            Op::Load { ptr, .. } => {
                let (add_id, acc) = recognize_access(f, *ptr, inst.ty)?;
                map[add_id.0 as usize] = Some(acc);
            }
            Op::Store { ptr, ty, .. } => {
                let (add_id, acc) = recognize_access(f, *ptr, *ty)?;
                map[add_id.0 as usize] = Some(acc);
            }
            Op::ICmp(_, Ty::Ptr(_), ..) => {
                return Err(e_addr(
                    "comparing raw addresses has no representation under Logical addressing",
                ));
            }
            _ => {}
        }
    }
    for (idx, inst) in f.insts.iter().enumerate() {
        if matches!(inst.ty, Ty::Ptr(AddrSpace::Global)) && map[idx].is_none() {
            return Err(e_addr(
                "a pointer-typed value outside a recognized element-index computation has no \
                 representation under Logical addressing",
            ));
        }
    }
    Ok(map)
}

/// The one real element type each storage-buffer binding declares, derived from every recognized
/// access to it (see this file's header, "Element type inference"). Shared, verbatim, by
/// `check_module_glcompute` and the codegen that actually builds each binding's SPIR-V type.
fn resolve_binding_elem_tys(
    layout: &ResourceLayout,
    addr_map: &[Option<RecognizedAccess>],
) -> Result<Vec<Ty>, Diag> {
    let mut elem_ty: Vec<Option<Ty>> = vec![None; layout.ptr_params.len()];
    for acc in addr_map.iter().flatten() {
        let binding = layout
            .ptr_params
            .iter()
            .position(|&p| p == acc.param)
            .expect("recognize_access only ever names a parameter resource_layout already placed in ptr_params");
        match elem_ty[binding] {
            None => elem_ty[binding] = Some(acc.elem_ty),
            Some(prev) if prev == acc.elem_ty => {}
            Some(_) => {
                return Err(e_addr(
                    "the same storage-buffer binding is accessed at more than one element type",
                ))
            }
        }
    }
    elem_ty
        .into_iter()
        .enumerate()
        .map(|(i, t)| {
            t.ok_or_else(|| {
                e_addr(format!(
                    "pointer parameter {} (storage-buffer binding {i}) is never accessed in a \
                     recognized shape, so this backend cannot determine its element type",
                    layout.ptr_params[i]
                ))
            })
        })
        .collect()
}

// ---- module-level checks ----------------------------------------------------------------------

/// Single source of truth for what this path refuses, mirroring `super::check_module`'s role for
/// the `Kernel` path — shared verbatim by `Spirv::emit`'s `glcompute` branch (via
/// `emit_glcompute`) and by tests that want to assert a refusal without touching codegen.
pub(super) fn check_module_glcompute(module: &Module) -> Result<(), Diag> {
    for f in &module.funcs {
        if !f.is_kernel {
            return Err(Diag::new(ECode::UnsupportedFeature)
                .with_arg("host/non-kernel function compilation is not yet implemented"));
        }
        let layout = resource_layout(&f.params)?;
        for inst in &f.insts {
            check_inst(inst)?;
        }
        check_cfg(f)?;
        let addr_map = build_addr_map(f)?;
        resolve_binding_elem_tys(&layout, &addr_map)?;
    }
    Ok(())
}

// ---- module/context setup ----------------------------------------------------------------------

fn new_ctx_glcompute() -> Ctx {
    let mut b = Builder::new();
    b.set_version(1, 0);
    b.capability(Capability::Shader);
    b.capability(Capability::Int64);
    b.capability(Capability::Float64);
    b.memory_model(AddressingModel::Logical, MemoryModel::GLSL450);

    let (workgroup_id, local_invocation_id, num_workgroups) = declare_index_builtins(&mut b);

    Ctx {
        b,
        workgroup_id,
        local_invocation_id,
        num_workgroups,
    }
}

/// One `layout(set = 0, binding = N) buffer { T x[]; }` block: an `OpTypeRuntimeArray` of `T`
/// wrapped in a `BufferBlock`-decorated `OpTypeStruct` (the pre-1.3, SPIR-V-1.0-legal SSBO shape
/// — see this file's header), pointed to by one `Uniform`-storage-class `OpVariable` decorated
/// with this binding's `DescriptorSet`/`Binding`. `decorated` remembers which wrapping struct
/// type ids this call has already decorated (`rspirv` structurally dedupes the `OpTypeStruct`/
/// `OpTypeRuntimeArray` themselves when two bindings share an element type, but never dedupes
/// `OpDecorate`, so without this a shared type would otherwise get the same decoration repeated
/// once per binding that shares it).
fn build_binding(ctx: &mut Ctx, elem_ty: Ty, binding: u32, decorated: &mut Vec<Word>) -> Word {
    let elem_word = ctx.repr_ty(elem_ty);
    let stride = elem_byte_size(elem_ty)
        .expect("elem_ty was already confirmed representable by resolve_binding_elem_tys")
        as u32;
    let runtime_arr_ty = ctx.b.type_runtime_array(elem_word);
    let block_ty = ctx.b.type_struct(vec![runtime_arr_ty]);
    if !decorated.contains(&block_ty) {
        ctx.b.decorate(
            runtime_arr_ty,
            Decoration::ArrayStride,
            vec![Operand::LiteralBit32(stride)],
        );
        ctx.b.decorate(block_ty, Decoration::BufferBlock, vec![]);
        ctx.b.member_decorate(
            block_ty,
            0,
            Decoration::Offset,
            vec![Operand::LiteralBit32(0)],
        );
        decorated.push(block_ty);
    }
    let block_ptr_ty = ctx.b.type_pointer(None, StorageClass::Uniform, block_ty);
    let var = ctx
        .b
        .variable(block_ptr_ty, None, StorageClass::Uniform, None);
    ctx.b.decorate(
        var,
        Decoration::DescriptorSet,
        vec![Operand::LiteralBit32(0)],
    );
    ctx.b.decorate(
        var,
        Decoration::Binding,
        vec![Operand::LiteralBit32(binding)],
    );
    var
}

/// The single `layout(push_constant) uniform PC { ... } pc;` block: one `OpTypeStruct` member
/// per scalar parameter (in `layout.push_const`'s already-packed order), each `Offset`-decorated
/// at its packed byte offset, wrapped in a `PushConstant`-storage-class `OpVariable`.
fn build_push_const(ctx: &mut Ctx, params: &[Ty], layout: &[(usize, u32, u32)]) -> Word {
    let member_tys: Vec<Word> = layout
        .iter()
        .map(|&(p, _, _)| ctx.repr_ty(params[p]))
        .collect();
    let struct_ty = ctx.b.type_struct(member_tys);
    ctx.b.decorate(struct_ty, Decoration::Block, vec![]);
    for (member_idx, &(_, offset, _)) in layout.iter().enumerate() {
        ctx.b.member_decorate(
            struct_ty,
            member_idx as u32,
            Decoration::Offset,
            vec![Operand::LiteralBit32(offset)],
        );
    }
    let ptr_ty = ctx
        .b
        .type_pointer(None, StorageClass::PushConstant, struct_ty);
    ctx.b
        .variable(ptr_ty, None, StorageClass::PushConstant, None)
}

/// Builds a real `OpAccessChain` into `binding_var`'s single runtime-array member at element
/// `index_word` (always truncated to 32 bits first — see this file's header) and returns the
/// resulting `Uniform`-storage-class pointer, typed to `elem_word`.
fn access_chain_ptr(ctx: &mut Ctx, elem_word: Word, binding_var: Word, index_word: Word) -> Word {
    let uint_ty = ctx.ty_uint();
    let zero = ctx.b.constant_bit32(uint_ty, 0);
    let idx32 = ctx.b.u_convert(uint_ty, None, index_word).unwrap();
    let ptr_ty = ctx.b.type_pointer(None, StorageClass::Uniform, elem_word);
    ctx.b
        .access_chain(ptr_ty, None, binding_var, vec![zero, idx32])
        .unwrap()
}

/// The stable parts of a function's `glcompute` lowering state — everything that never changes
/// once computed, as opposed to `params`/`done` (see `lower_function_glcompute`), which grow as
/// codegen proceeds and so cannot be borrowed alongside a mutation of themselves.
struct GlBindings<'a> {
    layout: &'a ResourceLayout,
    addr_map: &'a [Option<RecognizedAccess>],
    bindings: &'a [Word],
}

fn recognized(ptr: ValRef, addr_map: &[Option<RecognizedAccess>]) -> RecognizedAccess {
    let id = match ptr {
        ValRef::Val(id) => id,
        ValRef::Param(_) => {
            unreachable!("check_module_glcompute already refused a bare-parameter address")
        }
    };
    addr_map[id.0 as usize]
        .expect("check_module_glcompute already confirmed this Load/Store's address is recognized")
}

fn binding_of(layout: &ResourceLayout, param: usize) -> usize {
    layout.ptr_params.iter().position(|&p| p == param).expect(
        "recognize_access only ever names a parameter resource_layout already placed in ptr_params",
    )
}

#[allow(clippy::too_many_arguments)]
fn lower_glcompute_load(
    ctx: &mut Ctx,
    gb: &GlBindings,
    ptr: ValRef,
    ty: Ty,
    volatile: bool,
    params: &[Word],
    done: &[Option<Word>],
) -> Word {
    let acc = recognized(ptr, gb.addr_map);
    let index_word = resolve(acc.index, params, done);
    let binding = binding_of(gb.layout, acc.param);
    let elem_word = ctx.repr_ty(ty);
    let ac = access_chain_ptr(ctx, elem_word, gb.bindings[binding], index_word);
    ctx.b
        .load(elem_word, None, ac, memory_access(volatile), vec![])
        .unwrap()
}

#[allow(clippy::too_many_arguments)]
fn lower_glcompute_store(
    ctx: &mut Ctx,
    gb: &GlBindings,
    ptr: ValRef,
    val: ValRef,
    ty: Ty,
    volatile: bool,
    params: &[Word],
    done: &[Option<Word>],
) {
    let acc = recognized(ptr, gb.addr_map);
    let index_word = resolve(acc.index, params, done);
    let binding = binding_of(gb.layout, acc.param);
    let elem_word = ctx.repr_ty(ty);
    let ac = access_chain_ptr(ctx, elem_word, gb.bindings[binding], index_word);
    let val_word = resolve(val, params, done);
    ctx.b
        .store(ac, val_word, memory_access(volatile), vec![])
        .unwrap();
}

fn lower_function_glcompute(
    ctx: &mut Ctx,
    f: &Function,
    layout: &ResourceLayout,
    addr_map: &[Option<RecognizedAccess>],
    elem_tys: &[Ty],
) {
    let mut decorated_block_tys: Vec<Word> = Vec::new();
    let bindings: Vec<Word> = elem_tys
        .iter()
        .enumerate()
        .map(|(i, &ty)| build_binding(ctx, ty, i as u32, &mut decorated_block_tys))
        .collect();
    let gb = GlBindings {
        layout,
        addr_map,
        bindings: &bindings,
    };

    let push_const_var = if layout.push_const.is_empty() {
        None
    } else {
        Some(build_push_const(ctx, &f.params, &layout.push_const))
    };

    let void_ty = ctx.ty_void();
    let fn_ty = ctx.b.type_function(void_ty, vec![]);
    let fn_id = ctx
        .b
        .begin_function(void_ty, None, FunctionControl::NONE, fn_ty)
        .unwrap();
    ctx.b.name(fn_id, f.name.clone());

    let labels: Vec<Word> = (0..f.blocks.len()).map(|_| ctx.b.id()).collect();
    let mut used_builtins: Vec<Word> = Vec::new();
    let mut done: Vec<Option<Word>> = vec![None; f.insts.len()];

    // Pointer parameters have no real value under `Logical` addressing (see this file's
    // header); every legitimate use of one is intercepted before `resolve` is ever called on
    // it (the recognized outer `Add`, handled inline by `lower_glcompute_load`/`_store` below).
    // A shared `OpUndef` fills the slot so `resolve`'s signature (and `super::lower_inst`'s)
    // needs no change to accommodate this path — if `check_module_glcompute` ever had a gap and
    // this were read anyway, the result is a `SPIR-V`-legal-but-undefined value, never a panic.
    let ulong_ty = ctx.ty_ulong();
    let poison = ctx.b.undef(ulong_ty, None);
    let mut full_params: Vec<Word> = vec![poison; f.params.len()];

    for bidx in 0..f.blocks.len() {
        ctx.b.begin_block(Some(labels[bidx])).unwrap();
        if bidx == 0 {
            if let Some(pc_var) = push_const_var {
                for (member_idx, &(param_idx, _, _)) in layout.push_const.iter().enumerate() {
                    let elem_word = ctx.repr_ty(f.params[param_idx]);
                    let member_ptr_ty =
                        ctx.b
                            .type_pointer(None, StorageClass::PushConstant, elem_word);
                    let uint_ty = ctx.ty_uint();
                    let member_const = ctx.b.constant_bit32(uint_ty, member_idx as u32);
                    let ac = ctx
                        .b
                        .access_chain(member_ptr_ty, None, pc_var, vec![member_const])
                        .unwrap();
                    let loaded = ctx.b.load(elem_word, None, ac, None, vec![]).unwrap();
                    full_params[param_idx] = loaded;
                }
            }
        }
        for &inst_id in &f.blocks[bidx].insts {
            let inst = &f.insts[inst_id.0 as usize];
            let result = if addr_map[inst_id.0 as usize].is_some() {
                // The outer `Bin::Add` of a recognized access: never a real SPIR-V value under
                // `Logical` addressing (see this file's header, "What actually gets skipped") —
                // the Load/Store consuming it builds a real `OpAccessChain` in its place below.
                None
            } else {
                match &inst.op {
                    Op::Load { ptr, volatile, .. } => Some(lower_glcompute_load(
                        ctx,
                        &gb,
                        *ptr,
                        inst.ty,
                        *volatile,
                        &full_params,
                        &done,
                    )),
                    Op::Store {
                        ptr,
                        val,
                        ty,
                        volatile,
                        ..
                    } => {
                        lower_glcompute_store(
                            ctx,
                            &gb,
                            *ptr,
                            *val,
                            *ty,
                            *volatile,
                            &full_params,
                            &done,
                        );
                        None
                    }
                    _ => lower_inst(ctx, inst, &labels, &full_params, &done, &mut used_builtins),
                }
            };
            done[inst_id.0 as usize] = result;
        }
        lower_term(ctx, f, bidx, &labels, &full_params, &done);
    }

    ctx.b.end_function().unwrap();

    ctx.b.entry_point(
        ExecutionModel::GLCompute,
        fn_id,
        f.name.clone(),
        used_builtins,
    );
    ctx.b
        .execution_mode(fn_id, ExecutionMode::LocalSize, [1u32, 1, 1]);
}

pub(super) fn emit_glcompute(module: &Module) -> Result<Artifact, Diag> {
    check_module_glcompute(module)?;
    let ssa_module = construct_ssa(module);
    let mut ctx = new_ctx_glcompute();
    for f in &ssa_module.funcs {
        let layout = resource_layout(&f.params)?;
        let addr_map = build_addr_map(f)?;
        let elem_tys = resolve_binding_elem_tys(&layout, &addr_map)?;
        lower_function_glcompute(&mut ctx, f, &layout, &addr_map, &elem_tys);
    }
    let spirv_module = ctx.b.module();
    let words = Assemble::assemble(&spirv_module);
    let mut bytes = Vec::with_capacity(words.len() * 4);
    for w in words {
        bytes.extend_from_slice(&w.to_le_bytes());
    }
    Ok(Artifact::bytes(ArtifactKind::SpirV, bytes))
}
