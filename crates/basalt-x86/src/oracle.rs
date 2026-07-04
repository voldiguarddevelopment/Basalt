// The x86-64 oracle: a stack-everything, zero-register-allocation `Backend` impl. This is the
// project's correctness truth source, not a performance path — see the workspace-level design
// notes for why it must stay deliberately dumb. A second, regalloc-based x86-64 backend is
// expected to land in this same crate later (hence `X86Oracle`'s name — it will not be the
// only x86-64 `Backend` impl for long).
//
// # The SIMT-via-a-native-loop threading model
//
// A BIR kernel function is a single-thread program that reads `tid.x`/`bid.x`/etc. Real GPU
// hardware runs many threads of that program concurrently; this oracle instead runs every
// thread of a flat, single-block launch one at a time, wrapped in a native loop:
//
//   every BIR `Function` becomes one native function whose signature is the function's own
//   `params` (classified into the SysV integer/SSE argument sequence below) plus one trailing
//   `nthreads` parameter — how many threads this flat 1-D launch has. `blockDim.{y,z}` and
//   `gridDim.{y,z}` are fixed at 1; `blockIdx.{x,y,z}` is fixed at 0 (single block). This is an
//   intentional, documented scope limit: a real multi-block launch is future work, not a bug.
//
// `tid.x` reads the loop's own counter; `bdim.x` reads `nthreads`; every other GPU index op is
// the constant `0` or `1` per the table above. `barrier` is a genuine no-op (a `nop`, purely as
// a disassembly landmark): since threads run one at a time to completion before the next
// starts, thread N's stores are already visible to thread N+1 by construction — there is no
// actual concurrency for a barrier to guard against. `shuffle`/`ballot`/`vote.any`/`vote.all`
// are refused (`E090`): they are inherently warp-collective, needing several threads' values
// live at once, which a one-thread-at-a-time interpreter cannot express. Atomics do not need a
// `lock` prefix for the same reason (no real concurrency to race against) and are implemented
// as an ordinary load-compute-store sequence returning the pre-modification value, matching
// CUDA's atomic-RMW-returns-old semantics.
//
// # Stack-everything: zero register allocation
//
// Every instruction with a result gets its own fixed-size (8-byte, regardless of its actual
// bit width — see below) stack slot, written immediately after the instruction computes its
// value. Every later reference to that value reloads it from the slot into a scratch register
// right before the consuming instruction needs it; nothing is ever kept live in a register
// across an instruction boundary. A fixed pool of scratch GP registers (`rax`/`rcx`/`rdx`/
// `rbx`/`r10`, plus `xmm0`-`xmm2` for float work) is reused freely per instruction template
// with no cross-instruction lifetime tracking at all — that absence is the whole point of an
// oracle.
//
// Slots are uniformly 8 bytes wide rather than packed to each value's real bit width: every
// operation that cares about a value's width (arithmetic, compares, shifts, casts) reads or
// writes *exactly* that many bytes directly against memory (`mov al, [slot]` for an i8, never
// a wider read followed by masking), so nothing ever depends on whatever garbage sits in the
// unused remainder of an 8-byte slot holding a narrower value. The only place a value's width
// actually changes is an explicit `Cast`, which is exactly BIR's designed mechanism for that —
// this backend never has to invent sign/zero-extension rules for ordinary arithmetic.
//
// # Synthesizing local/param/shared/constant addresses
//
// `basalt-sema`'s lowering pass has no `alloca`; it hands every local/parameter storage
// location a small integer slot id and materializes that variable's address, wherever BIR
// needs it, as `const.i ptr.<space> (slot_id * 65536)` — an ordinary integer constant whose
// declared type happens to be a pointer, always immediately consumed as a `load`/`store`'s
// `ptr` operand. `65536 * slot_id` is not a real address (it stands in for a stack layout the
// lowering pass never computes tightly) — this backend treats each distinct `(space,
// const-value)` pair as an opaque slot identifier and, the first time it is seen, assigns it a
// real 8-byte cell in its own frame (`Frame::const_addr_disp`). `AddrSpace::Shared` and
// `AddrSpace::Constant` are folded into the same real-stack-memory treatment as `Local`: since
// only one thread ever executes at a time in this oracle, there is no actual shared-vs-local
// distinction that matters for correctness here (a real "many threads see the same shared
// buffer concurrently" model is exactly the multi-block/concurrent future work `barrier`'s
// no-op already assumes away).
//
// Rather than special-casing every `load`/`store` site to recognize this pattern, the constant
// itself is lowered to a genuine address the moment it is produced: `lea reg, [rbp - real_off]`
// into the instruction's own ordinary result slot, exactly like any other instruction. Every
// consumer downstream — a direct `load`/`store`, or pointer arithmetic on top of it (the
// documented path for indexing into a local array) — then sees a real, usable stack address
// and needs no special-casing at all. `AddrSpace::Global` values are never synthesized this
// way; they are real addresses from the moment they arrive (an incoming pointer argument, or
// arithmetic on one), so `load`/`store` uses exactly one address-handling path regardless of
// space: reload the `ptr` operand into a register, dereference it.
//
// # Calling convention
//
// SysV x86-64: `f32`/`f64` params consume `xmm0..xmm7` in order; everything else (all integer
// widths, every pointer) consumes `rdi, rsi, rdx, rcx, r8, r9` in order. `nthreads` always
// takes the next integer-class register after the function's own params. More than 6
// integer-class or 8 SSE-class arguments (stack-passed arguments) is refused (`E093`) rather
// than implemented — not a case any kernel in scope needs. Returns: `void` -> nothing,
// `f32`/`f64` -> `xmm0`, every other scalar/pointer -> `rax`. A `Vec` return type is refused
// (`E091`); vector codegen is out of scope for this backend.
//
// `div`/`rem` are always lowered as signed (`idiv`) — BIR's `Bin` op carries no
// signed/unsigned distinction for these (a documented gap in the lowering pass itself), so
// this backend picks the one interpretation and documents it, matching the lowering pass's own
// stance rather than inventing a `udiv` BIR has no way to ask for.
//
// `Select` lowers via a branch (test the i1 condition, jump over one arm), not a conditional
// move — one code path for both integer and float destinations rather than a `cmovcc` encoder
// plus an SSE blend encoder for the float case.
//
// `f16` is refused (`E091`) anywhere it would need real arithmetic (an instruction result, a
// `cast`/`fcmp`/`store`'s explicit operand type): baseline SSE2 has no scalar half-float
// arithmetic path (that needs the F16C extension), and this backend does not assume F16C is
// present.
//
// # Multi-function modules: one host function, plus the kernel(s) it launches
//
// A module of more than one function is refused (`E093`) unless it is shaped exactly like
// "one host (`is_kernel == false`) function, plus every kernel it actually launches via a
// real `Op::KernelLaunch`, and nothing else" (`classify_module`) — general multi-function
// support (two unrelated kernels, a `__device__` helper call, dynamic parallelism) remains
// out of scope and refuses just as it always has.
//
// Both the host function and every kernel it launches are lowered into one shared `.text`
// blob, each its own named `ElfSymbol` (see `basalt-backend::elf`) at its own offset — no
// relocation is needed since every offset is known once the whole blob is laid out. A
// kernel keeps its existing per-thread-loop lowering unchanged; the host function is
// lowered as an ordinary function with no such loop (it runs once, like any C function).
// Every intra-function label (`Enc::label`'s own names, never visible in the emitted bytes
// themselves) is qualified by its owning function's name so two functions sharing one `Enc`
// never collide; each function's own entry point is additionally labeled with its bare
// name, matching both the `ElfSymbol` name and the string `Op::KernelLaunch::kernel` names
// as its `call` target.
//
// `Op::KernelLaunch` lowers to a genuine `call rel32` (`Enc::call`) to the launched kernel's
// own label: each of the launch's `args` reloads into the *launched kernel's own*
// SysV-classified argument registers (the callee's own declared parameter types drive the
// classification, exactly matching how that kernel's own entry point already expects to
// receive them), and the kernel's own trailing `nthreads` register receives the flattened
// `grid`x`block` product (all six components multiplied together) — this backend's kernels
// already treat `blockIdx`/`gridDim` as fixed at 0/1 (see the module header above), so the
// real total number of loop iterations a launch means is that full product, not just
// `block.x`. `shared`/`stream` carry no meaning under this backend's single-threaded,
// one-block execution model: a launch is only accepted if both materialize to their
// documented default (a literal `Op::ConstInt(0)`, exactly what a source launch omitting
// either one already lowers to); anything else — a nonzero constant, or a genuinely dynamic
// value — is refused (`E093`) rather than silently ignored. `Op::CudaDeviceSynchronize`
// lowers to a real `nop`: every launch this backend can even accept already runs to
// completion synchronously inside its own `call`, so there is genuinely nothing left to
// wait for, not a stubbed-out placeholder. `Op::CudaMalloc`/`CudaMemcpy`/`CudaFree` remain
// refused (`E090`) inside a host function: they need a real relocation against an external
// libc symbol, which is a separate, later piece of work (see `PLAN.md`'s P13-T1c-ii).
//
// A kernel launching another kernel, or containing any of these five ops at all, is still
// refused exactly as before — dynamic parallelism is out of scope, and this backend has no
// call machinery for a kernel to use one from inside its own per-thread loop.
//
// # `mma`
//
// Lowered as a genuine triple-nested runtime loop (`for i in 0..m { for j in 0..n { for k in
// 0..k { ... } } }`), not unrolled — `m`/`n`/`k` are ordinary compile-time `u32` fields on the
// op, but this backend still emits real loop counters (their own stack homes, exactly like
// the outer per-thread loop's own `loopctr_home`) so code size stays flat regardless of tile
// size. Every element access recomputes its byte address from scratch each iteration (row/col
// times the operand's own leading dimension, times its element width, plus the base pointer)
// — no cleverness, no strength reduction, matching this backend's whole stance. `A`/`B`
// addressing follows `layout_a`/`layout_b`; `C`/`D` are always row-major (see `Op::Mma`'s own
// doc comment).
//
// Supported `(in_dtype, acc_dtype)` pairs: both integer or both float (never mixed — BIR gives
// this op no cast step to bridge them), `acc_dtype` at least as wide as `in_dtype` (never
// narrowing, which would make overflow behavior a silent guess), and never `i1`/`f16` in
// either position (a 1-bit product is not a sensible matmul input, and `f16` needs F16C like
// everywhere else in this backend). Anything else is refused (`E091`). Within a supported
// pair, the multiply-accumulate itself always runs at `acc_dtype`'s width: narrower inputs are
// widened first (`movsx` for integers, `cvtss2sd` for the one legal float widening, `f32` ->
// `f64`) so the running sum is never computed at less precision than the type BIR asked for.

use std::collections::HashMap;

use basalt_backend::{
    write_elf_object, Architecture, Artifact, ArtifactKind, Backend, ElfObjectSpec, ElfSymbol,
    EmitOpts, Endianness, Support,
};
use basalt_bir::{
    AddrSpace, AtomicOp, BinOp, CastOp, FCmpPred, Function, ICmpPred, InstId, MmaLayout, Module,
    Op, Scalar, Term, Ty, ValRef,
};
use basalt_diag::{Diag, ECode};

use crate::enc::{
    cc, AluOp, Enc, Rm, ShiftKind, SseArith, INT_ARG_REGS, R10, RAX, RBP, RBX, RCX, RDX,
    SSE_ARG_REGS, W,
};

/// The x86-64 stack-everything oracle backend: correct-first, never clever. See the module
/// header for the full design; `name()` returns `"x86-oracle"` so a later `--cpu` CLI wire-up
/// can address it unambiguously alongside the (not yet written) regalloc-based x86-64 backend.
#[derive(Debug, Default, Clone, Copy)]
pub struct X86Oracle;

impl Backend for X86Oracle {
    fn name(&self) -> &'static str {
        "x86-oracle"
    }

    fn supports(&self, module: &Module) -> Support {
        match check_module(module) {
            Ok(_) => Support::Supported,
            Err(diag) => Support::Unsupported(diag.code),
        }
    }

    fn emit(&self, module: &Module, _opts: &EmitOpts) -> Result<Artifact, Diag> {
        let shape = check_module(module)?;
        let (text, symbols) = match shape {
            ModuleShape::SingleKernel(f) => {
                let text = emit_function(f)?;
                let size = text.len() as u64;
                (
                    text,
                    vec![ElfSymbol {
                        name: f.name.clone(),
                        offset: 0,
                        size,
                    }],
                )
            }
            ModuleShape::HostAndKernels { host, kernels } => emit_host_and_kernels(host, &kernels)?,
        };
        let spec =
            ElfObjectSpec::new_multi(Architecture::X86_64, Endianness::Little, symbols, text);
        let bytes = write_elf_object(&spec)?;
        Ok(Artifact::bytes(ArtifactKind::Object, bytes))
    }
}

/// The exact module shapes this backend accepts, returned by `check_module` on success: the
/// original single-kernel-only shape, or "one host function plus the kernel(s) it actually
/// launches" (see the module header's own section on this).
enum ModuleShape<'a> {
    SingleKernel(&'a Function),
    HostAndKernels {
        host: &'a Function,
        kernels: Vec<&'a Function>,
    },
}

/// Single source of truth for what this backend refuses, shared verbatim by `supports()` and
/// `emit()` so the two can never drift apart. Returns the module shape to lower on success.
fn check_module(module: &Module) -> Result<ModuleShape<'_>, Diag> {
    if module.funcs.len() == 1 && module.funcs[0].is_kernel {
        let f = &module.funcs[0];
        check_function(f, false, &[])?;
        return Ok(ModuleShape::SingleKernel(f));
    }
    if module.funcs.len() == 1 {
        return Err(Diag::new(ECode::UnsupportedFeature).with_arg(
            "host/non-kernel function compilation needs at least one kernel it launches",
        ));
    }

    let hosts: Vec<&Function> = module.funcs.iter().filter(|f| !f.is_kernel).collect();
    let host = match hosts.as_slice() {
        [host] => *host,
        [] => {
            return Err(Diag::new(ECode::UnsupportedFeature).with_arg(
                "multi-function module: general multi-kernel support is out of scope; needs \
                 exactly one host function launching kernel(s) it names",
            ))
        }
        _ => {
            return Err(Diag::new(ECode::UnsupportedFeature)
                .with_arg("multi-function module: more than one host (non-kernel) function"))
        }
    };

    let mut launched: Vec<&str> = Vec::new();
    for inst in &host.insts {
        if let Op::KernelLaunch { kernel, .. } = &inst.op {
            launched.push(kernel.as_str());
        }
    }
    if launched.is_empty() {
        return Err(
            Diag::new(ECode::UnsupportedFeature).with_arg("host function launches no kernels")
        );
    }

    let kernels: Vec<&Function> = module.funcs.iter().filter(|f| f.is_kernel).collect();
    for k in &kernels {
        if !launched.contains(&k.name.as_str()) {
            return Err(Diag::new(ECode::UnsupportedFeature).with_arg(
                "multi-function module: kernel present but never launched by the host function \
                 (general multi-kernel/device-helper-call support is out of scope)",
            ));
        }
    }
    for name in &launched {
        if !kernels.iter().any(|k| k.name == *name) {
            return Err(Diag::new(ECode::UnsupportedFeature)
                .with_arg("kernel launch names a function not present in this module"));
        }
    }

    check_function(host, true, &kernels)?;
    for k in &kernels {
        check_function(k, false, &[])?;
    }

    Ok(ModuleShape::HostAndKernels { host, kernels })
}

/// Whether `v` is the launch's own documented default for `shared`/`stream` (a literal
/// `Op::ConstInt(0)` — the exact materialization `basalt-sema`'s own lowering produces for a
/// source launch that names neither, see `Op::KernelLaunch`'s doc comment) — the only shape
/// this backend accepts for either operand.
fn launch_operand_is_default(f: &Function, v: ValRef) -> bool {
    match v {
        ValRef::Val(id) => matches!(f.insts[id.0 as usize].op, Op::ConstInt(0)),
        ValRef::Param(_) => false,
    }
}

/// The per-function checks shared by every function this backend ever lowers, kernel or
/// host. `is_host` and `launch_targets` (the kernels visible for a host function's own
/// `Op::KernelLaunch` calls, empty for a kernel) select the one place kernel and host
/// bodies are actually allowed to differ: which of the five kernel-launch/CUDA-Runtime-API
/// ops are real here versus still refused.
fn check_function(f: &Function, is_host: bool, launch_targets: &[&Function]) -> Result<(), Diag> {
    if matches!(f.ret, Ty::Vec(..)) {
        return Err(Diag::new(ECode::UnsupportedType).with_arg("vector-typed return value"));
    }
    if f.params.iter().any(|t| matches!(t, Ty::Vec(..))) {
        return Err(Diag::new(ECode::UnsupportedType).with_arg("vector-typed parameter"));
    }
    if ty_is_f16(f.ret) {
        return Err(Diag::new(ECode::UnsupportedType).with_arg("f16 return value needs F16C"));
    }
    if f.params.iter().any(|&t| ty_is_f16(t)) {
        return Err(Diag::new(ECode::UnsupportedType).with_arg("f16 parameter needs F16C"));
    }
    if classify_params(&f.params).is_none() {
        return Err(Diag::new(ECode::UnsupportedFeature).with_arg(
            "argument classes overflow SysV's 6 integer / 8 SSE registers (incl. trailing nthreads)",
        ));
    }

    for inst in &f.insts {
        if matches!(inst.ty, Ty::Vec(..)) {
            return Err(
                Diag::new(ECode::UnsupportedType).with_arg("vector-typed instruction result")
            );
        }
        if ty_is_f16(inst.ty) {
            return Err(Diag::new(ECode::UnsupportedType).with_arg("f16 arithmetic needs F16C"));
        }
        match &inst.op {
            Op::Shuffle(..) | Op::Ballot(..) | Op::VoteAny(..) | Op::VoteAll(..) => {
                return Err(Diag::new(ECode::UnsupportedOp).with_arg(
                    "warp-collective op has no meaning under one-thread-at-a-time execution",
                ));
            }
            Op::Cast(_, sty, _) | Op::FCmp(_, sty, _, _) if ty_is_f16(*sty) => {
                return Err(Diag::new(ECode::UnsupportedType).with_arg("f16 operand needs F16C"));
            }
            Op::Store { ty: sty, .. } if ty_is_f16(*sty) => {
                return Err(Diag::new(ECode::UnsupportedType).with_arg("f16 store needs F16C"));
            }
            Op::Mma {
                in_dtype,
                acc_dtype,
                ..
            } if !mma_dtypes_supported(*in_dtype, *acc_dtype) => {
                return Err(Diag::new(ECode::UnsupportedType).with_arg(
                    "mma dtype pair: in_dtype/acc_dtype must both be integer or both float, \
                     acc_dtype at least as wide as in_dtype, and neither i1 nor f16",
                ));
            }
            Op::KernelLaunch {
                kernel,
                shared,
                stream,
                args,
                ..
            } if is_host => {
                if !launch_targets.iter().any(|k| &k.name == kernel) {
                    return Err(Diag::new(ECode::UnsupportedFeature)
                        .with_arg("kernel launch names a function not present in this module"));
                }
                let target = launch_targets
                    .iter()
                    .find(|k| &k.name == kernel)
                    .expect("just checked above");
                if args.len() != target.params.len() {
                    return Err(Diag::new(ECode::UnsupportedFeature).with_arg(
                        "kernel launch argument count does not match the launched kernel's own \
                         signature",
                    ));
                }
                if !launch_operand_is_default(f, *shared) || !launch_operand_is_default(f, *stream)
                {
                    return Err(Diag::new(ECode::UnsupportedFeature).with_arg(
                        "non-default shared-memory/stream operand has no meaning under this \
                         backend's single-threaded, one-block execution model",
                    ));
                }
            }
            Op::CudaDeviceSynchronize if is_host => {}
            Op::KernelLaunch { .. }
            | Op::CudaMalloc { .. }
            | Op::CudaMemcpy { .. }
            | Op::CudaFree { .. }
            | Op::CudaDeviceSynchronize => {
                return Err(Diag::new(ECode::UnsupportedOp).with_arg(if is_host {
                    "cudaMalloc/cudaMemcpy/cudaFree need a real relocation against an external \
                     libc symbol this backend does not emit yet"
                } else {
                    "kernel launch / CUDA Runtime API calls inside a kernel (dynamic \
                     parallelism) are out of scope for this backend"
                }));
            }
            _ => {}
        }
    }

    Ok(())
}

fn ty_is_f16(ty: Ty) -> bool {
    matches!(ty, Ty::Scalar(Scalar::F16))
}

/// `(is_float, byte width)` for the scalar types `mma` accepts at all — `None` for `i1`/`f16`,
/// which are never valid in either of `mma`'s dtype fields (see the module header's `# mma`
/// section).
fn mma_scalar_class(s: Scalar) -> Option<(bool, u8)> {
    match s {
        Scalar::I8 => Some((false, 1)),
        Scalar::I16 => Some((false, 2)),
        Scalar::I32 => Some((false, 4)),
        Scalar::I64 => Some((false, 8)),
        Scalar::F32 => Some((true, 4)),
        Scalar::F64 => Some((true, 8)),
        Scalar::I1 | Scalar::F16 => None,
    }
}

/// Whether this backend's oracle lowering supports multiplying-accumulating `in_dtype`
/// operands into an `acc_dtype` accumulator: both integer or both float, `acc_dtype` no
/// narrower than `in_dtype`.
fn mma_dtypes_supported(in_dtype: Scalar, acc_dtype: Scalar) -> bool {
    match (mma_scalar_class(in_dtype), mma_scalar_class(acc_dtype)) {
        (Some((in_float, in_w)), Some((acc_float, acc_w))) => {
            in_float == acc_float && acc_w >= in_w
        }
        _ => false,
    }
}

fn is_float(ty: Ty) -> bool {
    matches!(ty, Ty::Scalar(Scalar::F32) | Ty::Scalar(Scalar::F64))
}

fn is_f64(ty: Ty) -> bool {
    matches!(ty, Ty::Scalar(Scalar::F64))
}

/// The exact byte width used by every memory access for a value of this type (never rounded
/// up beyond it — see the module header's width-exactness design). Every `Ty` this backend
/// accepts reaches here (`Vec`/`Void`/`f16` are all refused in `check_module` before codegen
/// starts).
fn width_of(ty: Ty) -> W {
    match ty {
        Ty::Scalar(Scalar::I1 | Scalar::I8) => W::B1,
        Ty::Scalar(Scalar::I16) => W::B2,
        Ty::Scalar(Scalar::I32 | Scalar::F32) => W::B4,
        Ty::Scalar(Scalar::I64 | Scalar::F64) | Ty::Ptr(_) => W::B8,
        Ty::Scalar(Scalar::F16) | Ty::Vec(..) | Ty::Void => {
            unreachable!("width_of called on a type check_module should have refused")
        }
    }
}

/// `W`'s byte count, for `mma`'s address arithmetic (an `i64` multiplicand, not a `W` itself).
fn w_bytes(w: W) -> i64 {
    match w {
        W::B1 => 1,
        W::B2 => 2,
        W::B4 => 4,
        W::B8 => 8,
    }
}

fn local_like(space: AddrSpace) -> bool {
    matches!(
        space,
        AddrSpace::Local | AddrSpace::Param | AddrSpace::Shared | AddrSpace::Constant
    )
}

fn space_tag(space: AddrSpace) -> u8 {
    match space {
        AddrSpace::Global => 0,
        AddrSpace::Shared => 1,
        AddrSpace::Constant => 2,
        AddrSpace::Local => 3,
        AddrSpace::Param => 4,
    }
}

#[derive(Clone, Copy)]
enum ArgLoc {
    Int(u8),
    Sse(u8),
}

/// The one slice of a launched kernel's own signature `lower_kernel_launch` needs to
/// classify the call's own argument registers exactly like that kernel's own entry point
/// does. An owned copy (not a borrow of the kernel `Function` itself) so `CodeGen` needs no
/// extra lifetime parameter beyond the one already tying it to the function it is currently
/// lowering.
struct KernelSig {
    name: String,
    params: Vec<Ty>,
}

/// Classifies `params` into the SysV integer/SSE argument sequence and returns the location
/// the trailing `nthreads` parameter would land in. `None` means the signature overflows the
/// register-passing convention (more than 6 integer-class or 8 SSE-class arguments, counting
/// `nthreads`) — this backend does not implement stack-passed arguments.
fn classify_params(params: &[Ty]) -> Option<(Vec<ArgLoc>, ArgLoc)> {
    let mut int_idx = 0usize;
    let mut sse_idx = 0usize;
    let mut locs = Vec::with_capacity(params.len());
    for &ty in params {
        if is_float(ty) {
            if sse_idx >= SSE_ARG_REGS.len() {
                return None;
            }
            locs.push(ArgLoc::Sse(SSE_ARG_REGS[sse_idx]));
            sse_idx += 1;
        } else {
            if int_idx >= INT_ARG_REGS.len() {
                return None;
            }
            locs.push(ArgLoc::Int(INT_ARG_REGS[int_idx]));
            int_idx += 1;
        }
    }
    if int_idx >= INT_ARG_REGS.len() {
        return None;
    }
    Some((locs, ArgLoc::Int(INT_ARG_REGS[int_idx])))
}

/// The extra stack homes one `mma` instruction needs for its own triple loop: `i`/`j`/`k`
/// loop counters (one per nesting level, the same "a live scheduling value gets a stack home"
/// treatment as the outer per-thread loop's `loopctr_home`) plus a running-sum accumulator
/// reloaded and re-stored every `k` iteration rather than ever held live in a register.
#[derive(Clone, Copy)]
struct MmaSlots {
    i: i32,
    j: i32,
    k: i32,
    acc: i32,
}

/// This function's real native stack frame: every fixed home plus one 8-byte slot per BIR
/// instruction result and one per synthesized local/param/shared/constant address. See the
/// module header for why every slot is uniformly 8 bytes.
struct Frame {
    param_home: Vec<i32>,
    nthreads_home: i32,
    loopctr_home: i32,
    retval_home: Option<i32>,
    inst_slot: Vec<i32>,
    const_addr_disp: HashMap<(u8, i64), i32>,
    mma_slots: HashMap<u32, MmaSlots>,
    frame_size: i32,
}

fn next_slot(offset: &mut i32) -> i32 {
    *offset += 8;
    -*offset
}

impl Frame {
    fn build(f: &Function) -> Frame {
        let mut offset: i32 = 0;

        let param_home: Vec<i32> = f.params.iter().map(|_| next_slot(&mut offset)).collect();
        let nthreads_home = next_slot(&mut offset);
        let loopctr_home = next_slot(&mut offset);
        let retval_home = if matches!(f.ret, Ty::Void) {
            None
        } else {
            Some(next_slot(&mut offset))
        };
        let inst_slot: Vec<i32> = f.insts.iter().map(|_| next_slot(&mut offset)).collect();

        let mut const_addr_disp = HashMap::new();
        for inst in &f.insts {
            if let (Op::ConstInt(n), Ty::Ptr(space)) = (&inst.op, inst.ty) {
                if local_like(space) {
                    let key = (space_tag(space), *n);
                    const_addr_disp
                        .entry(key)
                        .or_insert_with(|| next_slot(&mut offset));
                }
            }
        }

        let mut mma_slots = HashMap::new();
        for (idx, inst) in f.insts.iter().enumerate() {
            if matches!(inst.op, Op::Mma { .. }) {
                let slots = MmaSlots {
                    i: next_slot(&mut offset),
                    j: next_slot(&mut offset),
                    k: next_slot(&mut offset),
                    acc: next_slot(&mut offset),
                };
                mma_slots.insert(idx as u32, slots);
            }
        }

        let frame_size = (offset + 15) & !15;
        Frame {
            param_home,
            nthreads_home,
            loopctr_home,
            retval_home,
            inst_slot,
            const_addr_disp,
            mma_slots,
            frame_size,
        }
    }
}

/// Per-edge phi lowering: for each `(from_block, to_block)` edge, the list of (destination
/// slot, incoming value, its type) copies to run right before the jump that takes that edge —
/// the standard "every predecessor writes the phi's own slot" technique for a design where
/// every SSA value already lives in memory.
type PhiCopies = HashMap<(u32, u32), Vec<(i32, ValRef, Ty)>>;

fn build_phi_copies(f: &Function, frame: &Frame) -> PhiCopies {
    let mut map: PhiCopies = HashMap::new();
    for (bidx, block) in f.blocks.iter().enumerate() {
        for &inst_id in &block.insts {
            let inst = &f.insts[inst_id.0 as usize];
            if let Op::Phi(preds) = &inst.op {
                let dest = frame.inst_slot[inst_id.0 as usize];
                for &(pred_block, val) in preds {
                    map.entry((pred_block.0, bidx as u32))
                        .or_default()
                        .push((dest, val, inst.ty));
                }
            }
        }
    }
    map
}

struct CodeGen<'a, 'e> {
    f: &'a Function,
    frame: Frame,
    enc: &'e mut Enc,
    label_counter: u32,
    phi_copies: PhiCopies,
    /// Qualifies every intra-function label this function's own body defines or references
    /// (see `lbl`) — this function's own name, so two functions sharing one `Enc` (see
    /// `emit_host_and_kernels`) never collide.
    label_prefix: String,
    /// Where `Term::Ret` actually jumps: a kernel's own per-thread loop increment step (the
    /// kernel keeps running the next thread), or a host function's own epilogue (an
    /// ordinary function just returns).
    ret_target: String,
    /// The kernel(s) a host function's own `Op::KernelLaunch` instructions may call. Always
    /// empty for a kernel body (`check_module` refuses `Op::KernelLaunch` there).
    launch_targets: Vec<KernelSig>,
}

/// Lowers one function's full body — prologue, params, blocks, epilogue — into `enc`,
/// starting with a label at `f.name` naming its own entry point (a `call` target for a
/// host's own launches, and this function's own `ElfSymbol` name in the multi-function
/// case). A kernel (`is_host = false`) keeps the existing per-thread-loop wrapper unchanged;
/// a host function (`is_host = true`) is lowered as an ordinary function that runs once, no
/// loop, no synthesized trailing `nthreads` parameter.
fn emit_function_body(
    f: &Function,
    is_host: bool,
    launch_targets: Vec<KernelSig>,
    enc: &mut Enc,
) -> Result<(), Diag> {
    let (param_locs, nthreads_loc) =
        classify_params(&f.params).expect("check_module already validated the signature");

    let frame = Frame::build(f);
    let phi_copies = build_phi_copies(f, &frame);
    let mut cg = CodeGen {
        f,
        frame,
        enc,
        label_counter: 0,
        phi_copies,
        label_prefix: format!("{}$", f.name),
        ret_target: String::new(),
        launch_targets,
    };

    cg.enc.label(&f.name);
    cg.enc.push_reg(RBP);
    cg.enc.mov_rbp_rsp();
    cg.enc.sub_rsp_imm(cg.frame.frame_size);

    for (i, loc) in param_locs.iter().enumerate() {
        let disp = cg.frame.param_home[i];
        let ty = f.params[i];
        match *loc {
            ArgLoc::Int(r) => cg.enc.mov_rbp_reg(width_of(ty), disp, r),
            ArgLoc::Sse(r) => {
                if is_f64(ty) {
                    cg.enc.movsd_store(Rm::RbpDisp(disp), r);
                } else {
                    cg.enc.movss_store(Rm::RbpDisp(disp), r);
                }
            }
        }
    }

    if is_host {
        cg.ret_target = cg.lbl("__epilogue");
    } else {
        if let ArgLoc::Int(r) = nthreads_loc {
            cg.enc.mov_rbp_reg(W::B8, cg.frame.nthreads_home, r);
        }

        cg.enc.mov_reg_imm32(W::B8, RAX, 0);
        cg.enc.mov_rbp_reg(W::B8, cg.frame.loopctr_home, RAX);

        let loop_check = cg.lbl("__loop_check");
        let loop_end = cg.lbl("__loop_end");
        cg.enc.label(&loop_check);
        cg.enc.mov_reg_rbp(W::B8, RAX, cg.frame.loopctr_home);
        cg.enc.mov_reg_rbp(W::B8, RCX, cg.frame.nthreads_home);
        cg.enc.alu_reg_reg(AluOp::Cmp, W::B8, RAX, RCX);
        cg.enc.jcc(cc::GE, &loop_end);

        cg.ret_target = cg.lbl("__loop_incr");
    }

    for (bidx, block) in f.blocks.iter().enumerate() {
        let label = cg.block_label(bidx as u32);
        cg.enc.label(&label);
        for &inst_id in &block.insts {
            cg.lower_inst(inst_id);
        }
        cg.lower_term(bidx as u32, &block.term);
    }

    if is_host {
        let epilogue = cg.lbl("__epilogue");
        cg.enc.label(&epilogue);
    } else {
        let loop_incr = cg.lbl("__loop_incr");
        let loop_check = cg.lbl("__loop_check");
        let loop_end = cg.lbl("__loop_end");
        cg.enc.label(&loop_incr);
        cg.enc.mov_reg_rbp(W::B8, RAX, cg.frame.loopctr_home);
        cg.enc.alu_reg_imm32(AluOp::Add, W::B8, RAX, 1);
        cg.enc.mov_rbp_reg(W::B8, cg.frame.loopctr_home, RAX);
        cg.enc.jmp(&loop_check);
        cg.enc.label(&loop_end);
    }

    if !matches!(f.ret, Ty::Void) {
        let disp = cg
            .frame
            .retval_home
            .expect("non-void return always gets a retval home");
        if is_float(f.ret) {
            if is_f64(f.ret) {
                cg.enc.movsd_load(0, Rm::RbpDisp(disp));
            } else {
                cg.enc.movss_load(0, Rm::RbpDisp(disp));
            }
        } else {
            cg.enc.mov_reg_rbp(width_of(f.ret), RAX, disp);
        }
    }
    cg.enc.mov_rsp_rbp();
    cg.enc.pop_reg(RBP);
    cg.enc.ret();

    Ok(())
}

/// The `ModuleShape::SingleKernel` path: one function, its own fresh `Enc`, no label
/// qualification needed (nothing else shares the buffer) — kept as its own entry point so
/// this shape's emitted bytes never change shape merely because the multi-function path
/// now exists alongside it.
fn emit_function(f: &Function) -> Result<Vec<u8>, Diag> {
    let mut enc = Enc::new();
    emit_function_body(f, false, Vec::new(), &mut enc)?;
    Ok(enc.finish())
}

/// The `ModuleShape::HostAndKernels` path: the host function followed by every kernel it
/// launches, all lowered into one shared `Enc` so the host's own `call`s can reach them (see
/// the module header). Returns the combined `.text` bytes plus one `ElfSymbol` per function,
/// each sized by the gap to the next function's own entry point (or to the end of the
/// buffer, for whichever function was laid out last).
fn emit_host_and_kernels(
    host: &Function,
    kernels: &[&Function],
) -> Result<(Vec<u8>, Vec<ElfSymbol>), Diag> {
    let launch_targets: Vec<KernelSig> = kernels
        .iter()
        .map(|k| KernelSig {
            name: k.name.clone(),
            params: k.params.clone(),
        })
        .collect();

    let mut enc = Enc::new();
    emit_function_body(host, true, launch_targets, &mut enc)?;
    for k in kernels {
        emit_function_body(k, false, Vec::new(), &mut enc)?;
    }

    let mut names: Vec<&str> = vec![host.name.as_str()];
    names.extend(kernels.iter().map(|k| k.name.as_str()));
    let mut offsets: Vec<(String, u64)> = names
        .iter()
        .map(|name| {
            let off = enc.label_offset(name).unwrap_or_else(|| {
                panic!("codegen bug: `{name}`'s own entry label was never defined")
            });
            ((*name).to_string(), off as u64)
        })
        .collect();
    let total = enc.pos() as u64;
    let bytes = enc.finish();

    offsets.sort_by_key(|(_, off)| *off);
    let symbols = offsets
        .iter()
        .enumerate()
        .map(|(i, (name, off))| {
            let end = offsets.get(i + 1).map_or(total, |(_, o)| *o);
            ElfSymbol {
                name: name.clone(),
                offset: *off,
                size: end - off,
            }
        })
        .collect();
    Ok((bytes, symbols))
}

impl<'a, 'e> CodeGen<'a, 'e> {
    /// Qualifies an intra-function label name with this function's own `label_prefix`.
    /// Label spelling has no bearing on the emitted bytes (`Enc::finish` resolves fixups
    /// purely by numeric offset), so this exists solely to keep two functions sharing one
    /// `Enc` from ever defining the same label name twice.
    fn lbl(&self, s: &str) -> String {
        format!("{}{s}", self.label_prefix)
    }

    fn block_label(&self, id: u32) -> String {
        self.lbl(&format!("bb{id}"))
    }

    fn fresh_label(&mut self, prefix: &str) -> String {
        self.label_counter += 1;
        self.lbl(&format!("__{prefix}_{}", self.label_counter))
    }

    fn valref_ty(&self, v: ValRef) -> Ty {
        match v {
            ValRef::Param(i) => self.f.params[i as usize],
            ValRef::Val(id) => self.f.insts[id.0 as usize].ty,
        }
    }

    fn operand_disp(&self, v: ValRef) -> i32 {
        match v {
            ValRef::Param(i) => self.frame.param_home[i as usize],
            ValRef::Val(id) => self.frame.inst_slot[id.0 as usize],
        }
    }

    fn reload_gpr(&mut self, v: ValRef, dst: u8, w: W) {
        let disp = self.operand_disp(v);
        self.enc.mov_reg_rbp(w, dst, disp);
    }

    fn reload_gpr_zx(&mut self, v: ValRef, dst: u8, dst_w: W, src_w: W) {
        let disp = self.operand_disp(v);
        self.enc.movzx(dst_w, src_w, dst, Rm::RbpDisp(disp));
    }

    fn reload_gpr_sx(&mut self, v: ValRef, dst: u8, dst_w: W, src_w: W) {
        let disp = self.operand_disp(v);
        self.enc.movsx(dst_w, src_w, dst, Rm::RbpDisp(disp));
    }

    fn reload_xmm(&mut self, v: ValRef, dst_xmm: u8, is_f64_: bool) {
        let disp = self.operand_disp(v);
        if is_f64_ {
            self.enc.movsd_load(dst_xmm, Rm::RbpDisp(disp));
        } else {
            self.enc.movss_load(dst_xmm, Rm::RbpDisp(disp));
        }
    }

    fn store_result(&mut self, id: InstId, src: u8, w: W) {
        let disp = self.frame.inst_slot[id.0 as usize];
        self.enc.mov_rbp_reg(w, disp, src);
    }

    fn store_result_xmm(&mut self, id: InstId, src_xmm: u8, is_f64_: bool) {
        let disp = self.frame.inst_slot[id.0 as usize];
        if is_f64_ {
            self.enc.movsd_store(Rm::RbpDisp(disp), src_xmm);
        } else {
            self.enc.movss_store(Rm::RbpDisp(disp), src_xmm);
        }
    }

    /// Copies `v` (of type `ty`) straight into instruction `id`'s own result slot — used by
    /// `select`'s two arms.
    fn copy_value(&mut self, v: ValRef, id: InstId, ty: Ty) {
        if is_float(ty) {
            let f64_ = is_f64(ty);
            self.reload_xmm(v, 0, f64_);
            self.store_result_xmm(id, 0, f64_);
        } else {
            let w = width_of(ty);
            self.reload_gpr(v, RAX, w);
            self.store_result(id, RAX, w);
        }
    }

    fn lower_inst(&mut self, id: InstId) {
        let f = self.f;
        let inst = &f.insts[id.0 as usize];
        let ty = inst.ty;
        match &inst.op {
            Op::ConstInt(n) => {
                let n = *n;
                // `basalt-sema`'s own lowering synthesizes a placeholder `const.i void 0` /
                // `const.f void 0` as the discarded "value" of a void expression used as a
                // bare statement (a kernel launch, `__syncthreads()`, ...) — see
                // `basalt_sema::lower::zero_of`. `Ty::Void` means "no SSA value" by BIR's own
                // convention (`Store`/`Mma`/`Barrier` are all `Ty::Void` for the same reason),
                // so nothing ever legitimately reads this instruction's own result; skip it
                // exactly like `Op::Phi`'s own "nothing to do at the definition site" below.
                if matches!(ty, Ty::Void) {
                    return;
                }
                if let Ty::Ptr(space) = ty {
                    if local_like(space) {
                        let key = (space_tag(space), n);
                        let disp = *self
                            .frame
                            .const_addr_disp
                            .get(&key)
                            .expect("Frame::build pre-scans every local-slot constant");
                        self.enc.lea_rbp(RAX, disp);
                        self.store_result(id, RAX, W::B8);
                        return;
                    }
                }
                self.enc.movabs(RAX, n);
                self.store_result(id, RAX, width_of(ty));
            }
            Op::ConstFloat(v) => {
                let v = *v;
                if matches!(ty, Ty::Void) {
                    return;
                }
                if is_f64(ty) {
                    self.enc.movabs(RAX, v.to_bits() as i64);
                    self.store_result(id, RAX, W::B8);
                } else {
                    let bits = (v as f32).to_bits();
                    self.enc.mov_reg_imm32(W::B4, RAX, bits as i32);
                    self.store_result(id, RAX, W::B4);
                }
            }
            Op::Bin(op, a, b) => {
                let (op, a, b) = (*op, *a, *b);
                self.lower_bin(id, op, a, b, ty);
            }
            Op::ICmp(pred, cty, a, b) => {
                let (pred, cty, a, b) = (*pred, *cty, *a, *b);
                self.lower_icmp(id, pred, cty, a, b);
            }
            Op::FCmp(pred, cty, a, b) => {
                let (pred, cty, a, b) = (*pred, *cty, *a, *b);
                self.lower_fcmp(id, pred, cty, a, b);
            }
            Op::Select(c, a, b) => {
                let (c, a, b) = (*c, *a, *b);
                self.reload_gpr(c, RAX, W::B1);
                self.enc.test_reg_reg(W::B1, RAX);
                let else_label = self.fresh_label("select_false");
                let end_label = self.fresh_label("select_end");
                self.enc.jcc(cc::E, &else_label);
                self.copy_value(a, id, ty);
                self.enc.jmp(&end_label);
                self.enc.label(&else_label);
                self.copy_value(b, id, ty);
                self.enc.label(&end_label);
            }
            Op::Cast(cop, sty, v) => {
                let (cop, sty, v) = (*cop, *sty, *v);
                self.lower_cast(id, cop, sty, v, ty);
            }
            Op::Load { ptr, .. } => {
                let ptr = *ptr;
                self.lower_load(id, ptr, ty);
            }
            Op::Store {
                ptr, val, ty: sty, ..
            } => {
                let (ptr, val, sty) = (*ptr, *val, *sty);
                self.lower_store(ptr, val, sty);
            }
            Op::Phi(_) => {
                // Nothing to do at the definition site: every predecessor writes this
                // instruction's own result slot directly before jumping here (see
                // `emit_phi_copies`), so by the time control reaches this block the slot
                // already holds the right value.
            }
            Op::TidX => {
                let d = self.frame.loopctr_home;
                self.lower_reload_dyn(id, d, ty);
            }
            Op::BdimX => {
                let d = self.frame.nthreads_home;
                self.lower_reload_dyn(id, d, ty);
            }
            Op::TidY | Op::TidZ | Op::BidX | Op::BidY | Op::BidZ => {
                self.enc.movabs(RAX, 0);
                self.store_result(id, RAX, width_of(ty));
            }
            Op::BdimY | Op::BdimZ | Op::GdimX | Op::GdimY | Op::GdimZ => {
                self.enc.movabs(RAX, 1);
                self.store_result(id, RAX, width_of(ty));
            }
            Op::Barrier => self.enc.nop(),
            Op::Shuffle(..) | Op::Ballot(..) | Op::VoteAny(..) | Op::VoteAll(..) => {
                unreachable!("check_module refuses these before codegen starts")
            }
            Op::Atomic(aop, ptr, val, _space) => {
                let (aop, ptr, val) = (*aop, *ptr, *val);
                self.lower_atomic(id, aop, ptr, val, ty);
            }
            Op::AtomicCas(ptr, cmp, newv, _space) => {
                let (ptr, cmp, newv) = (*ptr, *cmp, *newv);
                self.lower_atomic_cas(id, ptr, cmp, newv, ty);
            }
            Op::Mma {
                a,
                b,
                c,
                d,
                m,
                n,
                k,
                in_dtype,
                acc_dtype,
                layout_a,
                layout_b,
            } => {
                let (a, b, c, d, m, n, k, in_dtype, acc_dtype, layout_a, layout_b) = (
                    *a, *b, *c, *d, *m, *n, *k, *in_dtype, *acc_dtype, *layout_a, *layout_b,
                );
                self.lower_mma(
                    id, a, b, c, d, m, n, k, in_dtype, acc_dtype, layout_a, layout_b,
                );
            }
            Op::KernelLaunch {
                kernel,
                grid,
                block,
                shared,
                stream,
                args,
            } => {
                let (kernel, grid, block, shared, stream) =
                    (kernel.clone(), *grid, *block, *shared, *stream);
                self.lower_kernel_launch(&kernel, grid, block, shared, stream, args);
            }
            // Every launch this backend can even accept already runs to completion
            // synchronously inside its own `call` (see the module header) — genuinely
            // nothing left to wait for, so this is a real no-op, not a stub.
            Op::CudaDeviceSynchronize => self.enc.nop(),
            Op::CudaMalloc { .. } | Op::CudaMemcpy { .. } | Op::CudaFree { .. } => {
                unreachable!("check_module refuses these before codegen starts")
            }
        }
    }

    /// `tid.x`/`bdim.x`: reload a live scheduling value (the loop counter / thread count)
    /// from its always-8-byte home, truncating on store to whatever width this op's result
    /// type actually declares.
    fn lower_reload_dyn(&mut self, id: InstId, src_disp: i32, ty: Ty) {
        self.enc.mov_reg_rbp(W::B8, RAX, src_disp);
        self.store_result(id, RAX, width_of(ty));
    }

    fn lower_bin(&mut self, id: InstId, op: BinOp, a: ValRef, b: ValRef, ty: Ty) {
        if is_float(ty) {
            let f64_ = is_f64(ty);
            self.reload_xmm(a, 0, f64_);
            self.reload_xmm(b, 1, f64_);
            match op {
                BinOp::FAdd => self.enc.sse_arith(SseArith::Add, f64_, 0, Rm::Direct(1)),
                BinOp::FSub => self.enc.sse_arith(SseArith::Sub, f64_, 0, Rm::Direct(1)),
                BinOp::FMul => self.enc.sse_arith(SseArith::Mul, f64_, 0, Rm::Direct(1)),
                BinOp::FDiv => self.enc.sse_arith(SseArith::Div, f64_, 0, Rm::Direct(1)),
                BinOp::FRem => self.lower_frem(f64_),
                _ => unreachable!("float-typed Bin with a non-float BinOp"),
            }
            self.store_result_xmm(id, 0, f64_);
            return;
        }

        let w = width_of(ty);
        match op {
            BinOp::Add | BinOp::Sub | BinOp::And | BinOp::Or | BinOp::Xor => {
                self.reload_gpr(a, RAX, w);
                self.reload_gpr(b, RCX, w);
                let aop = match op {
                    BinOp::Add => AluOp::Add,
                    BinOp::Sub => AluOp::Sub,
                    BinOp::And => AluOp::And,
                    BinOp::Or => AluOp::Or,
                    BinOp::Xor => AluOp::Xor,
                    _ => unreachable!(),
                };
                self.enc.alu_reg_reg(aop, w, RAX, RCX);
                self.store_result(id, RAX, w);
            }
            BinOp::Mul => {
                // IMUL's two-operand form has no 8-bit encoding; promote to 32-bit (a
                // product's low N bits depend only on the low N bits of its factors, so
                // zero-extending narrow factors before a wider multiply is exact).
                if w == W::B1 {
                    self.reload_gpr_zx(a, RAX, W::B4, W::B1);
                    self.reload_gpr_zx(b, RCX, W::B4, W::B1);
                    self.enc.imul_reg_reg(W::B4, RAX, RCX);
                } else {
                    self.reload_gpr(a, RAX, w);
                    self.reload_gpr(b, RCX, w);
                    self.enc.imul_reg_reg(w, RAX, RCX);
                }
                self.store_result(id, RAX, w);
            }
            BinOp::Div | BinOp::Rem => {
                // `idiv r/m8` splits its result across AL/AH, which is awkward to reach
                // uniformly under this backend's always-REX byte-register policy (AH is
                // only reachable with no REX prefix at all). Sidestep it by promoting an
                // 8-bit divide to 32-bit: dividing two 8-bit values can never produce a
                // quotient/remainder wider than 8 bits, so sign-extending first is exact.
                let dw = if w == W::B1 { W::B4 } else { w };
                if w == W::B1 {
                    self.reload_gpr_sx(a, RAX, W::B4, W::B1);
                    self.reload_gpr_sx(b, R10, W::B4, W::B1);
                } else {
                    self.reload_gpr(a, RAX, w);
                    self.reload_gpr(b, R10, w);
                }
                self.enc.cdq(dw);
                self.enc.idiv_reg(dw, R10);
                let result_reg = if matches!(op, BinOp::Div) { RAX } else { RDX };
                self.store_result(id, result_reg, w);
            }
            BinOp::Shl | BinOp::Lshr | BinOp::Ashr => {
                self.reload_gpr(a, RAX, w);
                self.reload_gpr(b, RCX, w);
                let kind = match op {
                    BinOp::Shl => ShiftKind::Shl,
                    BinOp::Lshr => ShiftKind::Shr,
                    BinOp::Ashr => ShiftKind::Sar,
                    _ => unreachable!(),
                };
                self.enc.shift_cl(kind, w, RAX);
                self.store_result(id, RAX, w);
            }
            _ => unreachable!("integer-typed Bin with a float BinOp"),
        }
    }

    /// Software float remainder: `q = trunc(a / b); result = a - q*b`. x86 SSE has no scalar
    /// fmod instruction (that is an x87 `fprem`-only feature, and this backend never touches
    /// the x87 stack) — this is the standard divide/truncate/multiply/subtract emulation.
    /// Not bit-exact to IEEE `fmod` for ratios whose truncated quotient does not fit an i64,
    /// which is far outside the magnitude range any kernel this oracle targets would hit.
    /// Entered with `xmm0 = a`, `xmm1 = b`; leaves the result in `xmm0`.
    fn lower_frem(&mut self, f64_: bool) {
        self.enc.sse_move(2, 0, f64_); // xmm2 = a (b in xmm1 stays put)
        self.enc.sse_arith(SseArith::Div, f64_, 0, Rm::Direct(1)); // xmm0 = a/b
        self.enc.cvtt_to_si(f64_, W::B8, RAX, Rm::Direct(0));
        self.enc.cvt_si_to(f64_, W::B8, 0, Rm::Direct(RAX)); // xmm0 = trunc(a/b)
        self.enc.sse_arith(SseArith::Mul, f64_, 0, Rm::Direct(1)); // xmm0 = q*b
        self.enc.sse_move(1, 2, f64_); // xmm1 = a
        self.enc.sse_arith(SseArith::Sub, f64_, 1, Rm::Direct(0)); // xmm1 = a - q*b
        self.enc.sse_move(0, 1, f64_); // xmm0 = result
    }

    fn lower_icmp(&mut self, id: InstId, pred: ICmpPred, cty: Ty, a: ValRef, b: ValRef) {
        let w = width_of(cty);
        self.reload_gpr(a, RAX, w);
        self.reload_gpr(b, RCX, w);
        self.enc.alu_reg_reg(AluOp::Cmp, w, RAX, RCX);
        let code = match pred {
            ICmpPred::Eq => cc::E,
            ICmpPred::Ne => cc::NE,
            ICmpPred::Slt => cc::L,
            ICmpPred::Sle => cc::LE,
            ICmpPred::Sgt => cc::G,
            ICmpPred::Sge => cc::GE,
            ICmpPred::Ult => cc::B,
            ICmpPred::Ule => cc::BE,
            ICmpPred::Ugt => cc::A,
            ICmpPred::Uge => cc::AE,
        };
        self.enc.setcc(code, RAX);
        self.store_result(id, RAX, W::B1);
    }

    /// `ucomiss`/`ucomisd a, b` sets `(ZF,PF,CF)` to exactly one of four combinations:
    /// unordered `(1,1,1)`, `a>b` `(0,0,0)`, `a<b` `(0,0,1)`, `a==b` `(1,0,0)`. Every ordered
    /// predicate this enum has reduces to a single `setcc` or an AND of two — see the
    /// derivation in the module header's design notes (kept here as the one place the
    /// combinations are actually used).
    fn lower_fcmp(&mut self, id: InstId, pred: FCmpPred, cty: Ty, a: ValRef, b: ValRef) {
        let f64_ = is_f64(cty);
        self.reload_xmm(a, 0, f64_);
        self.reload_xmm(b, 1, f64_);
        if f64_ {
            self.enc.ucomisd(0, Rm::Direct(1));
        } else {
            self.enc.ucomiss(0, Rm::Direct(1));
        }
        match pred {
            // CF=0 & ZF=0 only in the a>b case (unordered forces both to 1).
            FCmpPred::Ogt => self.enc.setcc(cc::A, RAX),
            // CF=0 only in the a>b or a==b cases (unordered forces CF=1).
            FCmpPred::Oge => self.enc.setcc(cc::AE, RAX),
            FCmpPred::Ord => self.enc.setcc(cc::NP, RAX),
            FCmpPred::Uno => self.enc.setcc(cc::P, RAX),
            // CF=1 & ZF=0 only in the a<b case (unordered has ZF=1 too).
            FCmpPred::Olt => {
                self.enc.setcc(cc::B, RAX);
                self.enc.setcc(cc::NE, RCX);
                self.enc.alu_reg_reg(AluOp::And, W::B1, RAX, RCX);
            }
            // (CF=1|ZF=1) & PF=0: below-or-equal, but ordered.
            FCmpPred::Ole => {
                self.enc.setcc(cc::BE, RAX);
                self.enc.setcc(cc::NP, RCX);
                self.enc.alu_reg_reg(AluOp::And, W::B1, RAX, RCX);
            }
            // ZF=1 & PF=0: unordered also sets ZF=1, so PF=0 is needed to exclude it.
            FCmpPred::Oeq => {
                self.enc.setcc(cc::E, RAX);
                self.enc.setcc(cc::NP, RCX);
                self.enc.alu_reg_reg(AluOp::And, W::B1, RAX, RCX);
            }
            FCmpPred::One => {
                self.enc.setcc(cc::NE, RAX);
                self.enc.setcc(cc::NP, RCX);
                self.enc.alu_reg_reg(AluOp::And, W::B1, RAX, RCX);
            }
        }
        self.store_result(id, RAX, W::B1);
    }

    fn lower_cast(&mut self, id: InstId, cop: CastOp, sty: Ty, v: ValRef, dty: Ty) {
        match cop {
            CastOp::Trunc => {
                // Two's-complement truncation is just "read the low N bytes" — since every
                // slot already only has its real value in its low `width_of(src)` bytes,
                // and dst is narrower, this is a plain narrow read straight from the
                // source's own slot.
                let dw = width_of(dty);
                self.reload_gpr(v, RAX, dw);
                self.store_result(id, RAX, dw);
            }
            CastOp::Zext => {
                let sw = width_of(sty);
                let dw = width_of(dty);
                let disp = self.operand_disp(v);
                match sw {
                    W::B1 | W::B2 => self.enc.movzx(dw, sw, RAX, Rm::RbpDisp(disp)),
                    // A plain 32-bit write already zero-extends the upper 32 bits of the
                    // 64-bit register natively; no movzx opcode exists for this case.
                    W::B4 | W::B8 => self.enc.mov_reg_rbp(sw, RAX, disp),
                }
                self.store_result(id, RAX, dw);
            }
            CastOp::Sext => {
                let sw = width_of(sty);
                let dw = width_of(dty);
                let disp = self.operand_disp(v);
                match sw {
                    W::B1 | W::B2 => self.enc.movsx(dw, sw, RAX, Rm::RbpDisp(disp)),
                    W::B4 => self.enc.movsx(W::B8, W::B4, RAX, Rm::RbpDisp(disp)),
                    W::B8 => self.enc.mov_reg_rbp(W::B8, RAX, disp),
                }
                self.store_result(id, RAX, dw);
            }
            CastOp::FpTrunc => {
                self.reload_xmm(v, 0, true);
                self.enc.cvtsd2ss(0, Rm::Direct(0));
                self.store_result_xmm(id, 0, false);
            }
            CastOp::FpExt => {
                self.reload_xmm(v, 0, false);
                self.enc.cvtss2sd(0, Rm::Direct(0));
                self.store_result_xmm(id, 0, true);
            }
            CastOp::FpToSi => {
                let src_f64 = is_f64(sty);
                self.reload_xmm(v, 0, src_f64);
                let gpr_w = if width_of(dty) == W::B8 { W::B8 } else { W::B4 };
                self.enc.cvtt_to_si(src_f64, gpr_w, RAX, Rm::Direct(0));
                self.store_result(id, RAX, width_of(dty));
            }
            CastOp::FpToUi => self.lower_fp_to_ui(id, sty, v, dty),
            CastOp::SiToFp => {
                let dst_f64 = is_f64(dty);
                let sw = width_of(sty);
                if sw == W::B1 || sw == W::B2 {
                    self.reload_gpr_sx(v, RAX, W::B4, sw);
                } else {
                    self.reload_gpr(v, RAX, sw);
                }
                let gpr_w = if sw == W::B8 { W::B8 } else { W::B4 };
                self.enc.cvt_si_to(dst_f64, gpr_w, 0, Rm::Direct(RAX));
                self.store_result_xmm(id, 0, dst_f64);
            }
            CastOp::UiToFp => self.lower_ui_to_fp(id, sty, v, dty),
            CastOp::Bitcast => {
                let src_float = is_float(sty);
                let dst_float = is_float(dty);
                let w = width_of(dty);
                match (src_float, dst_float) {
                    (false, true) => {
                        self.reload_gpr(v, RAX, w);
                        self.enc.movd_to_xmm(w, 0, Rm::Direct(RAX));
                        self.store_result_xmm(id, 0, w == W::B8);
                    }
                    (true, false) => {
                        self.reload_xmm(v, 0, is_f64(sty));
                        self.enc.movd_from_xmm(w, Rm::Direct(RAX), 0);
                        self.store_result(id, RAX, w);
                    }
                    _ => {
                        // Same-kind reinterpret at equal width: a plain copy.
                        self.reload_gpr(v, RAX, w);
                        self.store_result(id, RAX, w);
                    }
                }
            }
        }
    }

    /// `fptoui`: baseline SSE2 has no unsigned float-to-int conversion. For a destination
    /// narrower than 64 bits, every possible value fits exactly in a signed i64, so the
    /// ordinary signed truncating conversion is already exact. For a full 64-bit unsigned
    /// destination, this is the textbook two-path fixup: values below 2^63 convert directly;
    /// values at or above 2^63 are shifted into signed range first, converted, then have the
    /// sign bit XORed back onto the integer result.
    fn lower_fp_to_ui(&mut self, id: InstId, sty: Ty, v: ValRef, dty: Ty) {
        let src_f64 = is_f64(sty);
        self.reload_xmm(v, 0, src_f64);
        let dw = width_of(dty);
        if dw != W::B8 {
            self.enc.cvtt_to_si(src_f64, W::B8, RAX, Rm::Direct(0));
            self.store_result(id, RAX, dw);
            return;
        }

        let two_pow_63_bits: i64 = if src_f64 {
            9223372036854775808.0_f64.to_bits() as i64
        } else {
            9223372036854775808.0_f32.to_bits() as i64
        };
        self.enc.movabs(RCX, two_pow_63_bits);
        self.enc
            .movd_to_xmm(if src_f64 { W::B8 } else { W::B4 }, 1, Rm::Direct(RCX));
        if src_f64 {
            self.enc.ucomisd(0, Rm::Direct(1));
        } else {
            self.enc.ucomiss(0, Rm::Direct(1));
        }
        let hi_label = self.fresh_label("fptoui_hi");
        let done_label = self.fresh_label("fptoui_done");
        self.enc.jcc(cc::AE, &hi_label); // CF=0: value >= 2^63 (ordered; unordered has CF=1)
        self.enc.cvtt_to_si(src_f64, W::B8, RAX, Rm::Direct(0));
        self.enc.jmp(&done_label);
        self.enc.label(&hi_label);
        self.enc.sse_arith(SseArith::Sub, src_f64, 0, Rm::Direct(1));
        self.enc.cvtt_to_si(src_f64, W::B8, RAX, Rm::Direct(0));
        self.enc.movabs(RCX, i64::MIN);
        self.enc.alu_reg_reg(AluOp::Xor, W::B8, RAX, RCX);
        self.enc.label(&done_label);
        self.store_result(id, RAX, W::B8);
    }

    /// `uitofp`: for a source narrower than 64 bits, zero-extending into a 64-bit signed
    /// register is exact (every value fits) and a plain signed conversion follows. For a
    /// full 64-bit unsigned source, the textbook fixup: values with the top bit clear convert
    /// directly; otherwise halve the value (keeping the dropped bit, for correct rounding),
    /// convert, then double the float result.
    fn lower_ui_to_fp(&mut self, id: InstId, sty: Ty, v: ValRef, dty: Ty) {
        let dst_f64 = is_f64(dty);
        let sw = width_of(sty);
        if sw != W::B8 {
            match sw {
                W::B1 | W::B2 => self.reload_gpr_zx(v, RAX, W::B8, sw),
                W::B4 => {
                    let disp = self.operand_disp(v);
                    self.enc.mov_reg_rbp(W::B4, RAX, disp); // zero-extends to 64 bits natively
                }
                W::B8 => unreachable!(),
            }
            self.enc.cvt_si_to(dst_f64, W::B8, 0, Rm::Direct(RAX));
            self.store_result_xmm(id, 0, dst_f64);
            return;
        }

        self.reload_gpr(v, RAX, W::B8);
        self.enc.test_reg_reg(W::B8, RAX);
        let hi_label = self.fresh_label("uitofp_hi");
        let done_label = self.fresh_label("uitofp_done");
        self.enc.jcc(cc::S, &hi_label);
        self.enc.cvt_si_to(dst_f64, W::B8, 0, Rm::Direct(RAX));
        self.enc.jmp(&done_label);
        self.enc.label(&hi_label);
        self.enc.mov_reg_reg(W::B8, RCX, RAX);
        self.enc.alu_reg_imm32(AluOp::And, W::B8, RCX, 1);
        self.enc.shift1(ShiftKind::Shr, W::B8, RAX);
        self.enc.alu_reg_reg(AluOp::Or, W::B8, RAX, RCX);
        self.enc.cvt_si_to(dst_f64, W::B8, 0, Rm::Direct(RAX));
        self.enc.sse_arith(SseArith::Add, dst_f64, 0, Rm::Direct(0));
        self.enc.label(&done_label);
        self.store_result_xmm(id, 0, dst_f64);
    }

    /// Every address space this backend touches is, by the time a value reaches here, a
    /// genuine usable stack or heap address (see the module header) — one path handles
    /// `load`/`store` for `global`/`local`/`param`/`shared`/`constant` alike.
    fn lower_load(&mut self, id: InstId, ptr: ValRef, ty: Ty) {
        let addr_disp = self.operand_disp(ptr);
        self.enc.mov_reg_rbp(W::B8, R10, addr_disp);
        if is_float(ty) {
            let f64_ = is_f64(ty);
            if f64_ {
                self.enc.movsd_load(0, Rm::IndBase(R10));
            } else {
                self.enc.movss_load(0, Rm::IndBase(R10));
            }
            self.store_result_xmm(id, 0, f64_);
        } else {
            let w = width_of(ty);
            self.enc.mov_reg_ind(w, RAX, R10);
            self.store_result(id, RAX, w);
        }
    }

    fn lower_store(&mut self, ptr: ValRef, val: ValRef, ty: Ty) {
        let addr_disp = self.operand_disp(ptr);
        self.enc.mov_reg_rbp(W::B8, R10, addr_disp);
        if is_float(ty) {
            let f64_ = is_f64(ty);
            self.reload_xmm(val, 0, f64_);
            if f64_ {
                self.enc.movsd_store(Rm::IndBase(R10), 0);
            } else {
                self.enc.movss_store(Rm::IndBase(R10), 0);
            }
        } else {
            let w = width_of(ty);
            self.reload_gpr(val, RAX, w);
            self.enc.mov_ind_reg(w, R10, RAX);
        }
    }

    /// Ordinary (non-`lock`-prefixed) load-compute-store: correct here because exactly one
    /// thread ever executes at a time (see the module header), returning the
    /// pre-modification value to match CUDA's atomic-RMW-returns-old semantics. `And`/`Or`/
    /// `Xor` on a float type operate on its raw bit pattern (well-defined, if an unusual
    /// thing for a real kernel to ask for).
    fn lower_atomic(&mut self, id: InstId, op: AtomicOp, ptr: ValRef, val: ValRef, ty: Ty) {
        let addr_disp = self.operand_disp(ptr);
        self.enc.mov_reg_rbp(W::B8, R10, addr_disp);

        if is_float(ty) {
            let f64_ = is_f64(ty);
            if f64_ {
                self.enc.movsd_load(0, Rm::IndBase(R10));
            } else {
                self.enc.movss_load(0, Rm::IndBase(R10));
            }
            self.enc.sse_move(2, 0, f64_); // xmm2 = old, kept for the return value
            self.reload_xmm(val, 1, f64_);
            match op {
                AtomicOp::Add => self.enc.sse_arith(SseArith::Add, f64_, 0, Rm::Direct(1)),
                AtomicOp::Sub => self.enc.sse_arith(SseArith::Sub, f64_, 0, Rm::Direct(1)),
                AtomicOp::Exch => self.enc.sse_move(0, 1, f64_),
                AtomicOp::Min | AtomicOp::Max => {
                    if f64_ {
                        self.enc.ucomisd(0, Rm::Direct(1));
                    } else {
                        self.enc.ucomiss(0, Rm::Direct(1));
                    }
                    let skip = self.fresh_label("atomic_minmax_skip");
                    // Min keeps old when old<=val (BE); Max keeps old when old>=val (AE,
                    // ordered-only since unordered forces CF=1).
                    let skip_cc = if matches!(op, AtomicOp::Min) {
                        cc::BE
                    } else {
                        cc::AE
                    };
                    self.enc.jcc(skip_cc, &skip);
                    self.enc.sse_move(0, 1, f64_);
                    self.enc.label(&skip);
                }
                AtomicOp::And | AtomicOp::Or | AtomicOp::Xor => {
                    let w = if f64_ { W::B8 } else { W::B4 };
                    self.enc.movd_from_xmm(w, Rm::Direct(RAX), 0);
                    self.enc.movd_from_xmm(w, Rm::Direct(RCX), 1);
                    let aop = match op {
                        AtomicOp::And => AluOp::And,
                        AtomicOp::Or => AluOp::Or,
                        AtomicOp::Xor => AluOp::Xor,
                        _ => unreachable!(),
                    };
                    self.enc.alu_reg_reg(aop, w, RAX, RCX);
                    self.enc.movd_to_xmm(w, 0, Rm::Direct(RAX));
                }
            }
            if f64_ {
                self.enc.movsd_store(Rm::IndBase(R10), 0);
            } else {
                self.enc.movss_store(Rm::IndBase(R10), 0);
            }
            self.store_result_xmm(id, 2, f64_);
            return;
        }

        let w = width_of(ty);
        self.enc.mov_reg_ind(w, RAX, R10); // old
        self.enc.mov_reg_reg(w, RBX, RAX); // stashed for the return value
        self.reload_gpr(val, RCX, w);
        match op {
            AtomicOp::Add => self.enc.alu_reg_reg(AluOp::Add, w, RAX, RCX),
            AtomicOp::Sub => self.enc.alu_reg_reg(AluOp::Sub, w, RAX, RCX),
            AtomicOp::And => self.enc.alu_reg_reg(AluOp::And, w, RAX, RCX),
            AtomicOp::Or => self.enc.alu_reg_reg(AluOp::Or, w, RAX, RCX),
            AtomicOp::Xor => self.enc.alu_reg_reg(AluOp::Xor, w, RAX, RCX),
            AtomicOp::Exch => self.enc.mov_reg_reg(w, RAX, RCX),
            AtomicOp::Min | AtomicOp::Max => {
                // Signed compare, matching this backend's uniform signed choice for
                // div/rem: BIR's `AtomicOp` has no unsigned min/max variant either.
                self.enc.alu_reg_reg(AluOp::Cmp, w, RAX, RCX);
                let skip = self.fresh_label("atomic_minmax_skip");
                let skip_cc = if matches!(op, AtomicOp::Min) {
                    cc::LE
                } else {
                    cc::GE
                };
                self.enc.jcc(skip_cc, &skip);
                self.enc.mov_reg_reg(w, RAX, RCX);
                self.enc.label(&skip);
            }
        }
        self.enc.mov_ind_reg(w, R10, RAX); // store new
        self.store_result(id, RBX, w); // return old
    }

    /// `atomicCAS` compares and swaps the raw bit pattern regardless of `ty` — real hardware
    /// CAS is always an integer operation even for float/double CAS in CUDA, so there is
    /// nothing float-specific to do here at all: reload every operand at `ty`'s width exactly
    /// as raw bytes, compare, conditionally store, and return the old bytes (which a float
    /// consumer will reinterpret correctly on its own next reload).
    fn lower_atomic_cas(&mut self, id: InstId, ptr: ValRef, cmp: ValRef, newv: ValRef, ty: Ty) {
        let w = width_of(ty);
        let addr_disp = self.operand_disp(ptr);
        self.enc.mov_reg_rbp(W::B8, R10, addr_disp);
        self.enc.mov_reg_ind(w, RAX, R10);
        self.reload_gpr(cmp, RCX, w);
        self.enc.alu_reg_reg(AluOp::Cmp, w, RAX, RCX);
        let mismatch = self.fresh_label("cas_mismatch");
        self.enc.jcc(cc::NE, &mismatch);
        self.reload_gpr(newv, RDX, w);
        self.enc.mov_ind_reg(w, R10, RDX);
        self.enc.label(&mismatch);
        self.store_result(id, RAX, w);
    }

    /// Lowers a real `Op::KernelLaunch` to a genuine intra-object `call` — reached only for
    /// a host function's own instructions (`check_module` refuses this op inside a kernel
    /// body). `shared`/`stream` carry no meaning under this backend's execution model and
    /// have already been confirmed default by `check_function`, so there is nothing left to
    /// do with them here.
    ///
    /// Each of `args` reloads straight into the *launched kernel's own* SysV-classified
    /// argument register — `target`'s own declared parameter types drive the
    /// classification, exactly matching how that kernel's own entry point (`emit_function_
    /// body`) already expects to receive them. `nthreads` (the kernel's own trailing
    /// argument) receives the flattened `grid`x`block` product: this backend's kernels treat
    /// `blockIdx`/`gridDim` as fixed at 0/1 (see the module header), so the real number of
    /// per-thread loop iterations a launch means is that full six-way product, not just
    /// `block.x`. The product is computed in `RAX`/`R10` — scratch registers that are never
    /// themselves a SysV argument-carrying register — so it makes no difference whether it
    /// runs before or after the param registers below are populated.
    fn lower_kernel_launch(
        &mut self,
        kernel: &str,
        grid: [ValRef; 3],
        block: [ValRef; 3],
        shared: ValRef,
        stream: ValRef,
        args: &[ValRef],
    ) {
        let _ = (shared, stream);

        let target_idx = self
            .launch_targets
            .iter()
            .position(|k| k.name == kernel)
            .expect("check_function already validated every launch names a real kernel");
        let target_params = self.launch_targets[target_idx].params.clone();
        let (param_locs, nthreads_loc) = classify_params(&target_params)
            .expect("check_function already validated the launched kernel's own signature");
        debug_assert_eq!(args.len(), param_locs.len());

        self.reload_gpr(grid[0], RAX, W::B4);
        for &dim in grid[1..].iter().chain(block.iter()) {
            self.reload_gpr(dim, R10, W::B4);
            self.enc.imul_reg_reg(W::B4, RAX, R10);
        }
        if let ArgLoc::Int(r) = nthreads_loc {
            self.enc.mov_reg_reg(W::B8, r, RAX);
        }

        for (i, loc) in param_locs.iter().enumerate() {
            let arg = args[i];
            let ty = target_params[i];
            match *loc {
                ArgLoc::Int(r) => self.reload_gpr(arg, r, width_of(ty)),
                ArgLoc::Sse(r) => self.reload_xmm(arg, r, is_f64(ty)),
            }
        }

        self.enc.call(kernel);
    }

    /// `dst := [rbp + base_disp] + (row*leading_dim + col) * elem_bytes`, the address of one
    /// element of an `mma` operand tile — always recomputed from scratch (see the module
    /// header's `# mma` section). `RAX`/`RCX` are scratch; `dst` may be either of them, in
    /// which case the final copy into `dst` is skipped.
    fn mma_addr(
        &mut self,
        dst: u8,
        base_disp: i32,
        row_disp: i32,
        col_disp: i32,
        leading_dim: u32,
        elem_bytes: i64,
    ) {
        self.enc.mov_reg_rbp(W::B8, RAX, row_disp);
        self.enc.movabs(RCX, leading_dim as i64);
        self.enc.imul_reg_reg(W::B8, RAX, RCX);
        self.enc.mov_reg_rbp(W::B8, RCX, col_disp);
        self.enc.alu_reg_reg(AluOp::Add, W::B8, RAX, RCX);
        self.enc.movabs(RCX, elem_bytes);
        self.enc.imul_reg_reg(W::B8, RAX, RCX);
        self.enc.mov_reg_rbp(W::B8, RCX, base_disp);
        self.enc.alu_reg_reg(AluOp::Add, W::B8, RAX, RCX);
        if dst != RAX {
            self.enc.mov_reg_reg(W::B8, dst, RAX);
        }
    }

    /// Loads an `mma` integer operand element from `[addr_reg]`, sign-extending from `in_w`
    /// up to `target_w` if they differ (matches this backend's uniform signed stance —
    /// `BinOp::Div`/`Rem` and the atomic min/max ops make the same choice, since BIR gives
    /// `mma` no signed/unsigned distinction to ask for either).
    fn load_mma_int(&mut self, dst: u8, addr_reg: u8, in_w: W, target_w: W) {
        if in_w == target_w {
            self.enc.mov_reg_ind(target_w, dst, addr_reg);
        } else {
            self.enc.movsx(target_w, in_w, dst, Rm::IndBase(addr_reg));
        }
    }

    /// Loads an `mma` float operand element from `[addr_reg]` into `dst_xmm`, widening `f32`
    /// -> `f64` if `in_dtype`/`acc_dtype` differ (the only legal float widening `mma_dtypes_
    /// supported` admits).
    fn load_mma_float(&mut self, dst_xmm: u8, addr_reg: u8, in_f64: bool, acc_f64: bool) {
        if in_f64 == acc_f64 {
            if acc_f64 {
                self.enc.movsd_load(dst_xmm, Rm::IndBase(addr_reg));
            } else {
                self.enc.movss_load(dst_xmm, Rm::IndBase(addr_reg));
            }
        } else {
            debug_assert!(!in_f64 && acc_f64, "f64 input into an f32 accumulator");
            self.enc.movss_load(dst_xmm, Rm::IndBase(addr_reg));
            self.enc.cvtss2sd(dst_xmm, Rm::Direct(dst_xmm));
        }
    }

    /// `mma`: a genuine triple-nested runtime loop over `i in 0..m`, `j in 0..n`, `k in 0..k`
    /// — see the module header's `# mma` section for the addressing and dtype-widening rules
    /// this implements. `check_module` has already refused any `(in_dtype, acc_dtype)` pair
    /// this can't lower, so every arm below is exhaustive over what actually reaches here.
    #[allow(clippy::too_many_arguments)]
    fn lower_mma(
        &mut self,
        id: InstId,
        a: ValRef,
        b: ValRef,
        c: ValRef,
        d: ValRef,
        m: u32,
        n: u32,
        k: u32,
        in_dtype: Scalar,
        acc_dtype: Scalar,
        layout_a: MmaLayout,
        layout_b: MmaLayout,
    ) {
        let a_disp = self.operand_disp(a);
        let b_disp = self.operand_disp(b);
        let c_disp = self.operand_disp(c);
        let d_disp = self.operand_disp(d);
        let slots = *self
            .frame
            .mma_slots
            .get(&id.0)
            .expect("Frame::build reserves i/j/k/acc slots for every mma instruction");

        let in_ty = Ty::Scalar(in_dtype);
        let acc_ty = Ty::Scalar(acc_dtype);
        let in_w = width_of(in_ty);
        let acc_w = width_of(acc_ty);
        let in_bytes = w_bytes(in_w);
        let acc_bytes = w_bytes(acc_w);
        let acc_float = is_float(acc_ty);
        let acc_f64 = is_f64(acc_ty);
        let in_f64 = is_f64(in_ty);
        // `imul` has no 8-bit two-operand form (same constraint `lower_bin`'s `BinOp::Mul`
        // works around): an i8 accumulator's multiply-accumulate runs at 32 bits internally,
        // truncating back to 8 bits only when the running sum is written back to its slot.
        let cw = if acc_w == W::B1 { W::B4 } else { acc_w };

        let (a_row, a_col, a_leading) = match layout_a {
            MmaLayout::RowMajor => (slots.i, slots.k, k),
            MmaLayout::ColMajor => (slots.k, slots.i, m),
        };
        let (b_row, b_col, b_leading) = match layout_b {
            MmaLayout::RowMajor => (slots.k, slots.j, n),
            MmaLayout::ColMajor => (slots.j, slots.k, k),
        };

        let i_check = self.fresh_label("mma_i_check");
        let i_end = self.fresh_label("mma_i_end");
        let j_check = self.fresh_label("mma_j_check");
        let j_end = self.fresh_label("mma_j_end");
        let k_check = self.fresh_label("mma_k_check");
        let k_end = self.fresh_label("mma_k_end");

        self.enc.mov_reg_imm32(W::B8, RAX, 0);
        self.enc.mov_rbp_reg(W::B8, slots.i, RAX);
        self.enc.label(&i_check);
        self.enc.mov_reg_rbp(W::B8, RAX, slots.i);
        self.enc.movabs(RCX, m as i64);
        self.enc.alu_reg_reg(AluOp::Cmp, W::B8, RAX, RCX);
        self.enc.jcc(cc::GE, &i_end);

        self.enc.mov_reg_imm32(W::B8, RAX, 0);
        self.enc.mov_rbp_reg(W::B8, slots.j, RAX);
        self.enc.label(&j_check);
        self.enc.mov_reg_rbp(W::B8, RAX, slots.j);
        self.enc.movabs(RCX, n as i64);
        self.enc.alu_reg_reg(AluOp::Cmp, W::B8, RAX, RCX);
        self.enc.jcc(cc::GE, &j_end);

        self.enc.mov_reg_imm32(acc_w, RAX, 0);
        self.enc.mov_rbp_reg(acc_w, slots.acc, RAX);

        self.enc.mov_reg_imm32(W::B8, RAX, 0);
        self.enc.mov_rbp_reg(W::B8, slots.k, RAX);
        self.enc.label(&k_check);
        self.enc.mov_reg_rbp(W::B8, RAX, slots.k);
        self.enc.movabs(RCX, k as i64);
        self.enc.alu_reg_reg(AluOp::Cmp, W::B8, RAX, RCX);
        self.enc.jcc(cc::GE, &k_end);

        if acc_float {
            self.mma_addr(R10, a_disp, a_row, a_col, a_leading, in_bytes);
            self.load_mma_float(0, R10, in_f64, acc_f64);
            self.mma_addr(R10, b_disp, b_row, b_col, b_leading, in_bytes);
            self.load_mma_float(1, R10, in_f64, acc_f64);
            self.enc.sse_arith(SseArith::Mul, acc_f64, 0, Rm::Direct(1));
            if acc_f64 {
                self.enc.movsd_load(2, Rm::RbpDisp(slots.acc));
            } else {
                self.enc.movss_load(2, Rm::RbpDisp(slots.acc));
            }
            self.enc.sse_arith(SseArith::Add, acc_f64, 2, Rm::Direct(0));
            if acc_f64 {
                self.enc.movsd_store(Rm::RbpDisp(slots.acc), 2);
            } else {
                self.enc.movss_store(Rm::RbpDisp(slots.acc), 2);
            }
        } else {
            self.mma_addr(R10, a_disp, a_row, a_col, a_leading, in_bytes);
            self.load_mma_int(RDX, R10, in_w, cw);
            self.mma_addr(R10, b_disp, b_row, b_col, b_leading, in_bytes);
            self.load_mma_int(RAX, R10, in_w, cw);
            self.enc.imul_reg_reg(cw, RDX, RAX);
            if cw == acc_w {
                self.enc.mov_reg_rbp(acc_w, RCX, slots.acc);
            } else {
                self.enc.movzx(cw, acc_w, RCX, Rm::RbpDisp(slots.acc));
            }
            self.enc.alu_reg_reg(AluOp::Add, cw, RCX, RDX);
            self.enc.mov_rbp_reg(acc_w, slots.acc, RCX);
        }

        self.enc.mov_reg_rbp(W::B8, RAX, slots.k);
        self.enc.alu_reg_imm32(AluOp::Add, W::B8, RAX, 1);
        self.enc.mov_rbp_reg(W::B8, slots.k, RAX);
        self.enc.jmp(&k_check);
        self.enc.label(&k_end);

        if acc_float {
            self.mma_addr(R10, c_disp, slots.i, slots.j, n, acc_bytes);
            if acc_f64 {
                self.enc.movsd_load(0, Rm::IndBase(R10));
            } else {
                self.enc.movss_load(0, Rm::IndBase(R10));
            }
            if acc_f64 {
                self.enc.movsd_load(1, Rm::RbpDisp(slots.acc));
            } else {
                self.enc.movss_load(1, Rm::RbpDisp(slots.acc));
            }
            self.enc.sse_arith(SseArith::Add, acc_f64, 0, Rm::Direct(1));
            self.mma_addr(R10, d_disp, slots.i, slots.j, n, acc_bytes);
            if acc_f64 {
                self.enc.movsd_store(Rm::IndBase(R10), 0);
            } else {
                self.enc.movss_store(Rm::IndBase(R10), 0);
            }
        } else {
            self.mma_addr(R10, c_disp, slots.i, slots.j, n, acc_bytes);
            self.enc.mov_reg_ind(acc_w, RAX, R10);
            self.enc.mov_reg_rbp(acc_w, RCX, slots.acc);
            self.enc.alu_reg_reg(AluOp::Add, acc_w, RAX, RCX);
            self.mma_addr(R10, d_disp, slots.i, slots.j, n, acc_bytes);
            self.enc.mov_ind_reg(acc_w, R10, RAX);
        }

        self.enc.mov_reg_rbp(W::B8, RAX, slots.j);
        self.enc.alu_reg_imm32(AluOp::Add, W::B8, RAX, 1);
        self.enc.mov_rbp_reg(W::B8, slots.j, RAX);
        self.enc.jmp(&j_check);
        self.enc.label(&j_end);

        self.enc.mov_reg_rbp(W::B8, RAX, slots.i);
        self.enc.alu_reg_imm32(AluOp::Add, W::B8, RAX, 1);
        self.enc.mov_rbp_reg(W::B8, slots.i, RAX);
        self.enc.jmp(&i_check);
        self.enc.label(&i_end);
    }

    fn emit_phi_copies(&mut self, from: u32, to: u32) {
        let Some(copies) = self.phi_copies.get(&(from, to)).cloned() else {
            return;
        };
        for (dest_disp, val, ty) in copies {
            if is_float(ty) {
                let f64_ = is_f64(ty);
                self.reload_xmm(val, 0, f64_);
                if f64_ {
                    self.enc.movsd_store(Rm::RbpDisp(dest_disp), 0);
                } else {
                    self.enc.movss_store(Rm::RbpDisp(dest_disp), 0);
                }
            } else {
                let w = width_of(ty);
                self.reload_gpr(val, RAX, w);
                self.enc.mov_rbp_reg(w, dest_disp, RAX);
            }
        }
    }

    fn lower_term(&mut self, from_block: u32, term: &Term) {
        match term {
            Term::Br(target) => {
                self.emit_phi_copies(from_block, target.0);
                let label = self.block_label(target.0);
                self.enc.jmp(&label);
            }
            Term::CondBr(cond, t, f) => {
                self.reload_gpr(*cond, RAX, W::B1);
                self.enc.test_reg_reg(W::B1, RAX);
                let false_prep = self.fresh_label("condbr_false");
                self.enc.jcc(cc::E, &false_prep);
                self.emit_phi_copies(from_block, t.0);
                let t_label = self.block_label(t.0);
                self.enc.jmp(&t_label);
                self.enc.label(&false_prep);
                self.emit_phi_copies(from_block, f.0);
                let f_label = self.block_label(f.0);
                self.enc.jmp(&f_label);
            }
            Term::Switch(scrut, default, cases) => {
                let ty = self.valref_ty(*scrut);
                let w = width_of(ty);
                self.reload_gpr(*scrut, RAX, w);
                for &(case_val, target) in cases {
                    self.enc.movabs(RCX, case_val);
                    self.enc.alu_reg_reg(AluOp::Cmp, w, RAX, RCX);
                    let skip = self.fresh_label("switch_skip");
                    self.enc.jcc(cc::NE, &skip);
                    self.emit_phi_copies(from_block, target.0);
                    let target_label = self.block_label(target.0);
                    self.enc.jmp(&target_label);
                    self.enc.label(&skip);
                }
                self.emit_phi_copies(from_block, default.0);
                let default_label = self.block_label(default.0);
                self.enc.jmp(&default_label);
            }
            Term::Ret(v) => {
                if let Some(val) = v {
                    let rty = self.f.ret;
                    let disp = self
                        .frame
                        .retval_home
                        .expect("non-void Ret always has a retval home");
                    if is_float(rty) {
                        let f64_ = is_f64(rty);
                        self.reload_xmm(*val, 0, f64_);
                        if f64_ {
                            self.enc.movsd_store(Rm::RbpDisp(disp), 0);
                        } else {
                            self.enc.movss_store(Rm::RbpDisp(disp), 0);
                        }
                    } else {
                        let w = width_of(rty);
                        self.reload_gpr(*val, RAX, w);
                        self.enc.mov_rbp_reg(w, disp, RAX);
                    }
                }
                // A kernel's own thread must still advance the loop rather than actually
                // return (`ret_target` is the loop's own increment step there); a host
                // function's own `ret_target` is its real epilogue, since it runs once like
                // any ordinary function. Either way, never a bare `ret` here.
                let target = self.ret_target.clone();
                self.enc.jmp(&target);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use basalt_bir::{Block, BlockId, Inst, LaunchBounds};
    use object::read::{Object as ReadObject, ObjectSection, ObjectSymbol};

    fn wrap(f: Function) -> Module {
        Module {
            funcs: vec![f],
            launch_bounds: None,
            shared_mem_bytes: 0,
            target_dtypes: vec![],
        }
    }

    fn parses_as_elf_with_symbol<'a>(bytes: &'a [u8], symbol: &str) -> object::read::File<'a> {
        let file = object::read::File::parse(bytes).expect("parses as an object file");
        assert_eq!(file.format(), object::BinaryFormat::Elf);
        assert_eq!(file.architecture(), object::Architecture::X86_64);
        let text = file
            .section_by_name(".text")
            .expect(".text section present");
        assert!(!text.data().unwrap().is_empty(), ".text must not be empty");
        let sym = file
            .symbols()
            .find(|s| s.name() == Ok(symbol))
            .unwrap_or_else(|| panic!("symbol `{symbol}` present"));
        assert_eq!(sym.size(), text.data().unwrap().len() as u64);
        file
    }

    // ---- fixtures --------------------------------------------------------------------

    /// `func @ret_const() -> i32 { bb0: %0 = const.i i32 42; ret %0 }`
    ///
    /// Expected native shape (params=[], so the only incoming register is `nthreads` in
    /// `rdi`):
    /// ```asm
    /// push rbp; mov rbp,rsp; sub rsp,FRAME_SIZE
    /// mov [rbp-nthreads_home], rdi
    /// mov qword [rbp-loopctr_home], 0
    /// __loop_check: mov rax,[loopctr]; cmp rax,[nthreads]; jge __loop_end
    /// bb0:
    ///   movabs rax, 42; mov [slot0], rax        ; %0 = const.i 42, stored (only low 4
    ///                                            ; bytes of slot0 are ever meaningfully
    ///                                            ; read back, since %0 : i32)
    ///   mov eax,[slot0]; mov [retval_home], eax  ; ret %0 (i32-width)
    ///   jmp __loop_incr
    /// __loop_incr: mov rax,[loopctr]; add rax,1; mov [loopctr],rax; jmp __loop_check
    /// __loop_end: mov eax,[retval_home]; mov rsp,rbp; pop rbp; ret
    /// ```
    fn func_ret_const() -> Function {
        Function {
            is_kernel: true,
            name: "ret_const".into(),
            params: vec![],
            ret: Ty::Scalar(Scalar::I32),
            insts: vec![Inst {
                ty: Ty::Scalar(Scalar::I32),
                op: Op::ConstInt(42),
            }],
            blocks: vec![Block {
                insts: vec![InstId(0)],
                term: Term::Ret(Some(ValRef::Val(InstId(0)))),
            }],
        }
    }

    /// `func @add_i32(i32, i32) -> i32 { bb0: %0 = add i32 %arg0, %arg1; ret %0 }`
    ///
    /// Both params are integer-class, so `a` arrives in `edi`, `b` in `esi`, and — since
    /// SysV classification continues past the function's own params — `nthreads` lands in
    /// `rdx` (the third integer-class register), not `rdi`/`rsi`.
    /// ```asm
    /// mov [param0], edi; mov [param1], esi; mov [nthreads], rdx
    /// ...
    /// bb0:
    ///   mov eax,[param0]; mov ecx,[param1]; add eax,ecx    ; 32-bit add, no REX.W
    ///   mov [slot0], eax
    ///   mov eax,[slot0]; mov [retval_home], eax; jmp __loop_incr
    /// ```
    fn func_add_i32() -> Function {
        Function {
            is_kernel: true,
            name: "add_i32".into(),
            params: vec![Ty::Scalar(Scalar::I32), Ty::Scalar(Scalar::I32)],
            ret: Ty::Scalar(Scalar::I32),
            insts: vec![Inst {
                ty: Ty::Scalar(Scalar::I32),
                op: Op::Bin(BinOp::Add, ValRef::Param(0), ValRef::Param(1)),
            }],
            blocks: vec![Block {
                insts: vec![InstId(0)],
                term: Term::Ret(Some(ValRef::Val(InstId(0)))),
            }],
        }
    }

    /// `func @max_i32(i32, i32) -> i32`: `if (%arg0 > %arg1) return %arg0; else return
    /// %arg1;`, lowered to `icmp` + `condbr` + two single-instruction-free blocks.
    /// ```asm
    /// bb0:
    ///   mov eax,[param0]; mov ecx,[param1]; cmp eax,ecx; setg al   ; %0 = icmp sgt
    ///   mov [slot0], al
    ///   mov al,[slot0]; test al,al; jz .false
    ///   jmp bb1                                                    ; (true edge, no phis)
    /// .false:
    ///   jmp bb2                                                    ; (false edge, no phis)
    /// bb1: mov eax,[param0]; mov [retval_home],eax; jmp __loop_incr
    /// bb2: mov eax,[param1]; mov [retval_home],eax; jmp __loop_incr
    /// ```
    fn func_max_i32() -> Function {
        Function {
            is_kernel: true,
            name: "max_i32".into(),
            params: vec![Ty::Scalar(Scalar::I32), Ty::Scalar(Scalar::I32)],
            ret: Ty::Scalar(Scalar::I32),
            insts: vec![Inst {
                ty: Ty::Scalar(Scalar::I1),
                op: Op::ICmp(
                    ICmpPred::Sgt,
                    Ty::Scalar(Scalar::I32),
                    ValRef::Param(0),
                    ValRef::Param(1),
                ),
            }],
            blocks: vec![
                Block {
                    insts: vec![InstId(0)],
                    term: Term::CondBr(ValRef::Val(InstId(0)), BlockId(1), BlockId(2)),
                },
                Block {
                    insts: vec![],
                    term: Term::Ret(Some(ValRef::Param(0))),
                },
                Block {
                    insts: vec![],
                    term: Term::Ret(Some(ValRef::Param(1))),
                },
            ],
        }
    }

    /// `func @write_idx(ptr.global) -> void`: every thread writes its own flat index into
    /// `out[tid.x]` as an `i32` — exercises `tid.x`, `zext`, integer `mul`, a same-width
    /// `bitcast` reinterpreting a byte offset as a pointer, pointer `add`, and `store` to
    /// `global` memory, all inside the oracle's own per-thread loop (the "small loop using
    /// the thread index" case).
    fn func_write_idx() -> Function {
        Function {
            is_kernel: true,
            name: "write_idx".into(),
            params: vec![Ty::Ptr(AddrSpace::Global)],
            ret: Ty::Void,
            insts: vec![
                Inst {
                    ty: Ty::Scalar(Scalar::I32),
                    op: Op::TidX,
                },
                Inst {
                    ty: Ty::Scalar(Scalar::I64),
                    op: Op::Cast(
                        CastOp::Zext,
                        Ty::Scalar(Scalar::I32),
                        ValRef::Val(InstId(0)),
                    ),
                },
                Inst {
                    ty: Ty::Scalar(Scalar::I64),
                    op: Op::ConstInt(4),
                },
                Inst {
                    ty: Ty::Scalar(Scalar::I64),
                    op: Op::Bin(BinOp::Mul, ValRef::Val(InstId(1)), ValRef::Val(InstId(2))),
                },
                Inst {
                    ty: Ty::Ptr(AddrSpace::Global),
                    op: Op::Cast(
                        CastOp::Bitcast,
                        Ty::Scalar(Scalar::I64),
                        ValRef::Val(InstId(3)),
                    ),
                },
                Inst {
                    ty: Ty::Ptr(AddrSpace::Global),
                    op: Op::Bin(BinOp::Add, ValRef::Param(0), ValRef::Val(InstId(4))),
                },
                Inst {
                    ty: Ty::Void,
                    op: Op::Store {
                        ptr: ValRef::Val(InstId(5)),
                        val: ValRef::Val(InstId(0)),
                        ty: Ty::Scalar(Scalar::I32),
                        space: AddrSpace::Global,
                        align: 4,
                        volatile: false,
                    },
                },
            ],
            blocks: vec![Block {
                insts: (0..7).map(InstId).collect(),
                term: Term::Ret(None),
            }],
        }
    }

    /// `func @mma2x2(ptr.global, ptr.global, ptr.global, ptr.global) -> void`: one `mma`
    /// instruction, `D = A*B + C` at `M=N=K=2`, row-major `A`/`B`, `f32` throughout — small
    /// enough to hand-verify. With `A = [[1,2],[3,4]]`, `B = [[5,6],[7,8]]`,
    /// `C = [[0.5,0.5],[0.5,0.5]]`: `A*B = [[19,22],[43,50]]`, so
    /// `D = [[19.5,22.5],[43.5,50.5]]` (see `link_and_run.rs` for the executed proof).
    fn func_mma2x2() -> Function {
        let ptr_global = Ty::Ptr(AddrSpace::Global);
        Function {
            is_kernel: true,
            name: "mma2x2".into(),
            params: vec![ptr_global, ptr_global, ptr_global, ptr_global],
            ret: Ty::Void,
            insts: vec![Inst {
                ty: Ty::Void,
                op: Op::Mma {
                    a: ValRef::Param(0),
                    b: ValRef::Param(1),
                    c: ValRef::Param(2),
                    d: ValRef::Param(3),
                    m: 2,
                    n: 2,
                    k: 2,
                    in_dtype: Scalar::F32,
                    acc_dtype: Scalar::F32,
                    layout_a: MmaLayout::RowMajor,
                    layout_b: MmaLayout::RowMajor,
                },
            }],
            blocks: vec![Block {
                insts: vec![InstId(0)],
                term: Term::Ret(None),
            }],
        }
    }

    /// `func @mma_i8i32(ptr.global x4) -> void`: `M=N=K=2`, col-major `A`, row-major `B`,
    /// `i8` inputs into an `i32` accumulator — exercises the integer widening path
    /// (`load_mma_int`'s `movsx`) and a non-default layout in the same fixture.
    fn func_mma_i8_i32_colmajor_a() -> Function {
        let ptr_global = Ty::Ptr(AddrSpace::Global);
        Function {
            is_kernel: true,
            name: "mma_i8i32".into(),
            params: vec![ptr_global, ptr_global, ptr_global, ptr_global],
            ret: Ty::Void,
            insts: vec![Inst {
                ty: Ty::Void,
                op: Op::Mma {
                    a: ValRef::Param(0),
                    b: ValRef::Param(1),
                    c: ValRef::Param(2),
                    d: ValRef::Param(3),
                    m: 2,
                    n: 2,
                    k: 2,
                    in_dtype: Scalar::I8,
                    acc_dtype: Scalar::I32,
                    layout_a: MmaLayout::ColMajor,
                    layout_b: MmaLayout::RowMajor,
                },
            }],
            blocks: vec![Block {
                insts: vec![InstId(0)],
                term: Term::Ret(None),
            }],
        }
    }

    // ---- supports() --------------------------------------------------------------------

    #[test]
    fn supports_a_module_using_only_implemented_ops() {
        assert_eq!(
            X86Oracle.supports(&wrap(func_ret_const())),
            Support::Supported
        );
        assert_eq!(
            X86Oracle.supports(&wrap(func_add_i32())),
            Support::Supported
        );
        assert_eq!(
            X86Oracle.supports(&wrap(func_max_i32())),
            Support::Supported
        );
        assert_eq!(
            X86Oracle.supports(&wrap(func_write_idx())),
            Support::Supported
        );
        assert_eq!(X86Oracle.supports(&wrap(func_mma2x2())), Support::Supported);
        assert_eq!(
            X86Oracle.supports(&wrap(func_mma_i8_i32_colmajor_a())),
            Support::Supported
        );
    }

    #[test]
    fn refuses_mma_i1_input_with_e091() {
        let mut f = func_mma2x2();
        let Op::Mma {
            in_dtype,
            acc_dtype,
            ..
        } = &mut f.insts[0].op
        else {
            unreachable!()
        };
        *in_dtype = Scalar::I1;
        *acc_dtype = Scalar::I32;
        assert_eq!(
            X86Oracle.supports(&wrap(f)),
            Support::Unsupported(ECode::UnsupportedType)
        );
    }

    #[test]
    fn refuses_mma_f16_accumulator_with_e091() {
        let mut f = func_mma2x2();
        let Op::Mma { acc_dtype, .. } = &mut f.insts[0].op else {
            unreachable!()
        };
        *acc_dtype = Scalar::F16;
        assert_eq!(
            X86Oracle.supports(&wrap(f)),
            Support::Unsupported(ECode::UnsupportedType)
        );
    }

    #[test]
    fn refuses_mma_mixed_int_float_dtypes_with_e091() {
        let mut f = func_mma2x2();
        let Op::Mma {
            in_dtype,
            acc_dtype,
            ..
        } = &mut f.insts[0].op
        else {
            unreachable!()
        };
        *in_dtype = Scalar::I32;
        *acc_dtype = Scalar::F32;
        assert_eq!(
            X86Oracle.supports(&wrap(f)),
            Support::Unsupported(ECode::UnsupportedType)
        );
    }

    #[test]
    fn refuses_mma_narrowing_accumulator_with_e091() {
        let mut f = func_mma2x2();
        let Op::Mma {
            in_dtype,
            acc_dtype,
            ..
        } = &mut f.insts[0].op
        else {
            unreachable!()
        };
        *in_dtype = Scalar::F64;
        *acc_dtype = Scalar::F32;
        assert_eq!(
            X86Oracle.supports(&wrap(f)),
            Support::Unsupported(ECode::UnsupportedType)
        );
    }

    #[test]
    fn refuses_shuffle_with_e090() {
        let f = Function {
            is_kernel: true,
            name: "shuf".into(),
            params: vec![Ty::Scalar(Scalar::I32)],
            ret: Ty::Void,
            insts: vec![Inst {
                ty: Ty::Scalar(Scalar::I32),
                op: Op::Shuffle(
                    basalt_bir::ShuffleKind::Idx,
                    ValRef::Param(0),
                    ValRef::Param(0),
                ),
            }],
            blocks: vec![Block {
                insts: vec![InstId(0)],
                term: Term::Ret(None),
            }],
        };
        assert_eq!(
            X86Oracle.supports(&wrap(f)),
            Support::Unsupported(ECode::UnsupportedOp)
        );
    }

    /// P13-T1b's kernel-launch/CUDA-Runtime-API ops are sema-only today (see
    /// `basalt_bir::Op::KernelLaunch`'s own doc comment): a real oracle lowering needs a
    /// genuine call/return mechanism this backend does not have yet. Every backend refuses
    /// them cleanly.
    #[test]
    fn refuses_kernel_launch_and_cuda_runtime_api_ops_with_e090() {
        let f = Function {
            is_kernel: true,
            name: "launch_stub".into(),
            params: vec![],
            ret: Ty::Void,
            insts: vec![Inst {
                ty: Ty::Void,
                op: Op::CudaDeviceSynchronize,
            }],
            blocks: vec![Block {
                insts: vec![InstId(0)],
                term: Term::Ret(None),
            }],
        };
        assert_eq!(
            X86Oracle.supports(&wrap(f)),
            Support::Unsupported(ECode::UnsupportedOp)
        );
    }

    #[test]
    fn refuses_ballot_vote_with_e090() {
        for op in [
            Op::Ballot(ValRef::Param(0)),
            Op::VoteAny(ValRef::Param(0)),
            Op::VoteAll(ValRef::Param(0)),
        ] {
            let f = Function {
                is_kernel: true,
                name: "vote".into(),
                params: vec![Ty::Scalar(Scalar::I1)],
                ret: Ty::Void,
                insts: vec![Inst {
                    ty: Ty::Scalar(Scalar::I1),
                    op,
                }],
                blocks: vec![Block {
                    insts: vec![InstId(0)],
                    term: Term::Ret(None),
                }],
            };
            assert_eq!(
                X86Oracle.supports(&wrap(f)),
                Support::Unsupported(ECode::UnsupportedOp)
            );
        }
    }

    #[test]
    fn refuses_vector_result_with_e091() {
        let f = Function {
            is_kernel: true,
            name: "vecty".into(),
            params: vec![],
            ret: Ty::Void,
            insts: vec![Inst {
                ty: Ty::Vec(Scalar::F32, 4),
                op: Op::ConstFloat(1.0),
            }],
            blocks: vec![Block {
                insts: vec![InstId(0)],
                term: Term::Ret(None),
            }],
        };
        assert_eq!(
            X86Oracle.supports(&wrap(f)),
            Support::Unsupported(ECode::UnsupportedType)
        );
    }

    #[test]
    fn refuses_vector_return_with_e091() {
        let f = Function {
            is_kernel: true,
            name: "vecret".into(),
            params: vec![],
            ret: Ty::Vec(Scalar::I32, 4),
            insts: vec![],
            blocks: vec![Block {
                insts: vec![],
                term: Term::Ret(None),
            }],
        };
        assert_eq!(
            X86Oracle.supports(&wrap(f)),
            Support::Unsupported(ECode::UnsupportedType)
        );
    }

    #[test]
    fn refuses_multi_function_module_with_e093() {
        let module = Module {
            funcs: vec![func_ret_const(), func_add_i32()],
            launch_bounds: None,
            shared_mem_bytes: 0,
            target_dtypes: vec![],
        };
        assert_eq!(
            X86Oracle.supports(&module),
            Support::Unsupported(ECode::UnsupportedFeature)
        );
    }

    #[test]
    fn refuses_non_kernel_function_with_e093() {
        let mut f = func_ret_const();
        f.is_kernel = false;
        assert_eq!(
            X86Oracle.supports(&wrap(f)),
            Support::Unsupported(ECode::UnsupportedFeature)
        );
    }

    // ---- host function + real kernel-launch call (P13-T1c-i) ----------------------------

    /// A host function launching `add_i32` (see `func_add_i32`) with a real
    /// `Op::KernelLaunch`: `grid=(2,1,1)`, `block=(3,1,1)` (so `nthreads` at the call site
    /// should be the flattened product `6`), args `(10, 20)`, `shared`/`stream` both the
    /// documented default (`Op::ConstInt(0)`). Instruction indices: 0-2 grid.xyz, 3-5
    /// block.xyz, 6 shared, 7 stream, 8-9 the two launch args, 10 the launch itself.
    fn func_host_launches_add_i32() -> Function {
        let i32t = Ty::Scalar(Scalar::I32);
        let i64t = Ty::Scalar(Scalar::I64);
        let ptr_global = Ty::Ptr(AddrSpace::Global);
        Function {
            is_kernel: false,
            name: "host_main".into(),
            params: vec![],
            ret: Ty::Void,
            insts: vec![
                Inst {
                    ty: i32t,
                    op: Op::ConstInt(2),
                }, // 0: grid.x
                Inst {
                    ty: i32t,
                    op: Op::ConstInt(1),
                }, // 1: grid.y
                Inst {
                    ty: i32t,
                    op: Op::ConstInt(1),
                }, // 2: grid.z
                Inst {
                    ty: i32t,
                    op: Op::ConstInt(3),
                }, // 3: block.x
                Inst {
                    ty: i32t,
                    op: Op::ConstInt(1),
                }, // 4: block.y
                Inst {
                    ty: i32t,
                    op: Op::ConstInt(1),
                }, // 5: block.z
                Inst {
                    ty: i64t,
                    op: Op::ConstInt(0),
                }, // 6: shared (default)
                Inst {
                    ty: ptr_global,
                    op: Op::ConstInt(0),
                }, // 7: stream (default)
                Inst {
                    ty: i32t,
                    op: Op::ConstInt(10),
                }, // 8: arg a
                Inst {
                    ty: i32t,
                    op: Op::ConstInt(20),
                }, // 9: arg b
                Inst {
                    ty: Ty::Void,
                    op: Op::KernelLaunch {
                        kernel: "add_i32".into(),
                        grid: [
                            ValRef::Val(InstId(0)),
                            ValRef::Val(InstId(1)),
                            ValRef::Val(InstId(2)),
                        ],
                        block: [
                            ValRef::Val(InstId(3)),
                            ValRef::Val(InstId(4)),
                            ValRef::Val(InstId(5)),
                        ],
                        shared: ValRef::Val(InstId(6)),
                        stream: ValRef::Val(InstId(7)),
                        args: vec![ValRef::Val(InstId(8)), ValRef::Val(InstId(9))],
                    },
                }, // 10
            ],
            blocks: vec![Block {
                insts: (0..=10).map(InstId).collect(),
                term: Term::Ret(None),
            }],
        }
    }

    fn host_and_kernel_module(host: Function, kernels: Vec<Function>) -> Module {
        let mut funcs = vec![host];
        funcs.extend(kernels);
        Module {
            funcs,
            launch_bounds: None,
            shared_mem_bytes: 0,
            target_dtypes: vec![],
        }
    }

    #[test]
    fn supports_and_emits_host_function_launching_a_kernel() {
        let module = host_and_kernel_module(func_host_launches_add_i32(), vec![func_add_i32()]);
        assert_eq!(X86Oracle.supports(&module), Support::Supported);

        let artifact = X86Oracle
            .emit(&module, &EmitOpts::default())
            .expect("emit succeeds for a host function launching a real kernel");
        let bytes = artifact.as_bytes().unwrap();
        let file = object::read::File::parse(bytes).expect("parses as an object file");
        let text = file.section_by_name(".text").expect(".text present");
        let text_len = text.data().unwrap().len() as u64;

        let host_sym = file
            .symbols()
            .find(|s| s.name() == Ok("host_main"))
            .expect("host symbol present");
        let kernel_sym = file
            .symbols()
            .find(|s| s.name() == Ok("add_i32"))
            .expect("kernel symbol present");
        assert_ne!(host_sym.address(), kernel_sym.address());
        assert!(host_sym.size() > 0);
        assert!(kernel_sym.size() > 0);
        assert!(host_sym.address() + host_sym.size() <= text_len);
        assert!(kernel_sym.address() + kernel_sym.size() <= text_len);
    }

    #[test]
    fn refuses_kernel_present_but_never_launched_with_e093() {
        let module = host_and_kernel_module(
            func_host_launches_add_i32(),
            vec![func_add_i32(), func_max_i32()],
        );
        assert_eq!(
            X86Oracle.supports(&module),
            Support::Unsupported(ECode::UnsupportedFeature)
        );
    }

    #[test]
    fn refuses_launch_naming_a_function_not_in_the_module_with_e093() {
        let module = host_and_kernel_module(func_host_launches_add_i32(), vec![]);
        assert_eq!(
            X86Oracle.supports(&module),
            Support::Unsupported(ECode::UnsupportedFeature)
        );
    }

    #[test]
    fn refuses_non_default_shared_operand_with_e093() {
        let mut host = func_host_launches_add_i32();
        host.insts[6].op = Op::ConstInt(4096); // shared: non-default, non-zero
        let module = host_and_kernel_module(host, vec![func_add_i32()]);
        assert_eq!(
            X86Oracle.supports(&module),
            Support::Unsupported(ECode::UnsupportedFeature)
        );
    }

    #[test]
    fn refuses_non_default_stream_operand_with_e093() {
        let mut host = func_host_launches_add_i32();
        host.insts[7].op = Op::ConstInt(7); // stream: non-default, non-null
        let module = host_and_kernel_module(host, vec![func_add_i32()]);
        assert_eq!(
            X86Oracle.supports(&module),
            Support::Unsupported(ECode::UnsupportedFeature)
        );
    }

    #[test]
    fn refuses_cuda_malloc_inside_host_function_with_e090() {
        let mut host = func_host_launches_add_i32();
        let malloc_id = InstId(host.insts.len() as u32);
        host.insts.push(Inst {
            ty: Ty::Ptr(AddrSpace::Global),
            op: Op::CudaMalloc {
                size: ValRef::Val(InstId(6)),
            },
        });
        host.blocks[0].insts.push(malloc_id);
        let module = host_and_kernel_module(host, vec![func_add_i32()]);
        assert_eq!(
            X86Oracle.supports(&module),
            Support::Unsupported(ECode::UnsupportedOp)
        );
    }

    /// `Op::CudaDeviceSynchronize` inside a host function is a real no-op (every launch this
    /// backend accepts already runs synchronously inside its own `call`), not refused.
    #[test]
    fn supports_cuda_device_synchronize_inside_host_function() {
        let mut host = func_host_launches_add_i32();
        let sync_id = InstId(host.insts.len() as u32);
        host.insts.push(Inst {
            ty: Ty::Void,
            op: Op::CudaDeviceSynchronize,
        });
        host.blocks[0].insts.push(sync_id);
        let module = host_and_kernel_module(host, vec![func_add_i32()]);
        assert_eq!(X86Oracle.supports(&module), Support::Supported);
        X86Oracle
            .emit(&module, &EmitOpts::default())
            .expect("emit succeeds with a real cudaDeviceSynchronize no-op");
    }

    #[test]
    fn refuses_two_host_functions_with_e093() {
        let mut second_host = func_host_launches_add_i32();
        second_host.name = "other_host".into();
        let module = host_and_kernel_module(
            func_host_launches_add_i32(),
            vec![second_host, func_add_i32()],
        );
        assert_eq!(
            X86Oracle.supports(&module),
            Support::Unsupported(ECode::UnsupportedFeature)
        );
    }

    #[test]
    fn refuses_too_many_integer_params_with_e093() {
        // 6 integer-class params leaves no register for the trailing `nthreads` argument.
        let f = Function {
            is_kernel: true,
            name: "toomany".into(),
            params: vec![Ty::Scalar(Scalar::I32); 6],
            ret: Ty::Void,
            insts: vec![],
            blocks: vec![Block {
                insts: vec![],
                term: Term::Ret(None),
            }],
        };
        assert_eq!(
            X86Oracle.supports(&wrap(f)),
            Support::Unsupported(ECode::UnsupportedFeature)
        );
    }

    #[test]
    fn refuses_f16_arithmetic_with_e091() {
        let f = Function {
            is_kernel: true,
            name: "halfty".into(),
            params: vec![],
            ret: Ty::Void,
            insts: vec![Inst {
                ty: Ty::Scalar(Scalar::F16),
                op: Op::ConstFloat(1.0),
            }],
            blocks: vec![Block {
                insts: vec![],
                term: Term::Ret(None),
            }],
        };
        assert_eq!(
            X86Oracle.supports(&wrap(f)),
            Support::Unsupported(ECode::UnsupportedType)
        );
    }

    // ---- emit() -------------------------------------------------------------------------

    #[test]
    fn emits_valid_elf_for_ret_const() {
        let artifact = X86Oracle
            .emit(&wrap(func_ret_const()), &EmitOpts::default())
            .expect("emit succeeds");
        assert_eq!(artifact.kind, ArtifactKind::Object);
        parses_as_elf_with_symbol(artifact.as_bytes().unwrap(), "ret_const");
    }

    #[test]
    fn emits_valid_elf_for_add_i32() {
        let artifact = X86Oracle
            .emit(&wrap(func_add_i32()), &EmitOpts::default())
            .expect("emit succeeds");
        parses_as_elf_with_symbol(artifact.as_bytes().unwrap(), "add_i32");
    }

    #[test]
    fn emits_valid_elf_for_condbr() {
        let artifact = X86Oracle
            .emit(&wrap(func_max_i32()), &EmitOpts::default())
            .expect("emit succeeds");
        parses_as_elf_with_symbol(artifact.as_bytes().unwrap(), "max_i32");
    }

    #[test]
    fn emits_valid_elf_for_thread_index_loop() {
        let artifact = X86Oracle
            .emit(&wrap(func_write_idx()), &EmitOpts::default())
            .expect("emit succeeds");
        parses_as_elf_with_symbol(artifact.as_bytes().unwrap(), "write_idx");
    }

    #[test]
    fn emits_valid_elf_for_mma() {
        let artifact = X86Oracle
            .emit(&wrap(func_mma2x2()), &EmitOpts::default())
            .expect("emit succeeds");
        parses_as_elf_with_symbol(artifact.as_bytes().unwrap(), "mma2x2");
    }

    #[test]
    fn emits_valid_elf_for_mma_int_colmajor() {
        let artifact = X86Oracle
            .emit(&wrap(func_mma_i8_i32_colmajor_a()), &EmitOpts::default())
            .expect("emit succeeds");
        parses_as_elf_with_symbol(artifact.as_bytes().unwrap(), "mma_i8i32");
    }

    #[test]
    fn emit_refuses_mma_dtype_pair_supports_refuses() {
        let mut f = func_mma2x2();
        let Op::Mma { acc_dtype, .. } = &mut f.insts[0].op else {
            unreachable!()
        };
        *acc_dtype = Scalar::F16;
        assert_eq!(
            X86Oracle.supports(&wrap(f.clone())),
            Support::Unsupported(ECode::UnsupportedType)
        );
        let err = X86Oracle
            .emit(&wrap(f), &EmitOpts::default())
            .expect_err("must refuse, not guess");
        assert_eq!(err.code, ECode::UnsupportedType);
    }

    #[test]
    fn emit_refuses_what_supports_refuses() {
        let f = Function {
            is_kernel: true,
            name: "shuf".into(),
            params: vec![Ty::Scalar(Scalar::I32)],
            ret: Ty::Void,
            insts: vec![Inst {
                ty: Ty::Scalar(Scalar::I32),
                op: Op::Shuffle(
                    basalt_bir::ShuffleKind::Idx,
                    ValRef::Param(0),
                    ValRef::Param(0),
                ),
            }],
            blocks: vec![Block {
                insts: vec![InstId(0)],
                term: Term::Ret(None),
            }],
        };
        let err = X86Oracle
            .emit(&wrap(f), &EmitOpts::default())
            .expect_err("must refuse, not guess");
        assert_eq!(err.code, ECode::UnsupportedOp);
    }

    #[test]
    fn emit_is_deterministic() {
        let module = wrap(func_write_idx());
        let a = X86Oracle.emit(&module, &EmitOpts::default()).unwrap();
        let b = X86Oracle.emit(&module, &EmitOpts::default()).unwrap();
        assert_eq!(
            a, b,
            "same module in must yield byte-identical artifact out"
        );
    }

    #[test]
    fn name_is_stable() {
        assert_eq!(X86Oracle.name(), "x86-oracle");
    }

    /// Sanity check that the fixtures above are not vacuously trivial: a module can also
    /// carry the metadata BIR allows (launch bounds), which this backend simply ignores.
    #[test]
    fn ignores_launch_bounds_metadata() {
        let mut module = wrap(func_ret_const());
        module.launch_bounds = Some(LaunchBounds {
            max_threads: 128,
            min_blocks: 2,
        });
        assert_eq!(X86Oracle.supports(&module), Support::Supported);
    }
}
