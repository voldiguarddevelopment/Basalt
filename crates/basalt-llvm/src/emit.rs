// TargetMachine-based object emission: turns the in-memory IR `lower_module` builds into real
// object bytes for a concrete target triple, via LLVM's own codegen. `lower_module` itself
// never touches a `TargetMachine`, never emits bytes — this is the layer above it, and the
// only place in this crate that does.
//
// # Target choices
//
// Triple/CPU/reloc-mode are fixed per `LlvmTarget` rather than exposed as knobs — this lane's
// job is proving the object-emission path works and cross-checking against the hand-rolled
// oracle, not offering a target matrix:
//   - Nvptx: `nvptx64-nvidia-cuda` / `sm_70`, matching `basalt-ptx`'s own documented floor
//     (see that crate's `.target sm_70` emission) so the hand-rolled and LLVM-backed NVPTX
//     lanes agree on a baseline part.
//   - Amdgcn: `amdgcn-amd-amdhsa` / `gfx1100`, an RDNA3 part.
//   - X86: `x86_64-unknown-linux-gnu` / `x86-64`, the generic SysV baseline — this lane is a
//     correctness cross-check against the x86 oracle, not a performance target.
//
// AMDGCN code objects are conventionally position-independent (an HSACO loads at a
// runtime-chosen address), so `Amdgcn` uses `RelocMode::PIC`; the other two targets have no
// such requirement and use LLVM's plain default. Every target uses `OptimizationLevel::None`:
// this lane exists to prove correctness, not to compete on speed.
//
// # What `FileType::Object` actually produces per target
//
// x86/Amdgcn: a genuine ELF relocatable object, the same shape `write_elf_object` produces by
// hand elsewhere in this project. Nvptx: LLVM's NVPTX backend has **no object-file writer at
// all** — asked for `FileType::Object` it returns LLVM's own "TargetMachine can't emit a file
// of this type" error, a clean `LLVMString` `emit_object` turns into an ordinary `Err(Diag)`,
// not a crash and not a silent fallback to another format. This is confirmed empirically, not
// assumed (see `emit/tests.rs`): the identical `TargetMachine` asked for `FileType::Assembly`
// instead succeeds and prints ordinary PTX text (`.version 6.0`, `.target sm_70`, `.visible
// .func ...` — LLVM 18's NVPTX backend defaults to PTX ISA 6.0 when none is requested, older
// than the `.version 8.0` `basalt-ptx` emits by hand). So `emit_object(..., LlvmTarget::Nvptx)`
// always returns `Err(UnsupportedFeature)` today; the NVPTX backend itself works fine, only
// its object-file path is missing in this LLVM build.
//
// # X86 and BIR's GPU index ops
//
// A BIR kernel reads `tid.x`/`bid.x`/etc. directly; `lower_module` only knows how to turn
// those into real GPU intrinsics (see its own header), which have no meaning on x86. Before
// handing a module to `lower_module` for `LlvmTarget::X86`, `emit_object` checks whether the
// module actually uses one of those ops and, if so, runs it through `cpu_flatten` first — the
// same "wrap the whole kernel body in a native loop" convention `basalt-x86`'s oracle already
// uses, so the two independent x86 codegen paths are lowering the identical convention and
// their outputs are actually comparable. A module with no GPU index op (an ordinary scalar
// function) is left untouched.

use inkwell::context::Context;
use inkwell::targets::{
    CodeModel, FileType, InitializationConfig, RelocMode, Target, TargetMachine, TargetTriple,
};
use inkwell::OptimizationLevel;

use basalt_backend::{Artifact, ArtifactKind, Backend, EmitOpts, Support};
use basalt_bir::Module;
use basalt_diag::{Diag, ECode};

use crate::lower::{lower_module, GpuDialect};

/// Which real target `emit_object` compiles a module's LLVM IR down to. Distinct from
/// `GpuDialect`: that picks an intrinsic family at IR-build time (`lower_module`'s own
/// concern); this picks the `TargetMachine` that then compiles whichever IR came out of it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LlvmTarget {
    Nvptx,
    Amdgcn,
    X86,
}

impl LlvmTarget {
    fn triple(self) -> &'static str {
        match self {
            LlvmTarget::Nvptx => "nvptx64-nvidia-cuda",
            LlvmTarget::Amdgcn => "amdgcn-amd-amdhsa",
            LlvmTarget::X86 => "x86_64-unknown-linux-gnu",
        }
    }

    fn cpu(self) -> &'static str {
        match self {
            LlvmTarget::Nvptx => "sm_70",
            LlvmTarget::Amdgcn => "gfx1100",
            LlvmTarget::X86 => "x86-64",
        }
    }

    fn reloc_mode(self) -> RelocMode {
        match self {
            LlvmTarget::Amdgcn => RelocMode::PIC,
            LlvmTarget::Nvptx | LlvmTarget::X86 => RelocMode::Default,
        }
    }

    /// Which GPU intrinsic family `lower_module` should build the IR against. `X86` has no
    /// GPU concept at all; `lower_module` never consults `dialect` unless the module actually
    /// contains a GPU op, and a module reaching `lower_module` for `X86` never does by the
    /// time it gets there (see `emit_object`'s own use of `cpu_flatten`) — so `Nvptx` here is
    /// an arbitrary but inert placeholder, not a real choice.
    fn dialect(self) -> GpuDialect {
        match self {
            LlvmTarget::Nvptx => GpuDialect::Nvptx,
            LlvmTarget::Amdgcn => GpuDialect::Amdgpu,
            LlvmTarget::X86 => GpuDialect::Nvptx,
        }
    }

    fn initialize(self) {
        match self {
            LlvmTarget::Nvptx => Target::initialize_nvptx(&InitializationConfig::default()),
            LlvmTarget::Amdgcn => Target::initialize_amd_gpu(&InitializationConfig::default()),
            LlvmTarget::X86 => Target::initialize_x86(&InitializationConfig::default()),
        }
    }
}

fn target_machine(target: LlvmTarget) -> Result<(TargetMachine, TargetTriple), Diag> {
    target.initialize();
    let triple = TargetTriple::create(target.triple());
    let llvm_target = Target::from_triple(&triple).map_err(|e| {
        Diag::new(ECode::UnsupportedFeature).with_arg(format!(
            "this LLVM build has no target registered for triple `{}`: {e}",
            target.triple()
        ))
    })?;
    let tm = llvm_target
        .create_target_machine(
            &triple,
            target.cpu(),
            "",
            OptimizationLevel::None,
            target.reloc_mode(),
            CodeModel::Default,
        )
        .ok_or_else(|| {
            Diag::new(ECode::UnsupportedFeature).with_arg(format!(
                "LLVM could not create a target machine for `{}`/`{}`",
                target.triple(),
                target.cpu()
            ))
        })?;
    Ok((tm, triple))
}

/// Lowers `module` to LLVM IR (via `lower_module`, picking the GPU intrinsic dialect `target`
/// implies) and compiles it through a real `TargetMachine` into object bytes. For `X86`, a
/// module that actually uses a GPU index op is rewritten by `cpu_flatten` first (see the
/// module header). The module is verified before codegen ever starts — a well-formed module
/// out of `lower_module` should always pass, so a failure here is this crate's own bug, not a
/// normal refusal, and panics like this crate's other "should never happen" invariants rather
/// than threading it through `Result`. See the module header for what `FileType::Object`
/// actually contains per target.
pub fn emit_object(
    module: &Module,
    llvm_ctx: &Context,
    target: LlvmTarget,
) -> Result<Vec<u8>, Diag> {
    let flattened;
    let module = if target == LlvmTarget::X86 && crate::cpu_flatten::uses_gpu_index_ops(module) {
        flattened = crate::cpu_flatten::flatten_to_native_cpu_loop(module)?;
        &flattened
    } else {
        module
    };

    let llvm_mod = lower_module(module, llvm_ctx, target.dialect())?;
    llvm_mod
        .verify()
        .expect("this crate's own lowering always produces valid LLVM IR");

    let (tm, triple) = target_machine(target)?;
    llvm_mod.set_triple(&triple);
    llvm_mod.set_data_layout(&tm.get_target_data().get_data_layout());

    let buf = tm
        .write_to_memory_buffer(&llvm_mod, FileType::Object)
        .map_err(|e| {
            Diag::new(ECode::UnsupportedFeature)
                .with_arg(format!("LLVM object emission failed: {e}"))
        })?;
    Ok(buf.as_slice().to_vec())
}

/// The AMDGCN target-machine object-emission path, wrapped as an ordinary `Backend` so the
/// CLI can call it the same way it calls every hand-rolled backend. There is no hand-rolled
/// `basalt-amdgpu` backend yet (a later phase); until then this is the only path that turns
/// an AMDGCN-bound module into a real binary artifact.
#[derive(Debug, Default, Clone, Copy)]
pub struct LlvmAmdgcn;

impl Backend for LlvmAmdgcn {
    fn name(&self) -> &'static str {
        "llvm-amdgcn"
    }

    fn supports(&self, module: &Module) -> Support {
        let ctx = Context::create();
        match emit_object(module, &ctx, LlvmTarget::Amdgcn) {
            Ok(_) => Support::Supported,
            Err(diag) => Support::Unsupported(diag.code),
        }
    }

    fn emit(&self, module: &Module, _opts: &EmitOpts) -> Result<Artifact, Diag> {
        let ctx = Context::create();
        let bytes = emit_object(module, &ctx, LlvmTarget::Amdgcn)?;
        Ok(Artifact::bytes(ArtifactKind::Object, bytes))
    }
}

#[cfg(test)]
mod tests;
