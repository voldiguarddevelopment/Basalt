// basalt-backend: the `Backend` trait and the support types every target
// crate shares.
//
// Two pieces:
//   - `backend` — `Backend`, `Support`, `Artifact`/`Payload`/`ArtifactKind`,
//                 `EmitOpts`/`OptLevel`. The abstraction boundary between the shared
//                 frontend/sema/BIR/mid-end pipeline and every pluggable emitter.
//   - `elf`     — `write_elf_object`, a SysV-ELF writer (via the `object` crate)
//                 reused by every backend that emits a relocatable object (`basalt-x86`,
//                 `basalt-rv`, `basalt-amdgpu`'s HSACO container, ...).
//
// This crate has no target-specific knowledge; a backend crate implements `Backend` and,
// if it emits ELF, calls into `elf::write_elf_object`. Nothing here grows a match arm per
// target (backend isolation).

mod backend;
mod elf;

pub use backend::{Artifact, ArtifactKind, Backend, EmitOpts, OptLevel, Payload, Support};
pub use elf::{
    write_elf_object, Architecture, ElfObjectSpec, ElfRelocation, ElfSymbol, Endianness,
};
