// The `Backend` trait plus the three supporting types every backend
// impl shares: `Support` (the refusal contract), `Artifact` (what `emit`
// hands back — bytes or text, never a mix), and `EmitOpts` (the knobs `emit` reads).
//
// Nothing here touches a specific target. Adding a backend means writing a new crate that
// implements `Backend`; this file never grows a match arm per target.

use basalt_bir::Module;
use basalt_diag::{Diag, ECode};

/// Whether a backend can lower every op a module uses.
///
/// A backend must never guess at codegen for an op it does not implement — no
/// silently-wrong codegen. `supports` is the pre-flight check; `emit`
/// re-confirms the same refusal at the point it actually hits the unhandled construct, so
/// the two never drift apart.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Support {
    /// The backend claims to implement every op/type/address-space this module needs.
    Supported,
    /// The backend cannot lower something in this module. Carries the stable E-code that
    /// names *why* (`E090` unrecognized op, `E091` type, `E092` address space, `E093`
    /// feature, `E099` the rank-2-tile-without-matrix-path case) — never a guess.
    Unsupported(ECode),
}

impl Support {
    /// True for `Supported`.
    pub fn is_supported(self) -> bool {
        matches!(self, Support::Supported)
    }
}

/// What kind of payload an `Artifact` carries. Kept separate from the payload itself so
/// callers (the CLI, the diff harness) can branch on format without matching the enum.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArtifactKind {
    /// A relocatable/loadable object: SysV ELF, HSACO, RV ELF, ...
    Object,
    /// PTX assembly text.
    Ptx,
    /// Generated source text (e.g. Tensix Metalium C++).
    Source,
    /// A SPIR-V module. Binary by the SPIR-V spec, but kept as its own kind since it is
    /// neither a host-loadable object nor human-authored text.
    SpirV,
}

/// The output of `Backend::emit`.
///
/// Two payload shapes only: raw bytes (object files, SPIR-V words) or text (PTX, generated
/// C++). A backend picks whichever `Payload` variant matches what it actually produces —
/// never stringify bytes or byte-encode text to force a single shape.
///
/// # Determinism contract
///
/// For a fixed `(module, opts)`, `emit` must produce a byte-identical `Artifact` on every
/// call, on every machine: no timestamps, no hashmap-iteration-order-dependent layout, no
/// host paths or environment baked into the payload. This is exactly what the
/// built-twice double-build hash comparison enforces; a backend that fails it is broken,
/// not "usually right."
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Artifact {
    pub kind: ArtifactKind,
    pub payload: Payload,
}

/// The actual emitted content of an `Artifact`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Payload {
    Bytes(Vec<u8>),
    Text(String),
}

impl Artifact {
    /// Builds a binary artifact (object file, SPIR-V module, ...).
    pub fn bytes(kind: ArtifactKind, data: Vec<u8>) -> Artifact {
        Artifact {
            kind,
            payload: Payload::Bytes(data),
        }
    }

    /// Builds a text artifact (PTX, generated source).
    pub fn text(kind: ArtifactKind, text: String) -> Artifact {
        Artifact {
            kind,
            payload: Payload::Text(text),
        }
    }

    /// The payload as bytes, if this is a `Payload::Bytes` artifact.
    pub fn as_bytes(&self) -> Option<&[u8]> {
        match &self.payload {
            Payload::Bytes(b) => Some(b),
            Payload::Text(_) => None,
        }
    }

    /// The payload as text, if this is a `Payload::Text` artifact.
    pub fn as_text(&self) -> Option<&str> {
        match &self.payload {
            Payload::Text(t) => Some(t),
            Payload::Bytes(_) => None,
        }
    }
}

/// Optimization intent passed to `emit`. Backends that are deliberately dumb (the oracle)
/// are free to ignore this entirely; it exists for the backends that grow a scheduler.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum OptLevel {
    /// Correct-first, no scheduling — the oracle's only mode.
    #[default]
    None,
    Speed,
}

/// Options threaded through to `Backend::emit`.
///
/// Kept as a plain struct with a `Default` impl rather than a builder: every field has an
/// obvious no-op default, and callers (the CLI) only set what a given `--<flag>` implies.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct EmitOpts {
    /// Target sub-variant, e.g. a specific GFX ISA version or RV extension set. `None`
    /// means "the backend's default." Free-form because each backend owns its own variant
    /// namespace (backend isolation).
    pub target_variant: Option<String>,
    /// How aggressively to optimize. See `OptLevel`.
    pub opt_level: OptLevel,
    /// A filename hint for diagnostics/output naming. Never affects the emitted bytes
    /// themselves — the payload must not embed the output path.
    pub out_name_hint: Option<String>,
    /// `--snap`-style debug placeholder: when set, a backend may emit extra
    /// human-readable side-channel debug info (never into the artifact payload).
    pub snap: bool,
}

/// One codegen target. Implemented once per crate under `basalt-{x86,rv,ptx,amdgpu,spirv,
/// tensix,llvm,mlir,clif}`; the CLI registers an instance per `--<name>` flag.
///
/// See ARCHITECTURE §4 for the invariants this trait carries: no silently-wrong codegen,
/// deterministic output, oracle-validatable results.
pub trait Backend {
    /// Stable identifier used by `--<name>` flags and the diff harness.
    fn name(&self) -> &'static str;

    /// Does this backend claim to implement every op the module uses? Checked before
    /// `emit` is called; missing matrix codegen, e.g., reports `Unsupported(E099)` here
    /// rather than producing wrong output.
    fn supports(&self, module: &Module) -> Support;

    /// Lowers a validated BIR module to an artifact. Must be pure and deterministic: no I/O
    /// side effects beyond the returned `Artifact`, and byte-identical output for the same
    /// `(module, opts)` on every call.
    fn emit(&self, module: &Module, opts: &EmitOpts) -> Result<Artifact, Diag>;
}
