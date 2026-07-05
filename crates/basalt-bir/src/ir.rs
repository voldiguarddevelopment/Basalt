// Arena-based typed-SSA data model for BIR functions.
//
// Instructions and basic blocks live in per-function arenas (`Function::insts`,
// `Function::blocks`), addressed by index newtypes (`InstId`, `BlockId`) rather than
// `Rc`/`Box` graphs, preferring arenas and indices in IR-hot paths.
//
// A function's instruction arena is always populated in program order (construction only
// ever appends), so an `InstId` doubles as that instruction's unnamed SSA value number in
// the textual form printed by `print.rs` and read back by `parse.rs` — there is no separate
// value-numbering pass.

use crate::ty::{AddrSpace, Scalar, Ty};

/// Index of an instruction in a function's instruction arena; also that instruction's SSA
/// value number in the textual form (`%<id>`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct InstId(pub u32);

/// Index of a basic block in a function's block arena; also its textual label (`bb<id>`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct BlockId(pub u32);

/// A reference to an SSA value: either a function parameter or an instruction's result.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ValRef {
    Param(u32),
    Val(InstId),
}

/// Integer compare predicate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ICmpPred {
    Eq,
    Ne,
    Slt,
    Sle,
    Sgt,
    Sge,
    Ult,
    Ule,
    Ugt,
    Uge,
}

impl ICmpPred {
    pub const ALL: &'static [ICmpPred] = &[
        ICmpPred::Eq,
        ICmpPred::Ne,
        ICmpPred::Slt,
        ICmpPred::Sle,
        ICmpPred::Sgt,
        ICmpPred::Sge,
        ICmpPred::Ult,
        ICmpPred::Ule,
        ICmpPred::Ugt,
        ICmpPred::Uge,
    ];

    pub fn text(self) -> &'static str {
        match self {
            ICmpPred::Eq => "eq",
            ICmpPred::Ne => "ne",
            ICmpPred::Slt => "slt",
            ICmpPred::Sle => "sle",
            ICmpPred::Sgt => "sgt",
            ICmpPred::Sge => "sge",
            ICmpPred::Ult => "ult",
            ICmpPred::Ule => "ule",
            ICmpPred::Ugt => "ugt",
            ICmpPred::Uge => "uge",
        }
    }

    pub fn parse(s: &str) -> Option<ICmpPred> {
        ICmpPred::ALL.iter().copied().find(|p| p.text() == s)
    }
}

/// Float compare predicate (ordered variants plus `Ord`/`Uno` for NaN checks).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FCmpPred {
    Oeq,
    One,
    Olt,
    Ole,
    Ogt,
    Oge,
    Ord,
    Uno,
}

impl FCmpPred {
    pub const ALL: &'static [FCmpPred] = &[
        FCmpPred::Oeq,
        FCmpPred::One,
        FCmpPred::Olt,
        FCmpPred::Ole,
        FCmpPred::Ogt,
        FCmpPred::Oge,
        FCmpPred::Ord,
        FCmpPred::Uno,
    ];

    pub fn text(self) -> &'static str {
        match self {
            FCmpPred::Oeq => "oeq",
            FCmpPred::One => "one",
            FCmpPred::Olt => "olt",
            FCmpPred::Ole => "ole",
            FCmpPred::Ogt => "ogt",
            FCmpPred::Oge => "oge",
            FCmpPred::Ord => "ord",
            FCmpPred::Uno => "uno",
        }
    }

    pub fn parse(s: &str) -> Option<FCmpPred> {
        FCmpPred::ALL.iter().copied().find(|p| p.text() == s)
    }
}

/// A scalar/vector arithmetic or bitwise binary operation. Operand and result types are
/// always the same `Ty`, carried on the owning `Inst`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinOp {
    Add,
    Sub,
    Mul,
    Div,
    Rem,
    FAdd,
    FSub,
    FMul,
    FDiv,
    FRem,
    And,
    Or,
    Xor,
    Shl,
    Lshr,
    Ashr,
}

impl BinOp {
    pub const ALL: &'static [BinOp] = &[
        BinOp::Add,
        BinOp::Sub,
        BinOp::Mul,
        BinOp::Div,
        BinOp::Rem,
        BinOp::FAdd,
        BinOp::FSub,
        BinOp::FMul,
        BinOp::FDiv,
        BinOp::FRem,
        BinOp::And,
        BinOp::Or,
        BinOp::Xor,
        BinOp::Shl,
        BinOp::Lshr,
        BinOp::Ashr,
    ];

    pub fn text(self) -> &'static str {
        match self {
            BinOp::Add => "add",
            BinOp::Sub => "sub",
            BinOp::Mul => "mul",
            BinOp::Div => "div",
            BinOp::Rem => "rem",
            BinOp::FAdd => "fadd",
            BinOp::FSub => "fsub",
            BinOp::FMul => "fmul",
            BinOp::FDiv => "fdiv",
            BinOp::FRem => "frem",
            BinOp::And => "and",
            BinOp::Or => "or",
            BinOp::Xor => "xor",
            BinOp::Shl => "shl",
            BinOp::Lshr => "lshr",
            BinOp::Ashr => "ashr",
        }
    }

    pub fn parse(s: &str) -> Option<BinOp> {
        BinOp::ALL.iter().copied().find(|b| b.text() == s)
    }
}

/// A value-conversion operation. Source type is carried on `Op::Cast`; destination type is
/// the owning `Inst::ty`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CastOp {
    Trunc,
    Zext,
    Sext,
    FpTrunc,
    FpExt,
    FpToSi,
    FpToUi,
    SiToFp,
    UiToFp,
    Bitcast,
}

impl CastOp {
    pub const ALL: &'static [CastOp] = &[
        CastOp::Trunc,
        CastOp::Zext,
        CastOp::Sext,
        CastOp::FpTrunc,
        CastOp::FpExt,
        CastOp::FpToSi,
        CastOp::FpToUi,
        CastOp::SiToFp,
        CastOp::UiToFp,
        CastOp::Bitcast,
    ];

    pub fn text(self) -> &'static str {
        match self {
            CastOp::Trunc => "trunc",
            CastOp::Zext => "zext",
            CastOp::Sext => "sext",
            CastOp::FpTrunc => "fptrunc",
            CastOp::FpExt => "fpext",
            CastOp::FpToSi => "fptosi",
            CastOp::FpToUi => "fptoui",
            CastOp::SiToFp => "sitofp",
            CastOp::UiToFp => "uitofp",
            CastOp::Bitcast => "bitcast",
        }
    }

    pub fn parse(s: &str) -> Option<CastOp> {
        CastOp::ALL.iter().copied().find(|c| c.text() == s)
    }
}

/// Warp-shuffle variant.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShuffleKind {
    Idx,
    Up,
    Down,
    Xor,
}

impl ShuffleKind {
    pub const ALL: &'static [ShuffleKind] = &[
        ShuffleKind::Idx,
        ShuffleKind::Up,
        ShuffleKind::Down,
        ShuffleKind::Xor,
    ];

    pub fn text(self) -> &'static str {
        match self {
            ShuffleKind::Idx => "shuffle.idx",
            ShuffleKind::Up => "shuffle.up",
            ShuffleKind::Down => "shuffle.down",
            ShuffleKind::Xor => "shuffle.xor",
        }
    }

    pub fn parse(s: &str) -> Option<ShuffleKind> {
        ShuffleKind::ALL.iter().copied().find(|k| k.text() == s)
    }
}

/// Read-modify-write atomic operation (excludes compare-and-swap, which needs a third
/// operand and gets its own `Op::AtomicCas`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AtomicOp {
    Add,
    Sub,
    Exch,
    Min,
    Max,
    And,
    Or,
    Xor,
}

impl AtomicOp {
    pub const ALL: &'static [AtomicOp] = &[
        AtomicOp::Add,
        AtomicOp::Sub,
        AtomicOp::Exch,
        AtomicOp::Min,
        AtomicOp::Max,
        AtomicOp::And,
        AtomicOp::Or,
        AtomicOp::Xor,
    ];

    pub fn text(self) -> &'static str {
        match self {
            AtomicOp::Add => "atomic.add",
            AtomicOp::Sub => "atomic.sub",
            AtomicOp::Exch => "atomic.exch",
            AtomicOp::Min => "atomic.min",
            AtomicOp::Max => "atomic.max",
            AtomicOp::And => "atomic.and",
            AtomicOp::Or => "atomic.or",
            AtomicOp::Xor => "atomic.xor",
        }
    }

    pub fn parse(s: &str) -> Option<AtomicOp> {
        AtomicOp::ALL.iter().copied().find(|a| a.text() == s)
    }
}

/// Row- vs column-major addressing for an `mma` input tile's 2D-to-linear mapping. No
/// separate stride field: an operand's own natural extent (`A`'s `M`/`K`, `B`'s `K`/`N`)
/// is its leading dimension — row-major `A[i,j]` sits at `i*K + j`, col-major at `j*M + i`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MmaLayout {
    RowMajor,
    ColMajor,
}

impl MmaLayout {
    pub const ALL: &'static [MmaLayout] = &[MmaLayout::RowMajor, MmaLayout::ColMajor];

    pub fn text(self) -> &'static str {
        match self {
            MmaLayout::RowMajor => "row",
            MmaLayout::ColMajor => "col",
        }
    }

    pub fn parse(s: &str) -> Option<MmaLayout> {
        MmaLayout::ALL.iter().copied().find(|l| l.text() == s)
    }
}

/// One instruction's operation and operands. The result type lives on the owning `Inst`,
/// not here, except where an op's operand type must differ from its result type: `icmp`/
/// `fcmp` compare a wider type than the `i1` they produce, and casts convert between two
/// distinct types.
#[derive(Debug, Clone, PartialEq)]
pub enum Op {
    ConstInt(i64),
    ConstFloat(f64),
    Bin(BinOp, ValRef, ValRef),
    ICmp(ICmpPred, Ty, ValRef, ValRef),
    FCmp(FCmpPred, Ty, ValRef, ValRef),
    Select(ValRef, ValRef, ValRef),
    Cast(CastOp, Ty, ValRef),
    Load {
        ptr: ValRef,
        space: AddrSpace,
        align: u32,
        volatile: bool,
    },
    Store {
        ptr: ValRef,
        val: ValRef,
        ty: Ty,
        space: AddrSpace,
        align: u32,
        volatile: bool,
    },
    /// Incoming-value list, one `(predecessor block, value)` pair per predecessor.
    Phi(Vec<(BlockId, ValRef)>),
    TidX,
    TidY,
    TidZ,
    BidX,
    BidY,
    BidZ,
    BdimX,
    BdimY,
    BdimZ,
    GdimX,
    GdimY,
    GdimZ,
    Barrier,
    Shuffle(ShuffleKind, ValRef, ValRef),
    Ballot(ValRef),
    VoteAny(ValRef),
    VoteAll(ValRef),
    Atomic(AtomicOp, ValRef, ValRef, AddrSpace),
    AtomicCas(ValRef, ValRef, ValRef, AddrSpace),
    /// Tile-level matrix-multiply-accumulate: `D[m,n] = sum_k(A[m,k] * B[k,n]) + C[m,n]`,
    /// writing through `d` (which may alias `c`). Side-effecting like `Store` rather than
    /// value-producing — BIR has no register-fragment value class, so `a`/`b`/`c`/`d` are
    /// plain pointers into whatever address space they were already given, and the owning
    /// `Inst::ty` is `Ty::Void`. A hand-rolled tensor-core backend decomposes this into its
    /// own fragment load/execute/store sequence; that decomposition is backend-internal,
    /// not part of BIR.
    ///
    /// `layout_a`/`layout_b` only govern `A`/`B`; `C`/`D` (the `m`-by-`n` accumulator tile)
    /// have no layout attribute of their own and are always addressed row-major with `n` as
    /// their leading dimension — the one fixed convention for the op's output.
    Mma {
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
    },
    /// `kernel_name<<<grid, block[, shared[, stream]]>>>(args...)`. `kernel` names the
    /// launched `Function` by its own `name` field, spelled `@name` in textual BIR exactly
    /// like a function's own declaration line (see `print.rs`/`parse.rs`) — a genuine
    /// function *reference*, not a declaration, and the first thing in BIR that names a
    /// function from anywhere other than its own `func @name (...)` line. It is a plain
    /// `String`, not an arena index: nothing about parsing or printing this op needs to
    /// resolve the name against `Module::funcs` (that is a semantic question for whatever
    /// validates the module, not a syntactic one for this crate), so a launch may name a
    /// function printed earlier or later in the same module with no special handling either
    /// way — see `parse.rs`'s round-trip test for a launch referencing a function defined
    /// after it.
    ///
    /// `grid`/`block` are each a flattened `(x, y, z)` triple: BIR has no aggregate value
    /// type to hold a `dim3` directly (`basalt-sema/src/lower.rs`'s own module header has the
    /// full "no aggregate BIR type" story), so `basalt-sema` decomposes both the struct form
    /// and `dim3`'s single-argument implicit-constructor form (`kernel<<<256, 256>>>(...)`,
    /// equivalent to `dim3(256, 1, 1)`) into three scalar operands before this op is ever
    /// built. `shared` (dynamic shared-memory byte count) and `stream` are always concrete
    /// operands, never `Option`: a source launch that omits either lowers to a materialized
    /// default (`0` bytes; a null-stream sentinel) rather than leaving an optional field for
    /// this op's textual grammar to disambiguate.
    ///
    /// Side-effecting like `Store`/`Mma`, not value-producing (`Ty::Void`) — a real kernel
    /// launch has no ordinary-expression value in CUDA C++'s own grammar. **Sema-only today**:
    /// no hand-rolled backend lowers this op yet, including the x86 oracle (a real oracle
    /// lowering needs a genuine call/return mechanism this project has not built — see
    /// `PLAN.md`'s P13-T1c). Every backend refuses it with `E090` until that lands.
    KernelLaunch {
        kernel: String,
        grid: [ValRef; 3],
        block: [ValRef; 3],
        shared: ValRef,
        stream: ValRef,
        args: Vec<ValRef>,
    },
    /// `cudaMalloc(void** devPtr, size_t size)`. The interesting output is not this
    /// instruction's own SSA value — real `cudaMalloc` writes the freshly-allocated pointer
    /// *through* `devPtr` — so this op only produces that pointer as an ordinary `Ty::Ptr`
    /// value; `basalt-sema/src/lower.rs`'s lowering immediately follows it with a real `Store`
    /// of that value through `devPtr`'s own address, the same as any other pointer write.
    /// `size` is `size_t`, modeled as `Ty::Scalar(Scalar::I64)` per this project's existing
    /// `size_t` convention. The x86 oracle lowers this to a real call against libc's own
    /// `malloc` inside a host function (`--cpu`'s "device" memory is just host memory — see
    /// `basalt-x86::oracle`'s module header, P13-T1c-ii); every other backend
    /// (`basalt-ptx`/`basalt-spirv`/`basalt-rv`/`basalt-amdgpu`) still refuses it with `E090`.
    CudaMalloc {
        size: ValRef,
    },
    /// `cudaMemcpy(void* dst, const void* src, size_t count, enum cudaMemcpyKind kind)`.
    /// `kind` is an ordinary integer operand (`cudaMemcpyKind`'s stable values, `HostToHost=0`
    /// .. `Default=4` — see `basalt-sema/src/checker.rs`'s builtin seeding), not a dedicated
    /// enum field: BIR has no enum type, and a real lowering needs the value at runtime to
    /// pick a copy path anyway, not just at compile time. The x86 oracle lowers this to a real
    /// call against libc's own `memcpy` inside a host function, ignoring `kind` (there is no
    /// real host/device distinction to pick a copy path for under a `--cpu` target — see
    /// `basalt-x86::oracle`'s module header, P13-T1c-ii); every other backend still refuses it
    /// with `E090`.
    CudaMemcpy {
        dst: ValRef,
        src: ValRef,
        count: ValRef,
        kind: ValRef,
    },
    /// `cudaFree(void* devPtr)`. The x86 oracle lowers this to a real call against libc's own
    /// `free` inside a host function (P13-T1c-ii); every other backend still refuses it with
    /// `E090`.
    CudaFree {
        ptr: ValRef,
    },
    /// `cudaDeviceSynchronize(void)`. No operands. The x86 oracle lowers this to a real `nop`
    /// inside a host function (P13-T1c-i); every other backend still refuses it with `E090`.
    CudaDeviceSynchronize,
    /// An ordinary same-module function call: `func(args...)`. `func` names the callee's own
    /// `Function::name`, spelled `@name` in textual BIR exactly like `Op::KernelLaunch::kernel`
    /// — a genuine function *reference*, not a declaration, and printed/parsed with the same
    /// "plain `String`, nothing resolved against `Module::funcs` at parse time" convention (see
    /// `KernelLaunch`'s own doc comment above; a call may name a function defined earlier or
    /// later in the same module). The call's own result type lives on the owning `Inst::ty`
    /// exactly like every other value-producing op, `Ty::Void` for a void-returning callee.
    ///
    /// Which call *shapes* an actual backend accepts (whether a caller may itself be called,
    /// direct recursion, cross-module calls, ...) is not a BIR-level restriction — BIR has no
    /// call graph of its own to validate — it is entirely up to whatever backend claims to
    /// lower this op (`basalt-sema`'s own lowering pass, and each backend's `check_module`/
    /// `check_function`, document their own restrictions; see `basalt-x86::oracle`'s module
    /// header for the first backend that actually does this, P13-T-calls-i). Every other
    /// backend refuses this op outright with `E090`.
    Call {
        func: String,
        args: Vec<ValRef>,
    },
}

/// One arena-resident instruction: its result type (`Ty::Void` if it produces no value)
/// plus its operation.
#[derive(Debug, Clone, PartialEq)]
pub struct Inst {
    pub ty: Ty,
    pub op: Op,
}

/// A basic block's terminator. Every block ends in exactly one of these.
#[derive(Debug, Clone, PartialEq)]
pub enum Term {
    Br(BlockId),
    CondBr(ValRef, BlockId, BlockId),
    /// `Switch(scrutinee, default, cases)`; `cases` is `(match value, target block)` pairs.
    Switch(ValRef, BlockId, Vec<(i64, BlockId)>),
    Ret(Option<ValRef>),
}

/// A basic block: an ordered list of instruction ids (into the owning function's arena)
/// followed by exactly one terminator.
#[derive(Debug, Clone, PartialEq)]
pub struct Block {
    pub insts: Vec<InstId>,
    pub term: Term,
}

/// A function: typed params, a return type, and the block/instruction arenas that make up
/// its body. Block 0 is the entry block.
///
/// `is_kernel` is `true` for a function that originated from a `__global__`-qualified
/// declaration (a real, launchable GPU entry point) and `false` for everything else — plain,
/// `__host__`, `__device__`, or `__host__ __device__` functions are never launchable, even
/// though today's frontends only ever lower a `__global__` body into BIR. This is a genuine
/// BIR-level barrier, not a backend concern: a hand-rolled GPU backend that iterates every
/// function in a module must not treat a non-kernel one as a second entry point (see
/// `print.rs`/`parse.rs` for the textual spelling, and each backend's own `check_module` for
/// the refusal this field exists to make possible).
#[derive(Debug, Clone, PartialEq)]
pub struct Function {
    pub name: String,
    pub is_kernel: bool,
    pub params: Vec<Ty>,
    pub ret: Ty,
    pub blocks: Vec<Block>,
    pub insts: Vec<Inst>,
}

/// Kernel launch-bounds hint (module-level per ARCHITECTURE §3's metadata list).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LaunchBounds {
    pub max_threads: u32,
    pub min_blocks: u32,
}

/// A BIR module: its functions plus the module-level metadata called out in
/// ARCHITECTURE §3 (launch bounds, shared-mem bytes, target dtype set).
#[derive(Debug, Clone, PartialEq)]
pub struct Module {
    pub funcs: Vec<Function>,
    pub launch_bounds: Option<LaunchBounds>,
    pub shared_mem_bytes: u32,
    pub target_dtypes: Vec<Scalar>,
}

impl Inst {
    /// Whether this instruction produces an SSA value (i.e. has a `%<id> =` prefix in the
    /// textual form). Only `store` and `barrier` are void.
    pub fn has_result(&self) -> bool {
        self.ty != Ty::Void
    }
}

impl Op {
    /// Mnemonic for the zero-operand GPU index intrinsics (`tid.x`, `bid.y`, ...). Shared
    /// by the printer and parser so the mnemonic table exists in exactly one place.
    pub(crate) fn gpu_index_text(&self) -> Option<&'static str> {
        Some(match self {
            Op::TidX => "tid.x",
            Op::TidY => "tid.y",
            Op::TidZ => "tid.z",
            Op::BidX => "bid.x",
            Op::BidY => "bid.y",
            Op::BidZ => "bid.z",
            Op::BdimX => "bdim.x",
            Op::BdimY => "bdim.y",
            Op::BdimZ => "bdim.z",
            Op::GdimX => "gdim.x",
            Op::GdimY => "gdim.y",
            Op::GdimZ => "gdim.z",
            _ => return None,
        })
    }

    pub(crate) fn gpu_index_from_text(s: &str) -> Option<Op> {
        Some(match s {
            "tid.x" => Op::TidX,
            "tid.y" => Op::TidY,
            "tid.z" => Op::TidZ,
            "bid.x" => Op::BidX,
            "bid.y" => Op::BidY,
            "bid.z" => Op::BidZ,
            "bdim.x" => Op::BdimX,
            "bdim.y" => Op::BdimY,
            "bdim.z" => Op::BdimZ,
            "gdim.x" => Op::GdimX,
            "gdim.y" => Op::GdimY,
            "gdim.z" => Op::GdimZ,
            _ => return None,
        })
    }
}
