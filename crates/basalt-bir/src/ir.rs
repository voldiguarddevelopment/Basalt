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
