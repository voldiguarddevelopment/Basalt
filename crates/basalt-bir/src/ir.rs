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
#[derive(Debug, Clone, PartialEq)]
pub struct Function {
    pub name: String,
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
