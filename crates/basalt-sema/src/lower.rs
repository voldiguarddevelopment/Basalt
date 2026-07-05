// AST-to-BIR lowering: turns a type-checked `TranslationUnit` into a `basalt_bir::Module`.
// Entry point: `lower`. Assumes `checker::check` already ran and returned no fatal problems
// for the parts being lowered; given something it cannot make sense of, this pass degrades to
// a diagnostic plus a best-effort placeholder rather than panicking or guessing (the project's
// "no silently-wrong codegen" rule applies to backends, but the same spirit governs this pass:
// a `Module` containing an `E304` diagnostic is not meant to be handed to a backend).
//
// GPU intrinsics: `threadIdx.x`/`blockIdx.y`/`blockDim.z`/`gridDim.x`-style member access on
// one of the four dim3-like builtins (see `checker::CUDA_DIM3_BUILTINS`) lowers directly to
// BIR's `tid.*`/`bid.*`/`bdim.*`/`gdim.*` index ops (`gpu_index_op_for`), and a call to
// `__syncthreads` lowers to `barrier`. Warp shuffle (`__shfl`/`__shfl_up`/`__shfl_down`/
// `__shfl_xor`), warp vote (`__ballot`/`__any`/`__all`), and the atomic read-modify-write/
// compare-and-swap builtins (`atomicAdd`/`atomicSub`/`atomicExch`/`atomicMin`/`atomicMax`/
// `atomicAnd`/`atomicOr`/`atomicXor`/`atomicCAS`) lower the same way: `lower_call` recognizes
// each by callee name and emits the matching `shuffle.*`/`ballot`/`vote.*`/`atomic.*` op
// directly, rather than going through a real `call` instruction (BIR has none — see the gap
// noted below). The maskless legacy CUDA spellings are used for shuffle/vote rather than the
// `_sync` forms, since BIR's ops carry no warp-mask operand to put one in.
//
// All of the above are seeded into a device/kernel body's scope by
// `checker::seed_cuda_builtins` as ordinary `int`/`int*`-typed functions, not generic over
// every scalar type real CUDA overloads them across (e.g. `atomicAdd` on a `float*`). This
// pass lowers exactly what the checker typed them as; passing e.g. a `float` argument still
// type-checks (sema's `assignable` is permissive between scalar kinds) but is coerced to `int`
// via `coerce_to`'s ordinary numeric conversion, not a bit-level reinterpret — a known
// simplification inherited from the checker's monomorphic builtin signatures, not a new gap
// introduced here.
//
// Locals are stack slots, not SSA values. Every `VarDecl` (including parameters, which BIR
// itself passes as plain SSA values via `ValRef::Param`) gets a synthetic memory location and
// is accessed via `load`/`store`, mem2reg-style promotion to real SSA form is explicitly a
// later pass's job (ARCHITECTURE.md's mid-end). This keeps this pass a straightforward
// syntax-directed translation, at the cost of some redundant loads/stores and a trailing dead
// block after every `return`/`break`/`continue`/`goto` (this pass always opens a fresh block
// after a terminator, whether or not anything ever branches to it) that a later cleanup pass
// is expected to fold away.
//
// # Synthesizing stack-slot addresses: a BIR gap
//
// BIR has no `alloca`-style instruction — nothing that says "reserve a new, distinct storage
// location and hand back its address". Inventing one is out of scope here (ARCHITECTURE.md
// requires printer + parser + oracle lowering to land together for a new op). The workaround:
// each local variable is assigned a small integer slot index at lowering time, and its address
// is materialized as `const.i ptr.<space> (slot * SLOT_STRIDE)` — an ordinary integer constant
// whose declared type happens to be a pointer. Nothing in BIR's printer/parser stops this (a
// pointer's own type carries only an address space, never a pointee type, so its "value" was
// already opaque), and it gives every local a stable, distinct address without adding new BIR
// surface. `SLOT_STRIDE` is a generous fixed spacing (documented below) standing in for a real
// stack-frame layout a later pass would compute tightly.
//
// # Address spaces
//
// - A local variable's own slot: `AddrSpace::Local`, unless CUDA-qualified `__shared__`
//   (`AddrSpace::Shared`) or `__constant__` (`AddrSpace::Constant`).
// - A function parameter's own slot: `AddrSpace::Param`.
// - The pointee of a pointer *value* (a dereference, `->`, or indexing through a pointer
//   variable rather than an array): `AddrSpace::Global`, the common case for a CUDA kernel
//   argument pointing into device memory. Sema's own `Ty::Pointer` carries no address-space
//   annotation at all (a checker-era simplification), so this pass cannot do better without
//   deeper source-level qualifier tracking than the AST currently carries for arbitrary
//   pointer-typed *expressions* (only declarations carry `CudaQualifiers`).
// - Indexing/member access through an array or struct/union *lvalue* (not a pointer):
//   inherits the base's own space (still the same storage region).
//
// # Arrays and structs: no aggregate BIR type
//
// BIR's `Ty` is `Scalar | Ptr | Vec | Void` — no fixed-size array, no struct/record type, no
// `getelementptr`-style op. An array-typed local decays to its slot's base address (matching C);
// indexing and member access are lowered as manual pointer arithmetic (`add` typed as a
// pointer, offset computed from a packed, unpadded field layout this pass derives from the
// checker's own `StructInfo` — no alignment/padding rules are applied, a documented
// simplification). A *whole* aggregate used as a value (assigned, returned, passed, compared)
// has nothing to lower to (no aggregate value type, no memcpy-equivalent op) and is reported as
// `E304` instead of silently taking its address and mislabeling that as the value.
//
// Sema's own `Ty::Array` does not carry an element count (the checker never needed it). This
// pass does not need it either for indexing (only the element type matters for computing a
// byte offset), but `sizeof` on an array falls back to a single element's size with an `E304`
// diagnostic when the count would have mattered.
//
// # Other discovered BIR gaps
//
// - A call to a plain, resolvable same-module function lowers to a real `Op::Call`
//   (P13-T-calls-i) instead of the `E304` this pass used to raise unconditionally. A whole
//   struct/union/array argument or return value still has nothing to lower to (no aggregate
//   BIR value, same as everywhere else in this pass) and still reports `E304`. Everything else
//   about which *call graph shapes* an actual backend accepts — recursion, `__device__`-to-
//   `__device__` calls, cross-translation-unit calls — is deliberately not this pass's problem:
//   BIR's own `Op::Call` carries no such restriction, so this pass emits it generically for any
//   resolved same-module callee and leaves shape validation to whichever backend claims to
//   lower it (see `basalt-x86::oracle`'s module header for the first one that does, and its own
//   documented scope for this first slice).
// - `BinOp::Div`/`BinOp::Rem` do not distinguish signed from unsigned (unlike `icmp`, which has
//   separate signed/unsigned predicates, and `ashr`/`lshr`). A backend lowering a bare `div` has
//   no way to recover the operand's signedness from BIR alone; this pass always emits the one
//   opcode BIR provides and flags the gap here rather than inventing `udiv`/`sdiv` unilaterally.
// - No module-level data segment: a top-level (`Item::Var`) global has no storage to lower to,
//   so it is registered for name resolution only and every reference to it reports `E304`.
// - `switch` lowers to BIR's native `Term::Switch` (a real match, not a `condbr` chain) but only
//   recognizes `case`/`default` labels that are direct entries of the switch's own body block —
//   the overwhelmingly common style. A label buried inside a nested statement (Duff's-device
//   style) is not detected; this is a narrow, documented gap, not a claim of full support.
//
// New diagnostic code: `ECode::LoweringUnsupported` (`E304`), covering every gap above.

use std::collections::{HashMap, HashSet};

use basalt_bir::{
    AddrSpace as BSpace, AtomicOp as BAtomicOp, BinOp as BBin, Block as BBlock, BlockId,
    CastOp as BCast, FCmpPred, Function as BFunction, ICmpPred, Inst as BInst, InstId, Module,
    Op as BOp, Scalar as BScalar, ShuffleKind as BShuffleKind, Term as BTerm, Ty as BTy, ValRef,
};
use basalt_diag::{Diag, ECode};
use basalt_frontend_c::ast::{
    AssignOp, BinOp as ABin, EnumDecl, Expr, FunctionDecl, IncDecOp, Item, ScalarKind, Stmt,
    TagKind, TranslationUnit, Type, UnaryOp, VarDecl,
};
use basalt_frontend_c::{FloatLit, FloatSuffix, IntBase, IntLit, Span as FSpan};

use crate::checker::{
    collect_labels_many, compound_binop, conv_span, float_lit_ty, int_lit_ty, top_level_const,
    CUDA_ATOMIC_CAS_BUILTIN, CUDA_DEVICE_SYNCHRONIZE_BUILTIN, CUDA_DIM3_BUILTINS, CUDA_DIM3_STRUCT,
    CUDA_FREE_BUILTIN, CUDA_MALLOC_BUILTIN, CUDA_MEMCPY_BUILTIN, CUDA_MEMCPY_KIND_CONSTANTS,
};
use crate::scope::{FuncSig, ScopeStack, StructInfo, ValueSym};
use crate::ty::{assignable, is_signed_kind, promote, Ty};

/// Spacing between two locals' synthesized slot addresses (see the module header). Generous
/// enough that no array/struct this pass lays out (packed, no padding) can plausibly overrun
/// into a neighboring slot.
const SLOT_STRIDE: i64 = 1 << 16;

/// Lowers a type-checked translation unit to BIR. Returns the module built so far alongside
/// every diagnostic collected along the way; a non-empty diagnostic list means some part of the
/// input could not be soundly lowered (see the module header for exactly what) and the module
/// should not be treated as ready for a backend.
pub fn lower(tu: &TranslationUnit) -> (Module, Vec<Diag>) {
    let mut lw = Lowerer {
        scopes: ScopeStack::new(),
        diags: Vec::new(),
        funcs: Vec::new(),
        unlowered_globals: HashSet::new(),
        enum_values: HashMap::new(),
        insts: Vec::new(),
        blocks: Vec::new(),
        insts_by_block: HashMap::new(),
        cur: BlockId(0),
        locals: Vec::new(),
        next_slot: 0,
        ctrl_stack: Vec::new(),
        label_blocks: HashMap::new(),
        fn_ret: Ty::Scalar(ScalarKind::Void),
    };
    lw.scopes.push();
    lw.seed_cuda_runtime_api();
    lw.lower_items(&tu.items);
    lw.scopes.pop();
    let module = Module {
        funcs: lw.funcs,
        launch_bounds: None,
        shared_mem_bytes: 0,
        target_dtypes: Vec::new(),
    };
    (module, lw.diags)
}

#[derive(Clone)]
struct LocalSlot {
    slot: u32,
    ty: Ty,
    space: BSpace,
}

/// An addressable location: where it lives (`addr`, `space`) and what sema type is stored
/// there. `ty == Ty::Unknown` marks a location that could not be resolved — already
/// diagnosed, callers should propagate silently rather than reporting a second time.
struct LValue {
    addr: ValRef,
    ty: Ty,
    space: BSpace,
}

/// What a `break`/`continue` targets. `continue` always looks for the innermost `Loop` entry,
/// skipping over any `Switch` frames above it (a `switch` only ever redefines `break`).
enum CtrlCtx {
    Loop {
        break_bb: BlockId,
        continue_bb: BlockId,
    },
    Switch {
        break_bb: BlockId,
    },
}

enum CaseLabel {
    Value(i64),
    Default,
}

struct Lowerer {
    /// struct/union/enum/typedef tags plus the global function/variable namespace, built up
    /// front exactly like `checker::Checker` does, since struct layouts and function return
    /// types must be known before a body that references them is lowered.
    scopes: ScopeStack,
    diags: Vec<Diag>,
    funcs: Vec<BFunction>,
    /// Names of top-level `Item::Var` globals: registered for resolution so referencing one
    /// reports the specific "no data segment" gap instead of a generic undefined-symbol error.
    unlowered_globals: HashSet<String>,
    enum_values: HashMap<String, i64>,

    // Per-function transient state; reset at the top of `lower_function_body`.
    insts: Vec<BInst>,
    blocks: Vec<Option<BBlock>>,
    /// Instructions appended to a block that is open (allocated, not yet terminated) but not
    /// necessarily the *most recently* opened one — `if`/ternary lowering interleaves building
    /// two branch blocks before either is terminated, so this is keyed by block rather than
    /// being a single current-block buffer.
    insts_by_block: HashMap<u32, Vec<InstId>>,
    cur: BlockId,
    locals: Vec<HashMap<String, LocalSlot>>,
    next_slot: u32,
    ctrl_stack: Vec<CtrlCtx>,
    label_blocks: HashMap<String, BlockId>,
    fn_ret: Ty,
}

// ---- free helpers: type mapping, sizes, casts --------------------------------------------

fn to_bir_scalar(k: ScalarKind) -> Option<BScalar> {
    use ScalarKind::*;
    Some(match k {
        Void => return None,
        Bool => BScalar::I1,
        Char | SChar | UChar => BScalar::I8,
        Short | UShort => BScalar::I16,
        Int | UInt | WcharT => BScalar::I32,
        Long | ULong | LongLong | ULongLong => BScalar::I64,
        Float => BScalar::F32,
        // BIR has no 80/128-bit float type; `long double` truncates to `f64` (documented
        // lossy fallback, matching the "map to the nearest BIR scalar width" instruction).
        Double | LongDouble => BScalar::F64,
    })
}

/// Maps a sema type to the BIR type of its *value* (not its storage address — see
/// `LValue`/`slot_lvalue` for that). Pointer-like sema types always map to `Ptr(Global)`; the
/// module header explains why this pass cannot generally do better for an arbitrary pointer
/// *expression*. Array/struct/union map to the same, matching their pointer-decayed value.
fn to_bir_ty(ty: &Ty) -> BTy {
    match ty {
        Ty::Scalar(k) => to_bir_scalar(*k).map(BTy::Scalar).unwrap_or(BTy::Void),
        Ty::Pointer(..) | Ty::Array(_) | Ty::Struct(_) | Ty::Union(_) => BTy::Ptr(BSpace::Global),
        Ty::Enum(_) => BTy::Scalar(BScalar::I32),
        Ty::Function { .. } | Ty::Unknown => BTy::Scalar(BScalar::I32),
    }
}

fn is_aggregate(ty: &Ty) -> bool {
    matches!(ty, Ty::Array(_) | Ty::Struct(_) | Ty::Union(_))
}

fn is_signed(ty: &Ty) -> bool {
    match ty {
        Ty::Scalar(k) => is_signed_kind(*k),
        Ty::Enum(_) => true,
        _ => false,
    }
}

fn scalar_bits(s: BScalar) -> u32 {
    match s {
        BScalar::I1 => 1,
        BScalar::I8 => 8,
        BScalar::I16 | BScalar::F16 => 16,
        BScalar::I32 | BScalar::F32 => 32,
        BScalar::I64 | BScalar::F64 => 64,
    }
}

fn scalar_byte_size(s: BScalar) -> u32 {
    match s {
        BScalar::I1 | BScalar::I8 => 1,
        BScalar::I16 | BScalar::F16 => 2,
        BScalar::I32 | BScalar::F32 => 4,
        BScalar::I64 | BScalar::F64 => 8,
    }
}

fn natural_align(t: BTy) -> u32 {
    match t {
        BTy::Scalar(s) => scalar_byte_size(s),
        BTy::Ptr(_) => 8,
        BTy::Vec(s, n) => scalar_byte_size(s) * u32::from(n),
        BTy::Void => 1,
    }
}

fn is_float_scalar(s: BScalar) -> bool {
    matches!(s, BScalar::F16 | BScalar::F32 | BScalar::F64)
}

// ---- GPU intrinsic name tables ------------------------------------------------------------
//
// These map a builtin's *source* spelling straight to its BIR op, independent of
// `checker::seed_cuda_builtins`'s bookkeeping (which only exists to give the builtin a type
// for ordinary call-arity/argument checking). Lowering recognizes these names unconditionally,
// the same way it already does for `checker::CUDA_DIM3_BUILTINS` — this pass assumes the
// checker already rejected any use outside a device/kernel body, so it never re-checks that
// context here.

/// Maps a `threadIdx`/`blockIdx`/`blockDim`/`gridDim` builtin plus a `.x`/`.y`/`.z` field to
/// its BIR index intrinsic (e.g. `threadIdx.x` -> `Op::TidX`). `None` for anything else, so the
/// caller falls back to ordinary struct-member lowering.
fn gpu_index_op_for(builtin: &str, field: &str) -> Option<BOp> {
    Some(match (builtin, field) {
        ("threadIdx", "x") => BOp::TidX,
        ("threadIdx", "y") => BOp::TidY,
        ("threadIdx", "z") => BOp::TidZ,
        ("blockIdx", "x") => BOp::BidX,
        ("blockIdx", "y") => BOp::BidY,
        ("blockIdx", "z") => BOp::BidZ,
        ("blockDim", "x") => BOp::BdimX,
        ("blockDim", "y") => BOp::BdimY,
        ("blockDim", "z") => BOp::BdimZ,
        ("gridDim", "x") => BOp::GdimX,
        ("gridDim", "y") => BOp::GdimY,
        ("gridDim", "z") => BOp::GdimZ,
        _ => return None,
    })
}

/// Maps a shuffle builtin's name to BIR's `ShuffleKind`, or `None` if `name` is not one of
/// `checker::CUDA_SHUFFLE_BUILTINS`.
fn shuffle_kind_for(name: &str) -> Option<BShuffleKind> {
    Some(match name {
        "__shfl" => BShuffleKind::Idx,
        "__shfl_up" => BShuffleKind::Up,
        "__shfl_down" => BShuffleKind::Down,
        "__shfl_xor" => BShuffleKind::Xor,
        _ => return None,
    })
}

/// Maps a vote builtin's name to the `Op` tuple-variant constructor it lowers to (each takes
/// exactly the one predicate operand), or `None` if `name` is not one of
/// `checker::CUDA_VOTE_BUILTINS`.
fn vote_ctor_for(name: &str) -> Option<fn(ValRef) -> BOp> {
    Some(match name {
        "__ballot" => BOp::Ballot,
        "__any" => BOp::VoteAny,
        "__all" => BOp::VoteAll,
        _ => return None,
    })
}

/// Maps an atomic read-modify-write builtin's name to BIR's `AtomicOp`, or `None` if `name` is
/// not one of `checker::CUDA_ATOMIC_RMW_BUILTINS`. `atomicCAS` is handled separately (three
/// operands, `Op::AtomicCas` rather than `Op::Atomic`).
fn atomic_op_for(name: &str) -> Option<BAtomicOp> {
    Some(match name {
        "atomicAdd" => BAtomicOp::Add,
        "atomicSub" => BAtomicOp::Sub,
        "atomicExch" => BAtomicOp::Exch,
        "atomicMin" => BAtomicOp::Min,
        "atomicMax" => BAtomicOp::Max,
        "atomicAnd" => BAtomicOp::And,
        "atomicOr" => BAtomicOp::Or,
        "atomicXor" => BAtomicOp::Xor,
        _ => return None,
    })
}

// ---- linearizing the instruction arena into block-print order -----------------------------
//
// BIR requires a function's instruction arena to be appended strictly in the order it prints
// (block 0's instructions, then block 1's, ...), since a `%<id>` doubles as that arena index.
// This pass builds several blocks concurrently (`if`, ternary, and `&&`/`||` all leave more
// than one allocated block open across a span of lowering, e.g. filling `then_bb` and then
// `else_bb` before either is terminated), so `self.insts` ends up in *creation* order, which is
// not the same as block-print order once a block's own content is filled in later than a
// higher-numbered block's. This is fixed up once per function, right before it is handed back:
// walk the already-terminated blocks in block-id order, assign each instruction encountered a
// fresh id in that order, and rewrite every `ValRef::Val` (in every op's operands, `phi`'s
// incoming values, and every terminator) through the resulting remap table. Every cross-block
// value reference this pass ever produces (only `phi`, from `if`/ternary/`&&`/`||`'s merge
// blocks) already goes from a lower block id to a higher one, so this is a straightforward
// reindexing, not a real reschedule.

fn remap_valref(v: ValRef, remap: &[u32]) -> ValRef {
    match v {
        ValRef::Param(p) => ValRef::Param(p),
        ValRef::Val(id) => ValRef::Val(InstId(remap[id.0 as usize])),
    }
}

fn remap_op(op: BOp, remap: &[u32]) -> BOp {
    let rv = |v: ValRef| remap_valref(v, remap);
    match op {
        BOp::ConstInt(v) => BOp::ConstInt(v),
        BOp::ConstFloat(v) => BOp::ConstFloat(v),
        BOp::Bin(b, a, c) => BOp::Bin(b, rv(a), rv(c)),
        BOp::ICmp(p, ty, a, c) => BOp::ICmp(p, ty, rv(a), rv(c)),
        BOp::FCmp(p, ty, a, c) => BOp::FCmp(p, ty, rv(a), rv(c)),
        BOp::Select(c, a, b) => BOp::Select(rv(c), rv(a), rv(b)),
        BOp::Cast(c, ty, v) => BOp::Cast(c, ty, rv(v)),
        BOp::Load {
            ptr,
            space,
            align,
            volatile,
        } => BOp::Load {
            ptr: rv(ptr),
            space,
            align,
            volatile,
        },
        BOp::Store {
            ptr,
            val,
            ty,
            space,
            align,
            volatile,
        } => BOp::Store {
            ptr: rv(ptr),
            val: rv(val),
            ty,
            space,
            align,
            volatile,
        },
        BOp::Phi(incoming) => BOp::Phi(incoming.into_iter().map(|(bb, v)| (bb, rv(v))).collect()),
        BOp::Shuffle(k, a, b) => BOp::Shuffle(k, rv(a), rv(b)),
        BOp::Ballot(a) => BOp::Ballot(rv(a)),
        BOp::VoteAny(a) => BOp::VoteAny(rv(a)),
        BOp::VoteAll(a) => BOp::VoteAll(rv(a)),
        BOp::Atomic(a, ptr, v, space) => BOp::Atomic(a, rv(ptr), rv(v), space),
        BOp::AtomicCas(ptr, cmp, new, space) => BOp::AtomicCas(rv(ptr), rv(cmp), rv(new), space),
        BOp::Call { func, args } => BOp::Call {
            func,
            args: args.into_iter().map(rv).collect(),
        },
        other => other,
    }
}

fn remap_term(t: BTerm, remap: &[u32]) -> BTerm {
    match t {
        BTerm::Br(b) => BTerm::Br(b),
        BTerm::CondBr(c, t1, t2) => BTerm::CondBr(remap_valref(c, remap), t1, t2),
        BTerm::Switch(v, d, cases) => BTerm::Switch(remap_valref(v, remap), d, cases),
        BTerm::Ret(v) => BTerm::Ret(v.map(|x| remap_valref(x, remap))),
    }
}

fn linearize_by_block_order(
    blocks: Vec<BBlock>,
    old_insts: Vec<BInst>,
) -> (Vec<BBlock>, Vec<BInst>) {
    let mut remap = vec![0u32; old_insts.len()];
    let mut order: Vec<InstId> = Vec::with_capacity(old_insts.len());
    for b in &blocks {
        for &id in &b.insts {
            remap[id.0 as usize] = order.len() as u32;
            order.push(id);
        }
    }
    let new_insts: Vec<BInst> = order
        .iter()
        .map(|id| {
            let inst = &old_insts[id.0 as usize];
            BInst {
                ty: inst.ty,
                op: remap_op(inst.op.clone(), &remap),
            }
        })
        .collect();
    let new_blocks: Vec<BBlock> = blocks
        .into_iter()
        .map(|b| {
            let new_ids: Vec<InstId> = b
                .insts
                .iter()
                .map(|id| InstId(remap[id.0 as usize]))
                .collect();
            BBlock {
                insts: new_ids,
                term: remap_term(b.term, &remap),
            }
        })
        .collect();
    (new_blocks, new_insts)
}

fn switch_body_stmts(body: &Stmt) -> Vec<&Stmt> {
    match body {
        Stmt::Block { stmts, .. } => stmts.iter().collect(),
        other => vec![other],
    }
}

fn is_lvalue_shaped(e: &Expr) -> bool {
    matches!(
        e,
        Expr::Ident { .. } | Expr::Member { .. } | Expr::Index { .. }
    ) || matches!(
        e,
        Expr::Unary {
            op: UnaryOp::Deref,
            ..
        }
    )
}

/// Recovers an integer literal's numeric value from its lexed text (digits/prefix/suffix all
/// still present there; `IntLit` only classifies the shape). Best-effort: a literal wide
/// enough to overflow `i64` as signed is reinterpreted via `u64`, matching two's-complement
/// truncation rather than failing outright.
fn int_lit_value(lit: &IntLit) -> i64 {
    let mut s = lit.text.as_str();
    while let Some(c) = s.chars().next_back() {
        if c.is_ascii_alphabetic() {
            s = &s[..s.len() - 1];
        } else {
            break;
        }
    }
    let (radix, digits) = match lit.base {
        IntBase::Dec => (10, s),
        IntBase::Oct => (8, s),
        IntBase::Hex => (16, s.trim_start_matches("0x").trim_start_matches("0X")),
        IntBase::Bin => (2, s.trim_start_matches("0b").trim_start_matches("0B")),
    };
    i64::from_str_radix(digits, radix)
        .or_else(|_| u64::from_str_radix(digits, radix).map(|v| v as i64))
        .unwrap_or(0)
}

fn float_lit_value(lit: &FloatLit) -> f64 {
    let mut s = lit.text.as_str();
    if matches!(lit.suffix, FloatSuffix::F | FloatSuffix::L) {
        s = &s[..s.len() - 1];
    }
    s.parse::<f64>().unwrap_or(0.0)
}

// ---- Lowerer: item-level registration -----------------------------------------------------

impl Lowerer {
    /// Mirrors `checker::Checker::seed_cuda_runtime_api`'s scope-visible side (this pass has no
    /// need to register the four CUDA Runtime API calls themselves — `lower_call` recognizes
    /// them directly by name, exactly like `__syncthreads`): declares `dim3`
    /// (`CUDA_DIM3_STRUCT`) globally so `field_offset` can resolve a real dim3-typed launch-
    /// config value's `x`/`y`/`z` fields, and pre-populates `enum_values` with
    /// `cudaMemcpyKind`'s five named constants the same way `lower_enum_decl` would for a real
    /// `enum` — so a `cudaMemcpy` call naming one of them by name lowers to the same real
    /// integer constant `checker::Checker::seed_cuda_runtime_api` typed it as.
    fn seed_cuda_runtime_api(&mut self) {
        self.scopes.declare_struct(
            CUDA_DIM3_STRUCT,
            StructInfo {
                fields: vec![
                    ("x".to_string(), Ty::Scalar(ScalarKind::UInt), false),
                    ("y".to_string(), Ty::Scalar(ScalarKind::UInt), false),
                    ("z".to_string(), Ty::Scalar(ScalarKind::UInt), false),
                ],
            },
        );
        for &(name, value) in &CUDA_MEMCPY_KIND_CONSTANTS {
            self.enum_values.insert(name.to_string(), value);
        }
    }

    fn lower_items(&mut self, items: &[Item]) {
        for item in items {
            self.lower_item(item);
        }
    }

    fn lower_item(&mut self, item: &Item) {
        match item {
            Item::Struct(d) => {
                let mut fields = Vec::with_capacity(d.fields.len());
                for f in &d.fields {
                    let ty = self.resolve_type(&f.ty);
                    fields.push((f.name.clone(), ty, top_level_const(&f.ty)));
                }
                if let Some(name) = &d.name {
                    self.scopes.declare_struct(name, StructInfo { fields });
                }
            }
            Item::Union(d) => {
                let mut fields = Vec::with_capacity(d.fields.len());
                for f in &d.fields {
                    let ty = self.resolve_type(&f.ty);
                    fields.push((f.name.clone(), ty, top_level_const(&f.ty)));
                }
                if let Some(name) = &d.name {
                    self.scopes.declare_union(name, StructInfo { fields });
                }
            }
            Item::Enum(d) => self.lower_enum_decl(d),
            Item::Typedef(d) => {
                let ty = self.resolve_type(&d.ty);
                self.scopes.declare_typedef(&d.alias, ty);
            }
            Item::Namespace(ns) => {
                self.scopes.push();
                self.lower_items(&ns.items);
                self.scopes.pop();
            }
            Item::Function(f) => self.lower_function_item(f),
            Item::Var(v) => self.lower_global_var(v),
            // Out of scope, matching `checker::Checker::check_item`: never descended into.
            Item::Template(_) => {}
        }
    }

    fn lower_enum_decl(&mut self, d: &EnumDecl) {
        if let Some(name) = &d.name {
            self.scopes.declare_enum(name);
        }
        let enum_ty = match &d.name {
            Some(n) => Ty::Enum(n.clone()),
            None => Ty::Scalar(ScalarKind::Int),
        };
        let mut next = 0i64;
        for v in &d.variants {
            let value = match &v.init {
                Some(init) => self.const_eval_i64(init).unwrap_or_else(|| {
                    self.diag_unsupported(
                        v.span,
                        "non-constant enumerator initializer",
                        "falling back to auto-increment from the previous value",
                    );
                    next
                }),
                None => next,
            };
            self.enum_values.insert(v.name.clone(), value);
            next = value + 1;
            self.scopes
                .declare_value(&v.name, ValueSym::EnumConst(enum_ty.clone()));
        }
    }

    fn lower_global_var(&mut self, v: &VarDecl) {
        let ty = self.resolve_type(&v.ty);
        self.scopes.declare_value(&v.name, ValueSym::Var(ty, false));
        self.unlowered_globals.insert(v.name.clone());
        self.diag_unsupported(
            v.span,
            "global variable storage",
            "BIR has no module-level data segment yet",
        );
    }

    fn lower_function_item(&mut self, f: &FunctionDecl) {
        let ret = self.resolve_type(&f.ret);
        let param_tys: Vec<Ty> = f.params.iter().map(|p| self.resolve_type(&p.ty)).collect();
        let sig = FuncSig {
            ret: ret.clone(),
            params: param_tys.clone(),
            variadic: f.variadic,
            is_kernel: f.cuda_quals.is_global,
        };
        self.scopes.declare_value(&f.name, ValueSym::Func(sig));
        if let Some(body) = &f.body {
            self.lower_function_body(f, ret, &param_tys, body, f.cuda_quals.is_global);
        }
    }

    /// Mirrors `checker::Checker::resolve_type`'s shape but does not also re-visit an array's
    /// size expression: that visit exists in the checker purely to catch diagnostics inside the
    /// size expression, which this pass assumes already happened. Kept as its own small
    /// function rather than sharing code with the checker's version, which would need a
    /// callback threaded through for that one difference — more machinery than the ~25 lines
    /// it would save.
    fn resolve_type(&mut self, ty: &Type) -> Ty {
        match ty {
            Type::Scalar { kind, .. } => Ty::Scalar(*kind),
            Type::Tag {
                kind, name, span, ..
            } => {
                if name.is_empty() {
                    return Ty::Unknown;
                }
                let found = match kind {
                    TagKind::Struct => self.scopes.lookup_struct(name).is_some(),
                    TagKind::Union => self.scopes.lookup_union(name).is_some(),
                    TagKind::Enum => self.scopes.lookup_enum(name).is_some(),
                };
                if !found {
                    self.diags.push(
                        Diag::new(ECode::UndefinedSymbol)
                            .with_span(conv_span(*span))
                            .with_arg(name.clone()),
                    );
                    return Ty::Unknown;
                }
                match kind {
                    TagKind::Struct => Ty::Struct(name.clone()),
                    TagKind::Union => Ty::Union(name.clone()),
                    TagKind::Enum => Ty::Enum(name.clone()),
                }
            }
            Type::Named { name, span, .. } => {
                if let Some(t) = self.scopes.lookup_typedef(name) {
                    return t.clone();
                }
                if self.scopes.lookup_struct(name).is_some() {
                    return Ty::Struct(name.clone());
                }
                if self.scopes.lookup_union(name).is_some() {
                    return Ty::Union(name.clone());
                }
                if self.scopes.lookup_enum(name).is_some() {
                    return Ty::Enum(name.clone());
                }
                self.diags.push(
                    Diag::new(ECode::UndefinedSymbol)
                        .with_span(conv_span(*span))
                        .with_arg(name.clone()),
                );
                Ty::Unknown
            }
            Type::Pointer { pointee, .. } => Ty::Pointer(
                Box::new(self.resolve_type(pointee)),
                top_level_const(pointee),
            ),
            Type::Array { elem, .. } => Ty::Array(Box::new(self.resolve_type(elem))),
            Type::Instantiated { .. } => Ty::Unknown,
        }
    }

    // ---- struct/union layout (packed, no padding — see module header) -------------------

    fn size_of_ty(&self, ty: &Ty) -> u64 {
        match ty {
            Ty::Scalar(k) => to_bir_scalar(*k)
                .map(|s| u64::from(scalar_byte_size(s)))
                .unwrap_or(0),
            Ty::Pointer(..) => 8,
            Ty::Enum(_) => 4,
            // Documented gap: `Ty::Array` carries no element count, so this cannot compute a
            // real total size; treated as a single element (flagged at each `sizeof` call site
            // that takes this path, not silently here).
            Ty::Array(elem) => self.size_of_ty(elem),
            Ty::Struct(n) => self
                .scopes
                .lookup_struct(n)
                .map(|info| info.fields.iter().map(|(_, t, _)| self.size_of_ty(t)).sum())
                .unwrap_or(0),
            Ty::Union(n) => self
                .scopes
                .lookup_union(n)
                .map(|info| {
                    info.fields
                        .iter()
                        .map(|(_, t, _)| self.size_of_ty(t))
                        .max()
                        .unwrap_or(0)
                })
                .unwrap_or(0),
            Ty::Function { .. } | Ty::Unknown => 0,
        }
    }

    fn field_offset(&self, ty: &Ty, field: &str) -> Option<(u64, Ty)> {
        match ty {
            Ty::Struct(n) => {
                let info = self.scopes.lookup_struct(n)?;
                let mut off = 0u64;
                for (fname, fty, _) in &info.fields {
                    if fname == field {
                        return Some((off, fty.clone()));
                    }
                    off += self.size_of_ty(fty);
                }
                None
            }
            // Union fields overlay the same storage; every field starts at offset 0.
            Ty::Union(n) => {
                let info = self.scopes.lookup_union(n)?;
                info.fields
                    .iter()
                    .find(|(fname, _, _)| fname == field)
                    .map(|(_, fty, _)| (0u64, fty.clone()))
            }
            _ => None,
        }
    }

    // ---- constant folding (enumerators, case labels) -------------------------------------

    fn const_eval_i64(&self, e: &Expr) -> Option<i64> {
        match e {
            Expr::IntLit { value, .. } => Some(int_lit_value(value)),
            Expr::CharLit { value, .. } => Some(i64::from(value.value)),
            Expr::Unary { op, expr, .. } => {
                let v = self.const_eval_i64(expr)?;
                Some(match op {
                    UnaryOp::Neg => v.wrapping_neg(),
                    UnaryOp::Plus => v,
                    UnaryOp::BitNot => !v,
                    _ => return None,
                })
            }
            Expr::Binary { op, lhs, rhs, .. } => {
                let l = self.const_eval_i64(lhs)?;
                let r = self.const_eval_i64(rhs)?;
                Some(match op {
                    ABin::Add => l.wrapping_add(r),
                    ABin::Sub => l.wrapping_sub(r),
                    ABin::Mul => l.wrapping_mul(r),
                    ABin::Div if r != 0 => l.wrapping_div(r),
                    ABin::Rem if r != 0 => l.wrapping_rem(r),
                    ABin::BitAnd => l & r,
                    ABin::BitOr => l | r,
                    ABin::BitXor => l ^ r,
                    ABin::Shl => l.wrapping_shl(r as u32),
                    ABin::Shr => l.wrapping_shr(r as u32),
                    _ => return None,
                })
            }
            Expr::Ident { name, .. } => self.enum_values.get(name).copied(),
            _ => None,
        }
    }

    fn diag_unsupported(
        &mut self,
        span: FSpan,
        what: impl Into<String>,
        detail: impl Into<String>,
    ) {
        self.diags.push(
            Diag::new(ECode::LoweringUnsupported)
                .with_span(conv_span(span))
                .with_arg(what.into())
                .with_arg(detail.into()),
        );
    }
}

// ---- Lowerer: function bodies, blocks, and the stack-slot bookkeeping ----------------------

impl Lowerer {
    fn lower_function_body(
        &mut self,
        f: &FunctionDecl,
        ret: Ty,
        param_tys: &[Ty],
        body: &[Stmt],
        is_kernel: bool,
    ) {
        self.insts = Vec::new();
        self.blocks = Vec::new();
        self.insts_by_block = HashMap::new();
        self.locals = vec![HashMap::new()];
        self.next_slot = 0;
        self.ctrl_stack = Vec::new();
        self.label_blocks = HashMap::new();
        self.fn_ret = ret.clone();

        let entry = self.alloc_block();
        self.cur = entry;

        // Pre-allocate one block per label so a forward `goto` has somewhere to point (a label
        // may be declared after the `goto` that targets it). Sorted for determinism: `Module`
        // must print byte-identically regardless of `HashSet`'s iteration order.
        let mut labelset = HashSet::new();
        collect_labels_many(body, &mut labelset);
        let mut sorted_labels: Vec<&String> = labelset.iter().collect();
        sorted_labels.sort();
        for name in sorted_labels {
            let b = self.alloc_block();
            self.label_blocks.insert(name.clone(), b);
        }

        let mut bir_params = Vec::with_capacity(f.params.len());
        for (i, (p, pty)) in f.params.iter().zip(param_tys.iter()).enumerate() {
            bir_params.push(to_bir_ty(pty));
            if let Some(name) = &p.name {
                let slot = self.new_slot(pty.clone(), BSpace::Param);
                self.bind_local(name, slot.clone());
                // An aggregate-by-value parameter shares the "whole aggregate has no BIR
                // value" gap documented above: there is no value to store here, so the
                // parameter's slot is simply never initialized (a member access against it
                // would read synthetic, unwritten storage — a known, narrow limitation, not a
                // silent miscompile of anything this pass otherwise claims to support).
                if !is_aggregate(pty) && !pty.is_unknown() {
                    let lv = self.slot_lvalue(&slot);
                    self.store_addr(&lv, ValRef::Param(i as u32));
                }
            }
        }

        for s in body {
            self.lower_stmt(s);
        }
        self.close_fallthrough();

        let name = f.name.clone();
        let blocks: Vec<BBlock> = std::mem::take(&mut self.blocks)
            .into_iter()
            .enumerate()
            .map(|(i, b)| {
                b.unwrap_or_else(|| {
                    panic!(
                        "lowering bug: block bb{i} in '{name}' was allocated but never terminated"
                    )
                })
            })
            .collect();
        let (blocks, insts) = linearize_by_block_order(blocks, std::mem::take(&mut self.insts));

        let func = BFunction {
            name,
            is_kernel,
            params: bir_params,
            ret: to_bir_ty(&ret),
            blocks,
            insts,
        };
        self.funcs.push(func);
    }

    /// Closes off whatever block is open when a function body's statement list runs out
    /// (falling off the end without an explicit `return`, or the trailing dead block left open
    /// after the body's final statement if that was itself a terminator). Defaults to a zeroed
    /// return value for a non-void function — this pass does not check for a missing `return`
    /// on every control path (the checker does not either, see its module header).
    fn close_fallthrough(&mut self) {
        let fr = self.fn_ret.clone();
        let term = if matches!(fr, Ty::Scalar(ScalarKind::Void)) {
            BTerm::Ret(None)
        } else if is_aggregate(&fr) {
            self.diags.push(
                Diag::new(ECode::LoweringUnsupported)
                    .with_arg("whole-aggregate value")
                    .with_arg(
                        "function falls through its end without a return; cannot synthesize a struct/union/array return value",
                    ),
            );
            BTerm::Ret(None)
        } else {
            let z = self.zero_of(&fr);
            BTerm::Ret(Some(z))
        };
        self.terminate(term);
    }

    fn alloc_block(&mut self) -> BlockId {
        let id = BlockId(self.blocks.len() as u32);
        self.blocks.push(None);
        id
    }

    fn push(&mut self, ty: BTy, op: BOp) -> ValRef {
        let id = InstId(self.insts.len() as u32);
        self.insts.push(BInst { ty, op });
        self.insts_by_block.entry(self.cur.0).or_default().push(id);
        ValRef::Val(id)
    }

    fn push_void(&mut self, op: BOp) {
        let id = InstId(self.insts.len() as u32);
        self.insts.push(BInst { ty: BTy::Void, op });
        self.insts_by_block.entry(self.cur.0).or_default().push(id);
    }

    /// Finalizes the currently-open block (`self.cur`) with `term`. Callers must set
    /// `self.cur` to whichever block should receive subsequent instructions next — this never
    /// does that implicitly, since some callers (`if`, ternary, `&&`/`||`) need to leave more
    /// than one already-allocated block open across several steps.
    fn terminate(&mut self, term: BTerm) {
        let ids = self.insts_by_block.remove(&self.cur.0).unwrap_or_default();
        self.blocks[self.cur.0 as usize] = Some(BBlock { insts: ids, term });
    }

    fn zero_of(&mut self, ty: &Ty) -> ValRef {
        let bty = to_bir_ty(ty);
        if ty.is_float() {
            self.push(bty, BOp::ConstFloat(0.0))
        } else {
            self.push(bty, BOp::ConstInt(0))
        }
    }

    fn new_slot(&mut self, ty: Ty, space: BSpace) -> LocalSlot {
        let slot = self.next_slot;
        self.next_slot += 1;
        LocalSlot { slot, ty, space }
    }

    fn bind_local(&mut self, name: &str, slot: LocalSlot) {
        self.locals
            .last_mut()
            .expect("at least one local scope must be open while lowering a function body")
            .insert(name.to_string(), slot);
    }

    fn find_local(&self, name: &str) -> Option<LocalSlot> {
        self.locals
            .iter()
            .rev()
            .find_map(|scope| scope.get(name).cloned())
    }

    fn slot_addr(&mut self, s: &LocalSlot) -> ValRef {
        self.push(
            BTy::Ptr(s.space),
            BOp::ConstInt(i64::from(s.slot) * SLOT_STRIDE),
        )
    }

    fn slot_lvalue(&mut self, s: &LocalSlot) -> LValue {
        let addr = self.slot_addr(s);
        LValue {
            addr,
            ty: s.ty.clone(),
            space: s.space,
        }
    }

    fn lvalue_unknown(&mut self) -> LValue {
        let addr = self.push(BTy::Ptr(BSpace::Local), BOp::ConstInt(0));
        LValue {
            addr,
            ty: Ty::Unknown,
            space: BSpace::Local,
        }
    }

    fn load_addr(&mut self, lv: &LValue) -> (ValRef, Ty) {
        let bty = to_bir_ty(&lv.ty);
        let align = natural_align(bty);
        let v = self.push(
            bty,
            BOp::Load {
                ptr: lv.addr,
                space: lv.space,
                align,
                volatile: false,
            },
        );
        (v, lv.ty.clone())
    }

    fn store_addr(&mut self, lv: &LValue, val: ValRef) {
        let bty = to_bir_ty(&lv.ty);
        let align = natural_align(bty);
        self.push_void(BOp::Store {
            ptr: lv.addr,
            val,
            ty: bty,
            space: lv.space,
            align,
            volatile: false,
        });
    }
}

// ---- Lowerer: statements --------------------------------------------------------------------
//
// Every `lower_*` statement helper below leaves `self.cur` pointing at an open (allocated,
// not yet terminated) block when it returns — including the ones (`break`/`continue`/
// `return`/`goto`) that terminate the block they were called in: they immediately open a fresh
// one for whatever (reachable or not) code textually follows. This lets every caller just keep
// appending/terminating without checking whether the previous statement already closed things
// off.

impl Lowerer {
    fn lower_stmt(&mut self, s: &Stmt) {
        match s {
            Stmt::Expr { expr, .. } => {
                self.lower_expr(expr);
            }
            Stmt::Empty { .. } => {}
            Stmt::Block { stmts, .. } => {
                self.locals.push(HashMap::new());
                for st in stmts {
                    self.lower_stmt(st);
                }
                self.locals.pop();
            }
            Stmt::Decl { decls, .. } => {
                for d in decls {
                    self.lower_var_decl(d);
                }
            }
            Stmt::If {
                cond,
                then_branch,
                else_branch,
                span,
            } => self.lower_if(cond, then_branch, else_branch.as_deref(), *span),
            Stmt::While { cond, body, span } => self.lower_while(cond, body, *span),
            Stmt::DoWhile { body, cond, span } => self.lower_do_while(body, cond, *span),
            Stmt::For {
                init,
                cond,
                step,
                body,
                span,
            } => self.lower_for(init.as_deref(), cond.as_ref(), step.as_ref(), body, *span),
            Stmt::Switch { expr, body, span } => self.lower_switch(expr, body, *span),
            // Reached only when a `case`/`default` sits outside `lower_switch`'s flattened
            // top-level view (e.g. nested/Duff's-device style, see the module header) — the
            // label itself is not dispatched to, only its wrapped statement still lowers.
            Stmt::Case { stmt, .. } => self.lower_stmt(stmt),
            Stmt::Default { stmt, .. } => self.lower_stmt(stmt),
            Stmt::Break { span } => self.lower_break(*span),
            Stmt::Continue { span } => self.lower_continue(*span),
            Stmt::Return { expr, span } => self.lower_return(expr.as_ref(), *span),
            Stmt::Label { name, stmt, .. } => self.lower_label(name, stmt),
            Stmt::Goto { label, span } => self.lower_goto(label, *span),
        }
    }

    fn lower_var_decl(&mut self, v: &VarDecl) {
        let ty = self.resolve_type(&v.ty);
        if ty.is_unknown() {
            if let Some(init) = &v.init {
                self.lower_expr(init);
            }
            let slot = self.new_slot(Ty::Unknown, BSpace::Local);
            self.bind_local(&v.name, slot);
            return;
        }
        let space = if v.cuda_quals.is_shared {
            BSpace::Shared
        } else if v.cuda_quals.is_constant {
            BSpace::Constant
        } else {
            BSpace::Local
        };
        let slot = self.new_slot(ty.clone(), space);
        self.bind_local(&v.name, slot.clone());
        if let Some(init) = &v.init {
            if is_aggregate(&ty) {
                self.diag_unsupported(
                    v.span,
                    "aggregate initializer",
                    "struct/union/array initializers are not lowered",
                );
                self.lower_expr(init);
            } else {
                let (iv, ity) = self.lower_expr(init);
                if !ity.is_unknown() {
                    let coerced = self.coerce_to(iv, &ity, &ty);
                    let lv = self.slot_lvalue(&slot);
                    self.store_addr(&lv, coerced);
                }
            }
        }
    }

    fn lower_if(&mut self, cond: &Expr, then_s: &Stmt, else_s: Option<&Stmt>, _span: FSpan) {
        let (cv, cty) = self.lower_expr(cond);
        let branch_val = if cty.is_unknown() {
            self.push(BTy::Scalar(BScalar::I1), BOp::ConstInt(1))
        } else {
            self.truthy(cv, &cty)
        };
        let then_bb = self.alloc_block();
        let else_bb = self.alloc_block();
        let merge_bb = self.alloc_block();
        self.terminate(BTerm::CondBr(branch_val, then_bb, else_bb));

        self.cur = then_bb;
        self.lower_stmt(then_s);
        self.terminate(BTerm::Br(merge_bb));

        self.cur = else_bb;
        if let Some(e) = else_s {
            self.lower_stmt(e);
        }
        self.terminate(BTerm::Br(merge_bb));

        self.cur = merge_bb;
    }

    fn lower_while(&mut self, cond: &Expr, body: &Stmt, _span: FSpan) {
        let cond_bb = self.alloc_block();
        let body_bb = self.alloc_block();
        let exit_bb = self.alloc_block();
        self.terminate(BTerm::Br(cond_bb));

        self.cur = cond_bb;
        let (cv, cty) = self.lower_expr(cond);
        let branch_val = if cty.is_unknown() {
            self.push(BTy::Scalar(BScalar::I1), BOp::ConstInt(1))
        } else {
            self.truthy(cv, &cty)
        };
        self.terminate(BTerm::CondBr(branch_val, body_bb, exit_bb));

        self.cur = body_bb;
        self.ctrl_stack.push(CtrlCtx::Loop {
            break_bb: exit_bb,
            continue_bb: cond_bb,
        });
        self.lower_stmt(body);
        self.ctrl_stack.pop();
        self.terminate(BTerm::Br(cond_bb));

        self.cur = exit_bb;
    }

    fn lower_do_while(&mut self, body: &Stmt, cond: &Expr, _span: FSpan) {
        let body_bb = self.alloc_block();
        let cond_bb = self.alloc_block();
        let exit_bb = self.alloc_block();
        self.terminate(BTerm::Br(body_bb));

        self.cur = body_bb;
        self.ctrl_stack.push(CtrlCtx::Loop {
            break_bb: exit_bb,
            continue_bb: cond_bb,
        });
        self.lower_stmt(body);
        self.ctrl_stack.pop();
        self.terminate(BTerm::Br(cond_bb));

        self.cur = cond_bb;
        let (cv, cty) = self.lower_expr(cond);
        let branch_val = if cty.is_unknown() {
            self.push(BTy::Scalar(BScalar::I1), BOp::ConstInt(1))
        } else {
            self.truthy(cv, &cty)
        };
        self.terminate(BTerm::CondBr(branch_val, body_bb, exit_bb));

        self.cur = exit_bb;
    }

    fn lower_for(
        &mut self,
        init: Option<&Stmt>,
        cond: Option<&Expr>,
        step: Option<&Expr>,
        body: &Stmt,
        _span: FSpan,
    ) {
        self.locals.push(HashMap::new());
        if let Some(i) = init {
            self.lower_stmt(i);
        }
        let cond_bb = self.alloc_block();
        let body_bb = self.alloc_block();
        let step_bb = self.alloc_block();
        let exit_bb = self.alloc_block();
        self.terminate(BTerm::Br(cond_bb));

        self.cur = cond_bb;
        if let Some(c) = cond {
            let (cv, cty) = self.lower_expr(c);
            let branch_val = if cty.is_unknown() {
                self.push(BTy::Scalar(BScalar::I1), BOp::ConstInt(1))
            } else {
                self.truthy(cv, &cty)
            };
            self.terminate(BTerm::CondBr(branch_val, body_bb, exit_bb));
        } else {
            self.terminate(BTerm::Br(body_bb));
        }

        self.cur = body_bb;
        self.ctrl_stack.push(CtrlCtx::Loop {
            break_bb: exit_bb,
            continue_bb: step_bb,
        });
        self.lower_stmt(body);
        self.ctrl_stack.pop();
        self.terminate(BTerm::Br(step_bb));

        self.cur = step_bb;
        if let Some(s) = step {
            self.lower_expr(s);
        }
        self.terminate(BTerm::Br(cond_bb));

        self.cur = exit_bb;
        self.locals.pop();
    }

    fn lower_switch(&mut self, expr: &Expr, body: &Stmt, _span: FSpan) {
        let (sv, _sty) = self.lower_expr(expr);
        let stmts = switch_body_stmts(body);
        let exit_bb = self.alloc_block();
        let entry_blocks: Vec<BlockId> = stmts.iter().map(|_| self.alloc_block()).collect();

        let mut cases: Vec<(i64, BlockId)> = Vec::new();
        let mut default_bb: Option<BlockId> = None;
        for (i, st) in stmts.iter().enumerate() {
            let (labels, _inner) = self.peel_case_labels(st, i as i64);
            for lbl in labels {
                match lbl {
                    CaseLabel::Value(v) => cases.push((v, entry_blocks[i])),
                    CaseLabel::Default => default_bb = Some(entry_blocks[i]),
                }
            }
        }
        let default_target = default_bb.unwrap_or(exit_bb);
        self.terminate(BTerm::Switch(sv, default_target, cases));

        self.ctrl_stack.push(CtrlCtx::Switch { break_bb: exit_bb });
        for (i, st) in stmts.iter().enumerate() {
            self.cur = entry_blocks[i];
            let (_labels, inner) = self.peel_case_labels(st, i as i64);
            self.lower_stmt(inner);
            let fallthrough = entry_blocks.get(i + 1).copied().unwrap_or(exit_bb);
            self.terminate(BTerm::Br(fallthrough));
        }
        self.ctrl_stack.pop();

        self.cur = exit_bb;
    }

    /// Peels leading `case`/`default` wrappers off a switch body's top-level statement,
    /// returning every label found (in source order) plus the statement they ultimately wrap.
    /// `seed` (the statement's own index within the switch) keeps synthesized fallback values
    /// for non-constant case labels from colliding with each other across a single `switch`.
    fn peel_case_labels<'e>(&mut self, s: &'e Stmt, seed: i64) -> (Vec<CaseLabel>, &'e Stmt) {
        let mut labels = Vec::new();
        let mut cur = s;
        let mut n = 0i64;
        loop {
            match cur {
                Stmt::Case { value, stmt, span } => {
                    let v = self.const_eval_i64(value).unwrap_or_else(|| {
                        self.diag_unsupported(
                            *span,
                            "non-constant case label",
                            "value must be a compile-time integer constant",
                        );
                        i64::MIN + seed * 1000 + n
                    });
                    n += 1;
                    labels.push(CaseLabel::Value(v));
                    cur = stmt;
                }
                Stmt::Default { stmt, .. } => {
                    labels.push(CaseLabel::Default);
                    cur = stmt;
                }
                _ => break,
            }
        }
        (labels, cur)
    }

    fn lower_break(&mut self, span: FSpan) {
        let target = self.ctrl_stack.last().map(|c| match c {
            CtrlCtx::Loop { break_bb, .. } | CtrlCtx::Switch { break_bb } => *break_bb,
        });
        match target {
            Some(bb) => self.terminate(BTerm::Br(bb)),
            None => {
                self.diags.push(
                    Diag::new(ECode::TypeError)
                        .with_span(conv_span(span))
                        .with_arg("'break' used outside of a loop or switch"),
                );
                self.terminate(BTerm::Ret(None));
            }
        }
        self.cur = self.alloc_block();
    }

    fn lower_continue(&mut self, span: FSpan) {
        let target = self.ctrl_stack.iter().rev().find_map(|c| match c {
            CtrlCtx::Loop { continue_bb, .. } => Some(*continue_bb),
            CtrlCtx::Switch { .. } => None,
        });
        match target {
            Some(bb) => self.terminate(BTerm::Br(bb)),
            None => {
                self.diags.push(
                    Diag::new(ECode::TypeError)
                        .with_span(conv_span(span))
                        .with_arg("'continue' used outside of a loop"),
                );
                self.terminate(BTerm::Ret(None));
            }
        }
        self.cur = self.alloc_block();
    }

    fn lower_return(&mut self, expr: Option<&Expr>, span: FSpan) {
        let fr = self.fn_ret.clone();
        let is_void = matches!(fr, Ty::Scalar(ScalarKind::Void));
        let term = match expr {
            None => BTerm::Ret(None),
            Some(e) => {
                let (v, ty) = self.lower_expr(e);
                if ty.is_unknown() {
                    BTerm::Ret(if is_void {
                        None
                    } else {
                        Some(self.zero_of(&fr))
                    })
                } else if is_aggregate(&ty) {
                    self.diag_unsupported(
                        span,
                        "whole-aggregate return",
                        "returning a struct/union/array by value has no BIR representation",
                    );
                    BTerm::Ret(if is_void {
                        None
                    } else {
                        Some(self.zero_of(&fr))
                    })
                } else {
                    let coerced = self.coerce_to(v, &ty, &fr);
                    BTerm::Ret(Some(coerced))
                }
            }
        };
        self.terminate(term);
        self.cur = self.alloc_block();
    }

    fn lower_label(&mut self, name: &str, stmt: &Stmt) {
        let bb = *self
            .label_blocks
            .get(name)
            .expect("every label was pre-allocated from the same tree this walks");
        self.terminate(BTerm::Br(bb));
        self.cur = bb;
        self.lower_stmt(stmt);
    }

    fn lower_goto(&mut self, label: &str, span: FSpan) {
        match self.label_blocks.get(label).copied() {
            Some(bb) => self.terminate(BTerm::Br(bb)),
            None => {
                self.diags.push(
                    Diag::new(ECode::UndefinedSymbol)
                        .with_span(conv_span(span))
                        .with_arg(label.to_string()),
                );
                self.terminate(BTerm::Ret(None));
            }
        }
        self.cur = self.alloc_block();
    }
}

// ---- Lowerer: expressions -------------------------------------------------------------------

impl Lowerer {
    /// A universal fallback for anything this pass cannot make sense of: a well-formed but
    /// meaningless value, paired with `Ty::Unknown` so every later use of it silently
    /// propagates instead of reporting a second diagnostic (mirrors `checker::Ty::Unknown`'s
    /// own suppression rule).
    fn placeholder(&mut self) -> (ValRef, Ty) {
        (
            self.push(BTy::Scalar(BScalar::I32), BOp::ConstInt(0)),
            Ty::Unknown,
        )
    }

    fn diag_for_unresolved_name(&mut self, name: &str, span: FSpan) {
        if CUDA_DIM3_BUILTINS.contains(&name) {
            self.diag_unsupported(span, "GPU intrinsic", name);
        } else if self.unlowered_globals.contains(name) {
            self.diag_unsupported(span, "global variable storage", name);
        } else if matches!(self.scopes.lookup_value(name), Some(ValueSym::Func(_))) {
            self.diag_unsupported(span, "function-pointer value", name);
        } else {
            self.diags.push(
                Diag::new(ECode::UndefinedSymbol)
                    .with_span(conv_span(span))
                    .with_arg(name.to_string()),
            );
        }
    }

    fn lower_expr(&mut self, e: &Expr) -> (ValRef, Ty) {
        match e {
            Expr::IntLit { value, .. } => {
                let ty = int_lit_ty(value);
                let v = self.push(to_bir_ty(&ty), BOp::ConstInt(int_lit_value(value)));
                (v, ty)
            }
            Expr::FloatLit { value, .. } => {
                let ty = float_lit_ty(value);
                let v = self.push(to_bir_ty(&ty), BOp::ConstFloat(float_lit_value(value)));
                (v, ty)
            }
            Expr::CharLit { value, .. } => {
                let ty = Ty::Scalar(ScalarKind::Char);
                let v = self.push(to_bir_ty(&ty), BOp::ConstInt(i64::from(value.value)));
                (v, ty)
            }
            Expr::StrLit { span, .. } => {
                // Same underlying gap as a module-level global: nowhere in BIR to put the
                // bytes yet. A null-ish pointer placeholder lets the rest of the expression
                // still type-check further.
                self.diag_unsupported(
                    *span,
                    "string literal",
                    "no BIR data-segment representation",
                );
                let ty = Ty::Pointer(Box::new(Ty::Scalar(ScalarKind::Char)), true);
                let v = self.push(BTy::Ptr(BSpace::Global), BOp::ConstInt(0));
                (v, ty)
            }
            Expr::Ident { name, span } => self.lower_ident_value(name, *span),
            Expr::Index { .. } => {
                let lv = self.lower_lvalue(e);
                self.value_of_lvalue(lv, e.span())
            }
            Expr::Member {
                base, name, arrow, ..
            } => {
                if !arrow {
                    if let Expr::Ident {
                        name: base_name, ..
                    } = base.as_ref()
                    {
                        if let Some(op) = gpu_index_op_for(base_name, name) {
                            let v = self.push(BTy::Scalar(BScalar::I32), op);
                            return (v, Ty::Scalar(ScalarKind::UInt));
                        }
                    }
                }
                let lv = self.lower_lvalue(e);
                self.value_of_lvalue(lv, e.span())
            }
            Expr::Unary {
                op: UnaryOp::Deref, ..
            } => {
                let lv = self.lower_lvalue(e);
                self.value_of_lvalue(lv, e.span())
            }
            Expr::Unary {
                op: UnaryOp::Addr,
                expr,
                ..
            } => {
                let lv = self.lower_lvalue(expr);
                if lv.ty.is_unknown() {
                    self.placeholder()
                } else {
                    (lv.addr, Ty::Pointer(Box::new(lv.ty), false))
                }
            }
            Expr::Unary { op, expr, span } => self.lower_unary(*op, expr, *span),
            Expr::Binary { op, lhs, rhs, span } => self.lower_binary(*op, lhs, rhs, *span),
            Expr::Assign { op, lhs, rhs, span } => self.lower_assign(*op, lhs, rhs, *span),
            Expr::Ternary {
                cond,
                then_branch,
                else_branch,
                span,
            } => self.lower_ternary(cond, then_branch, else_branch, *span),
            Expr::Cast { ty, expr, span } => self.lower_cast(ty, expr, *span),
            Expr::PreIncDec { op, expr, span } => self.lower_incdec(*op, expr, *span, true),
            Expr::PostIncDec { op, expr, span } => self.lower_incdec(*op, expr, *span, false),
            Expr::SizeofExpr { expr, .. } => {
                // The operand is lowered for its type only; sizeof does not execute at
                // runtime, but this pass has no separate non-emitting type-inference path (it
                // computes types as a side effect of value lowering), so the operand's
                // instructions still end up in the function even though nothing references
                // their result. Harmless except for an operand with an observable side effect
                // (`sizeof(x++)`), which is both rare and already dubious style.
                let (_, ty) = self.lower_expr(expr);
                let bytes = if ty.is_unknown() {
                    0
                } else {
                    self.size_of_ty(&ty)
                };
                let v = self.push(BTy::Scalar(BScalar::I64), BOp::ConstInt(bytes as i64));
                (v, Ty::Scalar(ScalarKind::ULong))
            }
            Expr::SizeofType { ty, .. } => {
                let t = self.resolve_type(ty);
                let bytes = if t.is_unknown() {
                    0
                } else {
                    self.size_of_ty(&t)
                };
                let v = self.push(BTy::Scalar(BScalar::I64), BOp::ConstInt(bytes as i64));
                (v, Ty::Scalar(ScalarKind::ULong))
            }
            Expr::Call { callee, args, span } => self.lower_call(callee, args, *span),
            Expr::KernelLaunch {
                kernel,
                grid,
                block,
                shared,
                stream,
                args,
                span,
            } => self.lower_kernel_launch(
                kernel,
                grid,
                block,
                shared.as_deref(),
                stream.as_deref(),
                args,
                *span,
            ),
            Expr::Comma { exprs, .. } => {
                let mut last = self.placeholder();
                for ex in exprs {
                    last = self.lower_expr(ex);
                }
                last
            }
            Expr::Error { .. } => self.placeholder(),
        }
    }

    fn lower_ident_value(&mut self, name: &str, span: FSpan) -> (ValRef, Ty) {
        if let Some(slot) = self.find_local(name) {
            if slot.ty.is_unknown() {
                return self.placeholder();
            }
            if is_aggregate(&slot.ty) {
                let addr = self.slot_addr(&slot);
                return match &slot.ty {
                    Ty::Array(elem) => (addr, Ty::Pointer(elem.clone(), false)),
                    _ => {
                        self.diag_unsupported(
                            span,
                            "whole-aggregate value",
                            "struct/union used by value (no BIR aggregate type)",
                        );
                        self.placeholder()
                    }
                };
            }
            let lv = self.slot_lvalue(&slot);
            return self.load_addr(&lv);
        }
        if CUDA_DIM3_BUILTINS.contains(&name) {
            self.diag_unsupported(span, "GPU intrinsic", name);
            return self.placeholder();
        }
        if let Some(&v) = self.enum_values.get(name) {
            return (
                self.push(BTy::Scalar(BScalar::I32), BOp::ConstInt(v)),
                Ty::Scalar(ScalarKind::Int),
            );
        }
        self.diag_for_unresolved_name(name, span);
        self.placeholder()
    }

    /// Computes an expression's address plus what sema type lives there. Used directly for
    /// `&expr` and assignment targets, and recursively as the base of `Index`/`Member` chains.
    fn lower_lvalue(&mut self, e: &Expr) -> LValue {
        match e {
            Expr::Ident { name, span } => {
                if let Some(slot) = self.find_local(name) {
                    return self.slot_lvalue(&slot);
                }
                if CUDA_DIM3_BUILTINS.contains(&name.as_str()) {
                    self.diag_unsupported(*span, "GPU intrinsic", name);
                } else if self.enum_values.contains_key(name) {
                    self.diags.push(
                        Diag::new(ECode::TypeError)
                            .with_span(conv_span(*span))
                            .with_arg(format!("'{name}' is not addressable")),
                    );
                } else {
                    self.diag_for_unresolved_name(name, *span);
                }
                self.lvalue_unknown()
            }
            Expr::Unary {
                op: UnaryOp::Deref,
                expr,
                span,
            } => {
                let (v, ty) = self.lower_expr(expr);
                if ty.is_unknown() {
                    return self.lvalue_unknown();
                }
                match ty.deref_target() {
                    Some(target) => LValue {
                        addr: v,
                        ty: target,
                        space: BSpace::Global,
                    },
                    None => {
                        self.diag_unsupported(*span, "dereference", "operand is not a pointer");
                        self.lvalue_unknown()
                    }
                }
            }
            Expr::Index { base, index, span } => self.lower_index_lvalue(base, index, *span),
            Expr::Member {
                base,
                name,
                arrow,
                span,
            } => self.lower_member_lvalue(base, name, *arrow, *span),
            _ => {
                self.diags.push(
                    Diag::new(ECode::TypeError)
                        .with_span(conv_span(e.span()))
                        .with_arg("expression is not addressable"),
                );
                self.lvalue_unknown()
            }
        }
    }

    fn value_of_lvalue(&mut self, lv: LValue, span: FSpan) -> (ValRef, Ty) {
        if lv.ty.is_unknown() {
            return self.placeholder();
        }
        if is_aggregate(&lv.ty) {
            return match &lv.ty {
                Ty::Array(elem) => (lv.addr, Ty::Pointer(elem.clone(), false)),
                _ => {
                    self.diag_unsupported(
                        span,
                        "whole-aggregate value",
                        "struct/union used by value (no BIR aggregate type)",
                    );
                    self.placeholder()
                }
            };
        }
        self.load_addr(&lv)
    }

    fn lower_index_lvalue(&mut self, base: &Expr, index: &Expr, span: FSpan) -> LValue {
        let (idx_val, idx_ty) = self.lower_expr(index);
        if idx_ty.is_unknown() {
            return self.lvalue_unknown();
        }
        let (base_addr, base_space, elem_ty) = if is_lvalue_shaped(base) {
            let lv = self.lower_lvalue(base);
            if lv.ty.is_unknown() {
                return self.lvalue_unknown();
            }
            match lv.ty.clone() {
                Ty::Array(elem) => (lv.addr, lv.space, *elem),
                Ty::Pointer(elem, _) => {
                    let (v, _) = self.load_addr(&lv);
                    (v, BSpace::Global, *elem)
                }
                _ => {
                    self.diag_unsupported(span, "indexing", "base is not an array or pointer");
                    return self.lvalue_unknown();
                }
            }
        } else {
            let (v, ty) = self.lower_expr(base);
            if ty.is_unknown() {
                return self.lvalue_unknown();
            }
            match ty.deref_target() {
                Some(elem) => (v, BSpace::Global, elem),
                None => {
                    self.diag_unsupported(span, "indexing", "base is not an array or pointer");
                    return self.lvalue_unknown();
                }
            }
        };
        let esz = self.size_of_ty(&elem_ty) as i64;
        let idx64 = self.widen_index_i64(idx_val, &idx_ty);
        let i64t = BTy::Scalar(BScalar::I64);
        let esz_val = self.push(i64t, BOp::ConstInt(esz));
        let byte_off = self.push(i64t, BOp::Bin(BBin::Mul, idx64, esz_val));
        let ptrty = BTy::Ptr(base_space);
        let addr = self.push(ptrty, BOp::Bin(BBin::Add, base_addr, byte_off));
        LValue {
            addr,
            ty: elem_ty,
            space: base_space,
        }
    }

    fn lower_member_lvalue(&mut self, base: &Expr, name: &str, arrow: bool, span: FSpan) -> LValue {
        let (base_addr, base_space, struct_ty) = if arrow {
            let (v, ty) = self.lower_expr(base);
            if ty.is_unknown() {
                return self.lvalue_unknown();
            }
            match ty.deref_target() {
                Some(t) => (v, BSpace::Global, t),
                None => {
                    self.diag_unsupported(span, "member access", "'->' on a non-pointer type");
                    return self.lvalue_unknown();
                }
            }
        } else {
            let lv = self.lower_lvalue(base);
            if lv.ty.is_unknown() {
                return self.lvalue_unknown();
            }
            (lv.addr, lv.space, lv.ty)
        };
        if !matches!(struct_ty, Ty::Struct(_) | Ty::Union(_)) {
            self.diag_unsupported(span, "member access", "base is not a struct/union");
            return self.lvalue_unknown();
        }
        match self.field_offset(&struct_ty, name) {
            Some((off, fty)) => {
                let off_val = self.push(BTy::Scalar(BScalar::I64), BOp::ConstInt(off as i64));
                let addr = self.push(
                    BTy::Ptr(base_space),
                    BOp::Bin(BBin::Add, base_addr, off_val),
                );
                LValue {
                    addr,
                    ty: fty,
                    space: base_space,
                }
            }
            None => {
                self.diag_unsupported(span, "member access", format!("unknown field '{name}'"));
                self.lvalue_unknown()
            }
        }
    }
}

// ---- Lowerer: operators, casts, calls -------------------------------------------------------

impl Lowerer {
    /// Converts an already-lowered value from one sema type to another, choosing BIR's cast
    /// opcode from the source/destination scalar shapes. Used both for an explicit `(T)expr`
    /// cast and for C's implicit arithmetic promotion (widening both operands of a binary op
    /// to their common type before applying it).
    fn coerce_to(&mut self, v: ValRef, from: &Ty, to: &Ty) -> ValRef {
        let from_b = to_bir_ty(from);
        let to_b = to_bir_ty(to);
        if from_b == to_b {
            return v;
        }
        match (from_b, to_b) {
            (BTy::Scalar(fs), BTy::Scalar(ts)) => {
                let f_float = is_float_scalar(fs);
                let t_float = is_float_scalar(ts);
                let op = if f_float && t_float {
                    if scalar_bits(ts) > scalar_bits(fs) {
                        BCast::FpExt
                    } else {
                        BCast::FpTrunc
                    }
                } else if f_float {
                    if is_signed(to) {
                        BCast::FpToSi
                    } else {
                        BCast::FpToUi
                    }
                } else if t_float {
                    if is_signed(from) {
                        BCast::SiToFp
                    } else {
                        BCast::UiToFp
                    }
                } else if scalar_bits(ts) > scalar_bits(fs) {
                    if is_signed(from) {
                        BCast::Sext
                    } else {
                        BCast::Zext
                    }
                } else {
                    BCast::Trunc
                };
                self.push(to_b, BOp::Cast(op, from_b, v))
            }
            _ => {
                // Pointer<->pointer, pointer<->integer, enum<->int, or anything else this pass
                // does not model precisely: `bitcast` is the closest existing opcode. BIR's
                // pointers are opaque (no defined bit pattern), so a pointer/integer bitcast
                // here is at best a best-effort placeholder conversion, not a real reinterpret
                // — a documented limitation, not a claim of correctness.
                self.push(to_b, BOp::Cast(BCast::Bitcast, from_b, v))
            }
        }
    }

    fn widen_index_i64(&mut self, v: ValRef, ty: &Ty) -> ValRef {
        match to_bir_ty(ty) {
            BTy::Scalar(BScalar::I64) => v,
            BTy::Scalar(_) => self.coerce_to(v, ty, &Ty::Scalar(ScalarKind::Long)),
            _ => v,
        }
    }

    fn truthy(&mut self, v: ValRef, ty: &Ty) -> ValRef {
        let bty = to_bir_ty(ty);
        let i1 = BTy::Scalar(BScalar::I1);
        if ty.is_float() {
            let zero = self.push(bty, BOp::ConstFloat(0.0));
            self.push(i1, BOp::FCmp(FCmpPred::One, bty, v, zero))
        } else {
            let zero = self.push(bty, BOp::ConstInt(0));
            self.push(i1, BOp::ICmp(ICmpPred::Ne, bty, v, zero))
        }
    }

    fn lower_cast(&mut self, ty: &Type, expr: &Expr, span: FSpan) -> (ValRef, Ty) {
        let target = self.resolve_type(ty);
        let (v, src) = self.lower_expr(expr);
        if src.is_unknown() || target.is_unknown() {
            return self.placeholder();
        }
        if is_aggregate(&target) || is_aggregate(&src) {
            self.diag_unsupported(
                span,
                "cast involving a struct/union/array",
                "no BIR aggregate type",
            );
            return self.placeholder();
        }
        (self.coerce_to(v, &src, &target), target)
    }

    fn lower_unary(&mut self, op: UnaryOp, expr: &Expr, _span: FSpan) -> (ValRef, Ty) {
        let (v, ty) = self.lower_expr(expr);
        if ty.is_unknown() {
            return self.placeholder();
        }
        let bty = to_bir_ty(&ty);
        match op {
            UnaryOp::Plus => (v, ty),
            UnaryOp::Neg => {
                if ty.is_float() {
                    let zero = self.push(bty, BOp::ConstFloat(0.0));
                    (self.push(bty, BOp::Bin(BBin::FSub, zero, v)), ty)
                } else {
                    let zero = self.push(bty, BOp::ConstInt(0));
                    (self.push(bty, BOp::Bin(BBin::Sub, zero, v)), ty)
                }
            }
            UnaryOp::BitNot => {
                let allones = self.push(bty, BOp::ConstInt(-1));
                (self.push(bty, BOp::Bin(BBin::Xor, v, allones)), ty)
            }
            UnaryOp::Not => {
                // `!x` is `x == 0`, result an `int` per C — computed directly rather than
                // negating `truthy`'s `x != 0`, which would need a second comparison anyway.
                let i32t = BTy::Scalar(BScalar::I32);
                let zero_op = if ty.is_float() {
                    BOp::ConstFloat(0.0)
                } else {
                    BOp::ConstInt(0)
                };
                let zero = self.push(bty, zero_op);
                let cmp = if ty.is_float() {
                    self.push(
                        BTy::Scalar(BScalar::I1),
                        BOp::FCmp(FCmpPred::Oeq, bty, v, zero),
                    )
                } else {
                    self.push(
                        BTy::Scalar(BScalar::I1),
                        BOp::ICmp(ICmpPred::Eq, bty, v, zero),
                    )
                };
                (
                    self.push(i32t, BOp::Cast(BCast::Zext, BTy::Scalar(BScalar::I1), cmp)),
                    Ty::Scalar(ScalarKind::Int),
                )
            }
            UnaryOp::Deref | UnaryOp::Addr => {
                unreachable!("Deref/Addr are handled directly in lower_expr/lower_lvalue")
            }
        }
    }
}

// ---- Lowerer: binary operators ---------------------------------------------------------------

impl Lowerer {
    fn lower_binary(&mut self, op: ABin, lhs: &Expr, rhs: &Expr, span: FSpan) -> (ValRef, Ty) {
        if matches!(op, ABin::LogOr | ABin::LogAnd) {
            return self.lower_logical(op, lhs, rhs);
        }
        let (lv, lty) = self.lower_expr(lhs);
        let (rv, rty) = self.lower_expr(rhs);
        if lty.is_unknown() || rty.is_unknown() {
            return self.placeholder();
        }
        match op {
            ABin::Eq | ABin::Ne | ABin::Lt | ABin::Gt | ABin::Le | ABin::Ge => {
                self.lower_compare(op, lv, &lty, rv, &rty)
            }
            ABin::Add => self.lower_add(lv, &lty, rv, &rty, span),
            ABin::Sub => self.lower_sub(lv, &lty, rv, &rty, span),
            // The checker already rejected non-numeric `Mul`/`Div`/`Rem` and non-integer
            // bitwise/shift operands; this pass assumes that already happened and does not
            // re-validate it.
            ABin::Mul
            | ABin::Div
            | ABin::Rem
            | ABin::BitOr
            | ABin::BitXor
            | ABin::BitAnd
            | ABin::Shl
            | ABin::Shr => self.lower_arith(op, lv, &lty, rv, &rty),
            ABin::LogOr | ABin::LogAnd => unreachable!("handled above"),
        }
    }

    /// `&&`/`||`: lowered via a branch to a merge block with a `phi`, not eagerly, since C
    /// requires short-circuit evaluation (the right operand's side effects must not happen
    /// when the left operand alone decides the result).
    fn lower_logical(&mut self, op: ABin, lhs: &Expr, rhs: &Expr) -> (ValRef, Ty) {
        let (lv, lty) = self.lower_expr(lhs);
        if lty.is_unknown() {
            return self.placeholder();
        }
        let lb = self.truthy(lv, &lty);
        let i32t = BTy::Scalar(BScalar::I32);
        let short_const = if matches!(op, ABin::LogAnd) { 0 } else { 1 };
        let short_val = self.push(i32t, BOp::ConstInt(short_const));
        let short_bb = self.cur;

        let rhs_bb = self.alloc_block();
        let merge_bb = self.alloc_block();
        match op {
            ABin::LogAnd => self.terminate(BTerm::CondBr(lb, rhs_bb, merge_bb)),
            ABin::LogOr => self.terminate(BTerm::CondBr(lb, merge_bb, rhs_bb)),
            _ => unreachable!(),
        }

        self.cur = rhs_bb;
        let (rv, rty) = self.lower_expr(rhs);
        let rb32 = if rty.is_unknown() {
            self.push(i32t, BOp::ConstInt(0))
        } else {
            let rb = self.truthy(rv, &rty);
            self.push(i32t, BOp::Cast(BCast::Zext, BTy::Scalar(BScalar::I1), rb))
        };
        let rhs_end_bb = self.cur;
        self.terminate(BTerm::Br(merge_bb));

        self.cur = merge_bb;
        let phi = self.push(
            i32t,
            BOp::Phi(vec![(short_bb, short_val), (rhs_end_bb, rb32)]),
        );
        (phi, Ty::Scalar(ScalarKind::Int))
    }

    fn lower_compare(
        &mut self,
        op: ABin,
        lv: ValRef,
        lty: &Ty,
        rv: ValRef,
        rty: &Ty,
    ) -> (ValRef, Ty) {
        let i1 = BTy::Scalar(BScalar::I1);
        let is_ptr_cmp = lty.is_pointer_like() || rty.is_pointer_like();
        let cmp = if is_ptr_cmp {
            let cmp_ty = BTy::Ptr(BSpace::Global);
            let pred = match op {
                ABin::Eq => ICmpPred::Eq,
                ABin::Ne => ICmpPred::Ne,
                ABin::Lt => ICmpPred::Ult,
                ABin::Gt => ICmpPred::Ugt,
                ABin::Le => ICmpPred::Ule,
                ABin::Ge => ICmpPred::Uge,
                _ => unreachable!(),
            };
            self.push(i1, BOp::ICmp(pred, cmp_ty, lv, rv))
        } else if lty.is_float() || rty.is_float() {
            let ct = promote(lty, rty);
            let bty = to_bir_ty(&ct);
            let l2 = self.coerce_to(lv, lty, &ct);
            let r2 = self.coerce_to(rv, rty, &ct);
            let pred = match op {
                ABin::Eq => FCmpPred::Oeq,
                ABin::Ne => FCmpPred::One,
                ABin::Lt => FCmpPred::Olt,
                ABin::Gt => FCmpPred::Ogt,
                ABin::Le => FCmpPred::Ole,
                ABin::Ge => FCmpPred::Oge,
                _ => unreachable!(),
            };
            self.push(i1, BOp::FCmp(pred, bty, l2, r2))
        } else {
            let ct = promote(lty, rty);
            let bty = to_bir_ty(&ct);
            let l2 = self.coerce_to(lv, lty, &ct);
            let r2 = self.coerce_to(rv, rty, &ct);
            let signed = is_signed(&ct);
            let pred = match op {
                ABin::Eq => ICmpPred::Eq,
                ABin::Ne => ICmpPred::Ne,
                ABin::Lt => {
                    if signed {
                        ICmpPred::Slt
                    } else {
                        ICmpPred::Ult
                    }
                }
                ABin::Gt => {
                    if signed {
                        ICmpPred::Sgt
                    } else {
                        ICmpPred::Ugt
                    }
                }
                ABin::Le => {
                    if signed {
                        ICmpPred::Sle
                    } else {
                        ICmpPred::Ule
                    }
                }
                ABin::Ge => {
                    if signed {
                        ICmpPred::Sge
                    } else {
                        ICmpPred::Uge
                    }
                }
                _ => unreachable!(),
            };
            self.push(i1, BOp::ICmp(pred, bty, l2, r2))
        };
        let i32t = BTy::Scalar(BScalar::I32);
        (
            self.push(i32t, BOp::Cast(BCast::Zext, i1, cmp)),
            Ty::Scalar(ScalarKind::Int),
        )
    }

    /// Mul/Div/Rem/bitwise/shift/plain-arithmetic Add/Sub, after ruling out pointer arithmetic
    /// (the caller handles `Add`/`Sub` pointer cases before falling here). See the module
    /// header for the `div`/`rem` signedness gap this inherits from BIR.
    fn lower_arith(
        &mut self,
        op: ABin,
        lv: ValRef,
        lty: &Ty,
        rv: ValRef,
        rty: &Ty,
    ) -> (ValRef, Ty) {
        let result_ty = promote(lty, rty);
        let bty = to_bir_ty(&result_ty);
        let l2 = self.coerce_to(lv, lty, &result_ty);
        let r2 = self.coerce_to(rv, rty, &result_ty);
        let is_float = result_ty.is_float();
        let signed = is_signed(&result_ty);
        let bop = match op {
            ABin::Add => {
                if is_float {
                    BBin::FAdd
                } else {
                    BBin::Add
                }
            }
            ABin::Sub => {
                if is_float {
                    BBin::FSub
                } else {
                    BBin::Sub
                }
            }
            ABin::Mul => {
                if is_float {
                    BBin::FMul
                } else {
                    BBin::Mul
                }
            }
            ABin::Div => {
                if is_float {
                    BBin::FDiv
                } else {
                    BBin::Div
                }
            }
            ABin::Rem => {
                if is_float {
                    BBin::FRem
                } else {
                    BBin::Rem
                }
            }
            ABin::BitOr => BBin::Or,
            ABin::BitXor => BBin::Xor,
            ABin::BitAnd => BBin::And,
            ABin::Shl => BBin::Shl,
            ABin::Shr => {
                if signed {
                    BBin::Ashr
                } else {
                    BBin::Lshr
                }
            }
            ABin::Eq
            | ABin::Ne
            | ABin::Lt
            | ABin::Gt
            | ABin::Le
            | ABin::Ge
            | ABin::LogOr
            | ABin::LogAnd => {
                unreachable!("handled by lower_compare/lower_logical")
            }
        };
        (self.push(bty, BOp::Bin(bop, l2, r2)), result_ty)
    }

    fn lower_add(
        &mut self,
        lv: ValRef,
        lty: &Ty,
        rv: ValRef,
        rty: &Ty,
        span: FSpan,
    ) -> (ValRef, Ty) {
        if lty.is_pointer_like() && rty.is_integer() {
            return self.lower_ptr_offset(lv, lty, rv, rty);
        }
        if rty.is_pointer_like() && lty.is_integer() {
            return self.lower_ptr_offset(rv, rty, lv, lty);
        }
        if lty.is_arithmetic() && rty.is_arithmetic() {
            return self.lower_arith(ABin::Add, lv, lty, rv, rty);
        }
        self.diag_unsupported(
            span,
            "'+' operand combination",
            "not arithmetic or pointer+integer",
        );
        self.placeholder()
    }

    fn lower_sub(
        &mut self,
        lv: ValRef,
        lty: &Ty,
        rv: ValRef,
        rty: &Ty,
        span: FSpan,
    ) -> (ValRef, Ty) {
        if lty.is_pointer_like() && rty.is_pointer_like() {
            self.diag_unsupported(
                span,
                "pointer difference",
                "BIR pointers are opaque; no integer representation to subtract",
            );
            return self.placeholder();
        }
        if lty.is_pointer_like() && rty.is_integer() {
            let ity = Ty::Scalar(ScalarKind::Long);
            let i64t = BTy::Scalar(BScalar::I64);
            let r64 = self.widen_index_i64(rv, rty);
            let zero = self.push(i64t, BOp::ConstInt(0));
            let neg = self.push(i64t, BOp::Bin(BBin::Sub, zero, r64));
            return self.lower_ptr_offset(lv, lty, neg, &ity);
        }
        if lty.is_arithmetic() && rty.is_arithmetic() {
            return self.lower_arith(ABin::Sub, lv, lty, rv, rty);
        }
        self.diag_unsupported(
            span,
            "'-' operand combination",
            "not arithmetic or pointer-integer",
        );
        self.placeholder()
    }

    fn lower_ptr_offset(&mut self, pv: ValRef, pty: &Ty, iv: ValRef, ity: &Ty) -> (ValRef, Ty) {
        let elem = pty.deref_target().unwrap_or(Ty::Unknown);
        if elem.is_unknown() {
            return self.placeholder();
        }
        let esz = self.size_of_ty(&elem) as i64;
        let idx64 = self.widen_index_i64(iv, ity);
        let i64t = BTy::Scalar(BScalar::I64);
        let esz_val = self.push(i64t, BOp::ConstInt(esz));
        let byte_off = self.push(i64t, BOp::Bin(BBin::Mul, idx64, esz_val));
        // Arbitrary pointer-*value* arithmetic (as opposed to a known local's own storage,
        // handled through `LValue`) defaults to `Global` — see the module header.
        let addr = self.push(BTy::Ptr(BSpace::Global), BOp::Bin(BBin::Add, pv, byte_off));
        (addr, Ty::Pointer(Box::new(elem), false))
    }
}

// ---- Lowerer: assignment, increment/decrement, ternary, calls -------------------------------

impl Lowerer {
    fn lower_assign(&mut self, op: AssignOp, lhs: &Expr, rhs: &Expr, span: FSpan) -> (ValRef, Ty) {
        let lv = self.lower_lvalue(lhs);
        if lv.ty.is_unknown() {
            self.lower_expr(rhs);
            return self.placeholder();
        }
        if is_aggregate(&lv.ty) {
            self.diag_unsupported(
                span,
                "whole-aggregate assignment",
                "struct/union/array copy has no BIR representation",
            );
            self.lower_expr(rhs);
            return self.placeholder();
        }
        match op {
            AssignOp::Assign => {
                let (rv, rty) = self.lower_expr(rhs);
                if rty.is_unknown() {
                    return self.placeholder();
                }
                let coerced = self.coerce_to(rv, &rty, &lv.ty);
                self.store_addr(&lv, coerced);
                (coerced, lv.ty)
            }
            _ => {
                let bin = compound_binop(op);
                let (rv, rty) = self.lower_expr(rhs);
                if rty.is_unknown() {
                    return self.placeholder();
                }
                let (cur_v, cur_ty) = self.load_addr(&lv);
                let (result_v, result_ty) = match bin {
                    ABin::Add => self.lower_add(cur_v, &cur_ty, rv, &rty, span),
                    ABin::Sub => self.lower_sub(cur_v, &cur_ty, rv, &rty, span),
                    other => self.lower_arith(other, cur_v, &cur_ty, rv, &rty),
                };
                if result_ty.is_unknown() {
                    return self.placeholder();
                }
                let coerced = self.coerce_to(result_v, &result_ty, &lv.ty);
                self.store_addr(&lv, coerced);
                (coerced, lv.ty)
            }
        }
    }

    fn lower_incdec(
        &mut self,
        op: IncDecOp,
        expr: &Expr,
        _span: FSpan,
        is_pre: bool,
    ) -> (ValRef, Ty) {
        let lv = self.lower_lvalue(expr);
        if lv.ty.is_unknown() {
            return self.placeholder();
        }
        let (cur_v, cur_ty) = self.load_addr(&lv);
        let (new_v, new_ty) = if cur_ty.is_pointer() {
            let delta = if matches!(op, IncDecOp::Inc) { 1 } else { -1 };
            let deltav = self.push(BTy::Scalar(BScalar::I64), BOp::ConstInt(delta));
            self.lower_ptr_offset(cur_v, &cur_ty, deltav, &Ty::Scalar(ScalarKind::Long))
        } else if cur_ty.is_float() {
            let bty = to_bir_ty(&cur_ty);
            let one = self.push(bty, BOp::ConstFloat(1.0));
            let bop = if matches!(op, IncDecOp::Inc) {
                BBin::FAdd
            } else {
                BBin::FSub
            };
            (self.push(bty, BOp::Bin(bop, cur_v, one)), cur_ty.clone())
        } else {
            let bty = to_bir_ty(&cur_ty);
            let one = self.push(bty, BOp::ConstInt(1));
            let bop = if matches!(op, IncDecOp::Inc) {
                BBin::Add
            } else {
                BBin::Sub
            };
            (self.push(bty, BOp::Bin(bop, cur_v, one)), cur_ty.clone())
        };
        let coerced = self.coerce_to(new_v, &new_ty, &cur_ty);
        self.store_addr(&lv, coerced);
        if is_pre {
            (coerced, cur_ty)
        } else {
            (cur_v, cur_ty)
        }
    }

    fn lower_ternary(
        &mut self,
        cond: &Expr,
        then_e: &Expr,
        else_e: &Expr,
        span: FSpan,
    ) -> (ValRef, Ty) {
        let (cv, cty) = self.lower_expr(cond);
        if cty.is_unknown() {
            self.lower_expr(then_e);
            self.lower_expr(else_e);
            return self.placeholder();
        }
        let cb = self.truthy(cv, &cty);
        let then_bb = self.alloc_block();
        let else_bb = self.alloc_block();
        let merge_bb = self.alloc_block();
        self.terminate(BTerm::CondBr(cb, then_bb, else_bb));

        self.cur = then_bb;
        let (tv, tty) = self.lower_expr(then_e);
        let then_end = self.cur;

        self.cur = else_bb;
        let (ev, ety) = self.lower_expr(else_e);
        let else_end = self.cur;

        if tty.is_unknown() || ety.is_unknown() {
            self.cur = then_end;
            self.terminate(BTerm::Br(merge_bb));
            self.cur = else_end;
            self.terminate(BTerm::Br(merge_bb));
            self.cur = merge_bb;
            return self.placeholder();
        }

        let result_ty = if assignable(&tty, &ety) {
            tty.clone()
        } else if assignable(&ety, &tty) {
            ety.clone()
        } else {
            self.diag_unsupported(
                span,
                "ternary branch types",
                "incompatible types in the two branches",
            );
            self.cur = then_end;
            self.terminate(BTerm::Br(merge_bb));
            self.cur = else_end;
            self.terminate(BTerm::Br(merge_bb));
            self.cur = merge_bb;
            return self.placeholder();
        };
        if is_aggregate(&result_ty) {
            self.diag_unsupported(
                span,
                "whole-aggregate value",
                "ternary over a struct/union/array",
            );
            self.cur = then_end;
            self.terminate(BTerm::Br(merge_bb));
            self.cur = else_end;
            self.terminate(BTerm::Br(merge_bb));
            self.cur = merge_bb;
            return self.placeholder();
        }

        let bty = to_bir_ty(&result_ty);
        self.cur = then_end;
        let tv2 = self.coerce_to(tv, &tty, &result_ty);
        self.terminate(BTerm::Br(merge_bb));

        self.cur = else_end;
        let ev2 = self.coerce_to(ev, &ety, &result_ty);
        self.terminate(BTerm::Br(merge_bb));

        self.cur = merge_bb;
        let phi = self.push(bty, BOp::Phi(vec![(then_end, tv2), (else_end, ev2)]));
        (phi, result_ty)
    }

    /// GPU intrinsic calls (`__syncthreads`, shuffle/vote/atomic builtins, the CUDA Runtime
    /// API) each have a dedicated BIR op, so they are special-cased by callee name below rather
    /// than falling through to the generic path.
    ///
    /// The generic path (see the module header) resolves a plain named callee against
    /// `ValueSym::Func` and, so long as neither its return type nor any argument's type is a
    /// whole struct/union/array (BIR has no aggregate value type — the same `E304` every other
    /// whole-aggregate-value use in this pass already reports), emits a real `Op::Call`. An
    /// unresolvable callee (not a plain identifier, or not a known function — the checker
    /// already reported the latter) still lowers every argument for its own side effects,
    /// matching real evaluation order as far as it goes, and stands in a zeroed placeholder of
    /// the statically-known return type in place of the call itself.
    fn lower_call(&mut self, callee: &Expr, args: &[Expr], span: FSpan) -> (ValRef, Ty) {
        if let Expr::Ident { name, .. } = callee {
            let name = name.as_str();
            if name == "__syncthreads" {
                return self.lower_syncthreads_call(args);
            }
            if let Some(kind) = shuffle_kind_for(name) {
                return self.lower_shuffle_call(kind, args);
            }
            if let Some(ctor) = vote_ctor_for(name) {
                return self.lower_vote_call(ctor, args);
            }
            if let Some(aop) = atomic_op_for(name) {
                return self.lower_atomic_rmw_call(aop, args, span);
            }
            if name == CUDA_ATOMIC_CAS_BUILTIN {
                return self.lower_atomic_cas_call(args, span);
            }
            if name == CUDA_MALLOC_BUILTIN {
                return self.lower_cuda_malloc_call(args, span);
            }
            if name == CUDA_MEMCPY_BUILTIN {
                return self.lower_cuda_memcpy_call(args);
            }
            if name == CUDA_FREE_BUILTIN {
                return self.lower_cuda_free_call(args);
            }
            if name == CUDA_DEVICE_SYNCHRONIZE_BUILTIN {
                return self.lower_cuda_device_synchronize_call(args);
            }
        }
        let resolved = if let Expr::Ident { name, .. } = callee {
            match self.scopes.lookup_value(name).cloned() {
                Some(ValueSym::Func(sig)) => Some((name.clone(), sig.ret)),
                _ => None,
            }
        } else {
            self.lower_expr(callee);
            None
        };

        let arg_vals = self.lower_call_args(args);

        let Some((name, ret_ty)) = resolved else {
            return self.placeholder();
        };

        if is_aggregate(&ret_ty) || arg_vals.iter().any(|(_, t)| is_aggregate(t)) {
            self.diag_unsupported(
                span,
                "function call",
                "a whole struct/union/array argument or return value has no BIR value to lower to",
            );
            return if ret_ty.is_unknown() || is_aggregate(&ret_ty) {
                self.placeholder()
            } else {
                let v = self.zero_of(&ret_ty);
                (v, ret_ty)
            };
        }
        if ret_ty.is_unknown() {
            // Only reachable for a callee the checker already flagged (undefined symbol, or a
            // non-function value called); nothing sound to lower to.
            return self.placeholder();
        }

        let bargs: Vec<ValRef> = arg_vals.into_iter().map(|(v, _)| v).collect();
        if matches!(ret_ty, Ty::Scalar(ScalarKind::Void)) {
            self.push_void(BOp::Call {
                func: name,
                args: bargs,
            });
            (self.zero_of(&ret_ty), ret_ty)
        } else {
            let bty = to_bir_ty(&ret_ty);
            let v = self.push(
                bty,
                BOp::Call {
                    func: name,
                    args: bargs,
                },
            );
            (v, ret_ty)
        }
    }

    /// `__syncthreads()` -> `barrier`. Any arguments (there should be none — the checker
    /// enforces zero-arity) are still lowered first for their side effects, matching the
    /// generic call path's evaluation-order guarantee.
    fn lower_syncthreads_call(&mut self, args: &[Expr]) -> (ValRef, Ty) {
        for a in args {
            self.lower_expr(a);
        }
        self.push_void(BOp::Barrier);
        (
            self.zero_of(&Ty::Scalar(ScalarKind::Void)),
            Ty::Scalar(ScalarKind::Void),
        )
    }

    /// Lowers every call argument left to right, for evaluation-order and side-effect parity
    /// with the generic call path even when a builtin's fixed arity does not match what was
    /// actually written (malformed input the checker should already have flagged; this pass
    /// degrades to a placeholder rather than indexing out of bounds).
    fn lower_call_args(&mut self, args: &[Expr]) -> Vec<(ValRef, Ty)> {
        args.iter().map(|a| self.lower_expr(a)).collect()
    }

    /// `__shfl`/`__shfl_up`/`__shfl_down`/`__shfl_xor(value, lane_or_offset)` -> `shuffle.*`.
    /// Both operands are coerced to `int`, matching the `int`-only builtin signature
    /// `checker::seed_cuda_builtins` declares (see the module header).
    fn lower_shuffle_call(&mut self, kind: BShuffleKind, args: &[Expr]) -> (ValRef, Ty) {
        let vals = self.lower_call_args(args);
        if vals.len() != 2 || vals.iter().any(|(_, t)| t.is_unknown()) {
            return self.placeholder();
        }
        let ity = Ty::Scalar(ScalarKind::Int);
        let v = self.coerce_to(vals[0].0, &vals[0].1, &ity);
        let lane = self.coerce_to(vals[1].0, &vals[1].1, &ity);
        let i32t = BTy::Scalar(BScalar::I32);
        (self.push(i32t, BOp::Shuffle(kind, v, lane)), ity)
    }

    /// `__ballot`/`__any`/`__all(predicate)` -> `ballot`/`vote.any`/`vote.all`.
    fn lower_vote_call(&mut self, ctor: fn(ValRef) -> BOp, args: &[Expr]) -> (ValRef, Ty) {
        let vals = self.lower_call_args(args);
        if vals.len() != 1 || vals[0].1.is_unknown() {
            return self.placeholder();
        }
        let ity = Ty::Scalar(ScalarKind::Int);
        let pred = self.coerce_to(vals[0].0, &vals[0].1, &ity);
        let i32t = BTy::Scalar(BScalar::I32);
        (self.push(i32t, ctor(pred)), ity)
    }

    /// `atomicAdd`/`atomicSub`/`atomicExch`/`atomicMin`/`atomicMax`/`atomicAnd`/`atomicOr`/
    /// `atomicXor(address, value)` -> `atomic.*`. `address`'s own lowered value is used
    /// directly as the op's pointer operand (it is already the address, not something to load
    /// through); `value` is coerced to `int`, per the module header's documented `int`-only
    /// simplification.
    fn lower_atomic_rmw_call(
        &mut self,
        aop: BAtomicOp,
        args: &[Expr],
        span: FSpan,
    ) -> (ValRef, Ty) {
        let vals = self.lower_call_args(args);
        if vals.len() != 2 || vals.iter().any(|(_, t)| t.is_unknown()) {
            return self.placeholder();
        }
        let (addr, addr_ty) = &vals[0];
        if !addr_ty.is_pointer_like() {
            self.diag_unsupported(span, "atomic builtin", "first argument is not a pointer");
            return self.placeholder();
        }
        let ity = Ty::Scalar(ScalarKind::Int);
        let val = self.coerce_to(vals[1].0, &vals[1].1, &ity);
        let i32t = BTy::Scalar(BScalar::I32);
        (
            self.push(i32t, BOp::Atomic(aop, *addr, val, BSpace::Global)),
            ity,
        )
    }

    /// `atomicCAS(address, compare, value)` -> `atomic.cas`, same operand handling as
    /// `lower_atomic_rmw_call`.
    fn lower_atomic_cas_call(&mut self, args: &[Expr], span: FSpan) -> (ValRef, Ty) {
        let vals = self.lower_call_args(args);
        if vals.len() != 3 || vals.iter().any(|(_, t)| t.is_unknown()) {
            return self.placeholder();
        }
        let (addr, addr_ty) = &vals[0];
        if !addr_ty.is_pointer_like() {
            self.diag_unsupported(span, "atomic builtin", "first argument is not a pointer");
            return self.placeholder();
        }
        let ity = Ty::Scalar(ScalarKind::Int);
        let cmp = self.coerce_to(vals[1].0, &vals[1].1, &ity);
        let new = self.coerce_to(vals[2].0, &vals[2].1, &ity);
        let i32t = BTy::Scalar(BScalar::I32);
        (
            self.push(i32t, BOp::AtomicCas(*addr, cmp, new, BSpace::Global)),
            ity,
        )
    }

    /// Lowers a checked `kernel<<<grid, block[, shared[, stream]]>>>(args...)` to
    /// `Op::KernelLaunch`. Deliberately not routed through `lower_call` at all — a launch is
    /// never a generic call (see the module header and `basalt_bir::Op::KernelLaunch`'s own
    /// doc comment) — so this is its own top-level entry point from `lower_expr`, the same way
    /// `Expr::KernelLaunch` is its own top-level `Expr` variant rather than a shape of `Call`.
    #[allow(clippy::too_many_arguments)]
    fn lower_kernel_launch(
        &mut self,
        kernel: &Expr,
        grid: &Expr,
        block: &Expr,
        shared: Option<&Expr>,
        stream: Option<&Expr>,
        args: &[Expr],
        span: FSpan,
    ) -> (ValRef, Ty) {
        let void_ty = Ty::Scalar(ScalarKind::Void);
        let Expr::Ident { name, .. } = kernel else {
            self.lower_expr(kernel);
            for a in args {
                self.lower_expr(a);
            }
            self.diag_unsupported(
                span,
                "kernel launch",
                "launch target is not a named function",
            );
            return (self.zero_of(&void_ty), void_ty);
        };

        let grid_vals = self.lower_launch_config_dim(grid);
        let block_vals = self.lower_launch_config_dim(block);
        let shared_val = match shared {
            Some(e) => {
                let (v, t) = self.lower_expr(e);
                self.coerce_to(v, &t, &Ty::Scalar(ScalarKind::ULong))
            }
            None => self.push(BTy::Scalar(BScalar::I64), BOp::ConstInt(0)),
        };
        let stream_val = match stream {
            Some(e) => self.lower_expr(e).0,
            // No stream named: a null-stream sentinel — see `Op::KernelLaunch`'s own doc
            // comment on why this op has no `Option` operands.
            None => self.push(BTy::Ptr(BSpace::Global), BOp::ConstInt(0)),
        };
        let arg_vals: Vec<ValRef> = args.iter().map(|a| self.lower_expr(a).0).collect();

        self.push_void(BOp::KernelLaunch {
            kernel: name.clone(),
            grid: grid_vals,
            block: block_vals,
            shared: shared_val,
            stream: stream_val,
            args: arg_vals,
        });
        (self.zero_of(&void_ty), void_ty)
    }

    /// Lowers one launch-config dimension (`grid`/`block`) to its flattened `(x, y, z)` triple:
    /// `dim3`'s own single-argument implicit constructor for a bare integer (`(v, 1, 1)`, real
    /// CUDA's `kernel<<<256, 256>>>(...)` shape), or three real field loads for an actual
    /// `dim3`-typed value — `peek_launch_dim_is_dim3` tells the two shapes apart without
    /// lowering `e` twice (`checker::Checker::check_launch_config_dim` already validated `e` is
    /// one or the other before lowering ever runs).
    fn lower_launch_config_dim(&mut self, e: &Expr) -> [ValRef; 3] {
        let i32t = BTy::Scalar(BScalar::I32);
        if self.peek_launch_dim_is_dim3(e) {
            let lv = self.lower_lvalue(e);
            if lv.ty.is_unknown() {
                let z = self.push(i32t, BOp::ConstInt(0));
                return [z, z, z];
            }
            let x = self.load_dim3_field(&lv, "x");
            let y = self.load_dim3_field(&lv, "y");
            let z = self.load_dim3_field(&lv, "z");
            [x, y, z]
        } else {
            let (v, t) = self.lower_expr(e);
            let x = if t.is_unknown() {
                self.push(i32t, BOp::ConstInt(0))
            } else {
                self.coerce_to(v, &t, &Ty::Scalar(ScalarKind::UInt))
            };
            let one = self.push(i32t, BOp::ConstInt(1));
            [x, one, one]
        }
    }

    /// Best-effort static check of whether `e` is `dim3`-typed, without lowering it — enough to
    /// tell a launch-config dimension's two legal shapes apart before deciding whether it needs
    /// three field loads or a single value. Deliberately narrow, matching what
    /// `checker::Checker::check_launch_config_dim` actually accepts: only a local variable or
    /// one of the four `dim3`-typed builtins (`checker::CUDA_DIM3_BUILTINS`) can name a `dim3`
    /// value in source today (there is no `dim3(...)` constructor-call syntax modeled by this
    /// pass); anything else is assumed to be the integer shape.
    fn peek_launch_dim_is_dim3(&self, e: &Expr) -> bool {
        let Expr::Ident { name, .. } = e else {
            return false;
        };
        if let Some(slot) = self.find_local(name) {
            return slot.ty == Ty::Struct(CUDA_DIM3_STRUCT.to_string());
        }
        CUDA_DIM3_BUILTINS.contains(&name.as_str())
    }

    /// Loads one `x`/`y`/`z` field out of a `dim3`-typed `LValue`, the same base-address-plus-
    /// byte-offset technique `lower_member_lvalue` uses for ordinary struct member access.
    fn load_dim3_field(&mut self, base: &LValue, field: &str) -> ValRef {
        match self.field_offset(&base.ty, field) {
            Some((off, fty)) => {
                let off_val = self.push(BTy::Scalar(BScalar::I64), BOp::ConstInt(off as i64));
                let addr = self.push(
                    BTy::Ptr(base.space),
                    BOp::Bin(BBin::Add, base.addr, off_val),
                );
                let field_lv = LValue {
                    addr,
                    ty: fty,
                    space: base.space,
                };
                self.load_addr(&field_lv).0
            }
            None => self.push(BTy::Scalar(BScalar::I32), BOp::ConstInt(0)),
        }
    }

    /// `cudaMalloc(devPtr, size)` -> `Op::CudaMalloc` plus a real `Store` of the allocated
    /// pointer through `devPtr`'s own address (see `lower_cuda_malloc_devptr` and
    /// `Op::CudaMalloc`'s own doc comment on why the store, not this instruction's own SSA
    /// value, is the real output). `size` (`size_t`) is coerced to `i64`. The call's own
    /// expression value is a synthesized `cudaSuccess` (`0`) placeholder — this pass has no
    /// runtime failure to report honestly (see the module header), matching every other
    /// builtin here that stands in for a value with no BIR execution semantics attached yet.
    fn lower_cuda_malloc_call(&mut self, args: &[Expr], span: FSpan) -> (ValRef, Ty) {
        let ity = Ty::Scalar(ScalarKind::Int);
        if args.len() != 2 {
            for a in args {
                self.lower_expr(a);
            }
            self.diag_unsupported(span, "cudaMalloc", "expects exactly 2 arguments");
            return self.placeholder();
        }
        let (size_v, size_t) = self.lower_expr(&args[1]);
        let devptr_lv = self.lower_cuda_malloc_devptr(&args[0], span);
        if size_t.is_unknown() || devptr_lv.ty.is_unknown() {
            return self.placeholder();
        }
        let size = self.coerce_to(size_v, &size_t, &Ty::Scalar(ScalarKind::ULong));
        let ptr_ty = BTy::Ptr(BSpace::Global);
        let allocated = self.push(ptr_ty, BOp::CudaMalloc { size });
        self.store_addr(&devptr_lv, allocated);
        (self.zero_of(&ity), ity)
    }

    /// Resolves `cudaMalloc`'s first argument (`devPtr`, real signature `void**`) to the real
    /// `LValue` the allocated pointer must be stored through. The common real shape is
    /// `(void**)&d_a` — a cast wrapping `&lvalue` — unwrapped here so the store keeps the
    /// lvalue's own real address space (`Local`/`Shared`/...) rather than losing it through an
    /// ordinary value-lowering of the whole expression: this project's sema `Ty::Pointer`
    /// carries no address-space annotation of its own (see the module header), so once that
    /// information is gone it cannot be recovered downstream. Anything else (an already-
    /// `void**`-typed plain value, e.g. a parameter) still lowers correctly, defaulting to
    /// `AddrSpace::Global` — the same default every other pointer *value* gets in this pass.
    fn lower_cuda_malloc_devptr(&mut self, e: &Expr, span: FSpan) -> LValue {
        let mut inner = e;
        while let Expr::Cast { expr, .. } = inner {
            inner = expr;
        }
        if let Expr::Unary {
            op: UnaryOp::Addr,
            expr,
            ..
        } = inner
        {
            return self.lower_lvalue(expr);
        }
        let (v, ty) = self.lower_expr(e);
        if !ty.is_pointer_like() {
            if !ty.is_unknown() {
                self.diag_unsupported(span, "cudaMalloc", "first argument is not a pointer");
            }
            return self.lvalue_unknown();
        }
        LValue {
            addr: v,
            ty: Ty::Pointer(Box::new(Ty::Scalar(ScalarKind::Void)), false),
            space: BSpace::Global,
        }
    }

    /// `cudaMemcpy(dst, src, count, kind)` -> `Op::CudaMemcpy`. No real byte-copy semantics are
    /// modeled here (see the module header): the op simply carries the four operands
    /// faithfully, exactly like `cudaFree`/`cudaDeviceSynchronize` below — a real host-side
    /// copy is separate, later work. `kind` accepts either a bare integer literal or one of
    /// `checker::CUDA_MEMCPY_KIND_CONSTANTS`'s named constants; both already lower through the
    /// ordinary `Expr::Ident` enum-constant path `seed_cuda_runtime_api` wires up.
    fn lower_cuda_memcpy_call(&mut self, args: &[Expr]) -> (ValRef, Ty) {
        let vals = self.lower_call_args(args);
        if vals.len() != 4 || vals.iter().any(|(_, t)| t.is_unknown()) {
            return self.placeholder();
        }
        let count = self.coerce_to(vals[2].0, &vals[2].1, &Ty::Scalar(ScalarKind::ULong));
        let kind = self.coerce_to(vals[3].0, &vals[3].1, &Ty::Scalar(ScalarKind::Int));
        self.push_void(BOp::CudaMemcpy {
            dst: vals[0].0,
            src: vals[1].0,
            count,
            kind,
        });
        let ity = Ty::Scalar(ScalarKind::Int);
        (self.zero_of(&ity), ity)
    }

    /// `cudaFree(devPtr)` -> `Op::CudaFree`.
    fn lower_cuda_free_call(&mut self, args: &[Expr]) -> (ValRef, Ty) {
        let vals = self.lower_call_args(args);
        if vals.len() != 1 || vals[0].1.is_unknown() {
            return self.placeholder();
        }
        self.push_void(BOp::CudaFree { ptr: vals[0].0 });
        let ity = Ty::Scalar(ScalarKind::Int);
        (self.zero_of(&ity), ity)
    }

    /// `cudaDeviceSynchronize(void)` -> `Op::CudaDeviceSynchronize`. Any arguments (there
    /// should be none) are still lowered first for their side effects, matching
    /// `lower_syncthreads_call`'s own convention for a zero-arity builtin.
    fn lower_cuda_device_synchronize_call(&mut self, args: &[Expr]) -> (ValRef, Ty) {
        for a in args {
            self.lower_expr(a);
        }
        self.push_void(BOp::CudaDeviceSynchronize);
        let ity = Ty::Scalar(ScalarKind::Int);
        (self.zero_of(&ity), ity)
    }
}

#[cfg(test)]
mod tests {
    use basalt_diag::ECode;
    use basalt_frontend_c::ast::TranslationUnit;
    use basalt_frontend_c::{lex, parse};

    use super::lower;
    use crate::check;

    fn parse_ok(src: &str) -> TranslationUnit {
        let (tokens, lex_errs) = lex(src);
        assert!(lex_errs.is_empty(), "lex errors: {lex_errs:?}");
        let (tu, parse_errs) = parse(&tokens);
        assert!(parse_errs.is_empty(), "parse errors: {parse_errs:?}");
        tu
    }

    fn checked(src: &str) -> TranslationUnit {
        let tu = parse_ok(src);
        let diags = check(&tu);
        assert!(diags.is_empty(), "unexpected sema diagnostics: {diags:?}");
        tu
    }

    fn codes(diags: &[basalt_diag::Diag]) -> Vec<ECode> {
        diags.iter().map(|d| d.code).collect()
    }

    fn assert_roundtrip(m: &basalt_bir::Module) {
        let text = basalt_bir::print(m);
        let reparsed = match basalt_bir::parse(&text) {
            Ok(m) => m,
            Err(e) => panic!("parse(print(m)) failed: {e}\n--- printed BIR ---\n{text}"),
        };
        assert_eq!(
            &reparsed, m,
            "parse(print(m)) != m\n--- printed BIR ---\n{text}"
        );
    }

    #[test]
    fn lowers_trivial_constant_return() {
        let tu = checked("int f() { return 42; }");
        let (m, diags) = lower(&tu);
        assert!(diags.is_empty(), "{diags:?}");
        assert_eq!(m.funcs.len(), 1);
        assert_eq!(m.funcs[0].name, "f");
        assert_eq!(
            m.funcs[0].ret,
            basalt_bir::Ty::Scalar(basalt_bir::Scalar::I32)
        );
        let text = basalt_bir::print(&m);
        assert!(text.contains("const.i i32 42"), "{text}");
        assert_roundtrip(&m);
    }

    #[test]
    fn lowers_locals_arithmetic_and_return() {
        let tu = checked(
            r#"
            int f(int a, int b) {
                int x = a + b;
                int y = x * 2;
                return y;
            }
            "#,
        );
        let (m, diags) = lower(&tu);
        assert!(diags.is_empty(), "{diags:?}");
        let text = basalt_bir::print(&m);
        assert!(text.contains("add i32"), "{text}");
        assert!(text.contains("mul i32"), "{text}");
        assert_roundtrip(&m);
    }

    #[test]
    fn lowers_if_else() {
        let tu = checked(
            r#"
            int f(int a) {
                int r;
                if (a > 0) {
                    r = 1;
                } else {
                    r = -1;
                }
                return r;
            }
            "#,
        );
        let (m, diags) = lower(&tu);
        assert!(diags.is_empty(), "{diags:?}");
        let f = &m.funcs[0];
        assert!(f.blocks.len() >= 4, "{f:?}");
        assert!(f
            .blocks
            .iter()
            .any(|b| matches!(b.term, basalt_bir::Term::CondBr(..))));
        assert_roundtrip(&m);
    }

    #[test]
    fn lowers_while_with_break_and_continue() {
        let tu = checked(
            r#"
            int f(int n) {
                int i = 0;
                int sum = 0;
                while (i < n) {
                    i = i + 1;
                    if (i == 5) {
                        continue;
                    }
                    if (i == 10) {
                        break;
                    }
                    sum = sum + i;
                }
                return sum;
            }
            "#,
        );
        let (m, diags) = lower(&tu);
        assert!(diags.is_empty(), "{diags:?}");
        assert_roundtrip(&m);
    }

    #[test]
    fn lowers_for_loop() {
        let tu = checked(
            r#"
            int f(int n) {
                int sum = 0;
                for (int i = 0; i < n; i = i + 1) {
                    sum = sum + i;
                }
                return sum;
            }
            "#,
        );
        let (m, diags) = lower(&tu);
        assert!(diags.is_empty(), "{diags:?}");
        let text = basalt_bir::print(&m);
        assert!(text.contains("icmp"), "{text}");
    }

    #[test]
    fn lowers_ternary_via_compare_and_phi() {
        let tu = checked(
            r#"
            int f(int a, int b) {
                return a > b ? a : b;
            }
            "#,
        );
        let (m, diags) = lower(&tu);
        assert!(diags.is_empty(), "{diags:?}");
        let text = basalt_bir::print(&m);
        assert!(text.contains("icmp sgt"), "{text}");
        assert!(text.contains("phi"), "{text}");
        assert_roundtrip(&m);
    }

    #[test]
    fn plain_function_call_lowers_to_a_real_call_op() {
        // P13-T-calls-i: a same-module call to a resolvable named function now lowers to a
        // real `Op::Call` instead of unconditionally reporting `E304` — which *call graph
        // shapes* an actual backend accepts (this one is two plain functions, neither
        // `__global__` nor `__device__`) is validated at the backend layer, not here (see the
        // module header's own note on this).
        let tu = checked(
            r#"
            int g(int x) {
                return x;
            }
            int f() {
                return g(1);
            }
            "#,
        );
        let (m, diags) = lower(&tu);
        assert!(diags.is_empty(), "{diags:?}");
        let text = basalt_bir::print(&m);
        assert!(text.contains("call i32 @g ["), "{text}");
        assert_roundtrip(&m);
    }

    #[test]
    fn call_with_aggregate_argument_reports_diag() {
        let tu = checked(
            r#"
            struct Point { int x; int y; };
            int consume(struct Point p) {
                return p.x;
            }
            void f(struct Point p) {
                consume(p);
            }
            "#,
        );
        let (m, diags) = lower(&tu);
        assert!(
            codes(&diags).contains(&ECode::LoweringUnsupported),
            "{diags:?}"
        );
        assert_eq!(m.funcs.len(), 2);
    }

    #[test]
    fn call_with_aggregate_return_reports_diag() {
        let tu = checked(
            r#"
            struct Point { int x; int y; };
            struct Point make(void) {
                struct Point p;
                p.x = 1;
                p.y = 2;
                return p;
            }
            void f(void) {
                make();
            }
            "#,
        );
        let (m, diags) = lower(&tu);
        assert!(
            codes(&diags).contains(&ECode::LoweringUnsupported),
            "{diags:?}"
        );
        assert_eq!(m.funcs.len(), 2);
    }

    #[test]
    fn switch_with_fallthrough_and_break_lowers_to_native_switch() {
        let tu = checked(
            r#"
            int f(int x) {
                int r = 0;
                switch (x) {
                    case 1:
                        r = 1;
                        break;
                    case 2:
                    case 3:
                        r = 2;
                        break;
                    default:
                        r = -1;
                }
                return r;
            }
            "#,
        );
        let (m, diags) = lower(&tu);
        assert!(diags.is_empty(), "{diags:?}");
        let f = &m.funcs[0];
        assert!(
            f.blocks.iter().any(
                |b| matches!(&b.term, basalt_bir::Term::Switch(_, _, cases) if cases.len() == 3)
            ),
            "{f:?}"
        );
        assert_roundtrip(&m);
    }

    #[test]
    fn array_indexing_lowers_via_pointer_arithmetic() {
        let tu = checked(
            r#"
            int f(int *p) {
                int a[4];
                a[0] = 1;
                p[1] = a[0];
                return p[1];
            }
            "#,
        );
        let (m, diags) = lower(&tu);
        assert!(diags.is_empty(), "{diags:?}");
        let text = basalt_bir::print(&m);
        assert!(text.contains("mul i64"), "{text}");
        assert_roundtrip(&m);
    }

    #[test]
    fn struct_member_access_lowers_via_byte_offset() {
        let tu = checked(
            r#"
            struct Point { int x; int y; };
            int f(struct Point p) {
                p.x = 1;
                p.y = 2;
                return p.x + p.y;
            }
            "#,
        );
        let (m, diags) = lower(&tu);
        assert!(diags.is_empty(), "{diags:?}");
        assert_roundtrip(&m);
    }

    #[test]
    fn gpu_index_intrinsics_lower_to_dedicated_ops() {
        let tu = checked(
            r#"
            __global__ void kernel(int *out) {
                out[0] = threadIdx.x + blockIdx.y * blockDim.z + gridDim.x;
            }
            "#,
        );
        let (m, diags) = lower(&tu);
        assert!(diags.is_empty(), "{diags:?}");
        let text = basalt_bir::print(&m);
        assert!(text.contains("tid.x"), "{text}");
        assert!(text.contains("bid.y"), "{text}");
        assert!(text.contains("bdim.z"), "{text}");
        assert!(text.contains("gdim.x"), "{text}");
        assert_roundtrip(&m);
    }

    #[test]
    fn syncthreads_call_lowers_to_barrier_with_no_diagnostic() {
        let tu = checked(
            r#"
            __global__ void kernel() {
                __syncthreads();
            }
            "#,
        );
        let (m, diags) = lower(&tu);
        assert!(diags.is_empty(), "{diags:?}");
        let text = basalt_bir::print(&m);
        assert!(text.contains("barrier"), "{text}");
        assert_roundtrip(&m);
    }

    #[test]
    fn shuffle_builtin_lowers_to_shuffle_op() {
        let tu = checked(
            r#"
            __global__ void kernel(int *out) {
                int v = threadIdx.x;
                out[0] = __shfl_xor(v, 1);
            }
            "#,
        );
        let (m, diags) = lower(&tu);
        assert!(diags.is_empty(), "{diags:?}");
        let text = basalt_bir::print(&m);
        assert!(text.contains("shuffle.xor"), "{text}");
        assert_roundtrip(&m);
    }

    #[test]
    fn vote_builtin_lowers_to_ballot_op() {
        let tu = checked(
            r#"
            __global__ void kernel(int *out) {
                int p = threadIdx.x;
                out[0] = __ballot(p);
            }
            "#,
        );
        let (m, diags) = lower(&tu);
        assert!(diags.is_empty(), "{diags:?}");
        let text = basalt_bir::print(&m);
        assert!(text.contains("ballot"), "{text}");
        assert_roundtrip(&m);
    }

    #[test]
    fn atomic_rmw_builtin_lowers_to_atomic_op() {
        let tu = checked(
            r#"
            __global__ void kernel(int *addr) {
                int old = atomicAdd(addr, 1);
                *addr = old;
            }
            "#,
        );
        let (m, diags) = lower(&tu);
        assert!(diags.is_empty(), "{diags:?}");
        let text = basalt_bir::print(&m);
        assert!(text.contains("atomic.add"), "{text}");
        assert_roundtrip(&m);
    }

    #[test]
    fn atomic_cas_builtin_lowers_to_atomic_cas_op() {
        let tu = checked(
            r#"
            __global__ void kernel(int *addr) {
                int old = atomicCAS(addr, 0, 1);
                *addr = old;
            }
            "#,
        );
        let (m, diags) = lower(&tu);
        assert!(diags.is_empty(), "{diags:?}");
        let text = basalt_bir::print(&m);
        assert!(text.contains("atomic.cas"), "{text}");
        assert_roundtrip(&m);
    }

    /// `assignable` permits an integer literal in a pointer-typed parameter slot (modeling C's
    /// null-constant `0`, see `ty::assignable`), so `atomicAdd(0, 5)` type-checks in the sema
    /// pass with no diagnostic at all. Lowering still cannot make sense of a non-pointer
    /// address: this is the genuine, sema-valid-but-unlowerable case the module header's
    /// `int`-only-builtin simplification does not paper over, and it must degrade to `E304`
    /// rather than emitting a bogus `atomic.add` against a non-address value.
    #[test]
    fn atomic_with_non_pointer_address_reports_e304_without_panicking() {
        let tu = checked(
            r#"
            __global__ void kernel() {
                int r = atomicAdd(0, 5);
            }
            "#,
        );
        let (m, diags) = lower(&tu);
        assert!(
            codes(&diags).contains(&ECode::LoweringUnsupported),
            "{diags:?}"
        );
        assert_eq!(m.funcs.len(), 1);
    }

    // ---- P13-T1b: kernel launch + CUDA Runtime API -----------------------------------------

    #[test]
    fn kernel_launch_with_bare_integer_config_lowers_to_kernel_launch_op() {
        let tu = checked(
            r#"
            __global__ void vadd(float *a, float *b, float *c) {
                c[threadIdx.x] = a[threadIdx.x] + b[threadIdx.x];
            }
            void launch(float *a, float *b, float *c) {
                vadd<<<1, 256>>>(a, b, c);
            }
            "#,
        );
        let (m, diags) = lower(&tu);
        assert!(diags.is_empty(), "{diags:?}");
        let launcher = m.funcs.iter().find(|f| f.name == "launch").unwrap();
        let launch_inst = launcher
            .insts
            .iter()
            .find_map(|inst| match &inst.op {
                basalt_bir::Op::KernelLaunch {
                    kernel,
                    grid,
                    block,
                    args,
                    ..
                } => Some((kernel.clone(), *grid, *block, args.clone())),
                _ => None,
            })
            .expect("expected a KernelLaunch instruction");
        let (kernel, grid, block, args) = launch_inst;
        assert_eq!(kernel, "vadd");
        assert_eq!(args.len(), 3);

        // A bare-integer launch config (`<<<1, 256>>>`) flattens to `(v, 1, 1)` per dim3's own
        // single-argument implicit constructor — check every non-leading component really is
        // the constant `1`, and the leading component is the real grid/block value (not itself
        // hardcoded to 1).
        let const_int_value = |v: basalt_bir::ValRef| -> Option<i64> {
            let basalt_bir::ValRef::Val(id) = v else {
                return None;
            };
            match launcher.insts[id.0 as usize].op {
                basalt_bir::Op::ConstInt(n) => Some(n),
                _ => None,
            }
        };
        assert_eq!(const_int_value(grid[1]), Some(1));
        assert_eq!(const_int_value(grid[2]), Some(1));
        assert_eq!(const_int_value(block[1]), Some(1));
        assert_eq!(const_int_value(block[2]), Some(1));

        assert_roundtrip(&m);
    }

    /// A launch-config dimension that really is `dim3`-typed (rather than a bare integer)
    /// lowers to three real field loads, not the `(v, 1, 1)` flattening. `__basalt_cuda_dim3`
    /// is the synthetic struct name `checker`/`lower` register `dim3` builtin values under
    /// (see `checker::CUDA_DIM3_STRUCT`) — real CUDA-C has no way to spell a `dim3` local today
    /// (see this file's own module header's `Op::KernelLaunch` note), so this test exercises
    /// the mechanism directly against that internal name, the same way the sema layer would see
    /// it if a real `dim3`-typed value reached a launch.
    #[test]
    fn kernel_launch_with_dim3_typed_config_loads_real_fields() {
        let tu = checked(
            r#"
            __global__ void vadd(float *a) {
                a[threadIdx.x] = 0.0f;
            }
            void launch(float *a, __basalt_cuda_dim3 g, __basalt_cuda_dim3 b) {
                vadd<<<g, b>>>(a);
            }
            "#,
        );
        let (m, diags) = lower(&tu);
        assert!(diags.is_empty(), "{diags:?}");
        let text = basalt_bir::print(&m);
        // Three loads per dim3 config argument (x/y/z), six total, rather than any `const.i`
        // standing in for a flattened `1`.
        assert!(
            text.matches("load i32 ptr.param").count() >= 6,
            "expected at least 6 field loads, got:\n{text}"
        );
        assert_roundtrip(&m);
    }

    #[test]
    fn kernel_launch_with_shared_and_stream_lowers_those_operands() {
        let tu = checked(
            r#"
            __global__ void vadd(float *a) {
                a[threadIdx.x] = 0.0f;
            }
            void launch(float *a, int shared_bytes, void *stream) {
                vadd<<<1, 256, shared_bytes, stream>>>(a);
            }
            "#,
        );
        let (m, diags) = lower(&tu);
        assert!(diags.is_empty(), "{diags:?}");
        let text = basalt_bir::print(&m);
        assert!(text.contains("kernel.launch"), "{text}");
        assert_roundtrip(&m);
    }

    /// Omitting `shared`/`stream` still produces concrete operands (a `0`-byte default and a
    /// null-stream sentinel) — `Op::KernelLaunch` has no `Option` fields (see its own doc
    /// comment), so these must always be real, materialized values.
    #[test]
    fn kernel_launch_without_shared_or_stream_still_materializes_both_operands() {
        let tu = checked(
            r#"
            __global__ void vadd(float *a) {
                a[threadIdx.x] = 0.0f;
            }
            void launch(float *a) {
                vadd<<<1, 256>>>(a);
            }
            "#,
        );
        let (m, diags) = lower(&tu);
        assert!(diags.is_empty(), "{diags:?}");
        let launcher = m.funcs.iter().find(|f| f.name == "launch").unwrap();
        let found = launcher.insts.iter().any(|inst| {
            matches!(
                &inst.op,
                basalt_bir::Op::KernelLaunch {
                    shared: basalt_bir::ValRef::Val(_),
                    stream: basalt_bir::ValRef::Val(_),
                    ..
                }
            )
        });
        assert!(found, "expected concrete shared/stream operands");
        assert_roundtrip(&m);
    }

    /// `cudaMalloc`'s real pointer-to-pointer semantics: the allocated pointer is not just
    /// this instruction's own unused SSA result — it must be written through `devPtr`'s own
    /// address with a genuine `Op::Store`.
    #[test]
    fn cuda_malloc_stores_the_allocated_pointer_through_devptr() {
        let tu = checked(
            r#"
            void setup(int n) {
                float *d_a;
                cudaMalloc((void**)&d_a, n * sizeof(float));
            }
            "#,
        );
        let (m, diags) = lower(&tu);
        assert!(diags.is_empty(), "{diags:?}");
        let f = &m.funcs[0];
        let malloc_id = f
            .insts
            .iter()
            .position(|inst| matches!(inst.op, basalt_bir::Op::CudaMalloc { .. }))
            .expect("expected a CudaMalloc instruction");
        let stores_result = f.insts.iter().any(|inst| match &inst.op {
            basalt_bir::Op::Store { val, .. } => {
                *val == basalt_bir::ValRef::Val(basalt_bir::InstId(malloc_id as u32))
            }
            _ => false,
        });
        assert!(
            stores_result,
            "expected a Store writing CudaMalloc's own result through devPtr"
        );
        assert_roundtrip(&m);
    }

    #[test]
    fn cuda_memcpy_named_kind_constant_lowers_to_its_real_integer_value() {
        let tu = checked(
            r#"
            void setup(float *h_a, float *d_a, int n) {
                cudaMemcpy(d_a, h_a, n * sizeof(float), cudaMemcpyHostToDevice);
            }
            "#,
        );
        let (m, diags) = lower(&tu);
        assert!(diags.is_empty(), "{diags:?}");
        let f = &m.funcs[0];
        let kind = f
            .insts
            .iter()
            .find_map(|inst| match &inst.op {
                basalt_bir::Op::CudaMemcpy { kind, .. } => Some(*kind),
                _ => None,
            })
            .expect("expected a CudaMemcpy instruction");
        let basalt_bir::ValRef::Val(id) = kind else {
            panic!("kind operand should be a real instruction value");
        };
        assert_eq!(f.insts[id.0 as usize].op, basalt_bir::Op::ConstInt(1));
        assert_roundtrip(&m);
    }

    #[test]
    fn cuda_free_and_device_synchronize_lower_to_their_own_ops() {
        let tu = checked(
            r#"
            void teardown(float *d_a) {
                cudaDeviceSynchronize();
                cudaFree(d_a);
            }
            "#,
        );
        let (m, diags) = lower(&tu);
        assert!(diags.is_empty(), "{diags:?}");
        let f = &m.funcs[0];
        assert!(f
            .insts
            .iter()
            .any(|i| matches!(i.op, basalt_bir::Op::CudaDeviceSynchronize)));
        assert!(f
            .insts
            .iter()
            .any(|i| matches!(i.op, basalt_bir::Op::CudaFree { .. })));
        assert_roundtrip(&m);
    }
}
