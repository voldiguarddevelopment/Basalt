// Triton-to-BIR lowering: turns a `basalt_frontend_triton::ast::Module` plus the
// `triton_check::KernelShapes` already inferred for it into a `basalt_bir::Module`. Entry
// point: `lower_triton`. This is the third and last stage of the Triton pipeline (parse ->
// `check_triton` -> `lower_triton`), mirroring `lower.rs`'s role for the CUDA-C side, but as
// its own pass over its own AST rather than a shared `Lowerer` — the same "own AST, own sema,
// own lowering" precedent P10-T1/T2 already set for this frontend.
//
// # Scoping correction: `tl.dot` does not lower to BIR `mma`
//
// `TASKS.md`'s literal wording for this task is "`tl.dot` lowers to BIR `mma`". Independently
// verified against the actual backend sources (not taken on faith): `basalt-ptx`'s
// `check_no_mma` (`crates/basalt-ptx/src/emit.rs`) refuses any module containing `Op::Mma`
// outright, and `basalt-x86`'s second (SSA/regalloc) backend refuses it the same way
// (`crates/basalt-x86/src/regalloc.rs`, `check_module`'s `Op::Mma => Err(...)` arm). Only the
// x86-64 *oracle* (`crates/basalt-x86/src/oracle.rs`) actually lowers `Op::Mma`, via its own
// documented triple-loop expansion — so `Op::Mma` is not quite the universal dead end
// `TASKS.md` assumed, but it is still a dead end for this phase's own stated exit criteria:
// Phase 10 requires both `--cpu` *and* `--nvidia-ptx` to accept the same lowered BIR, and BIR
// is meant to be target-independent (one lowering, many backends — ARCHITECTURE.md §1), not
// something this pass special-cases per target. A `tl.dot` that lowered to `Op::Mma` would
// work under `--cpu` and refuse under `--nvidia-ptx`, which satisfies neither exit criterion
// honestly. `tl.dot` therefore always lowers to a real, scalar, runtime triple loop (the same
// `for m { for n { for k { ... } } }` shape `basalt-x86/tests/tiled_sgemm.rs`'s own hand-built
// fixture already uses and this project already trusts), which every hand-rolled backend can
// at least attempt without a fast-path refusal. `PLAN.md` carries this correction alongside
// Phase 10's own status, matching the precedent set by Phase 9's own "Correction" note.
//
// # Core lowering strategy: no tile-valued SSA register
//
// BIR's `Ty` (see `basalt-bir/src/ty.rs`) is `Scalar | Ptr | Vec | Void` — there is no tile
// type and no way to hold a whole rank-1/rank-2 Triton value "in a register". Every
// tile-shaped Triton value is therefore materialized into its own scratch memory location
// (an ordinary local slot, addressed exactly the way `lower.rs` already addresses every CUDA-C
// local: a synthesized `const.i ptr.<space> (slot * SLOT_STRIDE)` — see that module's own
// header for why this is BIR's accepted stand-in for a real `alloca`), filled element-by-
// element by a genuine BIR loop (phi-free: a stack-slot loop counter, `CondBr` back edge,
// body/exit blocks — precisely `lower_for`'s own construction technique, reused here for a
// different kind of loop). This is chosen deliberately over unrolling even when a dimension's
// extent is concretely known small: one code path, and it is exactly the "runtime K-loop"
// `TASKS.md`'s own P10-T4 exit criterion asks for. A `Dim::Const(n)` dimension's loop bound is
// the literal `n`; a `Dim::Symbolic(name)` dimension's bound is that name's own kernel
// parameter, loaded like any other scalar local at the point the loop opens — `constexpr`-ness
// only matters to `triton_check`'s shape *inference*, not to how the value is actually passed
// at the ABI level (it is still just an ordinary scalar argument at runtime). Only a bare
// symbolic name is resolvable this way; a compound symbolic dimension (`triton_ty::Dim`'s own
// rendered text for something like `"BLOCK * 2"`) is refused (`E304`) rather than re-parsed —
// this pass has no expression evaluator over dimension text, only a name lookup.
//
// One departure from `lower.rs`'s own precedent, found while building this: `lower_for`'s
// actual loop-counter representation is a stack-slot local (`load`/`store`), not a real SSA
// `phi` — `phi` in that pass is reserved for `if`/ternary/`&&`/`||` merges only. This pass
// follows the *actual* established idiom (stack-slot counters for every loop, Triton's source-
// level `for k in range(...)` included) rather than the phi-carried induction variable an
// earlier read of that file's header suggested; it is simpler, already proven correct by every
// existing CUDA-C loop test, and needs no merge-block bookkeeping at all.
//
// # Masked `tl.load`/`tl.store`: a real `CondBr`, not a `Select`
//
// A masked load must not execute the underlying `Load` at all when the mask is false — an
// out-of-bounds guard exists precisely so the address is never dereferenced, and a `Select`
// between "loaded value" and "other" would still have performed the (possibly invalid) load
// first. So a masked load/store lowers to a genuine two-way `CondBr` diamond (mask-true block
// performs the `Load`/`Store`; mask-false block does nothing, or evaluates `other`) rather
// than an unconditional access blended with `Select` — the one place this pass spends a merge
// block on a single element instead of just writing straight into memory. A masked load's
// result still has to come out as one SSA value for its caller to use, so the diamond writes
// through a small scratch scalar slot (mirroring every other value this pass carries across a
// block boundary) rather than a `phi` — again matching the project's `lower.rs`-established
// preference for memory over control-flow-merged values whenever a value must survive past a
// branch.
//
// # What this pass assumes about dtype (a real, documented gap)
//
// Triton kernel signatures carry no dtype at all (`triton_check`'s own `Elem::Unknown`
// documents the same limitation for shape inference), so this pass cannot know a pointer
// parameter's pointee width from the AST. Every pointer arithmetic offset is computed
// assuming 4-byte (`f32`/`i32`) elements — the only width this task's two proof kernels
// (masked vector-add, a small `tl.dot` matmul) need. A kernel over `f64`/`i64` data would
// silently get the wrong byte offsets under this assumption; there is no way to detect that
// case from the AST to refuse it instead, which is the one honest weak spot in this pass's
// "no silently-wrong codegen" stance — flagged here rather than hidden. `tl.load` results are
// assumed `f32` (real element data); `tl.arange`/`tl.program_id`/index arithmetic and ordinary
// (non-`constexpr`, non-pointer) kernel parameters are lowered as `i64` throughout, uniformly,
// so index math and pointer-offset math never need an inserted cast.
//
// # Pointer-vs-scalar kernel parameters: a usage heuristic, not a guess
//
// An ordinary kernel parameter's own AST carries no annotation distinguishing `float* x_ptr`
// from `int n`. This pass classifies a parameter as a pointer if its name appears anywhere in
// the pointer-argument subtree of some `tl.load`/`tl.store` call in the kernel body (a
// conservative usage scan, `collect_names`/`ptr_param_names`); every other non-`constexpr`
// parameter lowers as an ordinary `i64` scalar. This is a heuristic, not real type inference,
// but it is a usage fact about the source, not a name-spelling convention (`_ptr` suffix)
// guessed at — a parameter that is never used as a load/store address is never dereferenced
// either, so misclassifying it cannot produce a wrong *result*, only an unused wrong-typed
// argument slot.
//
// # Refusals (`ECode::LoweringUnsupported`, reusing `lower.rs`'s own E304 rather than minting
// a Triton-specific code — its own doc comment is already generic: "AST-to-BIR lowering hit a
// construct it does not yet lower")
//
// Refused outright rather than guessed at: any expression/statement whose shape came back
// `TileTy::Unknown` where a concrete shape was required (an already-diagnosed upstream
// problem, or a construct `triton_check` itself does not model — `tl.dot` nested anywhere
// other than directly as an assignment's right-hand side, an unrecognized callee, a rank > 2
// reshape); a compound (non-bare-name) symbolic tile dimension; an assignment target that
// is not a bare name; `while`; any comparison operator other than the six ordinary relational
// ones (`is`/`is not`/`in`/`not in` have no scalar-arithmetic meaning here).

use std::collections::HashMap;

use basalt_bir::{
    AddrSpace as BSpace, BinOp as BBin, Block as BBlock, BlockId, FCmpPred, Function as BFunction,
    ICmpPred, Inst as BInst, InstId, Module as BModule, Op as BOp, Scalar as BScalar,
    Term as BTerm, Ty as BTy, ValRef,
};
use basalt_diag::{Diag, ECode, Span};
use basalt_frontend_triton::ast::{
    BinOp as TBin, CmpOp, Expr, KernelFn, Keyword, Module as TModule, Stmt, UnaryOp,
};

use crate::triton_check::KernelShapes;
use crate::triton_ty::{broadcast, reshape, Dim, Elem, ReshapeStep, TileTy};

/// Lowers every kernel in `module` to a BIR function, given the shapes `check_triton` already
/// inferred for it (`shapes[i]` must correspond to `module.kernels[i]` — the same
/// correspondence `check_triton` itself returns). Never stops at the first problem: one
/// kernel's diagnostics do not suppress lowering the rest, matching every other pass in this
/// crate's "report many" contract.
pub fn lower_triton(module: &TModule, shapes: &[KernelShapes]) -> (BModule, Vec<Diag>) {
    let mut funcs = Vec::with_capacity(module.kernels.len());
    let mut diags = Vec::new();
    for (k, ks) in module.kernels.iter().zip(shapes.iter()) {
        let mut lw = Lowerer::new(ks);
        let f = lw.lower_kernel(k);
        diags.append(&mut lw.diags);
        funcs.push(f);
    }
    let bmodule = BModule {
        funcs,
        launch_bounds: None,
        shared_mem_bytes: 0,
        target_dtypes: Vec::new(),
    };
    (bmodule, diags)
}

/// Generous per-slot spacing, exactly `lower.rs`'s own `SLOT_STRIDE` (see that module's
/// header): far more than any tile this pass expects to build (a few thousand elements at
/// widest) can plausibly occupy, so no two distinct slots' address ranges ever overlap.
const SLOT_STRIDE: i64 = 1 << 16;

/// Byte offset, within the kernel's designated scratch pointer (see `Storage::Scratch`),
/// where tile scratch space begins — must clear whatever *real* payload that same parameter
/// also carries (the two proof kernels' widest real payload is a few thousand bytes; their own
/// C drivers document the exact arithmetic and size the buffer accordingly).
const SCRATCH_BASE_BYTES: i64 = 16384;

/// Byte stride between two distinct tiles' own scratch regions. Must clear the widest tile
/// either proof kernel materializes (masked `vector_add`'s 1024-element `i64` `offsets` tile,
/// 8192 bytes) with real margin — mirrors `SLOT_STRIDE`'s own role, just for real memory
/// instead of the oracle's synthetic per-constant cells.
const TILE_STRIDE: i64 = 16384;

/// How one bound name's storage is actually addressed.
///
/// A **scalar** (loop counter, ordinary local, spilled parameter) uses the existing
/// `lower.rs`-established synthetic-address idiom: `const.i ptr.<space> (slot * SLOT_STRIDE)`,
/// a single opaque key the oracle backend maps to its own real (but *single-cell*, 8-byte)
/// stack storage.
///
/// A **tile** (rank-1/rank-2) cannot use that same scheme for its *elements* — found the hard
/// way, building this pass's own end-to-end proof: `basalt-x86/src/oracle.rs`'s
/// `Frame::const_addr_disp` allocates exactly one 8-byte cell per distinct `(space,
/// const-value)` key, packed with zero padding against its neighbors (see that struct's own
/// `next_slot` helper). `SLOT_STRIDE`'s large spacing exists only so two *different* scalars
/// never pick the same literal key by coincidence; it does not reserve any real backing bytes
/// for pointer arithmetic to walk across, so treating a slot's synthesized address as the base
/// of a wide, indexable array — exactly what a materialized tile needs — silently overruns
/// into whichever neighboring scalar's cell happens to sit a few bytes further into the real
/// frame. This is a genuine, previously untested gap in the oracle's local-storage model (no
/// existing CUDA-C kernel in this project declares a real multi-element stack-local array wide
/// enough to expose it), not a misunderstanding of an already-proven mechanism — confirmed by
/// reading `Frame::build`'s actual allocation loop, not assumed.
///
/// The fix this pass takes is the same one `tiled_sgemm.rs`'s own hand-built fixture already
/// established for exactly this "BIR has no `alloca`" gap: borrow real, host-allocated,
/// genuinely contiguous memory from one of the kernel's own pointer parameters (the *last*
/// pointer-typed one — see `lower_kernel`) rather than the oracle's synthetic scheme. Every
/// tile gets a fixed, compile-time-assigned byte range within that pointer's target
/// (`SCRATCH_BASE_BYTES + ordinal * TILE_STRIDE`), and real pointer arithmetic on top of that
/// (`+ index * elem_bytes`) is safe because it walks through actual allocated bytes, not an
/// opaque single-cell key. The two end-to-end proof kernels' C drivers size their designated
/// scratch-bearing buffer accordingly (documented in each driver).
#[derive(Clone, Copy)]
enum Storage {
    Synthetic { slot: u32, space: BSpace },
    Scratch { ordinal: u32 },
}

/// One bound name's storage location, its static (checker-inferred) shape, the concrete BIR
/// scalar type its elements are stored as, and — for a rank-1/rank-2 tile — the already-
/// resolved runtime `ValRef` for each axis's extent (computed once, when the tile was created,
/// and reused by every later statement that addresses an element of it).
#[derive(Clone)]
struct VarSlot {
    storage: Storage,
    shape: TileTy,
    elem_bty: BTy,
    dims: Vec<ValRef>,
}

/// One open (not yet closed) tile-materialization loop level, opened by `open_tile_loop` and
/// closed, innermost first, by `close_tile_loop`.
struct LoopLevel {
    cond_bb: BlockId,
    step_bb: BlockId,
    exit_bb: BlockId,
    ctr_slot: u32,
}

struct Lowerer<'a> {
    shapes: &'a KernelShapes,
    diags: Vec<Diag>,
    vars: HashMap<String, VarSlot>,
    next_slot: u32,
    /// Next ordinal for a fresh tile's `Storage::Scratch` region; separate from `next_slot`
    /// (which also counts scalars) so the real scratch buffer only needs `TILE_STRIDE` bytes
    /// per tile actually materialized, not per synthetic-address user overall.
    next_tile_ordinal: u32,
    /// Index (into this function's own BIR params, not `k.params`) of the pointer parameter
    /// tile scratch space is carved out of — the last pointer-typed real parameter; `None`
    /// only for a kernel with no pointer parameter at all, in which case any tile
    /// materialization refuses rather than guessing where to put its scratch data.
    scratch_param_idx: Option<u32>,
    insts: Vec<BInst>,
    blocks: Vec<Option<BBlock>>,
    insts_by_block: HashMap<u32, Vec<InstId>>,
    cur: BlockId,
    /// Parameter slot/index pairs whose initial store is deferred until the entry block is
    /// actually open (params are registered, for name resolution, before any block exists).
    pending_param_stores: Vec<(VarSlot, u32)>,
    /// Same deferral, for a resolved-literal `constexpr` parameter's local slot (never a real
    /// BIR parameter — see `lower_kernel`).
    pending_const_inits: Vec<(VarSlot, i64)>,
    /// `(break target, continue target)` for the innermost enclosing Triton-source `for` loop.
    ctrl_stack: Vec<(BlockId, BlockId)>,
}

impl<'a> Lowerer<'a> {
    fn new(shapes: &'a KernelShapes) -> Lowerer<'a> {
        Lowerer {
            shapes,
            diags: Vec::new(),
            vars: HashMap::new(),
            next_slot: 0,
            next_tile_ordinal: 0,
            scratch_param_idx: None,
            insts: Vec::new(),
            blocks: Vec::new(),
            insts_by_block: HashMap::new(),
            cur: BlockId(0),
            pending_param_stores: Vec::new(),
            pending_const_inits: Vec::new(),
            ctrl_stack: Vec::new(),
        }
    }

    fn diag(&mut self, span: Span, what: impl Into<String>, detail: impl Into<String>) {
        self.diags.push(
            Diag::new(ECode::LoweringUnsupported)
                .with_span(span)
                .with_args([what.into(), detail.into()]),
        );
    }

    // ---- block/instruction arena plumbing (mirrors lower.rs's own linearization need: see
    // that module's header on why a flat creation-order arena and block-print order can
    // diverge once a loop's body itself opens further nested blocks before the loop's own
    // exit block is filled) --------------------------------------------------------------

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

    fn terminate(&mut self, term: BTerm) {
        let ids = self.insts_by_block.remove(&self.cur.0).unwrap_or_default();
        self.blocks[self.cur.0 as usize] = Some(BBlock { insts: ids, term });
    }

    /// Reorders `self.insts` into block-print order and rewrites every cross-reference
    /// accordingly. A private copy of `lower.rs`'s own `linearize_by_block_order`/`remap_*`
    /// (not exported by that module, and this pass's `Op` surface is a small subset of the
    /// CUDA-C lowerer's), kept deliberately small: this pass only ever emits `ConstInt`,
    /// `ConstFloat`, `Bin`, `ICmp`, `FCmp`, `Load`, `Store`, and `BidX`/`BidY`/`BidZ`.
    fn linearize(&mut self) {
        let blocks: Vec<BBlock> = std::mem::take(&mut self.blocks)
            .into_iter()
            .enumerate()
            .map(|(i, b)| {
                b.unwrap_or_else(|| panic!("triton lowering bug: bb{i} never terminated"))
            })
            .collect();
        let old_insts = std::mem::take(&mut self.insts);

        let mut remap = vec![0u32; old_insts.len()];
        let mut order: Vec<InstId> = Vec::with_capacity(old_insts.len());
        for b in &blocks {
            for &id in &b.insts {
                remap[id.0 as usize] = order.len() as u32;
                order.push(id);
            }
        }
        let rv = |v: ValRef, remap: &[u32]| match v {
            ValRef::Param(p) => ValRef::Param(p),
            ValRef::Val(id) => ValRef::Val(InstId(remap[id.0 as usize])),
        };
        let new_insts: Vec<BInst> = order
            .iter()
            .map(|id| {
                let inst = &old_insts[id.0 as usize];
                let op = match inst.op.clone() {
                    BOp::Bin(op, a, b) => BOp::Bin(op, rv(a, &remap), rv(b, &remap)),
                    BOp::ICmp(p, ty, a, b) => BOp::ICmp(p, ty, rv(a, &remap), rv(b, &remap)),
                    BOp::FCmp(p, ty, a, b) => BOp::FCmp(p, ty, rv(a, &remap), rv(b, &remap)),
                    BOp::Load {
                        ptr,
                        space,
                        align,
                        volatile,
                    } => BOp::Load {
                        ptr: rv(ptr, &remap),
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
                        ptr: rv(ptr, &remap),
                        val: rv(val, &remap),
                        ty,
                        space,
                        align,
                        volatile,
                    },
                    other => other,
                };
                BInst { ty: inst.ty, op }
            })
            .collect();
        let new_blocks: Vec<BBlock> = blocks
            .into_iter()
            .map(|b| {
                let insts = b
                    .insts
                    .iter()
                    .map(|id| InstId(remap[id.0 as usize]))
                    .collect();
                let term = match b.term {
                    BTerm::Br(t) => BTerm::Br(t),
                    BTerm::CondBr(c, t1, t2) => BTerm::CondBr(rv(c, &remap), t1, t2),
                    BTerm::Switch(v, d, cases) => BTerm::Switch(rv(v, &remap), d, cases),
                    BTerm::Ret(v) => BTerm::Ret(v.map(|x| rv(x, &remap))),
                };
                BBlock { insts, term }
            })
            .collect();
        self.insts = new_insts;
        self.blocks = new_blocks.into_iter().map(Some).collect();
    }

    // ---- slots ----------------------------------------------------------------------------

    fn new_slot_id(&mut self) -> u32 {
        let s = self.next_slot;
        self.next_slot += 1;
        s
    }

    fn new_tile_ordinal(&mut self) -> u32 {
        let o = self.next_tile_ordinal;
        self.next_tile_ordinal += 1;
        o
    }

    /// The scratch pointer's own value — `ValRef::Param` is a valid SSA reference from any
    /// block in the function (BIR params are not block-scoped), so this never needs its own
    /// load/slot round-trip the way a synthetic-address scalar does.
    fn scratch_base_ptr(&mut self, span: Span) -> ValRef {
        match self.scratch_param_idx {
            Some(idx) => ValRef::Param(idx),
            None => {
                self.diag(
                    span,
                    "tile materialization with no pointer parameter to carve scratch space from",
                    "this kernel has no pointer-typed parameter at all",
                );
                self.const_i64(0)
            }
        }
    }

    /// A scalar's own synthesized address (`Storage::Synthetic`): see that variant's own doc
    /// for why this is only ever safe for a single 8-byte value, never an indexed array.
    fn synthetic_addr(&mut self, slot: u32, space: BSpace) -> ValRef {
        self.push(
            BTy::Ptr(space),
            BOp::ConstInt(i64::from(slot) * SLOT_STRIDE),
        )
    }

    fn load_scalar(&mut self, v: &VarSlot) -> ValRef {
        let Storage::Synthetic { slot, space } = v.storage else {
            unreachable!("load_scalar is only ever called with Storage::Synthetic");
        };
        let addr = self.synthetic_addr(slot, space);
        self.push(
            v.elem_bty,
            BOp::Load {
                ptr: addr,
                space,
                align: bty_align(v.elem_bty),
                volatile: false,
            },
        )
    }

    fn store_scalar(&mut self, v: &VarSlot, val: ValRef) {
        let Storage::Synthetic { slot, space } = v.storage else {
            unreachable!("store_scalar is only ever called with Storage::Synthetic");
        };
        let addr = self.synthetic_addr(slot, space);
        self.push_void(BOp::Store {
            ptr: addr,
            val,
            ty: v.elem_bty,
            space,
            align: bty_align(v.elem_bty),
            volatile: false,
        });
    }

    /// The real address of tile `v`'s element at `idx` (one `ValRef` per axis, i64, row-major:
    /// the last axis is contiguous), carved out of the kernel's scratch pointer parameter —
    /// see `Storage::Scratch`'s own doc for why this, and not a synthesized local address, is
    /// what a tile's elements must live in. `idx.len()` must equal `v.dims.len()`.
    fn tile_elem_addr(&mut self, v: &VarSlot, idx: &[ValRef], span: Span) -> ValRef {
        let Storage::Scratch { ordinal } = v.storage else {
            unreachable!("tile_elem_addr is only ever called with Storage::Scratch");
        };
        let elem_bytes = bty_bytes(v.elem_bty);
        let base = self.scratch_base_ptr(span);
        let tile_off = self.const_i64(SCRATCH_BASE_BYTES + i64::from(ordinal) * TILE_STRIDE);
        let base = self.push(
            BTy::Ptr(BSpace::Global),
            BOp::Bin(BBin::Add, base, tile_off),
        );
        let lin = match idx.len() {
            1 => idx[0],
            2 => {
                let row_stride = v.dims[1];
                let scaled = self.push(I64, BOp::Bin(BBin::Mul, idx[0], row_stride));
                self.push(I64, BOp::Bin(BBin::Add, scaled, idx[1]))
            }
            n => unreachable!("tile rank is always 1 or 2, got {n}"),
        };
        let elem_bytes_val = self.const_i64(elem_bytes);
        let byte_off = self.push(I64, BOp::Bin(BBin::Mul, lin, elem_bytes_val));
        self.push(
            BTy::Ptr(BSpace::Global),
            BOp::Bin(BBin::Add, base, byte_off),
        )
    }

    fn const_i64(&mut self, v: i64) -> ValRef {
        self.push(I64, BOp::ConstInt(v))
    }

    fn zero_of(&mut self, ty: BTy) -> ValRef {
        match ty {
            BTy::Scalar(BScalar::F32) | BTy::Scalar(BScalar::F64) => {
                self.push(ty, BOp::ConstFloat(0.0))
            }
            _ => self.push(ty, BOp::ConstInt(0)),
        }
    }

    // ---- kernel entry -----------------------------------------------------------------------

    fn lower_kernel(&mut self, k: &KernelFn) -> BFunction {
        let ptr_names = ptr_param_names(k);

        let mut bir_params = Vec::with_capacity(k.params.len());
        let mut param_idx: u32 = 0;
        for p in &k.params {
            // A `constexpr` parameter that resolves to a literal default is, in real Triton,
            // never a runtime argument at all — it is specialized away before codegen. This
            // pass has no launch site to specialize from, but a literal default is the one
            // case it can still fully resolve on its own (mirrors `triton_check`'s own
            // `seed_param`); binding it to a genuine BIR function parameter anyway would waste
            // an integer-class ABI slot the oracle's calling convention caps at six (SysV's
            // `rdi..r9`, `nthreads` included) for no benefit, since the value is already known.
            // A symbolic `constexpr` (no literal default) has no such option: this pass's own
            // scoping decision (see the module header) is to pass it as an ordinary runtime
            // scalar, since that is the only way it can ever get a value at all.
            if p.is_constexpr {
                if let Some(lit) = p.default.as_ref().and_then(expr_as_i64) {
                    let slot = self.new_slot_id();
                    let vs = VarSlot {
                        storage: Storage::Synthetic {
                            slot,
                            space: BSpace::Local,
                        },
                        shape: TileTy::Scalar(Elem::Int),
                        elem_bty: I64,
                        dims: Vec::new(),
                    };
                    self.vars.insert(p.name.clone(), vs.clone());
                    self.pending_const_inits.push((vs, lit));
                    continue;
                }
            }

            let (bty, space_shape) = if p.is_constexpr {
                (BTy::Scalar(BScalar::I64), TileTy::Scalar(Elem::Int))
            } else if ptr_names.contains(p.name.as_str()) {
                (BTy::Ptr(BSpace::Global), TileTy::Scalar(Elem::Unknown))
            } else {
                (BTy::Scalar(BScalar::I64), TileTy::Scalar(Elem::Unknown))
            };
            bir_params.push(bty);
            if matches!(bty, BTy::Ptr(_)) {
                // The *last* pointer-typed real parameter wins if there is more than one —
                // see `Storage::Scratch`'s own doc for why any tile materialized by this
                // kernel borrows its scratch space from this one parameter.
                self.scratch_param_idx = Some(param_idx);
            }
            let slot = self.new_slot_id();
            let vs = VarSlot {
                storage: Storage::Synthetic {
                    slot,
                    space: BSpace::Param,
                },
                shape: space_shape,
                elem_bty: bty,
                dims: Vec::new(),
            };
            self.vars.insert(p.name.clone(), vs.clone());
            // Entry block does not exist yet the first time through; opened just below. Defer
            // the actual store until after `entry` is current.
            self.pending_param_stores.push((vs, param_idx));
            param_idx += 1;
        }

        let entry = self.alloc_block();
        self.cur = entry;
        for (vs, lit) in std::mem::take(&mut self.pending_const_inits) {
            let v = self.const_i64(lit);
            self.store_scalar(&vs, v);
        }
        for (vs, i) in std::mem::take(&mut self.pending_param_stores) {
            self.store_scalar(&vs, ValRef::Param(i));
        }

        for s in &k.body {
            self.lower_stmt(s);
        }
        self.terminate(BTerm::Ret(None));

        self.linearize();
        let (blocks, insts) = (
            std::mem::take(&mut self.blocks),
            std::mem::take(&mut self.insts),
        );
        BFunction {
            name: k.name.clone(),
            is_kernel: true,
            params: bir_params,
            ret: BTy::Void,
            blocks: blocks.into_iter().map(|b| b.expect("linearized")).collect(),
            insts,
        }
    }
}

// ---- tile-materialization loops (see the module header: always a real runtime loop, never
// unrolled, mirroring lower_for's own cond/body/step/exit shape) ---------------------------

impl<'a> Lowerer<'a> {
    /// `check_triton`'s own `expr_types` map, unmodified — correct for every construct that
    /// pass itself resolves concretely, but it deliberately gives `tl.dot`'s own call a bare
    /// `TileTy::Unknown` result (see that pass's module header: "Real matmul-shape inference
    /// ... is P10-T3's job"). Used only as `shape_of`'s leaf-level fallback, never directly by
    /// a caller that might be looking at a name a `tl.dot` assignment rebound.
    fn checker_shape(&self, e: &Expr) -> TileTy {
        self.shapes
            .expr_types
            .get(&e.span())
            .cloned()
            .unwrap_or(TileTy::Unknown)
    }

    /// This pass's own shape query — a real, if narrow, difference from just reading
    /// `check_triton`'s output back (found while building the matmul proof kernel: the
    /// checker's `Unknown` for a `tl.dot` call's own result also poisons every expression
    /// built from it afterward, e.g. `acc = tl.dot(a, b, acc)` followed by `acc = acc + c` —
    /// `broadcast`'s "either side unknown -> unknown" rule means the checker's own tracked
    /// type for `acc` stays `Unknown` from that point on, even though this pass knows exactly
    /// what shape it lowered `tl.dot`'s result to). A bare name is resolved against this
    /// pass's *own* live `vars` table (authoritative for what a name is bound to right now,
    /// including a `tl.dot` result this pass derived independently — see `lower_dot_stmt`)
    /// rather than the checker's possibly-stale record; every other node recomputes its shape
    /// compositionally, mirroring `triton_check`'s own broadcast/reshape rules, falling back to
    /// `checker_shape` only for the handful of leaf forms (`tl.arange`, `tl.zeros`) that pass
    /// already resolves concretely and this pass has no reason to redo.
    fn shape_of(&self, e: &Expr) -> TileTy {
        match e {
            Expr::Name { name, .. } => self
                .vars
                .get(name)
                .map(|v| v.shape.clone())
                .unwrap_or(TileTy::Unknown),
            Expr::IntLit { .. } => TileTy::Scalar(Elem::Int),
            Expr::FloatLit { .. } => TileTy::Scalar(Elem::Float),
            Expr::BoolLit { .. } => TileTy::Scalar(Elem::Bool),
            Expr::UnaryOp { operand, .. } => self.shape_of(operand),
            Expr::BinOp { lhs, rhs, .. } => {
                broadcast(&self.shape_of(lhs), &self.shape_of(rhs)).unwrap_or(TileTy::Unknown)
            }
            Expr::Compare {
                left, comparators, ..
            } => {
                let mut acc = self.shape_of(left);
                for c in comparators {
                    acc = broadcast(&acc, &self.shape_of(c)).unwrap_or(TileTy::Unknown);
                }
                acc
            }
            Expr::Ternary { body, orelse, .. } => {
                broadcast(&self.shape_of(body), &self.shape_of(orelse)).unwrap_or(TileTy::Unknown)
            }
            Expr::Subscript { value, index, .. } => match reshape_steps(index) {
                Some(steps) => reshape(&self.shape_of(value), &steps).unwrap_or(TileTy::Unknown),
                None => TileTy::Unknown,
            },
            Expr::Call { func, args, .. } => match attr_name(func) {
                Some("load") => args
                    .first()
                    .map(|p| self.shape_of(p))
                    .unwrap_or(TileTy::Unknown),
                Some("program_id") => TileTy::Scalar(Elem::Int),
                // `tl.arange`/`tl.zeros` are leaf tile constructors the checker already
                // resolves concretely from literal/constexpr bounds; nothing to recompute.
                Some("arange") | Some("zeros") => self.checker_shape(e),
                // Never queried this way in practice (`tl.dot` is only ever handled directly
                // as an assignment's right-hand side, before `shape_of` would be called on
                // the call expression itself); `Unknown` here is inert, not a guess.
                _ => TileTy::Unknown,
            },
            _ => self.checker_shape(e),
        }
    }

    fn resolve_dim(&mut self, d: &Dim, span: Span) -> ValRef {
        match d {
            Dim::Const(n) => self.const_i64(*n),
            Dim::Symbolic(name) => {
                if let Some(v) = self.vars.get(name).cloned() {
                    if matches!(v.shape, TileTy::Scalar(_)) {
                        return self.load_scalar(&v);
                    }
                }
                self.diag(
                    span,
                    format!("symbolic tile dimension '{name}'"),
                    "not a bare constexpr parameter name this pass can resolve to a runtime value (a compound dimension expression is not re-evaluated)",
                );
                self.const_i64(1)
            }
        }
    }

    /// Opens one nested runtime loop per entry of `dims` (outermost first), leaving `self.cur`
    /// pointing at the innermost body block. Returns each level's loaded induction variable,
    /// each level's resolved runtime bound (kept for row-major addressing of a rank-2 tile,
    /// where axis 1's bound is the row stride), and the loop-context handles `close_tile_loop`
    /// needs to close them again, innermost first.
    fn open_tile_loop(
        &mut self,
        dims: &[Dim],
        span: Span,
    ) -> (Vec<ValRef>, Vec<ValRef>, Vec<LoopLevel>) {
        let mut idx = Vec::with_capacity(dims.len());
        let mut bounds = Vec::with_capacity(dims.len());
        let mut levels = Vec::with_capacity(dims.len());
        for d in dims {
            let bound = self.resolve_dim(d, span);
            bounds.push(bound);

            let ctr_slot = self.new_slot_id();
            let ctr_vs = VarSlot {
                storage: Storage::Synthetic {
                    slot: ctr_slot,
                    space: BSpace::Local,
                },
                shape: TileTy::Scalar(Elem::Int),
                elem_bty: I64,
                dims: Vec::new(),
            };
            let zero = self.const_i64(0);
            self.store_scalar(&ctr_vs, zero);

            let cond_bb = self.alloc_block();
            let body_bb = self.alloc_block();
            let step_bb = self.alloc_block();
            let exit_bb = self.alloc_block();
            self.terminate(BTerm::Br(cond_bb));

            self.cur = cond_bb;
            let iv = self.load_scalar(&ctr_vs);
            let cmp = self.push(
                BTy::Scalar(BScalar::I1),
                BOp::ICmp(ICmpPred::Slt, I64, iv, bound),
            );
            self.terminate(BTerm::CondBr(cmp, body_bb, exit_bb));

            self.cur = body_bb;
            idx.push(iv);
            levels.push(LoopLevel {
                cond_bb,
                step_bb,
                exit_bb,
                ctr_slot,
            });
        }
        (idx, bounds, levels)
    }

    /// Closes loop levels innermost-first: `self.cur` must be the innermost body block (with
    /// its per-element work already appended) when this is called.
    fn close_tile_loop(&mut self, levels: Vec<LoopLevel>) {
        for lvl in levels.into_iter().rev() {
            self.terminate(BTerm::Br(lvl.step_bb));
            self.cur = lvl.step_bb;
            let ctr_vs = VarSlot {
                storage: Storage::Synthetic {
                    slot: lvl.ctr_slot,
                    space: BSpace::Local,
                },
                shape: TileTy::Scalar(Elem::Int),
                elem_bty: I64,
                dims: Vec::new(),
            };
            let iv = self.load_scalar(&ctr_vs);
            let one = self.const_i64(1);
            let next = self.push(I64, BOp::Bin(BBin::Add, iv, one));
            self.store_scalar(&ctr_vs, next);
            self.terminate(BTerm::Br(lvl.cond_bb));
            self.cur = lvl.exit_bb;
        }
    }

    fn get_or_new_scalar_var(&mut self, name: &str, elem_bty: BTy) -> VarSlot {
        if let Some(v) = self.vars.get(name) {
            if matches!(v.shape, TileTy::Scalar(_)) {
                let mut v2 = v.clone();
                v2.elem_bty = elem_bty;
                self.vars.insert(name.to_string(), v2.clone());
                return v2;
            }
        }
        let slot = self.new_slot_id();
        let vs = VarSlot {
            storage: Storage::Synthetic {
                slot,
                space: BSpace::Local,
            },
            shape: TileTy::Scalar(Elem::Unknown),
            elem_bty,
            dims: Vec::new(),
        };
        self.vars.insert(name.to_string(), vs.clone());
        vs
    }

    /// Binds `name` to a tile of `shape`, reusing its existing slot (in place, so a loop-
    /// carried rebind like `acc = tl.dot(a, b, acc)` mutates the same memory rather than
    /// leaking a fresh slot per iteration) whenever a same-rank binding for that name already
    /// exists; otherwise allocates a fresh one.
    fn get_or_new_tile_var(
        &mut self,
        name: &str,
        shape: &TileTy,
        elem_bty: BTy,
        resolved_dims: Vec<ValRef>,
    ) -> VarSlot {
        let rank = shape.rank();
        if let Some(v) = self.vars.get(name) {
            if v.shape.rank() == rank {
                let mut v2 = v.clone();
                v2.shape = shape.clone();
                v2.elem_bty = elem_bty;
                v2.dims = resolved_dims;
                self.vars.insert(name.to_string(), v2.clone());
                return v2;
            }
        }
        let ordinal = self.new_tile_ordinal();
        let vs = VarSlot {
            storage: Storage::Scratch { ordinal },
            shape: shape.clone(),
            elem_bty,
            dims: resolved_dims,
        };
        self.vars.insert(name.to_string(), vs.clone());
        vs
    }
}

fn tile_dims(shape: &TileTy) -> Vec<Dim> {
    match shape {
        TileTy::Rank1(d) => vec![d.clone()],
        TileTy::Rank2(d0, d1) => vec![d0.clone(), d1.clone()],
        _ => Vec::new(),
    }
}

const I64: BTy = BTy::Scalar(BScalar::I64);

/// The pointee element width this pass assumes for every Triton pointer parameter (see the
/// module header's dtype-assumption note): `f32`/`i32`, the only width this task's proof
/// kernels need.
const DATA_ELEM_BYTES: i64 = 4;

impl<'a> Lowerer<'a> {
    fn scale_by_elem_bytes(&mut self, v: ValRef) -> ValRef {
        let bytes = self.const_i64(DATA_ELEM_BYTES);
        self.push(I64, BOp::Bin(BBin::Mul, v, bytes))
    }
}

fn bty_bytes(t: BTy) -> i64 {
    match t {
        BTy::Scalar(BScalar::I1 | BScalar::I8) => 1,
        BTy::Scalar(BScalar::I16 | BScalar::F16) => 2,
        BTy::Scalar(BScalar::I32 | BScalar::F32) => 4,
        BTy::Scalar(BScalar::I64 | BScalar::F64) => 8,
        BTy::Ptr(_) => 8,
        _ => 8,
    }
}

fn bty_align(t: BTy) -> u32 {
    bty_bytes(t) as u32
}

// ---- statements -----------------------------------------------------------------------------

impl<'a> Lowerer<'a> {
    fn lower_stmts(&mut self, stmts: &[Stmt]) {
        for s in stmts {
            self.lower_stmt(s);
        }
    }

    fn lower_stmt(&mut self, s: &Stmt) {
        match s {
            Stmt::Expr { value, span } => self.lower_expr_stmt(value, *span),
            Stmt::Assign {
                targets,
                value,
                span,
            } => self.lower_assign(targets, value, *span),
            Stmt::AugAssign {
                target,
                op,
                value,
                span,
            } => self.lower_augassign(target, *op, value, *span),
            Stmt::AnnAssign {
                target,
                value,
                span,
                ..
            } => self.lower_ann_assign(target, value.as_ref(), *span),
            Stmt::If {
                test,
                body,
                orelse,
                span,
            } => self.lower_if(test, body, orelse, *span),
            Stmt::For {
                target,
                iter,
                body,
                orelse,
                span,
            } => self.lower_for(target, iter, body, orelse, *span),
            Stmt::While { span, .. } => {
                self.diag(*span, "'while' loop", "not supported by Triton lowering");
            }
            Stmt::Return { .. } => {
                // A kernel's own return value is never meaningful at runtime (Triton kernels
                // return nothing), but the control-flow effect of `return` itself is real: it
                // must actually end the current block, or code textually following it would
                // wrongly still execute.
                self.terminate(BTerm::Ret(None));
                self.cur = self.alloc_block();
            }
            // Debug-only in real Triton (compiled out under normal specialization); skipping it
            // never changes a kernel's numeric result, so this is a deliberate no-op rather than
            // a refusal.
            Stmt::Assert { .. } => {}
            Stmt::Pass { .. } => {}
            Stmt::Break { span } => {
                match self.ctrl_stack.last().copied() {
                    Some((break_bb, _)) => self.terminate(BTerm::Br(break_bb)),
                    None => {
                        self.diag(*span, "'break' outside of a loop", "not supported");
                        self.terminate(BTerm::Ret(None));
                    }
                }
                self.cur = self.alloc_block();
            }
            Stmt::Continue { span } => {
                match self.ctrl_stack.last().copied() {
                    Some((_, cont_bb)) => self.terminate(BTerm::Br(cont_bb)),
                    None => {
                        self.diag(*span, "'continue' outside of a loop", "not supported");
                        self.terminate(BTerm::Ret(None));
                    }
                }
                self.cur = self.alloc_block();
            }
            // Already diagnosed by the parser; nothing further to do.
            Stmt::Error { .. } => {}
        }
    }

    fn lower_expr_stmt(&mut self, value: &Expr, span: Span) {
        if let Expr::Call {
            func,
            args,
            keywords,
            span: call_span,
        } = value
        {
            if attr_name(func) == Some("store") {
                self.lower_store_call(args, keywords, *call_span);
                return;
            }
        }
        match self.shape_of(value) {
            TileTy::Scalar(_) => {
                self.eval_expr(value, &[]);
            }
            _ => self.diag(
                span,
                "bare tile-shaped expression statement",
                "not supported outside of 'tl.store' or an assignment",
            ),
        }
    }

    fn assign_target_names<'e>(&mut self, targets: &'e [Expr], span: Span) -> Option<Vec<&'e str>> {
        let mut names = Vec::with_capacity(targets.len());
        for t in targets {
            match t {
                Expr::Name { name, .. } => names.push(name.as_str()),
                _ => {
                    self.diag(
                        span,
                        "assignment target other than a bare name",
                        "tuple/attribute/subscript assignment targets are not lowered",
                    );
                    return None;
                }
            }
        }
        Some(names)
    }

    fn lower_assign(&mut self, targets: &[Expr], value: &Expr, span: Span) {
        let Some(names) = self.assign_target_names(targets, span) else {
            return;
        };
        if names.is_empty() {
            return;
        }

        if let Expr::Call {
            func,
            args,
            keywords,
            span: call_span,
        } = value
        {
            if attr_name(func) == Some("dot") {
                if names.len() != 1 {
                    self.diag(
                        span,
                        "'tl.dot' assigned to more than one target",
                        "not supported",
                    );
                    return;
                }
                self.lower_dot_stmt(names[0], args, keywords, *call_span);
                return;
            }
        }

        match self.shape_of(value) {
            TileTy::Scalar(_) => {
                let elem_bty = self.elem_bty_for(value);
                let v = self.eval_expr(value, &[]);
                for n in &names {
                    let vs = self.get_or_new_scalar_var(n, elem_bty);
                    self.store_scalar(&vs, v);
                }
            }
            shape @ (TileTy::Rank1(_) | TileTy::Rank2(..)) => {
                for n in &names {
                    self.lower_tile_materialize(n, value, &shape, span);
                }
            }
            TileTy::Unknown => self.diag(
                span,
                "assignment with unresolved tile shape",
                "right-hand side shape could not be determined (see earlier diagnostics)",
            ),
        }
    }

    /// The generic tile-assignment path: open a loop over `shape`'s own dims, evaluate `value`
    /// once per element, write it into `name`'s (fresh or reused) scratch slot.
    fn lower_tile_materialize(&mut self, name: &str, value: &Expr, shape: &TileTy, span: Span) {
        let dims_static = tile_dims(shape);
        let elem_bty = self.elem_bty_for(value);
        let (idx, bounds, levels) = self.open_tile_loop(&dims_static, span);
        let vs = self.get_or_new_tile_var(name, shape, elem_bty, bounds);
        let val = self.eval_expr(value, &idx);
        let addr = self.tile_elem_addr(&vs, &idx, span);
        self.push_void(BOp::Store {
            ptr: addr,
            val,
            ty: elem_bty,
            space: BSpace::Global,
            align: bty_align(elem_bty),
            volatile: false,
        });
        self.close_tile_loop(levels);
    }

    /// `name = tl.dot(a, b[, acc])`: `D[m,n] = sum_k A[m,k]*B[k,n] + (acc[m,n] or 0)`, a real
    /// triple-nested runtime loop (see the module header's `Op::Mma` scoping correction) —
    /// the outer two levels reuse `open_tile_loop`/`close_tile_loop` exactly like any other
    /// rank-2 materialization; the inner reduction over `k` opens a third, nested nested nested
    /// loop the same way, accumulating into a small scalar temp rather than the destination
    /// tile itself (so a self-referential `acc` argument sharing `name`'s own slot is read
    /// once, before anything is written back).
    fn lower_dot_stmt(&mut self, name: &str, args: &[Expr], _keywords: &[Keyword], span: Span) {
        if args.len() < 2 {
            self.diag(
                span,
                "'tl.dot' called with fewer than two arguments",
                "expected (a, b) or (a, b, acc)",
            );
            return;
        }
        let a_expr = &args[0];
        let b_expr = &args[1];
        let acc_expr = args.get(2);

        let (m, k1) = match self.shape_of(a_expr) {
            TileTy::Rank2(m, k) => (m, k),
            _ => {
                self.diag(span, "'tl.dot' first argument", "must be a rank-2 tile");
                return;
            }
        };
        let (k2, n) = match self.shape_of(b_expr) {
            TileTy::Rank2(k, n) => (k, n),
            _ => {
                self.diag(span, "'tl.dot' second argument", "must be a rank-2 tile");
                return;
            }
        };
        if k1 != k2 {
            self.diag(
                span,
                "'tl.dot' inner dimension mismatch",
                format!("{k1} vs {k2}"),
            );
            return;
        }

        let elem_bty = BTy::Scalar(BScalar::F32);
        let out_shape = TileTy::Rank2(m.clone(), n.clone());
        let (idx_mn, bounds_mn, levels_mn) = self.open_tile_loop(&tile_dims(&out_shape), span);
        let dest = self.get_or_new_tile_var(name, &out_shape, elem_bty, bounds_mn);

        let init = match acc_expr {
            Some(e) => self.eval_expr(e, &idx_mn),
            None => self.zero_of(elem_bty),
        };
        let sum_slot = self.new_slot_id();
        let sum_vs = VarSlot {
            storage: Storage::Synthetic {
                slot: sum_slot,
                space: BSpace::Local,
            },
            shape: TileTy::Scalar(Elem::Float),
            elem_bty,
            dims: Vec::new(),
        };
        self.store_scalar(&sum_vs, init);

        let (idx_k, _bounds_k, levels_k) = self.open_tile_loop(std::slice::from_ref(&k1), span);
        let a_val = self.eval_expr(a_expr, &[idx_mn[0], idx_k[0]]);
        let b_val = self.eval_expr(b_expr, &[idx_k[0], idx_mn[1]]);
        let prod = self.push(elem_bty, BOp::Bin(BBin::FMul, a_val, b_val));
        let cur_sum = self.load_scalar(&sum_vs);
        let new_sum = self.push(elem_bty, BOp::Bin(BBin::FAdd, cur_sum, prod));
        self.store_scalar(&sum_vs, new_sum);
        self.close_tile_loop(levels_k);

        let final_sum = self.load_scalar(&sum_vs);
        let addr = self.tile_elem_addr(&dest, &idx_mn, span);
        self.push_void(BOp::Store {
            ptr: addr,
            val: final_sum,
            ty: elem_bty,
            space: BSpace::Global,
            align: bty_align(elem_bty),
            volatile: false,
        });
        self.close_tile_loop(levels_mn);
    }

    fn lower_store_call(&mut self, args: &[Expr], keywords: &[Keyword], span: Span) {
        if args.len() < 2 {
            self.diag(
                span,
                "'tl.store' called with fewer than a pointer and a value argument",
                "cannot lower",
            );
            return;
        }
        let ptr_expr = &args[0];
        let val_expr = &args[1];
        let mask_kw = keywords
            .iter()
            .find(|k| k.name.as_deref() == Some("mask"))
            .cloned();
        let dims_static = match self.shape_of(val_expr) {
            TileTy::Scalar(_) => Vec::new(),
            TileTy::Rank1(d) => vec![d],
            TileTy::Rank2(d0, d1) => vec![d0, d1],
            TileTy::Unknown => {
                self.diag(
                    span,
                    "'tl.store' value with unresolved tile shape",
                    "cannot lower",
                );
                return;
            }
        };
        let elem_bty = self.elem_bty_for(val_expr);

        let (idx, levels) = if dims_static.is_empty() {
            (Vec::new(), Vec::new())
        } else {
            let (idx, _bounds, levels) = self.open_tile_loop(&dims_static, span);
            (idx, levels)
        };

        let addr = self.eval_expr(ptr_expr, &idx);
        let val = self.eval_expr(val_expr, &idx);
        match mask_kw {
            None => self.push_void(BOp::Store {
                ptr: addr,
                val,
                ty: elem_bty,
                space: BSpace::Global,
                align: bty_align(elem_bty),
                volatile: false,
            }),
            Some(mkw) => {
                let mask_val = self.eval_expr(&mkw.value, &idx);
                let store_bb = self.alloc_block();
                let skip_bb = self.alloc_block();
                let merge_bb = self.alloc_block();
                self.terminate(BTerm::CondBr(mask_val, store_bb, skip_bb));
                self.cur = store_bb;
                self.push_void(BOp::Store {
                    ptr: addr,
                    val,
                    ty: elem_bty,
                    space: BSpace::Global,
                    align: bty_align(elem_bty),
                    volatile: false,
                });
                self.terminate(BTerm::Br(merge_bb));
                self.cur = skip_bb;
                self.terminate(BTerm::Br(merge_bb));
                self.cur = merge_bb;
            }
        }
        if !levels.is_empty() {
            self.close_tile_loop(levels);
        }
    }

    fn lower_ann_assign(&mut self, target: &Expr, value: Option<&Expr>, span: Span) {
        let Expr::Name { name, .. } = target else {
            self.diag(
                span,
                "annotated assignment to a non-name target",
                "not supported",
            );
            return;
        };
        if let Some(v) = value {
            let elem_bty = self.elem_bty_for(v);
            let val = self.eval_expr(v, &[]);
            let vs = self.get_or_new_scalar_var(name, elem_bty);
            self.store_scalar(&vs, val);
        }
    }

    fn lower_augassign(&mut self, target: &Expr, op: TBin, value: &Expr, span: Span) {
        let Expr::Name { name, .. } = target else {
            self.diag(
                span,
                "augmented assignment to a non-name target",
                "not supported",
            );
            return;
        };
        if !matches!(self.shape_of(target), TileTy::Scalar(_)) {
            self.diag(
                span,
                "augmented assignment on a tile-ranked value",
                "only scalar accumulators are supported",
            );
            return;
        }
        let Some(vs) = self.vars.get(name).cloned() else {
            self.diag(
                span,
                "augmented assignment to an undefined name",
                name.clone(),
            );
            return;
        };
        let cur = self.load_scalar(&vs);
        let rhs = self.eval_expr(value, &[]);
        let result = self.apply_binop(op, cur, rhs, vs.elem_bty, span);
        self.store_scalar(&vs, result);
    }

    fn lower_if(&mut self, test: &Expr, body: &[Stmt], orelse: &[Stmt], span: Span) {
        if !matches!(self.shape_of(test), TileTy::Scalar(_)) {
            self.diag(
                span,
                "'if' on a tile-ranked condition",
                "not supported (Triton masking is expressed via tl.load/tl.store's mask=, not a Python 'if')",
            );
            return;
        }
        let cond = self.eval_expr(test, &[]);
        let then_bb = self.alloc_block();
        let else_bb = self.alloc_block();
        let merge_bb = self.alloc_block();
        self.terminate(BTerm::CondBr(cond, then_bb, else_bb));

        self.cur = then_bb;
        self.lower_stmts(body);
        self.terminate(BTerm::Br(merge_bb));

        self.cur = else_bb;
        self.lower_stmts(orelse);
        self.terminate(BTerm::Br(merge_bb));

        self.cur = merge_bb;
    }

    /// `for name in range(...): body` — a plain scalar loop (`lower_for`'s own precedent,
    /// stack-slot counter/cond/body/step/exit), not one of this pass's own tile-materialization
    /// loops. Used for a real Triton K-loop; any other iterable is refused.
    fn lower_for(
        &mut self,
        target: &Expr,
        iter: &Expr,
        body: &[Stmt],
        orelse: &[Stmt],
        span: Span,
    ) {
        if !orelse.is_empty() {
            self.diag(span, "'for ... else'", "not supported");
        }
        let Expr::Name { name, .. } = target else {
            self.diag(
                span,
                "'for' loop target other than a bare name",
                "not supported",
            );
            return;
        };
        let Expr::Call { func, args, .. } = iter else {
            self.diag(
                span,
                "'for' loop over something other than 'range(...)'",
                "not supported",
            );
            return;
        };
        if !matches!(&**func, Expr::Name { name, .. } if name == "range") {
            self.diag(
                span,
                "'for' loop over something other than 'range(...)'",
                "not supported",
            );
            return;
        }
        let (start, stop, step) = match args.len() {
            1 => (
                self.const_i64(0),
                self.eval_expr(&args[0], &[]),
                self.const_i64(1),
            ),
            2 => (
                self.eval_expr(&args[0], &[]),
                self.eval_expr(&args[1], &[]),
                self.const_i64(1),
            ),
            3 => (
                self.eval_expr(&args[0], &[]),
                self.eval_expr(&args[1], &[]),
                self.eval_expr(&args[2], &[]),
            ),
            _ => {
                self.diag(
                    span,
                    "'range(...)' with an unsupported argument count",
                    "expected 1, 2, or 3 arguments",
                );
                return;
            }
        };

        let ctr = self.get_or_new_scalar_var(name, I64);
        self.store_scalar(&ctr, start);

        let cond_bb = self.alloc_block();
        let body_bb = self.alloc_block();
        let step_bb = self.alloc_block();
        let exit_bb = self.alloc_block();
        self.terminate(BTerm::Br(cond_bb));

        self.cur = cond_bb;
        let iv = self.load_scalar(&ctr);
        let cmp = self.push(
            BTy::Scalar(BScalar::I1),
            BOp::ICmp(ICmpPred::Slt, I64, iv, stop),
        );
        self.terminate(BTerm::CondBr(cmp, body_bb, exit_bb));

        self.cur = body_bb;
        self.ctrl_stack.push((exit_bb, step_bb));
        self.lower_stmts(body);
        self.ctrl_stack.pop();
        self.terminate(BTerm::Br(step_bb));

        self.cur = step_bb;
        let iv2 = self.load_scalar(&ctr);
        let next = self.push(I64, BOp::Bin(BBin::Add, iv2, step));
        self.store_scalar(&ctr, next);
        self.terminate(BTerm::Br(cond_bb));

        self.cur = exit_bb;
    }
}

/// The final attribute name of a call's callee (`tl.load` -> `"load"`), matching
/// `triton_check`'s own `attr_name` (private to that module, so duplicated here rather than
/// exported purely for this).
fn attr_name(func: &Expr) -> Option<&str> {
    match func {
        Expr::Attribute { attr, .. } => Some(attr.as_str()),
        _ => None,
    }
}

// ---- expressions ------------------------------------------------------------------------------

impl<'a> Lowerer<'a> {
    /// The element `Ty` a value would be stored/loaded as. Consults an already-bound
    /// variable's own recorded type where possible (so a tile that was itself materialized
    /// under a specific dtype decision is addressed consistently everywhere it is later
    /// referenced) rather than re-deriving one from `TileTy::elem()` — the checker's `Elem` is
    /// coarser than this pass needs (it does not, for instance, know a `tl.load` defaults to
    /// `f32`; see the module header's dtype-assumption note).
    fn elem_bty_for(&mut self, e: &Expr) -> BTy {
        match e {
            Expr::Name { name, .. } => self
                .vars
                .get(name)
                .map(|v| v.elem_bty)
                .unwrap_or(BTy::Scalar(BScalar::F32)),
            Expr::IntLit { .. } => I64,
            Expr::FloatLit { .. } => BTy::Scalar(BScalar::F32),
            Expr::BoolLit { .. } => BTy::Scalar(BScalar::I1),
            Expr::BinOp { lhs, rhs, .. } => {
                let l = self.elem_bty_for(lhs);
                let r = self.elem_bty_for(rhs);
                combine_bty(l, r)
            }
            Expr::UnaryOp { operand, .. } => self.elem_bty_for(operand),
            Expr::Compare { .. } => BTy::Scalar(BScalar::I1),
            Expr::Ternary { body, orelse, .. } => {
                let b = self.elem_bty_for(body);
                let o = self.elem_bty_for(orelse);
                combine_bty(b, o)
            }
            Expr::Subscript { value, .. } => self.elem_bty_for(value),
            Expr::Call { func, keywords, .. } => match attr_name(func) {
                Some("load") => BTy::Scalar(BScalar::F32),
                Some("zeros") => zeros_dtype_bty(keywords),
                Some("dot") => BTy::Scalar(BScalar::F32),
                Some("arange") | Some("program_id") => I64,
                _ => BTy::Scalar(BScalar::F32),
            },
            _ => BTy::Scalar(BScalar::F32),
        }
    }

    /// `bty` is the *result's* own element type (`elem_bty_for` on the whole binary expression,
    /// not either operand alone) — deciding int-vs-float dispatch from it, rather than a
    /// separately-threaded flag, is what makes pointer arithmetic (`a_ptr + <index tile>`,
    /// `combine_bty`'s "a pointer always wins" rule) fall out for free: `bty` is `Ptr(_)`
    /// exactly when this is address arithmetic, and the result is produced with that same
    /// `Ptr` type (mirroring `tiled_sgemm.rs`'s own `addr()` helper) rather than a bare integer.
    fn apply_binop(&mut self, op: TBin, l: ValRef, r: ValRef, bty: BTy, span: Span) -> ValRef {
        if let BTy::Ptr(_) = bty {
            let bop = match op {
                TBin::Add => Some(BBin::Add),
                TBin::Sub => Some(BBin::Sub),
                _ => None,
            };
            return match bop {
                Some(b) => self.push(bty, BOp::Bin(b, l, r)),
                None => {
                    self.diag(
                        span,
                        format!("binary operator {op:?} on a pointer-typed tile"),
                        "only '+' and '-' are supported for address arithmetic",
                    );
                    self.push(bty, BOp::ConstInt(0))
                }
            };
        }
        let is_float = matches!(bty, BTy::Scalar(BScalar::F32 | BScalar::F64));
        let out_bty = if is_float {
            BTy::Scalar(BScalar::F32)
        } else {
            I64
        };
        let bop = match op {
            TBin::Add => Some(if is_float { BBin::FAdd } else { BBin::Add }),
            TBin::Sub => Some(if is_float { BBin::FSub } else { BBin::Sub }),
            TBin::Mul => Some(if is_float { BBin::FMul } else { BBin::Mul }),
            TBin::Div => Some(if is_float { BBin::FDiv } else { BBin::Div }),
            // BIR's `div`/`rem` carry no signed/unsigned distinction (a gap `lower.rs`'s own
            // header already documents for the CUDA-C side); this pass inherits the same gap
            // rather than inventing a distinction BIR has no way to express.
            TBin::FloorDiv => Some(if is_float { BBin::FDiv } else { BBin::Div }),
            TBin::Mod => Some(if is_float { BBin::FRem } else { BBin::Rem }),
            TBin::BitAnd if !is_float => Some(BBin::And),
            TBin::BitOr if !is_float => Some(BBin::Or),
            TBin::BitXor if !is_float => Some(BBin::Xor),
            TBin::LShift if !is_float => Some(BBin::Shl),
            TBin::RShift if !is_float => Some(BBin::Lshr),
            _ => None,
        };
        match bop {
            Some(b) => self.push(out_bty, BOp::Bin(b, l, r)),
            None => {
                self.diag(
                    span,
                    format!("binary operator {op:?}"),
                    "not supported by tile lowering in this element context",
                );
                self.push(out_bty, BOp::ConstInt(0))
            }
        }
    }

    fn apply_cmp(&mut self, op: CmpOp, l: ValRef, r: ValRef, is_float: bool, span: Span) -> ValRef {
        let i1 = BTy::Scalar(BScalar::I1);
        if is_float {
            let pred = match op {
                CmpOp::Eq => Some(FCmpPred::Oeq),
                CmpOp::NotEq => Some(FCmpPred::One),
                CmpOp::Lt => Some(FCmpPred::Olt),
                CmpOp::LtE => Some(FCmpPred::Ole),
                CmpOp::Gt => Some(FCmpPred::Ogt),
                CmpOp::GtE => Some(FCmpPred::Oge),
                _ => None,
            };
            match pred {
                Some(p) => self.push(i1, BOp::FCmp(p, BTy::Scalar(BScalar::F32), l, r)),
                None => {
                    self.diag(span, format!("comparison operator {op:?}"), "not supported");
                    self.push(i1, BOp::ConstInt(0))
                }
            }
        } else {
            let pred = match op {
                CmpOp::Eq => Some(ICmpPred::Eq),
                CmpOp::NotEq => Some(ICmpPred::Ne),
                CmpOp::Lt => Some(ICmpPred::Slt),
                CmpOp::LtE => Some(ICmpPred::Sle),
                CmpOp::Gt => Some(ICmpPred::Sgt),
                CmpOp::GtE => Some(ICmpPred::Sge),
                _ => None,
            };
            match pred {
                Some(p) => self.push(i1, BOp::ICmp(p, I64, l, r)),
                None => {
                    self.diag(span, format!("comparison operator {op:?}"), "not supported");
                    self.push(i1, BOp::ConstInt(0))
                }
            }
        }
    }

    /// Evaluates `e` for tile position `idx` (one `ValRef` per axis of `e`'s *own* inferred
    /// rank — a scalar sub-expression ignores `idx` entirely, which is exactly Triton's
    /// broadcast rule: `idx` always carries the enclosing loop's full induction-variable list,
    /// and each node truncates to however many of its leading axes it actually has).
    fn eval_expr(&mut self, e: &Expr, idx: &[ValRef]) -> ValRef {
        let rank = self.shape_of(e).rank().unwrap_or(0) as usize;
        let idx: &[ValRef] = if idx.len() > rank { &idx[..rank] } else { idx };
        match e {
            Expr::Name { name, span } => match self.vars.get(name).cloned() {
                Some(v) if matches!(v.shape, TileTy::Scalar(_)) => self.load_scalar(&v),
                Some(v) => {
                    let addr = self.tile_elem_addr(&v, idx, *span);
                    self.push(
                        v.elem_bty,
                        BOp::Load {
                            ptr: addr,
                            space: BSpace::Global,
                            align: bty_align(v.elem_bty),
                            volatile: false,
                        },
                    )
                }
                None => {
                    self.diag(
                        *span,
                        format!("reference to undefined name '{name}'"),
                        "cannot lower",
                    );
                    self.const_i64(0)
                }
            },
            Expr::IntLit { text, span } => match parse_int_text(text) {
                Some(v) => self.const_i64(v),
                None => {
                    self.diag(*span, "integer literal", "could not be parsed");
                    self.const_i64(0)
                }
            },
            Expr::FloatLit { text, span } => match text.parse::<f64>() {
                Ok(v) => self.push(BTy::Scalar(BScalar::F32), BOp::ConstFloat(v)),
                Err(_) => {
                    self.diag(*span, "float literal", "could not be parsed");
                    self.push(BTy::Scalar(BScalar::F32), BOp::ConstFloat(0.0))
                }
            },
            Expr::BoolLit { value, .. } => {
                self.push(BTy::Scalar(BScalar::I1), BOp::ConstInt(i64::from(*value)))
            }
            Expr::UnaryOp { op, operand, span } => {
                let is_float = matches!(
                    self.elem_bty_for(operand),
                    BTy::Scalar(BScalar::F32 | BScalar::F64)
                );
                let v = self.eval_expr(operand, idx);
                match op {
                    UnaryOp::UAdd => v,
                    UnaryOp::USub => {
                        let bty = if is_float {
                            BTy::Scalar(BScalar::F32)
                        } else {
                            I64
                        };
                        let zero = if is_float {
                            self.push(BTy::Scalar(BScalar::F32), BOp::ConstFloat(0.0))
                        } else {
                            self.const_i64(0)
                        };
                        self.apply_binop(TBin::Sub, zero, v, bty, *span)
                    }
                    UnaryOp::Not => {
                        let one = self.push(BTy::Scalar(BScalar::I1), BOp::ConstInt(1));
                        self.push(BTy::Scalar(BScalar::I1), BOp::Bin(BBin::Xor, v, one))
                    }
                    UnaryOp::Invert => {
                        let neg1 = self.const_i64(-1);
                        self.push(I64, BOp::Bin(BBin::Xor, v, neg1))
                    }
                }
            }
            Expr::BinOp { op, lhs, rhs, span } => {
                let bty = self.elem_bty_for(e);
                let lbty = self.elem_bty_for(lhs);
                let rbty = self.elem_bty_for(rhs);
                let mut l = self.eval_expr(lhs, idx);
                let mut r = self.eval_expr(rhs, idx);
                // Pointer arithmetic: BIR's `Bin::Add`/`Sub` do no implicit scaling the way C's
                // `ptr + i` does (`i` scaled by the pointee's size) — this pass has to insert
                // that scaling itself, on whichever side is the plain integer, using the same
                // assumed 4-byte element width documented in the module header. Without this, a
                // Triton kernel's ordinary `a_ptr + offsets` idiom computes a byte address that
                // is off by a factor of the element width — the actual bug this comment fixes,
                // caught by `masked_triton_vector_add_links_and_runs` segfaulting before this
                // scaling existed.
                if let BTy::Ptr(_) = bty {
                    if !matches!(lbty, BTy::Ptr(_)) {
                        l = self.scale_by_elem_bytes(l);
                    }
                    if !matches!(rbty, BTy::Ptr(_)) {
                        r = self.scale_by_elem_bytes(r);
                    }
                }
                self.apply_binop(*op, l, r, bty, *span)
            }
            Expr::Compare {
                left,
                ops,
                comparators,
                span,
            } => {
                let is_float = matches!(
                    self.elem_bty_for(left),
                    BTy::Scalar(BScalar::F32 | BScalar::F64)
                );
                let mut prev = self.eval_expr(left, idx);
                let mut result: Option<ValRef> = None;
                for (op, comp) in ops.iter().zip(comparators.iter()) {
                    let rv = self.eval_expr(comp, idx);
                    let c = self.apply_cmp(*op, prev, rv, is_float, *span);
                    result = Some(match result {
                        None => c,
                        Some(acc) => {
                            self.push(BTy::Scalar(BScalar::I1), BOp::Bin(BBin::And, acc, c))
                        }
                    });
                    prev = rv;
                }
                result.unwrap_or_else(|| self.push(BTy::Scalar(BScalar::I1), BOp::ConstInt(1)))
            }
            Expr::Subscript { value, index, span } => match reshape_steps(index) {
                Some(steps) => {
                    let inner_idx: Vec<ValRef> = steps
                        .iter()
                        .zip(idx.iter())
                        .filter(|(s, _)| **s == ReshapeStep::Keep)
                        .map(|(_, v)| *v)
                        .collect();
                    self.eval_expr(value, &inner_idx)
                }
                None => {
                    self.diag(
                        *span,
                        "tile subscript other than a '[:, None]'/'[None, :]' reshape",
                        "not supported",
                    );
                    self.const_i64(0)
                }
            },
            Expr::Call {
                func,
                args,
                keywords,
                span,
            } => self.eval_call(func, args, keywords, idx, *span),
            other => {
                self.diag(
                    other.span(),
                    "expression form",
                    "not supported by tile lowering",
                );
                self.const_i64(0)
            }
        }
    }

    fn eval_call(
        &mut self,
        func: &Expr,
        args: &[Expr],
        keywords: &[Keyword],
        idx: &[ValRef],
        span: Span,
    ) -> ValRef {
        match attr_name(func) {
            Some("arange") => {
                if args.len() != 2 || idx.len() != 1 {
                    self.diag(
                        span,
                        "'tl.arange' in an unexpected context",
                        "expected exactly two arguments in a rank-1 context",
                    );
                    return self.const_i64(0);
                }
                let lo = self.eval_expr(&args[0], &[]);
                self.push(I64, BOp::Bin(BBin::Add, lo, idx[0]))
            }
            Some("load") => self.eval_masked_load(args, keywords, idx, span),
            Some("program_id") => {
                let axis = args.first().and_then(expr_as_i64).unwrap_or(0);
                match axis {
                    0 => self.push(I64, BOp::BidX),
                    1 => self.push(I64, BOp::BidY),
                    2 => self.push(I64, BOp::BidZ),
                    _ => {
                        self.diag(span, "'tl.program_id' axis", "must be 0, 1, or 2");
                        self.const_i64(0)
                    }
                }
            }
            Some("zeros") => self.zero_of(zeros_dtype_bty(keywords)),
            Some("dot") => {
                self.diag(
                    span,
                    "'tl.dot' used somewhere other than directly as an assignment's right-hand side",
                    "not supported",
                );
                self.const_i64(0)
            }
            _ => {
                self.diag(
                    span,
                    "call",
                    "unrecognized or unsupported callee in tile lowering",
                );
                self.const_i64(0)
            }
        }
    }

    /// `tl.load(ptr, mask=.., other=..)` at tile position `idx`: the pointer address is always
    /// computed (pure integer arithmetic, cannot fault), but the actual `Load` only executes
    /// inside the mask-true arm of a real `CondBr` diamond — see the module header for why a
    /// `Select` would be unsafe here. The diamond's result is threaded back through a small
    /// scratch scalar slot (this pass's general "no phi" convention) rather than a `phi`.
    fn eval_masked_load(
        &mut self,
        args: &[Expr],
        keywords: &[Keyword],
        idx: &[ValRef],
        span: Span,
    ) -> ValRef {
        if args.is_empty() {
            self.diag(span, "'tl.load' with no pointer argument", "cannot lower");
            return self.const_i64(0);
        }
        let ptr_val = self.eval_expr(&args[0], idx);
        let mask_kw = keywords.iter().find(|k| k.name.as_deref() == Some("mask"));
        let other_kw = keywords.iter().find(|k| k.name.as_deref() == Some("other"));
        let elem_bty = BTy::Scalar(BScalar::F32);

        let tmp_slot = self.new_slot_id();
        let tmp_vs = VarSlot {
            storage: Storage::Synthetic {
                slot: tmp_slot,
                space: BSpace::Local,
            },
            shape: TileTy::Scalar(Elem::Float),
            elem_bty,
            dims: Vec::new(),
        };

        match mask_kw {
            None => {
                let v = self.push(
                    elem_bty,
                    BOp::Load {
                        ptr: ptr_val,
                        space: BSpace::Global,
                        align: bty_align(elem_bty),
                        volatile: false,
                    },
                );
                self.store_scalar(&tmp_vs, v);
            }
            Some(mkw) => {
                let mask_val = self.eval_expr(&mkw.value, idx);
                let load_bb = self.alloc_block();
                let other_bb = self.alloc_block();
                let merge_bb = self.alloc_block();
                self.terminate(BTerm::CondBr(mask_val, load_bb, other_bb));

                self.cur = load_bb;
                let v = self.push(
                    elem_bty,
                    BOp::Load {
                        ptr: ptr_val,
                        space: BSpace::Global,
                        align: bty_align(elem_bty),
                        volatile: false,
                    },
                );
                self.store_scalar(&tmp_vs, v);
                self.terminate(BTerm::Br(merge_bb));

                self.cur = other_bb;
                let ov = match other_kw {
                    Some(okw) => self.eval_expr(&okw.value, idx),
                    None => self.zero_of(elem_bty),
                };
                self.store_scalar(&tmp_vs, ov);
                self.terminate(BTerm::Br(merge_bb));

                self.cur = merge_bb;
            }
        }
        self.load_scalar(&tmp_vs)
    }
}

/// Combines two operands' element types for a binary op's own result. A pointer always wins
/// (matches C's own "pointer + integer = pointer" convention, and this is exactly how a real
/// Triton kernel's `a_ptrs = a_ptr + <index tile>` idiom builds up an address tile one step at
/// a time — mirrors `tiled_sgemm.rs`'s own `addr()` helper, which types pointer arithmetic as
/// `Ty::Ptr`, not a bare integer, for the same reason: a later `Load`/`Store` addressing this
/// value needs its space, not just its width).
fn combine_bty(a: BTy, b: BTy) -> BTy {
    if let BTy::Ptr(s) = a {
        return BTy::Ptr(s);
    }
    if let BTy::Ptr(s) = b {
        return BTy::Ptr(s);
    }
    if a == b {
        return a;
    }
    let is_f = |t: BTy| matches!(t, BTy::Scalar(BScalar::F32 | BScalar::F64));
    if is_f(a) || is_f(b) {
        BTy::Scalar(BScalar::F32)
    } else {
        I64
    }
}

fn zeros_dtype_bty(keywords: &[Keyword]) -> BTy {
    for kw in keywords {
        if kw.name.as_deref() == Some("dtype") {
            if let Expr::Attribute { attr, .. } = &kw.value {
                return match attr.as_str() {
                    "int8" | "int16" | "int32" | "int64" => I64,
                    _ => BTy::Scalar(BScalar::F32),
                };
            }
        }
    }
    BTy::Scalar(BScalar::F32)
}

fn expr_as_i64(e: &Expr) -> Option<i64> {
    match e {
        Expr::IntLit { text, .. } => parse_int_text(text),
        _ => None,
    }
}

/// One reshape-subscript index parsed into `[ReshapeStep]` — a private copy of
/// `triton_check`'s own `subscript_steps` (not exported by that module).
fn reshape_steps(index: &Expr) -> Option<Vec<ReshapeStep>> {
    fn step_of(e: &Expr) -> Option<ReshapeStep> {
        match e {
            Expr::Slice {
                lower: None,
                upper: None,
                step: None,
                ..
            } => Some(ReshapeStep::Keep),
            Expr::NoneLit { .. } => Some(ReshapeStep::Insert),
            _ => None,
        }
    }
    match index {
        Expr::Tuple { elts, .. } => elts.iter().map(step_of).collect(),
        other => step_of(other).map(|s| vec![s]),
    }
}

/// A private copy of `triton_check`'s own `parse_int_text` (not exported by that module).
fn parse_int_text(text: &str) -> Option<i64> {
    let cleaned: String = text.chars().filter(|c| *c != '_').collect();
    for (prefix, radix) in [
        ("0x", 16),
        ("0X", 16),
        ("0o", 8),
        ("0O", 8),
        ("0b", 2),
        ("0B", 2),
    ] {
        if let Some(digits) = cleaned.strip_prefix(prefix) {
            return i64::from_str_radix(digits, radix).ok();
        }
    }
    cleaned.parse::<i64>().ok()
}

/// Every non-`constexpr` parameter name that appears anywhere inside the pointer-argument
/// subtree of some `tl.load`/`tl.store` call in `k`'s body (see the module header's "usage
/// heuristic" section).
/// Every parameter name reachable, transitively, from a `tl.load`/`tl.store` pointer argument.
/// The direct case (`tl.load(a_ptr + offsets)`) alone misses the equally common idiom of
/// building the address in its own named step first (`a_ptrs = a_ptr + ...; tl.load(a_ptrs)`
/// — the real matmul proof kernel's own style, and `triton_check_tests.rs`'s own `MATMUL`
/// fixture's): `a_ptrs` itself is what appears in the load's argument subtree, not `a_ptr`. So
/// this also builds a simple name-flow map (`collect_defs_stmts`: assignment target -> every
/// name in its own right-hand side) and closes the direct set over it — a parameter counts as
/// a pointer if its name is reachable from some load/store's own pointer argument by following
/// zero or more assignments backward.
fn ptr_param_names(k: &KernelFn) -> std::collections::HashSet<String> {
    let mut direct = std::collections::HashSet::new();
    for s in &k.body {
        collect_ptr_names_stmt(s, &mut direct);
    }
    let mut defs: HashMap<String, std::collections::HashSet<String>> = HashMap::new();
    collect_defs_stmts(&k.body, &mut defs);

    let mut closed = std::collections::HashSet::new();
    let mut stack: Vec<String> = direct.into_iter().collect();
    while let Some(name) = stack.pop() {
        if closed.insert(name.clone()) {
            if let Some(sources) = defs.get(&name) {
                for s in sources {
                    if !closed.contains(s) {
                        stack.push(s.clone());
                    }
                }
            }
        }
    }
    closed
}

/// Builds `target name -> every name referenced in its own assigned expression`, recursing into
/// every nested statement body (`if`/`for`/`while`) — a plain, flow-insensitive name-flow map,
/// not real def-use/reaching-definitions analysis (this pass's own kernels never rebind a name
/// to two different underlying pointers, so the extra precision would buy nothing here).
fn collect_defs_stmts(
    stmts: &[Stmt],
    defs: &mut HashMap<String, std::collections::HashSet<String>>,
) {
    fn record(
        target: &Expr,
        value: &Expr,
        defs: &mut HashMap<String, std::collections::HashSet<String>>,
    ) {
        if let Expr::Name { name, .. } = target {
            let mut names = std::collections::HashSet::new();
            collect_all_names(value, &mut names);
            defs.entry(name.clone()).or_default().extend(names);
        }
    }
    for s in stmts {
        match s {
            Stmt::Assign { targets, value, .. } => {
                for t in targets {
                    record(t, value, defs);
                }
            }
            Stmt::AugAssign { target, value, .. } => record(target, value, defs),
            Stmt::AnnAssign {
                target,
                value: Some(v),
                ..
            } => record(target, v, defs),
            Stmt::AnnAssign { value: None, .. } => {}
            Stmt::If { body, orelse, .. }
            | Stmt::For { body, orelse, .. }
            | Stmt::While { body, orelse, .. } => {
                collect_defs_stmts(body, defs);
                collect_defs_stmts(orelse, defs);
            }
            _ => {}
        }
    }
}

fn collect_ptr_names_stmt(s: &Stmt, out: &mut std::collections::HashSet<String>) {
    match s {
        Stmt::Expr { value, .. } => collect_ptr_names_expr(value, out),
        Stmt::Assign { value, .. } => collect_ptr_names_expr(value, out),
        Stmt::AugAssign { value, .. } => collect_ptr_names_expr(value, out),
        Stmt::AnnAssign { value: Some(v), .. } => collect_ptr_names_expr(v, out),
        Stmt::AnnAssign { value: None, .. } => {}
        Stmt::If { body, orelse, .. } => {
            for st in body.iter().chain(orelse) {
                collect_ptr_names_stmt(st, out);
            }
        }
        Stmt::For { body, orelse, .. } => {
            for st in body.iter().chain(orelse) {
                collect_ptr_names_stmt(st, out);
            }
        }
        Stmt::While { body, orelse, .. } => {
            for st in body.iter().chain(orelse) {
                collect_ptr_names_stmt(st, out);
            }
        }
        _ => {}
    }
}

fn collect_ptr_names_expr(e: &Expr, out: &mut std::collections::HashSet<String>) {
    match e {
        Expr::Call {
            func,
            args,
            keywords,
            ..
        } => {
            let is_ptr_call = matches!(attr_name(func), Some("load") | Some("store"));
            if is_ptr_call {
                if let Some(ptr_arg) = args.first() {
                    collect_all_names(ptr_arg, out);
                }
            }
            for a in args {
                collect_ptr_names_expr(a, out);
            }
            for kw in keywords {
                collect_ptr_names_expr(&kw.value, out);
            }
        }
        Expr::BinOp { lhs, rhs, .. } => {
            collect_ptr_names_expr(lhs, out);
            collect_ptr_names_expr(rhs, out);
        }
        Expr::UnaryOp { operand, .. } => collect_ptr_names_expr(operand, out),
        Expr::Compare {
            left, comparators, ..
        } => {
            collect_ptr_names_expr(left, out);
            for c in comparators {
                collect_ptr_names_expr(c, out);
            }
        }
        Expr::Subscript { value, .. } => collect_ptr_names_expr(value, out),
        Expr::Ternary {
            test, body, orelse, ..
        } => {
            collect_ptr_names_expr(test, out);
            collect_ptr_names_expr(body, out);
            collect_ptr_names_expr(orelse, out);
        }
        Expr::Tuple { elts, .. } | Expr::List { elts, .. } => {
            for el in elts {
                collect_ptr_names_expr(el, out);
            }
        }
        _ => {}
    }
}

fn collect_all_names(e: &Expr, out: &mut std::collections::HashSet<String>) {
    match e {
        Expr::Name { name, .. } => {
            out.insert(name.clone());
        }
        Expr::BinOp { lhs, rhs, .. } => {
            collect_all_names(lhs, out);
            collect_all_names(rhs, out);
        }
        Expr::UnaryOp { operand, .. } => collect_all_names(operand, out),
        Expr::Subscript { value, .. } => collect_all_names(value, out),
        Expr::Call { args, .. } => {
            for a in args {
                collect_all_names(a, out);
            }
        }
        Expr::Tuple { elts, .. } | Expr::List { elts, .. } => {
            for el in elts {
                collect_all_names(el, out);
            }
        }
        _ => {}
    }
}
