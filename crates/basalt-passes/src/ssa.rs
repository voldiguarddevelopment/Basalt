// SSA construction: promotes basalt-sema's synthetic memory-slot pattern (see that crate's
// `lower.rs` header, "Locals are stack slots, not SSA values") into real SSA form.
//
// A "slot" is one distinct `(address space, synthesized address)` pair, where the address is
// the raw immediate baked into a `const.i ptr.<space> <n>` by the lowering pass — an opaque
// integer key, not a real pointer. A slot is promotable only if every use of its address
// constant is directly the `ptr` operand of a `load` or a `store` (never the stored value,
// never any other operand — see `analyze` below) and every such load/store agrees on one
// value type. `global`-space memory is never a slot candidate: it is a real dereference, not a
// synthesized local's home, and this pass leaves it untouched.
//
// Construction follows Braun, Buchwald, Hack, Leissa, Mallon, Zwinkau ("Simple and Efficient
// Construction of Static Single Assignment Form", 2013): a per-block "current value" lookup
// (`read_variable`/`resolve_load`) that inserts a placeholder `phi` at a multi-predecessor
// block before recursing into its predecessors (breaking cycles from loops), followed by a
// trivial-phi elimination fixed point. Because BIR hands over its whole CFG upfront rather
// than being built incrementally, there is no need for the paper's "sealed block" bookkeeping:
// every predecessor list is already complete, so a block's value can always be (recursively,
// lazily, memoized) resolved on demand regardless of the order this pass happens to visit
// blocks in.
//
// Two-stage output construction: this pass first walks the whole function purely to resolve
// values and discover every phi that will be needed (`force_resolve_all`), runs trivial-phi
// elimination over the complete set, and only then allocates the new function's instruction
// arena — one pass to assign every surviving instruction (translated original, or new phi) its
// final id in exactly the order it will be printed (blocks in order; a block's own phis, then
// its surviving body), and a second pass to fill in real operand lists now that every id in the
// function is known. This is what lets a loop header's phi reference a value produced later,
// in the loop's latch block, without needing to build the arena out of order.

use std::collections::{BTreeMap, BTreeSet, HashMap};

use basalt_bir::{
    AddrSpace, Block, BlockId, Function, Inst, InstId, Module, Op, Scalar, Term, Ty, ValRef,
};

/// Identity of one promotable memory slot: `(address-space tag, synthesized address)`. The
/// tag exists only because `AddrSpace` has no `Hash`/`Ord` impl of its own; `Global` never
/// gets a tag since it is never a slot.
type SlotKey = (u8, i64);

fn slot_space_tag(space: AddrSpace) -> Option<u8> {
    match space {
        AddrSpace::Global => None,
        AddrSpace::Shared => Some(0),
        AddrSpace::Constant => Some(1),
        AddrSpace::Local => Some(2),
        AddrSpace::Param => Some(3),
    }
}

/// Promotes every safely-promotable slot in `module`'s functions to real SSA values, replacing
/// eliminated `load`/`store` traffic with direct references and `phi`s at merge points.
/// `global`-space memory and any slot that fails the safety checks in `analyze` are copied
/// through unchanged.
pub fn construct_ssa(module: &Module) -> Module {
    Module {
        funcs: module.funcs.iter().map(construct_ssa_fn).collect(),
        launch_bounds: module.launch_bounds,
        shared_mem_bytes: module.shared_mem_bytes,
        target_dtypes: module.target_dtypes.clone(),
    }
}

fn construct_ssa_fn(f: &Function) -> Function {
    let mut cx = Ctx::new(f);
    cx.force_resolve_all();
    let repl = eliminate_trivial_phis(&cx.phis);
    cx.build(&repl)
}

/// A value resolved by SSA construction, not yet given a final arena id: either a function
/// parameter, an original instruction this pass keeps as-is, a phi this pass inserts (indexed
/// into `Ctx::phis`), or a synthesized zero standing in for a read of a never-written slot.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Sym {
    Param(u32),
    Kept(InstId),
    Phi(usize),
    Undef(Ty),
}

/// One phi this pass wants to insert, prior to trivial-phi elimination and prior to knowing
/// its final `InstId`.
struct PhiNode {
    ty: Ty,
    block: BlockId,
    operands: Vec<(BlockId, Sym)>,
}

fn successors(term: &Term) -> Vec<BlockId> {
    match term {
        Term::Br(b) => vec![*b],
        Term::CondBr(_, t, e) => vec![*t, *e],
        Term::Switch(_, default, cases) => {
            let mut out = vec![*default];
            out.extend(cases.iter().map(|(_, b)| *b));
            out
        }
        Term::Ret(_) => Vec::new(),
    }
}

fn preds_of(f: &Function) -> Vec<Vec<BlockId>> {
    let mut preds = vec![Vec::new(); f.blocks.len()];
    for (idx, block) in f.blocks.iter().enumerate() {
        let from = BlockId(idx as u32);
        for succ in successors(&block.term) {
            let into = &mut preds[succ.0 as usize];
            if !into.contains(&from) {
                into.push(from);
            }
        }
    }
    preds
}

/// Every `ValRef` operand of `op`, in structural order. Shared by escape analysis (anything
/// reached here other than a load/store's own special-cased `ptr` is, by definition, an
/// escaping use) and by the forced-resolution walk.
fn operand_refs(op: &Op) -> Vec<ValRef> {
    match op {
        Op::ConstInt(_) | Op::ConstFloat(_) => Vec::new(),
        Op::Bin(_, a, b) => vec![*a, *b],
        Op::ICmp(_, _, a, b) => vec![*a, *b],
        Op::FCmp(_, _, a, b) => vec![*a, *b],
        Op::Select(c, a, b) => vec![*c, *a, *b],
        Op::Cast(_, _, v) => vec![*v],
        Op::Load { ptr, .. } => vec![*ptr],
        Op::Store { ptr, val, .. } => vec![*ptr, *val],
        Op::Phi(incoming) => incoming.iter().map(|(_, v)| *v).collect(),
        Op::TidX
        | Op::TidY
        | Op::TidZ
        | Op::BidX
        | Op::BidY
        | Op::BidZ
        | Op::BdimX
        | Op::BdimY
        | Op::BdimZ
        | Op::GdimX
        | Op::GdimY
        | Op::GdimZ
        | Op::Barrier => Vec::new(),
        Op::Shuffle(_, a, b) => vec![*a, *b],
        Op::Ballot(a) | Op::VoteAny(a) | Op::VoteAll(a) => vec![*a],
        Op::Atomic(_, ptr, v, _) => vec![*ptr, *v],
        Op::AtomicCas(ptr, cmp, new, _) => vec![*ptr, *cmp, *new],
        Op::Mma { a, b, c, d, .. } => vec![*a, *b, *c, *d],
        Op::KernelLaunch {
            grid,
            block,
            shared,
            stream,
            args,
            ..
        } => {
            let mut v = Vec::with_capacity(8 + args.len());
            v.extend_from_slice(grid);
            v.extend_from_slice(block);
            v.push(*shared);
            v.push(*stream);
            v.extend(args.iter().copied());
            v
        }
        Op::CudaMalloc { size } => vec![*size],
        Op::CudaMemcpy {
            dst,
            src,
            count,
            kind,
        } => vec![*dst, *src, *count, *kind],
        Op::CudaFree { ptr } => vec![*ptr],
        Op::CudaDeviceSynchronize => Vec::new(),
    }
}

/// Rebuilds `op` with every `ValRef` operand passed through `f`, in the same structural order
/// `operand_refs` walks.
fn map_op(op: &Op, mut f: impl FnMut(ValRef) -> ValRef) -> Op {
    match op {
        Op::ConstInt(v) => Op::ConstInt(*v),
        Op::ConstFloat(v) => Op::ConstFloat(*v),
        Op::Bin(o, a, b) => Op::Bin(*o, f(*a), f(*b)),
        Op::ICmp(p, t, a, b) => Op::ICmp(*p, *t, f(*a), f(*b)),
        Op::FCmp(p, t, a, b) => Op::FCmp(*p, *t, f(*a), f(*b)),
        Op::Select(c, a, b) => Op::Select(f(*c), f(*a), f(*b)),
        Op::Cast(c, t, v) => Op::Cast(*c, *t, f(*v)),
        Op::Load {
            ptr,
            space,
            align,
            volatile,
        } => Op::Load {
            ptr: f(*ptr),
            space: *space,
            align: *align,
            volatile: *volatile,
        },
        Op::Store {
            ptr,
            val,
            ty,
            space,
            align,
            volatile,
        } => Op::Store {
            ptr: f(*ptr),
            val: f(*val),
            ty: *ty,
            space: *space,
            align: *align,
            volatile: *volatile,
        },
        Op::Phi(incoming) => Op::Phi(incoming.iter().map(|(bb, v)| (*bb, f(*v))).collect()),
        Op::TidX => Op::TidX,
        Op::TidY => Op::TidY,
        Op::TidZ => Op::TidZ,
        Op::BidX => Op::BidX,
        Op::BidY => Op::BidY,
        Op::BidZ => Op::BidZ,
        Op::BdimX => Op::BdimX,
        Op::BdimY => Op::BdimY,
        Op::BdimZ => Op::BdimZ,
        Op::GdimX => Op::GdimX,
        Op::GdimY => Op::GdimY,
        Op::GdimZ => Op::GdimZ,
        Op::Barrier => Op::Barrier,
        Op::Shuffle(k, a, b) => Op::Shuffle(*k, f(*a), f(*b)),
        Op::Ballot(a) => Op::Ballot(f(*a)),
        Op::VoteAny(a) => Op::VoteAny(f(*a)),
        Op::VoteAll(a) => Op::VoteAll(f(*a)),
        Op::Atomic(o, ptr, v, s) => Op::Atomic(*o, f(*ptr), f(*v), *s),
        Op::AtomicCas(ptr, cmp, new, s) => Op::AtomicCas(f(*ptr), f(*cmp), f(*new), *s),
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
        } => Op::Mma {
            a: f(*a),
            b: f(*b),
            c: f(*c),
            d: f(*d),
            m: *m,
            n: *n,
            k: *k,
            in_dtype: *in_dtype,
            acc_dtype: *acc_dtype,
            layout_a: *layout_a,
            layout_b: *layout_b,
        },
        Op::KernelLaunch {
            kernel,
            grid,
            block,
            shared,
            stream,
            args,
        } => Op::KernelLaunch {
            kernel: kernel.clone(),
            grid: [f(grid[0]), f(grid[1]), f(grid[2])],
            block: [f(block[0]), f(block[1]), f(block[2])],
            shared: f(*shared),
            stream: f(*stream),
            args: args.iter().map(|&v| f(v)).collect(),
        },
        Op::CudaMalloc { size } => Op::CudaMalloc { size: f(*size) },
        Op::CudaMemcpy {
            dst,
            src,
            count,
            kind,
        } => Op::CudaMemcpy {
            dst: f(*dst),
            src: f(*src),
            count: f(*count),
            kind: f(*kind),
        },
        Op::CudaFree { ptr } => Op::CudaFree { ptr: f(*ptr) },
        Op::CudaDeviceSynchronize => Op::CudaDeviceSynchronize,
    }
}

fn term_operand(term: &Term) -> Option<ValRef> {
    match term {
        Term::Br(_) => None,
        Term::CondBr(c, _, _) => Some(*c),
        Term::Switch(v, _, _) => Some(*v),
        Term::Ret(v) => *v,
    }
}

fn map_term(term: &Term, f: impl FnOnce(ValRef) -> ValRef) -> Term {
    match term {
        Term::Br(b) => Term::Br(*b),
        Term::CondBr(c, t, e) => Term::CondBr(f(*c), *t, *e),
        Term::Switch(v, default, cases) => Term::Switch(f(*v), *default, cases.clone()),
        Term::Ret(v) => Term::Ret(v.map(f)),
    }
}

/// Every `const.i ptr.<local/param/shared/constant> <n>` in `f`, keyed by its own `InstId`.
/// Candidacy alone says nothing about safety yet — `analyze` narrows this to the promotable
/// subset.
fn find_candidates(f: &Function) -> HashMap<u32, SlotKey> {
    let mut out = HashMap::new();
    for (idx, inst) in f.insts.iter().enumerate() {
        if let (Ty::Ptr(space), Op::ConstInt(v)) = (inst.ty, &inst.op) {
            if let Some(tag) = slot_space_tag(space) {
                out.insert(idx as u32, (tag, *v));
            }
        }
    }
    out
}

fn note_use(ty_of: &mut BTreeMap<SlotKey, Ty>, bad: &mut BTreeSet<SlotKey>, key: SlotKey, ty: Ty) {
    match ty_of.get(&key) {
        None => {
            ty_of.insert(key, ty);
        }
        Some(&seen) if seen == ty => {}
        Some(_) => {
            bad.insert(key);
        }
    }
}

/// Classifies every slot candidate in `f` as promotable (mapped to its single agreed value
/// type) or not. A slot is disqualified the moment its address constant is used anywhere other
/// than directly as a `load`'s or a `store`'s `ptr` operand (a `store`'s own *value* operand
/// counts as disqualifying — the address itself must never escape), or if its loads/stores
/// disagree on the value type.
fn analyze(f: &Function) -> (HashMap<u32, SlotKey>, BTreeMap<SlotKey, Ty>) {
    let candidates = find_candidates(f);
    let mut ty_of: BTreeMap<SlotKey, Ty> = BTreeMap::new();
    let mut bad: BTreeSet<SlotKey> = BTreeSet::new();

    for inst in &f.insts {
        match &inst.op {
            Op::Load { ptr, .. } => {
                if let ValRef::Val(id) = ptr {
                    if let Some(&key) = candidates.get(&id.0) {
                        note_use(&mut ty_of, &mut bad, key, inst.ty);
                    }
                }
                continue;
            }
            Op::Store { ptr, val, ty, .. } => {
                if let ValRef::Val(id) = ptr {
                    if let Some(&key) = candidates.get(&id.0) {
                        note_use(&mut ty_of, &mut bad, key, *ty);
                    }
                }
                if let ValRef::Val(id) = val {
                    if let Some(&key) = candidates.get(&id.0) {
                        bad.insert(key);
                    }
                }
                continue;
            }
            _ => {}
        }
        for v in operand_refs(&inst.op) {
            if let ValRef::Val(id) = v {
                if let Some(&key) = candidates.get(&id.0) {
                    bad.insert(key);
                }
            }
        }
    }
    for block in &f.blocks {
        if let Some(ValRef::Val(id)) = term_operand(&block.term) {
            if let Some(&key) = candidates.get(&id.0) {
                bad.insert(key);
            }
        }
    }

    ty_of.retain(|key, _| !bad.contains(key));
    (candidates, ty_of)
}

fn undef_op(ty: Ty) -> Op {
    match ty {
        Ty::Scalar(Scalar::F16) | Ty::Scalar(Scalar::F32) | Ty::Scalar(Scalar::F64) => {
            Op::ConstFloat(0.0)
        }
        _ => Op::ConstInt(0),
    }
}

fn follow(repl: &[Option<Sym>], mut s: Sym) -> Sym {
    while let Sym::Phi(i) = s {
        match repl[i] {
            Some(r) => s = r,
            None => break,
        }
    }
    s
}

/// Trivial-phi elimination to a fixed point: a phi whose non-self-referential operands
/// collapse (after chasing already-eliminated phis) to a single value is replaced everywhere
/// by that value. A phi with no non-self operand at all (unreachable except through its own
/// back edges) is replaced by an undef of its own type.
fn eliminate_trivial_phis(phis: &[PhiNode]) -> Vec<Option<Sym>> {
    let mut repl: Vec<Option<Sym>> = vec![None; phis.len()];
    let mut changed = true;
    while changed {
        changed = false;
        for i in 0..phis.len() {
            if repl[i].is_some() {
                continue;
            }
            let mut uniq: Option<Sym> = None;
            let mut trivial = true;
            for &(_, opnd) in &phis[i].operands {
                let resolved = follow(&repl, opnd);
                if resolved == Sym::Phi(i) {
                    continue;
                }
                match uniq {
                    None => uniq = Some(resolved),
                    Some(u) if u == resolved => {}
                    Some(_) => {
                        trivial = false;
                        break;
                    }
                }
            }
            if trivial {
                repl[i] = Some(uniq.unwrap_or(Sym::Undef(phis[i].ty)));
                changed = true;
            }
        }
    }
    repl
}

fn to_valref(
    repl: &[Option<Sym>],
    sym: Sym,
    final_old: &HashMap<u32, InstId>,
    final_phi: &[Option<InstId>],
    undef_ids: &[(Ty, InstId)],
) -> ValRef {
    match follow(repl, sym) {
        Sym::Param(p) => ValRef::Param(p),
        Sym::Kept(id) => ValRef::Val(
            *final_old
                .get(&id.0)
                .expect("kept operand must be materialized before use"),
        ),
        Sym::Phi(idx) => {
            ValRef::Val(final_phi[idx].expect("surviving phi must be materialized before use"))
        }
        Sym::Undef(ty) => ValRef::Val(
            undef_ids
                .iter()
                .find(|(t, _)| *t == ty)
                .map(|(_, id)| *id)
                .expect("undef constant must be materialized before use"),
        ),
    }
}

struct Ctx<'a> {
    f: &'a Function,
    preds: Vec<Vec<BlockId>>,
    promotable: BTreeMap<SlotKey, Ty>,
    slot_of_const: HashMap<u32, SlotKey>,
    inst_block: Vec<u32>,
    inst_pos: Vec<usize>,
    current_def: HashMap<(SlotKey, u32), Sym>,
    end_of_block: HashMap<(SlotKey, u32), Sym>,
    old_resolved: HashMap<u32, Sym>,
    phis: Vec<PhiNode>,
    undef_tys: Vec<Ty>,
}

impl<'a> Ctx<'a> {
    fn new(f: &'a Function) -> Self {
        let (candidates, promotable) = analyze(f);
        let slot_of_const = candidates
            .into_iter()
            .filter(|(_, key)| promotable.contains_key(key))
            .collect();

        let mut inst_block = vec![0u32; f.insts.len()];
        let mut inst_pos = vec![0usize; f.insts.len()];
        for (bidx, block) in f.blocks.iter().enumerate() {
            for (pos, id) in block.insts.iter().enumerate() {
                inst_block[id.0 as usize] = bidx as u32;
                inst_pos[id.0 as usize] = pos;
            }
        }

        Ctx {
            preds: preds_of(f),
            promotable,
            slot_of_const,
            inst_block,
            inst_pos,
            current_def: HashMap::new(),
            end_of_block: HashMap::new(),
            old_resolved: HashMap::new(),
            phis: Vec::new(),
            undef_tys: Vec::new(),
            f,
        }
    }

    fn slot_key_of(&self, ptr: ValRef) -> Option<SlotKey> {
        match ptr {
            ValRef::Param(_) => None,
            ValRef::Val(id) => self.slot_of_const.get(&id.0).copied(),
        }
    }

    /// Whether `id` is dropped entirely from the rebuilt function: a promoted slot's address
    /// constant, or a load/store through one.
    fn is_eliminated(&self, id: InstId) -> bool {
        match &self.f.insts[id.0 as usize].op {
            Op::ConstInt(_) => self.slot_of_const.contains_key(&id.0),
            Op::Load { ptr, .. } | Op::Store { ptr, .. } => self.slot_key_of(*ptr).is_some(),
            _ => false,
        }
    }

    fn undef_sym(&mut self, ty: Ty) -> Sym {
        if !self.undef_tys.contains(&ty) {
            self.undef_tys.push(ty);
        }
        Sym::Undef(ty)
    }

    /// Resolves an operand reference. A reference to a promoted slot's load is chased to
    /// whatever value reaches that load's position; everything else (params, and any
    /// instruction this pass keeps as-is) is memoized as itself.
    fn resolve(&mut self, v: ValRef) -> Sym {
        let id = match v {
            ValRef::Param(p) => return Sym::Param(p),
            ValRef::Val(id) => id,
        };
        if let Some(&s) = self.old_resolved.get(&id.0) {
            return s;
        }
        let load_ptr = match &self.f.insts[id.0 as usize].op {
            Op::Load { ptr, .. } => Some(*ptr),
            _ => None,
        };
        let sym = match load_ptr.and_then(|ptr| self.slot_key_of(ptr)) {
            Some(key) => {
                let block = BlockId(self.inst_block[id.0 as usize]);
                let pos = self.inst_pos[id.0 as usize];
                self.resolve_load(key, block, pos)
            }
            None => Sym::Kept(id),
        };
        self.old_resolved.insert(id.0, sym);
        sym
    }

    /// Resolves what a promoted-slot load at position `pos` of `block` reads: the nearest
    /// earlier store to the same slot within `block`, or (if none) the value flowing into
    /// `block` from its predecessors.
    fn resolve_load(&mut self, key: SlotKey, block: BlockId, pos: usize) -> Sym {
        let f = self.f;
        let mut found = None;
        for &id in f.blocks[block.0 as usize].insts[..pos].iter().rev() {
            if let Op::Store { ptr, val, .. } = &f.insts[id.0 as usize].op {
                if self.slot_key_of(*ptr) == Some(key) {
                    found = Some(*val);
                    break;
                }
            }
        }
        match found {
            Some(v) => self.resolve(v),
            None => self.read_variable_at_block_entry(key, block),
        }
    }

    /// The value of `key` flowing out of the end of `block` — its own last store, if any,
    /// else whatever flows into it.
    fn value_at_block_end(&mut self, key: SlotKey, block: BlockId) -> Sym {
        if let Some(&s) = self.end_of_block.get(&(key, block.0)) {
            return s;
        }
        let f = self.f;
        let mut found = None;
        for &id in f.blocks[block.0 as usize].insts.iter().rev() {
            if let Op::Store { ptr, val, .. } = &f.insts[id.0 as usize].op {
                if self.slot_key_of(*ptr) == Some(key) {
                    found = Some(*val);
                    break;
                }
            }
        }
        let result = match found {
            Some(v) => self.resolve(v),
            None => self.read_variable_at_block_entry(key, block),
        };
        self.end_of_block.insert((key, block.0), result);
        result
    }

    /// The value of `key` flowing into `block`, i.e. Braun et al.'s `readVariable` at the top
    /// of a block: one predecessor recurses directly, several insert a phi (memoized eagerly,
    /// breaking cycles from loops) and merge each predecessor's end-of-block value, none means
    /// an unreachable/uninitialized read.
    fn read_variable_at_block_entry(&mut self, key: SlotKey, block: BlockId) -> Sym {
        if let Some(&s) = self.current_def.get(&(key, block.0)) {
            return s;
        }
        let preds = self.preds[block.0 as usize].clone();
        let result = if preds.is_empty() || (preds.len() == 1 && preds[0] == block) {
            let ty = self.promotable[&key];
            self.undef_sym(ty)
        } else if preds.len() == 1 {
            self.value_at_block_end(key, preds[0])
        } else {
            let idx = self.phis.len();
            let ty = self.promotable[&key];
            self.phis.push(PhiNode {
                ty,
                block,
                operands: Vec::new(),
            });
            self.current_def.insert((key, block.0), Sym::Phi(idx));
            let mut operands = Vec::with_capacity(preds.len());
            for p in &preds {
                let v = self.value_at_block_end(key, *p);
                operands.push((*p, v));
            }
            self.phis[idx].operands = operands;
            Sym::Phi(idx)
        };
        self.current_def.insert((key, block.0), result);
        result
    }

    /// Forces resolution of every operand reachable from a kept instruction or a terminator,
    /// in program order. This is what discovers every phi this pass will ever need, so it must
    /// run to completion before trivial-phi elimination.
    fn force_resolve_all(&mut self) {
        let f = self.f;
        for block in &f.blocks {
            for &id in &block.insts {
                if self.is_eliminated(id) {
                    continue;
                }
                for v in operand_refs(&f.insts[id.0 as usize].op) {
                    self.resolve(v);
                }
            }
            if let Some(v) = term_operand(&block.term) {
                self.resolve(v);
            }
        }
    }

    fn build(mut self, repl: &[Option<Sym>]) -> Function {
        let f = self.f;

        let mut new_insts: Vec<Inst> = Vec::new();
        let mut final_old: HashMap<u32, InstId> = HashMap::new();
        let mut final_phi: Vec<Option<InstId>> = vec![None; self.phis.len()];
        let mut undef_ids: Vec<(Ty, InstId)> = Vec::new();
        let mut new_blocks: Vec<Block> = Vec::with_capacity(f.blocks.len());

        let mut phis_by_block: Vec<Vec<usize>> = vec![Vec::new(); f.blocks.len()];
        for (idx, phi) in self.phis.iter().enumerate() {
            if repl[idx].is_none() {
                phis_by_block[phi.block.0 as usize].push(idx);
            }
        }

        // Pass 1: allocate every surviving instruction's final id, in exactly the order it
        // will be printed. Content is filled in pass 2, once every id in the function exists
        // (a phi's back-edge operand may name an instruction from a block visited later here).
        for (bidx, block) in f.blocks.iter().enumerate() {
            let mut ids: Vec<InstId> = Vec::new();

            for &idx in &phis_by_block[bidx] {
                let id = InstId(new_insts.len() as u32);
                new_insts.push(Inst {
                    ty: self.phis[idx].ty,
                    op: Op::Phi(Vec::new()),
                });
                final_phi[idx] = Some(id);
                ids.push(id);
            }

            if bidx == 0 {
                for &ty in &self.undef_tys {
                    let id = InstId(new_insts.len() as u32);
                    new_insts.push(Inst {
                        ty,
                        op: undef_op(ty),
                    });
                    undef_ids.push((ty, id));
                    ids.push(id);
                }
            }

            for &old_id in &block.insts {
                if self.is_eliminated(old_id) {
                    continue;
                }
                let id = InstId(new_insts.len() as u32);
                new_insts.push(Inst {
                    ty: f.insts[old_id.0 as usize].ty,
                    op: Op::ConstInt(0),
                });
                final_old.insert(old_id.0, id);
                ids.push(id);
            }

            new_blocks.push(Block {
                insts: ids,
                term: block.term.clone(),
            });
        }

        // Pass 2: fill in real content now that every id exists.
        for (idx, phi) in self.phis.iter().enumerate() {
            let Some(new_id) = final_phi[idx] else {
                continue;
            };
            let operands = phi
                .operands
                .iter()
                .map(|&(bb, sym)| (bb, to_valref(repl, sym, &final_old, &final_phi, &undef_ids)))
                .collect();
            new_insts[new_id.0 as usize].op = Op::Phi(operands);
        }

        for (bidx, block) in f.blocks.iter().enumerate() {
            for &old_id in &block.insts {
                if self.is_eliminated(old_id) {
                    continue;
                }
                let new_id = final_old[&old_id.0];
                let old_op = f.insts[old_id.0 as usize].op.clone();
                let new_op = map_op(&old_op, |v| {
                    let sym = self.resolve(v);
                    to_valref(repl, sym, &final_old, &final_phi, &undef_ids)
                });
                new_insts[new_id.0 as usize].op = new_op;
            }
            let new_term = map_term(&block.term, |v| {
                let sym = self.resolve(v);
                to_valref(repl, sym, &final_old, &final_phi, &undef_ids)
            });
            new_blocks[bidx].term = new_term;
        }

        Function {
            name: f.name.clone(),
            is_kernel: f.is_kernel,
            params: f.params.clone(),
            ret: f.ret,
            blocks: new_blocks,
            insts: new_insts,
        }
    }
}
