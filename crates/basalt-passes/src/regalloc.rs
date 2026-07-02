// Target-independent SSA linear-scan register allocation (Poletto-Sarkar lineage): assigns
// every SSA value in a function to a single fixed location — an abstract register index
// within its class, or a spill slot — for the value's entire lifetime.
//
// Algorithm, in order:
//   1. Build the CFG (successors from each block's `Term`, predecessors by inversion).
//   2. Compute real per-block live-in/live-out sets over SSA values via the standard
//      backward dataflow (`live_out[B] = union of live_in[succ]`, `live_in[B] = uses[B] ∪
//      (live_out[B] - defs[B])`), iterated to a fixed point. A `phi`'s operand for
//      predecessor edge `P -> B` is attributed to `P`'s use set, not `B`'s: the value is
//      consumed at the end of `P` (this is exactly why BIR's `Phi` stores incoming values
//      keyed by predecessor block — see `ir::Op::Phi` — and it is also how the oracle
//      backend already resolves phis, by having each predecessor write its own incoming
//      value before branching). Attributing it to `B` instead would force spurious
//      liveness to leak backward through every predecessor of a loop header, including
//      predecessors that dominate the loop and can never actually observe a later
//      iteration's value.
//   3. Number every instruction (plus one reserved pseudo-position per block for its
//      terminator) by walking blocks in arena order and instructions in each block's own
//      order. This numbering is only a heuristic scaffold for interval bounds and sort
//      order — the real liveness facts come from step 2, not from this linearization, and
//      it must never be read as true execution order.
//   4. Build one conservative `[start, end]` interval per value from the block-level facts:
//      the interval always covers the value's own def position and every position where it
//      is used directly as an operand (a `phi` operand for edge `P -> B` counts as a direct
//      use at `P`'s reserved terminator position, whether or not that edge is also an
//      upward-exposed use per step 2); it is additionally widened to cover a block's first
//      position wherever the value is live-in, and a block's last position wherever it is
//      live-out.
//   5. Sort intervals by start (ties broken by value identity, for determinism) and run
//      textbook linear scan: an "active" list per register class, sorted by end; expire
//      active intervals whose end precedes the new interval's start; assign a free register
//      of the matching class if one exists; otherwise apply the Poletto-Sarkar spill rule —
//      compare the new interval's end against the active interval (same class) with the
//      furthest end, and spill whichever of the two ends furthest away.
//
// Simplifications, deliberate and documented (not defects — natural follow-ups for later
// passes, none of which this one needs to unblock):
//   - **Single interval per value, no live-range splitting.** A value gets exactly one
//     location for its whole lifetime. This can hold a register slightly longer than
//     strictly necessary across untaken control-flow paths (a conservative `[start, end]`
//     span can't express "live here, dead there, live again later" within one span). Full
//     Braun-Hack-style splitting is not implemented; this is the classical linear-scan
//     algorithm in that lineage, not a reimplementation of the fuller published pipeline.
//   - **One dedicated spill slot per spilled value.** No slot coalescing/reuse across
//     non-overlapping lifetimes. Costs stack space, never correctness.
//   - **Phi-copy insertion is out of scope.** A `phi`'s result gets a location like any
//     other value, but this pass never inserts the copies that would move each
//     predecessor's assigned location into the phi's location before that predecessor's
//     terminator. That is codegen's job (a later pass), analogous to how the oracle backend
//     already resolves phis by having each predecessor write its incoming value into the
//     phi's storage location.
//
// Architecture independence: register counts are plain parameters (`num_int_regs`,
// `num_float_regs`), and assigned register indices are abstract (`0..num_int_regs` /
// `0..num_float_regs`) — this module has no notion of any real ISA's register file or
// naming, by design (a later pass maps abstract indices onto a concrete target).
//
// Type classification: `Ty::Scalar(F16|F32|F64)` is float-class; everything else that can
// carry a value (`Ty::Scalar` of an integer width, `Ty::Ptr`, `Ty::Vec`) is int-class.
// `Ty::Vec` values are treated as an int-class placeholder here deliberately — this pass
// only ever assigns a value *a location*, never emits code, so there is nothing for it to
// get silently wrong about a vector's shape; a backend that cannot place a rank>1 tile in a
// plain register file refuses with its own E-code when it gets there, per this project's
// codegen-refusal contract. `Ty::Void` never reaches here: only instructions with a result
// (`Inst::has_result`) and function parameters are given intervals.

use std::collections::BTreeMap;

use basalt_bir::{BlockId, Function, Op, Scalar, Term, Ty, ValRef};

/// Which abstract register file a value belongs in.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum RegClass {
    Int,
    Float,
}

/// Where one SSA value lives for its entire lifetime.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Location {
    /// An abstract register index within its class (`0..num_{int,float}_regs`).
    Reg(RegClass, u32),
    /// A dedicated spill slot index within its class (`0..num_{int,float}_spills`).
    Spill(RegClass, u32),
}

/// Identity of one SSA value, uniform across function parameters and instruction results.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum ValueId {
    Param(u32),
    Val(u32),
}

impl From<ValRef> for ValueId {
    fn from(v: ValRef) -> Self {
        match v {
            ValRef::Param(p) => ValueId::Param(p),
            ValRef::Val(id) => ValueId::Val(id.0),
        }
    }
}

/// The result of allocating one function: every SSA value's location, plus how many spill
/// slots ended up used in each class (`locations` already tells you which values spilled;
/// these counts are the total footprint a codegen pass needs to reserve).
#[derive(Debug, Clone, PartialEq)]
pub struct Allocation {
    pub locations: BTreeMap<ValueId, Location>,
    pub num_int_spills: u32,
    pub num_float_spills: u32,
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

fn preds_of(f: &Function, succs: &[Vec<BlockId>]) -> Vec<Vec<BlockId>> {
    let mut preds = vec![Vec::new(); f.blocks.len()];
    for (idx, list) in succs.iter().enumerate() {
        let from = BlockId(idx as u32);
        for &into in list {
            let bucket = &mut preds[into.0 as usize];
            if !bucket.contains(&from) {
                bucket.push(from);
            }
        }
    }
    preds
}

/// Every `ValRef` operand of a non-`phi` op, in structural order. `phi` is deliberately
/// excluded — its operands are handled separately, attributed per predecessor edge.
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
        Op::Phi(_) => Vec::new(),
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

fn classify(ty: Ty) -> RegClass {
    match ty {
        Ty::Scalar(Scalar::F16) | Ty::Scalar(Scalar::F32) | Ty::Scalar(Scalar::F64) => {
            RegClass::Float
        }
        _ => RegClass::Int,
    }
}

fn value_ty(f: &Function, v: ValueId) -> Ty {
    match v {
        ValueId::Param(p) => f.params[p as usize],
        ValueId::Val(id) => f.insts[id as usize].ty,
    }
}

/// One value's conservative `[start, end]` live span, in the linearized position numbering.
struct Interval {
    value: ValueId,
    class: RegClass,
    start: usize,
    end: usize,
}

/// Per-block position bounds from the linearization in step 3: `start` is the position of
/// the block's first instruction (or, for an empty block, the position its terminator
/// pseudo-slot occupies), and `term_pos` is the reserved terminator pseudo-position, which
/// doubles as the block's last position (`end`, in interval-building terms).
struct BlockPos {
    start: usize,
    term_pos: usize,
}

fn number_positions(f: &Function) -> (Vec<usize>, Vec<BlockPos>) {
    let mut inst_pos = vec![0usize; f.insts.len()];
    let mut block_pos = Vec::with_capacity(f.blocks.len());
    let mut pos = 0usize;
    for block in &f.blocks {
        let start = pos;
        for &id in &block.insts {
            inst_pos[id.0 as usize] = pos;
            pos += 1;
        }
        let term_pos = pos;
        pos += 1;
        block_pos.push(BlockPos { start, term_pos });
    }
    (inst_pos, block_pos)
}

/// Runs the allocator over one function. `num_int_regs`/`num_float_regs` are the abstract
/// per-class register budgets; every value ends up with exactly one `Location` in
/// `Allocation::locations`.
pub fn allocate(function: &Function, num_int_regs: u32, num_float_regs: u32) -> Allocation {
    let n_blocks = function.blocks.len();
    let succs: Vec<Vec<BlockId>> = function
        .blocks
        .iter()
        .map(|b| successors(&b.term))
        .collect();
    let preds = preds_of(function, &succs);

    let mut inst_block = vec![BlockId(0); function.insts.len()];
    for (bidx, block) in function.blocks.iter().enumerate() {
        for &id in &block.insts {
            inst_block[id.0 as usize] = BlockId(bidx as u32);
        }
    }
    let def_block = |v: ValueId| -> BlockId {
        match v {
            ValueId::Param(_) => BlockId(0),
            ValueId::Val(id) => inst_block[id as usize],
        }
    };

    let (inst_pos, block_pos) = number_positions(function);

    // Step 2 setup: per-block def/use sets. A `phi` operand for predecessor edge `P -> B`
    // is attributed to `uses[P]` (only when not already locally available in `P`), never to
    // `uses[B]` — see the module header for why.
    let mut defs: Vec<Vec<ValueId>> = vec![Vec::new(); n_blocks];
    let mut uses: Vec<Vec<ValueId>> = vec![Vec::new(); n_blocks];

    for p in 0..function.params.len() as u32 {
        defs[0].push(ValueId::Param(p));
    }
    for (bidx, block) in function.blocks.iter().enumerate() {
        for &id in &block.insts {
            let inst = &function.insts[id.0 as usize];
            if inst.has_result() {
                defs[bidx].push(ValueId::Val(id.0));
            }
            match &inst.op {
                Op::Phi(incoming) => {
                    for &(pred, val) in incoming {
                        let vid = ValueId::from(val);
                        if def_block(vid) != pred {
                            uses[pred.0 as usize].push(vid);
                        }
                    }
                }
                other => {
                    for v in operand_refs(other) {
                        let vid = ValueId::from(v);
                        if def_block(vid) != BlockId(bidx as u32) {
                            uses[bidx].push(vid);
                        }
                    }
                }
            }
        }
        if let Some(v) = term_operand(&block.term) {
            let vid = ValueId::from(v);
            if def_block(vid) != BlockId(bidx as u32) {
                uses[bidx].push(vid);
            }
        }
    }

    // Step 2: backward fixed-point dataflow over `BTreeSet`s (deterministic iteration,
    // though only set membership/equality is ever observed here).
    use std::collections::BTreeSet;
    let mut live_in: Vec<BTreeSet<ValueId>> = vec![BTreeSet::new(); n_blocks];
    let mut live_out: Vec<BTreeSet<ValueId>> = vec![BTreeSet::new(); n_blocks];
    loop {
        let mut changed = false;
        for b in 0..n_blocks {
            let mut new_out = BTreeSet::new();
            for &s in &succs[b] {
                new_out.extend(live_in[s.0 as usize].iter().copied());
            }
            let mut new_in: BTreeSet<ValueId> = uses[b].iter().copied().collect();
            for v in &new_out {
                if !defs[b].contains(v) {
                    new_in.insert(*v);
                }
            }
            if new_in != live_in[b] || new_out != live_out[b] {
                changed = true;
            }
            live_in[b] = new_in;
            live_out[b] = new_out;
        }
        if !changed {
            break;
        }
    }
    let _ = preds; // CFG predecessors aren't needed once live-in/live-out are in hand.

    // Step 4: direct-use positions (always recorded, regardless of same-block/cross-block —
    // see the module header on why this differs from the `uses[B]` exclusion above).
    let mut max_use_pos: BTreeMap<ValueId, usize> = BTreeMap::new();
    let record_use = |m: &mut BTreeMap<ValueId, usize>, v: ValueId, pos: usize| {
        m.entry(v).and_modify(|p| *p = (*p).max(pos)).or_insert(pos);
    };
    for (bidx, block) in function.blocks.iter().enumerate() {
        for &id in &block.insts {
            let inst = &function.insts[id.0 as usize];
            match &inst.op {
                Op::Phi(incoming) => {
                    for &(pred, val) in incoming {
                        record_use(
                            &mut max_use_pos,
                            ValueId::from(val),
                            block_pos[pred.0 as usize].term_pos,
                        );
                    }
                }
                other => {
                    for v in operand_refs(other) {
                        record_use(&mut max_use_pos, ValueId::from(v), inst_pos[id.0 as usize]);
                    }
                }
            }
        }
        if let Some(v) = term_operand(&block.term) {
            record_use(&mut max_use_pos, ValueId::from(v), block_pos[bidx].term_pos);
        }
    }

    // Every value needing a location: function parameters, plus every instruction result.
    let mut values: Vec<ValueId> = (0..function.params.len() as u32)
        .map(ValueId::Param)
        .collect();
    for (idx, inst) in function.insts.iter().enumerate() {
        if inst.has_result() {
            values.push(ValueId::Val(idx as u32));
        }
    }

    let mut intervals: Vec<Interval> = Vec::with_capacity(values.len());
    for v in values {
        let def_pos = match v {
            ValueId::Param(_) => 0,
            ValueId::Val(id) => inst_pos[id as usize],
        };
        let mut start = def_pos;
        let mut end = max_use_pos.get(&v).copied().unwrap_or(def_pos).max(def_pos);
        for b in 0..n_blocks {
            if live_in[b].contains(&v) {
                start = start.min(block_pos[b].start);
                end = end.max(block_pos[b].start);
            }
            if live_out[b].contains(&v) {
                start = start.min(block_pos[b].term_pos);
                end = end.max(block_pos[b].term_pos);
            }
        }
        intervals.push(Interval {
            value: v,
            class: classify(value_ty(function, v)),
            start,
            end,
        });
    }

    // Step 5: sort by start, tie-broken by value identity for determinism.
    intervals.sort_by_key(|iv| (iv.start, iv.value));

    allocate_linear_scan(intervals, num_int_regs, num_float_regs)
}

/// Per-class bookkeeping during the scan: which value (if any) currently owns each abstract
/// register, and the active list of intervals currently holding one, kept sorted by end.
struct ClassState {
    reg_owner: Vec<Option<ValueId>>,
    active: Vec<(usize, ValueId, u32)>, // (end, value, reg)
    next_spill: u32,
}

impl ClassState {
    fn new(num_regs: u32) -> Self {
        ClassState {
            reg_owner: vec![None; num_regs as usize],
            active: Vec::new(),
            next_spill: 0,
        }
    }

    fn expire_before(&mut self, start: usize) {
        self.active.retain(|&(end, _, reg)| {
            let keep = end >= start;
            if !keep {
                self.reg_owner[reg as usize] = None;
            }
            keep
        });
    }

    fn free_reg(&self) -> Option<u32> {
        self.reg_owner
            .iter()
            .position(|slot| slot.is_none())
            .map(|i| i as u32)
    }

    fn insert_active(&mut self, end: usize, value: ValueId, reg: u32) {
        self.active.push((end, value, reg));
        self.active.sort();
    }

    fn furthest(&self) -> Option<(usize, ValueId, u32)> {
        self.active.last().copied()
    }
}

fn allocate_linear_scan(
    intervals: Vec<Interval>,
    num_int_regs: u32,
    num_float_regs: u32,
) -> Allocation {
    let mut locations: BTreeMap<ValueId, Location> = BTreeMap::new();
    let mut int_state = ClassState::new(num_int_regs);
    let mut float_state = ClassState::new(num_float_regs);

    for iv in intervals {
        let state = match iv.class {
            RegClass::Int => &mut int_state,
            RegClass::Float => &mut float_state,
        };
        state.expire_before(iv.start);

        if let Some(reg) = state.free_reg() {
            state.reg_owner[reg as usize] = Some(iv.value);
            state.insert_active(iv.end, iv.value, reg);
            locations.insert(iv.value, Location::Reg(iv.class, reg));
            continue;
        }

        match state.furthest() {
            Some((furthest_end, furthest_value, reg)) if furthest_end > iv.end => {
                // Evict the active interval that ends furthest away; the new interval takes
                // its register instead.
                state.active.pop();
                state.reg_owner[reg as usize] = Some(iv.value);
                state.insert_active(iv.end, iv.value, reg);
                let slot = state.next_spill;
                state.next_spill += 1;
                locations.insert(furthest_value, Location::Spill(iv.class, slot));
                locations.insert(iv.value, Location::Reg(iv.class, reg));
            }
            _ => {
                // Either no active interval exists (zero registers in this class) or the new
                // interval itself ends furthest away: it is the one that spills.
                let slot = state.next_spill;
                state.next_spill += 1;
                locations.insert(iv.value, Location::Spill(iv.class, slot));
            }
        }
    }

    Allocation {
        locations,
        num_int_spills: int_state.next_spill,
        num_float_spills: float_state.next_spill,
    }
}
