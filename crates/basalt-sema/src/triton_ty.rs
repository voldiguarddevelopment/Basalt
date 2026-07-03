// The tile-shape type representation for Triton kernels — distinct from `ty.rs`'s `Ty`
// (which is CUDA-C's checker type, built on `basalt_frontend_c::ast::ScalarKind`, and has no
// notion of a tile at all). A Triton kernel's values are tiles: rank-0 scalars, rank-1 blocks
// of `N` elements, or rank-2 `M`-by-`N` blocks (real Triton allows higher ranks; this project's
// scope, matching every other task's incremental-first discipline and `TASKS.md`'s own
// wording, stops at rank 2).
//
// # How concrete does a dimension size need to be?
//
// A dimension's extent (`Dim`) is either a concrete integer (`Const`) or a name traceable back
// to a `constexpr` kernel parameter or local (`Symbolic`) — never a value this pass had to
// guess at. This is a deliberate scope line, not laziness: a kernel definition, on its own, is
// not a launch. `BLOCK_SIZE: tl.constexpr` names a value that is only known once a caller
// specializes the kernel for a particular launch; this pass runs once, over the kernel
// definition itself. Two things justify carrying the symbolic name forward instead of
// demanding every dimension resolve to a literal:
//
//   - `TASKS.md`'s own exit criterion for the matmul path (P10-T4) is a *runtime* K-loop —
//     i.e. the pipeline this pass feeds is expected to work even when a dimension's extent is
//     never unrolled to a compile-time constant. Demanding full concrete resolution here would
//     make this pass strictly more demanding than the pipeline it feeds actually needs.
//   - `basalt_bir::Op::Mma` does need concrete `m`/`n`/`k: u32` operands (see `basalt-bir`'s
//     `ir.rs`) — but that requirement belongs to P10-T3's lowering decision of *whether* a
//     given `tl.dot` can take the `mma` fast path (all three dims concretely resolved) or must
//     fall back to a scalar/runtime loop (some dimension stayed symbolic). This pass's job
//     ends at hosting that decision honestly: resolve to `Const` wherever the source actually
//     provides a literal (a bare `tl.arange(0, 128)`, a constexpr param's literal default), and
//     otherwise keep the symbolic name rather than inventing a number.
//
// This pass's actual contract, then, is **rank consistency and broadcast-compatibility**,
// with concrete-dimension resolution as a bonus when the source happens to make it available
// — not full symbolic algebra (no simplification, no proving two differently-named symbolic
// dims equal) and not mandatory literal evaluation.
//
// # Element kind
//
// `Elem` is deliberately coarse (`Int`/`Float`/`Bool`/`Unknown`) — this pass has no way to
// distinguish an ordinary pointer-shaped kernel parameter from a scalar one purely from the
// AST (Triton kernel signatures are routinely unannotated: `def k(x_ptr, n, BLOCK: tl.constexpr)`
// carries no type on `x_ptr` or `n` at all), and dtype precision is not this pass's job — rank
// and broadcasting are. `Elem::Unknown` is this type system's `ty::Ty::Unknown` counterpart:
// compatible with everything, never itself the cause of a diagnostic. An ordinary kernel
// parameter, and anything loaded through `tl.load` (whose pointee dtype this pass does not
// track), gets `Elem::Unknown`.

use std::fmt;

/// A tile's element kind, tracked only as granularly as rank/broadcast checking needs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Elem {
    Int,
    Float,
    Bool,
    /// Dtype not tracked (an unannotated kernel parameter, a `tl.load` result, ...). Always
    /// compatible; never itself reported as a mismatch.
    Unknown,
}

impl Elem {
    /// The element kind of a value built from two others (an elementwise binary op, a
    /// broadcast, ...): identical kinds are kept, anything else falls back to `Unknown` rather
    /// than guessing which side "wins" — this pass does not check arithmetic-operand dtypes,
    /// only shapes, so nothing downstream depends on this being precise.
    fn combine(self, other: Elem) -> Elem {
        if self == other {
            self
        } else {
            Elem::Unknown
        }
    }
}

/// A tile dimension's extent.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Dim {
    /// A concretely known size, resolved from a literal somewhere in the source (an integer
    /// literal directly, or a `constexpr` parameter/local whose value the source happens to
    /// provide).
    Const(i64),
    /// A size traceable back to a `constexpr` name (a parameter with no resolvable literal
    /// value, or a small arithmetic expression over one), carried forward symbolically rather
    /// than guessed at. The string is a rendering of the source expression (e.g. `"BLOCK"`,
    /// `"BLOCK * 2"`), used only for display and for the conservative equality `broadcast_dim`
    /// applies — not a handle into any real symbolic-algebra system.
    Symbolic(String),
}

impl fmt::Display for Dim {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Dim::Const(v) => write!(f, "{v}"),
            Dim::Symbolic(s) => write!(f, "{s}"),
        }
    }
}

/// A Triton tile's shape type: rank 0 (scalar), 1, or 2, matching this task's documented
/// scope. `Unknown` is this type's own recovery placeholder (a construct genuinely out of
/// scope, or an earlier error) — compatible with everything, propagated rather than
/// re-diagnosed, mirroring `ty::Ty::Unknown`'s role in the CUDA-C checker.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TileTy {
    Scalar(Elem),
    Rank1(Dim),
    Rank2(Dim, Dim),
    Unknown,
}

impl TileTy {
    pub fn rank(&self) -> Option<u32> {
        match self {
            TileTy::Scalar(_) => Some(0),
            TileTy::Rank1(_) => Some(1),
            TileTy::Rank2(_, _) => Some(2),
            TileTy::Unknown => None,
        }
    }

    pub fn is_unknown(&self) -> bool {
        matches!(self, TileTy::Unknown)
    }

    /// This value's element kind, ignoring shape.
    pub fn elem(&self) -> Elem {
        match self {
            TileTy::Scalar(e) => *e,
            TileTy::Rank1(_) | TileTy::Rank2(_, _) => Elem::Unknown,
            TileTy::Unknown => Elem::Unknown,
        }
    }

    /// This value's shape as a list of dimensions (`[]` for a scalar), or `None` if the shape
    /// itself is unknown (a recovery case, not a genuine rank-0 shape).
    fn dims(&self) -> Option<Vec<Dim>> {
        match self {
            TileTy::Scalar(_) => Some(Vec::new()),
            TileTy::Rank1(d) => Some(vec![d.clone()]),
            TileTy::Rank2(d0, d1) => Some(vec![d0.clone(), d1.clone()]),
            TileTy::Unknown => None,
        }
    }

    fn from_dims(dims: Vec<Dim>, elem: Elem) -> TileTy {
        match dims.len() {
            0 => TileTy::Scalar(elem),
            1 => TileTy::Rank1(dims.into_iter().next().unwrap()),
            2 => {
                let mut it = dims.into_iter();
                let d0 = it.next().unwrap();
                let d1 = it.next().unwrap();
                TileTy::Rank2(d0, d1)
            }
            _ => unreachable!("TileTy has no rank above 2"),
        }
    }
}

impl fmt::Display for TileTy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TileTy::Scalar(_) => write!(f, "scalar"),
            TileTy::Rank1(d) => write!(f, "[{d}]"),
            TileTy::Rank2(d0, d1) => write!(f, "[{d0}, {d1}]"),
            TileTy::Unknown => write!(f, "<unknown>"),
        }
    }
}

/// Broadcasts two same-axis dimensions, NumPy-style: equal sizes match, a size-1 axis yields
/// to the other side's size. A concrete size can never be proven compatible with a differently
/// spelled symbolic one (or a different concrete size) at this stage — this pass has no
/// launch-time specialization to consult, so it refuses rather than guessing which of two
/// unequal-looking dims might turn out equal at runtime. Two symbolic dims are considered
/// equal only when their rendered text matches exactly (no algebraic simplification — `BLOCK`
/// and `0 + BLOCK` are not recognized as the same dimension).
fn broadcast_dim(a: &Dim, b: &Dim) -> Result<Dim, ()> {
    match (a, b) {
        (Dim::Const(1), other) | (other, Dim::Const(1)) => Ok(other.clone()),
        (Dim::Const(x), Dim::Const(y)) => {
            if x == y {
                Ok(Dim::Const(*x))
            } else {
                Err(())
            }
        }
        (Dim::Symbolic(x), Dim::Symbolic(y)) => {
            if x == y {
                Ok(Dim::Symbolic(x.clone()))
            } else {
                Err(())
            }
        }
        (Dim::Const(_), Dim::Symbolic(_)) | (Dim::Symbolic(_), Dim::Const(_)) => Err(()),
    }
}

/// Broadcasts two tile shapes the way a Triton elementwise op does: unlike NumPy, Triton does
/// not implicitly left-pad a lower-rank tensor's shape with size-1 axes — two tile-ranked
/// operands (rank 1 or 2) must already share a rank, and a genuine rank mismatch between them
/// is refused rather than reconciled by guessing which axis the shorter shape meant. A rank-0
/// scalar is the one exception (as in real Triton, and as a plain Python `int`/`float` literal
/// used in an expression already is): it broadcasts freely against any rank. `TileTy::Unknown`
/// on either side propagates as `Unknown` without a diagnostic, matching `ty::Ty::Unknown`'s
/// "already reported, do not pile on" role in the CUDA-C checker.
pub fn broadcast(a: &TileTy, b: &TileTy) -> Result<TileTy, ()> {
    if a.is_unknown() || b.is_unknown() {
        return Ok(TileTy::Unknown);
    }
    let elem = a.elem().combine(b.elem());
    let (da, db) = (a.dims().unwrap(), b.dims().unwrap());
    if da.is_empty() {
        return Ok(TileTy::from_dims(db, elem));
    }
    if db.is_empty() {
        return Ok(TileTy::from_dims(da, elem));
    }
    if da.len() != db.len() {
        return Err(());
    }
    let mut out = Vec::with_capacity(da.len());
    for (x, y) in da.iter().zip(db.iter()) {
        out.push(broadcast_dim(x, y)?);
    }
    Ok(TileTy::from_dims(out, elem))
}

/// One axis-insertion op parsed out of a `[:, None]`/`[None, :]`-style subscript: either an
/// existing axis carried through unchanged (a bare `:`), or a new size-1 axis inserted at this
/// position (a `None`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReshapeStep {
    Keep,
    Insert,
}

/// Applies a `[:, None]`/`[None, :]`-style reshape (already parsed into `steps`, one per index
/// component, in source order) to `base`. The number of `Keep` steps must equal `base`'s own
/// rank (each corresponds to one of `base`'s existing axes, in order) and the resulting rank
/// (`Keep` count plus `Insert` count) must stay within this task's rank-0/1/2 scope — both are
/// refused with `Err` rather than silently truncated or padded, leaving the caller to attach
/// the right E-code.
pub fn reshape(base: &TileTy, steps: &[ReshapeStep]) -> Result<TileTy, ()> {
    if base.is_unknown() {
        return Ok(TileTy::Unknown);
    }
    let base_dims = base.dims().unwrap();
    let keep_count = steps.iter().filter(|s| **s == ReshapeStep::Keep).count();
    if keep_count != base_dims.len() {
        return Err(());
    }
    if steps.len() > 2 {
        return Err(());
    }
    let mut base_iter = base_dims.into_iter();
    let mut out = Vec::with_capacity(steps.len());
    for step in steps {
        match step {
            ReshapeStep::Keep => out.push(base_iter.next().unwrap()),
            ReshapeStep::Insert => out.push(Dim::Const(1)),
        }
    }
    Ok(TileTy::from_dims(out, base.elem()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scalar_broadcasts_freely_against_any_rank() {
        let s = TileTy::Scalar(Elem::Int);
        let r1 = TileTy::Rank1(Dim::Const(4));
        assert_eq!(broadcast(&s, &r1), Ok(TileTy::Rank1(Dim::Const(4))));
        assert_eq!(broadcast(&r1, &s), Ok(TileTy::Rank1(Dim::Const(4))));
    }

    #[test]
    fn same_rank_equal_dims_match() {
        let a = TileTy::Rank1(Dim::Const(8));
        let b = TileTy::Rank1(Dim::Const(8));
        assert_eq!(broadcast(&a, &b), Ok(TileTy::Rank1(Dim::Const(8))));
    }

    #[test]
    fn rank_mismatch_between_two_tile_ranks_is_refused() {
        let a = TileTy::Rank1(Dim::Const(8));
        let b = TileTy::Rank2(Dim::Const(8), Dim::Const(8));
        assert!(broadcast(&a, &b).is_err());
    }

    #[test]
    fn unequal_concrete_dims_are_refused() {
        let a = TileTy::Rank1(Dim::Const(8));
        let b = TileTy::Rank1(Dim::Const(16));
        assert!(broadcast(&a, &b).is_err());
    }

    #[test]
    fn size_one_axis_broadcasts_against_a_symbolic_dim() {
        let a = TileTy::Rank2(Dim::Const(1), Dim::Symbolic("N".to_string()));
        let b = TileTy::Rank2(Dim::Symbolic("M".to_string()), Dim::Const(1));
        assert_eq!(
            broadcast(&a, &b),
            Ok(TileTy::Rank2(
                Dim::Symbolic("M".to_string()),
                Dim::Symbolic("N".to_string())
            ))
        );
    }

    #[test]
    fn concrete_vs_differently_named_symbolic_is_refused_not_guessed() {
        let a = TileTy::Rank1(Dim::Const(128));
        let b = TileTy::Rank1(Dim::Symbolic("BLOCK".to_string()));
        assert!(broadcast(&a, &b).is_err());
    }

    #[test]
    fn reshape_inserts_axis_for_matmul_style_tile_construction() {
        let row = TileTy::Rank1(Dim::Symbolic("BLOCK_M".to_string()));
        let reshaped = reshape(&row, &[ReshapeStep::Keep, ReshapeStep::Insert]).unwrap();
        assert_eq!(
            reshaped,
            TileTy::Rank2(Dim::Symbolic("BLOCK_M".to_string()), Dim::Const(1))
        );
    }

    #[test]
    fn reshape_beyond_rank_two_is_refused() {
        let row = TileTy::Rank1(Dim::Const(4));
        let steps = [ReshapeStep::Insert, ReshapeStep::Keep, ReshapeStep::Insert];
        assert!(reshape(&row, &steps).is_err());
    }
}
