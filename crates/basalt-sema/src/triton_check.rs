// Tile-shape inference over a `basalt_frontend_triton::ast::Module` — the sema-barrier task
// this crate's own header calls out ("Shape inference (Triton tiles) ... is later work
// layered on top of this"). Walks every kernel's body and assigns a `TileTy` (see
// `triton_ty.rs` for the type itself and the scoping rationale) to every expression, the same
// "report many, never stop at the first" standard `checker.rs` uses for the CUDA-C side.
//
// Public entry point: `check_triton(&Module) -> (Vec<KernelShapes>, Vec<Diag>)`. Each
// `KernelShapes` carries the kernel's parameter types plus a span-keyed map of every
// expression's inferred `TileTy` — a real, walkable, queryable representation P10-T3 can
// consult by re-walking the same AST and looking up `expr.span()`, without this pass having
// to hand back a second parallel shadow-AST duplicating every `Expr`/`Stmt` variant.
//
// # What is (and isn't) modeled
//
// Recognized shape-relevant builtins, dispatched on the final attribute name of the callee
// (e.g. `tl.load` and `some_alias.load` are both recognized as `load` — this pass cannot
// verify the base resolves to the real `triton.language` module without import tracking, the
// same limitation `basalt-frontend-triton`'s own `is_triton_jit` already accepted for the
// `@triton.jit` decorator itself, and no real kernel defines an unrelated same-named method to
// collide with one of these):
//   - `tl.program_id(axis)` -> rank-0 `Int`.
//   - `tl.arange(lo, hi)` -> rank-1 `[hi - lo]`, `lo`/`hi` resolved through `eval_dim_expr`.
//   - `tl.load(ptr, mask=..., other=...)` -> the pointer argument's own shape; `mask` (if
//     given) must be broadcast-compatible with it.
//   - `tl.store(ptr, value, mask=...)` -> `value` must be broadcast-compatible with `ptr`,
//     `mask` (if given) must be broadcast-compatible with `ptr`; result is a bare rank-0 value
//     never meant to be used further (`tl.store` returns nothing at runtime either).
//   - `tl.zeros(shape, dtype=...)` -> a literal shape tuple/scalar resolved through
//     `eval_dim_expr`, entry by entry.
//   - `tl.dot(a, b, ...)` -> deliberately `TileTy::Unknown`. Real matmul-shape inference (and
//     the decision of whether a given call can lower to BIR's `mma` versus a scalar/runtime
//     loop) is P10-T3's job, which needs concrete `m`/`n`/`k`; this pass only has to let the
//     tile machinery *building up to* a `tl.dot` call (the 2-D index-tile construction) work,
//     per this task's own stated scope. Arguments are still walked so an error inside them is
//     still reported.
//   - Anything else callable (`range()` in a `for` loop's `iter`, an elementwise math
//     intrinsic like `tl.exp`, an unrecognized `tl.*` function, ...) is walked for recovery
//     but never guessed at: `TileTy::Unknown`, no diagnostic. `Unknown` is not a guess — it is
//     this pass's explicit "don't know," the same role `ty::Ty::Unknown` plays for the CUDA-C
//     checker.
//
// `x[:, None]` / `x[None, :]` (and the identity/insert-only single-index forms `x[:]`,
// `x[None]`) reshape through `triton_ty::reshape`. Any other subscript form — a partial slice
// (`x[0:BLOCK]`), an integer index (`x[0]`), anything producing rank > 2 — is refused outright
// (`E307`) rather than silently reinterpreted as a full-tile identity or truncated to fit.
//
// The `@` operator (Python's matmul operator, distinct from `tl.dot`) is refused the same way:
// broadcasting it elementwise would silently produce the wrong shape, and this pass has no
// matmul-shape rule to apply to it directly (that machinery belongs to `tl.dot`/P10-T3).
//
// A bare-name assignment target rebinds that name's tracked shape (Python has no block
// scoping, and neither does a Triton kernel body); a `constexpr`-annotated `AnnAssign` also
// updates the compile-time-dimension table (`Checker::const_vals`), resolving to a concrete
// `Dim::Const` when the initializer is itself resolvable and staying `Dim::Symbolic` (named
// after the local) otherwise. A `str`-literal statement (a docstring) and a bare `None`/module
// reference intentionally type to `Unknown` without a diagnostic — they are not tile values,
// and are common enough in a real kernel body that flagging them would just be noise.
//
// Undefined-name references reuse `ECode::UndefinedSymbol` (`E301`) — the same code the
// CUDA-C checker already uses for exactly this condition; there is no Triton-specific flavor
// of "this name is not bound" worth a new code for.

use std::collections::{BTreeMap, HashMap};

use basalt_diag::{Diag, ECode, Span};
use basalt_frontend_triton::ast::{
    is_constexpr_annotation, BinOp, Expr, KernelFn, Keyword, Module, Param, Stmt, UnaryOp,
};

use crate::triton_ty::{broadcast, reshape, Dim, Elem, ReshapeStep, TileTy};

/// One kernel's inferred shapes: its parameters (in declaration order) plus every expression
/// in its body, keyed by span. `BTreeMap` (rather than `HashMap`) purely so this crate's own
/// output is itself iteration-order-stable, matching the project's general aversion to
/// hashmap-ordered output — nothing here currently iterates the map in a way that would
/// produce diverging results either way, but there is no reason to leave that to chance.
#[derive(Debug, Clone, PartialEq)]
pub struct KernelShapes {
    pub name: String,
    pub param_types: Vec<TileTy>,
    pub expr_types: BTreeMap<Span, TileTy>,
}

/// Infers tile shapes for every kernel in `module`, returning one `KernelShapes` per kernel
/// (in `module.kernels`'s own order) plus every diagnostic collected along the way. Never
/// stops at the first problem — matches `checker::check`'s own contract.
pub fn check_triton(module: &Module) -> (Vec<KernelShapes>, Vec<Diag>) {
    let mut diags = Vec::new();
    let shapes = module
        .kernels
        .iter()
        .map(|k| check_kernel(k, &mut diags))
        .collect();
    (shapes, diags)
}

fn check_kernel(k: &KernelFn, diags: &mut Vec<Diag>) -> KernelShapes {
    let mut ck = Checker {
        diags,
        vars: HashMap::new(),
        const_vals: HashMap::new(),
        expr_types: BTreeMap::new(),
    };
    let param_types = k.params.iter().map(|p| ck.seed_param(p)).collect();
    for s in &k.body {
        ck.check_stmt(s);
    }
    KernelShapes {
        name: k.name.clone(),
        param_types,
        expr_types: ck.expr_types,
    }
}

struct Checker<'a> {
    diags: &'a mut Vec<Diag>,
    /// Every name currently bound in the kernel body (parameters and locals share one flat
    /// namespace — a Triton kernel body has no nested block scoping to model).
    vars: HashMap<String, TileTy>,
    /// The compile-time-dimension table: names known to trace back to a `constexpr`
    /// parameter or local, with whatever `Dim` this pass could resolve for them.
    const_vals: HashMap<String, Dim>,
    expr_types: BTreeMap<Span, TileTy>,
}

impl<'a> Checker<'a> {
    fn diag(&mut self, code: ECode, span: Span, args: impl IntoIterator<Item = String>) {
        self.diags
            .push(Diag::new(code).with_span(span).with_args(args));
    }

    /// Seeds one kernel parameter's tracked shape (and, for a `constexpr` parameter, its
    /// compile-time-dimension entry) into a fresh checker before the body is walked.
    fn seed_param(&mut self, p: &Param) -> TileTy {
        if p.is_constexpr {
            // A `constexpr` parameter's real value is only known once a caller specializes
            // this kernel for a launch — this pass runs over the definition alone. A literal
            // default (`BLOCK: tl.constexpr = 128`, a real and common Triton pattern) is the
            // one place a kernel definition can honestly provide a concrete value; anything
            // else stays symbolic rather than being guessed at.
            let dim = p
                .default
                .as_ref()
                .and_then(|d| self.eval_dim_expr(d))
                .unwrap_or_else(|| Dim::Symbolic(p.name.clone()));
            self.const_vals.insert(p.name.clone(), dim);
            let ty = TileTy::Scalar(Elem::Int);
            self.vars.insert(p.name.clone(), ty.clone());
            ty
        } else {
            // An ordinary kernel parameter (pointer or runtime scalar) starts life as a
            // rank-0 value of untracked element kind — Triton kernel signatures are routinely
            // unannotated, so there is no honest way to tell a pointer parameter from a plain
            // scalar one from the AST alone (see the module header on `Elem`).
            let ty = TileTy::Scalar(Elem::Unknown);
            self.vars.insert(p.name.clone(), ty.clone());
            ty
        }
    }

    // ---- statements ---------------------------------------------------------------------

    fn check_stmt(&mut self, s: &Stmt) {
        match s {
            Stmt::Expr { value, .. } => {
                self.infer_expr(value);
            }
            Stmt::Assign { targets, value, .. } => {
                let vt = self.infer_expr(value);
                for t in targets {
                    self.bind_target(t, vt.clone());
                }
            }
            Stmt::AugAssign {
                target,
                op,
                value,
                span,
            } => {
                let cur = self.infer_expr(target);
                let vt = self.infer_expr(value);
                let result = self.binop_result(*op, &cur, &vt, *span);
                self.bind_target(target, result);
            }
            Stmt::AnnAssign {
                target,
                annotation,
                value,
                ..
            } => self.check_ann_assign(target, annotation, value.as_ref()),
            Stmt::If {
                test, body, orelse, ..
            } => {
                self.infer_expr(test);
                self.check_stmts(body);
                self.check_stmts(orelse);
            }
            Stmt::For {
                target,
                iter,
                body,
                orelse,
                ..
            } => self.check_for(target, iter, body, orelse),
            Stmt::While {
                test, body, orelse, ..
            } => {
                self.infer_expr(test);
                self.check_stmts(body);
                self.check_stmts(orelse);
            }
            Stmt::Return { value, .. } => {
                if let Some(v) = value {
                    self.infer_expr(v);
                }
            }
            Stmt::Assert { test, msg, .. } => {
                self.infer_expr(test);
                if let Some(m) = msg {
                    self.infer_expr(m);
                }
            }
            Stmt::Pass { .. } | Stmt::Break { .. } | Stmt::Continue { .. } => {}
            // Already reported by the parser (see `basalt-frontend-triton`'s own `Stmt::Error`
            // doc); nothing further for this pass to check.
            Stmt::Error { .. } => {}
        }
    }

    fn check_stmts(&mut self, stmts: &[Stmt]) {
        for s in stmts {
            self.check_stmt(s);
        }
    }

    /// A bare name rebinds its tracked shape (a Triton kernel body has no block scoping, so
    /// there is no shadowing question the way `basalt-frontend-c`'s checker has to answer).
    /// Any other target shape (tuple-unpacking, an attribute/subscript write target — neither
    /// is a real Triton kernel-body pattern) is walked for diagnostics but does not update the
    /// shape environment: this pass cannot decompose a single tile's shape across several
    /// bound names, and refusing to guess beats inventing a per-element shape.
    fn bind_target(&mut self, target: &Expr, ty: TileTy) {
        match target {
            Expr::Name { name, .. } => {
                self.vars.insert(name.clone(), ty);
            }
            Expr::Tuple { elts, .. } => {
                for e in elts {
                    if !matches!(e, Expr::Name { .. }) {
                        self.infer_expr(e);
                    }
                }
            }
            _ => {
                self.infer_expr(target);
            }
        }
    }

    fn check_ann_assign(&mut self, target: &Expr, annotation: &Expr, value: Option<&Expr>) {
        // `annotation` names a triton dtype/`constexpr` marker, not a tile value — it is
        // never walked through `infer_expr` (the same reason a `Call`'s own callee expression
        // isn't: `tl.constexpr`/`tl.float32`-style dotted paths refer to triton's type
        // namespace, and walking them would misreport `tl` as an undefined tile value).
        let Expr::Name { name, .. } = target else {
            if let Some(v) = value {
                self.infer_expr(v);
            }
            return;
        };
        if is_constexpr_annotation(annotation) {
            let dim = value
                .and_then(|v| {
                    self.infer_expr(v);
                    self.eval_dim_expr(v)
                })
                .unwrap_or_else(|| Dim::Symbolic(name.clone()));
            self.const_vals.insert(name.clone(), dim);
            self.vars.insert(name.clone(), TileTy::Scalar(Elem::Int));
        } else {
            let ty = match value {
                Some(v) => self.infer_expr(v),
                None => TileTy::Scalar(Elem::Unknown),
            };
            self.vars.insert(name.clone(), ty);
        }
    }

    /// `for target in iter: body`. `range(...)` (the overwhelming common case — a Triton
    /// kernel body has no other real iteration source) binds `target` to a rank-0 `Int` loop
    /// counter; anything else still gets a best-effort rank-0 `Unknown` binding rather than a
    /// refusal, since a `for` loop's own iteration semantics are not this pass's job to police.
    fn check_for(&mut self, target: &Expr, iter: &Expr, body: &[Stmt], orelse: &[Stmt]) {
        if let Expr::Call { func, args, .. } = iter {
            if matches!(&**func, Expr::Name { name, .. } if name == "range") {
                for a in args {
                    self.infer_expr(a);
                }
                self.bind_target(target, TileTy::Scalar(Elem::Int));
                self.check_stmts(body);
                self.check_stmts(orelse);
                return;
            }
        }
        self.infer_expr(iter);
        self.bind_target(target, TileTy::Scalar(Elem::Unknown));
        self.check_stmts(body);
        self.check_stmts(orelse);
    }

    // ---- expressions ----------------------------------------------------------------------

    fn infer_expr(&mut self, e: &Expr) -> TileTy {
        let ty = self.infer_expr_inner(e);
        self.expr_types.insert(e.span(), ty.clone());
        ty
    }

    fn infer_expr_inner(&mut self, e: &Expr) -> TileTy {
        match e {
            Expr::Name { name, span } => match self.vars.get(name) {
                Some(t) => t.clone(),
                None => {
                    self.diag(ECode::UndefinedSymbol, *span, [name.clone()]);
                    TileTy::Unknown
                }
            },
            Expr::IntLit { .. } => TileTy::Scalar(Elem::Int),
            Expr::FloatLit { .. } => TileTy::Scalar(Elem::Float),
            Expr::BoolLit { .. } => TileTy::Scalar(Elem::Bool),
            // Neither is a tile value on its own: `None` only means anything inside a
            // `[:, None]`-style reshape (handled directly by `infer_subscript`, which never
            // calls back into this function for it), and a bare string is a docstring far
            // more often than a real construct — flagging either would just be noise.
            Expr::NoneLit { .. } | Expr::StrLit { .. } => TileTy::Unknown,
            Expr::BoolOp { values, span, .. } => {
                let mut acc: Option<TileTy> = None;
                for v in values {
                    let vt = self.infer_expr(v);
                    acc = Some(match acc {
                        None => vt,
                        Some(prev) => self.broadcast_or_diag(&prev, &vt, *span),
                    });
                }
                as_bool(acc.unwrap_or(TileTy::Unknown))
            }
            Expr::UnaryOp { op, operand, .. } => {
                let t = self.infer_expr(operand);
                match op {
                    UnaryOp::Not => as_bool(t),
                    UnaryOp::Invert | UnaryOp::UAdd | UnaryOp::USub => t,
                }
            }
            Expr::BinOp { op, lhs, rhs, span } => {
                let l = self.infer_expr(lhs);
                let r = self.infer_expr(rhs);
                self.binop_result(*op, &l, &r, *span)
            }
            Expr::Compare {
                left,
                comparators,
                span,
                ..
            } => {
                let mut acc = self.infer_expr(left);
                for c in comparators {
                    let ct = self.infer_expr(c);
                    acc = self.broadcast_or_diag(&acc, &ct, *span);
                }
                as_bool(acc)
            }
            Expr::Ternary {
                test,
                body,
                orelse,
                span,
            } => {
                self.infer_expr(test);
                let bt = self.infer_expr(body);
                let ot = self.infer_expr(orelse);
                self.broadcast_or_diag(&bt, &ot, *span)
            }
            Expr::Call {
                func,
                args,
                keywords,
                span,
            } => self.infer_call(func, args, keywords, *span),
            // A bare attribute reference (module/namespace access, `tl.float32` used as a
            // `dtype=` argument, ...) is not itself a tile value; `value` is deliberately not
            // walked (see the module header on `Call`'s own callee).
            Expr::Attribute { .. } => TileTy::Unknown,
            Expr::Subscript { value, index, span } => self.infer_subscript(value, index, *span),
            // A slice only ever appears as one component of a reshape subscript, handled
            // directly by `infer_subscript`/`subscript_steps` without reaching this arm in
            // practice; if one somehow does (a slice expression used standalone), there is no
            // tile value to report.
            Expr::Slice { .. } => TileTy::Unknown,
            Expr::Tuple { elts, .. } | Expr::List { elts, .. } => {
                for el in elts {
                    self.infer_expr(el);
                }
                TileTy::Unknown
            }
            Expr::Error { .. } => TileTy::Unknown,
        }
    }

    fn binop_result(&mut self, op: BinOp, l: &TileTy, r: &TileTy, span: Span) -> TileTy {
        if op == BinOp::MatMul {
            // Python's `@` operator, distinct from `tl.dot`: broadcasting it elementwise
            // would silently produce the wrong shape, and the real matmul-shape rule belongs
            // to `tl.dot`/P10-T3, not to this generic binary-op path.
            self.diag(
                ECode::TileConstructUnsupported,
                span,
                ["the '@' operator (use 'tl.dot' for matrix multiply)".to_string()],
            );
            return TileTy::Unknown;
        }
        self.broadcast_or_diag(l, r, span)
    }

    fn broadcast_or_diag(&mut self, a: &TileTy, b: &TileTy, span: Span) -> TileTy {
        match broadcast(a, b) {
            Ok(t) => t,
            Err(()) => {
                self.diag(
                    ECode::TileShapeMismatch,
                    span,
                    [a.to_string(), b.to_string()],
                );
                TileTy::Unknown
            }
        }
    }

    fn infer_subscript(&mut self, value: &Expr, index: &Expr, span: Span) -> TileTy {
        let vt = self.infer_expr(value);
        match subscript_steps(index) {
            Some(steps) => match reshape(&vt, &steps) {
                Ok(t) => t,
                Err(()) => {
                    self.diag(
                        ECode::TileConstructUnsupported,
                        span,
                        [format!(
                            "reshape of a rank-{} tile to rank {} (via '[:, None]'/'[None, :]')",
                            vt.rank().map_or_else(|| "?".to_string(), |r| r.to_string()),
                            steps.len()
                        )],
                    );
                    TileTy::Unknown
                }
            },
            None => {
                self.diag(
                    ECode::TileConstructUnsupported,
                    span,
                    ["a tile subscript other than a '[:, None]'/'[None, :]'-style reshape (partial slicing or integer indexing)".to_string()],
                );
                TileTy::Unknown
            }
        }
    }

    fn infer_call(
        &mut self,
        func: &Expr,
        args: &[Expr],
        keywords: &[Keyword],
        span: Span,
    ) -> TileTy {
        match attr_name(func) {
            Some("program_id") => {
                for a in args {
                    self.infer_expr(a);
                }
                for kw in keywords {
                    self.infer_expr(&kw.value);
                }
                TileTy::Scalar(Elem::Int)
            }
            Some("arange") => self.infer_arange(args, span),
            Some("load") => self.infer_load(args, keywords, span),
            Some("store") => self.infer_store(args, keywords, span),
            Some("zeros") => self.infer_zeros(args, keywords, span),
            Some("dot") => {
                // Deliberately `Unknown` — see the module header. The tile machinery
                // building the arguments still gets checked; the call's own result shape is
                // P10-T3's job once concrete `m`/`n`/`k` matter.
                for a in args {
                    self.infer_expr(a);
                }
                for kw in keywords {
                    self.infer_expr(&kw.value);
                }
                TileTy::Unknown
            }
            _ => {
                for a in args {
                    self.infer_expr(a);
                }
                for kw in keywords {
                    self.infer_expr(&kw.value);
                }
                TileTy::Unknown
            }
        }
    }

    fn infer_arange(&mut self, args: &[Expr], span: Span) -> TileTy {
        if args.len() != 2 {
            self.diag(
                ECode::TileConstructUnsupported,
                span,
                [
                    "'tl.arange' called with other than exactly two positional arguments"
                        .to_string(),
                ],
            );
            for a in args {
                self.infer_expr(a);
            }
            return TileTy::Unknown;
        }
        let lo_dim = self.eval_dim_expr(&args[0]);
        let hi_dim = self.eval_dim_expr(&args[1]);
        self.infer_expr(&args[0]);
        self.infer_expr(&args[1]);
        match (lo_dim, hi_dim) {
            (Some(lo), Some(hi)) => match combine_dim(BinOp::Sub, &hi, &lo) {
                Some(d) => TileTy::Rank1(d),
                None => {
                    self.diag(
                        ECode::TileDimUnresolved,
                        span,
                        ["'tl.arange' bounds overflowed while resolving a size".to_string()],
                    );
                    TileTy::Unknown
                }
            },
            _ => {
                self.diag(
                    ECode::TileDimUnresolved,
                    span,
                    ["'tl.arange' bounds".to_string()],
                );
                TileTy::Unknown
            }
        }
    }

    fn infer_load(&mut self, args: &[Expr], keywords: &[Keyword], span: Span) -> TileTy {
        if args.is_empty() {
            self.diag(
                ECode::TileConstructUnsupported,
                span,
                ["'tl.load' called with no pointer argument".to_string()],
            );
            for kw in keywords {
                self.infer_expr(&kw.value);
            }
            return TileTy::Unknown;
        }
        let ptr_ty = self.infer_expr(&args[0]);
        for a in &args[1..] {
            self.infer_expr(a);
        }
        let mut mask_ty = None;
        for kw in keywords {
            let kt = self.infer_expr(&kw.value);
            if kw.name.as_deref() == Some("mask") {
                mask_ty = Some(kt);
            }
        }
        if let Some(mt) = mask_ty {
            self.broadcast_or_diag(&ptr_ty, &mt, span);
        }
        ptr_ty
    }

    fn infer_store(&mut self, args: &[Expr], keywords: &[Keyword], span: Span) -> TileTy {
        if args.len() < 2 {
            self.diag(
                ECode::TileConstructUnsupported,
                span,
                ["'tl.store' called with fewer than a pointer and a value argument".to_string()],
            );
            for a in args {
                self.infer_expr(a);
            }
            for kw in keywords {
                self.infer_expr(&kw.value);
            }
            return TileTy::Scalar(Elem::Unknown);
        }
        let ptr_ty = self.infer_expr(&args[0]);
        let value_ty = self.infer_expr(&args[1]);
        for a in &args[2..] {
            self.infer_expr(a);
        }
        self.broadcast_or_diag(&ptr_ty, &value_ty, span);
        let mut mask_ty = None;
        for kw in keywords {
            let kt = self.infer_expr(&kw.value);
            if kw.name.as_deref() == Some("mask") {
                mask_ty = Some(kt);
            }
        }
        if let Some(mt) = mask_ty {
            self.broadcast_or_diag(&ptr_ty, &mt, span);
        }
        TileTy::Scalar(Elem::Unknown)
    }

    fn infer_zeros(&mut self, args: &[Expr], keywords: &[Keyword], span: Span) -> TileTy {
        for kw in keywords {
            self.infer_expr(&kw.value);
        }
        let Some(shape_arg) = args.first() else {
            self.diag(
                ECode::TileConstructUnsupported,
                span,
                ["'tl.zeros' called with no shape argument".to_string()],
            );
            return TileTy::Unknown;
        };
        let shape_exprs: Vec<&Expr> = match shape_arg {
            Expr::Tuple { elts, .. } => elts.iter().collect(),
            other => vec![other],
        };
        let dims: Vec<Option<Dim>> = shape_exprs
            .iter()
            .map(|e| {
                let d = self.eval_dim_expr(e);
                self.infer_expr(e);
                d
            })
            .collect();
        if dims.len() > 2 {
            self.diag(
                ECode::TileConstructUnsupported,
                span,
                ["'tl.zeros' with a shape of rank > 2".to_string()],
            );
            return TileTy::Unknown;
        }
        if dims.iter().any(Option::is_none) {
            self.diag(
                ECode::TileDimUnresolved,
                span,
                ["'tl.zeros' shape".to_string()],
            );
            return TileTy::Unknown;
        }
        let dims: Vec<Dim> = dims.into_iter().map(Option::unwrap).collect();
        match dims.len() {
            0 => TileTy::Scalar(Elem::Unknown),
            1 => TileTy::Rank1(dims[0].clone()),
            2 => TileTy::Rank2(dims[0].clone(), dims[1].clone()),
            _ => unreachable!("checked above"),
        }
    }

    /// Resolves an expression used in a dimension-producing position (`tl.arange`'s bounds, a
    /// `tl.zeros` shape entry, a `constexpr` local's initializer) to a `Dim`. Returns `None`
    /// for anything this pass cannot make sense of as a compile-time size — a genuine runtime
    /// value (a loaded tensor element, a non-`constexpr` parameter) — rather than guessing.
    /// Never reports a diagnostic itself: callers decide whether a `None` here is actually an
    /// error (it is for `tl.arange`/`tl.zeros`; it is not for a `constexpr` parameter's
    /// optional default, which falls back to a symbolic name instead).
    fn eval_dim_expr(&self, e: &Expr) -> Option<Dim> {
        match e {
            Expr::IntLit { text, .. } => parse_int_text(text).map(Dim::Const),
            Expr::Name { name, .. } => self.const_vals.get(name).cloned(),
            Expr::UnaryOp {
                op: UnaryOp::USub,
                operand,
                ..
            } => match self.eval_dim_expr(operand)? {
                Dim::Const(v) => v.checked_neg().map(Dim::Const),
                Dim::Symbolic(s) => Some(Dim::Symbolic(format!("-{s}"))),
            },
            Expr::UnaryOp {
                op: UnaryOp::UAdd,
                operand,
                ..
            } => self.eval_dim_expr(operand),
            Expr::BinOp { op, lhs, rhs, .. } => {
                let l = self.eval_dim_expr(lhs)?;
                let r = self.eval_dim_expr(rhs)?;
                combine_dim(*op, &l, &r)
            }
            _ => None,
        }
    }
}

/// The tile-shape-relevant final attribute name of a call's callee expression (`tl.load` and
/// `some_alias.load` both give `Some("load")`; a bare `Name` — e.g. `range` — gives `None`).
/// See the module header for why matching on the final component alone, without verifying the
/// base resolves to the real `triton.language` module, is an accepted simplification here.
fn attr_name(func: &Expr) -> Option<&str> {
    match func {
        Expr::Attribute { attr, .. } => Some(attr.as_str()),
        _ => None,
    }
}

/// One reshape-subscript index parsed into `[ReshapeStep]`, or `None` if `index` is not built
/// entirely out of bare `:` slices and `None` literals (a partial slice, an integer index,
/// ...), which this pass refuses rather than reinterprets.
fn subscript_steps(index: &Expr) -> Option<Vec<ReshapeStep>> {
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

fn as_bool(t: TileTy) -> TileTy {
    match t {
        TileTy::Scalar(_) => TileTy::Scalar(Elem::Bool),
        other => other,
    }
}

/// Constant-folds two resolved dimensions under a dimension-arithmetic operator, or (if either
/// side is symbolic) renders a combined symbolic name. `None` for an operator that has no
/// sensible meaning in a dimension-size position (a bitwise op, `%`, `**`, ...) or for integer
/// overflow while folding two concrete sizes — both are "cannot resolve," not zero or a
/// wraparound value.
///
/// Identity-element simplifications (`x - 0`, `0 + x`, `x * 1`, `1 * x`, `x // 1` all fold to
/// `x` verbatim, even when `x` itself is symbolic) matter beyond tidiness: `tl.arange(0,
/// BLOCK)` is by far the most common way a `constexpr` name becomes a dimension, and without
/// this, that dimension would render as `"BLOCK - 0"` while a bare reference to the same
/// parameter elsewhere (`tl.zeros((BLOCK, ...))`) renders as `"BLOCK"` — two spellings of the
/// identical size that `broadcast_dim`'s conservative text-match rule would then wrongly treat
/// as incompatible. Beyond these identities this pass does no further simplification (it does
/// not know `BLOCK - BLOCK` is `0`, for instance) — deliberately: proving two differently
/// shaped symbolic expressions denote the same size in general is real symbolic algebra, which
/// is out of scope (see the module header in `triton_ty.rs`).
fn combine_dim(op: BinOp, l: &Dim, r: &Dim) -> Option<Dim> {
    match op {
        BinOp::Add if matches!(l, Dim::Const(0)) => return Some(r.clone()),
        BinOp::Add | BinOp::Sub if matches!(r, Dim::Const(0)) => return Some(l.clone()),
        BinOp::Mul if matches!(l, Dim::Const(1)) => return Some(r.clone()),
        BinOp::Mul | BinOp::FloorDiv if matches!(r, Dim::Const(1)) => return Some(l.clone()),
        _ => {}
    }
    let opstr = match op {
        BinOp::Add => "+",
        BinOp::Sub => "-",
        BinOp::Mul => "*",
        BinOp::FloorDiv => "//",
        _ => return None,
    };
    if let (Dim::Const(a), Dim::Const(b)) = (l, r) {
        let folded = match op {
            BinOp::Add => a.checked_add(*b),
            BinOp::Sub => a.checked_sub(*b),
            BinOp::Mul => a.checked_mul(*b),
            BinOp::FloorDiv if *b != 0 => Some(a.div_euclid(*b)),
            _ => None,
        };
        return folded.map(Dim::Const);
    }
    Some(Dim::Symbolic(format!("{l} {opstr} {r}")))
}

/// Parses a Triton/Python integer literal's exact source text (as `basalt-frontend-triton`
/// hands it back, `_`-separated digit groups and `0x`/`0o`/`0b` prefixes included, matching
/// what Python itself accepts) to a value usable as a `Dim::Const`.
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

#[cfg(test)]
#[path = "triton_check_tests.rs"]
mod tests;
