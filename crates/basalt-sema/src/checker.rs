// The type checker itself: builds a symbol table while walking a `TranslationUnit`, resolves
// every identifier and type reference, and assigns every expression a `Ty`. Errors are
// collected into a `Vec<Diag>` rather than raised — `check` always returns, and it visits
// every reachable item/statement/expression regardless of how many problems it finds along
// the way (error recovery: report many, never stop at the first, never hang — walking a
// finite tree can't loop by construction).
//
// Templates are explicitly out of scope: `Item::Template` is recognized and skipped without
// descending into its templated body (`T`/`N` aren't concrete yet, so nothing there can be
// checked for real), and `Type::Instantiated` (`Name<Arg, ...>`) always resolves to
// `Ty::Unknown` rather than being substituted and checked.
//
// Known simplifications, none of which affect the required behavior of this pass:
// - No real "usual arithmetic conversions" ladder: `ty::promote` picks a plausible common
//   type by rank, not the standard's signedness-aware rules.
// - A pointer is assignable from any integer type, not only a null-constant literal `0` (C
//   restricts implicit pointer/integer conversions to that case; this pass does not evaluate
//   constant expressions to tell a literal `0` from any other integer).
// - Name resolution is a single sequential pass: a struct/union/enum/typedef/function must be
//   declared textually before first use. A function may call itself (its own signature is
//   registered before its body is checked) but not a sibling defined later in the same file.
// - Re-declaring a function (a prototype followed by its definition, or two identical
//   prototypes) is treated as an ordinary redefinition (`E302`); there is no signature-
//   equality-based prototype/definition merging.
// - A missing `return` on a control path of a non-void function is not checked.
// - Anonymous struct/union fields are recorded but not merged into the enclosing scope, since
//   there is no name to bind them under.
//
// `break`/`continue` misuse and an undefined `goto` label reuse the existing sema codes rather
// than minting new ones: `break`/`continue` outside their required context is `E300` (a
// structural type/context error), an unknown label is `E301` (an unresolved symbol reference).
//
// CUDA support (ARCHITECTURE.md §6): `basalt-frontend-c` recognizes `__global__`/`__device__`/
// `__host__`/`__shared__`/`__constant__` positionally (see its `CudaQualifiers`) but does not
// judge them; this pass does. `check_function_cuda_quals`/`check_var_cuda_quals` reject
// nonsensical combinations (`__global__` combined with `__host__`/`__device__`, either of those
// on a variable, more than one memory-space qualifier on a variable, ...) and a non-`void`
// `__global__` function, all under the new `E303` code.
//
// `threadIdx`/`blockIdx`/`blockDim`/`gridDim` are modeled as ordinary values of a synthetic
// struct type (`CUDA_DIM3_STRUCT`, with `x`/`y`/`z` unsigned members) rather than special-cased
// in member-access checking: this reuses `check_member`'s existing struct-field lookup as-is,
// the same way a real `dim3` would work if the user had declared it. `__syncthreads` is an
// ordinary zero-parameter `void` function, so it reuses `check_call`'s existing arity check for
// free. Both are seeded into the fresh scope `check_function_body` pushes, and only for a
// `__global__`/`__device__` function — never for a plain host function. This is the stricter of
// the two designs the task allows: using a builtin outside a device context is deliberately an
// ordinary `E301` (undefined symbol), on the view that silently accepting `threadIdx` in host
// code would hide a real portability bug rather than catch one.

use std::collections::HashSet;

use basalt_diag::{Diag, ECode, Loc as DLoc, Span as DSpan};
use basalt_frontend_c::ast::{
    AssignOp, BinOp, EnumDecl, Expr, FunctionDecl, NamespaceDecl, ScalarKind, Stmt, StructDecl,
    TagKind, Type, TypedefDecl, UnaryOp, UnionDecl, VarDecl,
};
use basalt_frontend_c::ast::{Item, TranslationUnit};
use basalt_frontend_c::{FloatLit, FloatSuffix, IntLit, Span as FSpan};

use crate::scope::{FuncSig, ScopeStack, StructInfo, ValueSym};
use crate::ty::{assignable, promote, Ty};

/// Name the synthetic `x`/`y`/`z` struct backing `threadIdx`/`blockIdx`/`blockDim`/`gridDim` is
/// registered under, in the scope `check_function_body` pushes for a device/kernel body.
/// Distinct from `dim3` (the name CUDA's own headers use) since this pass never parses those
/// headers and a user's own same-named struct, if any, lives in an outer scope regardless.
const CUDA_DIM3_STRUCT: &str = "__basalt_cuda_dim3";

/// The four `dim3`-typed builtin values available inside a device/kernel body.
const CUDA_DIM3_BUILTINS: [&str; 4] = ["threadIdx", "blockIdx", "blockDim", "gridDim"];

fn conv_span(s: FSpan) -> DSpan {
    DSpan::new(
        DLoc::new(s.start.line, s.start.col),
        DLoc::new(s.end.line, s.end.col),
    )
}

fn int_lit_ty(lit: &IntLit) -> Ty {
    use ScalarKind::*;
    Ty::Scalar(match (lit.unsigned, lit.long_len) {
        (false, 0) => Int,
        (false, 1) => Long,
        (false, _) => LongLong,
        (true, 0) => UInt,
        (true, 1) => ULong,
        (true, _) => ULongLong,
    })
}

fn float_lit_ty(lit: &FloatLit) -> Ty {
    Ty::Scalar(match lit.suffix {
        FloatSuffix::F => ScalarKind::Float,
        FloatSuffix::L => ScalarKind::LongDouble,
        FloatSuffix::None => ScalarKind::Double,
    })
}

fn compound_binop(op: AssignOp) -> BinOp {
    match op {
        AssignOp::AddAssign => BinOp::Add,
        AssignOp::SubAssign => BinOp::Sub,
        AssignOp::MulAssign => BinOp::Mul,
        AssignOp::DivAssign => BinOp::Div,
        AssignOp::RemAssign => BinOp::Rem,
        AssignOp::AndAssign => BinOp::BitAnd,
        AssignOp::OrAssign => BinOp::BitOr,
        AssignOp::XorAssign => BinOp::BitXor,
        AssignOp::ShlAssign => BinOp::Shl,
        AssignOp::ShrAssign => BinOp::Shr,
        AssignOp::Assign => unreachable!("plain '=' has no underlying binary operator"),
    }
}

/// Recursively collects every label name reachable in a function body, for a two-pass
/// `goto`/label check: a label may be declared after the `goto` that targets it, or in a
/// sibling block, so this walks the whole body up front rather than tracking declare-order.
fn collect_labels_many(stmts: &[Stmt], out: &mut HashSet<String>) {
    for s in stmts {
        collect_labels(s, out);
    }
}

fn collect_labels(stmt: &Stmt, out: &mut HashSet<String>) {
    match stmt {
        Stmt::Block { stmts, .. } => collect_labels_many(stmts, out),
        Stmt::If {
            then_branch,
            else_branch,
            ..
        } => {
            collect_labels(then_branch, out);
            if let Some(e) = else_branch {
                collect_labels(e, out);
            }
        }
        Stmt::While { body, .. } | Stmt::DoWhile { body, .. } => collect_labels(body, out),
        Stmt::For { init, body, .. } => {
            if let Some(i) = init {
                collect_labels(i, out);
            }
            collect_labels(body, out);
        }
        Stmt::Switch { body, .. } => collect_labels(body, out),
        Stmt::Case { stmt, .. } | Stmt::Default { stmt, .. } => collect_labels(stmt, out),
        Stmt::Label { name, stmt, .. } => {
            out.insert(name.clone());
            collect_labels(stmt, out);
        }
        Stmt::Expr { .. }
        | Stmt::Empty { .. }
        | Stmt::Decl { .. }
        | Stmt::Break { .. }
        | Stmt::Continue { .. }
        | Stmt::Return { .. }
        | Stmt::Goto { .. } => {}
    }
}

struct Checker {
    scopes: ScopeStack,
    diags: Vec<Diag>,
    current_fn_ret: Option<Ty>,
    current_fn_labels: HashSet<String>,
    loop_depth: u32,
    switch_depth: u32,
}

/// Type-checks a translation unit, returning every diagnostic found. Never stops at the first
/// error and never panics on malformed-but-parseable input; a construct this pass cannot make
/// sense of degrades to `Ty::Unknown` rather than aborting the walk.
pub fn check(tu: &TranslationUnit) -> Vec<Diag> {
    let mut ck = Checker {
        scopes: ScopeStack::new(),
        diags: Vec::new(),
        current_fn_ret: None,
        current_fn_labels: HashSet::new(),
        loop_depth: 0,
        switch_depth: 0,
    };
    ck.scopes.push();
    ck.check_items(&tu.items);
    ck.scopes.pop();
    ck.diags
}

impl Checker {
    fn err_type(&mut self, span: FSpan, msg: impl Into<String>) {
        self.diags.push(
            Diag::new(ECode::TypeError)
                .with_span(conv_span(span))
                .with_arg(msg.into()),
        );
    }

    fn err_undef(&mut self, span: FSpan, name: &str) {
        self.diags.push(
            Diag::new(ECode::UndefinedSymbol)
                .with_span(conv_span(span))
                .with_arg(name.to_string()),
        );
    }

    fn err_redef(&mut self, span: FSpan, name: &str) {
        self.diags.push(
            Diag::new(ECode::Redefinition)
                .with_span(conv_span(span))
                .with_arg(name.to_string()),
        );
    }

    fn err_cuda(&mut self, span: FSpan, name: &str, detail: impl Into<String>) {
        self.diags.push(
            Diag::new(ECode::InvalidCudaQualifier)
                .with_span(conv_span(span))
                .with_arg(name.to_string())
                .with_arg(detail.into()),
        );
    }

    // ---- items ----------------------------------------------------------------------------

    fn check_items(&mut self, items: &[Item]) {
        for item in items {
            self.check_item(item);
        }
    }

    fn check_item(&mut self, item: &Item) {
        match item {
            Item::Struct(d) => self.check_struct_decl(d),
            Item::Union(d) => self.check_union_decl(d),
            Item::Enum(d) => self.check_enum_decl(d),
            Item::Typedef(d) => self.check_typedef_decl(d),
            Item::Namespace(ns) => self.check_namespace(ns),
            Item::Function(f) => self.check_function_item(f),
            Item::Var(v) => self.check_var_decl(v),
            // Templates are out of scope for this pass (see module header): recognized,
            // never descended into.
            Item::Template(_) => {}
        }
    }

    fn check_namespace(&mut self, ns: &NamespaceDecl) {
        self.scopes.push();
        self.check_items(&ns.items);
        self.scopes.pop();
    }

    fn check_struct_decl(&mut self, d: &StructDecl) {
        let mut fields = Vec::with_capacity(d.fields.len());
        for f in &d.fields {
            let ty = self.resolve_type(&f.ty);
            fields.push((f.name.clone(), ty));
        }
        if let Some(name) = &d.name {
            if self.scopes.declare_struct(name, StructInfo { fields }) {
                self.err_redef(d.span, name);
            }
        }
    }

    fn check_union_decl(&mut self, d: &UnionDecl) {
        let mut fields = Vec::with_capacity(d.fields.len());
        for f in &d.fields {
            let ty = self.resolve_type(&f.ty);
            fields.push((f.name.clone(), ty));
        }
        if let Some(name) = &d.name {
            if self.scopes.declare_union(name, StructInfo { fields }) {
                self.err_redef(d.span, name);
            }
        }
    }

    fn check_enum_decl(&mut self, d: &EnumDecl) {
        if let Some(name) = &d.name {
            if self.scopes.declare_enum(name) {
                self.err_redef(d.span, name);
            }
        }
        let enum_ty = match &d.name {
            Some(n) => Ty::Enum(n.clone()),
            None => Ty::Scalar(ScalarKind::Int),
        };
        for v in &d.variants {
            if let Some(init) = &v.init {
                let it = self.type_of(init);
                if !it.is_unknown() && !it.is_integer() {
                    self.err_type(v.span, "enum initializer must be an integer constant");
                }
            }
            if self
                .scopes
                .declare_value(&v.name, ValueSym::EnumConst(enum_ty.clone()))
            {
                self.err_redef(v.span, &v.name);
            }
        }
    }

    fn check_typedef_decl(&mut self, d: &TypedefDecl) {
        let ty = self.resolve_type(&d.ty);
        if self.scopes.declare_typedef(&d.alias, ty) {
            self.err_redef(d.span, &d.alias);
        }
    }

    fn check_var_decl(&mut self, v: &VarDecl) {
        self.check_var_cuda_quals(v);
        let ty = self.resolve_type(&v.ty);
        if let Some(init) = &v.init {
            let it = self.type_of(init);
            if !ty.is_unknown() && !it.is_unknown() && !assignable(&ty, &it) {
                self.err_type(
                    v.span,
                    format!("cannot initialize '{}' with this expression's type", v.name),
                );
            }
        }
        if self.scopes.declare_value(&v.name, ValueSym::Var(ty)) {
            self.err_redef(v.span, &v.name);
        }
    }

    /// Checks the CUDA qualifiers (if any) on a variable declaration: `__shared__`,
    /// `__constant__`, and `__device__` are the ones that make sense on a variable;
    /// `__global__`/`__host__` are function-only, and a variable naming more than one memory
    /// space at once (`__shared__ __constant__ int x;`) is nonsensical, since each names a
    /// distinct piece of memory the value could live in.
    fn check_var_cuda_quals(&mut self, v: &VarDecl) {
        let q = v.cuda_quals;
        if q.is_global || q.is_host {
            self.err_cuda(
                v.span,
                &v.name,
                "'__global__'/'__host__' apply to functions, not variables",
            );
        }
        let mem_spaces = [q.is_shared, q.is_constant, q.is_device]
            .into_iter()
            .filter(|b| *b)
            .count();
        if mem_spaces > 1 {
            self.err_cuda(
                v.span,
                &v.name,
                "a variable cannot combine more than one of '__shared__'/'__constant__'/'__device__'",
            );
        }
    }

    fn check_function_item(&mut self, f: &FunctionDecl) {
        let ret = self.resolve_type(&f.ret);
        self.check_function_cuda_quals(f, &ret);
        let param_tys: Vec<Ty> = f.params.iter().map(|p| self.resolve_type(&p.ty)).collect();
        let sig = FuncSig {
            ret: ret.clone(),
            params: param_tys.clone(),
            variadic: f.variadic,
        };
        if self.scopes.declare_value(&f.name, ValueSym::Func(sig)) {
            self.err_redef(f.span, &f.name);
        }
        if let Some(body) = &f.body {
            self.check_function_body(f, ret, &param_tys, body);
        }
    }

    /// Checks the CUDA qualifiers (if any) on a function declaration. Valid combinations are
    /// the empty set (an ordinary host function), `__host__` alone, `__device__` alone,
    /// `__global__` alone, and `__host__ __device__` together (real CUDA compiles that pair
    /// twice, once per target); `__global__` combined with either of the other two is rejected,
    /// as is either memory-space qualifier (`__shared__`/`__constant__`) landing on a function
    /// instead of a variable. A `__global__` kernel is further required to return `void`, the
    /// only return type CUDA allows for one.
    fn check_function_cuda_quals(&mut self, f: &FunctionDecl, ret: &Ty) {
        let q = f.cuda_quals;
        if q.is_shared || q.is_constant {
            self.err_cuda(
                f.span,
                &f.name,
                "'__shared__'/'__constant__' apply to variables, not functions",
            );
        }
        if q.is_global && (q.is_device || q.is_host) {
            self.err_cuda(
                f.span,
                &f.name,
                "'__global__' cannot be combined with '__host__' or '__device__'",
            );
        }
        if q.is_global && !ret.is_unknown() && !matches!(ret, Ty::Scalar(ScalarKind::Void)) {
            self.err_cuda(f.span, &f.name, "a '__global__' function must return void");
        }
    }

    fn check_function_body(&mut self, f: &FunctionDecl, ret: Ty, param_tys: &[Ty], body: &[Stmt]) {
        let prev_ret = self.current_fn_ret.replace(ret);
        let mut labels = HashSet::new();
        collect_labels_many(body, &mut labels);
        let prev_labels = std::mem::replace(&mut self.current_fn_labels, labels);
        let prev_loop = std::mem::replace(&mut self.loop_depth, 0);
        let prev_switch = std::mem::replace(&mut self.switch_depth, 0);

        self.scopes.push();
        if f.cuda_quals.is_global || f.cuda_quals.is_device {
            self.seed_cuda_builtins();
        }
        for (p, ty) in f.params.iter().zip(param_tys.iter()) {
            if let Some(name) = &p.name {
                if self.scopes.declare_value(name, ValueSym::Var(ty.clone())) {
                    self.err_redef(p.span, name);
                }
            }
        }
        for s in body {
            self.check_stmt(s);
        }
        self.scopes.pop();

        self.current_fn_ret = prev_ret;
        self.current_fn_labels = prev_labels;
        self.loop_depth = prev_loop;
        self.switch_depth = prev_switch;
    }

    /// Populates the function-body scope just pushed with the builtins available inside a
    /// kernel/device body: `threadIdx`/`blockIdx`/`blockDim`/`gridDim` as values of a synthetic
    /// `x`/`y`/`z` struct type, and `__syncthreads` as an ordinary zero-parameter `void`
    /// function. Both ride the checker's existing member-access and call-arity machinery
    /// instead of needing special cases there.
    fn seed_cuda_builtins(&mut self) {
        self.scopes.declare_struct(
            CUDA_DIM3_STRUCT,
            StructInfo {
                fields: vec![
                    ("x".to_string(), Ty::Scalar(ScalarKind::UInt)),
                    ("y".to_string(), Ty::Scalar(ScalarKind::UInt)),
                    ("z".to_string(), Ty::Scalar(ScalarKind::UInt)),
                ],
            },
        );
        let dim3 = Ty::Struct(CUDA_DIM3_STRUCT.to_string());
        for name in CUDA_DIM3_BUILTINS {
            self.scopes.declare_value(name, ValueSym::Var(dim3.clone()));
        }
        self.scopes.declare_value(
            "__syncthreads",
            ValueSym::Func(FuncSig {
                ret: Ty::Scalar(ScalarKind::Void),
                params: Vec::new(),
                variadic: false,
            }),
        );
    }

    // ---- statements -------------------------------------------------------------------------

    fn check_condition(&mut self, cond: &Expr) {
        let t = self.type_of(cond);
        if !t.is_unknown() && !t.is_scalar_condition() {
            self.err_type(
                cond.span(),
                "condition must have a scalar (arithmetic or pointer) type",
            );
        }
    }

    fn check_stmt(&mut self, stmt: &Stmt) {
        match stmt {
            Stmt::Expr { expr, .. } => {
                self.type_of(expr);
            }
            Stmt::Empty { .. } => {}
            Stmt::Block { stmts, .. } => {
                self.scopes.push();
                for s in stmts {
                    self.check_stmt(s);
                }
                self.scopes.pop();
            }
            Stmt::Decl { decls, .. } => {
                for d in decls {
                    self.check_var_decl(d);
                }
            }
            Stmt::If {
                cond,
                then_branch,
                else_branch,
                ..
            } => {
                self.check_condition(cond);
                self.check_stmt(then_branch);
                if let Some(e) = else_branch {
                    self.check_stmt(e);
                }
            }
            Stmt::While { cond, body, .. } => {
                self.check_condition(cond);
                self.loop_depth += 1;
                self.check_stmt(body);
                self.loop_depth -= 1;
            }
            Stmt::DoWhile { body, cond, .. } => {
                self.loop_depth += 1;
                self.check_stmt(body);
                self.loop_depth -= 1;
                self.check_condition(cond);
            }
            Stmt::For {
                init,
                cond,
                step,
                body,
                ..
            } => {
                self.scopes.push();
                if let Some(i) = init {
                    self.check_stmt(i);
                }
                if let Some(c) = cond {
                    self.check_condition(c);
                }
                if let Some(s) = step {
                    self.type_of(s);
                }
                self.loop_depth += 1;
                self.check_stmt(body);
                self.loop_depth -= 1;
                self.scopes.pop();
            }
            Stmt::Switch { expr, body, .. } => {
                let t = self.type_of(expr);
                if !t.is_unknown() && !t.is_integer() {
                    self.err_type(expr.span(), "switch expression must have an integer type");
                }
                self.switch_depth += 1;
                self.check_stmt(body);
                self.switch_depth -= 1;
            }
            Stmt::Case { value, stmt, .. } => {
                let t = self.type_of(value);
                if !t.is_unknown() && !t.is_integer() {
                    self.err_type(value.span(), "case label must be an integer constant");
                }
                self.check_stmt(stmt);
            }
            Stmt::Default { stmt, .. } => self.check_stmt(stmt),
            Stmt::Break { span } => {
                if self.loop_depth == 0 && self.switch_depth == 0 {
                    self.err_type(*span, "'break' used outside of a loop or switch");
                }
            }
            Stmt::Continue { span } => {
                if self.loop_depth == 0 {
                    self.err_type(*span, "'continue' used outside of a loop");
                }
            }
            Stmt::Return { expr, span } => {
                let rt = expr
                    .as_ref()
                    .map(|e| self.type_of(e))
                    .unwrap_or(Ty::Scalar(ScalarKind::Void));
                if let Some(fret) = self.current_fn_ret.clone() {
                    if !fret.is_unknown() && !rt.is_unknown() && !assignable(&fret, &rt) {
                        self.err_type(
                            *span,
                            "return expression's type does not match the function's declared return type",
                        );
                    }
                }
            }
            Stmt::Label { stmt, .. } => self.check_stmt(stmt),
            Stmt::Goto { label, span } => {
                if !self.current_fn_labels.contains(label) {
                    self.err_undef(*span, label);
                }
            }
        }
    }

    // ---- types ------------------------------------------------------------------------------

    fn resolve_type(&mut self, ty: &Type) -> Ty {
        match ty {
            Type::Scalar { kind, .. } => Ty::Scalar(*kind),
            Type::Tag {
                kind, name, span, ..
            } => {
                if name.is_empty() {
                    // Anonymous tag: the definition itself (if any) was already registered as
                    // its own `Item`; there is no name here to resolve against.
                    return Ty::Unknown;
                }
                let found = match kind {
                    TagKind::Struct => self.scopes.lookup_struct(name).is_some(),
                    TagKind::Union => self.scopes.lookup_union(name).is_some(),
                    TagKind::Enum => self.scopes.lookup_enum(name).is_some(),
                };
                if !found {
                    self.err_undef(*span, name);
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
                self.err_undef(*span, name);
                Ty::Unknown
            }
            Type::Pointer { pointee, .. } => Ty::Pointer(Box::new(self.resolve_type(pointee))),
            Type::Array { elem, size, .. } => {
                if let Some(sz) = size {
                    self.type_of(sz);
                }
                Ty::Array(Box::new(self.resolve_type(elem)))
            }
            // Template instantiation is out of scope for this pass (see module header): `T`
            // isn't a concrete type yet, so an instantiated type is always "unknown" rather
            // than something to substitute and check for real.
            Type::Instantiated { .. } => Ty::Unknown,
        }
    }

    // ---- expressions ------------------------------------------------------------------------

    fn is_lvalue(&self, expr: &Expr) -> bool {
        matches!(
            expr,
            Expr::Ident { .. } | Expr::Member { .. } | Expr::Index { .. }
        ) || matches!(
            expr,
            Expr::Unary {
                op: UnaryOp::Deref,
                ..
            }
        )
    }

    fn is_modifiable_lvalue(&self, expr: &Expr, ty: &Ty) -> bool {
        self.is_lvalue(expr) && !matches!(ty, Ty::Array(_) | Ty::Function { .. })
    }

    fn comparable(&self, l: &Ty, r: &Ty) -> bool {
        if l.is_arithmetic() && r.is_arithmetic() {
            return true;
        }
        if l.is_pointer_like() && r.is_pointer_like() {
            return true;
        }
        if l.is_pointer_like() && r.is_integer() {
            return true;
        }
        if r.is_pointer_like() && l.is_integer() {
            return true;
        }
        false
    }

    fn check_binary(&mut self, op: BinOp, l: &Ty, r: &Ty, span: FSpan) -> Ty {
        use BinOp::*;
        let unknown_operand = l.is_unknown() || r.is_unknown();
        match op {
            LogOr | LogAnd => {
                if !(unknown_operand || (l.is_scalar_condition() && r.is_scalar_condition())) {
                    self.err_type(span, "operands of a logical operator must be scalar");
                }
                Ty::Scalar(ScalarKind::Int)
            }
            Eq | Ne | Lt | Gt | Le | Ge => {
                if !unknown_operand && !self.comparable(l, r) {
                    self.err_type(span, "comparison between incompatible types");
                }
                Ty::Scalar(ScalarKind::Int)
            }
            BitOr | BitXor | BitAnd | Shl | Shr => {
                if !(unknown_operand || (l.is_integer() && r.is_integer())) {
                    self.err_type(span, "operands of a bitwise operator must be integers");
                }
                if unknown_operand {
                    Ty::Unknown
                } else {
                    promote(l, r)
                }
            }
            Add => {
                if l.is_pointer_like() && r.is_integer() {
                    Ty::Pointer(Box::new(l.deref_target().unwrap_or(Ty::Unknown)))
                } else if r.is_pointer_like() && l.is_integer() {
                    Ty::Pointer(Box::new(r.deref_target().unwrap_or(Ty::Unknown)))
                } else if l.is_arithmetic() && r.is_arithmetic() {
                    promote(l, r)
                } else if unknown_operand {
                    Ty::Unknown
                } else {
                    self.err_type(span, "invalid operand types for '+'");
                    Ty::Unknown
                }
            }
            Sub => {
                if l.is_pointer_like() && r.is_pointer_like() {
                    Ty::Scalar(ScalarKind::Long)
                } else if l.is_pointer_like() && r.is_integer() {
                    Ty::Pointer(Box::new(l.deref_target().unwrap_or(Ty::Unknown)))
                } else if l.is_arithmetic() && r.is_arithmetic() {
                    promote(l, r)
                } else if unknown_operand {
                    Ty::Unknown
                } else {
                    self.err_type(span, "invalid operand types for '-'");
                    Ty::Unknown
                }
            }
            Mul | Div => {
                if unknown_operand {
                    Ty::Unknown
                } else if !(l.is_arithmetic() && r.is_arithmetic()) {
                    self.err_type(span, "operands of an arithmetic operator must be numeric");
                    Ty::Unknown
                } else {
                    promote(l, r)
                }
            }
            Rem => {
                if unknown_operand {
                    Ty::Unknown
                } else if !(l.is_integer() && r.is_integer()) {
                    self.err_type(span, "operands of '%' must be integers");
                    Ty::Unknown
                } else {
                    promote(l, r)
                }
            }
        }
    }

    fn check_unary(&mut self, op: UnaryOp, expr: &Expr, span: FSpan) -> Ty {
        match op {
            UnaryOp::Plus | UnaryOp::Neg => {
                let t = self.type_of(expr);
                if !t.is_unknown() && !t.is_arithmetic() {
                    self.err_type(
                        span,
                        "operand of a unary '+'/'-' must have an arithmetic type",
                    );
                }
                t
            }
            UnaryOp::Not => {
                let t = self.type_of(expr);
                if !t.is_unknown() && !t.is_scalar_condition() {
                    self.err_type(span, "operand of '!' must be scalar");
                }
                Ty::Scalar(ScalarKind::Int)
            }
            UnaryOp::BitNot => {
                let t = self.type_of(expr);
                if !t.is_unknown() && !t.is_integer() {
                    self.err_type(span, "operand of '~' must have an integer type");
                }
                t
            }
            UnaryOp::Deref => {
                let t = self.type_of(expr);
                if t.is_unknown() {
                    return Ty::Unknown;
                }
                match t.deref_target() {
                    Some(inner) => inner,
                    None => {
                        self.err_type(span, "cannot dereference a non-pointer type");
                        Ty::Unknown
                    }
                }
            }
            UnaryOp::Addr => {
                let t = self.type_of(expr);
                if !self.is_lvalue(expr) {
                    self.err_type(span, "cannot take the address of a non-lvalue expression");
                }
                if t.is_unknown() {
                    Ty::Unknown
                } else {
                    Ty::Pointer(Box::new(t))
                }
            }
        }
    }

    fn check_incdec(&mut self, inner: &Expr, span: FSpan) -> Ty {
        let t = self.type_of(inner);
        if !self.is_lvalue(inner) {
            self.err_type(span, "operand of increment/decrement must be an lvalue");
        }
        if !(t.is_unknown() || t.is_arithmetic() || t.is_pointer()) {
            self.err_type(
                span,
                "operand of increment/decrement must be arithmetic or pointer",
            );
        }
        t
    }

    fn check_assign(&mut self, op: AssignOp, lhs: &Expr, rhs: &Expr, span: FSpan) -> Ty {
        let lhs_ty = self.type_of(lhs);
        let rhs_ty = self.type_of(rhs);
        if !self.is_modifiable_lvalue(lhs, &lhs_ty) {
            self.err_type(lhs.span(), "assignment target is not a modifiable lvalue");
        }
        match op {
            AssignOp::Assign => {
                if !lhs_ty.is_unknown() && !rhs_ty.is_unknown() && !assignable(&lhs_ty, &rhs_ty) {
                    self.err_type(span, "incompatible types in assignment");
                }
            }
            _ => {
                let bin = compound_binop(op);
                let result = self.check_binary(bin, &lhs_ty, &rhs_ty, span);
                if !lhs_ty.is_unknown() && !result.is_unknown() && !assignable(&lhs_ty, &result) {
                    self.err_type(span, "incompatible types in compound assignment");
                }
            }
        }
        lhs_ty
    }

    fn check_ternary(
        &mut self,
        cond: &Expr,
        then_branch: &Expr,
        else_branch: &Expr,
        span: FSpan,
    ) -> Ty {
        self.check_condition(cond);
        let tt = self.type_of(then_branch);
        let et = self.type_of(else_branch);
        if tt.is_unknown() {
            // `assignable` also treats an unknown `et` as compatible with any `tt`, so the
            // branch below already covers that case once `tt` itself is known.
            et
        } else if assignable(&tt, &et) {
            tt
        } else if assignable(&et, &tt) {
            et
        } else {
            self.err_type(
                span,
                "incompatible types in the two branches of a ternary expression",
            );
            Ty::Unknown
        }
    }

    fn check_call(&mut self, callee: &Expr, args: &[Expr], span: FSpan) -> Ty {
        if let Expr::Ident {
            name,
            span: ident_span,
        } = callee
        {
            match self.scopes.lookup_value(name).cloned() {
                Some(ValueSym::Func(sig)) => {
                    if args.len() < sig.params.len()
                        || (!sig.variadic && args.len() > sig.params.len())
                    {
                        self.err_type(
                            span,
                            format!(
                                "call to '{name}' expects {} argument(s), got {}",
                                sig.params.len(),
                                args.len()
                            ),
                        );
                    }
                    for (i, arg) in args.iter().enumerate() {
                        let at = self.type_of(arg);
                        if let Some(pt) = sig.params.get(i) {
                            if !at.is_unknown() && !pt.is_unknown() && !assignable(pt, &at) {
                                self.err_type(
                                    arg.span(),
                                    format!(
                                        "argument {} to '{name}' has an incompatible type",
                                        i + 1
                                    ),
                                );
                            }
                        }
                    }
                    sig.ret
                }
                Some(_) => {
                    self.err_type(*ident_span, format!("'{name}' is not a function"));
                    for arg in args {
                        self.type_of(arg);
                    }
                    Ty::Unknown
                }
                None => {
                    self.err_undef(*ident_span, name);
                    for arg in args {
                        self.type_of(arg);
                    }
                    Ty::Unknown
                }
            }
        } else {
            self.type_of(callee);
            for arg in args {
                self.type_of(arg);
            }
            Ty::Unknown
        }
    }

    fn check_index(&mut self, base: &Expr, index: &Expr, span: FSpan) -> Ty {
        let bt = self.type_of(base);
        let it = self.type_of(index);
        if !it.is_unknown() && !it.is_integer() {
            self.err_type(index.span(), "array index must have an integer type");
        }
        if bt.is_unknown() {
            return Ty::Unknown;
        }
        match bt.deref_target() {
            Some(elem) => elem,
            None => {
                self.err_type(span, "indexed value is not an array or pointer");
                Ty::Unknown
            }
        }
    }

    fn check_member(&mut self, base: &Expr, name: &str, arrow: bool, span: FSpan) -> Ty {
        let bt = self.type_of(base);
        if bt.is_unknown() {
            return Ty::Unknown;
        }
        let target = if arrow {
            match bt.deref_target() {
                Some(inner) => inner,
                None => {
                    self.err_type(span, "'->' used on a non-pointer type");
                    return Ty::Unknown;
                }
            }
        } else {
            if bt.is_pointer() {
                self.err_type(span, "'.' used on a pointer type; did you mean '->'");
                return Ty::Unknown;
            }
            bt
        };
        if target.is_unknown() {
            return Ty::Unknown;
        }
        let fields = match &target {
            Ty::Struct(n) => self.scopes.lookup_struct(n).map(|s| s.fields.clone()),
            Ty::Union(n) => self.scopes.lookup_union(n).map(|s| s.fields.clone()),
            _ => {
                self.err_type(span, "member access on a non-struct/union type");
                return Ty::Unknown;
            }
        };
        match fields.and_then(|fs| fs.into_iter().find(|(n, _)| n == name)) {
            Some((_, ty)) => ty,
            None => {
                self.err_undef(span, name);
                Ty::Unknown
            }
        }
    }

    fn type_of(&mut self, expr: &Expr) -> Ty {
        match expr {
            Expr::IntLit { value, .. } => int_lit_ty(value),
            Expr::FloatLit { value, .. } => float_lit_ty(value),
            Expr::CharLit { .. } => Ty::Scalar(ScalarKind::Char),
            Expr::StrLit { .. } => Ty::Pointer(Box::new(Ty::Scalar(ScalarKind::Char))),
            Expr::Ident { name, span } => match self.scopes.lookup_value(name) {
                Some(ValueSym::Var(t)) => t.clone(),
                Some(ValueSym::EnumConst(t)) => t.clone(),
                Some(ValueSym::Func(sig)) => Ty::Function {
                    ret: Box::new(sig.ret.clone()),
                    params: sig.params.clone(),
                    variadic: sig.variadic,
                },
                None => {
                    self.err_undef(*span, name);
                    Ty::Unknown
                }
            },
            Expr::Comma { exprs, .. } => {
                let mut last = Ty::Unknown;
                for e in exprs {
                    last = self.type_of(e);
                }
                last
            }
            Expr::Assign { op, lhs, rhs, span } => self.check_assign(*op, lhs, rhs, *span),
            Expr::Ternary {
                cond,
                then_branch,
                else_branch,
                span,
            } => self.check_ternary(cond, then_branch, else_branch, *span),
            Expr::Binary { op, lhs, rhs, span } => {
                let l = self.type_of(lhs);
                let r = self.type_of(rhs);
                self.check_binary(*op, &l, &r, *span)
            }
            Expr::Cast { ty, expr, .. } => {
                self.type_of(expr);
                self.resolve_type(ty)
            }
            Expr::Unary { op, expr, span } => self.check_unary(*op, expr, *span),
            Expr::PreIncDec { expr, span, .. } | Expr::PostIncDec { expr, span, .. } => {
                self.check_incdec(expr, *span)
            }
            Expr::SizeofExpr { expr, .. } => {
                self.type_of(expr);
                Ty::Scalar(ScalarKind::ULong)
            }
            Expr::SizeofType { ty, .. } => {
                self.resolve_type(ty);
                Ty::Scalar(ScalarKind::ULong)
            }
            Expr::Call { callee, args, span } => self.check_call(callee, args, *span),
            Expr::Index { base, index, span } => self.check_index(base, index, *span),
            Expr::Member {
                base,
                name,
                arrow,
                span,
            } => self.check_member(base, name, *arrow, *span),
            Expr::Error { .. } => Ty::Unknown,
        }
    }
}
