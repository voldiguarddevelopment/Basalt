// Turns Python source into this crate's own AST (see `ast.rs`) by walking the tree
// `ruff_python_parser` hands back. There is no separate lex/preprocess stage the way
// `basalt-frontend-c` has one: Python has no macro preprocessor, and `ruff_python_parser`
// only exposes a single "parse this source" entry point, so one function is the whole
// pipeline here.
//
// `ruff_python_parser::parse_unchecked_source` is used rather than the `Result`-returning
// `parse_module`: it always returns a (possibly partial) `ModModule` plus every syntax error
// collected along the way, instead of stopping at the first one. That is exactly
// `basalt-frontend-c`'s own "report many, never stop" shape, so this layer keeps it: a
// problem found while lowering one statement or expression never discards the rest of a
// kernel, and a `@triton.jit` function is always included in the output (best-effort) even
// when part of its body couldn't be lowered — the accompanying `Diag`s are what make the
// problem visible, not a missing kernel.
//
// Diagnostics are handed back as `basalt_diag::Diag` directly (unlike `basalt-frontend-c`,
// which keeps its own local error types and only meets `Diag` at the CLI boundary): this
// crate has no earlier stage of its own for a caller to unify with, so there is nothing to
// lose by reporting through the shared diagnostic type from the start.

use basalt_diag::{Diag, ECode, Loc, Span};
use ruff_python_ast as py;
use ruff_python_ast::PySourceType;
use ruff_text_size::{Ranged, TextRange};

use crate::ast::{self, is_constexpr_annotation};

/// Parses `src` as a Python source file and returns every `@triton.jit`-decorated function
/// found, plus every problem hit along the way. An empty `kernels` list with no diagnostics
/// simply means the file had no decorated functions in it — not an error.
pub fn parse(src: &str) -> (ast::Module, Vec<Diag>) {
    let mut lowerer = Lowerer::new(src);
    let parsed = ruff_python_parser::parse_unchecked_source(src, PySourceType::Python);

    for err in parsed.errors() {
        lowerer.diags.push(
            Diag::new(ECode::ParseError)
                .with_arg(err.error.to_string())
                .with_span(lowerer.span(err.range())),
        );
    }

    let mut kernels = Vec::new();
    for stmt in parsed.suite() {
        if let py::Stmt::FunctionDef(f) = stmt {
            if f.decorator_list
                .iter()
                .any(|d| is_triton_jit(&d.expression))
            {
                kernels.push(lowerer.lower_kernel(f));
            }
        }
    }

    (ast::Module { kernels }, lowerer.diags)
}

/// True for `@triton.jit` and `@triton.jit(...)` (the call form, used for e.g.
/// `@triton.jit(do_not_specialize=[...])`). Only the literal `triton.jit` attribute path is
/// recognized — an aliased import (`from triton import jit` then bare `@jit`) can't be told
/// apart from an unrelated decorator without tracking imports, which is out of scope here, so
/// it is deliberately not matched rather than guessed at.
fn is_triton_jit(expr: &py::Expr) -> bool {
    match expr {
        py::Expr::Attribute(a) => is_triton_name(&a.value) && a.attr.as_str() == "jit",
        py::Expr::Call(c) => is_triton_jit(&c.func),
        _ => false,
    }
}

fn is_triton_name(expr: &py::Expr) -> bool {
    matches!(expr, py::Expr::Name(n) if n.id.as_str() == "triton")
}

/// Byte-offset-to-line/col conversion plus the AST lowering itself. Built once per `parse`
/// call over the full source text.
struct Lowerer<'a> {
    src: &'a str,
    line_starts: Vec<u32>,
    diags: Vec<Diag>,
}

impl<'a> Lowerer<'a> {
    fn new(src: &'a str) -> Lowerer<'a> {
        let mut line_starts = vec![0u32];
        line_starts.extend(
            src.bytes()
                .enumerate()
                .filter(|(_, b)| *b == b'\n')
                .map(|(i, _)| (i + 1) as u32),
        );
        Lowerer {
            src,
            line_starts,
            diags: Vec::new(),
        }
    }

    /// 1-based line/col (col counted in chars, matching `basalt-frontend-c`'s `Loc`) for a
    /// 0-based byte offset.
    fn loc(&self, offset: u32) -> Loc {
        let line = match self.line_starts.binary_search(&offset) {
            Ok(i) => i,
            Err(i) => i - 1,
        };
        let line_start = self.line_starts[line] as usize;
        let col = self.src[line_start..offset as usize].chars().count() as u32 + 1;
        Loc::new(line as u32 + 1, col)
    }

    fn span(&self, range: TextRange) -> Span {
        Span::new(
            self.loc(range.start().to_u32()),
            self.loc(range.end().to_u32()),
        )
    }

    fn text_at(&self, range: TextRange) -> String {
        self.src[range.start().to_u32() as usize..range.end().to_u32() as usize].to_string()
    }

    fn unsupported_expr(&mut self, what: &str, range: TextRange) -> ast::Expr {
        let span = self.span(range);
        self.diags.push(
            Diag::new(ECode::ParseError)
                .with_arg(format!("{what} is not supported in a Triton kernel"))
                .with_span(span),
        );
        ast::Expr::Error { span }
    }

    fn unsupported_stmt(&mut self, what: &str, range: TextRange) -> ast::Stmt {
        let span = self.span(range);
        self.diags.push(
            Diag::new(ECode::ParseError)
                .with_arg(format!("{what} is not supported in a Triton kernel"))
                .with_span(span),
        );
        ast::Stmt::Error { span }
    }

    fn lower_kernel(&mut self, f: &py::StmtFunctionDef) -> ast::KernelFn {
        if f.is_async {
            self.diags.push(
                Diag::new(ECode::ParseError)
                    .with_arg("'async def' is not supported on a Triton kernel")
                    .with_span(self.span(f.range())),
            );
        }
        if let Some(type_params) = &f.type_params {
            if !type_params.is_empty() {
                self.diags.push(
                    Diag::new(ECode::ParseError)
                        .with_arg("generic type parameters are not supported on a Triton kernel")
                        .with_span(self.span(type_params.range())),
                );
            }
        }

        let mut params = Vec::new();
        for p in f
            .parameters
            .posonlyargs
            .iter()
            .chain(f.parameters.args.iter())
            .chain(f.parameters.kwonlyargs.iter())
        {
            params.push(self.lower_param(p));
        }
        if let Some(vararg) = &f.parameters.vararg {
            self.diags.push(
                Diag::new(ECode::ParseError)
                    .with_arg("'*args' is not supported in a Triton kernel signature")
                    .with_span(self.span(vararg.range())),
            );
        }
        if let Some(kwarg) = &f.parameters.kwarg {
            self.diags.push(
                Diag::new(ECode::ParseError)
                    .with_arg("'**kwargs' is not supported in a Triton kernel signature")
                    .with_span(self.span(kwarg.range())),
            );
        }

        ast::KernelFn {
            name: f.name.as_str().to_string(),
            params,
            returns: f.returns.as_deref().map(|e| self.lower_expr(e)),
            body: self.lower_body(&f.body),
            span: self.span(f.range()),
        }
    }

    fn lower_param(&mut self, p: &py::ParameterWithDefault) -> ast::Param {
        let annotation = p.annotation().map(|a| self.lower_expr(a));
        let is_constexpr = annotation.as_ref().is_some_and(is_constexpr_annotation);
        ast::Param {
            name: p.name().as_str().to_string(),
            annotation,
            is_constexpr,
            default: p.default().map(|d| self.lower_expr(d)),
            span: self.span(p.range()),
        }
    }

    fn lower_body(&mut self, body: &[py::Stmt]) -> Vec<ast::Stmt> {
        body.iter().map(|s| self.lower_stmt(s)).collect()
    }

    fn lower_stmt(&mut self, s: &py::Stmt) -> ast::Stmt {
        let span = self.span(s.range());
        match s {
            py::Stmt::Expr(e) => ast::Stmt::Expr {
                value: self.lower_expr(&e.value),
                span,
            },
            py::Stmt::Assign(a) => ast::Stmt::Assign {
                targets: a.targets.iter().map(|t| self.lower_expr(t)).collect(),
                value: self.lower_expr(&a.value),
                span,
            },
            py::Stmt::AugAssign(a) => ast::Stmt::AugAssign {
                target: self.lower_expr(&a.target),
                op: lower_binop(a.op),
                value: self.lower_expr(&a.value),
                span,
            },
            py::Stmt::AnnAssign(a) => ast::Stmt::AnnAssign {
                target: self.lower_expr(&a.target),
                annotation: self.lower_expr(&a.annotation),
                value: a.value.as_deref().map(|v| self.lower_expr(v)),
                span,
            },
            py::Stmt::If(i) => self.lower_if(i, span),
            py::Stmt::For(f) => {
                if f.is_async {
                    self.diags.push(
                        Diag::new(ECode::ParseError)
                            .with_arg("'async for' is not supported in a Triton kernel")
                            .with_span(span),
                    );
                }
                ast::Stmt::For {
                    target: self.lower_expr(&f.target),
                    iter: self.lower_expr(&f.iter),
                    body: self.lower_body(&f.body),
                    orelse: self.lower_body(&f.orelse),
                    span,
                }
            }
            py::Stmt::While(w) => ast::Stmt::While {
                test: self.lower_expr(&w.test),
                body: self.lower_body(&w.body),
                orelse: self.lower_body(&w.orelse),
                span,
            },
            py::Stmt::Return(r) => ast::Stmt::Return {
                value: r.value.as_deref().map(|v| self.lower_expr(v)),
                span,
            },
            py::Stmt::Assert(a) => ast::Stmt::Assert {
                test: self.lower_expr(&a.test),
                msg: a.msg.as_deref().map(|m| self.lower_expr(m)),
                span,
            },
            py::Stmt::Pass(_) => ast::Stmt::Pass { span },
            py::Stmt::Break(_) => ast::Stmt::Break { span },
            py::Stmt::Continue(_) => ast::Stmt::Continue { span },
            py::Stmt::FunctionDef(_) => {
                self.unsupported_stmt("a nested function definition", s.range())
            }
            py::Stmt::ClassDef(_) => self.unsupported_stmt("a class definition", s.range()),
            py::Stmt::Delete(_) => self.unsupported_stmt("'del'", s.range()),
            py::Stmt::TypeAlias(_) => self.unsupported_stmt("a 'type' alias statement", s.range()),
            py::Stmt::With(_) => self.unsupported_stmt("a 'with' statement", s.range()),
            py::Stmt::Match(_) => self.unsupported_stmt("a 'match' statement", s.range()),
            py::Stmt::Raise(_) => self.unsupported_stmt("a 'raise' statement", s.range()),
            py::Stmt::Try(_) => self.unsupported_stmt("a 'try' statement", s.range()),
            py::Stmt::Import(_) => self.unsupported_stmt("an 'import' statement", s.range()),
            py::Stmt::ImportFrom(_) => {
                self.unsupported_stmt("an 'import from' statement", s.range())
            }
            py::Stmt::Global(_) => self.unsupported_stmt("a 'global' statement", s.range()),
            py::Stmt::Nonlocal(_) => self.unsupported_stmt("a 'nonlocal' statement", s.range()),
            py::Stmt::IpyEscapeCommand(_) => {
                self.unsupported_stmt("an IPython escape command", s.range())
            }
        }
    }

    /// Flattens `i`'s `elif`/`else` chain into nested `If`s in `orelse`, the same shape plain
    /// Python `ast` itself uses for `elif`.
    fn lower_if(&mut self, i: &py::StmtIf, span: Span) -> ast::Stmt {
        let orelse = self.lower_elif_chain(&i.elif_else_clauses);
        ast::Stmt::If {
            test: self.lower_expr(&i.test),
            body: self.lower_body(&i.body),
            orelse,
            span,
        }
    }

    fn lower_elif_chain(&mut self, clauses: &[py::ElifElseClause]) -> Vec<ast::Stmt> {
        let Some((clause, rest)) = clauses.split_first() else {
            return Vec::new();
        };
        match &clause.test {
            // `elif test: body ...` — one more nested `If`, itself followed by whatever
            // `elif`/`else` clauses remain.
            Some(test) => {
                let span = self.span(clause.range());
                let test = self.lower_expr(test);
                let body = self.lower_body(&clause.body);
                let orelse = self.lower_elif_chain(rest);
                vec![ast::Stmt::If {
                    test,
                    body,
                    orelse,
                    span,
                }]
            }
            // A plain `else:`; nothing can follow it, so `rest` is always empty here.
            None => self.lower_body(&clause.body),
        }
    }

    fn lower_expr(&mut self, e: &py::Expr) -> ast::Expr {
        let span = self.span(e.range());
        match e {
            py::Expr::Name(n) => ast::Expr::Name {
                name: n.id.as_str().to_string(),
                span,
            },
            py::Expr::NumberLiteral(n) => match &n.value {
                py::Number::Int(_) => ast::Expr::IntLit {
                    text: self.text_at(e.range()),
                    span,
                },
                py::Number::Float(_) => ast::Expr::FloatLit {
                    text: self.text_at(e.range()),
                    span,
                },
                py::Number::Complex { .. } => {
                    self.unsupported_expr("a complex number literal", e.range())
                }
            },
            py::Expr::BooleanLiteral(b) => ast::Expr::BoolLit {
                value: b.value,
                span,
            },
            py::Expr::NoneLiteral(_) => ast::Expr::NoneLit { span },
            py::Expr::StringLiteral(_) => ast::Expr::StrLit {
                text: self.text_at(e.range()),
                span,
            },
            py::Expr::BoolOp(b) => ast::Expr::BoolOp {
                op: lower_boolop(b.op),
                values: b.values.iter().map(|v| self.lower_expr(v)).collect(),
                span,
            },
            py::Expr::UnaryOp(u) => ast::Expr::UnaryOp {
                op: lower_unaryop(u.op),
                operand: Box::new(self.lower_expr(&u.operand)),
                span,
            },
            py::Expr::BinOp(b) => ast::Expr::BinOp {
                op: lower_binop(b.op),
                lhs: Box::new(self.lower_expr(&b.left)),
                rhs: Box::new(self.lower_expr(&b.right)),
                span,
            },
            py::Expr::Compare(c) => ast::Expr::Compare {
                left: Box::new(self.lower_expr(&c.left)),
                ops: c.ops.iter().map(|op| lower_cmpop(*op)).collect(),
                comparators: c.comparators.iter().map(|c| self.lower_expr(c)).collect(),
                span,
            },
            py::Expr::If(i) => ast::Expr::Ternary {
                test: Box::new(self.lower_expr(&i.test)),
                body: Box::new(self.lower_expr(&i.body)),
                orelse: Box::new(self.lower_expr(&i.orelse)),
                span,
            },
            py::Expr::Call(c) => ast::Expr::Call {
                func: Box::new(self.lower_expr(&c.func)),
                args: c
                    .arguments
                    .args
                    .iter()
                    .map(|a| self.lower_expr(a))
                    .collect(),
                keywords: c
                    .arguments
                    .keywords
                    .iter()
                    .map(|k| self.lower_keyword(k))
                    .collect(),
                span,
            },
            py::Expr::Attribute(a) => ast::Expr::Attribute {
                value: Box::new(self.lower_expr(&a.value)),
                attr: a.attr.as_str().to_string(),
                span,
            },
            py::Expr::Subscript(s) => ast::Expr::Subscript {
                value: Box::new(self.lower_expr(&s.value)),
                index: Box::new(self.lower_expr(&s.slice)),
                span,
            },
            py::Expr::Slice(s) => ast::Expr::Slice {
                lower: s.lower.as_deref().map(|l| Box::new(self.lower_expr(l))),
                upper: s.upper.as_deref().map(|u| Box::new(self.lower_expr(u))),
                step: s.step.as_deref().map(|s| Box::new(self.lower_expr(s))),
                span,
            },
            py::Expr::Tuple(t) => ast::Expr::Tuple {
                elts: t.iter().map(|e| self.lower_expr(e)).collect(),
                span,
            },
            py::Expr::List(l) => ast::Expr::List {
                elts: l.iter().map(|e| self.lower_expr(e)).collect(),
                span,
            },
            py::Expr::Named(_) => {
                self.unsupported_expr("a named expression ('walrus' :=)", e.range())
            }
            py::Expr::Lambda(_) => self.unsupported_expr("a 'lambda' expression", e.range()),
            py::Expr::Dict(_) => self.unsupported_expr("a dict display", e.range()),
            py::Expr::Set(_) => self.unsupported_expr("a set display", e.range()),
            py::Expr::ListComp(_) => self.unsupported_expr("a list comprehension", e.range()),
            py::Expr::SetComp(_) => self.unsupported_expr("a set comprehension", e.range()),
            py::Expr::DictComp(_) => self.unsupported_expr("a dict comprehension", e.range()),
            py::Expr::Generator(_) => self.unsupported_expr("a generator expression", e.range()),
            py::Expr::Await(_) => self.unsupported_expr("an 'await' expression", e.range()),
            py::Expr::Yield(_) => self.unsupported_expr("a 'yield' expression", e.range()),
            py::Expr::YieldFrom(_) => self.unsupported_expr("a 'yield from' expression", e.range()),
            py::Expr::FString(_) => self.unsupported_expr("an f-string", e.range()),
            py::Expr::TString(_) => self.unsupported_expr("a t-string", e.range()),
            py::Expr::BytesLiteral(_) => self.unsupported_expr("a bytes literal", e.range()),
            py::Expr::EllipsisLiteral(_) => self.unsupported_expr("an ellipsis literal", e.range()),
            py::Expr::Starred(_) => self.unsupported_expr("a starred expression", e.range()),
            py::Expr::IpyEscapeCommand(_) => {
                self.unsupported_expr("an IPython escape command", e.range())
            }
        }
    }

    fn lower_keyword(&mut self, k: &py::Keyword) -> ast::Keyword {
        ast::Keyword {
            name: k.arg.as_ref().map(|a| a.as_str().to_string()),
            value: self.lower_expr(&k.value),
            span: self.span(k.range()),
        }
    }
}

fn lower_boolop(op: py::BoolOp) -> ast::BoolOp {
    match op {
        py::BoolOp::And => ast::BoolOp::And,
        py::BoolOp::Or => ast::BoolOp::Or,
    }
}

fn lower_binop(op: py::Operator) -> ast::BinOp {
    match op {
        py::Operator::Add => ast::BinOp::Add,
        py::Operator::Sub => ast::BinOp::Sub,
        py::Operator::Mult => ast::BinOp::Mul,
        py::Operator::MatMult => ast::BinOp::MatMul,
        py::Operator::Div => ast::BinOp::Div,
        py::Operator::Mod => ast::BinOp::Mod,
        py::Operator::Pow => ast::BinOp::Pow,
        py::Operator::LShift => ast::BinOp::LShift,
        py::Operator::RShift => ast::BinOp::RShift,
        py::Operator::BitOr => ast::BinOp::BitOr,
        py::Operator::BitXor => ast::BinOp::BitXor,
        py::Operator::BitAnd => ast::BinOp::BitAnd,
        py::Operator::FloorDiv => ast::BinOp::FloorDiv,
    }
}

fn lower_unaryop(op: py::UnaryOp) -> ast::UnaryOp {
    match op {
        py::UnaryOp::Invert => ast::UnaryOp::Invert,
        py::UnaryOp::Not => ast::UnaryOp::Not,
        py::UnaryOp::UAdd => ast::UnaryOp::UAdd,
        py::UnaryOp::USub => ast::UnaryOp::USub,
    }
}

fn lower_cmpop(op: py::CmpOp) -> ast::CmpOp {
    match op {
        py::CmpOp::Eq => ast::CmpOp::Eq,
        py::CmpOp::NotEq => ast::CmpOp::NotEq,
        py::CmpOp::Lt => ast::CmpOp::Lt,
        py::CmpOp::LtE => ast::CmpOp::LtE,
        py::CmpOp::Gt => ast::CmpOp::Gt,
        py::CmpOp::GtE => ast::CmpOp::GtE,
        py::CmpOp::Is => ast::CmpOp::Is,
        py::CmpOp::IsNot => ast::CmpOp::IsNot,
        py::CmpOp::In => ast::CmpOp::In,
        py::CmpOp::NotIn => ast::CmpOp::NotIn,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse_ok(src: &str) -> ast::Module {
        let (module, diags) = parse(src);
        assert!(diags.is_empty(), "unexpected diags: {diags:?}");
        module
    }

    const VECTOR_ADD: &str = r#"
import triton
import triton.language as tl


@triton.jit
def vector_add(a_ptr, b_ptr, c_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n
    a = tl.load(a_ptr + offsets, mask=mask)
    b = tl.load(b_ptr + offsets, mask=mask)
    tl.store(c_ptr + offsets, a + b, mask=mask)
"#;

    #[test]
    fn parses_vector_add_signature_and_body() {
        let module = parse_ok(VECTOR_ADD);
        assert_eq!(module.kernels.len(), 1);
        let k = &module.kernels[0];
        assert_eq!(k.name, "vector_add");

        assert_eq!(k.params.len(), 5);
        let names: Vec<&str> = k.params.iter().map(|p| p.name.as_str()).collect();
        assert_eq!(names, ["a_ptr", "b_ptr", "c_ptr", "n", "BLOCK_SIZE"]);
        assert!(!k.params[0].is_constexpr);
        assert!(!k.params[3].is_constexpr);
        assert!(k.params[4].is_constexpr);
        assert!(matches!(
            k.params[4].annotation,
            Some(ast::Expr::Attribute { ref attr, .. }) if attr == "constexpr"
        ));

        // pid = tl.program_id(axis=0)
        assert!(matches!(k.body[0], ast::Stmt::Assign { .. }));
        // mask = offsets < n
        let ast::Stmt::Assign { value, .. } = &k.body[3] else {
            panic!("expected assignment, got {:?}", k.body[3]);
        };
        assert!(matches!(value, ast::Expr::Compare { .. }));

        // tl.store(c_ptr + offsets, a + b, mask=mask)
        let ast::Stmt::Expr { value, .. } = &k.body[6] else {
            panic!("expected expression statement, got {:?}", k.body[6]);
        };
        let ast::Expr::Call {
            func,
            args,
            keywords,
            ..
        } = value
        else {
            panic!("expected call, got {value:?}");
        };
        assert!(matches!(**func, ast::Expr::Attribute { ref attr, .. } if attr == "store"));
        assert_eq!(args.len(), 2);
        assert_eq!(keywords.len(), 1);
        assert_eq!(keywords[0].name.as_deref(), Some("mask"));
    }

    #[test]
    fn undecorated_function_is_not_a_kernel() {
        let module = parse_ok(
            r#"
def helper(x):
    return x + 1
"#,
        );
        assert!(module.kernels.is_empty());
    }

    #[test]
    fn call_form_decorator_is_recognized() {
        let module = parse_ok(
            r#"
import triton


@triton.jit()
def kernel(x):
    return x
"#,
        );
        assert_eq!(module.kernels.len(), 1);
        assert_eq!(module.kernels[0].name, "kernel");
    }

    #[test]
    fn autotune_stacked_above_jit_still_counts_as_a_kernel() {
        let module = parse_ok(
            r#"
import triton


@triton.autotune(configs=[], key=[])
@triton.jit
def kernel(x, BLOCK: tl.constexpr):
    return x
"#,
        );
        assert_eq!(module.kernels.len(), 1);
    }

    #[test]
    fn if_elif_else_flattens_to_nested_if() {
        let module = parse_ok(
            r#"
import triton


@triton.jit
def kernel(x):
    if x > 0:
        y = 1
    elif x < 0:
        y = -1
    else:
        y = 0
"#,
        );
        let ast::Stmt::If { orelse, .. } = &module.kernels[0].body[0] else {
            panic!("expected if");
        };
        assert_eq!(orelse.len(), 1);
        let ast::Stmt::If { orelse, .. } = &orelse[0] else {
            panic!("expected nested elif-if");
        };
        assert_eq!(orelse.len(), 1);
        assert!(matches!(orelse[0], ast::Stmt::Assign { .. }));
    }

    #[test]
    fn malformed_python_reports_a_diag_without_panicking() {
        let (_module, diags) = parse(
            r#"
import triton

x = (1 +
"#,
        );
        assert!(!diags.is_empty());
        assert_eq!(diags[0].code, basalt_diag::ECode::ParseError);
    }

    #[test]
    fn lambda_in_body_is_a_clean_diagnostic_not_a_panic() {
        let (module, diags) = parse(
            r#"
import triton


@triton.jit
def kernel(x):
    f = lambda v: v + 1
    return f
"#,
        );
        assert_eq!(module.kernels.len(), 1);
        assert!(!diags.is_empty());
        assert_eq!(diags[0].code, basalt_diag::ECode::ParseError);
        let ast::Stmt::Assign { value, .. } = &module.kernels[0].body[0] else {
            panic!("expected assignment");
        };
        assert!(matches!(value, ast::Expr::Error { .. }));
    }

    #[test]
    fn with_statement_in_body_is_reported_not_dropped_silently() {
        let (module, diags) = parse(
            r#"
import triton


@triton.jit
def kernel(x):
    with x:
        pass
    return x
"#,
        );
        assert_eq!(module.kernels.len(), 1);
        assert_eq!(module.kernels[0].body.len(), 2);
        assert!(matches!(module.kernels[0].body[0], ast::Stmt::Error { .. }));
        assert!(!diags.is_empty());
    }

    #[test]
    fn aliased_bare_jit_import_is_not_guessed_at() {
        // `from triton import jit` then `@jit` is a real style but is not detected: telling
        // it apart from an unrelated `jit` decorator would require tracking imports, which is
        // out of scope here (see `is_triton_jit`).
        let module = parse_ok(
            r#"
from triton import jit


@jit
def kernel(x):
    return x
"#,
        );
        assert!(module.kernels.is_empty());
    }
}
