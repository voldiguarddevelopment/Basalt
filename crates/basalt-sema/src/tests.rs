// Fixture-driven tests: parse a small source snippet with the real `basalt-frontend-c`
// pipeline, run `check`, and assert on the collected `ECode`s (never on message text, per the
// project's diagnostics contract).

use basalt_diag::ECode;
use basalt_frontend_c::ast::TranslationUnit;
use basalt_frontend_c::{lex, parse};

use crate::check;

fn parse_ok(src: &str) -> TranslationUnit {
    let (tokens, lex_errs) = lex(src);
    assert!(lex_errs.is_empty(), "lex errors: {lex_errs:?}");
    let (tu, parse_errs) = parse(&tokens);
    assert!(parse_errs.is_empty(), "parse errors: {parse_errs:?}");
    tu
}

fn codes(diags: &[basalt_diag::Diag]) -> Vec<ECode> {
    diags.iter().map(|d| d.code).collect()
}

#[test]
fn valid_program_has_no_diagnostics() {
    let tu = parse_ok(
        r#"
        int add(int a, int b) {
            return a + b;
        }
        int main() {
            int x = 1;
            int y = 2;
            int z = add(x, y);
            if (z > 0) {
                z = z - 1;
            }
            return z;
        }
        "#,
    );
    let diags = check(&tu);
    assert!(diags.is_empty(), "expected no diagnostics, got {diags:?}");
}

#[test]
fn undefined_variable_reference_reports_e301() {
    let tu = parse_ok(
        r#"
        int main() {
            int x = y;
            return x;
        }
        "#,
    );
    let diags = check(&tu);
    assert!(codes(&diags).contains(&ECode::UndefinedSymbol));
}

#[test]
fn type_mismatch_assignment_reports_e300() {
    let tu = parse_ok(
        r#"
        struct Point { int x; int y; };
        int main() {
            struct Point p;
            int a = p;
            return a;
        }
        "#,
    );
    let diags = check(&tu);
    assert!(codes(&diags).contains(&ECode::TypeError));
}

#[test]
fn redefinition_in_same_scope_reports_e302() {
    let tu = parse_ok(
        r#"
        int main() {
            int x = 1;
            int x = 2;
            return x;
        }
        "#,
    );
    let diags = check(&tu);
    assert!(codes(&diags).contains(&ECode::Redefinition));
}

#[test]
fn shadowing_in_nested_block_is_not_an_error() {
    let tu = parse_ok(
        r#"
        int main() {
            int x = 1;
            {
                int x = 2;
                x = x + 1;
            }
            return x;
        }
        "#,
    );
    let diags = check(&tu);
    assert!(
        diags.is_empty(),
        "shadowing should not error, got {diags:?}"
    );
}

#[test]
fn call_arity_mismatch_reports_e300() {
    let tu = parse_ok(
        r#"
        int add(int a, int b) {
            return a + b;
        }
        int main() {
            int z = add(1);
            return z;
        }
        "#,
    );
    let diags = check(&tu);
    assert!(codes(&diags).contains(&ECode::TypeError));
}

#[test]
fn member_access_on_unknown_field_reports_e301() {
    let tu = parse_ok(
        r#"
        struct Point { int x; int y; };
        int main() {
            struct Point p;
            int a = p.z;
            return a;
        }
        "#,
    );
    let diags = check(&tu);
    assert!(codes(&diags).contains(&ECode::UndefinedSymbol));
}

#[test]
fn many_independent_errors_are_all_reported_without_hanging() {
    let tu = parse_ok(
        r#"
        struct Point { int x; int y; };
        int main() {
            int a = undefined_one;
            int a = 2;
            struct Point p;
            int b = p;
            int c = undefined_two;
            return c;
        }
        "#,
    );
    let diags = check(&tu);
    let cs = codes(&diags);
    assert!(cs.iter().filter(|c| **c == ECode::UndefinedSymbol).count() >= 2);
    assert!(cs.contains(&ECode::Redefinition));
    assert!(cs.contains(&ECode::TypeError));
    assert!(diags.len() >= 4);
}

#[test]
fn struct_typedef_and_enum_resolve_end_to_end() {
    // Local variables can't be declared with a bare typedef name in this frontend (see
    // `ast.rs`'s note on `Type::Named` / `parse::Parser::next_starts_type`: a bare identifier
    // is never recognized as a type without a symbol table, so `PointT p;` inside a function
    // body does not parse). A typedef name is unambiguous in item position (a function's
    // return/parameter types), so that's where this test exercises it; locals use the tag
    // form (`struct Point p;`), which the frontend does parse everywhere.
    let tu = parse_ok(
        r#"
        struct Point { int x; int y; };
        typedef struct Point PointT;
        enum Color { Red, Green, Blue };

        PointT make_point(int x, int y) {
            struct Point p;
            p.x = x;
            p.y = y;
            return p;
        }

        int main() {
            struct Point p;
            p.x = 1;
            p.y = 2;
            enum Color c = Red;
            int sum = p.x + p.y + c;
            return sum;
        }
        "#,
    );
    let diags = check(&tu);
    assert!(diags.is_empty(), "expected no diagnostics, got {diags:?}");
}

#[test]
fn break_outside_loop_reports_e300() {
    let tu = parse_ok(
        r#"
        int main() {
            break;
            return 0;
        }
        "#,
    );
    let diags = check(&tu);
    assert!(codes(&diags).contains(&ECode::TypeError));
}

#[test]
fn goto_undefined_label_reports_e301() {
    let tu = parse_ok(
        r#"
        int main() {
            goto nowhere;
            return 0;
        }
        "#,
    );
    let diags = check(&tu);
    assert!(codes(&diags).contains(&ECode::UndefinedSymbol));
}
