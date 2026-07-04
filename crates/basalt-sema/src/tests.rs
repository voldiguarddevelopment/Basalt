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

// ---- CUDA qualifiers and builtins -------------------------------------------------------

#[test]
fn valid_global_kernel_with_void_return_has_no_diagnostics() {
    let tu = parse_ok(
        r#"
        __global__ void kernel(float *out, int n) {
            int i = n;
            out[i] = 0.0f;
        }
        "#,
    );
    let diags = check(&tu);
    assert!(diags.is_empty(), "expected no diagnostics, got {diags:?}");
}

#[test]
fn global_kernel_with_non_void_return_reports_e303() {
    let tu = parse_ok(
        r#"
        __global__ int kernel(int n) {
            return n;
        }
        "#,
    );
    let diags = check(&tu);
    assert!(codes(&diags).contains(&ECode::InvalidCudaQualifier));
}

#[test]
fn combined_host_device_function_is_valid() {
    let tu = parse_ok(
        r#"
        __host__ __device__ int add(int a, int b) {
            return a + b;
        }
        "#,
    );
    let diags = check(&tu);
    assert!(diags.is_empty(), "expected no diagnostics, got {diags:?}");
}

#[test]
fn global_combined_with_device_reports_e303() {
    let tu = parse_ok(
        r#"
        __global__ __device__ void kernel(int n) {
            int x = n;
        }
        "#,
    );
    let diags = check(&tu);
    assert!(codes(&diags).contains(&ECode::InvalidCudaQualifier));
}

#[test]
fn shared_and_constant_variable_declarations_are_recognized_without_error() {
    let tu = parse_ok(
        r#"
        __device__ void kernel() {
            __shared__ float tile[16];
            __constant__ int table[4];
            tile[0] = 0.0f;
        }
        "#,
    );
    let diags = check(&tu);
    assert!(diags.is_empty(), "expected no diagnostics, got {diags:?}");
}

#[test]
fn shared_and_constant_together_on_one_variable_reports_e303() {
    let tu = parse_ok(
        r#"
        __device__ void kernel() {
            __shared__ __constant__ int x;
        }
        "#,
    );
    let diags = check(&tu);
    assert!(codes(&diags).contains(&ECode::InvalidCudaQualifier));
}

#[test]
fn kernel_body_can_use_dim3_builtins_as_unsigned_integers() {
    let tu = parse_ok(
        r#"
        __global__ void kernel(unsigned int *out) {
            unsigned int i = threadIdx.x;
            unsigned int j = blockIdx.y;
            unsigned int k = blockDim.z;
            unsigned int l = gridDim.x;
            out[0] = i + j + k + l;
        }
        "#,
    );
    let diags = check(&tu);
    assert!(diags.is_empty(), "expected no diagnostics, got {diags:?}");
}

#[test]
fn syncthreads_with_no_arguments_has_no_error() {
    let tu = parse_ok(
        r#"
        __global__ void kernel() {
            __syncthreads();
        }
        "#,
    );
    let diags = check(&tu);
    assert!(diags.is_empty(), "expected no diagnostics, got {diags:?}");
}

#[test]
fn syncthreads_with_an_argument_reports_arity_error() {
    let tu = parse_ok(
        r#"
        __global__ void kernel() {
            __syncthreads(1);
        }
        "#,
    );
    let diags = check(&tu);
    assert!(codes(&diags).contains(&ECode::TypeError));
}

// This crate deliberately makes the builtins available only inside a `__global__`/`__device__`
// body (see checker.rs's module header for the reasoning): an ordinary host function gets no
// special seeding, so `threadIdx`/`__syncthreads` there are unresolved identifiers exactly like
// any other undeclared name, reported the normal way (`E301`).
#[test]
fn builtins_are_not_available_in_an_ordinary_host_function() {
    let tu = parse_ok(
        r#"
        int host_fn() {
            int i = threadIdx.x;
            return i;
        }
        "#,
    );
    let diags = check(&tu);
    assert!(codes(&diags).contains(&ECode::UndefinedSymbol));
}

// ---- const enforcement (P13-T5) ----------------------------------------------------------

#[test]
fn assigning_to_a_const_variable_reports_e308() {
    let tu = parse_ok(
        r#"
        int main() {
            const int n = 5;
            n = 10;
            return n;
        }
        "#,
    );
    let diags = check(&tu);
    assert!(codes(&diags).contains(&ECode::ConstViolation));
}

#[test]
fn assigning_to_a_const_parameter_reports_e308() {
    let tu = parse_ok(
        r#"
        int f(const int n) {
            n = 10;
            return n;
        }
        "#,
    );
    let diags = check(&tu);
    assert!(codes(&diags).contains(&ECode::ConstViolation));
}

#[test]
fn assigning_through_a_pointer_to_const_reports_e308() {
    let tu = parse_ok(
        r#"
        int main() {
            int n = 5;
            const int *p = &n;
            *p = 10;
            return n;
        }
        "#,
    );
    let diags = check(&tu);
    assert!(codes(&diags).contains(&ECode::ConstViolation));
}

#[test]
fn reassigning_a_pointer_to_const_itself_is_not_a_const_violation() {
    let tu = parse_ok(
        r#"
        int main() {
            int n = 5;
            int m = 6;
            const int *p = &n;
            p = &m;
            return *p;
        }
        "#,
    );
    let diags = check(&tu);
    assert!(!codes(&diags).contains(&ECode::ConstViolation));
}

#[test]
fn reassigning_a_const_pointer_itself_reports_e308() {
    let tu = parse_ok(
        r#"
        int main() {
            int n = 5;
            int m = 6;
            int *const p = &n;
            p = &m;
            return *p;
        }
        "#,
    );
    let diags = check(&tu);
    assert!(codes(&diags).contains(&ECode::ConstViolation));
}

#[test]
fn assigning_through_a_const_pointer_to_its_non_const_pointee_is_not_a_const_violation() {
    let tu = parse_ok(
        r#"
        int main() {
            int n = 5;
            int *const p = &n;
            *p = 10;
            return n;
        }
        "#,
    );
    let diags = check(&tu);
    assert!(!codes(&diags).contains(&ECode::ConstViolation));
}

#[test]
fn assigning_to_a_const_struct_member_via_dot_reports_e308() {
    let tu = parse_ok(
        r#"
        struct Point { const int x; int y; };
        int main() {
            struct Point p;
            p.x = 10;
            return p.y;
        }
        "#,
    );
    let diags = check(&tu);
    assert!(codes(&diags).contains(&ECode::ConstViolation));
}

#[test]
fn assigning_to_a_const_struct_member_via_arrow_reports_e308() {
    let tu = parse_ok(
        r#"
        struct Point { const int x; int y; };
        int main() {
            struct Point p;
            struct Point *q = &p;
            q->x = 10;
            return q->y;
        }
        "#,
    );
    let diags = check(&tu);
    assert!(codes(&diags).contains(&ECode::ConstViolation));
}

#[test]
fn assigning_to_a_plain_non_const_struct_member_has_no_diagnostics() {
    let tu = parse_ok(
        r#"
        struct Point { int x; int y; };
        int main() {
            struct Point p;
            p.x = 10;
            return p.y;
        }
        "#,
    );
    let diags = check(&tu);
    assert!(diags.is_empty(), "expected no diagnostics, got {diags:?}");
}

#[test]
fn plain_variable_parameter_and_pointer_assignment_has_no_diagnostics() {
    let tu = parse_ok(
        r#"
        int f(int n) {
            n = n + 1;
            int *p = &n;
            *p = 2;
            return n;
        }
        "#,
    );
    let diags = check(&tu);
    assert!(diags.is_empty(), "expected no diagnostics, got {diags:?}");
}
