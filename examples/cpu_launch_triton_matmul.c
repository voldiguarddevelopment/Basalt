/* Host-side proof for `basalt_sema::lower_triton`'s `tl.dot` lowering
 * (crates/basalt-x86/tests/triton_link_and_run.rs): a real `@triton.jit` matmul kernel,
 * `D = A@B + C` at M=4, K=5, N=3 (deliberately non-square, to exercise row-major addressing
 * with distinct strides on every operand), one program covering the whole result in a single
 * `tl.dot` call — no masking, no Triton-level K-loop, since those are separately covered by
 * `cpu_launch_triton_vadd.c` and this file's job is `tl.dot` specifically.
 *
 * `tl.dot` never lowers to BIR's `Op::Mma` (see `triton_lower.rs`'s own module header for why:
 * `basalt-ptx` and the x86-64 regalloc backend both refuse `Op::Mma` outright, so Phase 10's
 * `--nvidia-ptx` exit criterion could never pass if it did) — this test is exactly what proves
 * the real scalar triple-loop it lowers to instead computes the right answer.
 *
 * A[i][j] = ((3*i+j) % 7) + 1, B[i][j] = ((i+5*j) % 7) + 1, C[i][j] = (i+j) % 4 (mirrors
 * `cpu_launch_tiled_sgemm.c`'s own small position-varying integers): every product and partial
 * sum along the 5-deep dot product stays far below 2^24, so the plain triple-loop reference
 * below is an exact, order-independent target — no ULP tolerance needed.
 *
 * The kernel's `M`/`N` are `constexpr` with a literal default (`M: tl.constexpr = 4`); a
 * resolved `constexpr` is never a runtime argument (see `triton_lower.rs`'s `lower_kernel`),
 * so only `K` (left symbolic, no default — this pass's only way to give a `constexpr` a
 * runtime value at all) actually reaches the ABI. Four pointers + `K` + `nthreads` is six
 * integer-class arguments, exactly the oracle's SysV register budget; an M/N/K all left
 * runtime would overflow it (`E093`) before this fix.
 *
 * `out_ptr` is also this kernel's *scratch* pointer (the last pointer-typed parameter — see
 * `triton_lower.rs`'s `Storage::Scratch`): this kernel materializes eleven tiles (`rm`, `rn`,
 * `rk`, `a_ptrs`, `b_ptrs`, `c_ptrs`, `a`, `b`, `c`, `acc`, `out_ptrs`), each carved out of real
 * bytes past `out_ptr`'s own real `M*N`-float payload at `SCRATCH_BASE_BYTES + ordinal *
 * TILE_STRIDE` (16384 bytes each — mirrors `cpu_launch_triton_vadd.c`'s own `C_WORDS`
 * convention, and ultimately `cpu_launch_tiled_sgemm.c`'s `D_WORDS`, for the same "BIR has no
 * alloca" reason). `OUT_WORDS` is deliberately generous; only `out[0..M*N)` is ever compared.
 */
#include <stdint.h>
#include <stdio.h>

extern void matmul_kernel(const float *a_ptr, const float *b_ptr, const float *c_ptr,
                           float *out_ptr, int64_t K, int64_t nthreads);

#define M 4
#define N 3
#define K 5
#define OUT_WORDS 65536 /* M*N real floats + generous tile-scratch headroom */

int main(void) {
    float a[M * K], b[K * N], c[M * N], out[OUT_WORDS], expected[M * N];

    for (int i = 0; i < M; i++) {
        for (int k = 0; k < K; k++) {
            a[i * K + k] = (float)((3 * i + k) % 7 + 1);
        }
    }
    for (int k = 0; k < K; k++) {
        for (int j = 0; j < N; j++) {
            b[k * N + j] = (float)((k + 5 * j) % 7 + 1);
        }
    }
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            c[i * N + j] = (float)((i + j) % 4);
        }
    }
    for (int i = 0; i < M * N; i++) {
        out[i] = -1.0f;
    }

    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = c[i * N + j];
            for (int k = 0; k < K; k++) {
                sum += a[i * K + k] * b[k * N + j];
            }
            expected[i * N + j] = sum;
        }
    }

    matmul_kernel(a, b, c, out, K, 1);

    int ok = 1;
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float got = out[i * N + j];
            float want = expected[i * N + j];
            if (got != want) {
                fprintf(stderr, "FAIL at (%d,%d): expected %f, got %f\n", i, j, (double)want,
                        (double)got);
                ok = 0;
            }
        }
    }
    if (!ok) {
        return 1;
    }
    printf("PASS: triton matmul_kernel computed D = A@B + C correctly (%dx%dx%d, single "
           "program, real tl.dot triple loop)\n",
           M, N, K);
    return 0;
}
