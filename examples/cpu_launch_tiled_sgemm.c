/* Host-side proof for the hand-built `tiled_sgemm_f32` BIR fixture (see
 * crates/basalt-x86/tests/tiled_sgemm.rs): `D = A@B + C` at M=N=K=32, row-major throughout,
 * f32 in/acc, decomposed into a 2x2 grid of 16x16 output tiles each accumulated over two
 * 16-deep K-steps. All four params are ptr.global (integer-class), so a/b/c/d arrive in
 * rdi/rsi/rdx/rcx and nthreads (unused — the kernel never reads tid.x/bdim.x) is the fifth
 * integer register, r8.
 *
 * `d` is allocated wider than the real 32x32 output: the kernel stages every tile into
 * scratch space appended past the real 1024 floats (see the BIR fixture's own doc comment for
 * why), so the host must provide that extra room even though only the first 1024 floats are
 * ever compared here.
 *
 * A[i][j] = ((3*i+j) % 7) + 1, B[i][j] = ((i+5*j) % 7) + 1, C[i][j] = (i+j) % 4 — small
 * position-varying integers, so every product and every partial sum along a 32-deep dot
 * product stays far below 2^24 (exactly representable in f32, order-independent), making the
 * reference computed by the plain triple loop below an exact, unambiguous target: no ULP
 * tolerance needed on this side.
 */
#include <stdint.h>
#include <stdio.h>

extern void tiled_sgemm_f32(const float *a, const float *b, const float *c, float *d,
                             int64_t nthreads);

#define N 32
#define D_WORDS (1024 + 768) /* 1024 real output floats + 768 words of tile scratch */

int main(void) {
    float a[N * N], b[N * N], c[N * N];
    float d[D_WORDS];
    float expected[N * N];

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            a[i * N + j] = (float)((3 * i + j) % 7 + 1);
            b[i * N + j] = (float)((i + 5 * j) % 7 + 1);
            c[i * N + j] = (float)((i + j) % 4);
        }
    }
    for (int i = 0; i < D_WORDS; i++) {
        d[i] = -1.0f;
    }

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            float sum = c[i * N + j];
            for (int k = 0; k < N; k++) {
                sum += a[i * N + k] * b[k * N + j];
            }
            expected[i * N + j] = sum;
        }
    }

    tiled_sgemm_f32(a, b, c, d, 1);

    int ok = 1;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            float got = d[i * N + j];
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
    printf("PASS: tiled_sgemm_f32 computed D = A@B + C correctly (32x32, 2x2 tiles, "
           "K-accumulated over 2 steps)\n");
    return 0;
}
