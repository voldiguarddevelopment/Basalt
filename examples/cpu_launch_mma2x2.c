/* Host-side proof for the hand-built `mma2x2` BIR fixture (see
 * crates/basalt-x86/tests/link_and_run.rs): `D = A*B + C` at M=N=K=2, row-major A/B, f32
 * throughout. All four params are ptr.global (integer-class), so a/b/c/d arrive in
 * rdi/rsi/rdx/rcx and nthreads (unused by mma2x2's own computation, which never reads
 * tid.x/bdim.x) is the fifth integer register, r8.
 *
 * A = [[1,2],[3,4]], B = [[5,6],[7,8]], C = [[0.5,0.5],[0.5,0.5]]:
 *   A*B = [[1*5+2*7, 1*6+2*8], [3*5+4*7, 3*6+4*8]] = [[19,22],[43,50]]
 *   D = A*B + C = [[19.5,22.5],[43.5,50.5]]
 */
#include <stdint.h>
#include <stdio.h>

extern void mma2x2(const float *a, const float *b, const float *c, float *d, int64_t nthreads);

int main(void) {
    float a[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float b[4] = {5.0f, 6.0f, 7.0f, 8.0f};
    float c[4] = {0.5f, 0.5f, 0.5f, 0.5f};
    float d[4] = {-1.0f, -1.0f, -1.0f, -1.0f};
    float expected[4] = {19.5f, 22.5f, 43.5f, 50.5f};

    mma2x2(a, b, c, d, 1);

    int ok = 1;
    for (int i = 0; i < 4; i++) {
        if (d[i] != expected[i]) {
            fprintf(stderr, "FAIL at index %d: expected %f, got %f\n", i, (double)expected[i],
                    (double)d[i]);
            ok = 0;
        }
    }
    if (!ok) {
        return 1;
    }
    printf("PASS: mma2x2 computed D = A*B + C correctly\n");
    return 0;
}
