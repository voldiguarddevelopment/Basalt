/* Host-side proof that the x86-64 oracle's emitted object for vector_add.cu links and runs.
 *
 * vector_add's BIR params are (ptr.global, ptr.global, ptr.global, i32) — all four
 * integer-class, so under SysV they consume rdi/rsi/rdx/rcx in order. The oracle's calling
 * convention (crates/basalt-x86/src/oracle.rs) always appends one trailing integer-class
 * `nthreads` argument after a function's own params, landing in the next integer register
 * (r8 here) and read back a full 8 bytes regardless of the C-side type — hence int64_t, not
 * int, so the caller is required to actually widen it into r8 rather than leaving the upper
 * 32 bits unspecified. The kernel itself returns void, so there is no return value to check
 * beyond what it wrote through the pointers.
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

extern void vector_add(const float *a, const float *b, float *c, int n, int64_t nthreads);

#define N 1024

int main(void) {
    float a[N], b[N], c[N];
    for (int i = 0; i < N; i++) {
        a[i] = (float)i;
        b[i] = (float)(i * 2);
        c[i] = -1.0f;
    }

    vector_add(a, b, c, N, (int64_t)N);

    for (int i = 0; i < N; i++) {
        float expected = a[i] + b[i];
        if (c[i] != expected) {
            fprintf(stderr, "FAIL at index %d: expected %f, got %f\n", i, (double)expected,
                    (double)c[i]);
            return 1;
        }
    }

    printf("PASS: vector_add matched a[i]+b[i] for all %d elements\n", N);
    return 0;
}
