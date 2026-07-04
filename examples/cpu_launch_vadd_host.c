/* Host-side proof that a real `.cu` host function (`tests/kernels/cpu_launch_vadd_host.cu`)
 * launching `vector_add.cu`'s own kernel via genuine `<<<>>>` syntax actually runs correctly.
 * Unlike `cpu_launch_vadd.c` (which calls the kernel directly, bypassing any launch machinery),
 * this shim calls `launch_vector_add` — an ordinary compiled function, not a kernel, so there is
 * no synthesized trailing `nthreads` argument here: the launch's own `grid`/`block` and the
 * resulting call to `vector_add` are entirely `cpu_launch_vadd_host.cu`'s and the oracle's own
 * doing.
 */
#include <stdio.h>

extern void launch_vector_add(float *a, float *b, float *c, int n);

#define N 1024

int main(void) {
    float a[N], b[N], c[N];
    for (int i = 0; i < N; i++) {
        a[i] = (float)i;
        b[i] = (float)(i * 2);
        c[i] = -1.0f;
    }

    launch_vector_add(a, b, c, N);

    for (int i = 0; i < N; i++) {
        float expected = a[i] + b[i];
        if (c[i] != expected) {
            fprintf(stderr, "FAIL at index %d: expected %f, got %f\n", i, (double)expected,
                    (double)c[i]);
            return 1;
        }
    }

    printf("PASS: launch_vector_add matched a[i]+b[i] for all %d elements via a real <<<>>> "
           "launch\n",
           N);
    return 0;
}
