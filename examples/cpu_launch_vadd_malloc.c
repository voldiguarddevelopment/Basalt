/* Host-side proof that a real `.cu` host function (`tests/kernels/cpu_launch_vadd_malloc.cu`)
 * allocating its own device buffers via real `cudaMalloc`/`cudaMemcpy`/`cudaFree` calls -- not
 * relying on this driver to pre-allocate them, unlike `cpu_launch_vadd_host.c` -- actually
 * allocates, copies, computes, copies back, and frees correctly. `h_a`/`h_b`/`h_c` here are
 * ordinary host buffers this driver owns; every `d_a`/`d_b`/`d_c` device buffer is allocated,
 * used, and released entirely inside `launch_vector_add_malloc` itself.
 */
#include <stdio.h>

extern void launch_vector_add_malloc(const float *h_a, const float *h_b, float *h_c, int n);

#define N 1024

int main(void) {
    float h_a[N], h_b[N], h_c[N];
    for (int i = 0; i < N; i++) {
        h_a[i] = (float)i;
        h_b[i] = (float)(i * 2);
        h_c[i] = -1.0f;
    }

    launch_vector_add_malloc(h_a, h_b, h_c, N);

    for (int i = 0; i < N; i++) {
        float expected = h_a[i] + h_b[i];
        if (h_c[i] != expected) {
            fprintf(stderr, "FAIL at index %d: expected %f, got %f\n", i, (double)expected,
                    (double)h_c[i]);
            return 1;
        }
    }

    printf("PASS: launch_vector_add_malloc allocated/copied/freed real device memory and "
           "matched a[i]+b[i] for all %d elements\n",
           N);
    return 0;
}
