/* Host-side proof that the x86-64 oracle's emitted object for device_helper_square.cu links and
 * runs, and that the __device__ helper's own return value actually reaches the kernel's output.
 *
 * square_vector's BIR params are (ptr.global, ptr.global, i32) -- all three integer-class, so
 * under SysV they consume rdi/rsi/rdx in order. The oracle's calling convention (see
 * crates/basalt-x86/src/oracle.rs's own module header) always appends one trailing
 * integer-class `nthreads` argument after a function's own params, landing in the next integer
 * register (rcx here) and read back a full 8 bytes regardless of the C-side type -- hence
 * int64_t, not int, so the caller is required to actually widen it into rcx rather than leaving
 * the upper 32 bits unspecified. `square` itself is never called directly from here: it is a
 * same-object helper only `square_vector`'s own emitted code calls, via a real intra-object
 * `call rel32` this test indirectly exercises.
 */
#include <stdint.h>
#include <stdio.h>

extern void square_vector(const int *in, int *out, int n, int64_t nthreads);

#define N 1024

int main(void) {
    int in[N], out[N];
    for (int i = 0; i < N; i++) {
        in[i] = i - (N / 2);
        out[i] = -1;
    }

    square_vector(in, out, N, (int64_t)N);

    for (int i = 0; i < N; i++) {
        int expected = in[i] * in[i];
        if (out[i] != expected) {
            fprintf(stderr, "FAIL at index %d: expected %d, got %d\n", i, expected, out[i]);
            return 1;
        }
    }

    printf(
        "PASS: square_vector called the __device__ helper square() and matched in[i]*in[i] "
        "for all %d elements\n",
        N);
    return 0;
}
