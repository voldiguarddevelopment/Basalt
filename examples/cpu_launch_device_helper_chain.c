/* Host-side proof that the x86-64 oracle's emitted object for device_helper_chain.cu links and
 * runs, and that a three-level-deep __device__-to-__device__ call chain (negate_then_scale ->
 * scale_then_inc -> inc) actually carries its argument and return value correctly through every
 * link, not just the one hop from the kernel to its first helper.
 *
 * chain_vector's BIR params are (ptr.global, ptr.global, i32) -- all three integer-class, so
 * under SysV they consume rdi/rsi/rdx in order, with the trailing `nthreads` argument (see
 * crates/basalt-x86/src/oracle.rs's own module header) landing in rcx and always read back a
 * full 8 bytes, hence int64_t here. `inc`/`scale_then_inc`/`negate_then_scale` are never called
 * directly from here: each is a same-object helper only its own caller's emitted code reaches,
 * via a real intra-object `call rel32` this test indirectly exercises three deep.
 */
#include <stdint.h>
#include <stdio.h>

extern void chain_vector(const int *in, int *out, int n, int64_t nthreads);

#define N 1024

int main(void) {
    int in[N], out[N];
    for (int i = 0; i < N; i++) {
        in[i] = i - (N / 2);
        out[i] = 0x7fffffff;
    }

    chain_vector(in, out, N, (int64_t)N);

    for (int i = 0; i < N; i++) {
        int expected = -2 * in[i] + 1;
        if (out[i] != expected) {
            fprintf(stderr, "FAIL at index %d: expected %d, got %d\n", i, expected, out[i]);
            return 1;
        }
    }

    printf(
        "PASS: chain_vector called negate_then_scale -> scale_then_inc -> inc and matched "
        "-2*in[i]+1 for all %d elements\n",
        N);
    return 0;
}
