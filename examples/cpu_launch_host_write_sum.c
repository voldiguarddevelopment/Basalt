/* Host-side proof that a hand-built host function's own real intra-object `call` to a
 * launched kernel (crates/basalt-x86/tests/link_and_run.rs's
 * `hand_built_host_launches_kernel_links_and_runs`) actually runs: `host_write_sum` takes one
 * pointer argument (an ordinary function, not a kernel — no synthesized trailing `nthreads`
 * argument here, unlike every other shim in this directory) and launches `write_sum(out, 10,
 * 20)` internally, which stores `10 + 20` through `out`.
 */
#include <stdint.h>
#include <stdio.h>

extern void host_write_sum(int32_t *out);

int main(void) {
    int32_t out = -1;

    host_write_sum(&out);

    if (out != 30) {
        fprintf(stderr, "FAIL: expected 30, got %d\n", out);
        return 1;
    }

    printf("PASS: host_write_sum wrote a+b via a real intra-object call\n");
    return 0;
}
