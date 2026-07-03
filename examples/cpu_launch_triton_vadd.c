/* Host-side proof for `basalt_sema::lower_triton`'s masked-load/store lowering
 * (crates/basalt-x86/tests/triton_link_and_run.rs): a real `@triton.jit` `vector_add` kernel,
 * launched with exactly one program (`tl.program_id` always reads 0 under the oracle's own
 * single-block scope — see `basalt-x86/src/oracle.rs`'s module header) whose block covers more
 * elements (BLOCK) than the real array holds (N). Every index in `[N, BLOCK)` is masked out by
 * `offsets < n`; the mask must genuinely prevent both the load (of `a`/`b`, which are only `N`
 * floats wide, not `BLOCK`) and the store (into `c`, which is `BLOCK` floats wide but pre-
 * poisoned) from executing for those lanes.
 *
 * Triton kernel params are lowered as: pointer params stay `ptr.global`; every ordinary
 * (non-`constexpr`) and `constexpr` scalar param lowers as `i64` uniformly (see
 * `triton_lower.rs`'s own module header) — hence `int64_t n`/`int64_t BLOCK_SIZE` here, not
 * `int`. All five real params are integer-class (three pointers, two `i64`), so `nthreads`
 * (unused: this kernel never reads `tid.x`/`bdim.x`) is the sixth integer register.
 *
 * `c_ptr` is also this kernel's *scratch* pointer (the last pointer-typed parameter — see
 * `triton_lower.rs`'s `Storage::Scratch`): since BIR has no `alloca`, every tile this kernel
 * materializes (`offsets`, `mask`, `a`, `b` — four tiles) is carved out of real bytes past
 * `c_ptr`'s own real `BLOCK`-float payload, at `SCRATCH_BASE_BYTES + ordinal * TILE_STRIDE`
 * (16384 bytes each). `c` must therefore actually be allocated far wider than its real `BLOCK`
 * elements; `C_WORDS` below is deliberately generous (mirrors `cpu_launch_tiled_sgemm.c`'s own
 * `D_WORDS` convention for exactly the same "BIR has no alloca" reason), and only `c[0..BLOCK)`
 * is ever compared against the expected result.
 */
#include <stdint.h>
#include <stdio.h>

extern void vector_add(const float *a_ptr, const float *b_ptr, float *c_ptr, int64_t n,
                        int64_t BLOCK_SIZE, int64_t nthreads);

#define N 1000
#define BLOCK 1024
#define C_WORDS 32768 /* BLOCK real floats + generous tile-scratch headroom */

int main(void) {
    float a[N], b[N];
    float c[C_WORDS];

    for (int i = 0; i < N; i++) {
        a[i] = (float)i;
        b[i] = (float)(i * 2);
    }
    for (int i = 0; i < C_WORDS; i++) {
        c[i] = -1.0f;
    }

    vector_add(a, b, c, N, BLOCK, 1);

    int ok = 1;
    for (int i = 0; i < BLOCK; i++) {
        float want = (i < N) ? (a[i] + b[i]) : -1.0f;
        if (c[i] != want) {
            fprintf(stderr, "FAIL at index %d: expected %f, got %f\n", i, (double)want,
                    (double)c[i]);
            ok = 0;
        }
    }
    if (!ok) {
        return 1;
    }
    printf("PASS: masked triton vector_add computed c[i]=a[i]+b[i] for i<%d and left the "
           "poisoned c[i] untouched for %d<=i<%d (BLOCK=%d, not a multiple of N)\n",
           N, N, BLOCK, BLOCK);
    return 0;
}
