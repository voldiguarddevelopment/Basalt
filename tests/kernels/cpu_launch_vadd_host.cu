// Real end-to-end host+device compile proof (P13-T1c-i): a genuine host function launching
// `vector_add.cu`'s own, unmodified kernel via real `<<<grid, block>>>` syntax. `a`/`b`/`c` are
// ordinary pointer parameters — the buffers themselves are pre-allocated by this test's own C
// driver (`examples/cpu_launch_vadd_host.c`), not by this file (no `cudaMalloc`, which is
// P13-T1c-ii's job). `block`/`grid` are plain scalar locals, not arrays: a local *array* this
// backend's own oracle would need a real multi-byte stack home for is a separate, general gap
// this task does not touch (see `crates/basalt-x86/src/oracle.rs`'s `Frame` — every local gets
// a uniform 8-byte slot today).
#include "vector_add.cu"

void launch_vector_add(float *a, float *b, float *c, int n) {
    int block = 256;
    int grid = (n + block - 1) / block;
    vector_add<<<grid, block>>>(a, b, c, n);
    // This backend's own launches already run to completion synchronously inside the `call`
    // above (see oracle.rs's own module header); real here anyway, to exercise the real
    // no-op lowering end to end rather than only in a unit test.
    cudaDeviceSynchronize();
}
