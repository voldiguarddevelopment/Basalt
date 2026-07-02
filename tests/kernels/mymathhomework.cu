// Compile-time arithmetic exercise: every value here is knowable at parse time, and one
// temporary is computed but never read. Exists to give the mid-end pipeline something concrete
// to fold and delete — `a` and `b` collapse to the single constant 20, and `unused` disappears
// entirely — rather than merely asserting the reduction in prose (see
// crates/basalt-passes/tests/pipeline.rs, which counts instructions before and after).
__global__ void mymathhomework(int *out) {
    int a = 2 + 3;
    int b = a * 4;
    int unused = a + b;
    out[0] = b;
}
