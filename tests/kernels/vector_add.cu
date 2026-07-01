// Elementwise vector addition: c[i] = a[i] + b[i] for i in [0, n).
//
// One thread per element, laid out over a 1-D grid. Each thread computes its own global
// index from its block/thread coordinates and guards against grids that overshoot n.
__global__ void vector_add(const float *a, const float *b, float *c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}
