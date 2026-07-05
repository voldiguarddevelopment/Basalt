// Real end-to-end proof of a __device__-to-__device__ call chain, at least two levels deep, all
// the way down to a real __global__ kernel (P13-T-calls-ii): `inc` is called by
// `scale_then_inc`, which is called by `negate_then_scale`, which `chain_vector` itself calls
// once per thread. Nothing here is a leaf call from the kernel anymore -- every intermediate
// link is a genuine __device__-to-__device__ Op::Call, so a wrong stack frame, argument
// register, or return-value handoff anywhere in the chain shows up as a wrong final value, not
// a silent pass. Every element uses a different, checkable value (including negatives) so a
// sign error at any link would not accidentally match.
__device__ int inc(int x) {
    return x + 1;
}

__device__ int scale_then_inc(int x) {
    return inc(x * 2);
}

__device__ int negate_then_scale(int x) {
    return scale_then_inc(-x);
}

__global__ void chain_vector(const int *in, int *out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = negate_then_scale(in[i]);
    }
}
