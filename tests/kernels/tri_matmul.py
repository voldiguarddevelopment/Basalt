# Triton matmul: D = A@B + C at a deliberately non-square M/N/K, one program covering the
# whole result in a single `tl.dot` call. Same kernel `crates/basalt-x86/tests/triton_link_and_run.rs`'s
# `triton_matmul_links_and_runs` already proves correct through the real
# `parse -> check_triton -> lower_triton -> X86Oracle` pipeline (as `MATMUL`) — kept byte-for-
# byte identical here so this file's own proof (via `--triton --cpu`/`--triton --nvidia-ptx`) is
# provably the same kernel. `tl.dot` never lowers to BIR's `Op::Mma` (see
# `crates/basalt-sema/src/triton_lower.rs`'s own module header): it is a real scalar triple
# loop, which is exactly what this file's PTX proof exercises on real GPU hardware.
import triton
import triton.language as tl


@triton.jit
def matmul_kernel(a_ptr, b_ptr, c_ptr, out_ptr, K: tl.constexpr, M: tl.constexpr = 4, N: tl.constexpr = 3):
    rm = tl.arange(0, M)
    rn = tl.arange(0, N)
    rk = tl.arange(0, K)
    a_ptrs = a_ptr + rm[:, None] * K + rk[None, :]
    b_ptrs = b_ptr + rk[:, None] * N + rn[None, :]
    c_ptrs = c_ptr + rm[:, None] * N + rn[None, :]
    a = tl.load(a_ptrs)
    b = tl.load(b_ptrs)
    c = tl.load(c_ptrs)
    acc = tl.zeros((M, N), dtype=tl.float32)
    acc = tl.dot(a, b, acc)
    acc = acc + c
    out_ptrs = out_ptr + rm[:, None] * N + rn[None, :]
    tl.store(out_ptrs, acc)
