#!/usr/bin/env python3
# Runs a single kernel out of a real AMDGCN HSACO object through tinygrad's instruction-level
# RDNA3 emulator (test/mockgpu/amd/emu.py, activated by DEV=MOCK+AMD) and prints the contents
# of every output buffer. This is a correctness harness, not a benchmark: no hardware is
# involved anywhere in this path, on purpose — there is no RDNA3 part on hand, and this is the
# emulator route instead of leaving AMDGCN codegen unvalidated.
#
# Loading goes through tinygrad's own AMDProgram/AMDDevice classes directly rather than
# tinygrad's compiler pipeline: `AMDProgram.__init__` takes raw ELF bytes and feeds them to
# `elf_loader` (tinygrad.runtime.support.elf), a compiler-agnostic relocatable-object loader
# that only understands section/symbol/relocation structure, not provenance. It was confirmed
# empirically that this accepts an externally-produced HSACO exactly as it would accept one
# from tinygrad's own AMDGPU renderer — the only relocation kind it applies by hand
# (R_AMDGPU_REL64, type 5) is exactly what a real `TargetMachine` emits here.
#
# Kernel-argument layout assumption: `tinygrad`'s own kernarg packer (`CLikeArgsState`) always
# lays every buffer argument down first, then every scalar argument after — it has no notion
# of a kernel's true declared argument order. This harness therefore only produces a correct
# kernarg segment for a kernel whose real signature already has every pointer parameter before
# every scalar parameter (true of every kernel this project currently emits, since BIR params
# keep their source order and this project's frontend does not interleave the two).
#
# Exit codes: 0 on a clean run (results printed, one buffer per line), 1 on a real failure
# (bad arguments, load error, kernel trap), 77 when the emulator path itself is unavailable
# in this environment (no tinygrad checkout with test/mockgpu, import failure) — a caller
# script should treat 77 as "skip this check", not "this check failed".

import argparse
import importlib.util
import os
import struct
import sys

SKIP = 77

# Must happen before any tinygrad submodule import: tinygrad.runtime.support.hcq decides at
# import time, from the DEV environment variable, whether to swap in the mocked KFD file
# interface.
os.environ.setdefault("DEV", "MOCK+AMD")

DTYPE_FMT = {"i32": "i", "i64": "q", "f32": "f", "f64": "d"}
DTYPE_SIZE = {"i32": 4, "i64": 8, "f32": 4, "f64": 8}


def find_tinygrad_src_root() -> str | None:
    """Locates a full tinygrad git checkout (the one directory layout that ships
    test/mockgpu — the pip package alone does not) without importing tinygrad itself, since
    importing it before DEV is set would be too late for the mock hook above."""
    spec = importlib.util.find_spec("tinygrad")
    if spec is None or spec.origin is None:
        return None
    root = os.path.dirname(os.path.dirname(os.path.abspath(spec.origin)))
    if os.path.isfile(os.path.join(root, "test", "mockgpu", "amd", "emu.py")):
        return root
    return None


def parse_buf(spec: str):
    mode, dtype, rest = spec.split(":", 2)
    if mode not in ("in", "out"):
        raise ValueError(f"buf mode must be 'in' or 'out', got {mode!r}")
    if dtype not in DTYPE_FMT:
        raise ValueError(f"unknown buf dtype {dtype!r}")
    if mode == "in":
        values = [float(x) if dtype.startswith("f") else int(x) for x in rest.split(",")]
    else:
        values = [0] * int(rest)
    return mode, dtype, values


def parse_scalar(spec: str):
    dtype, val = spec.split(":", 1)
    if dtype not in DTYPE_FMT:
        raise ValueError(f"unknown scalar dtype {dtype!r}")
    return dtype, (float(val) if dtype.startswith("f") else int(val))


def parse_triple(spec: str):
    parts = tuple(int(x) for x in spec.split(","))
    if len(parts) != 3:
        raise ValueError(f"expected X,Y,Z, got {spec!r}")
    return parts


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--hsaco", required=True, help="path to the HSACO/ELF object to load")
    ap.add_argument("--kernel", required=True, help="kernel symbol name inside the object")
    ap.add_argument(
        "--buf",
        action="append",
        default=[],
        metavar="in|out:dtype:data",
        help="one device buffer, in kernel-argument order; 'in:f32:1.0,2.0' or 'out:f32:1'",
    )
    ap.add_argument(
        "--scalar",
        action="append",
        default=[],
        metavar="dtype:value",
        help="one non-pointer kernel argument, in kernel-argument order",
    )
    ap.add_argument("--global", dest="global_size", default="1,1,1")
    ap.add_argument("--local", dest="local_size", default="1,1,1")
    args = ap.parse_args()

    tg_root = find_tinygrad_src_root()
    if tg_root is None:
        print("no tinygrad checkout with test/mockgpu on this interpreter's path", file=sys.stderr)
        return SKIP
    sys.path.insert(0, tg_root)

    try:
        from tinygrad import Device
        from tinygrad.runtime.ops_amd import AMDProgram
    except Exception as e:  # pragma: no cover - environment-dependent
        print(f"tinygrad's mock AMD backend is not importable here: {e}", file=sys.stderr)
        return SKIP

    try:
        bufs = [parse_buf(s) for s in args.buf]
        scalars = [parse_scalar(s) for s in args.scalar]
        global_size = parse_triple(args.global_size)
        local_size = parse_triple(args.local_size)
    except ValueError as e:
        print(f"bad argument: {e}", file=sys.stderr)
        return 1

    with open(args.hsaco, "rb") as f:
        lib = f.read()

    dev = Device["AMD"]
    prg = AMDProgram(dev, args.kernel, lib)

    dev_bufs = []
    for mode, dtype, values in bufs:
        raw = struct.pack(f"<{len(values)}{DTYPE_FMT[dtype]}", *values)
        buf = dev.allocator.alloc(max(len(raw), 1))
        dev.allocator._copyin(buf, memoryview(bytearray(raw)))
        dev_bufs.append((mode, dtype, len(values), buf))

    vals = tuple(v for _, v in scalars)

    prg(
        *[b for _, _, _, b in dev_bufs],
        global_size=global_size,
        local_size=local_size,
        vals=vals,
        wait=True,
    )

    for mode, dtype, count, buf in dev_bufs:
        if mode != "out":
            continue
        nbytes = count * DTYPE_SIZE[dtype]
        out = memoryview(bytearray(nbytes))
        dev.allocator._copyout(out, buf)
        unpacked = struct.unpack(f"<{count}{DTYPE_FMT[dtype]}", out)
        print(" ".join(f"{v:.6f}" if dtype.startswith("f") else str(v) for v in unpacked))

    return 0


if __name__ == "__main__":
    sys.exit(main())
