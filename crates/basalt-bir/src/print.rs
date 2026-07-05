// BIR textual printer. `print` renders a `Module` to the grammar documented in
// `lib.rs`'s crate header; `parse.rs` reads that text back. The two files are written to
// mirror each other one opcode at a time — when adding an `Op` variant, add its line here
// and its matching arm in `parse.rs` in the same change.
//
// Determinism: every collection printed here is a `Vec` walked in
// stored order — never a hashmap — so the same `Module` always prints the same bytes.

use std::fmt::Write as _;

use crate::ir::{Block, Function, Inst, Module, Op, Term, ValRef};

/// Renders `m` to BIR's textual form. Deterministic: the same `Module` always yields the
/// same string.
pub fn print(m: &Module) -> String {
    let mut out = String::new();
    out.push_str("module {\n");
    if let Some(lb) = m.launch_bounds {
        let _ = writeln!(
            out,
            "  launch_bounds max_threads={} min_blocks={}",
            lb.max_threads, lb.min_blocks
        );
    }
    let _ = writeln!(out, "  shared_mem_bytes {}", m.shared_mem_bytes);
    out.push_str("  target_dtypes");
    for d in &m.target_dtypes {
        let _ = write!(out, " {d}");
    }
    out.push('\n');
    for func in &m.funcs {
        print_func(&mut out, func);
    }
    out.push_str("}\n");
    out
}

fn print_func(out: &mut String, f: &Function) {
    out.push_str("\n  ");
    if !f.is_kernel {
        out.push_str("host ");
    }
    out.push_str("func @");
    out.push_str(&f.name);
    out.push('(');
    for (i, p) in f.params.iter().enumerate() {
        if i > 0 {
            out.push_str(", ");
        }
        let _ = write!(out, "{p}");
    }
    let _ = writeln!(out, ") -> {} {{", f.ret);
    for (idx, block) in f.blocks.iter().enumerate() {
        let _ = writeln!(out, "  bb{idx}:");
        print_block(out, block, &f.insts);
    }
    out.push_str("  }\n");
}

fn print_block(out: &mut String, block: &Block, insts: &[Inst]) {
    for &id in &block.insts {
        let inst = &insts[id.0 as usize];
        out.push_str("    ");
        if inst.has_result() {
            let _ = write!(out, "%{} = ", id.0);
        }
        print_op(out, inst);
        out.push('\n');
    }
    out.push_str("    ");
    print_term(out, &block.term);
    out.push('\n');
}

fn val(v: ValRef) -> String {
    match v {
        ValRef::Param(i) => format!("%arg{i}"),
        ValRef::Val(id) => format!("%{}", id.0),
    }
}

fn print_op(out: &mut String, inst: &Inst) {
    let ty = inst.ty;
    match &inst.op {
        Op::ConstInt(v) => {
            let _ = write!(out, "const.i {ty} {v}");
        }
        Op::ConstFloat(v) => {
            let _ = write!(out, "const.f {ty} {v}");
        }
        Op::Bin(op, a, b) => {
            let _ = write!(out, "{} {ty} {}, {}", op.text(), val(*a), val(*b));
        }
        Op::ICmp(pred, oty, a, b) => {
            let _ = write!(out, "icmp {} {oty} {}, {}", pred.text(), val(*a), val(*b));
        }
        Op::FCmp(pred, oty, a, b) => {
            let _ = write!(out, "fcmp {} {oty} {}, {}", pred.text(), val(*a), val(*b));
        }
        Op::Select(c, a, b) => {
            let _ = write!(out, "select {ty} {}, {}, {}", val(*c), val(*a), val(*b));
        }
        Op::Cast(op, src_ty, v) => {
            let _ = write!(out, "{} {ty} {src_ty} {}", op.text(), val(*v));
        }
        Op::Load {
            ptr,
            space,
            align,
            volatile,
        } => {
            let _ = write!(out, "load {ty} ptr.{space} {}, align {align}", val(*ptr));
            if *volatile {
                out.push_str(", volatile");
            }
        }
        Op::Store {
            ptr,
            val: v,
            ty: sty,
            space,
            align,
            volatile,
        } => {
            let _ = write!(
                out,
                "store {sty} ptr.{space} {}, {}, align {align}",
                val(*ptr),
                val(*v)
            );
            if *volatile {
                out.push_str(", volatile");
            }
        }
        Op::Phi(incoming) => {
            let _ = write!(out, "phi {ty} [");
            for (i, (bb, v)) in incoming.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                let _ = write!(out, "bb{} -> {}", bb.0, val(*v));
            }
            out.push(']');
        }
        Op::Barrier => out.push_str("barrier"),
        Op::Shuffle(kind, a, b) => {
            let _ = write!(out, "{} {ty} {}, {}", kind.text(), val(*a), val(*b));
        }
        Op::Ballot(a) => {
            let _ = write!(out, "ballot {ty} {}", val(*a));
        }
        Op::VoteAny(a) => {
            let _ = write!(out, "vote.any {ty} {}", val(*a));
        }
        Op::VoteAll(a) => {
            let _ = write!(out, "vote.all {ty} {}", val(*a));
        }
        Op::Atomic(op, ptr, v, space) => {
            let _ = write!(
                out,
                "{} {ty} ptr.{space} {}, {}",
                op.text(),
                val(*ptr),
                val(*v)
            );
        }
        Op::AtomicCas(ptr, cmp, new, space) => {
            let _ = write!(
                out,
                "atomic.cas {ty} ptr.{space} {}, {}, {}",
                val(*ptr),
                val(*cmp),
                val(*new)
            );
        }
        Op::Mma {
            a,
            b,
            c,
            d,
            m,
            n,
            k,
            in_dtype,
            acc_dtype,
            layout_a,
            layout_b,
        } => {
            let _ = write!(
                out,
                "mma {in_dtype} {acc_dtype} {} {} m {m} n {n} k {k} {}, {}, {}, {}",
                layout_a.text(),
                layout_b.text(),
                val(*a),
                val(*b),
                val(*c),
                val(*d)
            );
        }
        Op::KernelLaunch {
            kernel,
            grid,
            block,
            shared,
            stream,
            args,
        } => {
            let _ = write!(
                out,
                "kernel.launch @{kernel} grid {}, {}, {} block {}, {}, {} shared {} stream {} [",
                val(grid[0]),
                val(grid[1]),
                val(grid[2]),
                val(block[0]),
                val(block[1]),
                val(block[2]),
                val(*shared),
                val(*stream),
            );
            for (i, a) in args.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                out.push_str(&val(*a));
            }
            out.push(']');
        }
        Op::CudaMalloc { size } => {
            let _ = write!(out, "cuda.malloc {ty} {}", val(*size));
        }
        Op::CudaMemcpy {
            dst,
            src,
            count,
            kind,
        } => {
            let _ = write!(
                out,
                "cuda.memcpy {}, {}, {}, {}",
                val(*dst),
                val(*src),
                val(*count),
                val(*kind)
            );
        }
        Op::CudaFree { ptr } => {
            let _ = write!(out, "cuda.free {}", val(*ptr));
        }
        Op::CudaDeviceSynchronize => out.push_str("cuda.device_sync"),
        Op::Call { func, args } => {
            let _ = write!(out, "call {ty} @{func} [");
            for (i, a) in args.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                out.push_str(&val(*a));
            }
            out.push(']');
        }
        other => {
            // Zero-operand GPU index intrinsics (tid.x, bid.y, ...).
            let mnemonic = other
                .gpu_index_text()
                .expect("unhandled Op variant in printer");
            let _ = write!(out, "{mnemonic} {ty}");
        }
    }
}

fn print_term(out: &mut String, term: &Term) {
    match term {
        Term::Br(b) => {
            let _ = write!(out, "br bb{}", b.0);
        }
        Term::CondBr(c, t, f) => {
            let _ = write!(out, "condbr {}, bb{}, bb{}", val(*c), t.0, f.0);
        }
        Term::Switch(v, default, cases) => {
            let _ = write!(out, "switch {}, default bb{} [", val(*v), default.0);
            for (i, (case, bb)) in cases.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                let _ = write!(out, "{case} -> bb{}", bb.0);
            }
            out.push(']');
        }
        Term::Ret(None) => out.push_str("ret"),
        Term::Ret(Some(v)) => {
            let _ = write!(out, "ret {}", val(*v));
        }
    }
}
