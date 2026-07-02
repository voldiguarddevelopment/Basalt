// Hand-rolled NVIDIA PTX text emitter: this project's first real GPU target, and the first
// backend that runs on genuine SIMT hardware rather than emulating one thread at a time on a
// CPU core. See `emit.rs`'s header for the full design (register model, threading model,
// refusal surface, and every non-obvious instruction-mapping choice).

mod emit;

pub use emit::Ptx;
