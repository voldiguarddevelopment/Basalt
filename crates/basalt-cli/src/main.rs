// The `basalt` driver. Flag parsing mirrors BarraCUDA UX.
//
// `--ast`, `--sema`, and `--ir` are wired to the real frontend/sema pipeline: `--ast` dumps
// the parsed AST, `--sema` runs the type checker, and `--ir` lowers all the way to BIR and
// prints it (or, given a `.bir` file directly, parses and re-prints it, exercising the
// printer/parser round-trip as before). `--ir` deliberately prints the raw output of
// `basalt-sema`'s lowering, unoptimized: its job is to show exactly what the lowering pass
// produced, for inspection and for the round-trip check, not what a backend will actually
// build from. `--cpu` runs that same frontend/sema pipeline, then `basalt_passes::optimize`,
// and hands the result to the x86-64 oracle backend, writing an object file to `-o`.
// `--cpu-regalloc` does the same against the x86-64 regalloc backend instead (the CPU
// performance path). `--nvidia-ptx` runs the same pipeline against the PTX backend; its
// output is text like `--ir`'s, so it prints to stdout or `-o` rather than requiring `-o`
// the way the object-file modes do. Every backend that emits real machine code is handed the
// optimized module, unconditionally — there is no flag to opt out, since the whole point is
// that these cleanups are load-bearing infrastructure every target gets for free, not a
// feature a caller has to remember to ask for. Every other mode flag parses into `Config`
// cleanly and fails with a diagnostic at dispatch time rather than guessing at output (no
// silently-wrong behavior). `--amdgpu-bin` on its own runs the real hand-rolled
// `basalt-amdgpu` backend (`Amdgcn`, always built, no feature gate); `--llvm --amdgpu-bin`
// instead routes through `basalt-llvm`'s `TargetMachine`-based AMDGCN object emission, a
// second, independent lane kept for cross-checking rather than replaced outright. Every other
// combination of `--llvm` with a mode is a clean refusal, never a silent fallback to the
// non-LLVM path. `--spirv` runs the real hand-rolled `basalt-spirv` backend (`Spirv`, always
// built, no feature gate), following `--amdgpu-bin`'s binary-artifact/`-o`-mandatory convention.
// `--tensix` runs the real hand-rolled `basalt-tensix` backend (`Tensix`, always built, no
// feature gate), generating Metalium C++ for Tenstorrent's Tensix; its output is text like
// `--nvidia-ptx`'s, so it follows that mode's stdout-or-`-o` convention. `--tensix --tdf`
// is a modifier on the same mode (see `run_tensix`): instead of the single-core emitter, it
// runs `basalt_tensix::dump_tdf`'s real multi-core fission pass and prints the resulting
// region/channel/NoC-arc layout. `--rv-elf` runs the
// real hand-rolled `basalt-rv` backend (`Rv32`, always built, no feature gate), following
// `--amdgpu-bin`'s binary-artifact/`-o`-mandatory convention instead. `--triton` is a
// modifier following `--llvm`'s own pattern, but on the *frontend* side rather
// than the backend side: paired with `--cpu`, `--cpu-regalloc`, or `--nvidia-ptx` it redirects
// the input from the CUDA-C `run_frontend`/`basalt_sema::check`/`basalt_sema::lower` pipeline
// to the real Triton one (`basalt_frontend_triton::parse` -> `basalt_sema::check_triton` ->
// `basalt_sema::lower_triton`, see `run_triton_frontend`) and leaves backend selection
// completely untouched — the resulting `basalt_bir::Module` flows into whichever backend was
// selected, same as any other module (this is exactly why `--cpu-regalloc` is included
// alongside the exit criterion's own two named modes: the diff harness needs a real
// oracle-vs-regalloc cross-check for a Triton kernel too, the same way every CUDA-C kernel
// already gets one). Paired with any other mode it is a clean refusal, matching `--llvm`'s own
// refusal discipline exactly.
//
// Adding a real backend later is meant to be a small change: one new arm in `run`'s match
// over `Mode`.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use basalt_amdgpu::Amdgcn;
use basalt_backend::{Backend, EmitOpts, Support};
use basalt_diag::{Diag, ECode, LangTable};
use basalt_frontend_c::ast::TranslationUnit;
use basalt_frontend_c::PpOpts;
#[cfg(feature = "llvm")]
use basalt_llvm::LlvmAmdgcn;
use basalt_ptx::Ptx;
use basalt_rv::Rv32;
use basalt_spirv::Spirv;
use basalt_tensix::Tensix;
use basalt_x86::{X86Oracle, X86Regalloc};

/// A mode-selecting flag. Exactly one must be given; a second conflicts with the first.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Mode {
    Ast,
    Sema,
    Ir,
    Cpu,
    CpuRegalloc,
    Rv64,
    NvidiaPtx,
    AmdgpuBin,
    Tensix,
    RvElf,
    Spirv,
}

impl Mode {
    /// The flag text that selects this mode, for diagnostics.
    fn flag(self) -> &'static str {
        match self {
            Mode::Ast => "--ast",
            Mode::Sema => "--sema",
            Mode::Ir => "--ir",
            Mode::Cpu => "--cpu",
            Mode::CpuRegalloc => "--cpu-regalloc",
            Mode::Rv64 => "--rv64",
            Mode::NvidiaPtx => "--nvidia-ptx",
            Mode::AmdgpuBin => "--amdgpu-bin",
            Mode::Tensix => "--tensix",
            Mode::RvElf => "--rv-elf",
            Mode::Spirv => "--spirv",
        }
    }

    fn from_flag(flag: &str) -> Option<Mode> {
        Some(match flag {
            "--ast" => Mode::Ast,
            "--sema" => Mode::Sema,
            "--ir" => Mode::Ir,
            "--cpu" => Mode::Cpu,
            "--cpu-regalloc" => Mode::CpuRegalloc,
            "--rv64" => Mode::Rv64,
            "--nvidia-ptx" => Mode::NvidiaPtx,
            "--amdgpu-bin" => Mode::AmdgpuBin,
            "--tensix" => Mode::Tensix,
            "--rv-elf" => Mode::RvElf,
            "--spirv" => Mode::Spirv,
            _ => return None,
        })
    }
}

/// Parsed CLI state. `-I`/`-D` feed `run_frontend`'s `PpOpts`; `snap` is collected but still
/// unused until the corresponding tooling lands. `llvm` is the `--llvm` modifier: paired
/// with `Mode::AmdgpuBin` it selects the LLVM-backed AMDGCN object-emission path (see `run`);
/// paired with anything else, `run` refuses cleanly rather than silently ignoring it. `triton`
/// is the `--triton` modifier: paired with `Mode::Cpu`, `Mode::CpuRegalloc`, or
/// `Mode::NvidiaPtx` it selects the real Triton frontend/sema pipeline instead of the CUDA-C
/// one (see `run_triton_frontend`); paired with anything else, `run` refuses the same way
/// `llvm` does. `tdf` is the `--tdf` modifier: paired with `Mode::Tensix` it selects
/// `run_tensix`'s TDF-dump path instead of the single-core Metalium emitter (see `run_tensix`);
/// paired with anything else, `run` refuses the same way `llvm`/`triton` do.
#[derive(Debug, Default)]
struct Config {
    mode: Option<Mode>,
    input: Option<PathBuf>,
    output: Option<PathBuf>,
    include_dirs: Vec<PathBuf>,
    defines: Vec<String>,
    tdf: bool,
    snap: bool,
    llvm: bool,
    triton: bool,
}

/// Consumes the argument that follows a value-taking flag, erroring with `E101` if there
/// isn't one.
fn take_value(args: &[String], i: &mut usize, flag: &str) -> Result<String, Diag> {
    *i += 1;
    match args.get(*i) {
        Some(v) => Ok(v.clone()),
        None => Err(Diag::new(ECode::CliMissingArgument).with_arg(flag)),
    }
}

/// Parses the full flag surface into a `Config`. Loads any `--lang` table it encounters
/// into `table` immediately, so a later parse error in the same invocation renders against
/// the table the user asked for rather than the default.
fn parse_args(args: &[String], table: &mut LangTable) -> Result<Config, Diag> {
    let mut cfg = Config::default();
    let mut i = 0;
    while i < args.len() {
        let arg = args[i].as_str();
        if let Some(mode) = Mode::from_flag(arg) {
            if let Some(existing) = cfg.mode {
                return Err(Diag::new(ECode::CliConflictingFlags)
                    .with_arg(existing.flag())
                    .with_arg(mode.flag()));
            }
            cfg.mode = Some(mode);
        } else {
            match arg {
                "-o" => cfg.output = Some(PathBuf::from(take_value(args, &mut i, "-o")?)),
                "-I" => cfg
                    .include_dirs
                    .push(PathBuf::from(take_value(args, &mut i, "-I")?)),
                "-D" => cfg.defines.push(take_value(args, &mut i, "-D")?),
                "--lang" => {
                    let path = take_value(args, &mut i, "--lang")?;
                    *table = basalt_diag::load_lang_file(&path)?;
                }
                "--tdf" => cfg.tdf = true,
                "--snap" => cfg.snap = true,
                "--llvm" => cfg.llvm = true,
                "--triton" => cfg.triton = true,
                _ if arg.starts_with('-') => {
                    return Err(Diag::new(ECode::CliUnknownFlag).with_arg(arg));
                }
                _ => {
                    if cfg.input.is_some() {
                        return Err(Diag::new(ECode::CliInvalidArgument)
                            .with_arg("input")
                            .with_arg(arg));
                    }
                    cfg.input = Some(PathBuf::from(arg));
                }
            }
        }
        i += 1;
    }
    Ok(cfg)
}

/// Wraps a filesystem error as an `E500` diagnostic naming the path.
fn io_diag(path: &std::path::Path, err: std::io::Error) -> Diag {
    Diag::new(ECode::IoError)
        .with_arg(path.display().to_string())
        .with_arg(err.to_string())
}

/// Whether `input`'s extension marks it as BIR text rather than C/CUDA source. Every mode
/// that can take either kind of input (currently just `--ir`) uses this one check, so "is
/// this a `.bir` file" never drifts between call sites.
fn is_bir_input(input: &Path) -> bool {
    input.extension().is_some_and(|ext| ext == "bir")
}

/// Splits a `-D` argument into the `(name, value)` pair `PpOpts::defines` wants. `NAME=value`
/// splits on the first `=`; a bare `NAME` carries no value, matching a plain `-D NAME` on a
/// real C compiler.
fn split_define(raw: &str) -> (String, Option<String>) {
    match raw.split_once('=') {
        Some((name, value)) => (name.to_string(), Some(value.to_string())),
        None => (raw.to_string(), None),
    }
}

/// The result of turning source text into an AST: the best-effort `TranslationUnit` the
/// parser managed to build, plus every lex/preprocess/parse problem hit along the way,
/// already rendered to text. Those three stages report their own self-contained error types
/// (`LexError`/`PpError`/`ParseError`) rather than `basalt_diag::Diag` — there is no
/// conversion between them, so problems from this stage are carried as plain strings (via
/// each error's own `Display` impl) and printed as-is rather than run through a `LangTable`.
struct Frontend {
    tu: TranslationUnit,
    problems: Vec<String>,
}

impl Frontend {
    fn has_problems(&self) -> bool {
        !self.problems.is_empty()
    }
}

/// Runs the shared lex/preprocess/parse pipeline over `src`. `preprocess` lexes internally
/// and reports any scanning problem as its own `PpError::Lex` variant, so a separate `lex`
/// pass first would only double-report the same errors; preprocessing then parsing is the
/// whole pipeline. Neither stage stops at its first problem, and `parse` always returns a
/// `TranslationUnit` (best-effort on a broken token stream) rather than an `Option`, so this
/// never has "nothing to hand to the next stage".
fn run_frontend(src: &str, input: &Path, cfg: &Config) -> Frontend {
    let opts = PpOpts {
        include_dirs: cfg.include_dirs.clone(),
        defines: cfg.defines.iter().map(|d| split_define(d)).collect(),
        base_dir: input.parent().map(Path::to_path_buf),
    };
    let (tokens, pp_errors) = basalt_frontend_c::preprocess(src, &opts);
    let mut problems: Vec<String> = pp_errors.iter().map(|e| e.to_string()).collect();
    let (tu, parse_errors) = basalt_frontend_c::parse(&tokens);
    problems.extend(parse_errors.iter().map(|e| e.to_string()));
    Frontend { tu, problems }
}

/// Runs the real Triton pipeline (`basalt_frontend_triton::parse` -> `basalt_sema::check_triton`
/// -> `basalt_sema::lower_triton`) over `src`, the `--triton` counterpart to `run_frontend` +
/// `basalt_sema::check` + `basalt_sema::lower` combined. Every diagnostic from any of the three
/// stages is rendered against `table` and printed, exactly like the CUDA-C path's own
/// frontend/sema/lower diagnostics; returns the lowered module only if all three stages came
/// back clean, so a caller can treat `None` as "already reported, just fail" the same way the
/// CUDA-C path's own inline check does.
fn run_triton_frontend(src: &str, table: &LangTable) -> Option<basalt_bir::Module> {
    let (module, parse_diags) = basalt_frontend_triton::parse(src);
    let (shapes, check_diags) = basalt_sema::check_triton(&module);
    let (bir, lower_diags) = basalt_sema::lower_triton(&module, &shapes);

    for d in parse_diags
        .iter()
        .chain(check_diags.iter())
        .chain(lower_diags.iter())
    {
        eprintln!("{}", d.render(table));
    }

    if parse_diags.is_empty() && check_diags.is_empty() && lower_diags.is_empty() {
        Some(bir)
    } else {
        None
    }
}

/// Diagnoses a `.bir` file handed to a mode that expects C/CUDA source.
fn mismatched_bir_input(mode: Mode) -> Diag {
    Diag::new(ECode::CliInvalidArgument)
        .with_arg(mode.flag())
        .with_arg("input file has a .bir extension; this mode expects C/CUDA source")
}

/// `--ir <file>`: given a `.bir` file, parse it and print it back out (to `-o`, or stdout) —
/// this both validates the input and exercises the printer/parser round-trip. Given C/CUDA
/// source, runs the full pipeline (lex/preprocess/parse, check, lower) and prints the
/// resulting BIR. Lowering degrades gracefully rather than panicking, so the best-effort BIR
/// is printed even when sema or lowering reported problems — the caller still gets something
/// to inspect — but any such problem still makes this exit non-zero.
fn run_ir(
    input: &Path,
    output: Option<&Path>,
    cfg: &Config,
    table: &LangTable,
) -> Result<ExitCode, Diag> {
    let src = fs::read_to_string(input).map_err(|e| io_diag(input, e))?;

    if is_bir_input(input) {
        let module = basalt_bir::parse(&src)
            .map_err(|e| Diag::new(ECode::BirParseError).with_arg(e.to_string()))?;
        let text = basalt_bir::print(&module);
        write_output(output, &text)?;
        return Ok(ExitCode::SUCCESS);
    }

    let fe = run_frontend(&src, input, cfg);
    let sema_diags = basalt_sema::check(&fe.tu);
    let (module, lower_diags) = basalt_sema::lower(&fe.tu);
    let text = basalt_bir::print(&module);
    write_output(output, &text)?;

    for p in &fe.problems {
        eprintln!("{p}");
    }
    for d in sema_diags.iter().chain(lower_diags.iter()) {
        eprintln!("{}", d.render(table));
    }

    let clean = !fe.has_problems() && sema_diags.is_empty() && lower_diags.is_empty();
    Ok(if clean {
        ExitCode::SUCCESS
    } else {
        ExitCode::FAILURE
    })
}

/// `--ast <file>`: lex+preprocess+parse the input and dump the resulting `TranslationUnit`
/// via its `Debug` impl (`{:#?}`) — no custom pretty-printer, `Debug` on the AST types is
/// already a faithful, readable dump. Exits non-zero if any problem was found, but the
/// best-effort AST is printed regardless, since it's the thing worth inspecting when
/// something went wrong.
fn run_ast(input: &Path, output: Option<&Path>, cfg: &Config) -> Result<ExitCode, Diag> {
    if is_bir_input(input) {
        return Err(mismatched_bir_input(Mode::Ast));
    }
    let src = fs::read_to_string(input).map_err(|e| io_diag(input, e))?;
    let fe = run_frontend(&src, input, cfg);
    let text = format!("{:#?}\n", fe.tu);
    write_output(output, &text)?;

    for p in &fe.problems {
        eprintln!("{p}");
    }
    Ok(if fe.has_problems() {
        ExitCode::FAILURE
    } else {
        ExitCode::SUCCESS
    })
}

/// `--sema <file>`: lex+preprocess+parse the input, then run the type checker over the
/// result. Every problem — frontend problems as-is, sema diagnostics rendered against
/// `table` — is printed to stdout, one per line; a clean run prints a plain "no diagnostics"
/// line instead of nothing, so success is never silent. Exits non-zero if anything at all was
/// reported, at any stage.
fn run_sema(input: &Path, cfg: &Config, table: &LangTable) -> Result<ExitCode, Diag> {
    if is_bir_input(input) {
        return Err(mismatched_bir_input(Mode::Sema));
    }
    let src = fs::read_to_string(input).map_err(|e| io_diag(input, e))?;
    let fe = run_frontend(&src, input, cfg);
    let sema_diags = basalt_sema::check(&fe.tu);

    for p in &fe.problems {
        println!("{p}");
    }
    for d in &sema_diags {
        println!("{}", d.render(table));
    }
    let clean = !fe.has_problems() && sema_diags.is_empty();
    if clean {
        println!("no diagnostics");
    }
    Ok(if clean {
        ExitCode::SUCCESS
    } else {
        ExitCode::FAILURE
    })
}

/// `--cpu <file>`: runs the same frontend/sema pipeline as `--ir` (lex/preprocess/parse, check,
/// lower for C/CUDA source; parse directly for a `.bir` file), then runs `basalt_passes::optimize`
/// over the resulting module before handing it to the x86-64 oracle backend and writing the
/// emitted object bytes to `-o`. Unlike `--ir`, this mode never emits a best-effort artifact: an
/// object file is either right or not written at all, so any frontend/sema/lowering problem
/// exits non-zero without touching `-o`. `-o` itself is mandatory here (`E101`) — a raw object is
/// not something to dump to a terminal — and a module this backend cannot lower is refused with
/// its own `Diag`, rendered the same way every other error in this file is.
fn run_cpu(
    input: &Path,
    output: Option<&Path>,
    cfg: &Config,
    table: &LangTable,
) -> Result<ExitCode, Diag> {
    let output = output.ok_or_else(|| Diag::new(ECode::CliMissingArgument).with_arg("-o"))?;

    let module = if is_bir_input(input) {
        let src = fs::read_to_string(input).map_err(|e| io_diag(input, e))?;
        basalt_bir::parse(&src)
            .map_err(|e| Diag::new(ECode::BirParseError).with_arg(e.to_string()))?
    } else if cfg.triton {
        let src = fs::read_to_string(input).map_err(|e| io_diag(input, e))?;
        match run_triton_frontend(&src, table) {
            Some(module) => module,
            None => return Ok(ExitCode::FAILURE),
        }
    } else {
        let src = fs::read_to_string(input).map_err(|e| io_diag(input, e))?;
        let fe = run_frontend(&src, input, cfg);
        let sema_diags = basalt_sema::check(&fe.tu);
        let (module, lower_diags) = basalt_sema::lower(&fe.tu);

        for p in &fe.problems {
            eprintln!("{p}");
        }
        for d in sema_diags.iter().chain(lower_diags.iter()) {
            eprintln!("{}", d.render(table));
        }

        if fe.has_problems() || !sema_diags.is_empty() || !lower_diags.is_empty() {
            return Ok(ExitCode::FAILURE);
        }
        module
    };
    let module = basalt_passes::optimize(&module);

    let backend = X86Oracle;
    match backend.supports(&module) {
        Support::Supported => {}
        Support::Unsupported(code) => return Err(Diag::new(code).with_arg(backend.name())),
    }

    let artifact = backend.emit(&module, &EmitOpts::default())?;
    let bytes = artifact
        .as_bytes()
        .expect("X86Oracle::emit always produces a Payload::Bytes artifact");
    fs::write(output, bytes).map_err(|e| io_diag(output, e))?;
    Ok(ExitCode::SUCCESS)
}

/// `--cpu-regalloc <file>`: identical pipeline and `-o`/error-handling contract to `--cpu`,
/// but hands the module to the x86-64 regalloc backend (`X86Regalloc`) instead of the oracle —
/// the CPU performance path, sharing every bit of frontend/sema/lowering plumbing `--cpu`
/// already has.
fn run_cpu_regalloc(
    input: &Path,
    output: Option<&Path>,
    cfg: &Config,
    table: &LangTable,
) -> Result<ExitCode, Diag> {
    let output = output.ok_or_else(|| Diag::new(ECode::CliMissingArgument).with_arg("-o"))?;

    let module = if is_bir_input(input) {
        let src = fs::read_to_string(input).map_err(|e| io_diag(input, e))?;
        basalt_bir::parse(&src)
            .map_err(|e| Diag::new(ECode::BirParseError).with_arg(e.to_string()))?
    } else if cfg.triton {
        let src = fs::read_to_string(input).map_err(|e| io_diag(input, e))?;
        match run_triton_frontend(&src, table) {
            Some(module) => module,
            None => return Ok(ExitCode::FAILURE),
        }
    } else {
        let src = fs::read_to_string(input).map_err(|e| io_diag(input, e))?;
        let fe = run_frontend(&src, input, cfg);
        let sema_diags = basalt_sema::check(&fe.tu);
        let (module, lower_diags) = basalt_sema::lower(&fe.tu);

        for p in &fe.problems {
            eprintln!("{p}");
        }
        for d in sema_diags.iter().chain(lower_diags.iter()) {
            eprintln!("{}", d.render(table));
        }

        if fe.has_problems() || !sema_diags.is_empty() || !lower_diags.is_empty() {
            return Ok(ExitCode::FAILURE);
        }
        module
    };
    let module = basalt_passes::optimize(&module);

    let backend = X86Regalloc;
    match backend.supports(&module) {
        Support::Supported => {}
        Support::Unsupported(code) => return Err(Diag::new(code).with_arg(backend.name())),
    }

    let artifact = backend.emit(&module, &EmitOpts::default())?;
    let bytes = artifact
        .as_bytes()
        .expect("X86Regalloc::emit always produces a Payload::Bytes artifact");
    fs::write(output, bytes).map_err(|e| io_diag(output, e))?;
    Ok(ExitCode::SUCCESS)
}

/// `--nvidia-ptx <file>`: same frontend/sema/lower/optimize pipeline as `--cpu`, handed to the
/// PTX backend instead of the x86-64 oracle. Unlike `--cpu`, the artifact here is text, not an
/// object file, so it follows `--ir`'s output convention: printed to stdout, or written to `-o`
/// if given, and `-o` is optional. A module the backend can't lower is refused the same way
/// `--cpu` refuses one, with the backend's own `Diag`.
fn run_nvidia_ptx(
    input: &Path,
    output: Option<&Path>,
    cfg: &Config,
    table: &LangTable,
) -> Result<ExitCode, Diag> {
    let module = if is_bir_input(input) {
        let src = fs::read_to_string(input).map_err(|e| io_diag(input, e))?;
        basalt_bir::parse(&src)
            .map_err(|e| Diag::new(ECode::BirParseError).with_arg(e.to_string()))?
    } else if cfg.triton {
        let src = fs::read_to_string(input).map_err(|e| io_diag(input, e))?;
        match run_triton_frontend(&src, table) {
            Some(module) => module,
            None => return Ok(ExitCode::FAILURE),
        }
    } else {
        let src = fs::read_to_string(input).map_err(|e| io_diag(input, e))?;
        let fe = run_frontend(&src, input, cfg);
        let sema_diags = basalt_sema::check(&fe.tu);
        let (module, lower_diags) = basalt_sema::lower(&fe.tu);

        for p in &fe.problems {
            eprintln!("{p}");
        }
        for d in sema_diags.iter().chain(lower_diags.iter()) {
            eprintln!("{}", d.render(table));
        }

        if fe.has_problems() || !sema_diags.is_empty() || !lower_diags.is_empty() {
            return Ok(ExitCode::FAILURE);
        }
        module
    };
    let module = basalt_passes::optimize(&module);

    let backend = Ptx;
    match backend.supports(&module) {
        Support::Supported => {}
        Support::Unsupported(code) => return Err(Diag::new(code).with_arg(backend.name())),
    }

    let artifact = backend.emit(&module, &EmitOpts::default())?;
    let text = artifact
        .as_text()
        .expect("Ptx::emit always produces a Payload::Text artifact");
    write_output(output, text)?;
    Ok(ExitCode::SUCCESS)
}

/// `--llvm --amdgpu-bin <file>`: same frontend/sema/lower/optimize pipeline as `--cpu`, handed
/// to `basalt-llvm`'s `TargetMachine`-based AMDGCN object-emission path (`LlvmAmdgcn`) instead
/// of the hand-rolled `basalt-amdgpu` backend `run_amdgpu_bin` below uses — kept as a second,
/// independent lane for cross-checking rather than replaced. `-o` is mandatory, matching
/// `--cpu`'s own object-file convention.
#[cfg(feature = "llvm")]
fn run_llvm_amdgpu_bin(
    input: &Path,
    output: Option<&Path>,
    cfg: &Config,
    table: &LangTable,
) -> Result<ExitCode, Diag> {
    let output = output.ok_or_else(|| Diag::new(ECode::CliMissingArgument).with_arg("-o"))?;

    let module = if is_bir_input(input) {
        let src = fs::read_to_string(input).map_err(|e| io_diag(input, e))?;
        basalt_bir::parse(&src)
            .map_err(|e| Diag::new(ECode::BirParseError).with_arg(e.to_string()))?
    } else {
        let src = fs::read_to_string(input).map_err(|e| io_diag(input, e))?;
        let fe = run_frontend(&src, input, cfg);
        let sema_diags = basalt_sema::check(&fe.tu);
        let (module, lower_diags) = basalt_sema::lower(&fe.tu);

        for p in &fe.problems {
            eprintln!("{p}");
        }
        for d in sema_diags.iter().chain(lower_diags.iter()) {
            eprintln!("{}", d.render(table));
        }

        if fe.has_problems() || !sema_diags.is_empty() || !lower_diags.is_empty() {
            return Ok(ExitCode::FAILURE);
        }
        module
    };
    let module = basalt_passes::optimize(&module);

    let backend = LlvmAmdgcn;
    match backend.supports(&module) {
        Support::Supported => {}
        Support::Unsupported(code) => return Err(Diag::new(code).with_arg(backend.name())),
    }

    let artifact = backend.emit(&module, &EmitOpts::default())?;
    let bytes = artifact
        .as_bytes()
        .expect("LlvmAmdgcn::emit always produces a Payload::Bytes artifact");
    fs::write(output, bytes).map_err(|e| io_diag(output, e))?;
    Ok(ExitCode::SUCCESS)
}

/// `--amdgpu-bin <file>` without `--llvm`: same frontend/sema/lower/optimize pipeline as
/// `--cpu`, handed to the real hand-rolled `basalt-amdgpu` backend (`Amdgcn`) — no LLVM
/// anywhere on this path, matching this backend's own role as the "no LLVM" flagship (see
/// `ARCHITECTURE.md`). `-o` is mandatory, matching `--cpu`'s own object-file convention.
fn run_amdgpu_bin(
    input: &Path,
    output: Option<&Path>,
    cfg: &Config,
    table: &LangTable,
) -> Result<ExitCode, Diag> {
    let output = output.ok_or_else(|| Diag::new(ECode::CliMissingArgument).with_arg("-o"))?;

    let module = if is_bir_input(input) {
        let src = fs::read_to_string(input).map_err(|e| io_diag(input, e))?;
        basalt_bir::parse(&src)
            .map_err(|e| Diag::new(ECode::BirParseError).with_arg(e.to_string()))?
    } else {
        let src = fs::read_to_string(input).map_err(|e| io_diag(input, e))?;
        let fe = run_frontend(&src, input, cfg);
        let sema_diags = basalt_sema::check(&fe.tu);
        let (module, lower_diags) = basalt_sema::lower(&fe.tu);

        for p in &fe.problems {
            eprintln!("{p}");
        }
        for d in sema_diags.iter().chain(lower_diags.iter()) {
            eprintln!("{}", d.render(table));
        }

        if fe.has_problems() || !sema_diags.is_empty() || !lower_diags.is_empty() {
            return Ok(ExitCode::FAILURE);
        }
        module
    };
    let module = basalt_passes::optimize(&module);

    let backend = Amdgcn;
    match backend.supports(&module) {
        Support::Supported => {}
        Support::Unsupported(code) => return Err(Diag::new(code).with_arg(backend.name())),
    }

    let artifact = backend.emit(&module, &EmitOpts::default())?;
    let bytes = artifact
        .as_bytes()
        .expect("Amdgcn::emit always produces a Payload::Bytes artifact");
    fs::write(output, bytes).map_err(|e| io_diag(output, e))?;
    Ok(ExitCode::SUCCESS)
}

/// `--spirv <file>`: same frontend/sema/lower/optimize pipeline as `--cpu`, handed to the real
/// hand-rolled `basalt-spirv` backend (`Spirv`, always built, no feature gate). The emitted
/// SPIR-V module is binary, like `--amdgpu-bin`'s HSACO, so this follows that mode's `-o`-
/// mandatory, raw-bytes-to-file convention rather than `--nvidia-ptx`'s stdout-text one.
fn run_spirv(
    input: &Path,
    output: Option<&Path>,
    cfg: &Config,
    table: &LangTable,
) -> Result<ExitCode, Diag> {
    let output = output.ok_or_else(|| Diag::new(ECode::CliMissingArgument).with_arg("-o"))?;

    let module = if is_bir_input(input) {
        let src = fs::read_to_string(input).map_err(|e| io_diag(input, e))?;
        basalt_bir::parse(&src)
            .map_err(|e| Diag::new(ECode::BirParseError).with_arg(e.to_string()))?
    } else {
        let src = fs::read_to_string(input).map_err(|e| io_diag(input, e))?;
        let fe = run_frontend(&src, input, cfg);
        let sema_diags = basalt_sema::check(&fe.tu);
        let (module, lower_diags) = basalt_sema::lower(&fe.tu);

        for p in &fe.problems {
            eprintln!("{p}");
        }
        for d in sema_diags.iter().chain(lower_diags.iter()) {
            eprintln!("{}", d.render(table));
        }

        if fe.has_problems() || !sema_diags.is_empty() || !lower_diags.is_empty() {
            return Ok(ExitCode::FAILURE);
        }
        module
    };
    let module = basalt_passes::optimize(&module);

    let backend = Spirv;
    match backend.supports(&module) {
        Support::Supported => {}
        Support::Unsupported(code) => return Err(Diag::new(code).with_arg(backend.name())),
    }

    let artifact = backend.emit(&module, &EmitOpts::default())?;
    let bytes = artifact
        .as_bytes()
        .expect("Spirv::emit always produces a Payload::Bytes artifact");
    fs::write(output, bytes).map_err(|e| io_diag(output, e))?;
    Ok(ExitCode::SUCCESS)
}

/// `--tensix <file>`: same frontend/sema/lower/optimize pipeline as `--cpu`, handed to the real
/// hand-rolled `basalt-tensix` backend (`Tensix`, always built, no feature gate). Its output is
/// generated Metalium C++ text, like `--nvidia-ptx`'s PTX, so it follows that mode's stdout-or-
/// `-o` convention rather than `--spirv`'s binary-artifact/`-o`-mandatory one. No `--triton`
/// pairing is wired for this mode (see the `--triton` refusal in `run`, below).
///
/// `--tensix --tdf <file>` is a modifier on this same mode, following `--llvm`/`--triton`'s own
/// pattern of a flag that changes what an existing mode does rather than adding a new one:
/// instead of the single-core `Tensix` backend, the same optimized module is handed to
/// `basalt_tensix::dump_tdf`, which runs the TDF (Tile-DataFlow) fission pass and prints the
/// resulting regions/channels/NoC-arc layout (plus the two kernel bodies the pass produced) as
/// text — same stdout-or-`-o` convention, since this is text output too. A module the fission
/// pass cannot fission is refused the same way an unsupported module refuses under plain
/// `--tensix`.
fn run_tensix(
    input: &Path,
    output: Option<&Path>,
    cfg: &Config,
    table: &LangTable,
) -> Result<ExitCode, Diag> {
    let module = if is_bir_input(input) {
        let src = fs::read_to_string(input).map_err(|e| io_diag(input, e))?;
        basalt_bir::parse(&src)
            .map_err(|e| Diag::new(ECode::BirParseError).with_arg(e.to_string()))?
    } else {
        let src = fs::read_to_string(input).map_err(|e| io_diag(input, e))?;
        let fe = run_frontend(&src, input, cfg);
        let sema_diags = basalt_sema::check(&fe.tu);
        let (module, lower_diags) = basalt_sema::lower(&fe.tu);

        for p in &fe.problems {
            eprintln!("{p}");
        }
        for d in sema_diags.iter().chain(lower_diags.iter()) {
            eprintln!("{}", d.render(table));
        }

        if fe.has_problems() || !sema_diags.is_empty() || !lower_diags.is_empty() {
            return Ok(ExitCode::FAILURE);
        }
        module
    };
    let module = basalt_passes::optimize(&module);

    if cfg.tdf {
        let text = basalt_tensix::dump_tdf(&module)?;
        write_output(output, &text)?;
        return Ok(ExitCode::SUCCESS);
    }

    let backend = Tensix;
    match backend.supports(&module) {
        Support::Supported => {}
        Support::Unsupported(code) => return Err(Diag::new(code).with_arg(backend.name())),
    }

    let artifact = backend.emit(&module, &EmitOpts::default())?;
    let text = artifact
        .as_text()
        .expect("Tensix::emit always produces a Payload::Text artifact");
    write_output(output, text)?;
    Ok(ExitCode::SUCCESS)
}

/// `--rv-elf <file>`: same frontend/sema/lower/optimize pipeline as `--cpu`, handed to the
/// real hand-rolled `basalt-rv` backend (`Rv32`, always built, no feature gate — RV32IM has no
/// LLVM-free build concern here, this is a from-scratch encoder just like `--amdgpu-bin`/
/// `--spirv`). Follows `--amdgpu-bin`'s binary-artifact/`-o`-mandatory convention: an ELF
/// object is either right or not written at all.
fn run_rv_elf(
    input: &Path,
    output: Option<&Path>,
    cfg: &Config,
    table: &LangTable,
) -> Result<ExitCode, Diag> {
    let output = output.ok_or_else(|| Diag::new(ECode::CliMissingArgument).with_arg("-o"))?;

    let module = if is_bir_input(input) {
        let src = fs::read_to_string(input).map_err(|e| io_diag(input, e))?;
        basalt_bir::parse(&src)
            .map_err(|e| Diag::new(ECode::BirParseError).with_arg(e.to_string()))?
    } else {
        let src = fs::read_to_string(input).map_err(|e| io_diag(input, e))?;
        let fe = run_frontend(&src, input, cfg);
        let sema_diags = basalt_sema::check(&fe.tu);
        let (module, lower_diags) = basalt_sema::lower(&fe.tu);

        for p in &fe.problems {
            eprintln!("{p}");
        }
        for d in sema_diags.iter().chain(lower_diags.iter()) {
            eprintln!("{}", d.render(table));
        }

        if fe.has_problems() || !sema_diags.is_empty() || !lower_diags.is_empty() {
            return Ok(ExitCode::FAILURE);
        }
        module
    };
    let module = basalt_passes::optimize(&module);

    let backend = Rv32;
    match backend.supports(&module) {
        Support::Supported => {}
        Support::Unsupported(code) => return Err(Diag::new(code).with_arg(backend.name())),
    }

    let artifact = backend.emit(&module, &EmitOpts::default())?;
    let bytes = artifact
        .as_bytes()
        .expect("Rv32::emit always produces a Payload::Bytes artifact");
    fs::write(output, bytes).map_err(|e| io_diag(output, e))?;
    Ok(ExitCode::SUCCESS)
}

/// Writes `text` to `output` if given, else stdout — the common tail of every mode that
/// produces a single text artifact.
fn write_output(output: Option<&Path>, text: &str) -> Result<(), Diag> {
    match output {
        Some(path) => fs::write(path, text).map_err(|e| io_diag(path, e)),
        None => {
            print!("{text}");
            Ok(())
        }
    }
}

/// Dispatches on the selected mode. `--ast`/`--sema`/`--ir`/`--cpu`/`--cpu-regalloc`/
/// `--nvidia-ptx`/`--amdgpu-bin`/`--spirv`/`--tensix`/`--rv-elf` run the real pipeline; every
/// other mode has no implementation yet (backends land in later phases) — refuse cleanly
/// rather than emit anything. `--triton`
/// (only meaningful with `--cpu`/`--nvidia-ptx`) and `--tdf` (only meaningful with `--tensix`)
/// are handled above this match, before the mode dispatch even begins.
fn run(cfg: &Config, table: &LangTable) -> Result<ExitCode, Diag> {
    let mode = cfg
        .mode
        .ok_or_else(|| Diag::new(ECode::CliMissingArgument).with_arg("mode"))?;
    let input = || {
        cfg.input
            .as_deref()
            .ok_or_else(|| Diag::new(ECode::CliMissingArgument).with_arg("input file"))
    };
    // `--llvm` only has a wired path for `Mode::AmdgpuBin`; paired with any other mode it must
    // refuse outright rather than silently falling through to that mode's non-LLVM behavior.
    if cfg.llvm && mode != Mode::AmdgpuBin {
        return Err(Diag::new(ECode::UnsupportedFeature).with_arg("--llvm"));
    }
    // `--triton` only has a wired path for `Mode::Cpu`/`Mode::CpuRegalloc`/`Mode::NvidiaPtx`;
    // paired with any other mode it must refuse outright, matching `--llvm`'s own refusal
    // discipline above.
    if cfg.triton && mode != Mode::Cpu && mode != Mode::CpuRegalloc && mode != Mode::NvidiaPtx {
        return Err(Diag::new(ECode::UnsupportedFeature).with_arg("--triton"));
    }
    // `--tdf` only has a wired path for `Mode::Tensix`; paired with any other mode it must
    // refuse outright, matching `--llvm`/`--triton`'s own refusal discipline above.
    if cfg.tdf && mode != Mode::Tensix {
        return Err(Diag::new(ECode::UnsupportedFeature).with_arg("--tdf"));
    }
    match mode {
        Mode::Ir => run_ir(input()?, cfg.output.as_deref(), cfg, table),
        Mode::Ast => run_ast(input()?, cfg.output.as_deref(), cfg),
        Mode::Sema => run_sema(input()?, cfg, table),
        Mode::Cpu => run_cpu(input()?, cfg.output.as_deref(), cfg, table),
        Mode::CpuRegalloc => run_cpu_regalloc(input()?, cfg.output.as_deref(), cfg, table),
        Mode::NvidiaPtx => run_nvidia_ptx(input()?, cfg.output.as_deref(), cfg, table),
        #[cfg(feature = "llvm")]
        Mode::AmdgpuBin if cfg.llvm => {
            run_llvm_amdgpu_bin(input()?, cfg.output.as_deref(), cfg, table)
        }
        Mode::AmdgpuBin => run_amdgpu_bin(input()?, cfg.output.as_deref(), cfg, table),
        Mode::Spirv => run_spirv(input()?, cfg.output.as_deref(), cfg, table),
        Mode::Tensix => run_tensix(input()?, cfg.output.as_deref(), cfg, table),
        Mode::RvElf => run_rv_elf(input()?, cfg.output.as_deref(), cfg, table),
        other => Err(Diag::new(ECode::UnsupportedFeature).with_arg(other.flag())),
    }
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut table = LangTable::default_en();

    let outcome = parse_args(&args, &mut table).and_then(|cfg| run(&cfg, &table));
    match outcome {
        Ok(code) => code,
        Err(diag) => {
            eprintln!("{}", diag.render(&table));
            ExitCode::FAILURE
        }
    }
}
