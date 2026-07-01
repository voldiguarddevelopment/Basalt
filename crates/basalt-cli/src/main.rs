// The `basalt` driver. Flag parsing mirrors BarraCUDA UX.
//
// `--ast`, `--sema`, and `--ir` are wired to the real frontend/sema pipeline: `--ast` dumps
// the parsed AST, `--sema` runs the type checker, and `--ir` lowers all the way to BIR and
// prints it (or, given a `.bir` file directly, parses and re-prints it, exercising the
// printer/parser round-trip as before). Every other mode flag parses into `Config` cleanly
// and fails with a diagnostic at dispatch time rather than guessing at output (no
// silently-wrong behavior).
//
// Adding a real backend later is meant to be a small change: one new arm in `run`'s match
// over `Mode`.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use basalt_diag::{Diag, ECode, LangTable};
use basalt_frontend_c::ast::TranslationUnit;
use basalt_frontend_c::PpOpts;

/// A mode-selecting flag. Exactly one must be given; a second conflicts with the first.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Mode {
    Ast,
    Sema,
    Ir,
    Cpu,
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

/// Parsed CLI state. `-I`/`-D` feed `run_frontend`'s `PpOpts`; `tdf`/`snap` are collected but
/// still unused until the corresponding tooling lands.
#[derive(Debug, Default)]
struct Config {
    mode: Option<Mode>,
    input: Option<PathBuf>,
    output: Option<PathBuf>,
    include_dirs: Vec<PathBuf>,
    defines: Vec<String>,
    tdf: bool,
    snap: bool,
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

/// Dispatches on the selected mode. `--ast`/`--sema`/`--ir` run the real pipeline; every
/// other mode has no implementation yet (backends land in later phases) — refuse cleanly
/// rather than emit anything.
fn run(cfg: &Config, table: &LangTable) -> Result<ExitCode, Diag> {
    let mode = cfg
        .mode
        .ok_or_else(|| Diag::new(ECode::CliMissingArgument).with_arg("mode"))?;
    let input = || {
        cfg.input
            .as_deref()
            .ok_or_else(|| Diag::new(ECode::CliMissingArgument).with_arg("input file"))
    };
    match mode {
        Mode::Ir => run_ir(input()?, cfg.output.as_deref(), cfg, table),
        Mode::Ast => run_ast(input()?, cfg.output.as_deref(), cfg),
        Mode::Sema => run_sema(input()?, cfg, table),
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
