// The `basalt` driver. Flag parsing mirrors BarraCUDA UX.
// The full flag surface parses, but only `--ir` is wired to
// anything — it reads a BIR file, parses it, and prints it back out, which both validates
// the file and exercises the printer/parser round-trip. Every other mode flag parses into
// `Config` cleanly and fails with a diagnostic at dispatch time rather than guessing at
// output (no silently-wrong behavior).
//
// Adding a real backend later is meant to be a small change: one new arm in `run`'s match
// over `Mode`.

use std::fs;
use std::path::PathBuf;
use std::process::ExitCode;

use basalt_diag::{Diag, ECode, LangTable};

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

/// Parsed CLI state. `-I`/`-D` are collected but unused until the preprocessor lands;
/// `tdf`/`snap` are collected for the same reason.
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

/// `--ir <file>`: parse the BIR file and print it back out (to `-o`, or stdout). Because
/// the printer and parser mirror each other exactly, this both validates the input and
/// pretty-prints it.
fn run_ir(input: &std::path::Path, output: Option<&std::path::Path>) -> Result<(), Diag> {
    let src = fs::read_to_string(input).map_err(|e| io_diag(input, e))?;
    let module = basalt_bir::parse(&src)
        .map_err(|e| Diag::new(ECode::BirParseError).with_arg(e.to_string()))?;
    let text = basalt_bir::print(&module);
    match output {
        Some(path) => fs::write(path, &text).map_err(|e| io_diag(path, e))?,
        None => print!("{text}"),
    }
    Ok(())
}

/// Dispatches on the selected mode. Every mode besides `--ir` has no implementation yet
/// (frontend/backends land in later phases) — refuse cleanly rather than emit anything.
fn run(cfg: &Config) -> Result<(), Diag> {
    let mode = cfg
        .mode
        .ok_or_else(|| Diag::new(ECode::CliMissingArgument).with_arg("mode"))?;
    match mode {
        Mode::Ir => {
            let input = cfg
                .input
                .as_deref()
                .ok_or_else(|| Diag::new(ECode::CliMissingArgument).with_arg("input file"))?;
            run_ir(input, cfg.output.as_deref())
        }
        other => Err(Diag::new(ECode::UnsupportedFeature).with_arg(other.flag())),
    }
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut table = LangTable::default_en();

    let outcome = parse_args(&args, &mut table).and_then(|cfg| run(&cfg));
    match outcome {
        Ok(()) => ExitCode::SUCCESS,
        Err(diag) => {
            eprintln!("{}", diag.render(&table));
            ExitCode::FAILURE
        }
    }
}
