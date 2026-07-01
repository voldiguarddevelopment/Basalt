// Integration tests for the flag surface and `--ir`. Runs the built
// `basalt` binary as a subprocess (`CARGO_BIN_EXE_basalt`) rather than calling into the
// crate directly, since the thing under test is the actual CLI contract (exit codes,
// stdout/stderr, argv parsing) rather than any internal function.

use std::path::PathBuf;
use std::process::Command;

fn fixture_path() -> PathBuf {
    PathBuf::from(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../tests/kernels/hand_written.bir"
    ))
}

fn basalt() -> Command {
    Command::new(env!("CARGO_BIN_EXE_basalt"))
}

#[test]
fn ir_flag_round_trips_the_fixture_and_prints_it() {
    let out = basalt()
        .arg("--ir")
        .arg(fixture_path())
        .output()
        .expect("failed to run basalt");
    assert!(
        out.status.success(),
        "basalt --ir <fixture> did not exit 0: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let printed = String::from_utf8(out.stdout).expect("stdout must be valid UTF-8");
    assert!(printed.contains("func @select_max"));
}

#[test]
fn ir_flag_output_is_idempotent_fed_back_through_the_cli() {
    let first = basalt()
        .arg("--ir")
        .arg(fixture_path())
        .output()
        .expect("failed to run basalt");
    assert!(first.status.success());

    let tmp = std::env::temp_dir().join(format!(
        "basalt-cli-test-{}-idempotent.bir",
        std::process::id()
    ));
    std::fs::write(&tmp, &first.stdout).expect("failed to write intermediate fixture");

    let second = basalt()
        .arg("--ir")
        .arg(&tmp)
        .output()
        .expect("failed to run basalt a second time");
    let _ = std::fs::remove_file(&tmp);

    assert!(
        second.status.success(),
        "second --ir pass did not exit 0: {}",
        String::from_utf8_lossy(&second.stderr)
    );
    assert_eq!(
        first.stdout, second.stdout,
        "feeding --ir output back through --ir must reproduce it byte-for-byte"
    );
}

#[test]
fn unknown_flag_exits_nonzero_and_reports_e100() {
    let out = basalt()
        .arg("--this-flag-does-not-exist")
        .output()
        .expect("failed to run basalt");
    assert!(!out.status.success());
    let stderr = String::from_utf8(out.stderr).expect("stderr must be valid UTF-8");
    assert!(
        stderr.contains("E100"),
        "expected E100 in stderr, got: {stderr}"
    );
}

#[test]
fn missing_argument_to_a_value_flag_reports_e101() {
    let out = basalt()
        .arg("--lang")
        .output()
        .expect("failed to run basalt");
    assert!(!out.status.success());
    let stderr = String::from_utf8(out.stderr).expect("stderr must be valid UTF-8");
    assert!(
        stderr.contains("E101"),
        "expected E101 in stderr, got: {stderr}"
    );
}
