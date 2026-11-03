use std::process::{Command, Output};

#[path = "../../runtime/tests/support/symbolic_fixture.rs"]
mod symbolic_fixture;

fn run(path: &std::path::Path, extra: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_bml-cli"))
        .args(["run", "-m"])
        .arg(path)
        .args(["-p", "AB", "-n", "2", "--temp", "0", "--context-size", "8"])
        .args(extra)
        .output()
        .unwrap()
}
fn stdout(result: &Output) -> String {
    String::from_utf8_lossy(&result.stdout).into_owned()
}
fn stderr(result: &Output) -> String {
    String::from_utf8_lossy(&result.stderr).into_owned()
}
fn converted() -> symbolic_fixture::Fixture {
    let source = symbolic_fixture::fixture(0.25);
    // Deliberately not .bmlgguf: identification must use the magic, not extension.
    let output = symbolic_fixture::Fixture(source.0.with_extension("prepared"));
    let result = Command::new(env!("CARGO_BIN_EXE_bml-cli"))
        .args(["convert-weights", "-m"])
        .arg(&source.0)
        .arg("-o")
        .arg(&output.0)
        .args(["--max-cache-bytes", "1048576"])
        .output()
        .unwrap();
    assert!(result.status.success(), "{}", stderr(&result));
    std::fs::remove_file(&source.0).unwrap();
    output
}

#[test]
fn run_generates_from_converted_container_without_source_gguf() {
    let output = converted();
    let result = run(
        &output.0,
        &[
            "--weight-evaluation",
            "checked-exact-f32",
            "--max-evaluated-weights",
            "10000",
        ],
    );
    assert!(
        result.status.success(),
        "{}\n{}",
        stdout(&result),
        stderr(&result)
    );
    let text = stdout(&result);
    assert!(text.contains("bmlgguf-checked-exact-f32-symbolic-causal-v1"));
    assert!(text.contains("max_evaluated_weights: 10000"));
    assert!(text.contains("Tokenizado: 4 tokens"));
    assert!(text.contains("Generados 2 tokens desde BMLGGUF"));
}

#[test]
fn run_requires_explicit_policy_and_cap_and_never_modifies_container() {
    let output = converted();
    let original = std::fs::read(&output.0).unwrap();
    let implicit = run(&output.0, &[]);
    assert_eq!(implicit.status.code(), Some(1));
    assert!(stderr(&implicit).contains("BMLGGUF requiere --weight-evaluation"));
    for extra in [
        vec!["--weight-evaluation", "checked-exact-f32"],
        vec!["--max-evaluated-weights", "10000"],
        vec![
            "--weight-evaluation",
            "approximate",
            "--max-evaluated-weights",
            "10000",
        ],
        vec![
            "--weight-evaluation",
            "checked-exact-f32",
            "--max-evaluated-weights",
            "0",
        ],
        vec![
            "--weight-evaluation",
            "checked-exact-f32",
            "--max-evaluated-weights",
            "10000",
            "--bml",
        ],
        vec![
            "--weight-evaluation",
            "checked-exact-f32",
            "--max-evaluated-weights",
            "10000",
            "--weights",
            "missing.gguf",
        ],
    ] {
        assert_eq!(run(&output.0, &extra).status.code(), Some(2));
    }
    let limited = run(
        &output.0,
        &[
            "--weight-evaluation",
            "checked-exact-f32",
            "--max-evaluated-weights",
            "1",
        ],
    );
    assert_eq!(limited.status.code(), Some(1));
    assert!(stderr(&limited).contains("materialization limit"));
    assert!(!stdout(&limited).contains("Generados"));
    assert_eq!(std::fs::read(&output.0).unwrap(), original);
}

#[test]
fn explicit_prepared_run_rejects_corruption_without_generation() {
    let output = converted();
    let mut bytes = std::fs::read(&output.0).unwrap();
    *bytes.last_mut().unwrap() ^= 1;
    std::fs::write(&output.0, bytes).unwrap();
    let result = run(
        &output.0,
        &[
            "--weight-evaluation",
            "checked-exact-f32",
            "--max-evaluated-weights",
            "10000",
        ],
    );
    assert_eq!(result.status.code(), Some(1));
    assert!(stderr(&result).contains("integrity failure"));
    assert!(!stdout(&result).contains("Generados"));
}
