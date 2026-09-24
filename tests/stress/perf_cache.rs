//! Opt-in Linux profiling of an actual evaluator child.
use bml_compiler::{RpnOp, RpnProgram};
#[test]
#[ignore = "requires Linux perf and PMU access"]
fn perf_stat_cache_events_on_program() {
    let output = std::process::Command::new("perf")
        .args(["stat", "-x,", "-e", "cache-references,cache-misses", "--"])
        .arg(std::env::current_exe().unwrap())
        .args(["--ignored", "--exact", "profiled_evaluator_child"])
        .env("BML_PERF_CHILD", "1")
        .output()
        .expect("install perf");
    let log = String::from_utf8_lossy(&output.stderr);
    assert!(output.status.success(), "perf failed: {log}");
    assert!(
        !log.contains("<not supported>") && !log.contains("<not counted>"),
        "{log}"
    );
    assert!(
        log.contains("cache-misses") && log.contains("cache-references"),
        "{log}"
    );
    println!("{log}");
}
#[test]
#[ignore = "child of perf_stat_cache_events_on_program"]
fn profiled_evaluator_child() {
    assert_eq!(std::env::var("BML_PERF_CHILD").as_deref(), Ok("1"));
    let mut p = RpnProgram::new();
    p.push(RpnOp::One);
    for _ in 0..1000 {
        p.ops.extend([RpnOp::One, RpnOp::Swap, RpnOp::Bml]);
    }
    for _ in 0..1000 {
        assert_eq!(std::hint::black_box(p.evaluate()), 1.0);
    }
}
