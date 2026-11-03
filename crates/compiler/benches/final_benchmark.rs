//! Synthetic RPN dispatch throughput. This is not transformer inference.
//!
//! Each sample executes a finite recurrence with a known exact result. Criterion
//! reports dispatched instructions/s; no conversion to tokens, model throughput,
//! machine counts or provider costs is justified by this workload.
//! Real inference comparisons are recorded in docs/benchmarks/.

use bml_compiler::{RpnOp, RpnProgram};
use bml_runtime::Runtime;
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};

fn build_program(bml_operations: usize) -> RpnProgram {
    let mut program = RpnProgram::new();
    program.push(RpnOp::One);
    for _ in 0..bml_operations {
        // B(1, 1) = 2, B(1, 2) = 1: the recurrence stays finite.
        program.push(RpnOp::One);
        program.push(RpnOp::Swap);
        program.push(RpnOp::Bml);
    }
    assert_eq!(program.len(), 1 + 3 * bml_operations);
    program
}

fn bench_final(c: &mut Criterion) {
    let mut group = c.benchmark_group("synthetic_rpn_dispatch");
    for bml_operations in [1_000, 1_001, 10_000, 100_000] {
        let program = build_program(bml_operations);
        let expected = if bml_operations % 2 == 0 { 1.0 } else { 2.0 };
        let mut runtime = Runtime::new(8192, 16);
        // Validate the actual measured engine, before entering the timed region.
        assert_eq!(runtime.execute(black_box(&program)), expected);
        group.throughput(Throughput::Elements(program.len() as u64));
        group.bench_with_input(
            BenchmarkId::new("instructions", program.len()),
            &program,
            |b, program| b.iter(|| black_box(runtime.execute(black_box(program)))),
        );
    }
    group.finish();
}

criterion_group!(benches, bench_final);
criterion_main!(benches);
