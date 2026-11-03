//! Equal-work comparison of direct scalar BML arithmetic and compiled RPN.
//!
//! The legacy target name is retained for command compatibility; these are
//! exp2/log2 operations, not fused multiply-add. Construction is outside timing.
//! Both programs retain their constant pool and receive runtime variable inputs.

use bml_domain::EvalContext;
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};

#[path = "support/recurrence.rs"]
mod recurrence;
use recurrence::{build_program, direct};

fn bench_equal_work(c: &mut Criterion) {
    let mut group = c.benchmark_group("equal_bml_recurrence");
    for steps in [10, 101, 1_000, 10_001] {
        let cons = build_program(steps, true);
        let raw = build_program(steps, false);
        group.throughput(Throughput::Elements(steps as u64));
        for initial in [1.0, 2.0] {
            // Independent integer oracle: B(1,1)=2 and B(1,2)=1.
            // This avoids accepting matching infinities from divergent chains.
            let expected = if steps % 2 == 0 {
                initial
            } else {
                3.0 - initial
            };
            assert_eq!(direct(steps, initial, 1.0), expected);
            for program in [&cons, &raw] {
                let inputs = [initial];
                let ctx = EvalContext::new(&inputs, &program.const_pool);
                assert_eq!(program.evaluate_with_ctx(&ctx), expected);
            }
            let case = format!("{steps}_steps_input_{initial}");
            group.bench_function(BenchmarkId::new("direct_scalar", &case), |b| {
                b.iter(|| black_box(direct(black_box(steps), black_box(initial), black_box(1.0))))
            });
            for (name, program) in [("rpn_deduplicated", &cons), ("rpn_unshared", &raw)] {
                group.bench_function(BenchmarkId::new(name, &case), |b| {
                    b.iter(|| {
                        let inputs = [black_box(initial)];
                        let ctx = EvalContext::new(&inputs, black_box(&program.const_pool));
                        black_box(black_box(program).evaluate_with_ctx(&ctx))
                    })
                });
            }
        }
    }
    group.finish();
}

criterion_group!(benches, bench_equal_work);
criterion_main!(benches);
