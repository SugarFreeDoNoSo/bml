//! Phase-separated symbolic workload measurements; no inference throughput claim.
use bml_domain::number::{
    Bml, BmlMatrix, ExactRational, ExactSearchOptions, ExactSearcher, ExactSynthesisOptions,
    MatrixLimits, StreamLimits,
};
use bml_domain::symbolic::Basis;
use std::{
    alloc::{GlobalAlloc, Layout, System},
    hint::black_box,
    sync::atomic::{AtomicUsize, Ordering},
    time::Instant,
};
struct MeasuredAllocator;
static CALLS: AtomicUsize = AtomicUsize::new(0);
static BYTES: AtomicUsize = AtomicUsize::new(0);
static LIVE: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);
fn allocated(size: usize) {
    CALLS.fetch_add(1, Ordering::Relaxed);
    BYTES.fetch_add(size, Ordering::Relaxed);
    let live = LIVE.fetch_add(size, Ordering::Relaxed) + size;
    PEAK.fetch_max(live, Ordering::Relaxed);
}
// SAFETY: every pointer/layout is forwarded unchanged to System; only atomic
// counters are added. No allocator callback allocates or inspects user memory.
unsafe impl GlobalAlloc for MeasuredAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        // SAFETY: caller supplies a valid Layout; System owns the returned memory.
        let pointer = unsafe { System.alloc(layout) };
        if !pointer.is_null() {
            allocated(layout.size());
        }
        pointer
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        // SAFETY: same GlobalAlloc contract, delegated to System.
        let pointer = unsafe { System.alloc_zeroed(layout) };
        if !pointer.is_null() {
            allocated(layout.size());
        }
        pointer
    }
    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        // SAFETY: caller returns a live System allocation with its original layout.
        unsafe { System.dealloc(pointer, layout) };
        LIVE.fetch_sub(layout.size(), Ordering::Relaxed);
    }
    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        // SAFETY: caller meets GlobalAlloc realloc requirements; failed realloc
        // leaves the original allocation live, as required by System.
        let result = unsafe { System.realloc(pointer, layout, size) };
        if !result.is_null() {
            LIVE.fetch_sub(layout.size(), Ordering::Relaxed);
            allocated(size);
        }
        result
    }
}
#[global_allocator]
static ALLOCATOR: MeasuredAllocator = MeasuredAllocator;
fn snapshot() -> (usize, usize, usize) {
    (
        CALLS.load(Ordering::Relaxed),
        BYTES.load(Ordering::Relaxed),
        LIVE.load(Ordering::Relaxed),
    )
}
fn begin() -> (Instant, (usize, usize, usize)) {
    let before = snapshot();
    PEAK.store(before.2, Ordering::Relaxed);
    (Instant::now(), before)
}
fn finish(start: (Instant, (usize, usize, usize))) -> String {
    let ns = start.0.elapsed().as_nanos();
    let after = snapshot();
    let peak = PEAK.load(Ordering::Relaxed).saturating_sub(start.1 .2);
    format!("{{\"ns\":{ns},\"allocation_calls\":{},\"requested_bytes\":{},\"peak_additional_live_bytes\":{peak}}}",after.0-start.1.0,after.1-start.1.1)
}
fn phase(
    first: &mut bool,
    name: &str,
    work: usize,
    unit: &str,
    iterations: usize,
    repetitions: usize,
    mut f: impl FnMut(),
) {
    if !*first {
        println!(",");
    }
    *first = false;
    f(); // warm-up, excluded; each cold search explicitly starts from an empty cache
    println!(
        "{{\"name\":\"{name}\",\"work_per_iteration\":{work},\"unit\":\"{unit}\",\"samples\":["
    );
    for repetition in 0..repetitions {
        if repetition != 0 {
            println!(",");
        }
        let start = begin();
        for _ in 0..iterations {
            f();
        }
        println!("{}", finish(start));
    }
    println!("]}}");
}
fn matrices() -> (BmlMatrix, BmlMatrix) {
    let limits = MatrixLimits::default();
    let convert = |values: &[&str]| {
        values
            .iter()
            .map(|text| Bml::from_decimal_str_exact(text).unwrap().into_term())
            .collect()
    };
    (
        BmlMatrix::new(2, 3, convert(&["1", "2", "1", "2", "1", "2"]), limits).unwrap(),
        BmlMatrix::new(3, 2, convert(&["2", "1", "1", "2", "2", "1"]), limits).unwrap(),
    )
}
fn main() {
    let iterations: usize = std::env::args()
        .nth(1)
        .map(|s| s.parse().expect("iteration count"))
        .unwrap_or(10);
    assert!((1..=1000).contains(&iterations));
    let repetitions = 7;
    let input = ["0", "1", "2", "3", "4", "12.375", "-0.125", "0.1", "2049"];
    let options = ExactSynthesisOptions::default();
    let preparation = begin();
    let targets: Vec<_> = input
        .iter()
        .map(|s| ExactRational::from_decimal_str(s, options.rational).unwrap())
        .collect();
    let certified: Vec<_> = input
        .iter()
        .map(|s| Bml::from_decimal_str_exact(s).unwrap())
        .collect();
    let terms: Vec<_> = certified.iter().map(|v| v.term().clone()).collect();
    let (a, b) = matrices();
    let prep = finish(preparation);
    let matrix_product = (&a * &b).unwrap();
    // Independent exact integer matrix oracle: standard sums of integer products.
    let aa = [1i64, 2, 1, 2, 1, 2];
    let bb = [2i64, 1, 1, 2, 2, 1];
    for row in 0..2 {
        for col in 0..2 {
            let expected: i64 = (0..3).map(|k| aa[row * 3 + k] * bb[k * 2 + col]).sum();
            assert_eq!(
                matrix_product[(row, col)]
                    .certify_exact(options)
                    .unwrap()
                    .target(),
                &ExactRational::from_i64(expected)
            );
        }
    }
    let mut direct_errors = 0;
    let mut direct_max_absolute_error = 0f64;
    for (i, value) in certified.iter().enumerate() {
        let expected = input[i].parse::<f64>().unwrap();
        assert_eq!(
            value.to_f64_simplified(options).unwrap().to_bits(),
            expected.to_bits()
        );
        match value.term().to_f64() {
            Ok(actual) => {
                direct_max_absolute_error = direct_max_absolute_error.max((actual - expected).abs())
            }
            Err(_) => direct_errors += 1,
        }
    }
    let mut seed = 0xb01u64;
    let mut operands = Vec::new();
    let mut basis_oracle = Vec::new();
    for _ in 0..1024 {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        let x = ((seed >> 32) % 9) as i32 - 4;
        let y = ((seed >> 40) % 5) as i32;
        let exp = if x >= 0 {
            (1u64 << x) as f64
        } else {
            1.0 / ((1u64 << (-x)) as f64)
        };
        operands.push((f64::from(x), (1u64 << y) as f64));
        basis_oracle.push(exp - f64::from(y));
    }
    for (input, expected) in operands.iter().zip(&basis_oracle) {
        assert_eq!(Basis::Bml.apply_real(input.0, input.1).unwrap(), *expected);
    }
    let mut basis_output = vec![0.0; operands.len()];
    println!("{{\"iterations\":{iterations},\"repetitions\":{repetitions},\"seed\":2817,\"threads\":1,\"initial_preparation\":{prep},\"direct_errors\":{direct_errors},\"direct_max_absolute_error\":{direct_max_absolute_error},\"correctness_gate\":\"passed\",\"input_terms\":[");
    for (i, value) in terms.iter().enumerate() {
        if i != 0 {
            println!(",");
        }
        let c = value.tree_cost(options.max_bits).unwrap();
        println!("{{\"decimal\":\"{}\",\"bits\":{},\"operators\":{},\"depth\":{},\"words\":{},\"word_capacity\":{}}}",input[i],c.bits,c.operators,c.depth,c.words,c.word_capacity);
    }
    let output_bits = matrix_product.total_bits().unwrap();
    let input_bits = a.total_bits().unwrap() + b.total_bits().unwrap();
    println!("],\"matrix\":{{\"left_shape\":[2,3],\"right_shape\":[3,2],\"output_shape\":[2,2],\"input_bits\":{input_bits},\"output_bits\":{output_bits}}},\"phases\":[");
    let mut first = true;
    phase(
        &mut first,
        "parse_exact_targets",
        input.len(),
        "targets",
        iterations,
        repetitions,
        || {
            for text in black_box(&input) {
                black_box(ExactRational::from_decimal_str(text, options.rational).unwrap());
            }
        },
    );
    phase(
        &mut first,
        "exact_synthesis",
        input.len(),
        "certified_constructions",
        iterations,
        repetitions,
        || {
            for text in black_box(&input) {
                black_box(Bml::from_decimal_str_exact(text).unwrap());
            }
        },
    );
    phase(
        &mut first,
        "search_cold",
        input.len(),
        "target_searches",
        iterations,
        repetitions,
        || {
            for target in black_box(&targets) {
                black_box(ExactSearcher::new(ExactSearchOptions::default()).search(target));
            }
        },
    );
    let mut warm = ExactSearcher::new(ExactSearchOptions::default());
    for target in &targets {
        assert!(warm.search(target).best.is_some());
    }
    phase(
        &mut first,
        "search_warm",
        input.len(),
        "target_searches",
        iterations,
        repetitions,
        || {
            for target in black_box(&targets) {
                black_box(warm.search(target));
            }
        },
    );
    phase(
        &mut first,
        "proof_replay",
        input.len(),
        "certificates",
        iterations,
        repetitions,
        || {
            for value in black_box(&certified) {
                black_box(
                    value
                        .derivation()
                        .verify(value.term(), value.target(), options)
                        .unwrap(),
                );
            }
        },
    );
    phase(
        &mut first,
        "stream_roundtrip",
        input.len(),
        "term_roundtrips",
        iterations,
        repetitions,
        || {
            let bytes = Bml::encode_stream(black_box(&terms), StreamLimits::default()).unwrap();
            black_box(Bml::decode_stream(&bytes, StreamLimits::default()).unwrap());
        },
    );
    phase(
        &mut first,
        "direct_tree_evaluation",
        input.len(),
        "expression_attempts",
        iterations,
        repetitions,
        || {
            for value in black_box(&terms) {
                let _ = black_box(value.to_f64());
            }
        },
    );
    phase(
        &mut first,
        "simplified_evaluation",
        input.len(),
        "certified_expressions",
        iterations,
        repetitions,
        || {
            for value in black_box(&certified) {
                black_box(value.to_f64_simplified(options).unwrap());
            }
        },
    );
    phase(
        &mut first,
        "basis_scalar_kernel",
        operands.len(),
        "B_applications",
        iterations,
        repetitions,
        || {
            for (i, (x, y)) in black_box(&operands).iter().enumerate() {
                basis_output[i] = Basis::Bml.apply_real(*x, *y).unwrap();
            }
            black_box(&basis_output);
        },
    );
    phase(
        &mut first,
        "matrix_preparation",
        12,
        "prepared_elements",
        iterations,
        repetitions,
        || {
            black_box(matrices());
        },
    );
    phase(
        &mut first,
        "matrix_operation_prepared_codes",
        1,
        "matrix_products",
        iterations,
        repetitions,
        || {
            black_box((black_box(&a) * black_box(&b)).unwrap());
        },
    );
    phase(
        &mut first,
        "matrix_complete",
        1,
        "matrix_products_including_preparation",
        iterations,
        repetitions,
        || {
            let (left, right) = matrices();
            black_box((&left * &right).unwrap());
        },
    );
    println!("]}}");
}
