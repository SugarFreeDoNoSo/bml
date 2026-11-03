//! Reproducible code-comparison experiment; no numerical BML throughput claim.
use bml_domain::number::Bml;
use bml_kernels::{compare_codes, Backend, CompareLimits};
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
fn chain(n: usize, seed: u64) -> Bml {
    let mut result = Bml::from_u64(seed).unwrap();
    for _ in 0..n {
        result = Bml::one().op(&result).unwrap();
    }
    result
}
fn main() {
    let iterations: usize = std::env::args()
        .nth(1)
        .map(|s| s.parse().expect("iteration count"))
        .unwrap_or(100);
    assert!((1..=10000).contains(&iterations));
    let repetitions = 7;
    let pairs = 512;
    println!("{{\"iterations\":{iterations},\"repetitions\":{repetitions},\"pairs\":{pairs},\"avx2_available\":{},\"neon_available\":{},\"workloads\":[",Backend::Avx2.available(),Backend::Neon.available());
    for (index, (name, depth, different)) in [
        ("equal-long", 2048, false),
        ("late-difference", 2048, true),
        ("short-mixed", 3, true),
    ]
    .into_iter()
    .enumerate()
    {
        if index != 0 {
            println!(",");
        }
        let start = begin();
        let a = chain(depth, 24);
        let b = chain(depth, if different { 20 } else { 24 });
        let left = vec![a; pairs];
        let right = vec![b; pairs];
        let preparation = finish(start);
        let logical_bits: usize = left.iter().chain(&right).map(Bml::bit_len).sum();
        println!("{{\"name\":\"{name}\",\"logical_input_bits\":{logical_bits},\"preparation\":{preparation},\"backends\":[");
        let mut first = true;
        for backend in [Backend::Scalar, Backend::Avx2, Backend::Neon]
            .into_iter()
            .filter(|b| b.available())
        {
            if !first {
                println!(",");
            }
            first = false;
            let limits = CompareLimits::default();
            let expected = compare_codes(&left, &right, Backend::Scalar, limits)
                .unwrap()
                .orderings;
            assert_eq!(
                compare_codes(&left, &right, backend, limits)
                    .unwrap()
                    .orderings,
                expected
            );
            // Warm-up is excluded from the seven measured repetitions.
            black_box(compare_codes(black_box(&left), black_box(&right), backend, limits).unwrap());
            println!("{{\"backend\":\"{backend:?}\",\"samples\":[");
            for repetition in 0..repetitions {
                if repetition != 0 {
                    println!(",");
                }
                let start = begin();
                for _ in 0..iterations {
                    black_box(
                        compare_codes(black_box(&left), black_box(&right), backend, limits)
                            .unwrap(),
                    );
                }
                println!("{}", finish(start));
            }
            println!("]}}");
        }
        println!("]}}");
    }
    println!("]}}");
}
