//! Instrumented execution metrics; requested heap bytes are not RSS or machine code.
use bml_compiler::RpnOp;
use bml_domain::EvalContext;
#[path = "../benches/support/recurrence.rs"]
mod recurrence;
use recurrence::{build_program, direct};
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

fn main() {
    // Positive control: the counter must observe allocation and deallocation.
    let before = snapshot();
    let control = black_box(vec![0u8; black_box(4096)]);
    assert!(snapshot().0 > before.0);
    assert!(snapshot().2 >= before.2 + 4096);
    drop(control);
    assert_eq!(snapshot().2, before.2);
    println!("[ ");
    let mut first = true;
    for steps in [10, 101, 1000, 10001] {
        let cons = build_program(steps, true);
        let raw = build_program(steps, false);
        for initial in [1.0, 2.0] {
            let expected = if steps % 2 == 0 {
                initial
            } else {
                3.0 - initial
            };
            for (name, program) in [
                ("direct_scalar", None),
                ("rpn_deduplicated", Some(&cons)),
                ("rpn_unshared", Some(&raw)),
            ] {
                let evaluate = || match program {
                    None => direct(black_box(steps), black_box(initial), black_box(1.0)),
                    Some(p) => {
                        let inputs = [black_box(initial)];
                        let context = EvalContext::new(&inputs, black_box(&p.const_pool));
                        black_box(p).evaluate_with_ctx(&context)
                    }
                };
                let actual = evaluate();
                assert_eq!(actual, expected);
                let (bytecode, pool, capacity) = program.map_or((0, 0, 0), |p| {
                    (
                        p.ops.len() * std::mem::size_of::<RpnOp>(),
                        p.const_pool.len() * std::mem::size_of::<f64>(),
                        p.ops.capacity() * std::mem::size_of::<RpnOp>()
                            + p.const_pool.capacity() * std::mem::size_of::<f64>(),
                    )
                });
                if !first {
                    println!(",");
                }
                first = false;
                println!("{{\"variant\":\"{name}\",\"steps\":{steps},\"initial\":{initial},\"absolute_error\":{},\"bytecode_payload_bytes\":{bytecode},\"constant_pool_payload_bytes\":{pool},\"program_heap_capacity_bytes\":{capacity},\"iterations_per_sample\":10,\"samples\":[", (actual-expected).abs());
                for repetition in 0..7 {
                    if repetition != 0 {
                        println!(",");
                    }
                    let start = begin();
                    for _ in 0..10 {
                        black_box(evaluate());
                    }
                    let report = finish(start);
                    println!("{report}");
                }
                println!("]}}");
            }
        }
    }
    println!("]");
}
