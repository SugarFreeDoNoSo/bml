//! Public signature planning must not materialize output-offset arrays.
#![deny(unsafe_op_in_unsafe_fn)]

use bml_compiler::symbolic_kernels::{Operation, OutputLayout, RmsBounds};
use std::{
    alloc::{GlobalAlloc, Layout, System},
    cell::Cell,
    hint::black_box,
};

thread_local! {
    // Harness and other test threads cannot affect this observation.
    static ALLOCATION_CALLS: Cell<Option<usize>> = const { Cell::new(None) };
}

struct CountedSystem;

fn record_allocation() {
    let _ = ALLOCATION_CALLS.try_with(|calls| {
        if let Some(count) = calls.get() {
            calls.set(Some(count.saturating_add(1)));
        }
    });
}

// SAFETY: each operation forwards its pointer/layout contract to System. The
// const-initialized thread-local counter does not allocate; try_with tolerates
// thread-local destruction without accessing caller-owned allocation memory.
unsafe impl GlobalAlloc for CountedSystem {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        record_allocation();
        unsafe { System.alloc(layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        record_allocation();
        unsafe { System.alloc_zeroed(layout) }
    }

    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        record_allocation();
        unsafe { System.realloc(pointer, layout, size) }
    }

    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        unsafe { System.dealloc(pointer, layout) }
    }
}

#[global_allocator]
static ALLOCATOR: CountedSystem = CountedSystem;

fn measured<T>(run: impl FnOnce() -> T) -> (T, usize) {
    struct DisableOnDrop;
    impl Drop for DisableOnDrop {
        fn drop(&mut self) {
            ALLOCATION_CALLS.with(|calls| calls.set(None));
        }
    }
    ALLOCATION_CALLS.with(|calls| {
        assert!(calls.get().is_none(), "nested allocation measurement");
        calls.set(Some(0));
    });
    let guard = DisableOnDrop;
    let output = run();
    let calls = ALLOCATION_CALLS.with(|calls| calls.replace(None).unwrap());
    drop(guard);
    (output, calls)
}

#[test]
fn buffer_literal_preflight_accepts_u32_max_count_without_allocating() {
    let limit = u32::MAX as usize;
    // These are real public Softmax signatures. Each setup owns one literal,
    // irrespective of width; no Buffer signature or large array is created.
    let exact = Operation::Softmax.signature(limit - 1, 0.5).unwrap();
    let excessive = Operation::Softmax.signature(limit, 0.5).unwrap();
    let overflow = Operation::Softmax.signature(usize::MAX, 0.5).unwrap();
    for signature in [&exact, &excessive, &overflow] {
        assert_eq!(signature.fixed, [std::f64::consts::LOG2_E]);
    }

    let (allocated, observed_calls) = measured(|| black_box(vec![7u8; black_box(128)]));
    assert!(
        observed_calls > 0,
        "allocator observer must see real allocations"
    );
    drop(allocated);

    let ((exact_count, excessive_count, overflow_count), allocations) = measured(|| {
        (
            black_box(&exact).buffer_literal_count(),
            black_box(&excessive).buffer_literal_count(),
            black_box(&overflow).buffer_literal_count(),
        )
    });
    assert_eq!(allocations, 0, "count planning must not allocate");
    assert_eq!(exact_count, Ok(limit));
    assert_eq!(
        excessive_count,
        Err(if usize::BITS > 32 {
            "symbolic output constant index overflow"
        } else {
            "symbolic output constant size overflow"
        })
    );
    assert_eq!(
        overflow_count,
        Err("symbolic output constant size overflow")
    );
    assert_eq!(exact.fixed.len(), 1);
    assert_eq!(excessive.fixed.len(), 1);
    assert_eq!(overflow.fixed.len(), 1);
}

#[test]
fn small_buffer_factories_preserve_prefix_offsets_and_admission_errors() {
    let three_offsets: &[f64] = &[0.0, 1.0, 2.0];
    let two_offsets: &[f64] = &[0.0, 1.0];
    for (operation, width, bounds, offsets) in [
        (Operation::Softmax, 3, None, three_offsets),
        (Operation::Norm, 3, None, three_offsets),
        (
            Operation::Norm,
            3,
            Some(RmsBounds {
                input: 8.0,
                weight: 0.5,
            }),
            three_offsets,
        ),
        (Operation::Rope, 2, None, two_offsets),
    ] {
        let base = operation
            .signature_with_rms_bounds(width, 0.5, bounds)
            .unwrap();
        let count = base.buffer_literal_count().unwrap();
        let buffered = operation
            .signature_for_layout_with_rms_bounds(width, 0.5, OutputLayout::Buffer, bounds)
            .unwrap();
        assert_eq!(buffered.fixed.len(), count);
        assert_eq!(
            (buffered.inputs, buffered.external, buffered.outputs),
            (base.inputs, base.external, base.outputs)
        );
        assert_eq!(&buffered.fixed[..base.fixed.len()], base.fixed.as_slice());
        assert_eq!(&buffered.fixed[base.fixed.len()..], offsets);
    }

    // Layout admission and base-parameter errors keep their original order.
    assert_eq!(
        Operation::Norm
            .signature_for_layout(0, f64::NAN, OutputLayout::IndexedBuffer)
            .unwrap_err(),
        "indexed buffer layout requires the complete bounded profile"
    );
    assert_eq!(
        Operation::Norm
            .signature_for_layout(0, 0.0, OutputLayout::Buffer)
            .unwrap_err(),
        "empty symbolic operation width"
    );
    assert_eq!(
        Operation::Norm
            .signature_for_layout(1, 0.0, OutputLayout::Buffer)
            .unwrap_err(),
        "invalid symbolic RMS epsilon"
    );
    assert_eq!(
        Operation::Dot
            .signature_for_layout(1, 0.5, OutputLayout::Buffer)
            .unwrap_err(),
        "buffer output is unsupported for this symbolic operation"
    );
}
