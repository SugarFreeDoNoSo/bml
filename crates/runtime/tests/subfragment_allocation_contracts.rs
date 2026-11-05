//! Measure the public subfragment path, including nested loops and shared state.
#![deny(unsafe_op_in_unsafe_fn)]

use bml_compiler::{distributed::SubFragment, Fragment, RpnOp, RpnProgram};
use bml_domain::EvalContext;
use bml_runtime::{buffer::ResultBuffer, Runtime};
use std::{
    alloc::{GlobalAlloc, Layout, System},
    cell::Cell,
    hint::black_box,
};

thread_local! {
    // Restrict observation to the test thread so harness allocations cannot
    // turn a correct zero-allocation execution into a spurious failure.
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

// SAFETY: each operation forwards its pointer/layout contract to System;
// the const-initialized thread-local counter does not allocate or access memory
// owned by the caller. try_with also tolerates thread-local destruction.
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

fn subfragment(sub_id: u32, ops: Vec<RpnOp>) -> SubFragment {
    SubFragment {
        fragment_id: 0,
        sub_id,
        layer_start: 0,
        layer_end: 1,
        ops,
        weight_refs: vec![],
        depends_on: if sub_id == 0 {
            vec![]
        } else {
            vec![sub_id - 1]
        },
    }
}

#[test]
fn borrowed_subfragments_preserve_stack_and_buffer_without_allocating() {
    // Positive control: prove that the observer sees Vec allocations of the
    // kind formerly performed by cloning every subfragment's opcode array.
    let (allocation, calls) = measured(|| black_box(vec![7u8; black_box(128)]));
    assert!(
        calls > 0,
        "allocation observer did not see its positive control"
    );
    drop(allocation);

    let fragments = vec![
        subfragment(0, vec![RpnOp::Var(0), RpnOp::One, RpnOp::Bml]),
        subfragment(
            1,
            vec![
                RpnOp::Loop {
                    count: 2,
                    body_len: 5,
                },
                RpnOp::Drop,
                RpnOp::Loop {
                    count: 2,
                    body_len: 3,
                },
                RpnOp::Drop,
                RpnOp::Var(1),
                RpnOp::FAdd,
            ],
        ),
        subfragment(
            2,
            vec![
                RpnOp::Dup,
                RpnOp::Zero,
                RpnOp::StoreResult { slot: 0 },
                RpnOp::Loop {
                    count: 3,
                    body_len: 6,
                },
                RpnOp::Dup,
                RpnOp::VarIndexed { base: 8 },
                RpnOp::Const(0),
                RpnOp::FMul,
                RpnOp::Swap,
                RpnOp::StoreResult { slot: 1 },
            ],
        ),
        subfragment(
            3,
            vec![
                RpnOp::Zero,
                RpnOp::VarIndexed { base: 4 },
                RpnOp::FAdd,
                RpnOp::One,
                RpnOp::VarIndexed { base: 4 },
                RpnOp::FAdd,
            ],
        ),
    ];
    let inputs = [2.0, 4.0];
    let constants = [0.5];
    let context = EvalContext::new(&inputs, &constants);
    let mut buffer = ResultBuffer::new(3, 4);
    for slot in 0..3 {
        for offset in 0..4 {
            buffer.write(slot, offset, -999.0);
        }
    }
    for (offset, value) in [6.0, 10.0, 14.0].into_iter().enumerate() {
        buffer.write(2, offset as u32, value);
    }
    let mut runtime = Runtime::new(16, 2);
    let mut one = RpnProgram::new();
    one.push(RpnOp::One);
    assert_eq!(runtime.execute(&one), 1.0);
    // An unfinished preceding fragment must be discarded at call entry.
    runtime.execute_fragment(&Fragment {
        ops: vec![RpnOp::One],
    });
    assert_eq!(runtime.stack_depth(), 1);
    let capacity = runtime.stack_capacity();

    let (sum, calls) = measured(|| {
        let mut sum = 0.0;
        for _ in 0..32 {
            sum += black_box(runtime.execute_sub_fragments(
                black_box(&fragments),
                black_box(&context),
                black_box(&mut buffer),
            ));
        }
        sum
    });
    assert_eq!(calls, 0, "subfragment execution allocated {calls} times");
    // B(2,1)=4; four additions of input 4 give 20. The buffer loop
    // writes [3,5,7], and the last fragment returns 20+3+5=28.
    assert_eq!(sum, 32.0 * 28.0);
    assert_eq!(runtime.stack_depth(), 0);
    assert_eq!(runtime.stack_capacity(), capacity);
    assert_eq!(buffer.read(0, 0), 20.0);
    assert_eq!((buffer.read(0, 1), buffer.read(0, 3)), (-999.0, -999.0));
    assert_eq!(
        (buffer.read(1, 0), buffer.read(1, 1), buffer.read(1, 2)),
        (3.0, 5.0, 7.0)
    );
    assert_eq!(buffer.read(1, 3), -999.0);
    assert_eq!(
        (buffer.read(2, 0), buffer.read(2, 1), buffer.read(2, 2)),
        (6.0, 10.0, 14.0)
    );
    assert_eq!(runtime.results()[0], 1.0);
    assert!(runtime.results()[1].is_nan());

    runtime.execute_fragment(&Fragment {
        ops: vec![RpnOp::One],
    });
    let (empty, calls) = measured(|| runtime.execute_sub_fragments(&[], &context, &mut buffer));
    assert!(empty.is_nan());
    assert_eq!(runtime.stack_depth(), 0);
    assert_eq!(calls, 0);
}
