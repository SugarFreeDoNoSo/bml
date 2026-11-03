//! Linked executable used to inspect the actual release hot-loop symbol.
//! Runtime black boxes keep the dispatch input opaque to optimization.
use bml_compiler::{RpnOp, RpnProgram};
use bml_runtime::HotLoop;
use std::hint::black_box;

fn main() {
    let program = RpnProgram {
        const_pool: vec![],
        ops: vec![RpnOp::One, RpnOp::One, RpnOp::Bml],
    };
    let mut hot = HotLoop::with_capacity(64);
    let value = hot.execute(black_box(&program));
    assert_eq!(black_box(value), 2.0);
    println!("hot-loop probe: {value}");
}
