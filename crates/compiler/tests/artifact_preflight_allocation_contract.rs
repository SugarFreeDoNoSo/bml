//! The legacy scalar context plan must be rejected before materializing its
//! prefix table. A tiny invalid bundle must not allocate in proportion to context.
use bml_compiler::{symbolic_artifact, symbolic_kernels::KernelLimits};
use std::{
    alloc::{GlobalAlloc, Layout, System},
    sync::atomic::{AtomicBool, AtomicUsize, Ordering},
};
#[path = "../../runtime/tests/support/symbolic_fixture.rs"]
mod symbolic_fixture;

struct CountedSystem;
static TRACK: AtomicBool = AtomicBool::new(false);
static BYTES: AtomicUsize = AtomicUsize::new(0);

fn record(bytes: usize) {
    if TRACK.load(Ordering::Relaxed) {
        BYTES.fetch_add(bytes, Ordering::Relaxed);
    }
}

// SAFETY: every allocation operation preserves System's pointer/layout contract;
// the counters observe requested sizes without reading or altering allocations.
unsafe impl GlobalAlloc for CountedSystem {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        record(layout.size());
        unsafe { System.alloc(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        record(size);
        unsafe { System.realloc(ptr, layout, size) }
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        record(layout.size());
        unsafe { System.alloc_zeroed(layout) }
    }
}

#[global_allocator]
static ALLOCATOR: CountedSystem = CountedSystem;

#[test]
fn oversized_legacy_context_is_rejected_before_allocating_the_prefix_plan() {
    let source = symbolic_fixture::fixture(0.25);
    let mut bytes = std::fs::read(&source.0).unwrap();
    let key = b"llama.context_length";
    let at = bytes.windows(key.len()).position(|w| w == key).unwrap() + key.len();
    assert_eq!(&bytes[at..at + 4], &4u32.to_le_bytes());
    bytes[at + 4..at + 8].copy_from_slice(&250_000u32.to_le_bytes());
    std::fs::write(&source.0, bytes).unwrap();
    let output = source.0.with_extension("preflight");
    let mut manifest =
        symbolic_artifact::compile_directory(&source.0, &output, 1, KernelLimits::default())
            .unwrap();
    manifest.version = 1;
    manifest.profile = "llama-symbolic-bml-f64-f32kv-v1".into();
    manifest.context_length = 250_000;
    std::fs::write(
        output.join(symbolic_artifact::MANIFEST),
        serde_json::to_vec(&manifest).unwrap(),
    )
    .unwrap();
    BYTES.store(0, Ordering::Relaxed);
    TRACK.store(true, Ordering::Relaxed);
    let result = symbolic_artifact::load_directory(&output, None);
    TRACK.store(false, Ordering::Relaxed);
    let allocated = BYTES.load(Ordering::Relaxed);
    std::fs::remove_dir_all(output).unwrap();
    assert_eq!(
        result.err().unwrap(),
        "symbolic AOT program budget exceeded"
    );
    // Generous allowance for parsing and validating this tiny fixture, including
    // cumulative realloc requests. The rejected prefix table alone exceeds it.
    assert!(
        allocated < 8 * 1024 * 1024,
        "preflight allocated {allocated} bytes"
    );
}
