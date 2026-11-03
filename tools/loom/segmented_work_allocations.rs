//! Observation only: relaxed counters do not publish queue payloads.
use loom::sync::atomic::{AtomicUsize, Ordering};

loom::lazy_static! {
    static ref ALLOCATED: AtomicUsize = AtomicUsize::new(0);
    static ref FREED: Vec<AtomicUsize> = (0..128).map(|_| AtomicUsize::new(0)).collect();
}

pub fn allocate() -> usize {
    let id = ALLOCATED.fetch_add(1, Ordering::Relaxed);
    assert!(id < FREED.len(), "allocation observation budget exceeded");
    id
}

pub fn free(id: usize) {
    assert_eq!(
        FREED[id].fetch_add(1, Ordering::Relaxed),
        0,
        "segment released more than once"
    );
}

pub fn assert_balanced() {
    let allocated = ALLOCATED.load(Ordering::Relaxed);
    assert!(allocated > 0);
    for id in 0..allocated {
        assert_eq!(
            FREED[id].load(Ordering::Relaxed),
            1,
            "segment allocation {id} was not released"
        );
    }
}
