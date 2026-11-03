// Loom synchronization for the shared ArrayQueue implementation.

pub(super) use crossbeam_utils::CachePadded;
pub(super) use loom::cell::UnsafeCell;
pub(super) use loom::sync::atomic::{self, AtomicUsize, Ordering};

// Backoff does not decide queue admission/ownership. Yield through Loom so a
// thread awaiting a published stamp permits the publishing thread to proceed.
pub(super) struct Backoff;
impl Backoff {
    pub(super) fn new() -> Self {
        Self
    }
    pub(super) fn spin(&self) {
        loom::thread::yield_now();
    }
    pub(super) fn snooze(&self) {
        loom::thread::yield_now();
    }
}

pub(super) fn atomic_get(value: &mut AtomicUsize) -> usize {
    value.with_mut(|inner| *inner)
}

pub(super) fn atomic_set(value: &mut AtomicUsize, next: usize) {
    value.with_mut(|inner| *inner = next);
}
