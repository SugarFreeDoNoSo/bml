// Native synchronization for the shared ArrayQueue implementation.

pub(super) use core::sync::atomic::{self, AtomicUsize, Ordering};
pub(super) use crossbeam_utils::{Backoff, CachePadded};

pub(super) struct UnsafeCell<T>(core::cell::UnsafeCell<T>);
impl<T> UnsafeCell<T> {
    #[inline]
    pub(super) fn new(value: T) -> Self {
        Self(core::cell::UnsafeCell::new(value))
    }
    #[inline]
    pub(super) fn with<R>(&self, f: impl FnOnce(*const T) -> R) -> R {
        f(self.0.get().cast_const())
    }
    #[inline]
    pub(super) fn with_mut<R>(&self, f: impl FnOnce(*mut T) -> R) -> R {
        f(self.0.get())
    }
}

#[inline]
pub(super) fn atomic_get(value: &mut AtomicUsize) -> usize {
    *value.get_mut()
}

#[inline]
pub(super) fn atomic_set(value: &mut AtomicUsize, next: usize) {
    *value.get_mut() = next;
}
