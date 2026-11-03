//! Bounded pending-work queue using the vendored Crossbeam ArrayQueue algorithm.
//!
//! The public queue always uses native synchronization, including with the
//! `loom` feature enabled. Model tests compile the same algorithm source with
//! instrumented synchronization. This crate contains the upstream unsafe slot
//! accesses; consumers such as bml-runtime retain their deny(unsafe_code) policy.
#![deny(unsafe_op_in_unsafe_fn)]

mod native {
    mod sync {
        include!("native_sync.rs");
    }
    pub(super) mod implementation {
        include!("array_queue.rs");
    }
}

pub use native::implementation::{ArrayQueue, IntoIter};

#[cfg(all(test, feature = "loom"))]
#[allow(dead_code)] // The shared upstream algorithm also exposes exclusive/iterator APIs.
mod model {
    mod sync {
        include!("loom_sync.rs");
    }
    mod implementation {
        include!("array_queue.rs");
    }
    mod tests {
        include!("loom_tests.rs");
    }
}

#[cfg(test)]
mod native_tests {
    use super::ArrayQueue;

    #[test]
    fn native_queue_remains_usable_with_every_feature() {
        // In particular, this executes outside loom::model under --all-features.
        let queue = ArrayQueue::new(3);
        for value in 0..3 {
            assert_eq!(queue.push(value), Ok(()));
        }
        assert_eq!(queue.push(3), Err(3));
        for value in 0..3 {
            assert_eq!(queue.pop(), Some(value));
        }
        assert!(queue.is_empty());
    }
}
