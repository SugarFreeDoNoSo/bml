// The model namespace compiles the same algorithm source as the native queue.
use super::implementation::ArrayQueue;
use loom::{sync::Arc, thread};

#[test]
fn bounded_admission_preserves_rejected_ownership() {
    loom::model(|| {
        let queue = Arc::new(ArrayQueue::new(1));
        let other = queue.clone();
        let first = thread::spawn(move || other.push(11));
        let second = queue.push(22);
        let first = first.join().unwrap();
        assert_ne!(first.is_ok(), second.is_ok());
        let (accepted, rejected) = if first.is_ok() { (11, 22) } else { (22, 11) };
        assert_eq!(first.err().or(second.err()), Some(rejected));
        assert_eq!(queue.len(), 1);
        assert_eq!(queue.pop(), Some(accepted));
        assert_eq!(queue.pop(), None);
        assert!(queue.is_empty());
    });
}

#[test]
fn simultaneous_consumers_transfer_each_item_once() {
    loom::model(|| {
        let queue = Arc::new(ArrayQueue::new(2));
        queue.push((11, 101)).unwrap();
        queue.push((22, 202)).unwrap();
        let other = queue.clone();
        let first = thread::spawn(move || other.pop());
        let second = queue.pop();
        let first = first.join().unwrap();
        let mut values = [first.unwrap(), second.unwrap()];
        values.sort();
        assert_eq!(values, [(11, 101), (22, 202)]);
        assert_eq!(queue.pop(), None);
    });
}

#[test]
fn publication_and_slot_reuse_preserve_payload() {
    loom::model(|| {
        let queue = Arc::new(ArrayQueue::new(1));
        let other = queue.clone();
        let producer = thread::spawn(move || {
            other.push((11, 101)).unwrap();
        });
        let value = queue.pop();
        producer.join().unwrap();
        let first = value.or_else(|| queue.pop());
        assert_eq!(first, Some((11, 101)));
        assert_eq!(queue.pop(), None);
        queue.push((22, 202)).unwrap();
        assert_eq!(queue.pop(), Some((22, 202)));
        assert!(queue.is_empty());
    });
}

#[test]
fn publication_precedes_producer_completion_without_join_synchronization() {
    use loom::sync::atomic::{AtomicBool, Ordering};
    loom::model(|| {
        let queue = Arc::new(ArrayQueue::new(1));
        let consumed = Arc::new(AtomicBool::new(false));
        let other = queue.clone();
        let done = consumed.clone();
        let producer = thread::spawn(move || {
            other.push((11, 101)).unwrap();
            // Keep the publisher alive until the payload has been consumed.
            // This flag controls termination; it does not publish queue data.
            while !done.load(Ordering::Relaxed) {
                thread::yield_now();
            }
        });
        let payload = loop {
            if let Some(value) = queue.pop() {
                break value;
            }
            thread::yield_now();
        };
        assert_eq!(payload, (11, 101));
        consumed.store(true, Ordering::Relaxed);
        producer.join().unwrap();
        assert!(queue.is_empty());
    });
}

#[test]
fn concurrent_reuse_publishes_the_next_lap_after_consumption() {
    loom::model(|| {
        let queue = Arc::new(ArrayQueue::new(1));
        queue.push((11, 101)).unwrap();
        let other = queue.clone();
        let producer = thread::spawn(move || {
            let mut pending = (22, 202);
            while let Err(value) = other.push(pending) {
                pending = value;
                thread::yield_now();
            }
        });
        assert_eq!(queue.pop(), Some((11, 101)));
        let next = queue.pop();
        producer.join().unwrap();
        assert_eq!(next.or_else(|| queue.pop()), Some((22, 202)));
        assert_eq!(queue.pop(), None);
        queue.push((33, 303)).unwrap();
        assert_eq!(queue.pop(), Some((33, 303)));
    });
}

#[test]
fn consumed_and_retained_values_are_dropped_exactly_once() {
    use loom::sync::atomic::{AtomicUsize, Ordering};
    struct Counted {
        drops: Arc<AtomicUsize>,
    }
    impl Drop for Counted {
        fn drop(&mut self) {
            assert_eq!(self.drops.fetch_add(1, Ordering::SeqCst), 0);
        }
    }
    loom::model(|| {
        let first = Arc::new(AtomicUsize::new(0));
        let second = Arc::new(AtomicUsize::new(0));
        let queue = Arc::new(ArrayQueue::new(2));
        assert!(queue
            .push(Counted {
                drops: first.clone()
            })
            .is_ok());
        assert!(queue
            .push(Counted {
                drops: second.clone()
            })
            .is_ok());
        let other = queue.clone();
        let consumer = thread::spawn(move || drop(other.pop()));
        consumer.join().unwrap();
        drop(queue);
        assert_eq!(first.load(Ordering::SeqCst), 1);
        assert_eq!(second.load(Ordering::SeqCst), 1);
    });
}
