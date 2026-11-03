//! Slot publication/reuse probes. Epoch/reclamation remains uninstrumented.
use crate::{Steal, Worker, SLOT_CAPACITY};

#[derive(Debug, PartialEq, Eq)]
struct Payload {
    id: usize,
    checksum: usize,
}
impl Payload {
    fn new(id: usize) -> Self {
        Self {
            id,
            checksum: id ^ 0x5a5a,
        }
    }
    fn check(self) -> usize {
        assert_eq!(self.checksum, self.id ^ 0x5a5a);
        self.id
    }
}

fn model(check: impl Fn() + Sync + Send + 'static) {
    let mut builder = loom::model::Builder::new();
    builder.preemption_bound = Some(2);
    builder.max_branches = 10_000;
    builder.check(check);
}

#[test]
fn concurrent_push_publishes_payload_before_steal() {
    model(|| {
        let owner = Worker::<Payload>::new_fifo();
        let stealer = owner.stealer();
        // No publication flag or pre-read join: visibility must come from the
        // real push release fence/back store and thief acquire path alone.
        let thief = loom::thread::spawn(move || match stealer.steal() {
            Steal::Success(value) => Some(value.check()),
            Steal::Retry | Steal::Empty => None,
        });
        owner.push(Payload::new(7));
        let stolen = thief.join().unwrap();
        let local = owner.pop().map(Payload::check);
        assert_eq!(stolen.into_iter().chain(local).collect::<Vec<_>>(), [7]);
    });
}

#[test]
fn owner_reuses_every_slot_without_concurrent_readers() {
    model(|| {
        let owner = Worker::new_fifo();
        for id in 0..(SLOT_CAPACITY + 1) {
            owner.push(Payload::new(id));
            assert_eq!(owner.pop().unwrap().check(), id);
        }
        assert!(matches!(owner.stealer().steal(), Steal::Empty));
    });
}

#[test]
fn stale_thief_competes_with_wrapped_slot_reuse() {
    model(|| {
        let owner = Worker::new_fifo();
        owner.push(Payload::new(0));
        let stealer = owner.stealer();
        let thief = loom::thread::spawn(move || match stealer.steal() {
            Steal::Success(value) => Some(value.check()),
            Steal::Retry | Steal::Empty => None,
        });
        let mut found = Vec::new();
        found.extend(owner.pop().map(Payload::check));
        // At most one live element at a time: capacity does not grow or shrink.
        // The thief can retain a stale front/index while the owner wraps it.
        for id in 1..=SLOT_CAPACITY {
            owner.push(Payload::new(id));
            found.extend(owner.pop().map(Payload::check));
        }
        found.extend(thief.join().unwrap());
        while let Some(value) = owner.pop() {
            found.push(value.check());
        }
        found.sort_unstable();
        assert_eq!(found, (0..=SLOT_CAPACITY).collect::<Vec<_>>());
    });
}
