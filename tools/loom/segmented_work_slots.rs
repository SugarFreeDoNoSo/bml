use crate::{SegQueue, BLOCK_CAP};
use loom::sync::atomic::{AtomicUsize, Ordering};
use loom::sync::Arc;

struct DropPayload {
    id: usize,
    drops: Arc<Vec<AtomicUsize>>,
}

impl Drop for DropPayload {
    fn drop(&mut self) {
        assert_eq!(
            self.drops[self.id].fetch_add(1, Ordering::Relaxed),
            0,
            "payload destroyed more than once"
        );
    }
}

#[test]
fn drop_once_across_retirement_and_abandoned_tail() {
    model(|| {
        let count = BLOCK_CAP + 3;
        let drops = Arc::new((0..count).map(|_| AtomicUsize::new(0)).collect::<Vec<_>>());
        let owner = Arc::new(SegQueue::new());
        for id in 0..count {
            owner.push(DropPayload {
                id,
                drops: drops.clone(),
            });
        }
        // Leave two positions in the first segment for competing consumers.
        for _ in 0..BLOCK_CAP - 2 {
            drop(owner.pop().unwrap());
        }
        let thief = owner.clone();
        let handle = loom::thread::spawn(move || {
            let held = thief.pop().unwrap();
            loom::thread::yield_now();
            drop(held);
        });
        drop(owner.pop().unwrap());
        // The last handle must destroy the three unconsumed tail payloads.
        drop(owner);
        handle.join().unwrap();
        for counter in drops.iter() {
            assert_eq!(
                counter.load(Ordering::Relaxed),
                1,
                "payload was not destroyed"
            );
        }
    });
}

#[test]
fn popped_payload_survives_queue_destruction() {
    model(|| {
        let drops = Arc::new(vec![AtomicUsize::new(0), AtomicUsize::new(0)]);
        let queue = SegQueue::new();
        for id in 0..2 {
            queue.push(DropPayload {
                id,
                drops: drops.clone(),
            });
        }
        let held = queue.pop().unwrap();
        drop(queue);
        assert_eq!(drops[0].load(Ordering::Relaxed), 0);
        assert_eq!(drops[1].load(Ordering::Relaxed), 1);
        drop(held);
        assert_eq!(drops[0].load(Ordering::Relaxed), 1);
    });
}

#[derive(Debug)]
struct Payload(usize, usize);
impl Payload {
    fn new(id: usize) -> Self {
        Self(id, id ^ 0x5a5a)
    }
    fn check(self) -> usize {
        assert_eq!(self.1, self.0 ^ 0x5a5a);
        self.0
    }
}
fn model(check: impl Fn() + Sync + Send + 'static) {
    let mut builder = loom::model::Builder::new();
    builder.preemption_bound = Some(2);
    builder.max_branches = 10_000;
    builder.check(move || {
        check();
        crate::allocations::assert_balanced();
    });
}

#[test]
fn concurrent_push_publishes_payload_before_steal() {
    model(|| {
        let owner = Arc::new(SegQueue::<Payload>::new());
        let stealer = owner.clone();
        let thief = loom::thread::spawn(move || stealer.pop().map(Payload::check));
        owner.push(Payload::new(7));
        let stolen = thief.join().unwrap();
        let local = owner.pop().map(Payload::check);
        assert_eq!(stolen.into_iter().chain(local).collect::<Vec<_>>(), [7]);
    });
}

#[test]
fn delayed_thief_competes_with_segment_turnover() {
    model(|| {
        let owner = Arc::new(SegQueue::new());
        // Advance close to the boundary before introducing concurrent access.
        for id in 0..(BLOCK_CAP - 1) {
            owner.push(Payload::new(id));
            assert_eq!(owner.pop().unwrap().check(), id);
        }
        owner.push(Payload::new(0));
        let stealer = owner.clone();
        let thief = loom::thread::spawn(move || stealer.pop().map(Payload::check));
        let mut found = Vec::new();
        found.extend(owner.pop().map(Payload::check));
        for id in 1..=2 {
            owner.push(Payload::new(id));
            found.extend(owner.pop().map(Payload::check));
        }
        found.extend(thief.join().unwrap());
        while let Some(value) = owner.pop() {
            found.push(value.check());
        }
        found.sort_unstable();
        assert_eq!(found, [0, 1, 2]);
    });
}

#[test]
fn two_thieves_and_owner_transfer_each_item_once() {
    model(|| {
        let owner = Arc::new(SegQueue::new());
        owner.push(Payload::new(0));
        owner.push(Payload::new(1));
        let mut handles = Vec::new();
        for _ in 0..2 {
            let thief = owner.clone();
            handles.push(loom::thread::spawn(move || thief.pop().map(Payload::check)));
        }
        let mut found = Vec::new();
        found.extend(owner.pop().map(Payload::check));
        for handle in handles {
            found.extend(handle.join().unwrap());
        }
        while let Some(value) = owner.pop() {
            found.push(value.check());
        }
        found.sort_unstable();
        assert_eq!(found, [0, 1]);
    });
}
