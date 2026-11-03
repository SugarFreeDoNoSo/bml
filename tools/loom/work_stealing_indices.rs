//! Index-ownership probe of the pinned upstream FIFO deque, not a slot-memory model.
use crate::{Steal, Worker};

fn explore(items: usize, thieves: usize) {
    let mut model = loom::model::Builder::new();
    model.preemption_bound = Some(2);
    model.max_branches = 10_000;
    model.check(move || {
        let owner = Worker::new_fifo();
        // Published before spawning: this probe deliberately excludes concurrent
        // slot writes/reuse and buffer resizing (upstream minimum capacity 64).
        for id in 0..items {
            owner.push(id);
        }
        let mut handles = Vec::new();
        for _ in 0..thieves {
            let stealer = owner.stealer();
            handles.push(loom::thread::spawn(move || match stealer.steal() {
                Steal::Success(id) => Some(id),
                Steal::Empty | Steal::Retry => None,
            }));
        }
        let mut found = Vec::new();
        if let Some(id) = owner.pop() {
            found.push(id);
        }
        for handle in handles {
            found.extend(handle.join().unwrap());
        }
        while let Some(id) = owner.pop() {
            found.push(id);
        }
        found.sort_unstable();
        assert_eq!(
            found,
            (0..items).collect::<Vec<_>>(),
            "lost or duplicated ownership"
        );
        assert!(matches!(owner.stealer().steal(), Steal::Empty));
    });
}

#[test]
fn fifo_last_item_owner_and_thief() {
    explore(1, 1);
}

#[test]
fn fifo_last_item_owner_and_two_thieves() {
    explore(1, 2);
}

#[test]
fn fifo_two_items_owner_and_two_thieves() {
    explore(2, 2);
}
