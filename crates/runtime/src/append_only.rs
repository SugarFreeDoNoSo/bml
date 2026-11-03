//! Bounded, immutable-after-publication numerical results.
//! Slots are reserved once and never recycled; capacity exhaustion is explicit.
#[cfg(all(test, feature = "loom"))]
use loom::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
#[cfg(not(all(test, feature = "loom")))]
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};

struct Slot {
    bits: AtomicU64,
    ready: AtomicBool,
}
/// A complete record cannot fit; no slots were reserved or changed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AppendFull;
impl std::fmt::Display for AppendFull {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("append-only result capacity exhausted")
    }
}
impl std::error::Error for AppendFull {}
/// Concurrent result storage with stable indices and no overwrite or wraparound.
/// Publication uses release/acquire ordering; readers never see a partially
/// published scalar. All IEEE bit patterns, including NaN and signed zero, survive.
pub struct AppendOnlyResults {
    slots: Vec<Slot>,
    reserved: AtomicUsize,
}
impl AppendOnlyResults {
    /// Reserve all slots up front, returning allocator failure to the caller.
    pub fn new(capacity: usize) -> Result<Self, std::collections::TryReserveError> {
        let mut slots = Vec::new();
        slots.try_reserve_exact(capacity)?;
        for _ in 0..capacity {
            slots.push(Slot {
                bits: AtomicU64::new(0),
                ready: AtomicBool::new(false),
            });
        }
        Ok(Self {
            slots,
            reserved: AtomicUsize::new(0),
        })
    }
    /// Publish once into the next available slot without allocating. On full
    /// capacity returns the original value; existing slots remain unchanged.
    pub fn append(&self, value: f64) -> Result<usize, f64> {
        let index = self.reserve(1).map_err(|_| value)?;
        self.publish(index, value);
        Ok(index)
    }
    fn reserve(&self, count: usize) -> Result<usize, AppendFull> {
        self.reserved
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |index| {
                index
                    .checked_add(count)
                    .filter(|end| *end <= self.slots.len())
            })
            .map_err(|_| AppendFull)
    }
    fn publish(&self, index: usize, value: f64) {
        let slot = &self.slots[index];
        slot.bits.store(value.to_bits(), Ordering::Relaxed);
        slot.ready.store(true, Ordering::Release);
    }
    /// Reserve and publish a scalar followed by its associated fields, without
    /// allocation. Capacity failure reserves nothing. The returned range is
    /// complete; observers of reserved_count may see individual slots while the
    /// record is being published, so it is not an atomic multi-value snapshot.
    pub fn append_record(
        &self,
        value: f64,
        fields: &[f64],
    ) -> Result<std::ops::Range<usize>, AppendFull> {
        let count = fields.len().checked_add(1).ok_or(AppendFull)?;
        let start = self.reserve(count)?;
        self.publish(start, value);
        for (offset, field) in fields.iter().enumerate() {
            self.publish(start + 1 + offset, *field);
        }
        Ok(start..start + count)
    }
    /// Read a published slot, or None if absent/out of bounds/still being written.
    pub fn get(&self, index: usize) -> Option<f64> {
        let slot = self.slots.get(index)?;
        slot.ready
            .load(Ordering::Acquire)
            .then(|| f64::from_bits(slot.bits.load(Ordering::Relaxed)))
    }
    /// Fixed slot capacity, independent of publication progress.
    pub fn capacity(&self) -> usize {
        self.slots.len()
    }
    /// Number of reserved slots; a producer may not yet have published its value.
    pub fn reserved_count(&self) -> usize {
        self.reserved.load(Ordering::Relaxed)
    }
}

#[cfg(all(test, not(feature = "loom")))]
mod tests {
    use super::*;
    #[test]
    fn no_overwrite_and_ieee_payload_preservation() {
        let output = AppendOnlyResults::new(2).unwrap();
        let nan = f64::from_bits(0x7ff8_0000_0000_0123);
        assert_eq!(output.append(-0.), Ok(0));
        assert_eq!(output.append(nan), Ok(1));
        assert_eq!(output.append(99.), Err(99.));
        assert_eq!(output.get(0).unwrap().to_bits(), (-0f64).to_bits());
        assert_eq!(output.get(1).unwrap().to_bits(), nan.to_bits());
        assert!(output.get(2).is_none());
        assert_eq!(output.reserved_count(), 2);
        assert_eq!(AppendOnlyResults::new(0).unwrap().append(1.), Err(1.));
    }
    #[test]
    fn multicore_append_publishes_each_result_at_a_unique_stable_index() {
        let output = std::sync::Arc::new(AppendOnlyResults::new(4000).unwrap());
        let handles: Vec<_> = (0..4)
            .map(|worker| {
                let output = output.clone();
                std::thread::spawn(move || {
                    let mut runtime = crate::Runtime::new(4, 0);
                    let mut program = bml_compiler::RpnProgram {
                        const_pool: vec![0.],
                        ops: vec![bml_compiler::RpnOp::Const(0)],
                    };
                    for item in 0..1000 {
                        let value = (worker * 1000 + item) as f64;
                        program.const_pool[0] = value;
                        let index = runtime.execute_append_only(&program, &output).unwrap();
                        assert_eq!(output.get(index), Some(value));
                    }
                })
            })
            .collect();
        for handle in handles {
            handle.join().unwrap();
        }
        let mut values: Vec<_> = (0..4000).map(|i| output.get(i).unwrap() as usize).collect();
        values.sort_unstable();
        assert_eq!(values, (0..4000).collect::<Vec<_>>());
        assert_eq!(output.append(4001.), Err(4001.));
    }
}
#[cfg(all(test, feature = "loom"))]
mod loom_tests {
    use super::*;
    use loom::{sync::Arc, thread};
    #[test]
    fn concurrent_complete_records_receive_disjoint_stable_ranges() {
        loom::model(|| {
            let output = Arc::new(AppendOnlyResults::new(4).unwrap());
            let other = output.clone();
            let first = thread::spawn(move || other.append_record(11.0, &[-0.0]).unwrap());
            let second = output.append_record(21.0, &[22.0]).unwrap();
            let first = first.join().unwrap();
            assert!(first.end <= second.start || second.end <= first.start);
            assert_eq!(output.get(first.start), Some(11.0));
            assert_eq!(
                output.get(first.start + 1).unwrap().to_bits(),
                (-0.0f64).to_bits()
            );
            assert_eq!(output.get(second.start), Some(21.0));
            assert_eq!(output.get(second.start + 1), Some(22.0));
            assert_eq!(output.reserved_count(), 4);
        });
    }
    #[test]
    fn record_reservations_do_not_overlap_or_consume_capacity_on_failure() {
        loom::model(|| {
            let output = Arc::new(AppendOnlyResults::new(3).unwrap());
            let other = output.clone();
            let first = thread::spawn(move || other.append_record(11.0, &[12.0]));
            let second = output.append_record(21.0, &[22.0]);
            let first = first.join().unwrap();
            assert_ne!(first.is_ok(), second.is_ok());
            let expected = if first.is_ok() {
                (11.0, 12.0)
            } else {
                (21.0, 22.0)
            };
            assert_eq!(output.get(0), Some(expected.0));
            assert_eq!(output.get(1), Some(expected.1));
            assert_eq!(output.reserved_count(), 2);
            assert_eq!(output.append(33.0), Ok(2));
            assert_eq!(output.get(2), Some(33.0));
        });
    }
    fn execute(value: f64, output: &AppendOnlyResults) -> usize {
        let program = bml_compiler::RpnProgram {
            const_pool: vec![value],
            ops: vec![bml_compiler::RpnOp::Const(0)],
        };
        crate::Runtime::new(4, 0)
            .execute_append_only(&program, output)
            .unwrap()
    }
    #[test]
    fn concurrent_publish_and_read_explores_real_storage_atomics() {
        loom::model(|| {
            let output = Arc::new(AppendOnlyResults::new(2).unwrap());
            let a = output.clone();
            let b = output.clone();
            let first = thread::spawn(move || execute(11., &a));
            let second = thread::spawn(move || execute(22., &b));
            if let Some(value) = output.get(0) {
                assert!(value == 11. || value == 22.);
            }
            let i = first.join().unwrap();
            let j = second.join().unwrap();
            assert_ne!(i, j);
            assert_eq!(output.get(i), Some(11.));
            assert_eq!(output.get(j), Some(22.));
            assert_eq!(output.append(33.), Err(33.));
        });
    }
    #[test]
    fn saturation_race_preserves_the_winner() {
        loom::model(|| {
            let output = Arc::new(AppendOnlyResults::new(1).unwrap());
            let other = output.clone();
            let first = thread::spawn(move || other.append(11.));
            let second = output.append(22.);
            let first = first.join().unwrap();
            assert_ne!(first.is_ok(), second.is_ok());
            assert_eq!(output.get(0), Some(if first.is_ok() { 11. } else { 22. }));
        });
    }
}
