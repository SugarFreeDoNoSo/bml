//! Conservative dependencies for independently executable operation fragments.
//!
//! Metadata is descriptive: indexed loads use absolute buffer addresses and
//! may cross slot boundaries. Only bytecode and the actual buffer layout prove
//! independence. Unknown indices read every reachable slot. Stack-carrying or
//! malformed fragments retain the historical sequential execution path.

use crate::buffer::{PartitionedBuffer, ResultBuffer};
use bml_compiler::{op_fragments::OperationFragment, RpnOp};
use bml_domain::EvalContext;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum IndexRange {
    Known(u32, u32),
    Unknown,
}
impl IndexRange {
    fn constant(value: f64) -> Self {
        if value.is_finite() && value >= 0.0 && value <= f64::from(u32::MAX) && value.fract() == 0.0
        {
            Self::Known(value as u32, value as u32)
        } else {
            Self::Unknown
        }
    }

    fn arithmetic(self, other: Self, multiply: bool) -> Self {
        if let (Self::Known(a, b), Self::Known(c, d)) = (self, other) {
            let lower = if multiply {
                a.checked_mul(c)
            } else {
                a.checked_add(c)
            };
            let upper = if multiply {
                b.checked_mul(d)
            } else {
                b.checked_add(d)
            };
            if let (Some(lower), Some(upper)) = (lower, upper) {
                return Self::Known(lower, upper);
            }
        }
        Self::Unknown
    }
}

pub(crate) struct Access {
    reads: Vec<bool>,
    writes: Vec<bool>,
    pub(crate) stack_capacity: usize,
}
impl Access {
    fn conflicts(&self, other: &Self) -> bool {
        (0..self.reads.len()).any(|i| {
            (self.writes[i] && (other.reads[i] || other.writes[i]))
                || (self.reads[i] && other.writes[i])
        })
    }

    fn read(&mut self, base: u32, offset: IndexRange, buffer: &ResultBuffer) {
        if buffer.len() == 0 || buffer.slot_size() == 0 {
            return;
        }
        let (low, high) = match offset {
            IndexRange::Known(low, high) => (low, high),
            // Match the legacy saturating f64 -> u32 conversion, including
            // NaN/negative values -> zero. Do not use context_index here.
            IndexRange::Unknown => (0, u32::MAX),
        };
        let Some(start) = base.checked_add(low).map(|n| n as usize) else {
            return;
        };
        let end = base.saturating_add(high) as usize;
        let end = end.min(buffer.len() - 1);
        if start <= end {
            for slot in start / buffer.slot_size()..=end / buffer.slot_size() {
                self.reads[slot] = true;
            }
        }
    }
}

struct Analyzer<'a> {
    context: &'a EvalContext<'a>,
    buffer: &'a ResultBuffer,
    access: Access,
    fuel: usize,
}
impl Analyzer<'_> {
    fn visit(&mut self, ops: &[RpnOp], stack: &mut Vec<IndexRange>, depth: usize) -> Option<()> {
        if depth > 32 {
            return None;
        }
        let mut pc = 0;
        while pc < ops.len() {
            self.fuel = self.fuel.checked_sub(1)?;
            match ops[pc] {
                RpnOp::Zero => stack.push(IndexRange::Known(0, 0)),
                RpnOp::One => stack.push(IndexRange::Known(1, 1)),
                RpnOp::Const(id) => stack.push(IndexRange::constant(self.context.get_const(id))),
                RpnOp::Var(id) => stack.push(IndexRange::constant(self.context.get_var(id))),
                RpnOp::ContextVarIndexed { .. } | RpnOp::ContextConstIndexed { .. } => {
                    stack.pop()?;
                    stack.push(IndexRange::Unknown);
                }
                RpnOp::VarIndexed { base } => {
                    self.access.read(base, stack.pop()?, self.buffer);
                    stack.push(IndexRange::Unknown);
                }
                RpnOp::StoreResult { slot } => {
                    stack.pop()?;
                    stack.pop()?;
                    if let Some(write) = self.access.writes.get_mut(slot as usize) {
                        *write = true;
                    }
                }
                RpnOp::Bml | RpnOp::FAdd | RpnOp::FMul => {
                    let right = stack.pop()?;
                    let left = stack.pop()?;
                    stack.push(match ops[pc] {
                        RpnOp::FAdd => left.arithmetic(right, false),
                        RpnOp::FMul => left.arithmetic(right, true),
                        _ => IndexRange::Unknown,
                    });
                }
                RpnOp::Dup => stack.push(*stack.last()?),
                RpnOp::Pick { depth } => {
                    let index = stack.len().checked_sub(depth as usize + 1)?;
                    stack.push(stack[index]);
                }
                RpnOp::Drop => {
                    stack.pop()?;
                }
                RpnOp::Swap => {
                    let len = stack.len();
                    let second = len.checked_sub(2)?;
                    stack.swap(second, len - 1);
                }
                RpnOp::Loop { count, body_len } => {
                    let end = (pc + 1).checked_add(body_len as usize)?;
                    let body = ops.get(pc + 1..end)?;
                    if count > 0 {
                        // Widen changing loop-carried values to Unknown. Each
                        // stack position widens at most once; the fixed point
                        // includes the entry and every possible iteration.
                        loop {
                            let mut next = stack.clone();
                            next.push(IndexRange::Known(0, count - 1));
                            self.access.stack_capacity = self.access.stack_capacity.max(next.len());
                            self.visit(body, &mut next, depth + 1)?;
                            if next.len() != stack.len() {
                                return None;
                            }
                            if count == 1 {
                                *stack = next;
                                break;
                            }
                            let mut changed = false;
                            for (old, new) in stack.iter_mut().zip(next) {
                                if *old != new && *old != IndexRange::Unknown {
                                    *old = IndexRange::Unknown;
                                    changed = true;
                                }
                            }
                            if !changed {
                                break;
                            }
                        }
                    }
                    pc = end;
                    continue;
                }
            }
            self.access.stack_capacity = self.access.stack_capacity.max(stack.len());
            pc += 1;
        }
        Some(())
    }
}

pub(crate) fn analyze(
    fragments: &[OperationFragment],
    context: &EvalContext<'_>,
    buffer: &ResultBuffer,
) -> Option<Vec<Access>> {
    fragments
        .iter()
        .map(|fragment| {
            let context = fragment.context(context);
            let mut analyzer = Analyzer {
                context: &context,
                buffer,
                access: Access {
                    reads: vec![false; buffer.n_slots()],
                    writes: vec![false; buffer.n_slots()],
                    stack_capacity: 0,
                },
                // Analysis is bounded independently of loop trip counts. Programs
                // exceeding this proof budget can still execute sequentially.
                fuel: fragment.fragment.ops.len().checked_mul(8)?.max(64),
            };
            let mut stack = Vec::new();
            analyzer.visit(&fragment.fragment.ops, &mut stack, 0)?;
            stack.is_empty().then_some(analyzer.access)
        })
        .collect()
}

/// A wave is a contiguous prefix so RAW, WAR and WAW preserve program order.
pub(crate) fn wave_len(accesses: &[Access], cores: usize) -> usize {
    let mut count = 1;
    while count < accesses.len().min(cores.max(1)) {
        if accesses[..count]
            .iter()
            .any(|old| old.conflicts(&accesses[count]))
        {
            break;
        }
        count += 1;
    }
    count
}

/// Scoped workers borrow input slots and own disjoint output slices. They
/// neither clone tensor data nor hold a per-fragment mutex during arithmetic.
pub(crate) fn run_wave<F>(
    fragments: &[OperationFragment],
    accesses: &[Access],
    buffer: &mut ResultBuffer,
    execute: F,
) where
    F: Fn(usize, &OperationFragment, &mut PartitionedBuffer<'_>) + Sync,
{
    let mut owners = vec![None; buffer.n_slots()];
    for (worker, access) in accesses.iter().enumerate() {
        for (slot, &write) in access.writes.iter().enumerate() {
            if write {
                assert!(owners[slot].replace(worker).is_none());
            }
        }
    }
    std::thread::scope(|scope| {
        let execute = &execute;
        let handles: Vec<_> = buffer
            .partition(&owners, fragments.len())
            .into_iter()
            .zip(fragments)
            .enumerate()
            .map(|(worker, (mut view, fragment))| {
                scope.spawn(move || execute(worker, fragment, &mut view))
            })
            .collect();
        for handle in handles {
            handle.join().unwrap();
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::HotLoop;
    use bml_compiler::{op_fragments::FragmentMeta, Fragment};
    use std::sync::{Condvar, Mutex};
    use std::time::Duration;

    fn operation(base: u32, output: u32) -> OperationFragment {
        OperationFragment {
            const_pool: None,
            fragment: Fragment {
                ops: vec![
                    RpnOp::Zero,
                    RpnOp::VarIndexed { base },
                    RpnOp::One,
                    RpnOp::FAdd,
                    RpnOp::Zero,
                    RpnOp::StoreResult { slot: output },
                ],
            },
            meta: FragmentMeta {
                name: "test".into(),
                input_slot: 0,
                output_slot: 0,
                weight_base: 0,
                n_rows: 0,
                n_cols: 0,
            },
        }
    }

    #[test]
    fn independent_workers_overlap_and_write_the_original_storage() {
        let fragments = [operation(0, 1), operation(0, 2)];
        let context = EvalContext::new(&[], &[]);
        let mut buffer = ResultBuffer::new(3, 1);
        buffer.write(0, 0, 10.0);
        let address = buffer.data().as_ptr();
        let accesses = analyze(&fragments, &context, &buffer).unwrap();
        assert_eq!(wave_len(&accesses, 2), 2);
        let arrived = Mutex::new(0);
        let ready = Condvar::new();
        run_wave(
            &fragments,
            &accesses,
            &mut buffer,
            |worker, fragment, view| {
                let mut count = arrived.lock().unwrap();
                *count += 1;
                ready.notify_all();
                // A serial/mutex-held executor fails instead of hanging forever.
                let (count, _) = ready
                    .wait_timeout_while(count, Duration::from_secs(10), |count| *count != 2)
                    .unwrap();
                assert_eq!(*count, 2, "both workers must enter before either leaves");
                drop(count);
                let mut hot = HotLoop::with_capacity(accesses[worker].stack_capacity);
                hot.execute_fragment_store(&fragment.fragment, &context, view);
                assert_eq!(hot.stack_depth(), 0);
            },
        );
        assert_eq!(buffer.data(), &[10.0, 11.0, 11.0]);
        assert_eq!(buffer.data().as_ptr(), address);
    }

    #[test]
    fn all_three_conflict_directions_end_the_wave() {
        let context = EvalContext::new(&[], &[]);
        let buffer = ResultBuffer::new(4, 1);
        for fragments in [
            [operation(0, 1), operation(1, 2)], // RAW
            [operation(1, 2), operation(0, 1)], // WAR
            [operation(0, 1), operation(2, 1)], // WAW
        ] {
            let accesses = analyze(&fragments, &context, &buffer).unwrap();
            assert_eq!(wave_len(&accesses, 2), 1);
        }
    }

    #[test]
    fn dependency_analysis_uses_each_operations_actual_constant_context() {
        let context = EvalContext::new(&[], &[0.0]);
        let buffer = ResultBuffer::new(4, 2);
        let first = operation(0, 2);
        let mut next = operation(0, 3);
        next.fragment.ops[0] = RpnOp::Const(0);
        next.const_pool = Some(vec![4.0]); // Actual read is slot 2, not slot 0.
        let accesses = analyze(&[first, next], &context, &buffer).unwrap();
        assert_eq!(wave_len(&accesses, 2), 1);
    }

    #[test]
    fn malformed_and_stack_carrying_programs_are_not_partitioned() {
        let context = EvalContext::new(&[], &[]);
        let buffer = ResultBuffer::new(4, 1);
        for ops in [
            vec![RpnOp::One],
            vec![RpnOp::Dup],
            vec![
                RpnOp::Loop {
                    count: 2,
                    body_len: 1,
                },
                RpnOp::One,
            ],
            vec![RpnOp::Loop {
                count: 0,
                body_len: 50,
            }],
        ] {
            let mut fragment = operation(0, 1);
            fragment.fragment.ops = ops;
            assert!(analyze(&[fragment], &context, &buffer).is_none());
        }
    }
}
