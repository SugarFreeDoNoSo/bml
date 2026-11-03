//! Bounded request admission and deadline-based batching.
//!
//! A single consumer owns node selection. Producers never block: full queues
//! return the original request for HTTP 429 handling or an explicit retry.
//! This groups requests; a backend must execute each returned batch. It does not
//! imply fused model execution or a network transport.
use crossbeam::channel::{bounded, Receiver, RecvTimeoutError, Sender, TrySendError};
use std::time::{Duration, Instant};

/// Limits on queued requests and each emitted batch.
#[derive(Debug, Clone, Copy)]
pub struct RequestBatchConfig {
    /// Maximum requests waiting in the channel, excluding an active batch.
    pub queue_capacity: usize,
    /// Maximum requests returned by one collection call.
    pub max_batch_size: usize,
    /// Collection deadline measured from receipt of the first request.
    pub window: Duration,
    /// Number of execution destinations, assigned round-robin.
    pub nodes: usize,
}
impl Default for RequestBatchConfig {
    fn default() -> Self {
        Self {
            queue_capacity: 64,
            max_batch_size: 16,
            window: Duration::from_millis(10),
            nodes: 1,
        }
    }
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// Invalid scheduler configuration.
pub enum BatchConfigError {
    /// Admission requires a nonzero queue capacity.
    ZeroCapacity,
    /// Each batch must allow at least one request.
    ZeroBatchSize,
    /// At least one execution destination is required.
    ZeroNodes,
    /// The collection window exceeds the platform clock range.
    WindowTooLarge,
}

/// Nonblocking admission failure. Ownership is always returned to the caller.
#[derive(Debug, PartialEq, Eq)]
pub enum AdmissionError<T> {
    /// The bounded queue has no available slot.
    Full(T),
    /// The consumer has been dropped.
    Closed(T),
}

/// Cloneable producer handle with nonblocking admission.
pub struct RequestSubmitter<T> {
    sender: Sender<T>,
}
impl<T> Clone for RequestSubmitter<T> {
    fn clone(&self) -> Self {
        Self {
            sender: self.sender.clone(),
        }
    }
}
impl<T> RequestSubmitter<T> {
    /// Enqueue a request or return its ownership when full or closed.
    pub fn try_submit(&self, request: T) -> Result<(), AdmissionError<T>> {
        self.sender.try_send(request).map_err(|e| match e {
            TrySendError::Full(request) => AdmissionError::Full(request),
            TrySendError::Disconnected(request) => AdmissionError::Closed(request),
        })
    }
}

#[derive(Debug, PartialEq, Eq)]
/// Requests collected within one window and assigned to one destination.
pub struct RequestBatch<T> {
    /// Zero-based round-robin execution destination.
    pub node: usize,
    /// Requests in channel receipt order.
    pub requests: Vec<T>,
}

/// Single-consumer batch builder. Dropping it closes all producer handles.
/// Queue capacity bounds waiting requests; an emitted/in-progress batch is
/// separate and bounded by `max_batch_size`.
pub struct RequestBatcher<T> {
    receiver: Receiver<T>,
    config: RequestBatchConfig,
    next_node: usize,
}
impl<T> RequestBatcher<T> {
    /// Validate limits and create a bounded channel with one consumer.
    pub fn new(
        config: RequestBatchConfig,
    ) -> Result<(RequestSubmitter<T>, Self), BatchConfigError> {
        if config.queue_capacity == 0 {
            return Err(BatchConfigError::ZeroCapacity);
        }
        if config.max_batch_size == 0 {
            return Err(BatchConfigError::ZeroBatchSize);
        }
        if config.nodes == 0 {
            return Err(BatchConfigError::ZeroNodes);
        }
        if Instant::now().checked_add(config.window).is_none() {
            return Err(BatchConfigError::WindowTooLarge);
        }
        let (sender, receiver) = bounded(config.queue_capacity);
        Ok((
            RequestSubmitter { sender },
            Self {
                receiver,
                config,
                next_node: 0,
            },
        ))
    }

    /// Wait for the first request, then collect until the batch is full or the
    /// fixed window expires. Continued arrivals never extend the deadline.
    /// Returns a partial batch on producer shutdown, then `None` after draining.
    pub fn next_batch(&mut self) -> Option<RequestBatch<T>> {
        let first = self.receiver.recv().ok()?;
        let started = Instant::now();
        let mut requests = vec![first];
        while requests.len() < self.config.max_batch_size {
            let remaining = self.config.window.saturating_sub(started.elapsed());
            if remaining.is_zero() {
                break;
            }
            match self.receiver.recv_timeout(remaining) {
                Ok(request) => requests.push(request),
                Err(RecvTimeoutError::Timeout | RecvTimeoutError::Disconnected) => break,
            }
        }
        let node = self.next_node;
        self.next_node = if node == self.config.nodes - 1 {
            0
        } else {
            node + 1
        };
        Some(RequestBatch { node, requests })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Barrier};

    #[test]
    fn bounded_admission_preserves_rejected_request_and_closes() {
        let config = RequestBatchConfig {
            queue_capacity: 1,
            max_batch_size: 1,
            ..Default::default()
        };
        let (tx, mut batches) = RequestBatcher::new(config).unwrap();
        tx.try_submit(String::from("first")).unwrap();
        assert_eq!(
            tx.try_submit(String::from("retry")),
            Err(AdmissionError::Full(String::from("retry")))
        );
        assert_eq!(batches.next_batch().unwrap().requests, ["first"]);
        tx.try_submit(String::from("next")).unwrap();
        drop(batches);
        assert_eq!(
            tx.try_submit(String::from("closed")),
            Err(AdmissionError::Closed(String::from("closed")))
        );
    }

    #[test]
    fn ten_concurrent_requests_are_batched_without_loss() {
        let (tx, mut batches) = RequestBatcher::new(RequestBatchConfig {
            max_batch_size: 8,
            nodes: 2,
            ..Default::default()
        })
        .unwrap();
        let barrier = Arc::new(Barrier::new(10));
        let handles: Vec<_> = (0..10)
            .map(|id| {
                let tx = tx.clone();
                let barrier = barrier.clone();
                std::thread::spawn(move || {
                    barrier.wait();
                    tx.try_submit(id).unwrap();
                })
            })
            .collect();
        for handle in handles {
            handle.join().unwrap();
        }
        // Admission is deterministic; do not assume OS threads wake within 10ms.
        drop(tx);
        let first = batches.next_batch().unwrap();
        let second = batches.next_batch().unwrap();
        assert_eq!((first.node, second.node), (0, 1));
        assert_eq!((first.requests.len(), second.requests.len()), (8, 2));
        let mut all = first.requests;
        all.extend(second.requests);
        all.sort_unstable();
        assert_eq!(all, (0..10).collect::<Vec<_>>());
        assert!(batches.next_batch().is_none());
    }

    #[test]
    fn collection_window_expires_with_live_producer() {
        let (tx, mut batches) = RequestBatcher::new(RequestBatchConfig::default()).unwrap();
        tx.try_submit(7).unwrap();
        let start = Instant::now();
        let batch = batches.next_batch().unwrap();
        assert!(start.elapsed() >= Duration::from_millis(10));
        assert_eq!(batch.requests, [7]);
        // No upper latency claim: the OS can deschedule the test thread.
        tx.try_submit(8).unwrap();
    }

    #[test]
    fn round_robin_wraps_and_preserves_fifo() {
        let (tx, mut batches) = RequestBatcher::new(RequestBatchConfig {
            max_batch_size: 1,
            nodes: 3,
            ..Default::default()
        })
        .unwrap();
        for id in 0..8 {
            tx.try_submit(id).unwrap();
        }
        drop(tx);
        for id in 0..8 {
            assert_eq!(
                batches.next_batch(),
                Some(RequestBatch {
                    node: id % 3,
                    requests: vec![id]
                })
            );
        }
        assert!(batches.next_batch().is_none());
    }

    #[test]
    fn invalid_limits_and_zero_window_are_explicit() {
        let base = RequestBatchConfig::default();
        for (config, expected) in [
            (
                RequestBatchConfig {
                    queue_capacity: 0,
                    ..base
                },
                BatchConfigError::ZeroCapacity,
            ),
            (
                RequestBatchConfig {
                    max_batch_size: 0,
                    ..base
                },
                BatchConfigError::ZeroBatchSize,
            ),
            (
                RequestBatchConfig { nodes: 0, ..base },
                BatchConfigError::ZeroNodes,
            ),
            (
                RequestBatchConfig {
                    window: Duration::MAX,
                    ..base
                },
                BatchConfigError::WindowTooLarge,
            ),
        ] {
            assert_eq!(RequestBatcher::<()>::new(config).err(), Some(expected));
        }
        let (tx, mut batches) = RequestBatcher::new(RequestBatchConfig {
            window: Duration::ZERO,
            ..base
        })
        .unwrap();
        tx.try_submit(1).unwrap();
        tx.try_submit(2).unwrap();
        drop(tx);
        assert_eq!(batches.next_batch().unwrap().requests, [1]);
        assert_eq!(batches.next_batch().unwrap().requests, [2]);
    }
}
