//! Cola lock-free con work-stealing para distribución de fragmentos.
//!
//! Usa `crossbeam-queue::SegQueue` para una cola FIFO lock-free por nodo.
//! Cuando un nodo vacía su cola, puede robar trabajo de otro nodo.
//!
//! # Diseño
//!
//! Cada nodo y sus handles de robo comparten una cola segmentada. Pop reclama
//! el índice antes de leer el slot; los segmentos agotados se retiran, sin la
//! lectura especulativa seguida de CAS del antiguo deque circular. Los handles
//! de robo solo permiten extraer trabajo; el propietario publica con push.

use bml_compiler::Fragment;
use crossbeam::{deque::Steal, queue::SegQueue};
use std::sync::Arc;

/// Bounded multi-producer queue for pending work shared by connection threads.
/// Each successful pop transfers ownership once; transport failures after pop
/// need application-level acknowledgement/recovery and are not exactly-once delivery.
pub struct PendingWork<T> {
    queue: bml_queue::ArrayQueue<T>,
}
impl<T> PendingWork<T> {
    /// Construct a nonempty-capacity queue; zero capacity is rejected.
    pub fn new(capacity: usize) -> Option<Self> {
        (capacity > 0).then(|| Self {
            queue: bml_queue::ArrayQueue::new(capacity),
        })
    }
    /// Admit work or return it unchanged when full.
    pub fn push(&self, work: T) -> Result<(), T> {
        self.queue.push(work)
    }
    /// Remove pending work for local execution or remote stealing.
    pub fn pop(&self) -> Option<T> {
        self.queue.pop()
    }
    /// Snapshot of queued work, excluding work already owned by a consumer.
    pub fn len(&self) -> usize {
        self.queue.len()
    }
    /// Whether no work is currently queued.
    pub fn is_empty(&self) -> bool {
        self.queue.is_empty()
    }
}

/// Handle clonable de extracción FIFO para otros nodos.
///
/// Reemplaza el handle específico del deque. La cola segmentada reclama la
/// posición antes de leer datos; esta interfaz no expone publicación remota.
#[derive(Clone)]
pub struct WorkStealer {
    queue: Arc<SegQueue<Fragment>>,
}

impl WorkStealer {
    /// Transfiere un fragmento o informa vacío. No devuelve Retry.
    pub fn steal(&self) -> Steal<Fragment> {
        match self.queue.pop() {
            Some(fragment) => Steal::Success(fragment),
            None => Steal::Empty,
        }
    }
}

/// Cola FIFO segmentada de un nodo, compartida con handles de robo.
pub struct WorkQueue {
    queue: Arc<SegQueue<Fragment>>,
    stealer: WorkStealer,
}

impl WorkQueue {
    /// Crea una nueva cola de trabajo.
    pub fn new() -> Self {
        let queue = Arc::new(SegQueue::new());
        let stealer = WorkStealer {
            queue: Arc::clone(&queue),
        };
        Self { queue, stealer }
    }

    /// Empuja un fragmento a la cola local.
    pub fn push(&self, fragment: Fragment) {
        self.queue.push(fragment);
    }

    /// Saca un fragmento de la cola local (no bloqueante).
    pub fn pop(&self) -> Option<Fragment> {
        self.queue.pop()
    }

    /// Retorna el stealer para que otros nodos puedan robar.
    pub fn stealer(&self) -> &WorkStealer {
        &self.stealer
    }

    /// Intenta robar un fragmento de otro nodo.
    pub fn steal_from(&self, other: &WorkStealer) -> Option<Fragment> {
        other.queue.pop()
    }
}

impl Default for WorkQueue {
    fn default() -> Self {
        Self::new()
    }
}

/// Conjunto de colas para múltiples nodos.
///
/// El coordinador empuja trabajo a las colas, y los workers
/// pueden robar entre sí.
pub struct WorkQueueSet {
    queues: Vec<WorkQueue>,
}

impl WorkQueueSet {
    /// Crea un conjunto de N colas.
    pub fn new(n: usize) -> Self {
        let queues = (0..n).map(|_| WorkQueue::new()).collect();
        Self { queues }
    }

    /// Número de colas.
    pub fn len(&self) -> usize {
        self.queues.len()
    }

    /// Empuja un fragmento a la cola del nodo `node_idx`.
    pub fn push(&self, node_idx: usize, fragment: Fragment) {
        self.queues[node_idx].push(fragment);
    }

    /// Saca un fragmento de la cola del nodo `node_idx`.
    pub fn pop(&self, node_idx: usize) -> Option<Fragment> {
        self.queues[node_idx].pop()
    }

    /// Roba trabajo de otro nodo para el nodo `node_idx`.
    ///
    /// Intenta robar de todos los demás nodos en orden round-robin.
    pub fn steal_for(&self, node_idx: usize) -> Option<Fragment> {
        let n = self.queues.len();
        for i in 1..n {
            let target = (node_idx + i) % n;
            let stealer = self.queues[target].stealer();
            if let Some(frag) = self.queues[node_idx].steal_from(stealer) {
                return Some(frag);
            }
        }
        None
    }

    /// Obtiene una referencia a la cola de un nodo.
    pub fn get(&self, node_idx: usize) -> &WorkQueue {
        &self.queues[node_idx]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bml_compiler::RpnOp;

    fn make_fragment(n: usize) -> Fragment {
        Fragment {
            ops: (0..n).map(|_| RpnOp::One).collect(),
        }
    }

    #[test]
    fn segmented_work_ownership_is_unique_across_turnover_and_owner_drop() {
        let owner = WorkQueue::new();
        let thieves: Vec<_> = (0..2).map(|_| owner.stealer().clone()).collect();
        for id in 1..=96 {
            owner.push(make_fragment(id));
        }
        let handles: Vec<_> = thieves
            .into_iter()
            .map(|thief| {
                std::thread::spawn(move || {
                    let mut ids = Vec::new();
                    while let Steal::Success(fragment) = thief.steal() {
                        ids.push(fragment.ops.len());
                    }
                    ids
                })
            })
            .collect();
        let mut ids = Vec::new();
        if let Some(fragment) = owner.pop() {
            ids.push(fragment.ops.len());
        }
        // Handles retain ownership of the queue after the original node exits.
        drop(owner);
        for handle in handles {
            ids.extend(handle.join().unwrap());
        }
        ids.sort_unstable();
        assert_eq!(ids, (1..=96).collect::<Vec<_>>());
    }

    #[test]
    fn push_pop_single_queue() {
        let q = WorkQueue::new();
        q.push(make_fragment(3));
        q.push(make_fragment(5));

        let f1 = q.pop().unwrap();
        assert_eq!(f1.ops.len(), 3);
        let f2 = q.pop().unwrap();
        assert_eq!(f2.ops.len(), 5);
        assert!(q.pop().is_none());
    }

    #[test]
    fn steal_from_other_queue() {
        let q1 = WorkQueue::new();
        let q2 = WorkQueue::new();

        // q1 tiene trabajo, q2 está vacía
        q1.push(make_fragment(10));
        q1.push(make_fragment(20));

        // q2 roba de q1
        let stolen = q2.steal_from(q1.stealer());
        assert!(stolen.is_some());
        assert_eq!(stolen.unwrap().ops.len(), 10); // FIFO: roba el primero

        // q1 todavía tiene trabajo
        assert_eq!(q1.pop().unwrap().ops.len(), 20);
    }

    #[test]
    fn queue_set_distribute_and_steal() {
        let set = WorkQueueSet::new(3);

        // Distribuir 6 fragmentos round-robin
        for i in 0..6 {
            set.push(i % 3, make_fragment(i + 1));
        }

        // Nodo 0 tiene 2 fragmentos (1, 4)
        assert_eq!(set.pop(0).unwrap().ops.len(), 1);
        assert_eq!(set.pop(0).unwrap().ops.len(), 4);
        assert!(set.pop(0).is_none());

        // Nodo 0 roba de otros
        let stolen = set.steal_for(0);
        assert!(stolen.is_some());
    }

    #[test]
    fn empty_queue_steal_returns_none() {
        let set = WorkQueueSet::new(2);
        // Todas vacías
        assert!(set.steal_for(0).is_none());
    }

    #[test]
    fn concurrent_push_pop() {
        use std::sync::Arc;
        use std::thread;

        // Crear 4 workers y sus stealers
        let workers: Vec<_> = (0..4).map(|_| WorkQueue::new()).collect();
        let stealers: Vec<_> = workers.iter().map(|w| w.stealer().clone()).collect();
        let stealers = Arc::new(stealers);

        // Llenar colas
        for (i, w) in workers.iter().enumerate() {
            for _ in 0..25 {
                w.push(make_fragment(1));
            }
        }

        // Mover workers a threads
        let handles: Vec<_> = workers
            .into_iter()
            .enumerate()
            .map(|(idx, worker)| {
                let stealers = Arc::clone(&stealers);
                thread::spawn(move || {
                    let mut count = 0;
                    loop {
                        if worker.pop().is_some() {
                            count += 1;
                        } else {
                            // Intentar robar de otros
                            let mut stolen = false;
                            for i in 1..4 {
                                let target = (idx + i) % 4;
                                loop {
                                    match stealers[target].steal() {
                                        Steal::Success(_) => {
                                            count += 1;
                                            stolen = true;
                                            break;
                                        }
                                        Steal::Empty => break,
                                        Steal::Retry => continue,
                                    }
                                }
                                if stolen {
                                    break;
                                }
                            }
                            if !stolen {
                                break;
                            }
                        }
                    }
                    count
                })
            })
            .collect();

        let total: usize = handles.into_iter().map(|h| h.join().unwrap()).sum();
        assert_eq!(total, 100, "todos los fragmentos deben ser consumidos");
    }
}
