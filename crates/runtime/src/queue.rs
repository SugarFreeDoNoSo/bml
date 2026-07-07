//! Cola lock-free con work-stealing para distribución de fragmentos.
//!
//! Usa `crossbeam-deque` (Chase-Lev deque) para una cola lock-free por nodo.
//! Cuando un nodo vacía su cola, puede robar trabajo de otro nodo.
//!
//! # Diseño
//!
//! Cada nodo tiene un `Worker` con su cola local. El coordinador empuja
//! fragmentos a las colas via `Stealer`. Los workers pueden robar de
//! otros workers cuando su cola está vacía.

use bml_compiler::Fragment;
use crossbeam::deque::{Steal, Stealer, Worker};

/// Cola de trabajo de un nodo.
///
/// Contiene un `Worker` (local) y un `Stealer` (para que otros roben).
pub struct WorkQueue {
    worker: Worker<Fragment>,
    stealer: Stealer<Fragment>,
}

impl WorkQueue {
    /// Crea una nueva cola de trabajo.
    pub fn new() -> Self {
        let worker = Worker::new_fifo();
        let stealer = worker.stealer();
        Self { worker, stealer }
    }

    /// Empuja un fragmento a la cola local.
    pub fn push(&self, fragment: Fragment) {
        self.worker.push(fragment);
    }

    /// Saca un fragmento de la cola local (no bloqueante).
    pub fn pop(&self) -> Option<Fragment> {
        self.worker.pop()
    }

    /// Retorna el stealer para que otros nodos puedan robar.
    pub fn stealer(&self) -> &Stealer<Fragment> {
        &self.stealer
    }

    /// Intenta robar un fragmento de otro nodo.
    pub fn steal_from(&self, other: &Stealer<Fragment>) -> Option<Fragment> {
        loop {
            match other.steal() {
                Steal::Success(fragment) => return Some(fragment),
                Steal::Empty => return None,
                Steal::Retry => continue,
            }
        }
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
        let workers: Vec<_> = (0..4).map(|_| Worker::<Fragment>::new_fifo()).collect();
        let stealers: Vec<Stealer<Fragment>> = workers.iter().map(|w| w.stealer()).collect();
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
