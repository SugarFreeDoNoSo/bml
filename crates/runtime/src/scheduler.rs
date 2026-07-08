//! Scheduler de waves: ejecuta sub-fragmentos en paralelo respetando
//! dependencias del DAG del transformer.
//!
//! # Modelo
//!
//! El transformer tiene etapas paralelas (Q, K, V, gate, up) y seriales
//! (attention, output, residual). El scheduler agrupa sub-fragmentos
//! sin dependencias entre sí en "waves" que se ejecutan en paralelo,
//! con barreras entre waves.
//!
//! ```
//! Wave 1 (paralela): Q, K, V, gate, up    → 4 threads
//! ─── barrera ───
//! Wave 2 (serial):   attention            → 1 thread
//! ─── barrera ───
//! Wave 3 (serial):   output + residual    → 1 thread
//! ```

use bml_compiler::distributed::SubFragment;
use bml_domain::EvalContext;
use crate::buffer::ResultBuffer;
use crate::hot_loop::HotLoop;
use std::sync::{Arc, Mutex, Barrier};
use std::thread;

/// Scheduler de waves para sub-fragmentos con dependencias.
pub struct WaveScheduler {
    /// Sub-fragmentos a ejecutar.
    sub_fragments: Vec<SubFragment>,
    /// Mapa: sub_id → índice en sub_fragments.
    id_to_idx: std::collections::HashMap<u32, usize>,
}

impl WaveScheduler {
    /// Crea un scheduler a partir de una lista de sub-fragmentos.
    pub fn new(sub_fragments: Vec<SubFragment>) -> Self {
        let id_to_idx = sub_fragments
            .iter()
            .enumerate()
            .map(|(i, sf)| (sf.sub_id, i))
            .collect();
        Self {
            sub_fragments,
            id_to_idx,
        }
    }

    /// Calcula las waves de ejecución.
    ///
    /// Una wave es un conjunto de sub-fragmentos cuyas dependencias
    /// ya fueron completadas por waves anteriores.
    ///
    /// # Retorna
    ///
    /// `Vec<Vec<u32>>` donde cada Vec interno es una wave de sub_ids.
    pub fn compute_waves(&self) -> Vec<Vec<u32>> {
        let mut waves = Vec::new();
        let mut completed: std::collections::HashSet<u32> = std::collections::HashSet::new();
        let mut remaining: std::collections::HashSet<u32> =
            self.sub_fragments.iter().map(|sf| sf.sub_id).collect();

        while !remaining.is_empty() {
            // Sub-fragmentos cuyas dependencias ya están completadas
            let wave: Vec<u32> = remaining
                .iter()
                .filter(|&&sid| {
                    let idx = self.id_to_idx[&sid];
                    let sf = &self.sub_fragments[idx];
                    sf.depends_on.iter().all(|dep| completed.contains(dep))
                })
                .copied()
                .collect();

            if wave.is_empty() {
                // Deadlock: dependencias circulares
                panic!("Deadlock: dependencias circulares en sub-fragmentos: {:?}", remaining);
            }

            for sid in &wave {
                completed.insert(*sid);
                remaining.remove(sid);
            }

            waves.push(wave);
        }

        waves
    }

    /// Ejecuta los sub-fragmentos respetando las waves.
    ///
    /// Cada wave se ejecuta en paralelo con `n_cores` threads.
    /// Barrera entre waves.
    pub fn execute(
        &self,
        n_cores: usize,
        ctx: &EvalContext,
        buf: &mut ResultBuffer,
    ) {
        let waves = self.compute_waves();

        for wave in &waves {
            if wave.len() <= 1 || n_cores <= 1 {
                // Ejecución serial
                for &sid in wave {
                    let sf = &self.sub_fragments[self.id_to_idx[&sid]];
                    let frag = bml_compiler::Fragment { ops: sf.ops.clone() };
                    let mut hot = HotLoop::with_capacity(8192);
                    hot.execute_fragment_full(&frag, ctx, buf);
                }
            } else {
                // Ejecución paralela con barrera
                self.execute_wave_parallel(wave, n_cores, ctx, buf);
            }
        }
    }

    fn execute_wave_parallel(
        &self,
        wave: &[u32],
        n_cores: usize,
        ctx: &EvalContext,
        buf: &mut ResultBuffer,
    ) {
        let n_threads = wave.len().min(n_cores);
        let barrier = Arc::new(Barrier::new(n_threads));
        let buf_arc = Arc::new(Mutex::new(std::mem::replace(
            buf,
            ResultBuffer::new(0, 0),
        )));

        let ctx_inputs = ctx.inputs.to_vec();
        let ctx_weights = ctx.weights.to_vec();

        let handles: Vec<_> = (0..n_threads)
            .map(|tid| {
                let sid = wave[tid];
                let sf = self.sub_fragments[self.id_to_idx[&sid]].clone();
                let barrier = Arc::clone(&barrier);
                let buf = Arc::clone(&buf_arc);
                let inputs = ctx_inputs.clone();
                let weights = ctx_weights.clone();

                thread::spawn(move || {
                    let ctx = EvalContext::new(&inputs, &weights);
                    let frag = bml_compiler::Fragment { ops: sf.ops.clone() };
                    let mut hot = HotLoop::with_capacity(8192);

                    // Warmup
                    hot.execute_fragment_full(&frag, &ctx, &mut ResultBuffer::new(0, 0));

                    // Barrera: todos listos
                    barrier.wait();

                    // Ejecutar
                    let mut buf_guard = buf.lock().unwrap();
                    hot.execute_fragment_full(&frag, &ctx, &mut buf_guard);
                })
            })
            .collect();

        for h in handles {
            h.join().unwrap();
        }

        // Recuperar buf del Mutex
        let recovered = Arc::try_unwrap(buf_arc)
            .map_err(|_| "buf_arc aún compartido")
            .unwrap()
            .into_inner()
            .unwrap();
        *buf = recovered;
    }

    /// Número de waves.
    pub fn num_waves(&self) -> usize {
        self.compute_waves().len()
    }

    /// Número máximo de sub-fragmentos en una wave (paralelismo máximo).
    pub fn max_parallelism(&self) -> usize {
        self.compute_waves()
            .iter()
            .map(|w| w.len())
            .max()
            .unwrap_or(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bml_compiler::distributed::SubFragment;
    use bml_compiler::rpn::RpnOp;

    fn make_sub(sub_id: u32, depends_on: Vec<u32>) -> SubFragment {
        SubFragment {
            fragment_id: 0,
            sub_id,
            layer_start: 0,
            layer_end: 1,
            ops: vec![RpnOp::One],
            weight_refs: vec![],
            depends_on,
        }
    }

    #[test]
    fn wave_serial_chain() {
        // A → B → C (todo serial)
        let subs = vec![
            make_sub(0, vec![]),
            make_sub(1, vec![0]),
            make_sub(2, vec![1]),
        ];
        let sched = WaveScheduler::new(subs);
        let waves = sched.compute_waves();

        assert_eq!(waves.len(), 3);
        assert_eq!(waves[0], vec![0]);
        assert_eq!(waves[1], vec![1]);
        assert_eq!(waves[2], vec![2]);
    }

    #[test]
    fn wave_parallel() {
        // A, B independientes → C depende de ambos
        let subs = vec![
            make_sub(0, vec![]),
            make_sub(1, vec![]),
            make_sub(2, vec![0, 1]),
        ];
        let sched = WaveScheduler::new(subs);
        let waves = sched.compute_waves();

        assert_eq!(waves.len(), 2);
        assert!(waves[0].contains(&0));
        assert!(waves[0].contains(&1));
        assert_eq!(waves[1], vec![2]);
    }

    #[test]
    fn wave_all_parallel() {
        // A, B, C, D todos independientes
        let subs = vec![
            make_sub(0, vec![]),
            make_sub(1, vec![]),
            make_sub(2, vec![]),
            make_sub(3, vec![]),
        ];
        let sched = WaveScheduler::new(subs);
        let waves = sched.compute_waves();

        assert_eq!(waves.len(), 1);
        assert_eq!(waves[0].len(), 4);
        assert_eq!(sched.max_parallelism(), 4);
    }

    #[test]
    fn wave_diamond() {
        // A → B, A → C, B → D, C → D (diamond)
        let subs = vec![
            make_sub(0, vec![]),
            make_sub(1, vec![0]),
            make_sub(2, vec![0]),
            make_sub(3, vec![1, 2]),
        ];
        let sched = WaveScheduler::new(subs);
        let waves = sched.compute_waves();

        assert_eq!(waves.len(), 3);
        assert_eq!(waves[0], vec![0]);
        assert_eq!(waves[1].len(), 2); // B y C en paralelo
        assert_eq!(waves[2], vec![3]);
        assert_eq!(sched.max_parallelism(), 2);
    }

    #[test]
    fn wave_num_waves() {
        let subs = vec![
            make_sub(0, vec![]),
            make_sub(1, vec![0]),
            make_sub(2, vec![0]),
            make_sub(3, vec![1, 2]),
        ];
        let sched = WaveScheduler::new(subs);
        assert_eq!(sched.num_waves(), 3);
    }
}
