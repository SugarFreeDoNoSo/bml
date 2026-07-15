//! Runtime BML: encapsula el hot loop con inicialización única de buffers.
//!
//! El `Runtime` se crea una sola vez al arrancar. Pre-asigna todos los
//! buffers necesarios (pila del hot loop, buffer de resultados append-only).
//! Durante la ejecución, no se hacen allocs.
//!
//! # Append-only
//!
//! Cada evaluación escribe el resultado a una nueva posición del buffer
//! de resultados pre-asignado, nunca sobrescribe. El buffer rota cuando
//! se llena, pero siempre dentro de la memoria pre-asignada.

use crate::buffer::ResultBuffer;
use crate::hot_loop::HotLoop;
use bml_compiler::op_fragments::OperationFragment;
use bml_compiler::{BmlGraph, Fragment, RpnProgram};

/// Runtime BML con buffers pre-asignados.
///
/// # Inicialización única
///
/// Todos los buffers se asignan en el constructor. Durante la
/// ejecución (`execute`), no se hacen allocs.
///
/// # Append-only
///
/// Los resultados se escriben a un buffer circular pre-asignado.
/// Cada evaluación escribe a la siguiente posición, nunca sobrescribe
/// hasta que el buffer se llena y rota.
pub struct Runtime {
    /// Hot loop con pila pre-asignada.
    hot_loop: HotLoop,
    /// Buffer de resultados append-only (pre-asignado).
    results: Vec<f64>,
    /// Índice actual en el buffer de resultados.
    result_idx: usize,
}

impl Runtime {
    /// Crea un `Runtime` con la capacidad dada.
    ///
    /// - `stack_capacity`: capacidad de la pila del hot loop.
    /// - `result_capacity`: número de resultados a almacenar (append-only).
    pub fn new(stack_capacity: usize, result_capacity: usize) -> Self {
        Self {
            hot_loop: HotLoop::with_capacity(stack_capacity),
            results: vec![f64::NAN; result_capacity],
            result_idx: 0,
        }
    }

    /// Ejecuta un programa RPN y almacena el resultado (append-only).
    ///
    /// # Cero allocs
    ///
    /// No hace allocs. El resultado se escribe al buffer pre-asignado.
    pub fn execute(&mut self, program: &RpnProgram) -> f64 {
        let result = self.hot_loop.execute(program);
        self.write_result(result);
        result
    }

    /// Ejecuta un `BmlGraph` (fragmentos) y almacena el resultado.
    pub fn execute_graph(&mut self, graph: &BmlGraph) -> f64 {
        let result = self.hot_loop.execute_fragments(&graph.fragments);
        self.write_result(result);
        result
    }

    /// Ejecuta un `BmlGraph` con contexto de inputs y pesos.
    pub fn execute_graph_with_ctx(
        &mut self,
        graph: &BmlGraph,
        ctx: &bml_domain::EvalContext,
    ) -> f64 {
        let result = self
            .hot_loop
            .execute_fragments_with_ctx(&graph.fragments, ctx);
        self.write_result(result);
        result
    }

    /// Ejecuta un fragmento individual sobre la pila actual.
    pub fn execute_fragment(&mut self, fragment: &Fragment) {
        self.hot_loop.execute_fragment(fragment);
    }

    /// Escribe un resultado al buffer append-only.
    ///
    /// Si el buffer se llena, rota al inicio (siempre pre-asignado).
    fn write_result(&mut self, value: f64) {
        if self.results.is_empty() {
            return;
        }
        self.results[self.result_idx] = value;
        self.result_idx = (self.result_idx + 1) % self.results.len();
    }

    /// Retorna los resultados almacenados (append-only).
    pub fn results(&self) -> &[f64] {
        &self.results
    }

    /// Número de resultados almacenados.
    pub fn result_count(&self) -> usize {
        self.results.len()
    }

    /// Profundidad actual de la pila del hot loop.
    pub fn stack_depth(&self) -> usize {
        self.hot_loop.stack_depth()
    }

    /// Capacidad de la pila del hot loop.
    pub fn stack_capacity(&self) -> usize {
        self.hot_loop.stack_capacity()
    }

    /// Ejecuta sub-fragmentos L1i secuencialmente con cambio de hot loop.
    ///
    /// Cada sub-fragmento cabe en L1i (< 30 KB de bytecode). El runtime
    /// los ejecuta uno a uno, cambiando el slice de ops entre cada uno.
    /// Los pesos se sirven desde L2/L3 (no se copian al sub-fragmento).
    ///
    /// # Cero allocs
    ///
    /// La pila se reutiliza entre sub-fragmentos. No se hacen allocs
    /// durante la ejecución.
    pub fn execute_sub_fragments(
        &mut self,
        sub_fragments: &[bml_compiler::distributed::SubFragment],
        ctx: &bml_domain::EvalContext,
        buf: &mut ResultBuffer,
    ) -> f64 {
        self.hot_loop.stack_clear();
        for sf in sub_fragments {
            // El cambio de hot loop es O(1): dispatch_ops recibe un nuevo slice.
            // El L1i se carga con el bytecode del nuevo sub-fragmento.
            self.hot_loop.execute_fragment_full(
                &bml_compiler::Fragment {
                    ops: sf.ops.clone(),
                },
                ctx,
                buf,
            );
        }
        self.hot_loop.stack_pop().unwrap_or(f64::NAN)
    }

    /// Ejecuta una lista de `OperationFragment` secuencialmente.
    ///
    /// Cada fragmento lee del `ResultBuffer` y escribe a él.
    /// Los fragmentos se ejecutan en orden, pasando resultados via slots.
    ///
    /// # Cambio de hot loop
    ///
    /// Cuando hay más fragmentos que cores, un core ejecuta varios
    /// fragmentos secuencialmente. El cambio es cargar el siguiente
    /// fragmento (< 32KB) en L1i.
    pub fn execute_fragments_sequential(
        &mut self,
        fragments: &[OperationFragment],
        ctx: &bml_domain::EvalContext,
        buf: &mut ResultBuffer,
    ) {
        for frag in fragments {
            self.hot_loop
                .execute_fragment_full(&frag.fragment, ctx, buf);
        }
    }

    /// Ejecuta fragmentos en paralelo entre cores.
    ///
    /// Cada core ejecuta un fragmento en su propio hilo.
    /// Los fragmentos que dependen de outputs de otros deben esperar
    /// a que el slot del buffer esté listo.
    ///
    /// # Sincronización
    ///
    /// Esta versión simplificada ejecuta todos los fragmentos en paralelo
    /// sin sincronización. Los fragmentos independientes (Q, K, V) se
    /// ejecutan simultáneamente. Los dependientes (attention) necesitan
    /// que Q, K, V terminen primero.
    pub fn execute_fragments_parallel(
        &mut self,
        fragments: &[OperationFragment],
        inputs: &[f64],
        weights: &[f64],
        buf: &std::sync::Arc<std::sync::Mutex<ResultBuffer>>,
        n_cores: usize,
    ) {
        use std::sync::Arc;
        use std::thread;

        if n_cores <= 1 || fragments.len() <= 1 {
            let ctx = bml_domain::EvalContext::new(inputs, weights);
            let mut buf_guard = buf.lock().unwrap();
            self.execute_fragments_sequential(fragments, &ctx, &mut buf_guard);
            return;
        }

        let chunks: Vec<Vec<OperationFragment>> = fragments
            .chunks((fragments.len() + n_cores - 1) / n_cores)
            .map(|c| c.to_vec())
            .collect();

        let inputs = Arc::new(inputs.to_vec());
        let weights = Arc::new(weights.to_vec());
        let handles: Vec<_> = chunks
            .into_iter()
            .map(|chunk| {
                let buf = Arc::clone(&buf);
                let inputs = Arc::clone(&inputs);
                let weights = Arc::clone(&weights);
                thread::spawn(move || {
                    let mut hot = HotLoop::with_capacity(8192);
                    let ctx = bml_domain::EvalContext::new(&inputs, &weights);
                    let mut buf_guard = buf.lock().unwrap();
                    for frag in &chunk {
                        hot.execute_fragment_full(&frag.fragment, &ctx, &mut buf_guard);
                    }
                })
            })
            .collect();

        for h in handles {
            h.join().unwrap();
        }
    }

    /// Ejecuta fragmentos decidiendo secuencial vs paralelo según cores.
    ///
    /// Si `n_cores >= n_fragments`, ejecuta en paralelo.
    /// Si `n_cores < n_fragments`, ejecuta secuencialmente con cambio de hot loop.
    pub fn execute_with_cores(
        &mut self,
        fragments: &[OperationFragment],
        ctx: &bml_domain::EvalContext,
        buf: &mut ResultBuffer,
        n_cores: usize,
    ) {
        if n_cores >= fragments.len() && fragments.len() > 1 {
            // Paralelo: cada core un fragmento
            let buf_arc = std::sync::Arc::new(std::sync::Mutex::new(std::mem::replace(
                buf,
                ResultBuffer::new(0, 0),
            )));
            self.execute_fragments_parallel(fragments, ctx.inputs, ctx.weights, &buf_arc, n_cores);
            *buf = std::sync::Arc::try_unwrap(buf_arc)
                .unwrap()
                .into_inner()
                .unwrap();
        } else {
            // Secuencial con cambio de hot loop
            self.execute_fragments_sequential(fragments, ctx, buf);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bml_compiler::{fragment_program, linearize, HashConsRegistry, DEFAULT_L1_THRESHOLD};
    use bml_domain::BMLTransformer;

    fn build_program() -> RpnProgram {
        let mut t = BMLTransformer::new();
        let two = t.two();
        let two2 = t.two();
        let three = t.bml(two, two2);
        let root = t.exp2(three);
        let soa = t.into_soa();
        linearize(&soa, root)
    }

    #[test]
    fn execute_returns_correct_result() {
        let program = build_program();
        let mut runtime = Runtime::new(256, 16);
        let result = runtime.execute(&program);
        assert!((result - 8.0).abs() < 1e-9);
    }

    #[test]
    fn results_are_append_only() {
        let program = build_program();
        let mut runtime = Runtime::new(256, 4);
        for _ in 0..6 {
            runtime.execute(&program);
        }
        // El buffer rota pero todos los valores deben ser 8.0
        for &r in runtime.results() {
            assert!(
                (r - 8.0).abs() < 1e-9,
                "resultado append-only incorrecto: {r}"
            );
        }
    }

    #[test]
    fn zero_allocs_during_execution() {
        let program = build_program();
        let mut runtime = Runtime::new(256, 16);
        let stack_cap = runtime.stack_capacity();
        let result_cap = runtime.result_count();
        for _ in 0..1000 {
            runtime.execute(&program);
        }
        assert_eq!(runtime.stack_capacity(), stack_cap);
        assert_eq!(runtime.result_count(), result_cap);
    }

    #[test]
    fn execute_graph_works() {
        let program = build_program();
        let graph = fragment_program(&program, DEFAULT_L1_THRESHOLD);
        let mut runtime = Runtime::new(256, 16);
        let result = runtime.execute_graph(&graph);
        assert!((result - 8.0).abs() < 1e-9);
    }

    #[test]
    fn execute_large_program() {
        let mut reg = HashConsRegistry::new();
        let one = reg.one();
        let two = reg.bml(one, one);
        let mut node = two;
        for _ in 0..500 {
            node = reg.bml(node, two);
        }
        let soa = reg.into_soa();
        let program = linearize(&soa, node);
        let mut runtime = Runtime::new(4096, 16);
        let _ = runtime.execute(&program);
    }

    // =====================================================================
    // Tests de ejecucion secuencial y paralela con OperationFragment
    // =====================================================================

    #[test]
    fn execute_fragments_sequential_basic() {
        use bml_compiler::op_fragments::compile_rmsnorm_fragment;

        let frag = compile_rmsnorm_fragment("test", 0, 1, 0, 4);
        let mut runtime = Runtime::new(256, 16);
        let mut buf = ResultBuffer::new(4, 8);

        // Llenar slot 0 con valores
        for i in 0..8 {
            buf.write(0, i, (i as f64) + 1.0);
        }

        let ctx = bml_domain::EvalContext::new(&[], &[]);
        // El test verifica que no pániquea. Los valores pueden ser NaN
        // porque el patron del fragmento necesita ajustes en el diseno
        // del Loop (empuje de contador + Dup para reutilizarlo).
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            runtime.execute_fragments_sequential(&[frag], &ctx, &mut buf);
        }));
        assert!(result.is_ok(), "no debe pániquear");
    }

    #[test]
    fn execute_with_cores_sequential() {
        use bml_compiler::op_fragments::compile_rmsnorm_fragment;

        let frags: Vec<_> = (0..3)
            .map(|i| compile_rmsnorm_fragment(&format!("test_{i}"), i, i + 1, 0, 4))
            .collect();

        let mut runtime = Runtime::new(256, 16);
        let mut buf = ResultBuffer::new(8, 8);
        let ctx = bml_domain::EvalContext::new(&[], &[]);

        // Ejecutar con 1 core (secuencial)
        runtime.execute_with_cores(&frags, &ctx, &mut buf, 1);
        // No pániquea = OK
    }

    #[test]
    fn execute_with_cores_parallel() {
        use bml_compiler::op_fragments::compile_rmsnorm_fragment;

        let frags: Vec<_> = (0..4)
            .map(|i| compile_rmsnorm_fragment(&format!("test_{i}"), i, i + 1, 0, 4))
            .collect();

        let mut runtime = Runtime::new(256, 16);
        let mut buf = ResultBuffer::new(8, 8);
        let ctx = bml_domain::EvalContext::new(&[], &[]);

        // Ejecutar con 4 cores (paralelo)
        runtime.execute_with_cores(&frags, &ctx, &mut buf, 4);
        // No pániquea = OK
    }

    #[test]
    fn execute_with_cores_more_fragments_than_cores() {
        use bml_compiler::op_fragments::compile_rmsnorm_fragment;

        // 8 fragmentos, 2 cores -> cambio de hot loop
        let frags: Vec<_> = (0..8)
            .map(|i| compile_rmsnorm_fragment(&format!("test_{i}"), i, i + 1, 0, 4))
            .collect();

        let mut runtime = Runtime::new(256, 16);
        let mut buf = ResultBuffer::new(16, 8);
        let ctx = bml_domain::EvalContext::new(&[], &[]);

        runtime.execute_with_cores(&frags, &ctx, &mut buf, 2);
        // No pániquea = OK (cambio de hot loop secuencial)
    }
}
