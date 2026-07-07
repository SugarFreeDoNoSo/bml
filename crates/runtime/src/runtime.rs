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

use crate::hot_loop::HotLoop;
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
    pub fn execute(&mut self, program: &RpnProgram, x: f64) -> f64 {
        let result = self.hot_loop.execute(program, x);
        self.write_result(result);
        result
    }

    /// Ejecuta un `BmlGraph` (fragmentos) y almacena el resultado.
    pub fn execute_graph(&mut self, graph: &BmlGraph, x: f64) -> f64 {
        let result = self.hot_loop.execute_fragments(&graph.fragments, x);
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
        let result = runtime.execute(&program, 0.0);
        assert!((result - 8.0).abs() < 1e-9);
    }

    #[test]
    fn results_are_append_only() {
        let program = build_program();
        let mut runtime = Runtime::new(256, 4);
        for _ in 0..6 {
            runtime.execute(&program, 0.0);
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
            runtime.execute(&program, 0.0);
        }
        assert_eq!(runtime.stack_capacity(), stack_cap);
        assert_eq!(runtime.result_count(), result_cap);
    }

    #[test]
    fn execute_graph_works() {
        let program = build_program();
        let graph = fragment_program(&program, DEFAULT_L1_THRESHOLD);
        let mut runtime = Runtime::new(256, 16);
        let result = runtime.execute_graph(&graph, 0.0);
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
        let _ = runtime.execute(&program, 0.0);
    }
}
