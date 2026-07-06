//! Pruebas de integración: ejecutar `.bmlgraph` completo.
//!
//! Verifica que el runtime puede ejecutar un grafo BML serializado
//! y deserializado, produciendo el mismo resultado que la evaluación
//! de referencia del DAG original.

use bml_compiler::{fragment_program, linearize, BmlGraph, HashConsRegistry, DEFAULT_L1_THRESHOLD};
use bml_runtime::Runtime;

fn build_graph(n_ops: usize) -> BmlGraph {
    let mut reg = HashConsRegistry::new();
    let one = reg.one();
    let two = reg.bml(one, one);
    let mut node = two;
    let iterations = n_ops / 3;
    for _ in 0..iterations {
        node = reg.bml(node, two);
    }
    let soa = reg.into_soa();
    let program = linearize(&soa, node);
    fragment_program(&program, DEFAULT_L1_THRESHOLD)
}

#[test]
fn execute_bmlgraph_small() {
    let graph = build_graph(100);
    let mut runtime = Runtime::new(256, 16);
    let result = runtime.execute_graph(&graph, 0.0);
    let expected = graph.evaluate(0.0);
    assert_eq!(result.to_bits(), expected.to_bits());
}

#[test]
fn execute_bmlgraph_large() {
    let graph = build_graph(10_000);
    let mut runtime = Runtime::new(4096, 16);
    let result = runtime.execute_graph(&graph, 0.0);
    let expected = graph.evaluate(0.0);
    assert_eq!(result.to_bits(), expected.to_bits());
}

#[test]
fn execute_serialized_deserialized_bmlgraph() {
    let graph = build_graph(1000);
    let bytes = graph.serialize();
    let restored = BmlGraph::deserialize(&bytes, DEFAULT_L1_THRESHOLD).unwrap();
    let mut runtime = Runtime::new(256, 16);
    let result = runtime.execute_graph(&restored, 0.0);
    let expected = graph.evaluate(0.0);
    assert_eq!(result.to_bits(), expected.to_bits());
}

#[test]
fn execute_multiple_graphs_same_runtime() {
    let graph1 = build_graph(100);
    let graph2 = build_graph(200);
    let mut runtime = Runtime::new(256, 16);

    let r1 = runtime.execute_graph(&graph1, 0.0);
    let e1 = graph1.evaluate(0.0);
    assert_eq!(r1.to_bits(), e1.to_bits());

    let r2 = runtime.execute_graph(&graph2, 0.0);
    let e2 = graph2.evaluate(0.0);
    assert_eq!(r2.to_bits(), e2.to_bits());
}

#[test]
fn runtime_handles_empty_graph() {
    let graph = BmlGraph::new(DEFAULT_L1_THRESHOLD);
    let mut runtime = Runtime::new(256, 16);
    let result = runtime.execute_graph(&graph, 0.0);
    assert!(result.is_nan(), "grafo vacío debe retornar NaN");
}
