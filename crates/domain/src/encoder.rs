//! Encoder de valores reales a árboles BML.
//!
//! # Completitud funcional
//!
//! El paper (ArXiv 2603.21852v2) prueba que `{1, BML}` tiene completitud
//! funcional: cualquier real se puede construir como un árbol
//! `S → 1 | BML(S, S)`.
//!
//! En práctica, usamos `Const(id)` para almacenar el valor en el const pool
//! del compilador. El árbol sigue siendo BML válido, y el hash consing
//! deduplica constantes idénticas.
//!
//! # Compresión para modelo cuantizado
//!
//! Para Q4_0 (16 valores de peso), solo se crean 14 entradas únicas en el
//! const pool (0 y 1 son `Zero`/`One`). Los 1B pesos son referencias
//! (`Const(id)`) a esas 14 entradas → ~14 bits por peso vs 32 bits (f32).

use crate::ast::{NodeId, NodeKind, Node};

/// Encoder de reales a árboles BML.
pub struct RealEncoder {
    nodes: Vec<NodeData>,
    const_pool: std::collections::HashMap<u64, u32>,
    const_values: Vec<f64>,
}

#[derive(Debug, Clone, Copy)]
struct NodeData {
    kind: NodeKind,
    left: Option<NodeId>,
    right: Option<NodeId>,
}

impl RealEncoder {
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            const_pool: std::collections::HashMap::new(),
            const_values: Vec::new(),
        }
    }

    pub fn with_capacity(cap: usize) -> Self {
        Self {
            nodes: Vec::with_capacity(cap),
            const_pool: std::collections::HashMap::with_capacity(256),
            const_values: Vec::with_capacity(256),
        }
    }

    /// Codifica un f64 como un nodo BML.
    ///
    /// - 0.0 → `Zero` (1 nodo)
    /// - 1.0 → `One` (1 nodo)
    /// - Otro → `Const(id)` (1 nodo + entrada en const pool)
    pub fn encode_f64(&mut self, val: f64) -> NodeId {
        if val == 0.0 {
            return self.push(NodeKind::Zero, None, None);
        }
        if val == 1.0 {
            return self.push(NodeKind::One, None, None);
        }
        let bits = val.to_bits();
        if let Some(&id) = self.const_pool.get(&bits) {
            return self.push(NodeKind::Const(id), None, None);
        }
        let id = self.const_values.len() as u32;
        self.const_values.push(val);
        self.const_pool.insert(bits, id);
        self.push(NodeKind::Const(id), None, None)
    }

    /// Construye `BML(left, right)`.
    pub fn bml(&mut self, left: NodeId, right: NodeId) -> NodeId {
        self.push(NodeKind::Bml, Some(left), Some(right))
    }

    fn push(&mut self, kind: NodeKind, left: Option<NodeId>, right: Option<NodeId>) -> NodeId {
        let id = self.nodes.len() as NodeId;
        self.nodes.push(NodeData { kind, left, right });
        id
    }

    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    pub fn const_count(&self) -> usize {
        self.const_values.len()
    }

    pub fn const_values(&self) -> &[f64] {
        &self.const_values
    }

    /// Construye un Node del ast público para evaluación.
    pub fn build_node(&self, id: NodeId) -> Node {
        let nd = self.nodes[id as usize];
        Node { id, kind: nd.kind, left: nd.left, right: nd.right }
    }

    /// Construye todos los Node para evaluación.
    pub fn build_nodes(&self) -> Vec<Node> {
        (0..self.nodes.len() as NodeId)
            .map(|i| self.build_node(i))
            .collect()
    }
}

impl Default for RealEncoder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::{evaluate, EvalContext};

    fn eval(encoder: &RealEncoder, root: NodeId) -> f64 {
        let nodes = encoder.build_nodes();
        let consts = encoder.const_values();
        evaluate(&nodes, root, &EvalContext::new(&[], consts))
    }

    #[test]
    fn encode_zero() {
        let mut enc = RealEncoder::new();
        let root = enc.encode_f64(0.0);
        assert_eq!(root, 0);
        assert_eq!(enc.node_count(), 1);
        assert_eq!(enc.const_count(), 0);
        assert!(eval(&enc, root) == 0.0);
    }

    #[test]
    fn encode_one() {
        let mut enc = RealEncoder::new();
        let root = enc.encode_f64(1.0);
        assert_eq!(root, 0);
        assert_eq!(enc.node_count(), 1);
        assert_eq!(enc.const_count(), 0);
        assert!(eval(&enc, root) == 1.0);
    }

    #[test]
    fn encode_arbitrary() {
        for &v in &[0.0, 1.0, 2.0, -1.0, 0.5, 0.0078125, 3.14159, -2.71828] {
            let mut enc = RealEncoder::new();
            let root = enc.encode_f64(v);
            let val = eval(&enc, root);
            assert!((val - v).abs() < 1e-12, "encode({v}) = {val}");
        }
    }

    #[test]
    fn deduplication() {
        let mut enc = RealEncoder::new();
        let _a = enc.encode_f64(0.0078125);
        let _b = enc.encode_f64(0.0078125);
        let _c = enc.encode_f64(0.0078125);
        assert_eq!(enc.const_count(), 1);
        assert_eq!(enc.node_count(), 3);
    }

    #[test]
    fn q4_0_weights() {
        let weights: [f64; 16] = [
            -8.0, -7.0, -6.0, -5.0, -4.0, -3.0, -2.0, -1.0,
            0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0,
        ];
        let mut enc = RealEncoder::new();
        for &w in &weights {
            let _ = enc.encode_f64(w);
        }
        // 0 y 1 son Zero/One, los otros 14 son Const
        assert_eq!(enc.const_count(), 14);
        assert_eq!(enc.node_count(), 16);
    }

    #[test]
    fn bml_identity() {
        let mut enc = RealEncoder::new();
        let one = enc.encode_f64(1.0);
        let bml = enc.bml(one, one);
        let val = eval(&enc, bml);
        assert!((val - 2.0).abs() < 1e-12, "BML(1,1) = {val}");
    }
}
