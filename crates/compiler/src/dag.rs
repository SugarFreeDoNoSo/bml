//! DAG estático de nodos BML.
//!
//! Un `Dag` es un grafo acíclico dirigido de nodos BML construido sobre
//! el layout SoA de [`bml_domain::NodeSoA`]. El DAG es estático: una vez
//! construido, no se modifican sus nodos. La deduplicación de sub-árboles
//! se realiza vía [`crate::HashConsRegistry`] durante la construcción.

use bml_domain::{EvalContext, Node, NodeId, NodeKind, NodeSoA};

/// DAG estático de nodos BML.
///
/// Envuelve un [`NodeSoA`] con la raíz del grafo. Los nodos se agregan
/// a través de un [`crate::HashConsRegistry`] que deduplica sub-árboles
/// idénticos; el `Dag` en sí es de solo lectura una vez construido.
#[derive(Debug, Clone)]
pub struct Dag {
    /// Layout SoA con los nodos del grafo.
    pub soa: NodeSoA,
    /// Identificador del nodo raíz del DAG.
    pub root: NodeId,
}

impl Dag {
    /// Crea un `Dag` a partir de un `NodeSoA` y un `NodeId` raíz.
    ///
    /// El caller es responsable de que `root` sea válido dentro de `soa`.
    pub fn new(soa: NodeSoA, root: NodeId) -> Self {
        Self { soa, root }
    }

    /// Número de nodos del DAG.
    pub fn len(&self) -> usize {
        self.soa.len()
    }

    /// Returns `true` si el DAG no tiene nodos.
    pub fn is_empty(&self) -> bool {
        self.soa.is_empty()
    }

    /// Obtiene el nodo lógico en el índice dado.
    pub fn get(&self, id: NodeId) -> Node {
        self.soa.get(id)
    }

    /// Evalúa el DAG recursivamente (referencia, no hot loop).
    ///
    /// Equivalente a [`bml_domain::ast::evaluate`] pero leyendo del SoA.
    pub fn evaluate(&self, ctx: &EvalContext) -> f64 {
        evaluate_soa(&self.soa, self.root, ctx)
    }
}

/// Evalúa un nodo del SoA recursivamente.
///
/// Esta es la evaluación de referencia sobre el layout SoA. El hot loop
/// del runtime (Hito 5) usará la representación RPN en su lugar.
fn evaluate_soa(soa: &NodeSoA, id: NodeId, ctx: &EvalContext) -> f64 {
    let idx = id as usize;
    match soa.kinds[idx] {
        NodeKind::One => 1.0,
        NodeKind::Zero => 0.0,
        NodeKind::Var(var_id) => ctx.get_var(var_id),
        NodeKind::Const(const_id) => ctx.get_const(const_id),
        NodeKind::Bml => {
            let l = evaluate_soa(soa, soa.lefts[idx], ctx);
            let r = evaluate_soa(soa, soa.rights[idx], ctx);
            bml_domain::bml(l, r)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bml_domain::BMLTransformer;

    fn build_two_dag() -> Dag {
        // bml(1, 1) = 2
        let mut t = BMLTransformer::new();
        let root = t.two();
        Dag::new(t.into_soa(), root)
    }

    #[test]
    fn evaluate_two() {
        let dag = build_two_dag();
        let ctx = EvalContext::new(&[], &[]);
        assert!((dag.evaluate(&ctx) - 2.0).abs() < 1e-12);
    }

    #[test]
    fn evaluate_exp2() {
        // 2^3 = 8, donde 3 = bml(2, 2)
        let mut t = BMLTransformer::new();
        let two = t.two();
        let two2 = t.two();
        let three = t.bml(two, two2);
        let root = t.exp2(three);
        let dag = Dag::new(t.into_soa(), root);
        let ctx = EvalContext::new(&[], &[]);
        assert!((dag.evaluate(&ctx) - 8.0).abs() < 1e-9);
    }

    #[test]
    fn dag_len() {
        let dag = build_two_dag();
        // bml(1,1): 3 nodos (root + 2 ones)
        assert_eq!(dag.len(), 3);
    }
}
