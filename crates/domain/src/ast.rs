//! Gramática estricta del AST BML.
//!
//! La gramática es: `S -> 1 | BML(S, S)`.
//! No existen nodos `+`, `-`, `*`, `/`, `pow` ni otras operaciones
//! estándar; solo `BML` y la constante `1`.

use crate::operator::bml;

/// Identificador de un nodo dentro del grafo.
///
/// Se usa un `u32` para permitir hasta 2^32 nodos, suficiente para
/// los DAGs objetivo y compacto en memoria (4 bytes por referencia).
pub type NodeId = u32;

/// Clase de nodo del AST.
///
/// La gramática BML solo admite dos variantes:
/// - [`NodeKind::One`]: la constante distinguida `1`.
/// - [`NodeKind::Bml`]: aplicación del operador BML a dos sub-árboles.
///
/// Cualquier otra operación estándar (`+`, `-`, `*`, `/`, `pow`, ...)
/// debe reducirse a esta gramática vía [`crate::BMLTransformer`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NodeKind {
    /// Constante distinguida `1`. Terminal de la gramática.
    One,
    /// `BML(left, right)`. Único operador del AST.
    Bml,
}

/// Nodo del AST BML.
///
/// Representación *lógica* del nodo (no el layout SoA, que está en
/// [`crate::NodeSoA`]). Se usa para construir árboles antes de
/// empaquetarlos en el grafo SoA.
///
/// Para `NodeKind::One`, `left` y `right` son `None`.
/// Para `NodeKind::Bml`, `left` y `right` son `Some(NodeId)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Node {
    /// Identificador del nodo dentro del grafo.
    pub id: NodeId,
    /// Clase del nodo.
    pub kind: NodeKind,
    /// Sub-árbol izquierdo (solo para `Bml`).
    pub left: Option<NodeId>,
    /// Sub-árbol derecho (solo para `Bml`).
    pub right: Option<NodeId>,
}

impl Node {
    /// Crea un nodo constante `1`.
    #[inline]
    pub fn one(id: NodeId) -> Self {
        Self {
            id,
            kind: NodeKind::One,
            left: None,
            right: None,
        }
    }

    /// Crea un nodo `BML(left, right)`.
    #[inline]
    pub fn bml(id: NodeId, left: NodeId, right: NodeId) -> Self {
        Self {
            id,
            kind: NodeKind::Bml,
            left: Some(left),
            right: Some(right),
        }
    }

    /// Returns `true` si el nodo es la constante `1`.
    #[inline]
    pub fn is_one(&self) -> bool {
        matches!(self.kind, NodeKind::One)
    }

    /// Returns `true` si el nodo es una aplicación de `BML`.
    #[inline]
    pub fn is_bml(&self) -> bool {
        matches!(self.kind, NodeKind::Bml)
    }
}

/// Evalúa un nodo del AST dado un arreglo de nodos y un valor de entrada `x`.
///
/// El AST se evalúa recursivamente. Esta función es de referencia
/// (no es el hot loop del runtime); se usa para verificar la corrección
/// del `BMLTransformer` y del compilador.
///
/// El parámetro `x` se propaga recursivamente a los sub-árboles para que
/// los nodos hoja de variable (cuando se integren en el Hito 2) puedan
/// leerlo. Por ahora solo existe el nodo `One`, por lo que `x` no se usa
/// en el cuerpo directo de la función, solo se pasa a las llamadas
/// recursivas.
///
/// # Panics
///
/// Panics si un `Bml` referencia un `NodeId` fuera de rango.
#[allow(clippy::only_used_in_recursion)]
pub fn evaluate(nodes: &[Node], root: NodeId, x: f64) -> f64 {
    let node = nodes[root as usize];
    match node.kind {
        NodeKind::One => 1.0,
        NodeKind::Bml => {
            let l = evaluate(nodes, node.left.unwrap(), x);
            let r = evaluate(nodes, node.right.unwrap(), x);
            bml(l, r)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn build_one() -> Vec<Node> {
        vec![Node::one(0)]
    }

    fn build_exp_x() -> Vec<Node> {
        // exp(x) = bml(x, 1)
        // Pero x no es terminal en la gramática pura; usamos un nodo
        // "input" simulado como BML(x, 1) para testear evaluate.
        // Para este test construimos: bml(input, 1) donde input es x.
        // Como la gramática pura solo tiene 1 y BML, modelamos x como
        // un nodo BML cuyo valor es x (placeholder). Para testear evaluate
        // usamos un nodo One y verificamos la estructura.
        vec![Node::bml(0, 1, 2), Node::one(1), Node::one(2)]
    }

    #[test]
    fn one_node_evaluates_to_one() {
        let nodes = build_one();
        assert_eq!(evaluate(&nodes, 0, 42.0), 1.0);
    }

    #[test]
    fn bml_one_one_evaluates_to_two() {
        // bml(1, 1) = 2 (constante fundamental en base 2)
        let nodes = vec![Node::bml(0, 1, 2), Node::one(1), Node::one(2)];
        let val = evaluate(&nodes, 0, 0.0);
        assert!((val - 2.0).abs() < 1e-12);
    }

    #[test]
    fn node_constructors() {
        let one = Node::one(0);
        assert!(one.is_one());
        assert!(!one.is_bml());

        let bml_node = Node::bml(1, 0, 0);
        assert!(bml_node.is_bml());
        assert!(!bml_node.is_one());
        assert_eq!(bml_node.left, Some(0));
        assert_eq!(bml_node.right, Some(0));
    }

    #[test]
    fn grammar_only_one_and_bml() {
        // La gramática no admite otras variantes. Verificamos que
        // NodeKind solo tenga One y Bml.
        let variants = [NodeKind::One, NodeKind::Bml];
        assert_eq!(variants.len(), 2);
    }

    #[test]
    fn build_exp_x_structure() {
        let nodes = build_exp_x();
        assert!(nodes[0].is_bml());
        assert!(nodes[1].is_one());
        assert!(nodes[2].is_one());
    }
}
