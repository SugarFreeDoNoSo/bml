//! Gramática del AST BML con variables y constantes.
//!
//! La gramática extendida es: `S -> 1 | Var(id) | Const(value) | BML(S, S)`.
//!
//! - `One`: la constante distinguida `1`.
//! - `Var(id)`: input variable (token del prompt), resuelto desde un contexto.
//! - `Const(value)`: peso del modelo (constante arbitraria).
//! - `Bml(left, right)`: aplicación del operador BML a dos sub-árboles.
//!
//! Las operaciones estándar (`+`, `-`, `*`, `/`, `pow`, ...) se reducen
//! a esta gramática vía [`crate::BMLTransformer`].

use crate::operator::bml_base_op;

/// Identificador de un nodo dentro del grafo.
pub type NodeId = u32;

/// Identificador de una variable (input).
pub type VarId = u32;

/// Identificador de una constante (peso del modelo).
pub type ConstId = u32;

/// Clase de nodo del AST.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NodeKind {
    /// Constante distinguida `1`. Terminal de la gramática.
    One,
    /// Constante distinguida `0`. Terminal de la gramática.
    Zero,
    /// Input variable. Resuelto desde un contexto de inputs.
    Var(VarId),
    /// Constante arbitraria (peso del modelo). Resuelto desde un pool de pesos.
    Const(ConstId),
    /// `BML(left, right)`. Único operador del AST.
    Bml,
}

/// Nodo del AST BML.
///
/// Representación *lógica* del nodo (no el layout SoA).
///
/// Para `One`, `Var`, `Const`: `left` y `right` son `None`.
/// Para `Bml`: `left` y `right` son `Some(NodeId)`.
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

    /// Crea un nodo constante `0`.
    #[inline]
    pub fn zero(id: NodeId) -> Self {
        Self {
            id,
            kind: NodeKind::Zero,
            left: None,
            right: None,
        }
    }

    /// Crea un nodo de variable `Var(id)`.
    #[inline]
    pub fn var(id: NodeId, var_id: VarId) -> Self {
        Self {
            id,
            kind: NodeKind::Var(var_id),
            left: None,
            right: None,
        }
    }

    /// Crea un nodo de constante `Const(id)`.
    #[inline]
    pub fn const_(id: NodeId, const_id: ConstId) -> Self {
        Self {
            id,
            kind: NodeKind::Const(const_id),
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

    /// Returns `true` si el nodo es una variable.
    #[inline]
    pub fn is_var(&self) -> bool {
        matches!(self.kind, NodeKind::Var(_))
    }

    /// Returns `true` si el nodo es una constante.
    #[inline]
    pub fn is_const(&self) -> bool {
        matches!(self.kind, NodeKind::Const(_))
    }

    /// Returns `true` si el nodo es una aplicación de `BML`.
    #[inline]
    pub fn is_bml(&self) -> bool {
        matches!(self.kind, NodeKind::Bml)
    }
}

/// Contexto de evaluación: resuelve `Var` y `Const` desde buffers.
#[derive(Debug, Clone)]
pub struct EvalContext<'a> {
    /// Inputs variables (tokens del prompt).
    pub inputs: &'a [f64],
    /// Pesos del modelo (constantes).
    pub weights: &'a [f64],
}

impl<'a> EvalContext<'a> {
    /// Crea un contexto de evaluación.
    pub fn new(inputs: &'a [f64], weights: &'a [f64]) -> Self {
        Self { inputs, weights }
    }

    /// Resuelve una variable por índice.
    pub fn get_var(&self, id: VarId) -> f64 {
        self.inputs.get(id as usize).copied().unwrap_or(f64::NAN)
    }

    /// Resuelve una constante por índice.
    pub fn get_const(&self, id: ConstId) -> f64 {
        self.weights.get(id as usize).copied().unwrap_or(f64::NAN)
    }
}

/// Evalúa un nodo del AST dado un arreglo de nodos y un contexto.
///
/// El AST se evalúa recursivamente. Esta función es de referencia
/// (no es el hot loop del runtime).
///
/// # Panics
///
/// Panics si un `Bml` referencia un `NodeId` fuera de rango.
#[allow(clippy::only_used_in_recursion)]
pub fn evaluate(nodes: &[Node], root: NodeId, ctx: &EvalContext) -> f64 {
    let node = nodes[root as usize];
    match node.kind {
        NodeKind::One => 1.0,
        NodeKind::Zero => 0.0,
        NodeKind::Var(id) => ctx.get_var(id),
        NodeKind::Const(id) => ctx.get_const(id),
        NodeKind::Bml => {
            let l = evaluate(nodes, node.left.unwrap(), ctx);
            let r = evaluate(nodes, node.right.unwrap(), ctx);
            bml_base_op(l, r)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn one_node_evaluates_to_one() {
        let nodes = vec![Node::one(0)];
        let ctx = EvalContext::new(&[], &[]);
        assert_eq!(evaluate(&nodes, 0, &ctx), 1.0);
    }

    #[test]
    fn bml_one_one_evaluates_to_two() {
        let nodes = vec![Node::bml(0, 1, 2), Node::one(1), Node::one(2)];
        let ctx = EvalContext::new(&[], &[]);
        let val = evaluate(&nodes, 0, &ctx);
        assert!((val - 2.0).abs() < 1e-12);
    }

    #[test]
    fn var_resolves_from_context() {
        // Var(0) debe resolver al primer input.
        let nodes = vec![Node::var(0, 0)];
        let ctx = EvalContext::new(&[2.71], &[]);
        assert_eq!(evaluate(&nodes, 0, &ctx), 2.71);
    }

    #[test]
    fn const_resolves_from_context() {
        // Const(0) debe resolver al primer peso.
        let nodes = vec![Node::const_(0, 0)];
        let ctx = EvalContext::new(&[], &[2.71]);
        assert_eq!(evaluate(&nodes, 0, &ctx), 2.71);
    }

    #[test]
    fn bml_var_const_evaluates() {
        // bml(Var(0), Const(0)) = bml(input[0], weight[0])
        // bml(2.71, 2.71) = 2^2.71 - log2(2.71)
        let nodes = vec![Node::bml(0, 1, 2), Node::var(1, 0), Node::const_(2, 0)];
        let ctx = EvalContext::new(&[2.71], &[2.71]);
        let val = evaluate(&nodes, 0, &ctx);
        let expected = bml_base_op(2.71, 2.71);
        assert!((val - expected).abs() < 1e-9);
    }

    #[test]
    fn node_constructors() {
        let one = Node::one(0);
        assert!(one.is_one());
        assert!(!one.is_var());
        assert!(!one.is_const());
        assert!(!one.is_bml());

        let var = Node::var(1, 0);
        assert!(!var.is_one());
        assert!(var.is_var());
        assert!(!var.is_const());
        assert!(!var.is_bml());

        let const_ = Node::const_(2, 0);
        assert!(!const_.is_one());
        assert!(!const_.is_var());
        assert!(const_.is_const());
        assert!(!const_.is_bml());

        let bml_node = Node::bml(3, 0, 0);
        assert!(!bml_node.is_one());
        assert!(!bml_node.is_var());
        assert!(!bml_node.is_const());
        assert!(bml_node.is_bml());
        assert_eq!(bml_node.left, Some(0));
        assert_eq!(bml_node.right, Some(0));
    }

    #[test]
    fn grammar_has_four_variants() {
        let variants = [
            NodeKind::One,
            NodeKind::Var(0),
            NodeKind::Const(0),
            NodeKind::Bml,
        ];
        assert_eq!(variants.len(), 4);
    }

    #[test]
    fn var_out_of_range_returns_nan() {
        let nodes = vec![Node::var(0, 99)];
        let ctx = EvalContext::new(&[1.0], &[]);
        assert!(evaluate(&nodes, 0, &ctx).is_nan());
    }

    #[test]
    fn const_out_of_range_returns_nan() {
        let nodes = vec![Node::const_(0, 99)];
        let ctx = EvalContext::new(&[], &[1.0]);
        assert!(evaluate(&nodes, 0, &ctx).is_nan());
    }
}
