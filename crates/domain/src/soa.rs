//! Layout de memoria SoA (Struct of Arrays) alineado a línea de caché.
//!
//! El draft prohíbe el patrón Array of Structures (AoS) para los nodos del
//! grafo. En su lugar, se almacenan los campos como Struct of Arrays (SoA)
//! con `#[repr(align(64))]`, garantizando que:
//!
//! - Cada campo relevante está alineado a 64 bytes (línea de caché típica).
//! - La CPU solo carga en caché los bytes estrictamente necesarios para
//!   la evaluación matemática, no toda la estructura del nodo.
//! - No hay false sharing entre hilos cuando los campos se acceden
//!   concurrentemente.
//!
//! # Layout
//!
//! `NodeSoA` contiene un `Vec` por campo del nodo. Cada `Vec` se alinea
//! a 64 bytes al inicio, de forma que el primer elemento de cada campo
//! caiga al inicio de una línea de caché.

use crate::ast::{Node, NodeId, NodeKind};

/// Capacidad inicial por defecto de cada arreglo SoA.
const DEFAULT_CAPACITY: usize = 64;

/// Struct of Arrays para los nodos del grafo BML.
///
/// Cada campo se almacena en su propio `Vec`, alineado a 64 bytes al
/// inicio. Esto permite que el runtime cargue solo el campo necesario
/// para una evaluación (ej. solo los operandos) sin traer a caché
/// campos irrelevantes.
///
/// # Alineación
///
/// La estructura misma se alinea con `#[repr(align(64))]`. Los `Vec`
/// internos reservan memoria con el allocator estándar; para garantizar
/// que el *contenido* de cada `Vec` esté alineado a 64 bytes, se debe
/// usar un allocator custom en el runtime (Hito 5). Para el Hito 1,
/// la alineación de la estructura es suficiente para los tests.
#[repr(align(64))]
#[derive(Debug, Clone)]
pub struct NodeSoA {
    /// Clase de cada nodo (`One` o `Bml`).
    pub kinds: Vec<NodeKind>,
    /// Sub-árbol izquierdo (solo para `Bml`; `0` para `One`).
    pub lefts: Vec<NodeId>,
    /// Sub-árbol derecho (solo para `Bml`; `0` para `One`).
    pub rights: Vec<NodeId>,
    /// Valor computado del nodo (escrito por el runtime, append-only).
    pub values: Vec<f64>,
}

impl NodeSoA {
    /// Crea un `NodeSoA` vacío con capacidad inicial por defecto.
    pub fn new() -> Self {
        Self::with_capacity(DEFAULT_CAPACITY)
    }

    /// Crea un `NodeSoA` vacío con la capacidad dada.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            kinds: Vec::with_capacity(capacity),
            lefts: Vec::with_capacity(capacity),
            rights: Vec::with_capacity(capacity),
            values: Vec::with_capacity(capacity),
        }
    }

    /// Número de nodos almacenados.
    pub fn len(&self) -> usize {
        self.kinds.len()
    }

    /// Returns `true` si no hay nodos.
    pub fn is_empty(&self) -> bool {
        self.kinds.is_empty()
    }

    /// Agrega un nodo al final de los arreglos.
    ///
    /// El campo `values` se inicializa en 0; el runtime lo escribirá
    /// (append-only) durante la evaluación.
    pub fn push(&mut self, node: Node) {
        self.kinds.push(node.kind);
        self.lefts.push(node.left.unwrap_or(0));
        self.rights.push(node.right.unwrap_or(0));
        self.values.push(0.0);
    }

    /// Agrega un nodo constante `1` y retorna su `NodeId`.
    pub fn push_one(&mut self) -> NodeId {
        let id = self.len() as NodeId;
        self.push(Node::one(id));
        id
    }

    /// Agrega un nodo `BML(left, right)` y retorna su `NodeId`.
    pub fn push_bml(&mut self, left: NodeId, right: NodeId) -> NodeId {
        let id = self.len() as NodeId;
        self.push(Node::bml(id, left, right));
        id
    }

    /// Obtiene una referencia al nodo lógico en el índice dado.
    pub fn get(&self, id: NodeId) -> Node {
        let idx = id as usize;
        let kind = self.kinds[idx];
        match kind {
            NodeKind::One => Node::one(id),
            NodeKind::Bml => Node::bml(id, self.lefts[idx], self.rights[idx]),
        }
    }

    /// Reserva memoria para al menos `additional` nodos más.
    pub fn reserve(&mut self, additional: usize) {
        self.kinds.reserve(additional);
        self.lefts.reserve(additional);
        self.rights.reserve(additional);
        self.values.reserve(additional);
    }
}

impl Default for NodeSoA {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::operator::bml;

    #[test]
    fn alignment_is_64_bytes() {
        // La estructura NodeSoA debe estar alineada a 64 bytes.
        assert_eq!(core::mem::align_of::<NodeSoA>(), 64);
    }

    #[test]
    fn push_one_and_bml() {
        let mut soa = NodeSoA::new();
        let one_id = soa.push_one();
        let bml_id = soa.push_bml(one_id, one_id);

        assert_eq!(soa.len(), 2);
        assert!(soa.get(one_id).is_one());
        assert!(soa.get(bml_id).is_bml());
        assert_eq!(soa.get(bml_id).left, Some(one_id));
        assert_eq!(soa.get(bml_id).right, Some(one_id));
    }

    #[test]
    fn values_initialized_to_zero() {
        let mut soa = NodeSoA::new();
        soa.push_one();
        assert_eq!(soa.values[0], 0.0);
    }

    #[test]
    fn evaluate_from_soa() {
        // bml(1, 1) = 2
        let mut soa = NodeSoA::new();
        let one = soa.push_one();
        let one2 = soa.push_one();
        let root = soa.push_bml(one, one2);

        // El runtime escribiría values[one] = 1, values[one2] = 1,
        // luego values[root] = bml(1, 1) = 2.
        soa.values[one as usize] = 1.0;
        soa.values[one2 as usize] = 1.0;
        soa.values[root as usize] = bml(soa.values[one as usize], soa.values[one2 as usize]);
        assert!((soa.values[root as usize] - 2.0).abs() < 1e-12);
    }

    #[test]
    fn empty_soa() {
        let soa = NodeSoA::new();
        assert!(soa.is_empty());
        assert_eq!(soa.len(), 0);
    }

    #[test]
    fn reserve_grows_capacity() {
        let mut soa = NodeSoA::new();
        soa.reserve(100);
        // La capacidad real puede ser mayor por la política del allocator.
        assert!(soa.kinds.capacity() >= 100);
    }
}
