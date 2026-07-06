//! Hash Consing: registro global de sub-árboles BML.
//!
//! El Hash Consing es el mecanismo crítico del compilador BML. Mantiene
//! un registro de los sub-árboles BML ya construidos y deduplica los
//! que son estructuralmente idénticos, reutilizando el mismo `NodeId`.
//!
//! # Por qué importa
//!
//! Sin Hash Consing, dos sub-árboles idénticos (ej. `bml(1, 1)` apareciendo
//! N veces) se almacenarían N veces en el SoA. Con Hash Consing, se
//! almacenan una sola vez y todas las referencias apuntan al mismo nodo.
//! Esto reduce el tamaño del DAG y permite que el tiempo de evaluación
//! de operaciones repetidas crezca sub-linealmente.
//!
//! # Implementación
//!
//! La clave de hash es estructural: combina el `NodeKind` con los
//! `NodeId` de los hijos. Para `One`, la clave es solo `One`. Para
//! `Bml(left, right)`, la clave es `(Bml, left, right)`. Como los
//! `NodeId` ya son canónicos (gracias al Hash Consing recursivo), dos
//! sub-árboles idénticos producen la misma clave.

use bml_domain::{NodeId, NodeSoA};
use std::collections::HashMap;

/// Clave canónica de un sub-árbol BML.
///
/// Dos sub-árboles producen la misma clave si y solo si son
/// estructuralmente idénticos (mismo `NodeKind` y mismos hijos).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum ConsKey {
    /// Constante `1`.
    One,
    /// `BML(left, right)` con `NodeId` canónicos de los hijos.
    Bml(NodeId, NodeId),
}

/// Registro global de sub-árboles BML con Hash Consing.
///
/// Mantiene un [`NodeSoA`] interno y un mapa `ConsKey -> NodeId`. Cada
/// llamada a [`Self::one`] o [`Self::bml`] verifica si el sub-árbol ya
/// existe; si existe, retorna el `NodeId` existente; si no, lo crea.
///
/// Esto garantiza que el SoA nunca contenga dos sub-árboles
/// estructuralmente idénticos.
pub struct HashConsRegistry {
    soa: NodeSoA,
    /// Mapa de claves canónicas a `NodeId` en el SoA.
    table: HashMap<ConsKey, NodeId>,
}

impl HashConsRegistry {
    /// Crea un registro vacío.
    pub fn new() -> Self {
        Self {
            soa: NodeSoA::new(),
            table: HashMap::new(),
        }
    }

    /// Crea un registro con capacidad inicial.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            soa: NodeSoA::with_capacity(capacity),
            table: HashMap::with_capacity(capacity),
        }
    }

    /// Acceso de solo lectura al SoA construido.
    pub fn soa(&self) -> &NodeSoA {
        &self.soa
    }

    /// Consume el registro y retorna el SoA.
    pub fn into_soa(self) -> NodeSoA {
        self.soa
    }

    /// Número de nodos únicos en el registro.
    pub fn len(&self) -> usize {
        self.soa.len()
    }

    /// Returns `true` si el registro no tiene nodos.
    pub fn is_empty(&self) -> bool {
        self.soa.is_empty()
    }

    /// Obtiene o crea la constante `1` y retorna su `NodeId`.
    ///
    /// Como todos los nodos `One` son idénticos, siempre retorna el mismo
    /// `NodeId` (típicamente `0`).
    pub fn one(&mut self) -> NodeId {
        match self.table.get(&ConsKey::One) {
            Some(&id) => id,
            None => {
                let id = self.soa.push_one();
                self.table.insert(ConsKey::One, id);
                id
            }
        }
    }

    /// Obtiene o crea `BML(left, right)` y retorna su `NodeId`.
    ///
    /// Si ya existe un nodo `BML(left, right)` con los mismos `NodeId`
    /// canónicos, retorna el `NodeId` existente. Si no, lo crea.
    pub fn bml(&mut self, left: NodeId, right: NodeId) -> NodeId {
        let key = ConsKey::Bml(left, right);
        match self.table.get(&key) {
            Some(&id) => id,
            None => {
                let id = self.soa.push_bml(left, right);
                self.table.insert(key, id);
                id
            }
        }
    }

    /// Número de deduplicaciones realizadas.
    ///
    /// Es la diferencia entre el número de llamadas a `one`/`bml` y el
    /// número de nodos únicos en el SoA. Como no rastreamos las llamadas
    /// explícitamente, esto se infiere comparando con un registro sin
    /// deduplicación. Para tests, exponemos el tamaño del SoA.
    pub fn unique_count(&self) -> usize {
        self.soa.len()
    }
}

impl Default for HashConsRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    #[test]
    fn one_is_deduplicated() {
        let mut reg = HashConsRegistry::new();
        let a = reg.one();
        let b = reg.one();
        let c = reg.one();
        assert_eq!(a, b);
        assert_eq!(b, c);
        assert_eq!(reg.unique_count(), 1);
    }

    #[test]
    fn identical_bml_subtrees_are_deduplicated() {
        let mut reg = HashConsRegistry::new();
        let one = reg.one();
        let a = reg.bml(one, one); // bml(1, 1)
        let b = reg.bml(one, one); // bml(1, 1) de nuevo
        assert_eq!(a, b);
        assert_eq!(reg.unique_count(), 2); // one + bml
    }

    #[test]
    fn distinct_subtrees_are_not_deduplicated() {
        let mut reg = HashConsRegistry::new();
        let one = reg.one();
        let two = reg.bml(one, one); // bml(1, 1) = 2
        let three = reg.bml(two, two); // bml(2, 2) = 3
        let four = reg.bml(two, one); // bml(2, 1) = 4
        assert_ne!(two, three);
        assert_ne!(two, four);
        assert_ne!(three, four);
        assert_eq!(reg.unique_count(), 4); // one, two, three, four
    }

    #[test]
    fn order_matters_for_deduplication() {
        let mut reg = HashConsRegistry::new();
        let one = reg.one();
        let two = reg.bml(one, one);
        // bml(two, one) != bml(one, two) (no conmutativo)
        let a = reg.bml(two, one);
        let b = reg.bml(one, two);
        assert_ne!(a, b);
    }

    #[test]
    fn deeply_nested_deduplication() {
        // Construir un árbol con sub-árboles repetidos
        let mut reg = HashConsRegistry::new();
        let one = reg.one();
        let two = reg.bml(one, one);
        // bml(two, two) aparece 3 veces
        let x1 = reg.bml(two, two);
        let x2 = reg.bml(two, two);
        let x3 = reg.bml(two, two);
        assert_eq!(x1, x2);
        assert_eq!(x2, x3);
        // bml(x1, x1) se deduplica
        let y1 = reg.bml(x1, x1);
        let y2 = reg.bml(x1, x1);
        assert_eq!(y1, y2);
        // Sin deduplicación serían: 1 + 1 + 3 + 2 = 7 nodos
        // Con deduplicación: 1 (one) + 1 (two) + 1 (x) + 1 (y) = 4 nodos
        assert_eq!(reg.unique_count(), 4);
    }

    /// Propiedad: dos sub-árboles idénticos siempre se deduplican.
    #[allow(unused_doc_comments)]
    proptest! {
        #[test]
        fn proptest_identical_subtrees_dedup(
            n in 0u32..100,
        ) {
            let mut reg = HashConsRegistry::new();
            let one = reg.one();
            let two = reg.bml(one, one);
            // Llamar bml(two, two) N veces siempre retorna el mismo id
            let first = reg.bml(two, two);
            for _ in 0..n {
                assert_eq!(reg.bml(two, two), first);
            }
        }
    }

    /// Propiedad: el número de nodos únicos nunca excede el número de
    /// sub-árboles distintos construidos.
    #[allow(unused_doc_comments)]
    proptest! {
        #[test]
        fn proptest_unique_count_bounded(
            n in 1u32..100,
        ) {
            let mut reg = HashConsRegistry::new();
            let one = reg.one();
            let two = reg.bml(one, one);
            // Construir bml(two, two) n veces -> solo 1 nodo único
            for _ in 0..n {
                reg.bml(two, two);
            }
            // one + two + bml(two,two) = 3 nodos únicos
            assert_eq!(reg.unique_count(), 3);
        }
    }
}
