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

use bml_domain::{bml, NodeId, NodeKind, NodeSoA};
use std::collections::HashMap;

/// Clave canónica de un sub-árbol BML.
///
/// Dos sub-árboles producen la misma clave si y solo si son
/// estructuralmente idénticos (mismo `NodeKind` y mismos hijos).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum ConsKey {
    /// Constante `1`.
    One,
    /// Constante `0`.
    Zero,
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
    /// Pool de valores constantes precalculados.
    /// Cuando bml(Const(a), Const(b)) se evalúa en compile-time,
    /// el resultado se almacena aquí y se referencia como Const(id).
    const_pool: Vec<f64>,
    /// Mapa de valores constantes a su índice en el pool.
    /// Evita duplicar constantes con el mismo valor.
    const_table: HashMap<u64, bml_domain::ConstId>,
}

impl HashConsRegistry {
    /// Crea un registro vacío.
    pub fn new() -> Self {
        Self {
            soa: NodeSoA::new(),
            table: HashMap::new(),
            const_pool: Vec::new(),
            const_table: HashMap::new(),
        }
    }

    /// Crea un registro con capacidad inicial.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            soa: NodeSoA::with_capacity(capacity),
            table: HashMap::with_capacity(capacity),
            const_pool: Vec::with_capacity(capacity),
            const_table: HashMap::with_capacity(capacity),
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

    /// Obtiene o crea la constante `0` y retorna su `NodeId`.
    pub fn zero(&mut self) -> NodeId {
        match self.table.get(&ConsKey::Zero) {
            Some(&id) => id,
            None => {
                let id = self.soa.push_zero();
                self.table.insert(ConsKey::Zero, id);
                id
            }
        }
    }

    /// Obtiene o crea `BML(left, right)` y retorna su `NodeId`.
    ///
    /// Si ya existe un nodo `BML(left, right)` con los mismos `NodeId`
    /// canónicos, retorna el `NodeId` existente. Si no, lo crea.
    ///
    /// **Constant folding**: si ambos hijos son constantes (One o Const),
    /// el resultado se precalcula en compile-time y se almacena como
    /// `Const(id)` en el pool de constantes. El nodo `Bml` no se crea —
    /// se reemplaza por `Const`. Esto elimina el cómputo en runtime.
    pub fn bml(&mut self, left: NodeId, right: NodeId) -> NodeId {
        // Intentar constant folding: si ambos hijos son constantes,
        // precalcular el resultado.
        if let (Some(a), Some(b)) = (self.eval_const_node(left), self.eval_const_node(right)) {
            let result = bml(a, b);
            // Solo plegar si el resultado es finito (evitar inf/nan)
            if result.is_finite() {
                return self.const_value(result);
            }
        }

        // No se puede plegar: crear nodo Bml normal con Hash Consing
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

    /// Crea un nodo de variable `Var(var_id)` y retorna su `NodeId`.
    pub fn var(&mut self, var_id: bml_domain::VarId) -> NodeId {
        self.soa.push_var(var_id)
    }

    /// Crea o reutiliza una constante con valor `value` y retorna su `NodeId`.
    ///
    /// El valor se almacena en el pool de constantes. Si ya existe una
    /// constante con el mismo valor (comparado por bits), se reutiliza.
    pub fn const_value(&mut self, value: f64) -> NodeId {
        let bits = value.to_bits();
        if let Some(&const_id) = self.const_table.get(&bits) {
            // Ya existe un nodo Const con este valor
            // Buscar el NodeId correspondiente en el SoA
            // El const_id es el índice en el pool, el NodeId es el del nodo Const
            // Necesitamos un mapeo const_id -> NodeId
            // Por simplicidad, buscamos en el SoA
            for i in 0..self.soa.len() {
                let node = self.soa.get(i as NodeId);
                if let NodeKind::Const(cid) = node.kind {
                    if cid == const_id {
                        return i as NodeId;
                    }
                }
            }
        }
        // Crear nueva constante
        let const_id = self.const_pool.len() as bml_domain::ConstId;
        self.const_pool.push(value);
        self.const_table.insert(bits, const_id);
        let id = self.soa.push_const(const_id);
        id
    }

    /// Evalúa un nodo como constante si es posible.
    ///
    /// Retorna `Some(value)` si el nodo es `One` (valor 1.0) o `Const(id)`
    /// (valor del pool). Retorna `None` si es `Var` o `Bml`.
    fn eval_const_node(&self, id: NodeId) -> Option<f64> {
        let node = self.soa.get(id);
        match node.kind {
            NodeKind::One => Some(1.0),
            NodeKind::Zero => Some(0.0),
            NodeKind::Const(const_id) => self.const_pool.get(const_id as usize).copied(),
            NodeKind::Var(_) | NodeKind::Bml => None,
        }
    }

    /// Retorna el pool de constantes precalculadas.
    ///
    /// El runtime carga este pool al arrancar para resolver `Const(id)`.
    pub fn const_pool(&self) -> &[f64] {
        &self.const_pool
    }

    /// Consume el registro y retorna el SoA y el pool de constantes.
    pub fn into_soa_and_pool(self) -> (NodeSoA, Vec<f64>) {
        (self.soa, self.const_pool)
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

    // =====================================================================
    // Tests de constant folding
    // =====================================================================

    #[test]
    fn constant_folding_bml_one_one() {
        // bml(1, 1) = 2 debe plegarse a Const(2.0), no crear un nodo Bml.
        let mut reg = HashConsRegistry::new();
        let one = reg.one();
        let result = reg.bml(one, one);
        // El resultado debe ser un nodo Const, no Bml
        let node = reg.soa().get(result);
        assert!(
            node.is_const(),
            "bml(1,1) debe plegarse a Const, got {:?}",
            node.kind
        );
        // El valor debe ser 2.0
        let pool = reg.const_pool();
        let const_id = match node.kind {
            NodeKind::Const(id) => id,
            _ => panic!("no es Const"),
        };
        assert!((pool[const_id as usize] - 2.0).abs() < 1e-12);
    }

    #[test]
    fn constant_folding_chained() {
        // bml(bml(1,1), bml(1,1)) = bml(2, 2) = 3 debe plegarse completamente.
        let mut reg = HashConsRegistry::new();
        let one = reg.one();
        let two = reg.bml(one, one); // plegado a Const(2)
        let three = reg.bml(two, two); // plegado a Const(3)
        let node = reg.soa().get(three);
        assert!(node.is_const(), "bml(2,2) debe plegarse a Const");
        let pool = reg.const_pool();
        let const_id = match node.kind {
            NodeKind::Const(id) => id,
            _ => panic!("no es Const"),
        };
        assert!((pool[const_id as usize] - 3.0).abs() < 1e-12);
    }

    #[test]
    fn constant_folding_with_var_not_folded() {
        // bml(Var(0), Const(2)) no debe plegarse (Var no es constante).
        let mut reg = HashConsRegistry::new();
        let var = reg.var(0);
        let one = reg.one();
        let two = reg.bml(one, one); // Const(2)
        let result = reg.bml(var, two);
        // El resultado debe ser Bml, no Const
        let node = reg.soa().get(result);
        assert!(node.is_bml(), "bml(Var, Const) no debe plegarse");
    }

    #[test]
    fn constant_folding_deduplicates_constants() {
        // bml(1, 1) = 2 y bml(1, 1) = 2 deben dar el mismo Const.
        let mut reg = HashConsRegistry::new();
        let one = reg.one();
        let a = reg.bml(one, one);
        let b = reg.bml(one, one);
        assert_eq!(a, b, "constantes plegadas deben deduplicarse");
    }

    #[test]
    fn constant_folding_reduces_node_count() {
        // Sin folding: bml(bml(1,1), bml(1,1)) = 3 nodos (one, bml, bml)
        // Con folding: bml(1,1) -> Const(2), bml(Const(2), Const(2)) -> Const(3)
        // = 3 nodos (one, Const(2), Const(3)) pero 0 nodos Bml
        let mut reg = HashConsRegistry::new();
        let one = reg.one();
        let two = reg.bml(one, one);
        let _three = reg.bml(two, two);
        // Verificar que no hay nodos Bml en el SoA
        let mut bml_count = 0;
        for i in 0..reg.unique_count() {
            if reg.soa().get(i as NodeId).is_bml() {
                bml_count += 1;
            }
        }
        assert_eq!(
            bml_count, 0,
            "constant folding debe eliminar todos los Bml const-const"
        );
    }

    #[test]
    fn const_value_creates_and_reuses() {
        let mut reg = HashConsRegistry::new();
        let a = reg.const_value(1.5);
        let b = reg.const_value(1.5);
        assert_eq!(a, b, "mismo valor debe reutilizar el mismo nodo");
        let c = reg.const_value(2.5);
        assert_ne!(a, c, "distinto valor debe crear nodo distinto");
    }

    #[test]
    fn const_pool_accessible() {
        let mut reg = HashConsRegistry::new();
        reg.const_value(1.5);
        reg.const_value(2.5);
        let pool = reg.const_pool();
        assert_eq!(pool.len(), 2);
        assert!((pool[0] - 1.5).abs() < 1e-12);
        assert!((pool[1] - 2.5).abs() < 1e-12);
    }
}
