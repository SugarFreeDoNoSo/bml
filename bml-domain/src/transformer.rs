//! `BMLTransformer`: traduce operaciones estándar a la gramática BML.
//!
//! El transformer toma operaciones estándar (`+`, `-`, `*`, `/`, `pow`,
//! `exp2`, `log2`) y las traduce puramente a la gramática recursiva BML
//! `S -> 1 | BML(S, S)` usando solo el operador [`crate::bml`] y la
//! constante [`crate::operator::ONE`].
//!
//! # Identidades conocidas (base 2)
//!
//! - `2 = bml(1, 1)` (constante fundamental)
//! - `2^x = bml(x, 1)` (exponencial en base 2)
//! - `log2(x) = bml(1, bml(bml(1, x), 1))` (logaritmo en base 2)
//!
//! # Operaciones pendientes
//!
//! Las fórmulas exactas para `+`, `-`, `*`, `/`, `pow` en base 2 no
//! están en el paper fuente (que usa base E con profundidades 27, 83,
//! 41, 105, 49 respectivamente). Se derivarán del Supplementary
//! Information del paper o por búsqueda directa en Hito 2. Por ahora,
//! el transformer expone `exp2` y `log2` que sí tenemos verificadas.

use crate::ast::NodeId;
use crate::soa::NodeSoA;

/// Traductor de operaciones estándar a la gramática BML.
///
/// Mantiene un [`NodeSoA`] interno donde va agregando los nodos generados.
/// Cada método retorna el `NodeId` de la raíz del sub-árbol generado.
pub struct BMLTransformer {
    soa: NodeSoA,
}

impl BMLTransformer {
    /// Crea un transformer vacío.
    pub fn new() -> Self {
        Self {
            soa: NodeSoA::new(),
        }
    }

    /// Crea un transformer con capacidad inicial.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            soa: NodeSoA::with_capacity(capacity),
        }
    }

    /// Acceso de solo lectura al grafo SoA construido.
    pub fn soa(&self) -> &NodeSoA {
        &self.soa
    }

    /// Consume el transformer y retorna el grafo SoA.
    pub fn into_soa(self) -> NodeSoA {
        self.soa
    }

    // =====================================================================
    // Primitivas BML
    // =====================================================================

    /// Agrega la constante `1` al grafo y retorna su `NodeId`.
    pub fn one(&mut self) -> NodeId {
        self.soa.push_one()
    }

    /// Agrega `BML(left, right)` al grafo y retorna su `NodeId`.
    pub fn bml(&mut self, left: NodeId, right: NodeId) -> NodeId {
        self.soa.push_bml(left, right)
    }

    // =====================================================================
    // Identidades verificadas (base 2)
    // =====================================================================

    /// `2 = bml(1, 1)`.
    ///
    /// La constante fundamental en base 2 (análogo de `e = eml(1, 1)`).
    pub fn two(&mut self) -> NodeId {
        let one = self.one();
        let one2 = self.one();
        self.bml(one, one2)
    }

    /// `2^x = bml(x, 1)`.
    ///
    /// Exponencial en base 2. Aquí `x` es un `NodeId` que debe evaluar
    /// al exponente deseado.
    pub fn exp2(&mut self, x: NodeId) -> NodeId {
        let one = self.one();
        self.bml(x, one)
    }

    /// `log2(x) = bml(1, bml(bml(1, x), 1))`.
    ///
    /// Logaritmo en base 2. Verificación:
    /// - `bml(1, x) = 2 - log2(x)`
    /// - `bml(bml(1, x), 1) = 2^(2 - log2(x)) = 4/x` (en potencias de 2)
    /// - `bml(1, 4/x) = 2 - log2(4/x) = 2 - (2 - log2(x)) = log2(x)`
    pub fn log2(&mut self, x: NodeId) -> NodeId {
        let one = self.one();
        let one2 = self.one();
        let one3 = self.one();
        let inner = self.bml(one, x); // 2 - log2(x)
        let inner2 = self.bml(inner, one2); // 2^(2 - log2(x)) = 4/x
        self.bml(one3, inner2) // 2 - log2(4/x) = log2(x)
    }

    // =====================================================================
    // Operaciones estándar (TODO: derivar del Supplementary Information)
    // =====================================================================

    /// `x + y` en base BML.
    ///
    /// TODO: La fórmula exacta en base 2 no está en el paper fuente.
    /// El paper reporta profundidad 27 (direct search: 19) para `x + y`
    /// en base E. Se derivará en Hito 2.
    pub fn add(&mut self, _x: NodeId, _y: NodeId) -> NodeId {
        unimplemented!("BMLTransformer::add: fórmula pendiente de derivación (Hito 2)")
    }

    /// `x - y` en base BML.
    ///
    /// TODO: Profundidad 83 (direct search: 11) en base E. Pendiente.
    pub fn sub(&mut self, _x: NodeId, _y: NodeId) -> NodeId {
        unimplemented!("BMLTransformer::sub: fórmula pendiente de derivación (Hito 2)")
    }

    /// `x * y` en base BML.
    ///
    /// TODO: Profundidad 41 (direct search: 17) en base E. Pendiente.
    pub fn mul(&mut self, _x: NodeId, _y: NodeId) -> NodeId {
        unimplemented!("BMLTransformer::mul: fórmula pendiente de derivación (Hito 2)")
    }

    /// `x / y` en base BML.
    ///
    /// TODO: Profundidad 105 (direct search: 17) en base E. Pendiente.
    pub fn div(&mut self, _x: NodeId, _y: NodeId) -> NodeId {
        unimplemented!("BMLTransformer::div: fórmula pendiente de derivación (Hito 2)")
    }

    /// `x^y` en base BML.
    ///
    /// TODO: Profundidad 49 (direct search: 25) en base E. Pendiente.
    pub fn pow(&mut self, _x: NodeId, _y: NodeId) -> NodeId {
        unimplemented!("BMLTransformer::pow: fórmula pendiente de derivación (Hito 2)")
    }
}

impl Default for BMLTransformer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::evaluate;
    use crate::ast::Node;
    use crate::operator::ONE;

    #[test]
    fn two_equals_2() {
        let mut t = BMLTransformer::new();
        let root = t.two();
        let soa = t.into_soa();
        let nodes: Vec<Node> = (0..soa.len() as NodeId).map(|i| soa.get(i)).collect();
        assert!((evaluate(&nodes, root, 0.0) - 2.0).abs() < 1e-12);
    }

    #[test]
    fn exp2_of_3_is_8() {
        // 2^3 = 8
        // Para obtener 3, usamos bml(2, 2) = 2^2 - log2(2) = 4 - 1 = 3.
        let mut t = BMLTransformer::new();
        let two = t.two(); // 2
        let two2 = t.two(); // 2
        let three = t.bml(two, two2); // bml(2, 2) = 3
        let exp = t.exp2(three); // 2^3 = 8
        let soa = t.into_soa();
        let nodes: Vec<Node> = (0..soa.len() as NodeId).map(|i| soa.get(i)).collect();
        assert!((evaluate(&nodes, exp, 0.0) - 8.0).abs() < 1e-9);
    }

    #[test]
    fn log2_of_8_is_3() {
        // log2(8) = 3 (con f64 la identidad funciona para cualquier x > 0)
        let mut t = BMLTransformer::new();
        let two = t.two();
        let two2 = t.two();
        let three = t.bml(two, two2); // 3
        let eight = t.exp2(three); // 8
        let log = t.log2(eight); // log2(8) = 3
        let soa = t.into_soa();
        let nodes: Vec<Node> = (0..soa.len() as NodeId).map(|i| soa.get(i)).collect();
        assert!((evaluate(&nodes, log, 0.0) - 3.0).abs() < 1e-9);
    }

    #[test]
    fn log2_of_power_of_2() {
        // log2(2^k) = k. Construimos 2^k como exp2(k_node), donde k_node
        // se obtiene manualmente para k pequenos (sin `add` disponible):
        //   k=1: 1
        //   k=2: 2 = bml(1,1)
        //   k=3: 3 = bml(2, 2) = 2^2 - log2(2) = 4 - 1 = 3
        //   k=4: 4 = bml(2, 1) = 2^2 - 0 = 4
        type Case = (f64, Box<dyn Fn(&mut BMLTransformer) -> NodeId>);
        let cases: Vec<Case> = vec![
            (1.0, Box::new(|t| t.one())),
            (2.0, Box::new(|t| t.two())),
            (
                3.0,
                Box::new(|t| {
                    let two = t.two();
                    let two2 = t.two();
                    t.bml(two, two2)
                }),
            ),
            (
                4.0,
                Box::new(|t| {
                    let two = t.two();
                    t.exp2(two)
                }),
            ),
        ];
        for (k, build_k) in cases {
            let mut t = BMLTransformer::new();
            let k_node = build_k(&mut t);
            let pow = t.exp2(k_node); // 2^k
            let log = t.log2(pow); // log2(2^k) = k
            let soa = t.into_soa();
            let nodes: Vec<Node> = (0..soa.len() as NodeId).map(|i| soa.get(i)).collect();
            let result = evaluate(&nodes, log, 0.0);
            assert!(
                (result - k).abs() < 1e-9,
                "log2(2^{k}) = {result}, expected {k}"
            );
        }
    }

    #[test]
    fn transformer_preserves_grammar() {
        // Verificamos que todos los nodos generados son One o Bml
        let mut t = BMLTransformer::new();
        let _ = t.two();
        let one = t.one();
        let _ = t.exp2(one);
        let one2 = t.one();
        let _ = t.log2(one2);
        let soa = t.into_soa();
        for i in 0..soa.len() as NodeId {
            let node = soa.get(i);
            assert!(node.is_one() || node.is_bml(), "nodo {i} no es One ni Bml");
        }
    }

    #[test]
    fn one_evaluates_to_one() {
        let mut t = BMLTransformer::new();
        let one = t.one();
        let soa = t.into_soa();
        let nodes: Vec<Node> = (0..soa.len() as NodeId).map(|i| soa.get(i)).collect();
        assert_eq!(evaluate(&nodes, one, 0.0), ONE);
    }
}
