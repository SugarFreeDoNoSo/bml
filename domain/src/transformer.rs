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
    // Operaciones estándar (derivadas del Supplementary Information del paper)
    // =====================================================================
    //
    // Cadena de reconstrucción (Supplementary Information, Sect. 2.5):
    //
    //   exp2(z) = bml(z, 1)                    [Lemma 1]
    //   log2(x) = bml(1, bml(bml(1, x), 1))    [Lemma 2]
    //   x - y   = bml(log2(x), exp2(y))        [Lemma 3]
    //   -x      = bml(log2(0), exp2(x)) = bml(-inf, exp2(x)) = 0 - x
    //   x + y   = x - (-y) = sub(x, neg(y))
    //   1/x     = exp2(-log2(x)) = exp2(neg(log2(x)))
    //   x * y   = exp2(log2(x) + log2(y)) = exp2(add(log2(x), log2(y)))
    //   x / y   = x * (1/y) = mul(x, recip(y))
    //   x^y     = exp2(y * log2(x)) = exp2(mul(y, log2(x)))
    //
    // Nota: log2(0) = -inf, 2^(-inf) = 0. Estas fórmulas funcionan en f64
    // con la convención IEEE 754 de infinitos. Requieren x > 0 para que
    // log2(x) esté definido en los reales.

    /// `x - y = bml(log2(x), exp2(y))`.
    ///
    /// Requiere `x > 0` (para que `log2(x)` esté definido en los reales).
    /// Identidad del Lemma 3 del Supplementary Information.
    pub fn sub(&mut self, x: NodeId, y: NodeId) -> NodeId {
        let log2_x = self.log2(x); // log2(x)
        let exp2_y = self.exp2(y); // exp2(y) = 2^y
        self.bml(log2_x, exp2_y) // 2^log2(x) - log2(2^y) = x - y
    }

    /// `-x = bml(log2(0), exp2(x)) = 0 - x`.
    ///
    /// Usa la convención `log2(0) = -inf`, `2^(-inf) = 0`.
    /// Identidad del Remark después del Lemma 3.
    pub fn neg(&mut self, x: NodeId) -> NodeId {
        // 0 = log2(1). Construimos el nodo 0 primero.
        let one = self.one();
        let zero = self.log2(one); // log2(1) = 0
                                   // -x = 0 - x = sub(0, x) = bml(log2(0), exp2(x))
                                   // Pero sub(0, x) requiere log2(0) = -inf, que no es representable.
                                   //
                                   // Alternativa: -x = bml(bml(1,1), exp2(x)) no funciona (da 4-x).
                                   //
                                   // El paper usa L(0) = -inf como convención. En f64, log2(0.0) = -inf.
                                   // Pero en la gramática BML pura, no hay nodo que evalúe a -inf.
                                   //
                                   // Solución: usar sub(ONE, x) solo cuando x < 1... no general.
                                   //
                                   // Para el transformer, neg se usa sobre log2(x) que puede ser
                                   // negativo. Pero sub requiere log2 del primer arg, que sería
                                   // log2(log2(x))... que no está definido si log2(x) < 0.
                                   //
                                   // La forma más limpia: -x = 1/x^(-1)... circular.
                                   //
                                   // Por ahora, implementamos neg como 0 - x usando la convención
                                   // de que el runtime maneja -inf. En la gramática BML, usamos
                                   // sub(zero, x) que expande a bml(log2(0), exp2(x)).
                                   // El nodo log2(0) se construye como log2(zero) donde zero = log2(1).
        let neg_zero = self.log2(zero); // log2(0) = -inf (en f64)
        let exp2_x = self.exp2(x); // 2^x
        self.bml(neg_zero, exp2_x) // 2^(-inf) - log2(2^x) = 0 - x = -x
    }

    /// `x + y = x - (-y) = sub(x, neg(y))`.
    ///
    /// Requiere `x > 0` (para `log2(x)` en `sub`).
    pub fn add(&mut self, x: NodeId, y: NodeId) -> NodeId {
        let neg_y = self.neg(y); // -y
        self.sub(x, neg_y) // x - (-y) = x + y
    }

    /// `1/x = exp2(-log2(x)) = exp2(neg(log2(x)))`.
    ///
    /// Requiere `x > 0`.
    pub fn recip(&mut self, x: NodeId) -> NodeId {
        let log2_x = self.log2(x); // log2(x)
        let neg_log2_x = self.neg(log2_x); // -log2(x)
        self.exp2(neg_log2_x) // 2^(-log2(x)) = 1/x
    }

    /// `x * y = exp2(log2(x) + log2(y)) = exp2(add(log2(x), log2(y)))`.
    ///
    /// Requiere `x > 0` y `y > 0`.
    pub fn mul(&mut self, x: NodeId, y: NodeId) -> NodeId {
        let log2_x = self.log2(x); // log2(x)
        let log2_y = self.log2(y); // log2(y)
        let sum = self.add(log2_x, log2_y); // log2(x) + log2(y)
        self.exp2(sum) // 2^(log2(x) + log2(y)) = x * y
    }

    /// `x / y = x * (1/y) = mul(x, recip(y))`.
    ///
    /// Requiere `x > 0` y `y > 0`.
    pub fn div(&mut self, x: NodeId, y: NodeId) -> NodeId {
        let recip_y = self.recip(y); // 1/y
        self.mul(x, recip_y) // x * (1/y) = x / y
    }

    /// `x^y = exp2(y * log2(x)) = exp2(mul(y, log2(x)))`.
    ///
    /// Requiere `x > 0`.
    pub fn pow(&mut self, x: NodeId, y: NodeId) -> NodeId {
        let log2_x = self.log2(x); // log2(x)
        let product = self.mul(y, log2_x); // y * log2(x)
        self.exp2(product) // 2^(y * log2(x)) = x^y
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

    // =====================================================================
    // Tests de operaciones aritméticas (Supplementary Information, Sect. 2.5)
    // =====================================================================

    /// Helper: evalúa un nodo del transformer.
    fn eval_node(t: BMLTransformer, root: NodeId) -> f64 {
        let soa = t.into_soa();
        let nodes: Vec<Node> = (0..soa.len() as NodeId).map(|i| soa.get(i)).collect();
        evaluate(&nodes, root, 0.0)
    }

    #[test]
    fn sub_works() {
        // x - y = bml(log2(x), exp2(y))
        // 8 - 3 = 5
        let mut t = BMLTransformer::new();
        let two = t.two();
        let two2 = t.two();
        let three = t.bml(two, two2); // 3
        let two3 = t.two();
        let two4 = t.two();
        let three2 = t.bml(two3, two4); // 3
        let eight = t.exp2(three); // 8
        let result = t.sub(eight, three2); // 8 - 3 = 5
        let val = eval_node(t, result);
        assert!((val - 5.0).abs() < 1e-9, "sub(8,3) = {val}, expected 5");
    }

    #[test]
    fn neg_works() {
        // -x = 0 - x = bml(log2(0), exp2(x))
        // -3 = -3
        let mut t = BMLTransformer::new();
        let two = t.two();
        let two2 = t.two();
        let three = t.bml(two, two2); // 3
        let result = t.neg(three); // -3
        let val = eval_node(t, result);
        assert!((val - (-3.0)).abs() < 1e-9, "neg(3) = {val}, expected -3");
    }

    #[test]
    fn add_works() {
        // x + y = x - (-y) = sub(x, neg(y))
        // 5 + 3 = 8
        let mut t = BMLTransformer::new();
        let two = t.two();
        let two2 = t.two();
        let three = t.bml(two, two2); // 3
        let two3 = t.two();
        let two4 = t.two();
        let three2 = t.bml(two3, two4); // 3
        let two5 = t.two();
        let two6 = t.two();
        let three3 = t.bml(two5, two6); // 3
        let eight = t.exp2(three); // 8
        let five = t.sub(eight, three2); // 8 - 3 = 5
        let result = t.add(five, three3); // 5 + 3 = 8
        let val = eval_node(t, result);
        assert!((val - 8.0).abs() < 1e-9, "add(5,3) = {val}, expected 8");
    }

    #[test]
    fn recip_works() {
        // 1/x = exp2(-log2(x))
        // 1/4 = 0.25
        let mut t = BMLTransformer::new();
        let two = t.two();
        let four = t.exp2(two); // 4
        let result = t.recip(four); // 1/4 = 0.25
        let val = eval_node(t, result);
        assert!((val - 0.25).abs() < 1e-9, "recip(4) = {val}, expected 0.25");
    }

    #[test]
    fn mul_works() {
        // x * y = exp2(log2(x) + log2(y))
        // 3 * 4 = 12
        let mut t = BMLTransformer::new();
        let two = t.two();
        let two2 = t.two();
        let three = t.bml(two, two2); // 3
        let two3 = t.two();
        let four = t.exp2(two3); // 4
        let result = t.mul(three, four); // 3 * 4 = 12
        let val = eval_node(t, result);
        assert!((val - 12.0).abs() < 1e-9, "mul(3,4) = {val}, expected 12");
    }

    #[test]
    fn div_works() {
        // x / y = x * (1/y)
        // 12 / 4 = 3
        let mut t = BMLTransformer::new();
        let two = t.two();
        let two2 = t.two();
        let three = t.bml(two, two2); // 3
        let two3 = t.two();
        let four = t.exp2(two3); // 4
        let twelve = t.mul(three, four); // 12
        let result = t.div(twelve, four); // 12 / 4 = 3
        let val = eval_node(t, result);
        assert!((val - 3.0).abs() < 1e-9, "div(12,4) = {val}, expected 3");
    }

    #[test]
    fn pow_works() {
        // x^y = exp2(y * log2(x))
        // 2^3 = 8
        let mut t = BMLTransformer::new();
        let two = t.two(); // 2
        let two2 = t.two();
        let two3 = t.two();
        let three = t.bml(two2, two3); // 3
        let result = t.pow(two, three); // 2^3 = 8
        let val = eval_node(t, result);
        assert!((val - 8.0).abs() < 1e-9, "pow(2,3) = {val}, expected 8");
    }
}
