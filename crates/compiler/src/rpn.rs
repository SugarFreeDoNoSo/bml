//! Linealización de DAG a Notación Polaca Inversa (RPN).
//!
//! El compilador convierte el DAG deduplicado en un arreglo unidimensional
//! en RPN. El runtime itera secuencialmente sobre este arreglo sin saltos
//! ni recursión, lo cual es ideal para la caché L1 de instrucciones.
//!
//! # Formato
//!
//! Cada instrucción RPN es un [`RpnOp`]:
//! - [`RpnOp::One`]: empuja la constante `1` a la pila.
//! - [`RpnOp::Bml`]: saca dos valores de la pila, aplica `bml`, empuja el resultado.
//!
//! La linealización se hace con un recorrido post-order del DAG. Como el
//! DAG ya está deduplicado por Hash Consing, los sub-árboles compartidos
//! se emiten una sola vez, precedidos de un [`RpnOp::Dup`] que duplica el
//! valor en la pila para reutilizarlo.

use bml_domain::{NodeId, NodeKind, NodeSoA};

/// Operación RPN.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RpnOp {
    /// Empuja la constante `1` a la pila.
    One,
    /// Saca dos valores `a, b` de la pila y empuja `bml(a, b)`.
    Bml,
    /// Duplica el valor en el tope de la pila.
    ///
    /// Se emite antes de un sub-árbol compartido para reutilizar el
    /// resultado ya computado sin recalcularlo.
    Dup,
    /// Repite un bloque de `body_len` operaciones `count` veces.
    ///
    /// El bloque empieza en la posición inmediatamente siguiente al
    /// `Loop` en el arreglo de ops. Entre iteraciones, la pila mantiene
    /// su estado (no se limpia). Esto permite patrones repetidos con
    /// distintos operandos sin expandir el programa RPN.
    ///
    /// # Semántica
    ///
    /// ```text
    /// Loop(count, body_len)
    /// [body: body_len ops]
    /// ```
    ///
    /// El runtime ejecuta `body` `count` veces secuencialmente.
    Loop {
        /// Número de veces que se repite el cuerpo.
        count: u32,
        /// Número de operaciones del cuerpo (que siguen al Loop).
        body_len: u32,
    },
}

/// Programa RPN: arreglo unidimensional de operaciones.
#[derive(Debug, Clone)]
pub struct RpnProgram {
    /// Secuencia de operaciones a ejecutar.
    pub ops: Vec<RpnOp>,
}

impl RpnProgram {
    /// Crea un programa RPN vacío.
    pub fn new() -> Self {
        Self { ops: Vec::new() }
    }

    /// Número de operaciones.
    pub fn len(&self) -> usize {
        self.ops.len()
    }

    /// Returns `true` si el programa no tiene operaciones.
    pub fn is_empty(&self) -> bool {
        self.ops.is_empty()
    }

    /// Agrega una operación al final.
    pub fn push(&mut self, op: RpnOp) {
        self.ops.push(op);
    }

    /// Evalúa el programa RPN sobre una pila.
    ///
    /// Esta es la evaluación de referencia. El hot loop del runtime
    /// (Hito 5) hará lo mismo pero sobre buffers pre-asignados.
    ///
    /// # Panics
    ///
    /// Panics si el programa es inválido (pila vacía al hacer `Bml` o `Dup`).
    pub fn evaluate(&self, x: f64) -> f64 {
        let mut stack: Vec<f64> = Vec::with_capacity(self.ops.len());
        let mut i = 0;
        while i < self.ops.len() {
            match self.ops[i] {
                RpnOp::One => stack.push(1.0),
                RpnOp::Bml => {
                    let b = stack.pop().unwrap();
                    let a = stack.pop().unwrap();
                    stack.push(bml_domain::bml(a, b));
                }
                RpnOp::Dup => {
                    let v = *stack.last().unwrap();
                    stack.push(v);
                }
                RpnOp::Loop { count, body_len } => {
                    let body_start = i + 1;
                    let body_end = body_start + body_len as usize;
                    for _ in 0..count {
                        // Ejecutar el cuerpo del loop
                        let mut j = body_start;
                        while j < body_end {
                            match self.ops[j] {
                                RpnOp::One => stack.push(1.0),
                                RpnOp::Bml => {
                                    let b = stack.pop().unwrap();
                                    let a = stack.pop().unwrap();
                                    stack.push(bml_domain::bml(a, b));
                                }
                                RpnOp::Dup => {
                                    let v = *stack.last().unwrap();
                                    stack.push(v);
                                }
                                RpnOp::Loop { .. } => {
                                    // Loops anidados no soportados en el cuerpo por simplicidad.
                                    // Se podrían soportar con un stack de loop frames.
                                    panic!("loops anidados no soportados");
                                }
                            }
                            j += 1;
                        }
                    }
                    i = body_end;
                    continue;
                }
            }
            i += 1;
        }
        // El parámetro x se reserva para nodos de variable (futuro).
        let _ = x;
        stack.pop().unwrap_or(f64::NAN)
    }
}

impl Default for RpnProgram {
    fn default() -> Self {
        Self::new()
    }
}

/// Linealiza un DAG (dado como `NodeSoA` + raíz) a un programa RPN.
///
/// Recorre el DAG en post-order. Los sub-árboles compartidos (múltiples
/// padres) se emiten una sola vez y se reutilizan con `Dup`.
///
/// # Argumentos
///
/// - `soa`: El layout SoA con los nodos del DAG.
/// - `root`: El `NodeId` de la raíz del DAG.
///
/// # Retorna
///
/// Un [`RpnProgram`] que evalúa al mismo valor que el DAG original.
pub fn linearize(soa: &NodeSoA, root: NodeId) -> RpnProgram {
    let mut program = RpnProgram::new();
    let mut visited = vec![false; soa.len()];
    emit(soa, root, &mut visited, &mut program);
    program
}

/// Emite el código RPN para un nodo en post-order.
///
/// - Si el nodo ya fue visitado (compartido), emite `Dup` para reutilizarlo.
/// - Si no, lo visita y emite su código.
fn emit(soa: &NodeSoA, id: NodeId, visited: &mut [bool], program: &mut RpnProgram) {
    let idx = id as usize;
    if visited[idx] {
        program.push(RpnOp::Dup);
        return;
    }
    visited[idx] = true;
    match soa.kinds[idx] {
        NodeKind::One => program.push(RpnOp::One),
        NodeKind::Bml => {
            // Post-order: left, right, then Bml
            emit(soa, soa.lefts[idx], visited, program);
            emit(soa, soa.rights[idx], visited, program);
            program.push(RpnOp::Bml);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bml_domain::BMLTransformer;
    use proptest::prelude::*;

    fn build_two_program() -> (bml_domain::NodeSoA, NodeId, RpnProgram) {
        // bml(1, 1) = 2
        let mut t = BMLTransformer::new();
        let root = t.two();
        let soa = t.into_soa();
        let program = linearize(&soa, root);
        (soa, root, program)
    }

    #[test]
    fn linearize_two() {
        let (soa, root, program) = build_two_program();
        // bml(1, 1) -> One, One, Bml
        assert_eq!(program.ops, vec![RpnOp::One, RpnOp::One, RpnOp::Bml]);
        assert_eq!(program.evaluate(0.0), 2.0);
        // Coincide con la evaluación del DAG
        let dag = crate::Dag::new(soa, root);
        assert!((program.evaluate(0.0) - dag.evaluate(0.0)).abs() < 1e-12);
    }

    #[test]
    fn linearize_exp2() {
        // 2^3 = 8, donde 3 = bml(2, 2)
        let mut t = BMLTransformer::new();
        let two = t.two();
        let two2 = t.two();
        let three = t.bml(two, two2);
        let root = t.exp2(three);
        let soa = t.into_soa();
        let program = linearize(&soa, root);
        assert!((program.evaluate(0.0) - 8.0).abs() < 1e-9);
        let dag = crate::Dag::new(soa, root);
        assert!((program.evaluate(0.0) - dag.evaluate(0.0)).abs() < 1e-9);
    }

    #[test]
    fn shared_subtree_uses_dup() {
        // Construir un DAG con sub-árbol compartido.
        // bml(bml(1,1), bml(1,1)) — el bml(1,1) está compartido.
        // Con Hash Consing, bml(1,1) se emite una vez y se reutiliza con Dup.
        //
        // Recorrido post-order del DAG:
        //   root = bml(two, two)
        //   two = bml(one, one)
        //   one = One
        //
        // emit(root) -> emit(two) -> emit(one)=One, emit(one)=Dup (ya visitado), Bml
        //             -> emit(two)=Dup (ya visitado), Bml
        // Resultado: [One, Dup, Bml, Dup, Bml]
        use bml_domain::NodeSoA;
        let mut soa = NodeSoA::new();
        let one = soa.push_one(); // 0: One
        let two = soa.push_bml(one, one); // 1: bml(0, 0)
        let root = soa.push_bml(two, two); // 2: bml(1, 1)
        let program = linearize(&soa, root);
        assert_eq!(
            program.ops,
            vec![RpnOp::One, RpnOp::Dup, RpnOp::Bml, RpnOp::Dup, RpnOp::Bml]
        );
        // bml(2, 2) = 2^2 - log2(2) = 4 - 1 = 3
        assert!((program.evaluate(0.0) - 3.0).abs() < 1e-9);
    }

    #[test]
    fn rpn_matches_dag_evaluation() {
        // Para varios DAGs, la evaluación RPN debe coincidir con la del DAG.
        let mut t = BMLTransformer::new();
        let two = t.two();
        let two2 = t.two();
        let three = t.bml(two, two2);
        let eight = t.exp2(three);
        let log = t.log2(eight); // log2(8) = 3
        let soa = t.into_soa();
        let program = linearize(&soa, log);
        let dag = crate::Dag::new(soa, log);
        let rpn_val = program.evaluate(0.0);
        let dag_val = dag.evaluate(0.0);
        assert!(
            (rpn_val - dag_val).abs() < 1e-9,
            "RPN={rpn_val}, DAG={dag_val}"
        );
        assert!((rpn_val - 3.0).abs() < 1e-9);
    }

    /// Propiedad: la evaluación RPN siempre coincide con la del DAG.
    #[allow(unused_doc_comments)]
    proptest! {
        #[test]
        fn proptest_rpn_matches_dag(
            depth in 1u32..5,
        ) {
            // Construir un DAG anidado de profundidad `depth` usando bml(1,1) repetido.
            let mut t = BMLTransformer::new();
            let one = t.one();
            let mut node = t.bml(one, one); // 2
            for _ in 1..depth {
                // bml(node, node) — sub-árbol compartido
                node = t.bml(node, node);
            }
            let soa = t.into_soa();
            let program = linearize(&soa, node);
            let dag = crate::Dag::new(soa, node);
            let rpn_val = program.evaluate(0.0);
            let dag_val = dag.evaluate(0.0);
            // Ambos pueden ser inf o nan a profundidades grandes; comparamos
            // solo si son finitos, sino verificamos que ambos coinciden en
            // su clase (inf/nan).
            if rpn_val.is_finite() && dag_val.is_finite() {
                prop_assert!((rpn_val - dag_val).abs() < 1e-6, "RPN={rpn_val}, DAG={dag_val}");
            } else {
                prop_assert_eq!(rpn_val.is_infinite(), dag_val.is_infinite());
            }
        }
    }

    // =====================================================================
    // Tests de Loop (opcode para patrones repetidos)
    // =====================================================================

    #[test]
    fn loop_repeats_body() {
        // Loop(3, 2) repite [One, Bml] 3 veces.
        // Antes del loop, empujamos 1 (valor inicial).
        // Iter 1: push 1 -> [1,1], bml(1,1)=2 -> [2]
        // Iter 2: push 1 -> [2,1], bml(2,1)=4 -> [4]
        // Iter 3: push 1 -> [4,1], bml(4,1)=16 -> [16]
        let mut program = RpnProgram::new();
        program.push(RpnOp::One); // valor inicial
        program.push(RpnOp::Loop {
            count: 3,
            body_len: 2,
        });
        program.push(RpnOp::One);
        program.push(RpnOp::Bml);
        let result = program.evaluate(0.0);
        assert!(
            (result - 16.0).abs() < 1e-9,
            "Loop result = {result}, expected 16"
        );
    }

    #[test]
    fn loop_count_zero() {
        // Loop(0, 2) no ejecuta el cuerpo. Queda el valor inicial.
        let mut program = RpnProgram::new();
        program.push(RpnOp::One); // valor inicial
        program.push(RpnOp::Loop {
            count: 0,
            body_len: 2,
        });
        program.push(RpnOp::One);
        program.push(RpnOp::Bml);
        let result = program.evaluate(0.0);
        assert!(
            (result - 1.0).abs() < 1e-9,
            "Loop(0) = {result}, expected 1"
        );
    }

    #[test]
    fn loop_count_one() {
        let mut program = RpnProgram::new();
        program.push(RpnOp::One); // valor inicial
        program.push(RpnOp::Loop {
            count: 1,
            body_len: 2,
        });
        program.push(RpnOp::One);
        program.push(RpnOp::Bml);
        let result = program.evaluate(0.0);
        assert!(
            (result - 2.0).abs() < 1e-9,
            "Loop(1) = {result}, expected 2"
        );
    }

    #[test]
    fn loop_equivalent_to_unrolled() {
        let mut loop_prog = RpnProgram::new();
        loop_prog.push(RpnOp::One); // valor inicial
        loop_prog.push(RpnOp::Loop {
            count: 3,
            body_len: 2,
        });
        loop_prog.push(RpnOp::One);
        loop_prog.push(RpnOp::Bml);

        let mut unrolled = RpnProgram::new();
        unrolled.push(RpnOp::One); // valor inicial
        for _ in 0..3 {
            unrolled.push(RpnOp::One);
            unrolled.push(RpnOp::Bml);
        }

        let loop_val = loop_prog.evaluate(0.0);
        let unrolled_val = unrolled.evaluate(0.0);
        assert_eq!(
            loop_val.to_bits(),
            unrolled_val.to_bits(),
            "Loop = {loop_val}, unrolled = {unrolled_val}"
        );
    }

    #[test]
    fn loop_program_smaller_than_unrolled() {
        let n: usize = 100;
        let m: usize = 3;

        let mut loop_prog = RpnProgram::new();
        loop_prog.push(RpnOp::One); // valor inicial
        loop_prog.push(RpnOp::One); // segundo valor inicial
        loop_prog.push(RpnOp::Loop {
            count: n as u32,
            body_len: m as u32,
        });
        loop_prog.push(RpnOp::One);
        loop_prog.push(RpnOp::One);
        loop_prog.push(RpnOp::Bml);

        let mut unrolled = RpnProgram::new();
        unrolled.push(RpnOp::One); // valor inicial
        unrolled.push(RpnOp::One); // segundo valor inicial
        for _ in 0..n {
            unrolled.push(RpnOp::One);
            unrolled.push(RpnOp::One);
            unrolled.push(RpnOp::Bml);
        }

        // Loop prog: 2 init + 1 Loop + 3 body = 6 ops
        assert_eq!(loop_prog.len(), m + 3, "Loop program should be M+3 ops");
        // Unrolled: 2 init + N*M body = 2 + 300 = 302 ops
        assert_eq!(unrolled.len(), n * m + 2, "Unrolled should be N*M+2 ops");
        assert!(loop_prog.len() < unrolled.len(), "Loop should be smaller");
    }
}
