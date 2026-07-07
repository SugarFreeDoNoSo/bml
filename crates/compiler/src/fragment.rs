//! Micro-fragmentación AOT y formato binario `.bmlgraph`.
//!
//! El compilador empaqueta el DAG linealizado en fragmentos cuyo tamaño
//! de memoria pre-asignado no supera el umbral de caché objetivo (32 KB
//! para L1 por defecto, configurable a L3).
//!
//! # Formato `.bmlgraph`
//!
//! ```text
//! [magic: u32 = 0x4C4D4247]  // "BMLG" en little-endian
//! [version: u32]
//! [num_fragments: u32]
//! [fragment_0_size: u32]
//! [fragment_0_ops: ...]
//! [fragment_1_size: u32]
//! [fragment_1_ops: ...]
//! ...
//! ```
//!
//! Cada fragmento es un subconjunto contiguo de operaciones RPN. El
//! runtime ejecuta los fragmentos en orden, manteniendo el estado de
//! la pila entre fragmentos.

use crate::rpn::{RpnOp, RpnProgram};

/// Magic number de `.bmlgraph`: `0x4C4D4247` ("BMLG" en little-endian).
pub const BMLGRAPH_MAGIC: u32 = 0x4C4D4247;

/// Versión del formato `.bmlgraph`.
pub const BMLGRAPH_VERSION: u32 = 1;

/// Umbral por defecto: 32 KB (L1 cache).
pub const DEFAULT_L1_THRESHOLD: usize = 32 * 1024;

/// Umbral para L3 cache: 1 MB.
pub const L3_THRESHOLD: usize = 1024 * 1024;

/// Un fragmento del grafo BML.
///
/// Contiene un subconjunto contiguo de operaciones RPN. El tamaño
/// del fragmento (en bytes) no supera el umbral de caché objetivo.
#[derive(Debug, Clone, PartialEq)]
pub struct Fragment {
    /// Operaciones RPN del fragmento.
    pub ops: Vec<RpnOp>,
}

impl Fragment {
    /// Crea un fragmento vacío.
    pub fn new() -> Self {
        Self { ops: Vec::new() }
    }

    /// Número de operaciones.
    pub fn len(&self) -> usize {
        self.ops.len()
    }

    /// Returns `true` si el fragmento no tiene operaciones.
    pub fn is_empty(&self) -> bool {
        self.ops.is_empty()
    }

    /// Tamaño del fragmento en bytes.
    pub fn byte_size(&self) -> usize {
        self.ops.len() * std::mem::size_of::<RpnOp>()
    }

    /// Agrega una operación al fragmento.
    pub fn push(&mut self, op: RpnOp) {
        self.ops.push(op);
    }

    /// Evalúa el fragmento sobre una pila dada.
    ///
    /// A diferencia de [`RpnProgram::evaluate`], esta función recibe
    /// una pila preexistente (que puede tener valores de fragmentos
    /// anteriores) y la modifica in-place.
    pub fn evaluate_on_stack(&self, stack: &mut Vec<f64>) {
        let ctx = bml_domain::EvalContext::new(&[], &[]);
        self.evaluate_on_stack_with_ctx(stack, &ctx);
    }

    /// Evalúa el fragmento sobre una pila con contexto de inputs y pesos.
    pub fn evaluate_on_stack_with_ctx(&self, stack: &mut Vec<f64>, ctx: &bml_domain::EvalContext) {
        let mut i = 0;
        while i < self.ops.len() {
            match self.ops[i] {
                RpnOp::One => stack.push(1.0),
                RpnOp::Zero => stack.push(0.0),
                RpnOp::Var(id) => stack.push(ctx.get_var(id)),
                RpnOp::Const(id) => stack.push(ctx.get_const(id)),
                RpnOp::VarIndexed { base: _ } => {
                    let _offset = stack.pop().unwrap_or(0.0);
                    stack.push(f64::NAN);
                }
                RpnOp::StoreResult { slot: _ } => {
                    let _offset = stack.pop().unwrap_or(0.0);
                    let _value = stack.pop().unwrap_or(0.0);
                }
                RpnOp::Bml => {
                    let b = stack.pop().unwrap();
                    let a = stack.pop().unwrap();
                    stack.push(bml_domain::bml(a, b));
                }
                RpnOp::Dup => {
                    let v = *stack.last().unwrap();
                    stack.push(v);
                }
                RpnOp::FAdd => {
                    let b = stack.pop().unwrap_or(0.0);
                    let a = stack.pop().unwrap_or(0.0);
                    stack.push(a + b);
                }
                RpnOp::FMul => {
                    let b = stack.pop().unwrap_or(0.0);
                    let a = stack.pop().unwrap_or(0.0);
                    stack.push(a * b);
                }
                RpnOp::Pick { depth } => {
                    let d = depth as usize;
                    let idx = stack.len().saturating_sub(1 + d);
                    let v = stack.get(idx).copied().unwrap_or(0.0);
                    stack.push(v);
                }
                RpnOp::Drop => {
                    stack.pop();
                }
                RpnOp::Swap => {
                    let len = stack.len();
                    if len >= 2 {
                        stack.swap(len - 1, len - 2);
                    }
                }
                RpnOp::Loop { count, body_len } => {
                    let body_start = i + 1;
                    let body_end = body_start + body_len as usize;
                    for _ in 0..count {
                        let mut j = body_start;
                        while j < body_end {
                            match self.ops[j] {
                                RpnOp::One => stack.push(1.0),
                                RpnOp::Zero => stack.push(0.0),
                                RpnOp::Var(id) => stack.push(ctx.get_var(id)),
                                RpnOp::Const(id) => stack.push(ctx.get_const(id)),
                                RpnOp::VarIndexed { base: _ } => {
                                    let _offset = stack.pop().unwrap_or(0.0);
                                    stack.push(0.0);
                                }
                                RpnOp::StoreResult { slot: _ } => {
                                    let _offset = stack.pop().unwrap_or(0.0);
                                    let _value = stack.pop().unwrap_or(0.0);
                                }
                                RpnOp::Bml => {
                                    let b = stack.pop().unwrap();
                                    let a = stack.pop().unwrap();
                                    stack.push(bml_domain::bml(a, b));
                                }
                                RpnOp::Dup => {
                                    let v = *stack.last().unwrap();
                                    stack.push(v);
                                }
                                RpnOp::FAdd => {
                                    let b = stack.pop().unwrap_or(0.0);
                                    let a = stack.pop().unwrap_or(0.0);
                                    stack.push(a + b);
                                }
                                RpnOp::FMul => {
                                    let b = stack.pop().unwrap_or(0.0);
                                    let a = stack.pop().unwrap_or(0.0);
                                    stack.push(a * b);
                                }
                                RpnOp::Pick { depth } => {
                                    let d = depth as usize;
                                    let idx = stack.len().saturating_sub(1 + d);
                                    let v = stack.get(idx).copied().unwrap_or(0.0);
                                    stack.push(v);
                                }
                                RpnOp::Drop => {
                                    stack.pop();
                                }
                                RpnOp::Swap => {
                                    let len = stack.len();
                                    if len >= 2 {
                                        stack.swap(len - 1, len - 2);
                                    }
                                }
                                RpnOp::Loop { count: inner_count, body_len: inner_body_len } => {
                                    let inner_body_start = j + 1;
                                    let inner_body_end = inner_body_start + inner_body_len as usize;
                                    for _ in 0..inner_count {
                                        let mut k = inner_body_start;
                                        while k < inner_body_end {
                                            match self.ops[k] {
                                                RpnOp::One => stack.push(1.0),
                                                RpnOp::Zero => stack.push(0.0),
                                                RpnOp::Var(id) => stack.push(ctx.get_var(id)),
                                                RpnOp::Const(id) => stack.push(ctx.get_const(id)),
                                                RpnOp::VarIndexed { base: _ } => {
                                                    let _offset = stack.pop().unwrap_or(0.0);
                                                    stack.push(0.0);
                                                }
                                                RpnOp::StoreResult { slot: _ } => {
                                                    let _offset = stack.pop().unwrap_or(0.0);
                                                    let _value = stack.pop().unwrap_or(0.0);
                                                }
                                                RpnOp::Bml => {
                                                    let b = stack.pop().unwrap();
                                                    let a = stack.pop().unwrap();
                                                    stack.push(bml_domain::bml(a, b));
                                                }
                                                RpnOp::Dup => {
                                                    let v = *stack.last().unwrap();
                                                    stack.push(v);
                                                }
                                                RpnOp::FAdd => {
                                                    let b = stack.pop().unwrap_or(0.0);
                                                    let a = stack.pop().unwrap_or(0.0);
                                                    stack.push(a + b);
                                                }
                                                RpnOp::FMul => {
                                                    let b = stack.pop().unwrap_or(0.0);
                                                    let a = stack.pop().unwrap_or(0.0);
                                                    stack.push(a * b);
                                                }
                                                RpnOp::Pick { depth } => {
                                                    let d = depth as usize;
                                                    let idx = stack.len().saturating_sub(1 + d);
                                                    let v = stack.get(idx).copied().unwrap_or(0.0);
                                                    stack.push(v);
                                                }
                                                RpnOp::Drop => {
                                                    stack.pop();
                                                }
                                                RpnOp::Swap => {
                                                    let len = stack.len();
                                                    if len >= 2 {
                                                        stack.swap(len - 1, len - 2);
                                                    }
                                                }
                                                RpnOp::Loop { .. } => {
                                                    panic!("max 2 loop nesting levels");
                                                }
                                            }
                                            k += 1;
                                        }
                                    }
                                    j = inner_body_end;
                                    continue;
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
    }
}

impl Default for Fragment {
    fn default() -> Self {
        Self::new()
    }
}

/// Grafo BML fragmentado (`.bmlgraph`).
///
/// Contiene una lista de fragmentos, cada uno bajo el umbral de caché
/// objetivo. El runtime ejecuta los fragmentos en orden.
#[derive(Debug, Clone)]
pub struct BmlGraph {
    /// Fragmentos del grafo.
    pub fragments: Vec<Fragment>,
    /// Umbral de caché usado (en bytes).
    pub threshold: usize,
}

impl BmlGraph {
    /// Crea un `BmlGraph` vacío con el umbral dado.
    pub fn new(threshold: usize) -> Self {
        Self {
            fragments: Vec::new(),
            threshold,
        }
    }

    /// Número de fragmentos.
    pub fn num_fragments(&self) -> usize {
        self.fragments.len()
    }

    /// Tamaño total del grafo en bytes.
    pub fn total_byte_size(&self) -> usize {
        self.fragments.iter().map(|f| f.byte_size()).sum()
    }

    /// Verifica que todos los fragmentos están bajo el umbral.
    pub fn all_fragments_under_threshold(&self) -> bool {
        self.fragments
            .iter()
            .all(|f| f.byte_size() <= self.threshold)
    }

    /// Evalúa el grafo completo ejecutando los fragmentos en orden.
    ///
    /// La pila se mantiene entre fragmentos.
    pub fn evaluate(&self, x: f64) -> f64 {
        let mut stack: Vec<f64> = Vec::new();
        for fragment in &self.fragments {
            fragment.evaluate_on_stack(&mut stack);
        }
        let _ = x;
        stack.pop().unwrap_or(f64::NAN)
    }

    /// Serializa el grafo a bytes (formato `.bmlgraph`).
    pub fn serialize(&self) -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&BMLGRAPH_MAGIC.to_le_bytes());
        bytes.extend_from_slice(&BMLGRAPH_VERSION.to_le_bytes());
        bytes.extend_from_slice(&(self.fragments.len() as u32).to_le_bytes());
        for fragment in &self.fragments {
            bytes.extend_from_slice(&(fragment.ops.len() as u32).to_le_bytes());
            for op in &fragment.ops {
                match op {
                    RpnOp::One => bytes.push(0),
                    RpnOp::Zero => bytes.push(6),
                    RpnOp::Bml => bytes.push(1),
                    RpnOp::Dup => bytes.push(2),
                    RpnOp::Loop { count, body_len } => {
                        bytes.push(3);
                        bytes.extend_from_slice(&count.to_le_bytes());
                        bytes.extend_from_slice(&body_len.to_le_bytes());
                    }
                    RpnOp::Var(id) => {
                        bytes.push(4);
                        bytes.extend_from_slice(&id.to_le_bytes());
                    }
                    RpnOp::Const(id) => {
                        bytes.push(5);
                        bytes.extend_from_slice(&id.to_le_bytes());
                    }
                    RpnOp::VarIndexed { base } => {
                        bytes.push(7);
                        bytes.extend_from_slice(&base.to_le_bytes());
                    }
                    RpnOp::StoreResult { slot } => {
                        bytes.push(8);
                        bytes.extend_from_slice(&slot.to_le_bytes());
                    }
                    RpnOp::FAdd => bytes.push(9),
                    RpnOp::FMul => bytes.push(10),
                    RpnOp::Pick { depth } => {
                        bytes.push(11);
                        bytes.extend_from_slice(&depth.to_le_bytes());
                    }
                    RpnOp::Drop => bytes.push(12),
                    RpnOp::Swap => bytes.push(13),
                }
            }
        }
        bytes
    }

    /// Deserializa un grafo desde bytes (formato `.bmlgraph`).
    pub fn deserialize(bytes: &[u8], threshold: usize) -> Result<Self, String> {
        if bytes.len() < 12 {
            return Err("archivo demasiado pequeño".to_string());
        }
        let magic = u32::from_le_bytes(bytes[0..4].try_into().unwrap());
        if magic != BMLGRAPH_MAGIC {
            return Err(format!("magic inválido: 0x{magic:08X}"));
        }
        let version = u32::from_le_bytes(bytes[4..8].try_into().unwrap());
        if version != BMLGRAPH_VERSION {
            return Err(format!("versión no soportada: {version}"));
        }
        let num_fragments = u32::from_le_bytes(bytes[8..12].try_into().unwrap()) as usize;
        let mut offset = 12;
        let mut fragments = Vec::with_capacity(num_fragments);
        for _ in 0..num_fragments {
            if offset + 4 > bytes.len() {
                return Err("offset fuera de rango leyendo tamaño de fragmento".to_string());
            }
            let num_ops =
                u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap()) as usize;
            offset += 4;
            let mut ops = Vec::with_capacity(num_ops);
            for _ in 0..num_ops {
                if offset >= bytes.len() {
                    return Err("offset fuera de rango leyendo op".to_string());
                }
                let tag = bytes[offset];
                offset += 1;
                let op = match tag {
                    0 => RpnOp::One,
                    1 => RpnOp::Bml,
                    2 => RpnOp::Dup,
                    3 => {
                        if offset + 8 > bytes.len() {
                            return Err("offset fuera de rango leyendo Loop".to_string());
                        }
                        let count =
                            u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap());
                        offset += 4;
                        let body_len =
                            u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap());
                        offset += 4;
                        RpnOp::Loop { count, body_len }
                    }
                    4 => {
                        if offset + 4 > bytes.len() {
                            return Err("offset fuera de rango leyendo Var".to_string());
                        }
                        let id = u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap());
                        offset += 4;
                        RpnOp::Var(id)
                    }
                    6 => RpnOp::Zero,
                    5 => {
                        if offset + 4 > bytes.len() {
                            return Err("offset fuera de rango leyendo Const".to_string());
                        }
                        let id = u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap());
                        offset += 4;
                        RpnOp::Const(id)
                    }
                    7 => {
                        if offset + 4 > bytes.len() {
                            return Err("offset fuera de rango leyendo VarIndexed".to_string());
                        }
                        let base =
                            u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap());
                        offset += 4;
                        RpnOp::VarIndexed { base }
                    }
                    8 => {
                        if offset + 4 > bytes.len() {
                            return Err("offset fuera de rango leyendo StoreResult".to_string());
                        }
                        let slot =
                            u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap());
                        offset += 4;
                        RpnOp::StoreResult { slot }
                    }
                    9 => RpnOp::FAdd,
                    10 => RpnOp::FMul,
                    11 => {
                        if offset + 4 > bytes.len() {
                            return Err("offset fuera de rango leyendo Pick".to_string());
                        }
                        let depth =
                            u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap());
                        offset += 4;
                        RpnOp::Pick { depth }
                    }
                    12 => RpnOp::Drop,
                    13 => RpnOp::Swap,
                    _ => return Err(format!("tag desconocido: {tag}")),
                };
                ops.push(op);
            }
            fragments.push(Fragment { ops });
        }
        Ok(Self {
            fragments,
            threshold,
        })
    }
}

/// Particiona un programa RPN en fragmentos bajo el umbral de caché.
///
/// Recorre las operaciones del programa y las agrupa en fragmentos
/// cuyo tamaño en bytes no supere `threshold`. Las operaciones no se
/// parten: una operación va entera en un fragmento.
///
/// # Argumentos
///
/// - `program`: El programa RPN a particionar.
/// - `threshold`: Tamaño máximo de cada fragmento en bytes.
///
/// # Retorna
///
/// Un [`BmlGraph`] con los fragmentos.
pub fn fragment_program(program: &RpnProgram, threshold: usize) -> BmlGraph {
    let mut graph = BmlGraph::new(threshold);
    let mut current = Fragment::new();
    let op_size = std::mem::size_of::<RpnOp>();

    for &op in &program.ops {
        if current.byte_size() + op_size > threshold && !current.is_empty() {
            // El fragmento actual está lleno; cerrarlo y empezar uno nuevo.
            graph
                .fragments
                .push(std::mem::replace(&mut current, Fragment::new()));
        }
        current.push(op);
    }

    // Empujar el último fragmento si no está vacío.
    if !current.is_empty() {
        graph.fragments.push(current);
    }

    graph
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{linearize, HashConsRegistry};
    use proptest::prelude::*;

    fn build_program(n_ops: usize) -> RpnProgram {
        let mut reg = HashConsRegistry::new();
        let one = reg.one();
        let two = reg.bml(one, one);
        let mut node = two;
        let iterations = n_ops / 3;
        for _ in 0..iterations {
            node = reg.bml(node, two);
        }
        let soa = reg.into_soa();
        linearize(&soa, node)
    }

    #[test]
    fn fragment_small_program() {
        // Programa pequeño: cabe en un solo fragmento
        let program = build_program(100);
        let graph = fragment_program(&program, DEFAULT_L1_THRESHOLD);
        assert_eq!(graph.num_fragments(), 1);
        assert!(graph.all_fragments_under_threshold());
    }

    #[test]
    fn fragment_large_program() {
        // Programa grande: debe partirse en múltiples fragmentos.
        // Limitado a 10K ops para evitar stack overflow en linearize (recursivo).
        let program = build_program(10_000);
        let graph = fragment_program(&program, DEFAULT_L1_THRESHOLD);
        assert!(
            graph.num_fragments() > 1,
            "debería tener múltiples fragmentos"
        );
        assert!(
            graph.all_fragments_under_threshold(),
            "todos los fragmentos deben estar bajo el umbral"
        );
    }

    #[test]
    fn fragment_preserves_evaluation() {
        // La evaluación del grafo fragmentado debe coincidir con la del programa original.
        let program = build_program(1000);
        let graph = fragment_program(&program, DEFAULT_L1_THRESHOLD);
        let original_val = program.evaluate(0.0);
        let fragmented_val = graph.evaluate(0.0);
        assert_eq!(
            original_val.to_bits(),
            fragmented_val.to_bits(),
            "evaluación fragmentada {fragmented_val} != original {original_val}"
        );
    }

    #[test]
    fn fragment_threshold_configurable() {
        // Con umbral L3 (1 MB), un programa de 10K ops cabe en un solo fragmento.
        // Limitado para evitar stack overflow en linearize.
        let program = build_program(10_000);
        let graph = fragment_program(&program, L3_THRESHOLD);
        assert_eq!(graph.num_fragments(), 1, "debería caber en un fragmento L3");
        assert!(graph.all_fragments_under_threshold());
    }

    #[test]
    fn fragment_small_threshold() {
        // Con umbral muy pequeño, cada fragmento tiene pocas ops.
        // RpnOp ahora pesa 12 bytes (enum con Loop), así que usamos 24 bytes.
        let program = build_program(1000);
        let graph = fragment_program(&program, 24);
        assert!(graph.num_fragments() > 1);
        assert!(graph.all_fragments_under_threshold());
    }

    #[test]
    fn serialize_deserialize_roundtrip() {
        let program = build_program(1000);
        let graph = fragment_program(&program, DEFAULT_L1_THRESHOLD);
        let bytes = graph.serialize();
        let restored = BmlGraph::deserialize(&bytes, DEFAULT_L1_THRESHOLD).unwrap();
        assert_eq!(graph.fragments.len(), restored.fragments.len());
        for (a, b) in graph.fragments.iter().zip(restored.fragments.iter()) {
            assert_eq!(a.ops, b.ops);
        }
        // La evaluación debe coincidir
        assert_eq!(
            graph.evaluate(0.0).to_bits(),
            restored.evaluate(0.0).to_bits()
        );
    }

    #[test]
    fn deserialize_rejects_bad_magic() {
        let bytes = [0xDE, 0xAD, 0xBE, 0xEF, 1, 0, 0, 0, 0, 0, 0, 0];
        let result = BmlGraph::deserialize(&bytes, DEFAULT_L1_THRESHOLD);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("magic"));
    }

    #[test]
    fn bmlgraph_magic_value() {
        // "BMLG" en little-endian
        assert_eq!(BMLGRAPH_MAGIC, 0x4C4D4247);
    }

    #[test]
    fn empty_program_produces_empty_graph() {
        let program = RpnProgram::new();
        let graph = fragment_program(&program, DEFAULT_L1_THRESHOLD);
        assert_eq!(graph.num_fragments(), 0);
    }

    /// Propiedad: todos los fragmentos están bajo el umbral.
    proptest! {
        #[test]
        fn proptest_all_fragments_under_threshold(
            n_ops in 10u32..10000,
            threshold in 24u32..65536,
        ) {
            let program = build_program(n_ops as usize);
            let graph = fragment_program(&program, threshold as usize);
            prop_assert!(graph.all_fragments_under_threshold(),
                "fragmento excede umbral: threshold={threshold}");
        }
    }

    /// Propiedad: la evaluación fragmentada coincide con la original.
    #[allow(unused_doc_comments)]
    proptest! {
        #[test]
        fn proptest_fragment_preserves_evaluation(
            n_ops in 10u32..5000,
        ) {
            let program = build_program(n_ops as usize);
            let graph = fragment_program(&program, DEFAULT_L1_THRESHOLD);
            let original = program.evaluate(0.0);
            let fragmented = graph.evaluate(0.0);
            if original.is_finite() && fragmented.is_finite() {
                prop_assert!((original - fragmented).abs() < 1e-6);
            } else {
                prop_assert_eq!(original.to_bits(), fragmented.to_bits());
            }
        }
    }
}