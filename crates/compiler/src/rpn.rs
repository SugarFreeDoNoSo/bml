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
//! DAG ya está deduplicado por Hash Consing, cada valor conserva una posición
//! de pila y se reutiliza con Pick. Bytecode y pila son O(nodos alcanzables).
//! Inicializar posiciones y copiar constantes cuesta O(nodos totales + pool).
//! Una sola posición de resultado permanece al finalizar.

use bml_domain::{NodeId, NodeKind, NodeSoA};

/// Operación RPN.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RpnOp {
    /// Empuja la constante `1` a la pila.
    One,
    /// Empuja la constante `0` a la pila.
    Zero,
    /// Saca dos valores `a, b` de la pila y empuja `bml(a, b)`.
    Bml,
    /// Duplica el valor en el tope de la pila.
    ///
    /// Se emite antes de un sub-árbol compartido para reutilizar el
    /// resultado ya computado sin recalcularlo.
    Dup,
    /// Empuja el valor de una variable (input) a la pila.
    /// El índice se resuelve desde el contexto de inputs.
    Var(u32),
    /// Empuja el valor de una constante (peso del modelo) a la pila.
    /// El índice se resuelve desde el pool de pesos.
    Const(u32),
    /// Lee el buffer de resultados del runtime en `base + offset`.
    /// Consume el offset y conserva la conversión permisiva histórica a u32.
    /// No direcciona `EvalContext`; para inputs y constantes usar las cargas
    /// explícitas `ContextVarIndexed` y `ContextConstIndexed`.
    VarIndexed {
        /// Base del índice.
        base: u32,
    },
    /// Pop an exact nonnegative integer offset and load an input at base+offset.
    /// Unlike `VarIndexed`, this addresses `EvalContext::inputs`, not a buffer.
    /// Invalid indices produce NaN in IEEE evaluators and fail checked execution.
    ContextVarIndexed {
        /// First input index in the addressed vector.
        base: u32,
    },
    /// Pop an exact nonnegative integer offset and load an external constant.
    /// Invalid indices produce NaN in IEEE evaluators and fail checked execution.
    ContextConstIndexed {
        /// First constant index in the addressed vector.
        base: u32,
    },
    /// Writes a value to `ResultBuffer` at `slot[offset]`.
    ///
    /// Pops the offset from the top of the stack, then pops the value below it.
    /// Both operands are consumed; duplicate the value before pushing the
    /// offset if it must remain on the stack. Loops can supply dynamic offsets.
    StoreResult {
        /// Slot del buffer donde escribir.
        slot: u32,
    },
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
    /// Suma aritmética: saca dos valores y empuja su suma (a + b).
    FAdd,
    /// Multiplicación aritmética: saca dos valores y empuja su producto (a * b).
    FMul,
    /// Duplica el valor en una profundidad específica de la pila.
    ///
    /// `depth = 0` es el tope de la pila, `depth = 1` es el valor debajo, etc.
    /// El valor se copia al tope sin consumirse.
    Pick {
        /// Profundidad del valor a copiar (0 = tope).
        depth: u32,
    },
    /// Descarta el valor en el tope de la pila.
    Drop,
    /// Intercambia los dos valores en el tope de la pila: (a, b) -> (b, a).
    Swap,
    /// Select the larger of two finite operands, consuming both values.
    /// Nonfinite operands produce NaN; +0 wins a tie with -0.
    FMax,
}

/// Finite maximum selection shared by every RPN evaluator.
///
/// Both operands must be finite. Equal nonzero operands retain their value;
/// +0 wins a signed-zero tie, while two negative zeros remain negative zero.
#[inline]
pub fn finite_max(left: f64, right: f64) -> f64 {
    if !left.is_finite() || !right.is_finite() {
        f64::NAN
    } else if left == 0.0 && right == 0.0 {
        f64::from_bits(left.to_bits() & right.to_bits())
    } else {
        left.max(right)
    }
}

/// Resolve an exact, finite, nonnegative u32 offset without saturating casts.
/// Negative zero denotes index zero. Overflow of either operand or their sum
/// is rejected; the caller must additionally check its context slice length.
pub fn context_index(base: u32, offset: f64) -> Option<u32> {
    if !offset.is_finite() || offset < 0.0 || offset.fract() != 0.0 || offset > u32::MAX as f64 {
        return None;
    }
    base.checked_add(offset as u32)
}

/// Programa RPN: arreglo unidimensional de operaciones.
#[derive(Debug, Clone)]
pub struct RpnProgram {
    /// Owned constants; an explicit EvalContext overrides this pool.
    pub const_pool: Vec<f64>,
    /// Secuencia de operaciones a ejecutar.
    pub ops: Vec<RpnOp>,
}

impl RpnProgram {
    /// Crea un programa RPN vacío.
    pub fn new() -> Self {
        Self {
            ops: Vec::new(),
            const_pool: Vec::new(),
        }
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
    /// Evaluación de referencia sin variables, usando las constantes propias.
    /// Para evaluar con inputs y pesos, usar [`Self::evaluate_with_ctx`].
    ///
    /// # Panics
    ///
    /// Panics si el programa es inválido (pila vacía al hacer `Bml` o `Dup`).
    pub fn evaluate(&self) -> f64 {
        let ctx = bml_domain::EvalContext::new(&[], &self.const_pool);
        self.evaluate_with_ctx(&ctx)
    }

    /// Evalúa el programa RPN con un contexto de inputs y pesos.
    ///
    /// `Var(id)` se resuelve desde `ctx.inputs` y `Const(id)` desde `ctx.weights`.
    pub fn evaluate_with_ctx(&self, ctx: &bml_domain::EvalContext) -> f64 {
        let mut stack: Vec<f64> = Vec::with_capacity(self.ops.len());
        if uses_counter_loops(&self.ops) {
            return if evaluate_indexed_ops(&self.ops, &mut stack, ctx, 0).is_ok() {
                stack.pop().unwrap_or(f64::NAN)
            } else {
                f64::NAN
            };
        }
        let mut i = 0;
        while i < self.ops.len() {
            match self.ops[i] {
                RpnOp::One => stack.push(1.0),
                RpnOp::Zero => stack.push(0.0),
                RpnOp::Var(id) => stack.push(ctx.get_var(id)),
                RpnOp::Const(id) => stack.push(ctx.get_const(id)),
                RpnOp::ContextVarIndexed { base } => {
                    let offset = stack.pop().unwrap_or(f64::NAN);
                    stack.push(
                        crate::rpn::context_index(base, offset)
                            .map_or(f64::NAN, |id| ctx.get_var(id)),
                    );
                }
                RpnOp::ContextConstIndexed { base } => {
                    let offset = stack.pop().unwrap_or(f64::NAN);
                    stack.push(
                        crate::rpn::context_index(base, offset)
                            .map_or(f64::NAN, |id| ctx.get_const(id)),
                    );
                }
                RpnOp::VarIndexed { base: _ } => {
                    let _offset = stack.pop().unwrap_or(0.0) as u32;
                    stack.push(f64::NAN); // No result buffer is available in this evaluator.
                }
                RpnOp::StoreResult { slot: _ } => {
                    let _offset = stack.pop().unwrap_or(0.0);
                    let _value = stack.pop().unwrap_or(0.0);
                }
                RpnOp::Bml => {
                    let b = stack.pop().unwrap();
                    let a = stack.pop().unwrap();
                    stack.push(bml_domain::bml_base_op(a, b));
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
                RpnOp::FMax => {
                    let right = stack.pop().unwrap_or(f64::NAN);
                    let left = stack.pop().unwrap_or(f64::NAN);
                    stack.push(finite_max(left, right));
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
                                RpnOp::ContextVarIndexed { base } => {
                                    let offset = stack.pop().unwrap_or(f64::NAN);
                                    stack.push(
                                        crate::rpn::context_index(base, offset)
                                            .map_or(f64::NAN, |id| ctx.get_var(id)),
                                    );
                                }
                                RpnOp::ContextConstIndexed { base } => {
                                    let offset = stack.pop().unwrap_or(f64::NAN);
                                    stack.push(
                                        crate::rpn::context_index(base, offset)
                                            .map_or(f64::NAN, |id| ctx.get_const(id)),
                                    );
                                }
                                RpnOp::VarIndexed { base: _base } => {
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
                                    stack.push(bml_domain::bml_base_op(a, b));
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
                                RpnOp::FMax => {
                                    let right = stack.pop().unwrap_or(f64::NAN);
                                    let left = stack.pop().unwrap_or(f64::NAN);
                                    stack.push(finite_max(left, right));
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
                                RpnOp::Loop {
                                    count: inner_count,
                                    body_len: inner_body_len,
                                } => {
                                    let inner_body_start = j + 1;
                                    let inner_body_end = inner_body_start + inner_body_len as usize;
                                    for _loop_iter in 0..inner_count {
                                        let mut k = inner_body_start;
                                        while k < inner_body_end {
                                            match self.ops[k] {
                                                RpnOp::One => stack.push(1.0),
                                                RpnOp::Zero => stack.push(0.0),
                                                RpnOp::Var(id) => stack.push(ctx.get_var(id)),
                                                RpnOp::Const(id) => stack.push(ctx.get_const(id)),
                                                RpnOp::ContextVarIndexed { base } => {
                                                    let offset = stack.pop().unwrap_or(f64::NAN);
                                                    stack.push(
                                                        crate::rpn::context_index(base, offset)
                                                            .map_or(f64::NAN, |id| ctx.get_var(id)),
                                                    );
                                                }
                                                RpnOp::ContextConstIndexed { base } => {
                                                    let offset = stack.pop().unwrap_or(f64::NAN);
                                                    stack.push(
                                                        crate::rpn::context_index(base, offset)
                                                            .map_or(f64::NAN, |id| {
                                                                ctx.get_const(id)
                                                            }),
                                                    );
                                                }
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
                                                    stack.push(bml_domain::bml_base_op(a, b));
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
                                                RpnOp::FMax => {
                                                    let right = stack.pop().unwrap_or(f64::NAN);
                                                    let left = stack.pop().unwrap_or(f64::NAN);
                                                    stack.push(finite_max(left, right));
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
        stack.pop().unwrap_or(f64::NAN)
    }
}

pub(crate) fn uses_counter_loops(ops: &[RpnOp]) -> bool {
    ops.iter().any(|op| {
        matches!(
            op,
            RpnOp::ContextVarIndexed { .. } | RpnOp::ContextConstIndexed { .. } | RpnOp::FMax
        )
    })
}

// The historical reference interpreters repeat loop bodies without injecting
// counters. New indexed programs explicitly select the runtime counter semantics;
// leaving the old branch intact preserves previously constructed program results.
pub(crate) fn evaluate_indexed_ops(
    ops: &[RpnOp],
    stack: &mut Vec<f64>,
    ctx: &bml_domain::EvalContext,
    depth: usize,
) -> Result<(), ()> {
    let mut i = 0usize;
    while i < ops.len() {
        match ops[i] {
            RpnOp::One => stack.push(1.0),
            RpnOp::Zero => stack.push(0.0),
            RpnOp::Var(id) => stack.push(ctx.get_var(id)),
            RpnOp::Const(id) => stack.push(ctx.get_const(id)),
            RpnOp::ContextVarIndexed { base } | RpnOp::ContextConstIndexed { base } => {
                let offset = stack.pop().ok_or(())?;
                let value = context_index(base, offset).map_or(f64::NAN, |id| {
                    if matches!(ops[i], RpnOp::ContextVarIndexed { .. }) {
                        ctx.get_var(id)
                    } else {
                        ctx.get_const(id)
                    }
                });
                stack.push(value);
            }
            RpnOp::Bml | RpnOp::FAdd | RpnOp::FMul | RpnOp::FMax => {
                let right = stack.pop().ok_or(())?;
                let left = stack.pop().ok_or(())?;
                stack.push(match ops[i] {
                    RpnOp::Bml => bml_domain::bml_base_op(left, right),
                    RpnOp::FAdd => left + right,
                    RpnOp::FMax => finite_max(left, right),
                    _ => left * right,
                });
            }
            RpnOp::Dup => stack.push(*stack.last().ok_or(())?),
            RpnOp::Pick { depth } => {
                let index = stack.len().checked_sub(depth as usize + 1).ok_or(())?;
                stack.push(stack[index]);
            }
            RpnOp::Drop => {
                stack.pop().ok_or(())?;
            }
            RpnOp::Swap => {
                let n = stack.len();
                if n < 2 {
                    return Err(());
                }
                stack.swap(n - 1, n - 2);
            }
            RpnOp::VarIndexed { .. } => {
                stack.pop().ok_or(())?;
                stack.push(f64::NAN);
            }
            RpnOp::StoreResult { .. } => {
                stack.pop().ok_or(())?;
                stack.pop().ok_or(())?;
            }
            RpnOp::Loop { count, body_len } => {
                if depth >= 2 {
                    return Err(());
                }
                let end = (i + 1).checked_add(body_len as usize).ok_or(())?;
                let body = ops.get(i + 1..end).ok_or(())?;
                for index in 0..count {
                    stack.push(f64::from(index));
                    evaluate_indexed_ops(body, stack, ctx, depth + 1)?;
                }
                i = end;
                continue;
            }
        }
        i += 1;
    }
    Ok(())
}

impl Default for RpnProgram {
    fn default() -> Self {
        Self::new()
    }
}

/// Linealiza un DAG (dado como `NodeSoA` + raíz) a un programa RPN.
///
/// Recorre el DAG en post-order. Los sub-árboles compartidos (múltiples
/// padres) se emiten una sola vez y se reutilizan por su posición de pila.
/// Los hijos deben preceder al padre, como en NodeSoA::push_bml.
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
    program.const_pool.clone_from(&soa.const_pool);
    let mut slots = vec![None; soa.len()];
    let mut pending = vec![(root, false)];
    let mut height = 0usize;
    while let Some((id, expanded)) = pending.pop() {
        let idx = id as usize;
        if slots[idx].is_some() {
            continue;
        }
        if soa.kinds[idx] == NodeKind::Bml && !expanded {
            let left = soa.lefts[idx];
            let right = soa.rights[idx];
            assert!(
                left < id && right < id,
                "children must precede their parent"
            );
            pending.push((id, true));
            pending.push((right, false));
            pending.push((left, false));
            continue;
        }
        match soa.kinds[idx] {
            NodeKind::One => program.push(RpnOp::One),
            NodeKind::Zero => program.push(RpnOp::Zero),
            NodeKind::Var(var) => program.push(RpnOp::Var(var)),
            NodeKind::Const(value) => program.push(RpnOp::Const(value)),
            NodeKind::Bml => {
                let left = slots[soa.lefts[idx] as usize].unwrap();
                let right = slots[soa.rights[idx] as usize].unwrap();
                program.push(RpnOp::Pick {
                    depth: (height - 1 - left) as u32,
                });
                program.push(RpnOp::Pick {
                    depth: (height - right) as u32,
                });
                program.push(RpnOp::Bml);
            }
        }
        slots[idx] = Some(height);
        height += 1;
    }
    // Net stack effect +1, including when this program is concatenated.
    for _ in 1..height {
        program.push(RpnOp::Swap);
        program.push(RpnOp::Drop);
    }
    program
}

/// Lower several DAG roots into one program, reusing their shared node values.
///
/// Each output is written to `slot[index]`, in root order, using the value of
/// `Const(output_constants_start + index)` as its offset. The caller must bind
/// those constants to the exact consecutive offsets `0, 1, ...`. Existing
/// constant indices and the owned constant pool are preserved; this function
/// does not insert or remap the offset literals. An external execution context
/// may supply both the original constants and the offset literals.
///
/// Every reachable node is evaluated once with the same ordered operands as
/// [`linearize`]. Reusing the resulting f64 value preserves the legacy IEEE
/// arithmetic, including infinite intermediates; it adds no real-domain checks
/// or native arithmetic. After storing every output, the program discards its
/// retained values and leaves a single administrative `One` on the stack.
///
/// Empty outputs, malformed SoA columns, invalid roots, nonpreceding children,
/// and overflowing constant indices are rejected before emitting any bytecode.
/// Input/constant context lengths and buffer bounds still require runtime
/// validation. Construction and execution budgets remain the caller's concern.
pub fn linearize_outputs(
    soa: &NodeSoA,
    roots: &[NodeId],
    slot: u32,
    output_constants_start: u32,
) -> Result<RpnProgram, String> {
    let output_count = u32::try_from(roots.len()).map_err(|_| "too many DAG outputs")?;
    let last_output = output_count.checked_sub(1).ok_or("empty DAG outputs")?;
    output_constants_start
        .checked_add(last_output)
        .ok_or("output constant index overflow")?;
    if soa.len() > u32::MAX as usize
        || soa.lefts.len() != soa.len()
        || soa.rights.len() != soa.len()
    {
        return Err("invalid DAG columns".into());
    }
    // Validate all reachable references before creating a partial output
    // program. Const/Var indices refer to the caller's execution context, not
    // necessarily to the owned pool, so they are checked by the runtime.
    let mut reachable = vec![false; soa.len()];
    let mut pending_ids = Vec::new();
    for &root in roots {
        if root as usize >= soa.len() {
            return Err("unknown DAG output node".into());
        }
        pending_ids.push(root);
    }
    while let Some(id) = pending_ids.pop() {
        let index = id as usize;
        if reachable[index] {
            continue;
        }
        if soa.kinds[index] == NodeKind::Bml {
            let left = soa.lefts[index];
            let right = soa.rights[index];
            if left >= id || right >= id {
                return Err("DAG children must precede their parent".into());
            }
            pending_ids.push(right);
            pending_ids.push(left);
        }
        reachable[index] = true;
    }

    let mut program = RpnProgram::new();
    program.const_pool.clone_from(&soa.const_pool);
    let mut slots = vec![None; soa.len()];
    let mut pending = Vec::new();
    let mut height = 0usize;
    for (output, &root) in roots.iter().enumerate() {
        pending.push((root, false));
        while let Some((id, expanded)) = pending.pop() {
            let index = id as usize;
            if slots[index].is_some() {
                continue;
            }
            if soa.kinds[index] == NodeKind::Bml && !expanded {
                pending.push((id, true));
                pending.push((soa.rights[index], false));
                pending.push((soa.lefts[index], false));
                continue;
            }
            match soa.kinds[index] {
                NodeKind::One => program.push(RpnOp::One),
                NodeKind::Zero => program.push(RpnOp::Zero),
                NodeKind::Var(var) => program.push(RpnOp::Var(var)),
                NodeKind::Const(value) => program.push(RpnOp::Const(value)),
                NodeKind::Bml => {
                    let left = slots[soa.lefts[index] as usize].unwrap();
                    let right = slots[soa.rights[index] as usize].unwrap();
                    program.push(RpnOp::Pick {
                        depth: (height - 1 - left) as u32,
                    });
                    program.push(RpnOp::Pick {
                        depth: (height - right) as u32,
                    });
                    program.push(RpnOp::Bml);
                }
            }
            slots[index] = Some(height);
            height += 1;
        }
        program.push(RpnOp::Pick {
            depth: (height - 1 - slots[root as usize].unwrap()) as u32,
        });
        program.push(RpnOp::Const(output_constants_start + output as u32));
        program.push(RpnOp::StoreResult { slot });
    }
    for _ in 0..height {
        program.push(RpnOp::Drop);
    }
    program.push(RpnOp::One);
    Ok(program)
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
        assert!(program.len() <= 5 * soa.len());
        assert_eq!(program.evaluate(), 2.0);
        // Coincide con la evaluación del DAG
        let dag = crate::Dag::new(soa, root);
        assert!(
            (program.evaluate() - dag.evaluate(&bml_domain::EvalContext::new(&[], &[]))).abs()
                < 1e-12
        );
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
        assert!((program.evaluate() - 8.0).abs() < 1e-9);
        let dag = crate::Dag::new(soa, root);
        assert!(
            (program.evaluate() - dag.evaluate(&bml_domain::EvalContext::new(&[], &[]))).abs()
                < 1e-9
        );
    }

    #[test]
    fn shared_subtree_is_evaluated_once() {
        use bml_domain::NodeSoA;
        let mut soa = NodeSoA::new();
        let one = soa.push_one(); // 0: One
        let two = soa.push_bml(one, one); // 1: bml(0, 0)
        let root = soa.push_bml(two, two); // 2: bml(1, 1)
        let program = linearize(&soa, root);
        assert_eq!(
            program.ops.iter().filter(|op| **op == RpnOp::Bml).count(),
            2
        );
        assert!((program.evaluate() - 3.0).abs() < 1e-9);
    }

    #[test]
    fn rpn_matches_dag_evaluation() {
        let mut t = BMLTransformer::new();
        let two = t.two();
        let two2 = t.two();
        let three = t.bml(two, two2);
        let eight = t.exp2(three);
        let log = t.log2(eight); // log2(8) = 3
        let soa = t.into_soa();
        let program = linearize(&soa, log);
        let dag = crate::Dag::new(soa, log);
        let rpn_val = program.evaluate();
        let dag_val = dag.evaluate(&bml_domain::EvalContext::new(&[], &[]));
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
            let mut t = BMLTransformer::new();
            let one = t.one();
            let mut node = t.bml(one, one); // 2
            for _ in 1..depth {
                node = t.bml(node, node);
            }
            let soa = t.into_soa();
            let program = linearize(&soa, node);
            let dag = crate::Dag::new(soa, node);
            let rpn_val = program.evaluate();
            let dag_val = dag.evaluate(&bml_domain::EvalContext::new(&[], &[]));
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
        let mut program = RpnProgram::new();
        program.push(RpnOp::One); // valor inicial
        program.push(RpnOp::Loop {
            count: 3,
            body_len: 2,
        });
        program.push(RpnOp::One);
        program.push(RpnOp::Bml);
        let result = program.evaluate();
        assert!(
            (result - 16.0).abs() < 1e-9,
            "Loop result = {result}, expected 16"
        );
    }

    #[test]
    fn loop_count_zero() {
        let mut program = RpnProgram::new();
        program.push(RpnOp::One); // valor inicial
        program.push(RpnOp::Loop {
            count: 0,
            body_len: 2,
        });
        program.push(RpnOp::One);
        program.push(RpnOp::Bml);
        let result = program.evaluate();
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
        let result = program.evaluate();
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

        let loop_val = loop_prog.evaluate();
        let unrolled_val = unrolled.evaluate();
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

        assert_eq!(loop_prog.len(), m + 3, "Loop program should be M+3 ops");
        assert_eq!(unrolled.len(), n * m + 2, "Unrolled should be N*M+2 ops");
        assert!(loop_prog.len() < unrolled.len(), "Loop should be smaller");
    }

    #[test]
    fn fadd_and_fmul() {
        let mut program = RpnProgram::new();
        program.push(RpnOp::One); // 1
        program.push(RpnOp::One); // 1
        program.push(RpnOp::FAdd); // 2
        program.push(RpnOp::One); // 1
        program.push(RpnOp::FMul); // 2 * 1 = 2
        let result = program.evaluate();
        assert!((result - 2.0).abs() < 1e-9);
    }

    #[test]
    fn pick_and_drop() {
        let mut program = RpnProgram::new();
        program.push(RpnOp::One); // stack: [1]
        program.push(RpnOp::Zero); // stack: [1, 0]
        program.push(RpnOp::Pick { depth: 1 }); // stack: [1, 0, 1]
        program.push(RpnOp::Drop); // stack: [1, 0]
        program.push(RpnOp::Drop); // stack: [1]
        let result = program.evaluate();
        assert!((result - 1.0).abs() < 1e-9);
    }

    #[test]
    fn swap_works() {
        let mut program = RpnProgram::new();
        program.push(RpnOp::One); // stack: [1]
        program.push(RpnOp::Zero); // stack: [1, 0]
        program.push(RpnOp::Swap); // stack: [0, 1]
        program.push(RpnOp::Drop); // stack: [0]
        let result = program.evaluate();
        assert!((result - 0.0).abs() < 1e-9);
    }

    #[test]
    fn nested_loops_two_levels() {
        // Outer: 2 iterations, inner: 3 iterations, inner body: One, One, Bml
        // Each inner iteration: pushes One (1), One (1), Bml(1,1)=2 -> pushes 2
        // 3 inner iters push three 2's each outer iter
        let inner_body: Vec<RpnOp> = vec![RpnOp::One, RpnOp::One, RpnOp::Bml];
        let inner_body_len = inner_body.len() as u32;

        let mut outer_body = Vec::new();
        outer_body.push(RpnOp::Loop {
            count: 3,
            body_len: inner_body_len,
        });
        outer_body.extend(inner_body.iter().copied());

        let mut program = RpnProgram::new();
        program.push(RpnOp::One); // initial value
        program.push(RpnOp::Loop {
            count: 2,
            body_len: outer_body.len() as u32,
        });
        for op in &outer_body {
            program.push(*op);
        }
        let result = program.evaluate();
        // Outer iter 0: inner loop pushes 2,2,2 -> [1, 2, 2, 2]
        // Outer iter 1: inner loop pushes 2,2,2 -> [1, 2, 2, 2, 2, 2, 2]
        // Result = top = 2.0
        assert!((result - 2.0).abs() < 1e-9, "nested loops: {result}");
    }

    #[test]
    #[should_panic(expected = "max 2 loop nesting levels")]
    fn triple_nested_loop_panics() {
        let inner_body: Vec<RpnOp> = vec![RpnOp::One, RpnOp::Bml];
        let inner_body_len = inner_body.len() as u32;

        let mut middle_body = Vec::new();
        middle_body.push(RpnOp::Loop {
            count: 1,
            body_len: inner_body_len,
        });
        middle_body.extend(inner_body.iter().copied());

        let mut outer_body = Vec::new();
        outer_body.push(RpnOp::Loop {
            count: 1,
            body_len: middle_body.len() as u32,
        });
        outer_body.extend(middle_body.iter().copied());

        let mut program = RpnProgram::new();
        program.push(RpnOp::One);
        program.push(RpnOp::Loop {
            count: 1,
            body_len: outer_body.len() as u32,
        });
        for op in &outer_body {
            program.push(*op);
        }
        program.evaluate();
    }

    #[test]
    fn ops_within_nested_loops() {
        // Test FAdd, FMul, Pick, Drop, Swap inside nested loop
        let inner_body: Vec<RpnOp> = vec![
            RpnOp::One,               // push 1
            RpnOp::Swap,              // swap
            RpnOp::Pick { depth: 2 }, // copy from depth 2
            RpnOp::FAdd,              // add
            RpnOp::Drop,              // drop extra
        ];
        let inner_body_len = inner_body.len() as u32;

        let mut outer_body = Vec::new();
        outer_body.push(RpnOp::Loop {
            count: 2,
            body_len: inner_body_len,
        });
        outer_body.extend(inner_body.iter().copied());

        let mut program = RpnProgram::new();
        program.push(RpnOp::One); // initial
        program.push(RpnOp::One); // extra
        program.push(RpnOp::Loop {
            count: 1,
            body_len: outer_body.len() as u32,
        });
        for op in &outer_body {
            program.push(*op);
        }
        let _ = program.evaluate();
    }
}
