//! Hot loop RPN: intérprete iterativo con buffers pre-asignados.
//!
//! Este es el corazón del runtime. Itera secuencialmente sobre el
//! arreglo RPN sin saltos ni recursión, usando una pila pre-asignada.
//!
//! # Cero allocs en hot path
//!
//! La pila se pre-asigna con capacidad suficiente en el constructor.
//! La función [`HotLoop::execute`] no hace ninguna asignación de
//! memoria: solo push/pop sobre la pila pre-asignada.
//!
//! # Hot loop < 32 KB
//!
//! El cuerpo del loop en `execute` es deliberadamente minimalista:
//! un `match` sobre `RpnOp` con tres variantes. El tamaño del código
//! compilado debe ser < 32 KB para caber en L1i.

use bml_compiler::{Fragment, RpnOp, RpnProgram};

/// Intérprete RPN con pila pre-asignada.
///
/// El `HotLoop` se crea una sola vez al arrancar el runtime. La pila
/// se pre-asigna con capacidad suficiente para el programa más grande
/// esperado. La función `execute` no hace allocs.
pub struct HotLoop {
    /// Pila pre-asignada. Se reutiliza entre ejecuciones.
    stack: Vec<f64>,
}

impl HotLoop {
    /// Crea un `HotLoop` con la capacidad de pila dada.
    ///
    /// La pila se pre-asigna y nunca crece durante `execute`.
    pub fn with_capacity(stack_capacity: usize) -> Self {
        let mut stack = Vec::with_capacity(stack_capacity);
        // Pre-asignar y dejar vacío; los push/pop en execute no allocan.
        // Reservamos el espacio exacto para que Vec no reasigne.
        stack.resize(stack_capacity, 0.0);
        stack.clear();
        Self { stack }
    }

    /// Ejecuta un programa RPN completo.
    ///
    /// # Cero allocs
    ///
    /// Esta función no hace ninguna asignación de memoria. La pila
    /// ya está pre-asignada. Se limpia al inicio y se reutiliza.
    ///
    /// # Hot loop
    ///
    /// El loop principal itera sobre `program.ops` con un `match`
    /// sobre `RpnOp`. El código compilado debe ser < 32 KB.
    #[inline]
    pub fn execute(&mut self, program: &RpnProgram, x: f64) -> f64 {
        let ctx = bml_domain::EvalContext::new(&[], &[]);
        let mut buf = crate::buffer::ResultBuffer::new(0, 0);
        self.execute_full(program, &ctx, &mut buf)
    }

    /// Ejecuta un programa RPN con contexto de inputs y pesos.
    #[inline]
    pub fn execute_with_ctx(&mut self, program: &RpnProgram, ctx: &bml_domain::EvalContext) -> f64 {
        let mut buf = crate::buffer::ResultBuffer::new(0, 0);
        self.execute_full(program, ctx, &mut buf)
    }

    /// Ejecuta un programa RPN con contexto, buffer de resultados, y pool de pesos.
    #[inline]
    pub fn execute_full(
        &mut self,
        program: &RpnProgram,
        ctx: &bml_domain::EvalContext,
        buf: &mut crate::buffer::ResultBuffer,
    ) -> f64 {
        self.stack.clear();

        let mut i = 0;
        while i < program.ops.len() {
            match program.ops[i] {
                RpnOp::One => self.stack.push(1.0),
                RpnOp::Zero => self.stack.push(0.0),
                RpnOp::Var(id) => self.stack.push(ctx.get_var(id)),
                RpnOp::Const(id) => self.stack.push(ctx.get_const(id)),
                RpnOp::VarIndexed { base } => {
                    let offset = self.stack.pop().unwrap_or(0.0) as u32;
                    self.stack.push(buf.read_indexed(base, offset));
                }
                RpnOp::StoreResult { slot } => {
                    let offset = self.stack.pop().unwrap_or(0.0) as u32;
                    let value = self.stack.pop().unwrap_or(f64::NAN);
                    buf.write(slot, offset, value);
                }
                RpnOp::Bml => {
                    let len = self.stack.len();
                    let b = self.stack[len - 1];
                    let a = self.stack[len - 2];
                    self.stack.truncate(len - 2);
                    self.stack.push(bml_domain::bml(a, b));
                }
                RpnOp::Dup => {
                    let len = self.stack.len();
                    let v = self.stack[len - 1];
                    self.stack.push(v);
                }
                RpnOp::Loop { count, body_len } => {
                    let body_start = i + 1;
                    let body_end = body_start + body_len as usize;
                    for _ in 0..count {
                        let mut j = body_start;
                        while j < body_end {
                            match program.ops[j] {
                                RpnOp::One => self.stack.push(1.0),
                                RpnOp::Zero => self.stack.push(0.0),
                                RpnOp::Var(id) => self.stack.push(ctx.get_var(id)),
                                RpnOp::Const(id) => self.stack.push(ctx.get_const(id)),
                                RpnOp::VarIndexed { base } => {
                                    let offset = self.stack.pop().unwrap_or(0.0) as u32;
                                    self.stack.push(buf.read_indexed(base, offset));
                                }
                                RpnOp::StoreResult { slot } => {
                                    let offset = self.stack.pop().unwrap_or(0.0) as u32;
                                    let value = self.stack.pop().unwrap_or(f64::NAN);
                                    buf.write(slot, offset, value);
                                }
                                RpnOp::Bml => {
                                    let len = self.stack.len();
                                    let b = self.stack[len - 1];
                                    let a = self.stack[len - 2];
                                    self.stack.truncate(len - 2);
                                    self.stack.push(bml_domain::bml(a, b));
                                }
                                RpnOp::Dup => {
                                    let len = self.stack.len();
                                    let v = self.stack[len - 1];
                                    self.stack.push(v);
                                }
                                RpnOp::Loop { .. } => {
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

        self.stack.pop().unwrap_or(f64::NAN)
    }

    /// Ejecuta un fragmento sobre la pila actual (sin limpiar).
    ///
    /// A diferencia de `execute`, esta función no limpia la pila al
    /// inicio. Permite ejecutar fragmentos consecutivos manteniendo
    /// el estado de la pila entre ellos.
    #[inline]
    pub fn execute_fragment(&mut self, fragment: &Fragment) {
        let ctx = bml_domain::EvalContext::new(&[], &[]);
        let mut buf = crate::buffer::ResultBuffer::new(0, 0);
        self.execute_fragment_full(fragment, &ctx, &mut buf);
    }

    /// Ejecuta un fragmento con contexto de inputs y pesos.
    #[inline]
    pub fn execute_fragment_with_ctx(
        &mut self,
        fragment: &Fragment,
        ctx: &bml_domain::EvalContext,
    ) {
        let mut buf = crate::buffer::ResultBuffer::new(0, 0);
        self.execute_fragment_full(fragment, ctx, &mut buf);
    }

    /// Ejecuta un fragmento con contexto, buffer, y pool completos.
    #[inline]
    pub fn execute_fragment_full(
        &mut self,
        fragment: &Fragment,
        ctx: &bml_domain::EvalContext,
        buf: &mut crate::buffer::ResultBuffer,
    ) {
        let mut i = 0;
        while i < fragment.ops.len() {
            match fragment.ops[i] {
                RpnOp::One => self.stack.push(1.0),
                RpnOp::Zero => self.stack.push(0.0),
                RpnOp::Var(id) => self.stack.push(ctx.get_var(id)),
                RpnOp::Const(id) => self.stack.push(ctx.get_const(id)),
                RpnOp::VarIndexed { base } => {
                    let offset = self.stack.pop().unwrap_or(0.0) as u32;
                    self.stack.push(buf.read_indexed(base, offset));
                }
                RpnOp::StoreResult { slot } => {
                    let offset = self.stack.pop().unwrap_or(0.0) as u32;
                    let value = self.stack.pop().unwrap_or(f64::NAN);
                    buf.write(slot, offset, value);
                }
                RpnOp::Bml => {
                    let len = self.stack.len();
                    let b = self.stack[len - 1];
                    let a = self.stack[len - 2];
                    self.stack.truncate(len - 2);
                    self.stack.push(bml_domain::bml(a, b));
                }
                RpnOp::Dup => {
                    let len = self.stack.len();
                    let v = self.stack[len - 1];
                    self.stack.push(v);
                }
                RpnOp::Loop { count, body_len } => {
                    let body_start = i + 1;
                    let body_end = body_start + body_len as usize;
                    for _ in 0..count {
                        let mut j = body_start;
                        while j < body_end {
                            match fragment.ops[j] {
                                RpnOp::One => self.stack.push(1.0),
                                RpnOp::Zero => self.stack.push(0.0),
                                RpnOp::Var(id) => self.stack.push(ctx.get_var(id)),
                                RpnOp::Const(id) => self.stack.push(ctx.get_const(id)),
                                RpnOp::VarIndexed { base } => {
                                    let offset = self.stack.pop().unwrap_or(0.0) as u32;
                                    self.stack.push(buf.read_indexed(base, offset));
                                }
                                RpnOp::StoreResult { slot } => {
                                    let offset = self.stack.pop().unwrap_or(0.0) as u32;
                                    let value = self.stack.pop().unwrap_or(f64::NAN);
                                    buf.write(slot, offset, value);
                                }
                                RpnOp::Bml => {
                                    let len = self.stack.len();
                                    let b = self.stack[len - 1];
                                    let a = self.stack[len - 2];
                                    self.stack.truncate(len - 2);
                                    self.stack.push(bml_domain::bml(a, b));
                                }
                                RpnOp::Dup => {
                                    let len = self.stack.len();
                                    let v = self.stack[len - 1];
                                    self.stack.push(v);
                                }
                                RpnOp::Loop { .. } => {
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
    }

    /// Ejecuta una lista de fragmentos en orden.
    ///
    /// La pila se mantiene entre fragmentos.
    #[inline]
    pub fn execute_fragments(&mut self, fragments: &[Fragment], x: f64) -> f64 {
        let ctx = bml_domain::EvalContext::new(&[], &[]);
        let mut buf = crate::buffer::ResultBuffer::new(0, 0);
        self.execute_fragments_full(fragments, &ctx, &mut buf)
    }

    /// Ejecuta una lista de fragmentos con contexto.
    #[inline]
    pub fn execute_fragments_with_ctx(
        &mut self,
        fragments: &[Fragment],
        ctx: &bml_domain::EvalContext,
    ) -> f64 {
        let mut buf = crate::buffer::ResultBuffer::new(0, 0);
        self.execute_fragments_full(fragments, ctx, &mut buf)
    }

    /// Ejecuta una lista de fragmentos con contexto y buffer completos.
    #[inline]
    pub fn execute_fragments_full(
        &mut self,
        fragments: &[Fragment],
        ctx: &bml_domain::EvalContext,
        buf: &mut crate::buffer::ResultBuffer,
    ) -> f64 {
        self.stack.clear();
        for fragment in fragments {
            self.execute_fragment_full(fragment, ctx, buf);
        }
        self.stack.pop().unwrap_or(f64::NAN)
    }

    /// Profundidad actual de la pila.
    pub fn stack_depth(&self) -> usize {
        self.stack.len()
    }

    /// Capacidad de la pila.
    pub fn stack_capacity(&self) -> usize {
        self.stack.capacity()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bml_compiler::{linearize, HashConsRegistry};
    use bml_domain::BMLTransformer;

    fn build_two_program() -> RpnProgram {
        let mut t = BMLTransformer::new();
        let root = t.two();
        let soa = t.into_soa();
        linearize(&soa, root)
    }

    fn build_exp2_program() -> RpnProgram {
        // 2^3 = 8, donde 3 = bml(2, 2)
        let mut t = BMLTransformer::new();
        let two = t.two();
        let two2 = t.two();
        let three = t.bml(two, two2);
        let root = t.exp2(three);
        let soa = t.into_soa();
        linearize(&soa, root)
    }

    #[test]
    fn execute_two() {
        let program = build_two_program();
        let mut hot = HotLoop::with_capacity(256);
        let result = hot.execute(&program, 0.0);
        assert!((result - 2.0).abs() < 1e-12);
    }

    #[test]
    fn execute_exp2() {
        let program = build_exp2_program();
        let mut hot = HotLoop::with_capacity(256);
        let result = hot.execute(&program, 0.0);
        assert!((result - 8.0).abs() < 1e-9);
    }

    #[test]
    fn matches_rpn_program_evaluate() {
        let program = build_exp2_program();
        let mut hot = HotLoop::with_capacity(256);
        let hot_result = hot.execute(&program, 0.0);
        let rpn_result = program.evaluate(0.0);
        assert_eq!(hot_result.to_bits(), rpn_result.to_bits());
    }

    #[test]
    fn zero_allocs_in_hot_path() {
        // Verificamos que la pila no crece durante execute.
        let program = build_exp2_program();
        let mut hot = HotLoop::with_capacity(256);
        let cap_before = hot.stack_capacity();
        for _ in 0..1000 {
            hot.execute(&program, 0.0);
        }
        let cap_after = hot.stack_capacity();
        assert_eq!(cap_before, cap_after, "la pila crecio durante execute");
    }

    #[test]
    fn stack_cleared_between_executions() {
        let program = build_two_program();
        let mut hot = HotLoop::with_capacity(256);
        hot.execute(&program, 0.0);
        assert_eq!(hot.stack_depth(), 0);
        hot.execute(&program, 0.0);
        assert_eq!(hot.stack_depth(), 0);
    }

    #[test]
    fn execute_fragments_preserves_stack() {
        // Construir un programa y partirlo en fragmentos.
        use bml_compiler::{fragment_program, DEFAULT_L1_THRESHOLD};
        let program = build_exp2_program();
        let graph = fragment_program(&program, DEFAULT_L1_THRESHOLD);
        let mut hot = HotLoop::with_capacity(256);
        let result = hot.execute_fragments(&graph.fragments, 0.0);
        assert!((result - 8.0).abs() < 1e-9);
    }

    #[test]
    fn large_program_no_panic() {
        let mut reg = HashConsRegistry::new();
        let one = reg.one();
        let two = reg.bml(one, one);
        let mut node = two;
        for _ in 0..1000 {
            node = reg.bml(node, two);
        }
        let soa = reg.into_soa();
        let program = linearize(&soa, node);
        let mut hot = HotLoop::with_capacity(4096);
        let _ = hot.execute(&program, 0.0);
    }
}
