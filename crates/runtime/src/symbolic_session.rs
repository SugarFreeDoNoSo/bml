//! Explicit per-connection state for a symbolic causal layer range.
use crate::symbolic_causal::SymbolicCausalModel;
use bml_compiler::HashConsedKV;
use std::ops::Range;

/// One independently owned model and F32 KV cache for a contiguous layer range.
///
/// Preparation writes only an uncommitted KV row. A coordinator must commit
/// exactly once after all ranges and the output head succeed. Abort permits a
/// retry at the same position; reset allocates a fresh cache before replacing
/// the old session state. Lost commit responses still require invalidating the
/// distributed pipeline: this local object cannot establish remote agreement.
pub struct SymbolicLayerSession {
    model: SymbolicCausalModel,
    cache: HashConsedKV,
    layers: Range<u32>,
    context: u32,
    max_kv_bytes: usize,
    prepared: bool,
}

impl SymbolicLayerSession {
    /// Admit a range and allocate bounded KV only for its layers.
    pub fn new(
        model: SymbolicCausalModel,
        layers: Range<u32>,
        context: u32,
        max_kv_bytes: usize,
    ) -> Result<Self, String> {
        if layers.start >= layers.end
            || layers.end > model.compiler().config().n_layers
            || context == 0
            || context > model.context_limit()
        {
            return Err("invalid symbolic layer session range or context".into());
        }
        let cache = Self::allocate(&model, layers.clone(), context, max_kv_bytes)?;
        Ok(Self {
            model,
            cache,
            layers,
            context,
            max_kv_bytes,
            prepared: false,
        })
    }

    fn allocate(
        model: &SymbolicCausalModel,
        layers: Range<u32>,
        context: u32,
        max_kv_bytes: usize,
    ) -> Result<HashConsedKV, String> {
        HashConsedKV::try_new_layer_range(
            layers,
            model.compiler().n_kv_heads(),
            model.compiler().head_dim(),
            context,
            max_kv_bytes,
        )
    }

    /// Confirmed token position, excluding any prepared row.
    pub fn position(&self) -> u32 {
        self.cache.current_pos()
    }
    /// Configured contiguous layer range.
    pub fn layers(&self) -> Range<u32> {
        self.layers.clone()
    }
    /// Whether a successful row awaits commit or abort.
    pub fn is_prepared(&self) -> bool {
        self.prepared
    }
    /// Read-only model metadata and tokenizer for an owning coordinator.
    pub fn model(&self) -> &SymbolicCausalModel {
        &self.model
    }

    /// Execute one position without advancing it. Bad input or a failed BML
    /// leaves the committed position unchanged and allows a fresh preparation.
    pub fn prepare(&mut self, position: u32, hidden: &[f64]) -> Result<Vec<f64>, String> {
        if self.prepared || position != self.position() || position >= self.context {
            return Err("symbolic layer session position or pending state".into());
        }
        let output = self.model.forward_layer_range(
            hidden,
            position,
            &mut self.cache,
            self.layers.clone(),
        )?;
        if output.iter().any(|v| !v.is_finite()) {
            return Err("nonfinite symbolic layer output".into());
        }
        self.prepared = true;
        Ok(output)
    }

    /// Confirm the prepared row at exactly the supplied current position.
    pub fn commit(&mut self, position: u32) -> Result<(), String> {
        if !self.prepared || position != self.position() {
            return Err("no matching symbolic row to commit".into());
        }
        self.cache.advance();
        self.prepared = false;
        Ok(())
    }

    /// Discard a prepared row logically; its physical cells are overwritten on retry.
    pub fn abort(&mut self, position: u32) -> Result<(), String> {
        if !self.prepared || position != self.position() {
            return Err("no matching symbolic row to abort".into());
        }
        self.prepared = false;
        Ok(())
    }

    /// Replace all KV state with a fresh empty cache, preserving the loaded model.
    pub fn reset(&mut self) -> Result<(), String> {
        let fresh = Self::allocate(
            &self.model,
            self.layers.clone(),
            self.context,
            self.max_kv_bytes,
        )?;
        self.cache = fresh;
        self.prepared = false;
        Ok(())
    }
}
