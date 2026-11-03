//! Validated dense Llama models and a causal CPU execution session.
//!
//! Stored tensors retain their GGUF encoding. A projection decodes one row at
//! a time; activations and KV use f32, with f64 scalar reduction accumulators.

use crate::tokenizer::{Tokenizer, TokenizerSpec};
use crate::{Error, Result, Tensor};
use bml_gguf::{decode_f32, GgufDataType, GgufMetadataValue, GgufReader};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

/// Default cap on one session's numerical KV and transient payloads (512 MiB).
/// Model storage, tokenizer, allocator overhead and caller-owned outputs are
/// additional. Use [`Model::session_with_memory_limit`] to set an explicit cap.
pub const DEFAULT_SESSION_MEMORY_LIMIT: usize = 512 * 1024 * 1024;

/// The supported dense Llama profile, independent of a GGUF file or tokenizer.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Config {
    /// Width of the hidden state and token embeddings.
    pub embedding_length: usize,
    /// Number of transformer blocks.
    pub block_count: usize,
    /// Number of query heads.
    pub head_count: usize,
    /// Number of shared key/value heads; divides `head_count`.
    pub head_count_kv: usize,
    /// Intermediate width of each gated feed-forward network.
    pub feed_forward_length: usize,
    /// Maximum number of tokens in one session.
    pub context_length: usize,
    /// Number of vocabulary entries and output logits.
    pub vocab_size: usize,
    /// Rotary dimensions per head; this profile requires the complete head.
    pub rope_dimension_count: usize,
    /// Positive base of unscaled rotary position frequencies.
    pub rope_freq_base: f32,
    /// Positive epsilon added to the mean square in every RMSNorm.
    pub rms_norm_eps: f32,
}

impl Config {
    /// Validate dimensions, grouped attention, and numerical parameters.
    pub fn validate(&self) -> Result<()> {
        if [
            self.embedding_length,
            self.block_count,
            self.head_count,
            self.head_count_kv,
            self.feed_forward_length,
            self.context_length,
            self.vocab_size,
        ]
        .contains(&0)
        {
            return Err(Error("Llama dimensions and counts must be nonzero".into()));
        }
        if !self.embedding_length.is_multiple_of(self.head_count)
            || !self.head_count.is_multiple_of(self.head_count_kv)
        {
            return Err(Error(
                "incompatible embedding, query or KV head counts".into(),
            ));
        }
        let head_dim = self.embedding_length / self.head_count;
        if !head_dim.is_multiple_of(2) || self.rope_dimension_count != head_dim {
            return Err(Error(
                "only even, full-head, unscaled Llama RoPE is supported".into(),
            ));
        }
        if !self.rope_freq_base.is_finite()
            || self.rope_freq_base <= 0.0
            || !self.rms_norm_eps.is_finite()
            || self.rms_norm_eps <= 0.0
        {
            return Err(Error(
                "RoPE base and RMSNorm epsilon must be finite and positive".into(),
            ));
        }
        self.embedding_length
            .checked_mul(self.vocab_size)
            .and_then(|_| self.embedding_length.checked_mul(self.feed_forward_length))
            .and_then(|_| self.block_count.checked_mul(self.context_length))
            .and_then(|n| n.checked_mul(self.kv_width()))
            .and_then(|n| n.checked_mul(2 * std::mem::size_of::<f32>()))
            .ok_or_else(|| Error("model dimensions overflow addressable storage".into()))?;
        Ok(())
    }

    /// Width of one query, key or value head, or zero for an invalid zero count.
    /// Call [`Self::validate`] before using an externally constructed profile.
    pub fn head_dimension(&self) -> usize {
        self.embedding_length
            .checked_div(self.head_count)
            .unwrap_or(0)
    }

    fn kv_width(&self) -> usize {
        self.head_dimension() * self.head_count_kv
    }
}

/// An operation in the model's fixed execution plan.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum Operation {
    /// Look up one contiguous embedding row.
    Embedding,
    /// Normalize the attention input without changing its residual.
    AttentionRmsNorm(usize),
    /// Project normalized input to query, key and value heads.
    QueryKeyValue(usize),
    /// Rotate query and key pairs at the current absolute position.
    Rope(usize),
    /// Append KV and compute causal grouped-query attention.
    CausalAttention(usize),
    /// Project the concatenated attention heads.
    AttentionOutput(usize),
    /// Add the original block input to projected attention.
    AttentionResidual(usize),
    /// Normalize the feed-forward input, retaining its residual.
    FeedForwardRmsNorm(usize),
    /// Multiply SiLU of the gate projection by the up projection.
    SwiGlu(usize),
    /// Project the gated intermediate state back to embedding width.
    FeedForwardDown(usize),
    /// Add the attention residual to the feed-forward output.
    FeedForwardResidual(usize),
    /// Apply the learned final RMSNorm.
    OutputRmsNorm,
    /// Project to vocabulary logits, optionally using tied embedding weights.
    Logits,
}

/// Numeric payload sizes for a model and one execution session.
///
/// These are logical byte counts, not an RSS prediction. Allocator capacity,
/// vector headers, tokenizer, metadata and mapped-file padding are excluded.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MemoryPlan {
    /// Sum of encoded tensor payload lengths; quantized rows stay compressed.
    pub encoded_weight_bytes: usize,
    /// Exact scalar payload of the preallocated F32 key and value caches.
    pub kv_bytes: usize,
    /// Largest decoded row buffer, released before the next row is decoded.
    pub decoded_row_bytes: usize,
    /// Conservative upper bound on simultaneous transient scalar payloads.
    /// Includes one output logit vector and F64 attention scores.
    pub scratch_bytes_upper_bound: usize,
}

#[derive(Debug)]
struct Layer {
    attention_norm: usize,
    query: usize,
    key: usize,
    value: usize,
    attention_output: usize,
    feed_forward_norm: usize,
    gate: usize,
    up: usize,
    down: usize,
}

/// An immutable model borrowing its compressed, validated tensor resources.
pub struct Model<'a> {
    config: Config,
    tokenizer: Tokenizer,
    tensors: Vec<Tensor<'a>>,
    layers: Vec<Layer>,
    embedding: usize,
    output_norm: usize,
    output: usize,
    plan: Vec<Operation>,
    fingerprint: std::sync::OnceLock<[u8; 32]>,
}

impl<'a> Model<'a> {
    /// Import the supported dense Llama profile without expanding all weights.
    ///
    /// Unknown tensor roles, scaling variants, non-finite weights, missing
    /// resources and shape mismatches are errors, never silent defaults.
    pub fn from_gguf(reader: &'a GgufReader) -> Result<Self> {
        let (config, tokenizer) = Self::profile_metadata(reader)?;
        let tensors = reader
            .tensor_infos()
            .iter()
            .map(|info| {
                let view = reader.tensor_view(&info.name)?;
                Ok(Tensor {
                    name: info.name.clone(),
                    dims: view.ggml_dims().to_vec(),
                    dtype: view.dtype(),
                    data: view.bytes(),
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Self::from_tensors(config, tokenizer, tensors)
    }

    /// Validate architecture/configuration/tokenizer independently of weight storage.
    pub fn profile_metadata(reader: &impl crate::ModelMetadata) -> Result<(Config, TokenizerSpec)> {
        if reader.architecture() != Some("llama") {
            return Err(Error(
                "unsupported architecture: expected dense llama".into(),
            ));
        }
        reject_variants(reader)?;
        let tokenizer = TokenizerSpec::from_metadata(reader)?;
        let embedding_length = required_usize(reader, "llama.embedding_length")?;
        let head_count = required_usize(reader, "llama.attention.head_count")?;
        if head_count == 0 {
            return Err(Error("llama.attention.head_count must be nonzero".into()));
        }
        let head_dim = embedding_length / head_count;
        let config = Config {
            embedding_length,
            block_count: required_usize(reader, "llama.block_count")?,
            head_count,
            head_count_kv: optional_usize(reader, "llama.attention.head_count_kv")?
                .unwrap_or(head_count),
            feed_forward_length: required_usize(reader, "llama.feed_forward_length")?,
            context_length: required_usize(reader, "llama.context_length")?,
            vocab_size: optional_usize(reader, "llama.vocab_size")?
                .unwrap_or(tokenizer.tokens.len()),
            rope_dimension_count: optional_usize(reader, "llama.rope.dimension_count")?
                .unwrap_or(head_dim),
            rope_freq_base: optional_float(reader, "llama.rope.freq_base")?.unwrap_or(10000.0),
            rms_norm_eps: optional_float(reader, "llama.attention.layer_norm_rms_epsilon")?
                .ok_or_else(|| Error("missing llama.attention.layer_norm_rms_epsilon".into()))?,
        };
        config.validate()?;
        for key in ["llama.attention.key_length", "llama.attention.value_length"] {
            if optional_usize(reader, key)?.is_some_and(|value| value != head_dim) {
                return Err(Error(format!("unsupported {key}: expected {head_dim}")));
            }
        }
        Ok((config, tokenizer))
    }

    /// Validate a normalized resource manifest and borrow its tensor payloads.
    pub fn from_tensors(
        config: Config,
        tokenizer: TokenizerSpec,
        tensors: Vec<Tensor<'a>>,
    ) -> Result<Self> {
        config.validate()?;
        let tokenizer = Tokenizer::new(tokenizer)?;
        if tokenizer.vocab_size() != config.vocab_size {
            return Err(Error(
                "configuration and tokenizer vocabulary sizes differ".into(),
            ));
        }
        let mut indices = HashMap::new();
        for (index, tensor) in tensors.iter().enumerate() {
            if indices.insert(tensor.name.as_str(), index).is_some() {
                return Err(Error(format!("duplicate tensor {}", tensor.name)));
            }
        }
        let mut used = HashSet::new();
        let mut require = |name: &str, shape: &[usize]| -> Result<usize> {
            let index = *indices
                .get(name)
                .ok_or_else(|| Error(format!("missing tensor {name}")))?;
            let expected: Vec<u64> = shape.iter().map(|&n| n as u64).collect();
            if tensors[index].dims != expected {
                return Err(Error(format!(
                    "tensor {name} shape {:?}, expected {expected:?}",
                    tensors[index].dims
                )));
            }
            validate_tensor(&tensors[index])?;
            used.insert(index);
            Ok(index)
        };
        let width = config.embedding_length;
        let kv_width = config.kv_width();
        let ff = config.feed_forward_length;
        let embedding = require("token_embd.weight", &[width, config.vocab_size])?;
        let output_norm = require("output_norm.weight", &[width])?;
        let output = if indices.contains_key("output.weight") {
            require("output.weight", &[width, config.vocab_size])?
        } else {
            embedding
        };
        let mut layers = Vec::new();
        let mut plan = vec![Operation::Embedding];
        for layer in 0..config.block_count {
            let mut weight =
                |role: &str, shape: &[usize]| require(&format!("blk.{layer}.{role}.weight"), shape);
            layers.push(Layer {
                attention_norm: weight("attn_norm", &[width])?,
                query: weight("attn_q", &[width, width])?,
                key: weight("attn_k", &[width, kv_width])?,
                value: weight("attn_v", &[width, kv_width])?,
                attention_output: weight("attn_output", &[width, width])?,
                feed_forward_norm: weight("ffn_norm", &[width])?,
                gate: weight("ffn_gate", &[width, ff])?,
                up: weight("ffn_up", &[width, ff])?,
                down: weight("ffn_down", &[ff, width])?,
            });
            plan.extend([
                Operation::AttentionRmsNorm(layer),
                Operation::QueryKeyValue(layer),
                Operation::Rope(layer),
                Operation::CausalAttention(layer),
                Operation::AttentionOutput(layer),
                Operation::AttentionResidual(layer),
                Operation::FeedForwardRmsNorm(layer),
                Operation::SwiGlu(layer),
                Operation::FeedForwardDown(layer),
                Operation::FeedForwardResidual(layer),
            ]);
        }
        if let Some((_, tensor)) = tensors.iter().enumerate().find(|(i, _)| !used.contains(i)) {
            return Err(Error(format!("unsupported tensor role {}", tensor.name)));
        }
        plan.extend([Operation::OutputRmsNorm, Operation::Logits]);
        Ok(Self {
            config,
            tokenizer,
            tensors,
            layers,
            embedding,
            output_norm,
            output,
            plan,
            fingerprint: std::sync::OnceLock::new(),
        })
    }

    /// SHA-256 identity of numerical configuration and encoded tensor resources.
    /// Computed once on demand, in canonical tensor-name order. Includes encoding
    /// and dimensions; excludes tokenizer metadata and file layout. This detects
    /// accidental deployment mismatches, not malicious/unauthenticated servers.
    pub fn fingerprint(&self) -> [u8; 32] {
        *self.fingerprint.get_or_init(|| {
            use sha2::{Digest, Sha256};
            let mut hash = Sha256::new();
            hash.update(b"BML-numerical-model-v1\0");
            let c = &self.config;
            for value in [
                c.embedding_length,
                c.block_count,
                c.head_count,
                c.head_count_kv,
                c.feed_forward_length,
                c.context_length,
                c.vocab_size,
                c.rope_dimension_count,
            ] {
                hash.update((value as u64).to_le_bytes());
            }
            hash.update(c.rope_freq_base.to_bits().to_le_bytes());
            hash.update(c.rms_norm_eps.to_bits().to_le_bytes());
            let mut tensors: Vec<_> = self.tensors.iter().collect();
            tensors.sort_unstable_by(|a, b| a.name.cmp(&b.name));
            hash.update((tensors.len() as u64).to_le_bytes());
            for tensor in tensors {
                hash.update((tensor.name.len() as u64).to_le_bytes());
                hash.update(tensor.name.as_bytes());
                hash.update((tensor.dtype as u32).to_le_bytes());
                hash.update((tensor.dims.len() as u64).to_le_bytes());
                for dimension in &tensor.dims {
                    hash.update(dimension.to_le_bytes());
                }
                hash.update((tensor.data.len() as u64).to_le_bytes());
                hash.update(tensor.data);
            }
            hash.finalize().into()
        })
    }

    /// Validated architectural and numerical configuration.
    pub fn config(&self) -> &Config {
        &self.config
    }

    /// Exact tokenizer carried by the model resources.
    pub fn tokenizer(&self) -> &Tokenizer {
        &self.tokenizer
    }

    /// Compressed tensor resources in their original import order.
    pub fn tensors(&self) -> &[Tensor<'a>] {
        &self.tensors
    }

    /// Ordered tensor operations performed for each token.
    pub fn plan(&self) -> &[Operation] {
        &self.plan
    }

    /// Derive compressed storage, KV and transient numerical storage sizes.
    pub fn memory_plan(&self, context: usize) -> Result<MemoryPlan> {
        if context == 0 || context > self.config.context_length {
            return Err(Error(
                "session context must be within the model context limit".into(),
            ));
        }
        let calculate = || -> Option<MemoryPlan> {
            let width = self.config.embedding_length;
            let ff = self.config.feed_forward_length;
            let kv_width = self.config.kv_width();
            let row = width.max(ff);
            let encoded_weight_bytes = self
                .tensors
                .iter()
                .try_fold(0_usize, |sum, tensor| sum.checked_add(tensor.data.len()))?;
            let kv_bytes = self
                .config
                .block_count
                .checked_mul(context)?
                .checked_mul(kv_width)?
                .checked_mul(8)?;
            // Attention and FFN use separate scopes, so their temporary vectors
            // cannot overlap. Include a row decoder and scores conservatively.
            let attention = width
                .checked_mul(5)?
                .checked_add(kv_width.checked_mul(2)?)?;
            let feed_forward = width.checked_mul(3)?.checked_add(ff.checked_mul(2)?)?;
            let logits = width.checked_mul(2)?.checked_add(self.config.vocab_size)?;
            let transient = attention.max(feed_forward).max(logits).checked_add(row)?;
            Some(MemoryPlan {
                encoded_weight_bytes,
                kv_bytes,
                decoded_row_bytes: row.checked_mul(4)?,
                scratch_bytes_upper_bound: transient
                    .checked_mul(4)?
                    .checked_add(context.checked_mul(8)?)?,
            })
        };
        calculate().ok_or_else(|| Error("memory plan exceeds addressable byte counts".into()))
    }

    /// Create an independent causal session with bounded, non-wrapping KV.
    /// Uses [`DEFAULT_SESSION_MEMORY_LIMIT`] for its numerical payload budget.
    pub fn session(&self, context: usize) -> Result<Session<'_>> {
        self.session_with_memory_limit(context, DEFAULT_SESSION_MEMORY_LIMIT)
    }

    /// Create a session after checking KV and scratch against a caller budget.
    ///
    /// Oversized requests fail before cache allocation. Allocations are fallible,
    /// but a successful reservation cannot guarantee physical memory availability
    /// on an operating system that overcommits memory.
    pub fn session_with_memory_limit(
        &self,
        context: usize,
        maximum_bytes: usize,
    ) -> Result<Session<'_>> {
        let memory = self.memory_plan(context)?;
        let requested = memory
            .kv_bytes
            .checked_add(memory.scratch_bytes_upper_bound)
            .ok_or_else(|| Error("session memory size overflow".into()))?;
        if requested > maximum_bytes {
            return Err(Error(format!(
                "session needs at most {requested} numerical bytes, budget is {maximum_bytes}"
            )));
        }
        let per_layer = context
            .checked_mul(self.config.kv_width())
            .ok_or_else(|| Error("KV allocation size overflow".into()))?;
        let mut keys = Vec::new();
        let mut values = Vec::new();
        keys.try_reserve_exact(self.config.block_count)
            .map_err(|_| Error("KV allocation failed".into()))?;
        values
            .try_reserve_exact(self.config.block_count)
            .map_err(|_| Error("KV allocation failed".into()))?;
        for _ in 0..self.config.block_count {
            keys.push(zeros(per_layer)?);
            values.push(zeros(per_layer)?);
        }
        Ok(Session {
            model: self,
            keys,
            values,
            position: 0,
            context,
        })
    }

    /// Decode one validated token embedding for a distributed layer pipeline.
    pub fn embed_token(&self, token: u32) -> Result<Vec<f32>> {
        if token as usize >= self.config.vocab_size {
            return Err(Error("token ID outside model vocabulary".into()));
        }
        self.row(self.embedding, token as usize)
    }

    /// Apply final normalization and vocabulary projection after all layers.
    /// The caller is responsible for having executed the complete layer plan.
    pub fn final_logits(&self, hidden: &[f32]) -> Result<Vec<f32>> {
        if hidden.len() != self.config.embedding_length {
            return Err(Error("hidden state width mismatch".into()));
        }
        finite(hidden, "final hidden state")?;
        let normalized = self.normalize(self.output_norm, hidden)?;
        self.project(self.output, &normalized)
    }

    /// Create an independent causal session for one fixed, nonempty layer range.
    /// Currently retains this full model and allocates full-model KV under the
    /// supplied budget; this API does not claim memory sharding of model weights.
    pub fn layer_session(
        &self,
        layers: std::ops::Range<usize>,
        context: usize,
        maximum_bytes: usize,
    ) -> Result<LayerSession<'_>> {
        if layers.start >= layers.end || layers.end > self.config.block_count {
            return Err(Error("invalid layer session range".into()));
        }
        Ok(LayerSession {
            session: self.session_with_memory_limit(context, maximum_bytes)?,
            layers,
        })
    }

    fn row(&self, tensor: usize, row: usize) -> Result<Vec<f32>> {
        decode_row(&self.tensors[tensor], row)
    }

    fn project(&self, tensor: usize, input: &[f32]) -> Result<Vec<f32>> {
        let resource = &self.tensors[tensor];
        if input.len() != resource.dims[0] as usize {
            return Err(Error(format!(
                "projection input shape mismatch: {}",
                resource.name
            )));
        }
        let mut output = zeros(resource.dims[1] as usize)?;
        for (row, result) in output.iter_mut().enumerate() {
            *result = if resource.dtype == GgufDataType::F32 {
                // F32 needs no intermediate row allocation or copy.
                encoded_row(resource, row)?
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .zip(input)
                    .map(|(bytes, &x)| f64::from(f32::from_le_bytes(*bytes)) * f64::from(x))
                    .sum::<f64>() as f32
            } else {
                let weights = self.row(tensor, row)?;
                input
                    .iter()
                    .zip(weights)
                    .map(|(&x, w)| f64::from(x) * f64::from(w))
                    .sum::<f64>() as f32
            };
        }
        finite(&output, &resource.name)?;
        Ok(output)
    }

    fn normalize(&self, tensor: usize, input: &[f32]) -> Result<Vec<f32>> {
        let weights = self.row(tensor, 0)?;
        let mean = input.iter().map(|&v| f64::from(v).powi(2)).sum::<f64>() / input.len() as f64;
        let scale = (mean + f64::from(self.config.rms_norm_eps)).sqrt().recip() as f32;
        let mut output = zeros(input.len())?;
        for ((result, &value), weight) in output.iter_mut().zip(input).zip(weights) {
            *result = value * scale * weight;
        }
        finite(&output, "RMSNorm")?;
        Ok(output)
    }
}

/// Mutable causal attention state, independent from other sessions.
pub struct Session<'a> {
    model: &'a Model<'a>,
    keys: Vec<Vec<f32>>,
    values: Vec<Vec<f32>>,
    position: usize,
    context: usize,
}

/// Causal execution state for a fixed range of layers in a staged pipeline.
/// Each instance belongs to one logical request/session. The transport must
/// preserve session identity and reject cross-session routing mistakes.
pub struct LayerSession<'a> {
    session: Session<'a>,
    layers: std::ops::Range<usize>,
}
impl LayerSession<'_> {
    /// Execute the next token's hidden state through the fixed layer range.
    /// Rejects duplicate/out-of-order positions before changing KV. Numerical
    /// failure leaves the committed position unchanged; retry overwrites the
    /// uncommitted cache slot before it becomes visible to later tokens.
    pub fn forward(&mut self, position: usize, hidden: &[f32]) -> Result<Vec<f32>> {
        if position != self.session.position || position >= self.session.context {
            return Err(Error(
                "layer session position mismatch or context exhausted".into(),
            ));
        }
        if hidden.len() != self.session.model.config.embedding_length {
            return Err(Error("hidden state width mismatch".into()));
        }
        finite(hidden, "layer input")?;
        let mut output = zeros(hidden.len())?;
        output.copy_from_slice(hidden);
        for layer in self.layers.clone() {
            self.session.forward_layer(&mut output, layer)?;
        }
        self.session.position += 1;
        Ok(output)
    }
    /// Number of committed tokens for this layer range.
    pub fn position(&self) -> usize {
        self.session.position
    }
    /// Fixed half-open layer range handled by this session.
    pub fn layers(&self) -> std::ops::Range<usize> {
        self.layers.clone()
    }
    /// Discard causal context; only a new position-zero token may follow.
    pub fn reset(&mut self) {
        self.session.reset();
    }
    /// Allocated KV scalar bytes (currently full-model allocation).
    pub fn kv_bytes(&self) -> usize {
        self.session.kv_bytes()
    }
}

/// Signature of a source-generated native token evaluator.
pub type NativeToken = fn(&mut Session<'_>, u32) -> Result<Vec<f32>>;

/// One uncommitted token, consumed by generated, ordered native layer calls.
///
/// Dropping this value, an invalid layer order or a failed kernel leaves the
/// session position unchanged. KV writes at the uncommitted position are not
/// visible to a subsequent token until that token overwrites them.
pub struct TokenStep<'session, 'model> {
    session: &'session mut Session<'model>,
    hidden: Vec<f32>,
    next_layer: usize,
}

impl TokenStep<'_, '_> {
    /// Execute exactly the next Llama layer through the native BML CPU kernels.
    /// Generated code emits one concrete call for each model layer.
    pub fn layer(mut self, index: usize) -> Result<Self> {
        if index != self.next_layer || index >= self.session.model.config.block_count {
            return Err(Error(format!(
                "native layer order mismatch: expected {}, got {index}",
                self.next_layer
            )));
        }
        self.session.forward_layer(&mut self.hidden, index)?;
        self.next_layer += 1;
        Ok(self)
    }

    /// Apply final normalization and output projection, then commit one token.
    /// All model layers must have succeeded in order before this call.
    pub fn finish(self) -> Result<Vec<f32>> {
        if self.next_layer != self.session.model.config.block_count {
            return Err(Error(
                "native token plan ended before all layers executed".into(),
            ));
        }
        let model = self.session.model;
        let logits = model.final_logits(&self.hidden)?;
        self.session.position += 1;
        Ok(logits)
    }
}

impl<'model> Session<'model> {
    /// Append tokens and return the final token's logits.
    ///
    /// Empty input, invalid IDs and context overflow are rejected before any
    /// state is changed. A numerical failure rolls back the committed position;
    /// only uncommitted cache slots may have been written and are overwritten
    /// by the next successful call.
    pub fn prefill(&mut self, tokens: &[u32]) -> Result<Vec<f32>> {
        if tokens.is_empty() {
            return Err(Error("prefill requires at least one token".into()));
        }
        if self
            .position
            .checked_add(tokens.len())
            .is_none_or(|n| n > self.context)
        {
            return Err(Error("session context limit exceeded".into()));
        }
        if tokens
            .iter()
            .any(|&id| id as usize >= self.model.config.vocab_size)
        {
            return Err(Error("token ID outside the model vocabulary".into()));
        }
        let start = self.position;
        let mut logits = Vec::new();
        for &token in tokens {
            // Only the last token's logits are returned; release the previous
            // vocabulary-sized vector before computing the next token.
            drop(logits);
            match self.forward_token(token) {
                Ok(output) => {
                    logits = output;
                }
                Err(error) => {
                    self.position = start;
                    return Err(error);
                }
            }
        }
        Ok(logits)
    }

    /// Append one token and return its vocabulary logits.
    pub fn decode(&mut self, token: u32) -> Result<Vec<f32>> {
        self.prefill(&[token])
    }

    /// Validate one token and start an uncommitted native execution step.
    /// Its layers can be emitted directly into model-specific AOT Rust source.
    pub fn begin_token(&mut self, token: u32) -> Result<TokenStep<'_, 'model>> {
        if self.position >= self.context {
            return Err(Error("session context limit exceeded".into()));
        }
        if token as usize >= self.model.config.vocab_size {
            return Err(Error("token ID outside the model vocabulary".into()));
        }
        let hidden = self.model.embed_token(token)?;
        Ok(TokenStep {
            session: self,
            hidden,
            next_layer: 0,
        })
    }

    /// Discard the committed context. Old slots become inaccessible immediately.
    pub fn reset(&mut self) {
        self.position = 0;
    }

    /// Number of tokens committed to this session.
    pub fn position(&self) -> usize {
        self.position
    }

    /// Configured maximum number of tokens in this session.
    pub fn context(&self) -> usize {
        self.context
    }

    /// Bytes occupied by key and value scalar payloads, excluding Vec headers.
    pub fn kv_bytes(&self) -> usize {
        self.keys
            .iter()
            .chain(&self.values)
            .map(|v| v.len() * 4)
            .sum()
    }

    fn forward_token(&mut self, token: u32) -> Result<Vec<f32>> {
        let count = self.model.config.block_count;
        let mut step = self.begin_token(token)?;
        for layer in 0..count {
            step = step.layer(layer)?;
        }
        step.finish()
    }

    fn forward_layer(&mut self, hidden: &mut [f32], layer_index: usize) -> Result<()> {
        let model = self.model;
        let layer = &model.layers[layer_index];
        {
            let normed = model.normalize(layer.attention_norm, hidden)?;
            let mut query = model.project(layer.query, &normed)?;
            let mut key = model.project(layer.key, &normed)?;
            let value = model.project(layer.value, &normed)?;
            rope(&mut query, self.position, &model.config);
            rope(&mut key, self.position, &model.config);
            let attended = self.attention(layer_index, &query, &key, &value)?;
            let projected = model.project(layer.attention_output, &attended)?;
            add_residual(hidden, &projected)?;
        }
        {
            let normed = model.normalize(layer.feed_forward_norm, hidden)?;
            let mut gate = model.project(layer.gate, &normed)?;
            let up = model.project(layer.up, &normed)?;
            for (g, u) in gate.iter_mut().zip(up) {
                *g = silu(*g) * u;
            }
            finite(&gate, "SwiGLU")?;
            let projected = model.project(layer.down, &gate)?;
            add_residual(hidden, &projected)?;
        }
        Ok(())
    }

    fn attention(
        &mut self,
        layer: usize,
        query: &[f32],
        key: &[f32],
        value: &[f32],
    ) -> Result<Vec<f32>> {
        let config = &self.model.config;
        let head_dim = config.head_dimension();
        let kv_width = config.kv_width();
        let start = self.position * kv_width;
        self.keys[layer][start..start + kv_width].copy_from_slice(key);
        self.values[layer][start..start + kv_width].copy_from_slice(value);
        let grouping = config.head_count / config.head_count_kv;
        let scale = (head_dim as f64).sqrt().recip();
        let mut output = zeros(config.embedding_length)?;
        for head in 0..config.head_count {
            let kv_head = head / grouping;
            let q = &query[head * head_dim..(head + 1) * head_dim];
            let mut scores = Vec::new();
            scores
                .try_reserve_exact(self.position + 1)
                .map_err(|_| Error("attention allocation failed".into()))?;
            for pos in 0..=self.position {
                let start = pos * kv_width + kv_head * head_dim;
                let k = &self.keys[layer][start..start + head_dim];
                scores.push(
                    q.iter()
                        .zip(k)
                        .map(|(&a, &b)| f64::from(a) * f64::from(b))
                        .sum::<f64>()
                        * scale,
                );
            }
            let maximum = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let mut total = 0.0;
            for score in &mut scores {
                *score = (*score - maximum).exp();
                total += *score;
            }
            for dimension in 0..head_dim {
                let mut sum = 0.0;
                for (pos, score) in scores.iter().enumerate() {
                    let index = pos * kv_width + kv_head * head_dim + dimension;
                    sum += (score / total) * f64::from(self.values[layer][index]);
                }
                output[head * head_dim + dimension] = sum as f32;
            }
        }
        finite(&output, "causal attention")?;
        Ok(output)
    }
}

fn zeros(length: usize) -> Result<Vec<f32>> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(length)
        .map_err(|_| Error("allocation failed".into()))?;
    values.resize(length, 0.0);
    Ok(values)
}

fn finite(values: &[f32], operation: &str) -> Result<()> {
    if values.iter().any(|value| !value.is_finite()) {
        Err(Error(format!("non-finite value in {operation}")))
    } else {
        Ok(())
    }
}

fn encoding(dtype: GgufDataType) -> Result<(usize, usize)> {
    match dtype {
        GgufDataType::F32 => Ok((1, 4)),
        GgufDataType::F16 => Ok((1, 2)),
        GgufDataType::Q4_0 => Ok((32, 18)),
        GgufDataType::Q8_0 => Ok((32, 34)),
        GgufDataType::Q6_K => Ok((256, 210)),
        _ => Err(Error(format!("unsupported tensor encoding {dtype:?}"))),
    }
}

fn validate_tensor(tensor: &Tensor<'_>) -> Result<()> {
    let (block, bytes) = encoding(tensor.dtype)?;
    let width = usize::try_from(tensor.dims[0]).map_err(|_| Error("row size overflow".into()))?;
    let rows = tensor.dims.get(1).copied().unwrap_or(1);
    let rows = usize::try_from(rows).map_err(|_| Error("row count overflow".into()))?;
    if !width.is_multiple_of(block) {
        return Err(Error(format!(
            "tensor {} row does not contain complete quantization blocks",
            tensor.name
        )));
    }
    let expected = (width / block)
        .checked_mul(bytes)
        .and_then(|n| n.checked_mul(rows))
        .ok_or_else(|| Error("tensor length overflow".into()))?;
    if tensor.data.len() != expected {
        return Err(Error(format!(
            "tensor {} byte length {}, expected {expected}",
            tensor.name,
            tensor.data.len()
        )));
    }
    if tensor.dtype == GgufDataType::F32 {
        if tensor
            .data
            .as_chunks::<4>()
            .0
            .iter()
            .any(|bytes| !f32::from_le_bytes(*bytes).is_finite())
        {
            return Err(Error(format!("non-finite value in {}", tensor.name)));
        }
    } else {
        for row in 0..rows {
            finite(&decode_row(tensor, row)?, &tensor.name)?;
        }
    }
    Ok(())
}

fn decode_row(tensor: &Tensor<'_>, row: usize) -> Result<Vec<f32>> {
    Ok(decode_f32(
        tensor.dtype,
        encoded_row(tensor, row)?,
        tensor.dims[0] as usize,
    )?)
}

fn encoded_row<'a>(tensor: &Tensor<'a>, row: usize) -> Result<&'a [u8]> {
    let (block, bytes) = encoding(tensor.dtype)?;
    let width = tensor.dims[0] as usize;
    let row_bytes = (width / block) * bytes;
    let start = row
        .checked_mul(row_bytes)
        .ok_or_else(|| Error("row offset overflow".into()))?;
    let end = start
        .checked_add(row_bytes)
        .ok_or_else(|| Error("row range overflow".into()))?;
    let data = tensor
        .data
        .get(start..end)
        .ok_or_else(|| Error(format!("row outside tensor {}", tensor.name)))?;
    Ok(data)
}

fn add_residual(hidden: &mut [f32], projected: &[f32]) -> Result<()> {
    for (value, &residual) in hidden.iter_mut().zip(projected) {
        *value += residual;
    }
    finite(hidden, "residual addition")
}

fn silu(value: f32) -> f32 {
    if value >= 0.0 {
        value / (1.0 + (-value).exp())
    } else {
        let exponent = value.exp();
        value * exponent / (1.0 + exponent)
    }
}

fn rope(values: &mut [f32], position: usize, config: &Config) {
    let width = config.head_dimension();
    for head in values.chunks_exact_mut(width) {
        for (index, pair) in head.as_chunks_mut::<2>().0.iter_mut().enumerate() {
            let angle = position as f64
                / f64::from(config.rope_freq_base).powf((2 * index) as f64 / width as f64);
            let (sin, cos) = angle.sin_cos();
            let (first, second) = (f64::from(pair[0]), f64::from(pair[1]));
            pair[0] = (first * cos - second * sin) as f32;
            pair[1] = (first * sin + second * cos) as f32;
        }
    }
}

fn optional_usize(reader: &impl crate::ModelMetadata, name: &str) -> Result<Option<usize>> {
    let Some(value) = reader.get_metadata(name) else {
        return Ok(None);
    };
    let number = match value {
        GgufMetadataValue::U8(v) => u64::from(*v),
        GgufMetadataValue::U16(v) => u64::from(*v),
        GgufMetadataValue::U32(v) => u64::from(*v),
        GgufMetadataValue::U64(v) => *v,
        GgufMetadataValue::I8(v) if *v >= 0 => *v as u64,
        GgufMetadataValue::I16(v) if *v >= 0 => *v as u64,
        GgufMetadataValue::I32(v) if *v >= 0 => *v as u64,
        GgufMetadataValue::I64(v) if *v >= 0 => *v as u64,
        _ => return Err(Error(format!("{name} must be a nonnegative integer"))),
    };
    Ok(Some(
        usize::try_from(number).map_err(|_| Error(format!("{name} overflows usize")))?,
    ))
}

fn required_usize(reader: &impl crate::ModelMetadata, name: &str) -> Result<usize> {
    optional_usize(reader, name)?.ok_or_else(|| Error(format!("missing {name}")))
}

fn optional_float(reader: &impl crate::ModelMetadata, name: &str) -> Result<Option<f32>> {
    let Some(value) = reader.get_metadata(name) else {
        return Ok(None);
    };
    let number = match value {
        GgufMetadataValue::F32(v) => *v,
        GgufMetadataValue::F64(v) => *v as f32,
        _ => return Err(Error(format!("{name} must be a floating-point number"))),
    };
    if !number.is_finite() {
        return Err(Error(format!("{name} must be finite")));
    }
    Ok(Some(number))
}

fn reject_variants(reader: &impl crate::ModelMetadata) -> Result<()> {
    let supported = [
        "llama.embedding_length",
        "llama.block_count",
        "llama.attention.head_count",
        "llama.attention.head_count_kv",
        "llama.feed_forward_length",
        "llama.context_length",
        "llama.vocab_size",
        "llama.rope.dimension_count",
        "llama.rope.freq_base",
        "llama.attention.layer_norm_rms_epsilon",
        "llama.attention.key_length",
        "llama.attention.value_length",
        "llama.expert_count",
        "llama.expert_used_count",
        "llama.attention.sliding_window",
        "llama.rope.scaling.type",
        "llama.rope.scaling.factor",
        "llama.rope.freq_scale",
        "llama.attention.causal",
    ];
    for name in reader.metadata().keys() {
        if name.starts_with("llama.") && !supported.contains(&name.as_str()) {
            return Err(Error(format!("unsupported Llama parameter: {name}")));
        }
    }
    if let Some(value) = reader.get_metadata("llama.attention.causal") {
        if !matches!(value, GgufMetadataValue::Bool(true)) {
            return Err(Error("only causal Llama attention is supported".into()));
        }
    }
    for name in [
        "llama.expert_count",
        "llama.expert_used_count",
        "llama.attention.sliding_window",
    ] {
        if optional_usize(reader, name)?.is_some_and(|value| value != 0) {
            return Err(Error(format!("unsupported Llama variant: {name}")));
        }
    }
    if let Some(value) = reader.get_metadata("llama.rope.scaling.type") {
        if !matches!(value, GgufMetadataValue::String(kind) if kind == "none") {
            return Err(Error("only unscaled Llama RoPE is supported".into()));
        }
    }
    for name in ["llama.rope.scaling.factor", "llama.rope.freq_scale"] {
        if optional_float(reader, name)?.is_some_and(|value| value != 1.0) {
            return Err(Error(format!("unsupported Llama RoPE scaling: {name}")));
        }
    }
    for name in [
        "llama.attention.clamp_kqv",
        "llama.attention.softmax_scale",
        "llama.rope.dimension_sections",
    ] {
        if reader.get_metadata(name).is_some() {
            return Err(Error(format!("unsupported Llama parameter: {name}")));
        }
    }
    Ok(())
}
