//! Execute the stored projection programs of a compiled causal model artifact.
use crate::{
    buffer::ResultBuffer,
    graph::{GraphLimits, ValidatedGraph},
};
use bml_compiler::{
    causal_artifact::{self, Manifest},
    gguf_compiler::InferenceCompiler,
    HashConsedKV,
};
use std::{collections::BTreeMap, path::Path};

/// A model whose serialized row graphs have passed runtime preflight.
pub struct CompiledCausalModel {
    compiler: InferenceCompiler,
    manifest: Manifest,
    programs: Vec<ValidatedGraph>,
    bindings: BTreeMap<String, usize>,
}

impl CompiledCausalModel {
    /// Load the referenced model and validate all saved programs before execution.
    pub fn open(directory: &Path, source_override: Option<&Path>) -> Result<Self, String> {
        let artifact = causal_artifact::load_directory(directory, source_override)?;
        let mut programs = Vec::with_capacity(artifact.graphs.len());
        for (graph, descriptor) in artifact.graphs.into_iter().zip(&artifact.manifest.programs) {
            programs.push(
                ValidatedGraph::new(graph, 0, 2, descriptor.columns, GraphLimits::default())
                    .map_err(|e| format!("compiled projection preflight: {e}"))?,
            );
        }
        let bindings = artifact
            .manifest
            .projections
            .iter()
            .map(|p| (p.tensor.clone(), p.program))
            .collect();
        Ok(Self {
            compiler: artifact.compiler,
            manifest: artifact.manifest,
            programs,
            bindings,
        })
    }

    /// Read-only model configuration, tokenizer and source storage information.
    pub fn compiler(&self) -> &InferenceCompiler {
        &self.compiler
    }

    /// Metadata verified at load time.
    pub fn manifest(&self) -> &Manifest {
        &self.manifest
    }

    fn project(&self, input: &[f64], name: &str) -> Result<Vec<f64>, String> {
        let &index = self
            .bindings
            .get(name)
            .ok_or("missing compiled projection binding")?;
        let program = &self.programs[index];
        let columns = self.manifest.programs[index].columns;
        if input.len() != columns {
            return Err("compiled projection input width mismatch".into());
        }
        let mut buffer = ResultBuffer::new(2, columns);
        for (i, &value) in input.iter().enumerate() {
            buffer.write(0, i as u32, value);
        }
        self.compiler.project_rows(input, name, |_, row| {
            for (i, &value) in row.iter().enumerate() {
                buffer.write(1, i as u32, f64::from(value));
            }
            program.execute(&[], &mut buffer).map_err(|e| e.to_string())
        })
    }

    /// Execute a full causal prefix with the programs read from the artifact.
    pub fn forward(&self, input_ids: &[u32]) -> Result<Vec<f64>, String> {
        self.compiler
            .forward_with_projection(input_ids, &mut |x, name| self.project(x, name))
    }

    /// Execute prefill or incremental decode, committing KV positions on success.
    pub fn forward_cached(
        &self,
        input_ids: &[u32],
        cache: &mut HashConsedKV,
    ) -> Result<Vec<f64>, String> {
        self.compiler
            .forward_cached_with_projection(input_ids, cache, &mut |x, name| self.project(x, name))
    }
}
