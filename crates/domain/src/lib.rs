//! # bml-domain (BML = Binary-Minus-Log)
//!
//! Núcleo matemático del proyecto. [`symbolic`] distingue términos cerrados
//! de términos abiertos y permite interpretarlos con EML o BML explícitamente.
//! Su evaluador real es parcial y comprueba dominio, overflow y underflow;
//! no implementa la semántica compleja completa del artículo.
//!
//! Los módulos anteriores (`ast`, `encoder`, `soa`, `transformer`) siguen
//! disponibles durante la migración. Su AST incluye constantes arbitrarias
//! y no es la gramática pura de [`symbolic::ClosedTerm`].
//!
//! Cero dependencias externas. Solo stdlib de Rust.
//!
//! Referencia teórica: "All elementary functions from a single operator"
//! (ArXiv 2603.21852v2) — define EML en base E; BML es su adaptación a base 2.

#![deny(unsafe_code)]
#![warn(missing_docs)]

pub mod ast;
pub mod encoder;
pub mod operator;
pub mod soa;
pub mod symbolic;
pub mod transformer;

pub use ast::{ConstId, EvalContext, Node, NodeId, NodeKind, VarId};
pub use operator::bml_base_op;
pub use soa::NodeSoA;
pub use transformer::BMLTransformer;
