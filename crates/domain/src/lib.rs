//! # bml-domain (BML = Binary-Minus-Log)
//!
//! Núcleo matemático del proyecto BML. Define el operador fundamental
//! `bml(x, y) = 2^x - log2(y)` — el análogo de EML (Exp-Minus-Log,
//! `eml(x, y) = exp(x) - ln(y)`) reescrito en **base 2** para alinearse
//! con el formato IEEE 754 de `f64` y usar `exp2`/`log2` nativos de la FPU.
//! BML actúa como operador con completitud funcional (análogo continuo del
//! NAND lógico) junto con la constante 1.
//!
//! Contiene: la gramática estricta del AST (`S -> 1 | BML(S, S)`), el
//! layout de memoria SoA alineado a línea de caché, y el `BMLTransformer`
//! que reduce operaciones estándar a la gramática BML.
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
pub mod transformer;

pub use ast::{ConstId, EvalContext, Node, NodeId, NodeKind, VarId};
pub use operator::bml;
pub use soa::NodeSoA;
pub use transformer::BMLTransformer;
