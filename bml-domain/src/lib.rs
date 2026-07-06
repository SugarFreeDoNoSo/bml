//! # bml-domain
//!
//! Núcleo matemático del proyecto BML. Define el operador fundamental
//! `bml(x, y) = exp(x) - ln(y)` (análogo continuo del NAND lógico, con
//! completitud funcional), la gramática estricta del AST, el layout de
//! memoria SoA alineado a línea de caché, y el `BMLTransformer` que
//! reduce operaciones estándar a la gramática BML usando solo el
//! operador y la constante 1.
//!
//! Cero dependencias externas. Solo stdlib de Rust.
//!
//! Referencia teórica: "All elementary functions from a single operator"
//! (ArXiv 2603.21852v2). El operador EML se renombra aquí como `bml`
//! por convención del proyecto.

#![deny(unsafe_code)]
#![warn(missing_docs)]

pub mod ast;
pub mod operator;
pub mod soa;
pub mod transformer;

pub use ast::{Node, NodeKind};
pub use operator::bml;
pub use soa::NodeSoA;
pub use transformer::BMLTransformer;
