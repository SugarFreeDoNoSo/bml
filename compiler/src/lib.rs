//! # bml-compiler
//!
//! Compilador BML: transforma tensores parseados en un DAG estático,
//! aplica **Hash Consing** para deduplicar sub-árboles BML idénticos,
//! linealiza el DAG en Notación Polaca Inversa (RPN), y aplica
//! micro-fragmentación AOT para que el binario exportado (`.bmlgraph`)
//! caiga bajo el umbral de caché objetivo.

#![deny(unsafe_code)]
#![warn(missing_docs)]

pub mod dag;
pub mod fragment;
pub mod hash_cons;
pub mod rpn;

pub use dag::Dag;
pub use fragment::{
    fragment_program, BmlGraph, Fragment, BMLGRAPH_MAGIC, BMLGRAPH_VERSION, DEFAULT_L1_THRESHOLD,
    L3_THRESHOLD,
};
pub use hash_cons::HashConsRegistry;
pub use rpn::{linearize, RpnOp, RpnProgram};
