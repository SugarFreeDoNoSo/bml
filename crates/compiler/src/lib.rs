//! # bml-compiler
//!
//! Compilador BML: transforma tensores parseados en un DAG estático,
//! aplica **Hash Consing** para deduplicar sub-árboles BML idénticos,
//! linealiza el DAG en Notación Polaca Inversa (RPN), y aplica
//! micro-fragmentación AOT para que el binario exportado (`.bmlgraph`)
//! caiga bajo el umbral de caché objetivo.

#![deny(unsafe_code)]
#![warn(missing_docs)]

pub mod bml_ops;
pub mod dag;
pub mod distributed;
pub mod eml;
pub mod fragment;
pub mod gguf_compiler;
pub mod hardware;
pub mod hash_cons;
pub mod op_fragments;
pub mod rpn;
pub mod sampler;
pub mod tokenizer;

pub use dag::Dag;
pub use fragment::{
    fragment_program, BmlGraph, Fragment, BMLGRAPH_MAGIC, BMLGRAPH_VERSION, DEFAULT_L1_THRESHOLD,
    L3_THRESHOLD,
};
pub use hash_cons::HashConsRegistry;
pub use rpn::{linearize, RpnOp, RpnProgram};
