//! # bml-runtime
//!
//! Motor de ejecución L1 para grafos BML. Ejecuta programas RPN
//! linealizados con:
//!
//! - **Cero allocs en hot path**: los buffers se inicializan una sola
//!   vez al arrancar.
//! - **Hot loop < 32 KB**: el intérprete RPN debe caber en L1i.
//! - **Append-only**: cada evaluación escribe a una dirección
//!   pre-asignada nueva, nunca sobrescribe.
//! - **RPC distribuido**: interfaz para transmitir fragmentos
//!   `.bmlgraph` entre nodos.

#![deny(unsafe_code)]
#![warn(missing_docs)]

pub mod append_only;
pub mod buffer;
pub mod causal_artifact;
pub mod graph;
pub mod graph_rpc;
pub mod hot_loop;
pub mod net;
mod operation_schedule;
pub mod prepared_projection;
pub mod queue;
pub mod runtime;
pub mod scheduler;
pub mod shm;
pub mod symbolic_causal;

pub use hot_loop::HotLoop;
pub use runtime::Runtime;
pub use scheduler::WaveScheduler;
