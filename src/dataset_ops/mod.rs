//! Stateful, model-based fuzzing for Lance datasets.
//!
//! See the project README for supported operations, architecture, checks, and limitations.

mod compact;
mod data;
mod index;
mod migrate;
mod ops;
mod oracle;
mod read;
mod runner;
mod scan;

pub use ops::FuzzInput;
pub use runner::{run, run_blocking};
