//! Shared implementations for the Lance fuzz targets.
//!
//! The binaries under `fuzz_targets/` use these modules for coverage-guided discovery with
//! libFuzzer. That workflow requires nightly Rust and cargo-fuzz. The stable `replay` binary also
//! uses these modules to decode and execute one saved input.
//!
//! Stable replay is useful for deterministic panics and logical assertion failures, but it does
//! not include libFuzzer's coverage guidance or sanitizer instrumentation. It is therefore not a
//! substitute for cargo-fuzz when reproducing memory-safety failures.

pub mod dataset_ops;
