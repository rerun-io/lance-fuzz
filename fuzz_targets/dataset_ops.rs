//! libFuzzer driver for the stateful dataset-operation target.
//!
//! libFuzzer decodes each input as `lance_fuzz::dataset_ops::FuzzInput`; the shared module owns
//! the operation model and correctness checks. Use the stable `replay` binary to execute a saved
//! input without coverage guidance or sanitizer instrumentation.
#![no_main]

use libfuzzer_sys::fuzz_target;

fuzz_target!(|input: lance_fuzz::dataset_ops::FuzzInput| {
    lance_fuzz::dataset_ops::run_blocking(input);
});
