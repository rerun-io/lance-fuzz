//! Runs one saved `dataset_ops` input without cargo-fuzz.
//!
//! The binary reads an artifact, decodes the complete byte slice with
//! `Arbitrary::arbitrary_take_rest`, and calls the shared harness implementation.
//!
//! ```sh
//! cargo run --bin replay -- artifacts/dataset_ops/crash-<hash>
//! REPLAY_DUMP=1 cargo run --bin replay -- artifacts/dataset_ops/crash-<hash>
//! ```
//!
//! `REPLAY_DUMP` prints the decoded input without executing it. A successful execution prints
//! `replay completed without crashing`; argument and file errors exit with status 2. Decode
//! failures panic, while harness failures propagate as panics or process aborts.
//!
//! Use the development profile when reproducing assertions or arithmetic overflow so debug
//! assertions and overflow checks remain enabled. This stable runner has no libFuzzer mutation,
//! coverage, or sanitizer instrumentation, so memory-safety findings must also be checked with
//! cargo-fuzz.

use std::process::ExitCode;

use arbitrary::{Arbitrary, Unstructured};

fn main() -> ExitCode {
    let Some(path) = std::env::args().nth(1) else {
        eprintln!(
            "usage: replay <artifact-path>\n\
             replays a saved dataset_ops artifact once, on stable (no nightly / cargo-fuzz)."
        );
        return ExitCode::from(2);
    };

    let bytes = match std::fs::read(&path) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("cannot read artifact {path}: {e}");
            return ExitCode::from(2);
        }
    };

    // Match the typed fuzz target's whole-input decoding. The execution environment still differs
    // because this binary has no libFuzzer or sanitizer instrumentation.
    let input = lance_fuzz::dataset_ops::FuzzInput::arbitrary_take_rest(Unstructured::new(&bytes))
        .expect("decode dataset_ops input");

    // With REPLAY_DUMP set, print the decoded input (Debug) and exit without running it — used to
    // extract the concrete op sequences baked into the standalone bug repros.
    if std::env::var_os("REPLAY_DUMP").is_some() {
        println!("{input:#?}");
        return ExitCode::SUCCESS;
    }

    lance_fuzz::dataset_ops::run_blocking(input);

    println!("replay completed without crashing");
    ExitCode::SUCCESS
}
