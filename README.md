# Lance Fuzz

Stateful, model-based fuzzing for Lance datasets, inspired by ClickHouse's BuzzHouse.

Buzz-style fuzzing generates weighted sequences of operations against a data system while maintaining a model of the expected state.
After every successful operation, the harness updates its model and checks Lance through logical and structural oracles.

Resources:
- [FOSSDEM 2025: Fuzzing databases is difficult](https://www.youtube.com/watch?v=CW4Ntdtp7lg)
- [BuzzHouse: Bridging the database fuzzing gap for testing ClickHouse](https://clickhouse.com/blog/buzzhouse-bridging-the-database-fuzzing-gap-for-testing-clickhouse)
- BuzzHouse [source](https://github.com/ClickHouse/ClickHouse/tree/master/src/Client/BuzzHouse) and [CI job](https://github.com/ClickHouse/ClickHouse/blob/master/ci/jobs/buzzhouse_job.py)

## How it works

`dataset_ops` runs stateful operation sequences against a fresh Lance dataset.
It mirrors the expected rows in a `BTreeMap<u64, Row>` and compares Lance with that model throughout the session.
A finding is an error from an eligible operation, a logical mismatch, or a structural validation failure.

### How a session works

Each input is generated from an RNG seed, column specifications, and operation candidates.
The initial schema has one to five value columns, which `AddColumn` can grow to eight.
The session examines at most 64 candidates and executes the first 24 whose preconditions are met.

Session:

1. Creates a fresh dataset with a non-null `u64` `id` column.
2. Generates data and executes eligible operations in order.
3. Applies every successful mutation to the in-memory row model.
4. Verifies Lance after every executed operation.
5. Runs structural validation and reopens file-backed datasets for a final independent read.

### What it exercises

- **Writes:** Append, full and partial merge-insert, conditional update, replace-all, merge-delete, SQL delete, and SQL update.
- **Schema and history:** Add or remove columns, restore an earlier version, and migrate between Lance file versions.
- **Indices:** Create, optimize, drop, and query scalar or vector indices.
- **Maintenance:** Compact files with varied physical and concurrency settings.
- **Reads:** Filter, count, project, limit, offset, `take`, and KNN search.

Scalar index selection is type-aware and covers BTree, Bitmap, ZoneMap, BloomFilter, NGram, Inverted, and LabelList.
Float32 fixed-size lists can use an IVF_FLAT vector index.

### Oracle boundaries

The model checks logical dataset contents; it does not independently recompute generated values.

KNN validation checks result safety and cardinality — live, unique, non-null rows, and at most
`min(k, live_non_null_rows)` of them — not exact nearest-neighbor order, because IVF searches are
approximate and an existing index may not cover later writes.

The schema excludes types such as `Map` that cannot be written by every initial file version, and
storage coverage is limited to the local filesystem and the in-memory object store.

### Normalizations

Some model behavior is deliberately shaped around what Lance can represent:

- Null structs are modeled as valid structs with null children, because the Lance file format does
  not preserve struct-level validity.
- `SourceDedupeBehavior::FirstSeen` is exercised only with duplicate keys that already exist;
  duplicate unmatched source rows would both be inserted and violate the unique-key model.
- Equality predicates avoid integer literals outside the SQL parser's safe `i64` range.
- Inverted indices use the raw tokenizer with every normalization filter disabled, so a document is
  exactly one token and full-text matches are plain string equality. Full-text probes run only on
  plain string columns: list documents are tokenized inconsistently between the index build and the
  flat-match path, so they have no oracle.
- Dictionary arrays are canonicalized before comparison, because equivalent values can use different
  dictionary encodings.
- Local generator wrappers avoid zero-row generation failures, alignment problems for decimal
  buffers, timezone loss in generated timestamp arrays, and wall-clock-dependent temporal values.

### Configuration

`LANCE_FUZZ_STORAGE` selects one storage profile for the entire campaign.
Storage is not encoded in `FuzzInput`, so the same input can be replayed against every profile.

| Value | Storage |
|---|---|
| `memory` (default) | In-memory object store through `memory://` |
| `tmp` | Unique temporary directory addressed by filesystem path |
| `uri` | Unique temporary directory addressed through a `file://` URI |

The input seed selects physical settings without consuming the RNG stream used for data generation:

| Setting | Values |
|---|---|
| Lance file version | `2.0`, `2.1`, `2.2`, `2.3` |
| Rows per row group | `1`, `4`, `16` |
| Rows per file | `2`, `4`, `8` |
| Stable row IDs | Disabled or enabled |
| Compaction profile | Default or two independently selected knobs |

`SOFT_STATS` downgrades the row-count checks on merge, update, and delete from assertions to warnings.
Those counts are hard assertions by default; softening them lets a run continue to the scan and probe
checks, which distinguishes a genuine data divergence from a merely miscounted stat.

## Running

### Setup

Install [Pixi](https://pixi.sh), nightly Rust, and cargo-fuzz:

```sh
rustup install nightly
rustup +nightly component add llvm-tools-preview
cargo install cargo-fuzz
```

### Fuzzing

Build the fuzz target or run it across all CPU cores:

```sh
pixi run fuzz-build
pixi run fuzz       # run until stopped
pixi run fuzz 300   # run for 300 seconds
```

Fuzz output is written to `logs/`.

### Reproducing a finding

A finding is a saved artifact. Replay it on stable, without nightly or cargo-fuzz:

```sh
cargo run --bin replay -- <artifact-path>
REPLAY_DUMP=1 cargo run --bin replay -- <artifact-path>   # print the decoded input, do not run it
```

The dev profile keeps `debug_assert!` and overflow checks on. Replay has no coverage guidance or
sanitizer instrumentation, so memory-safety findings must still be confirmed under cargo-fuzz.

### Importing crashes

Import new findings from the ignored `artifacts/dataset_ops/` directory:

```sh
pixi run crash-import
uv run hack/crash-check.py dataset_ops --import path/to/artifact
```

By default, imports keep up to three inputs per crash class. Named artifacts are always kept.
Inputs are content-addressed, so re-importing them is safe and does not create duplicates.

Maintain the `bug` column in `manifest.tsv` by hand:

- `?`: not yet linked to a bug
- `open`: an uninvestigated divergence
- `harness`: a known harness false positive

### Coverage from the corpus

```sh
pixi run fuzz-coverage
pixi run fuzz-coverage-report
```

`pixi run fuzz-coverage [target] [min-dark-lines]` builds the target with coverage
instrumentation, then replays every input in `corpus/<target>`. It prints total Lance line coverage,
a per-crate table, and every fully uncovered file with at least 400 lines by default.

Collection can take several minutes. `pixi run fuzz-coverage-report` re-renders the report from
the last collection, so changing the dark-file threshold is fast. The full per-file report is
written to `coverage/<target>/per-file.txt`.

Coverage counters are tied to one build. Collect a fresh baseline after updating Lance or changing
the harness.

## Updating source code

Use the same upstream Git tag for every direct `lance-*` dependency and `lance-datagen` in
`Cargo.toml`. To fuzz a local Lance checkout instead, rename `.cargo/local-lance.toml` to
`.cargo/config.toml` — see the caveats in that file. Then refresh and verify the dependencies:

```sh
cargo update
pixi run fuzz-build
pixi run crash-check
```

After changing the source code, run:

```sh
cargo fmt --all --check
cargo nextest run --lib
pixi run fuzz-build
pixi run crash-check
```

Keep the `FuzzInput` decoding layout when possible so the existing corpus and crash suite retain
their meaning. If it must change, rebuild the corpus and re-import representative crashes with the
new harness.

Run the tracked crash suite after updating Lance or changing the harness:

```sh
pixi run crash-check
pixi run crash-check dataset_ops 3       # retry mismatches three times
uv run hack/crash-check.py dataset_ops index   # check matching inputs only
```

The report shows inputs that still reproduce, no longer crash, or have a different class.
`crash-check` exits with status 1 when an input no longer matches, so it can run in CI.

Crash classes are normalized across Lance versions, but serialized inputs depend on the `FuzzInput`
layout. If the layout changes, the same bytes can decode into a different session. The manifest
stores a hash of `src/dataset_ops/ops.rs` (the file defining the input types), and `crash-check`
warns if it changes. In that case,
re-import inputs instead of treating widespread class changes as Lance regressions.
