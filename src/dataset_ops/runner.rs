//! Session driver: runs one operation sequence and verifies the dataset after every step.

use std::collections::BTreeSet;
use std::sync::Arc;

use arrow::array::{
    ArrayRef, BooleanArray, Int64Array, RecordBatch, RecordBatchIterator, StringArray, UInt64Array,
};
use arrow::compute::cast;
use arrow::datatypes::{DataType, Field, Schema};
use lance::Dataset;
use lance::dataset::optimize::compact_files;
use lance::dataset::{
    NewColumnTransform, UpdateBuilder, WhenMatched, WhenNotMatched, WhenNotMatchedBySource,
    WriteMode,
};
use lance::index::DatasetIndexExt as _;
use lance_file::version::LanceFileVersion;
use lance_index::optimize::OptimizeOptions;
use rand_xoshiro::Xoshiro256PlusPlus;
use rand_xoshiro::rand_core::SeedableRng as _;

use super::compact::CompactionProfile;
use super::data::{
    WriteProfile, all_null_cell, build_write, full_row, make_generator, merge, source_reader,
};
use super::index::{
    IndexEntry, IndexKind, IndexQueryCheck, create_index, rebuild_indices, scalar_index_kinds,
    verify_index_query,
};
use super::migrate::{
    assert_data_file_versions, dataset_file_version, file_version_from_seed, migrate_dataset,
    next_file_version, requests_sparse_encoding, schema_signature, with_sparse_request,
};
use super::ops::{AddFill, ColSpec, DataParams, FuzzInput, LenClass, NullClass, Op, PrimSpec};
use super::oracle::{
    Column, Lit, Oracle, is_cell_null, is_float32_vector, lit_friendly, verify_full_scan,
    verify_probes,
};
use super::read::execute_read;

/// Cap on ops per input: keeps a single iteration fast while still allowing the multi-step
/// interleavings (upsert → drain → delete → probe) these bugs need.
const MAX_OPS: usize = 24;
/// Cap on candidate ops examined while looking for [`MAX_OPS`] eligible operations.
const MAX_OP_CANDIDATES: usize = 64;
/// Cap on rows per write op. Keys live in `u8` space so ops collide often.
const MAX_KEYS_PER_OP: usize = 16;
/// Value columns in the initial schema.
const MAX_VALUE_COLS: usize = 5;
/// Hard cap on value columns after `AddColumn` ops.
const MAX_TOTAL_COLS: usize = 8;
/// Minimum live rows before a vector index is created (IVF training needs data).
const MIN_ROWS_FOR_VECTOR_INDEX: usize = 64;

/// Stat check that is a hard assertion normally, but — when `SOFT_STATS` is set in the environment
/// — only warns and lets the run continue to the full-scan/probe oracle checks. Used to tell a
/// genuine data divergence (scan disagrees) from a merely-miscounted stat.
macro_rules! stat_eq {
    ($left:expr, $right:expr, $($fmt:tt)*) => {{
        let (l, r) = ($left, $right);
        if std::env::var_os("SOFT_STATS").is_some() {
            if l != r {
                eprintln!("SOFT_STATS mismatch [{}]: left={l} right={r}", format!($($fmt)*));
            }
        } else {
            assert_eq!(l, r, $($fmt)*);
        }
    }};
}

/// Where one session's dataset lives. Selected once per campaign, not per input.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Storage {
    /// The in-memory object store (`memory://`). The default.
    Memory,
    /// Local filesystem via a plain path under a unique temporary directory.
    Tmp,
    /// Local filesystem addressed through a `file://` URI, exercising URI parsing.
    Uri,
}

impl Storage {
    fn from_env() -> Self {
        match std::env::var("LANCE_FUZZ_STORAGE") {
            Err(std::env::VarError::NotPresent) => Self::Memory,
            Ok(v) if v == "memory" => Self::Memory,
            Ok(v) if v == "tmp" => Self::Tmp,
            Ok(v) if v == "uri" => Self::Uri,
            other => panic!("LANCE_FUZZ_STORAGE must be memory|tmp|uri, got {other:?}"),
        }
    }

    /// Dataset location for one iteration, plus the guard that owns the on-disk directory.
    /// The guard must outlive every dataset handle; dropping it removes the directory.
    fn location(self) -> (String, Option<tempfile::TempDir>) {
        match self {
            Self::Memory => ("memory://fuzz".to_owned(), None),
            Self::Tmp | Self::Uri => {
                let dir = tempfile::TempDir::new().expect("create temp dir");
                let path = dir.path().join("fuzz");
                let path = path.to_str().expect("utf8 temp path");
                let uri = match self {
                    Self::Tmp => path.to_owned(),
                    Self::Uri => format!("file://{path}"),
                    Self::Memory => unreachable!(),
                };
                (uri, Some(dir))
            }
        }
    }
}

/// State recorded after every commit, so restore can roll the oracle back with the dataset.
struct Snapshot {
    version: u64,
    file_version: LanceFileVersion,
    oracle: Oracle,
    columns: Vec<Column>,
    indices: Vec<IndexEntry>,
}

/// [`run`] on a fresh current-thread runtime, for callers without one (the libFuzzer target).
pub fn run_blocking(input: FuzzInput) {
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("build tokio runtime");
    rt.block_on(run(input));
}

/// Sorted index names currently registered on the dataset.
async fn load_index_names(dataset: &Dataset, ctx: &str) -> Vec<String> {
    let mut names: Vec<String> = dataset
        .load_indices()
        .await
        .unwrap_or_else(|e| panic!("{ctx}: load indices: {e}"))
        .iter()
        .map(|index| index.name.clone())
        .collect();
    names.sort_unstable();
    names
}

/// Whether the model holds at least one non-null value in column `col`. This is the real
/// precondition for IVF training: KMeans errors on zero vectors, while a single non-null
/// row trains fine (verified against Lance v11.0.0-beta.2).
fn has_non_null_value(oracle: &Oracle, columns: &[Column], col: &str) -> bool {
    columns
        .iter()
        .position(|c| c.name == col)
        .is_some_and(|j| oracle.values().any(|row| !is_cell_null(&row[j])))
}

/// Keys for one write op: capped, deduplicated, and ordered, so the model stays key-unique.
fn dedup_keys(keys: &[u8]) -> Vec<u64> {
    keys.iter()
        .take(MAX_KEYS_PER_OP)
        .map(|k| u64::from(*k))
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect()
}

// ---------------------------------------------------------------------------------------------
// Main loop
// ---------------------------------------------------------------------------------------------

/// Run one session. Any finding — an error from an eligible operation, a model mismatch, or a
/// structural validation failure — panics, so libFuzzer records the input.
pub async fn run(input: FuzzInput) {
    run_inner(input).await;
}

/// Runs one session and returns the number of executed operations (used by tests).
async fn run_inner(input: FuzzInput) -> usize {
    // 1..=MAX_VALUE_COLS value columns of arbitrary type.
    let mut specs: Vec<ColSpec> = input.schema.into_iter().take(MAX_VALUE_COLS).collect();
    if specs.is_empty() {
        specs.push(ColSpec::Prim(PrimSpec::UInt64));
    }
    let mut columns: Vec<Column> = specs
        .into_iter()
        .enumerate()
        .map(|(i, spec)| {
            let dt = make_generator(&spec, LenClass::Small, 0.0)
                .data_type()
                .clone();
            Column {
                name: format!("v{i}"),
                spec,
                dt,
            }
        })
        .collect();
    let mut next_col = columns.len();

    let mut rng = Xoshiro256PlusPlus::seed_from_u64(input.seed);

    // Fresh dataset location per iteration: no cross-iteration state. File-backed profiles own
    // a unique temp directory that is removed when `_tmp_dir` drops at the end of the run.
    let storage = Storage::from_env();
    let file_version = file_version_from_seed(input.seed);
    let write_profile = WriteProfile::from_seed(input.seed);
    let compaction_profile = CompactionProfile::from_seed(input.seed);
    let (location, _tmp_dir) = storage.location();

    // Sparse structural encoding is opt-in per field, and the request is persisted with the
    // dataset schema, so later appends, updates and compaction re-encode through it too.
    let sparse = requests_sparse_encoding(input.seed, file_version);
    let schema = Arc::new(Schema::new(
        std::iter::once(Field::new("id", DataType::UInt64, false))
            .chain(columns.iter().map(|c| {
                let field = Field::new(&c.name, c.dt.clone(), true);
                if sparse {
                    with_sparse_request(field)
                } else {
                    field
                }
            }))
            .collect::<Vec<_>>(),
    ));
    let empty = RecordBatch::new_empty(schema.clone());
    let reader = RecordBatchIterator::new([Ok(empty)], schema);
    let mut write_params = write_profile.write_params();
    write_params.mode = WriteMode::Overwrite;
    write_params.data_storage_version = Some(file_version);
    let mut dataset = Dataset::write(reader, &location, Some(write_params))
        .await
        .expect("create empty dataset");
    // Every iteration must start from an empty location. Lance guarantees that today even for
    // the fixed `memory://fuzz` URI (object-store registries are per-session and cache stores
    // only by weak reference), but the guarantee is subtle enough to pin: leftover state would
    // make a crash depend on prior iterations and break single-input replay.
    assert_eq!(
        dataset.version().version,
        1,
        "initial write reused existing dataset state"
    );

    let mut oracle = Oracle::new();
    // Every key that has ever been in the dataset. Keys in here but not in `oracle` are
    // deleted-but-once-present — the only keys whose stale scalar-index entries a point probe
    // can expose.
    let mut ever_inserted: BTreeSet<u64> = BTreeSet::new();
    let mut indices: Vec<IndexEntry> = Vec::new();
    let mut next_index = 0usize;
    let mut expected_file_version = file_version;
    let mut snapshots: Vec<Snapshot> = vec![Snapshot {
        version: dataset.version().version,
        file_version: expected_file_version,
        oracle: oracle.clone(),
        columns: columns.clone(),
        indices: indices.clone(),
    }];

    let mut executed_ops = 0;
    for (candidate, op) in input.ops.iter().take(MAX_OP_CANDIDATES).enumerate() {
        if executed_ops == MAX_OPS {
            break;
        }
        let step = executed_ops;
        let ctx = format!("step {step}, candidate {candidate}: op {op:?}");
        let n_cols = columns.len();

        // Each arm reports whether it executed: `false` means its preconditions were not met,
        // so the candidate is skipped without consuming the execution budget.
        let executed = match op {
            Op::Append { keys, data } => {
                let new_keys: Vec<u64> = dedup_keys(keys)
                    .into_iter()
                    .filter(|k| !oracle.contains_key(k))
                    .collect();
                if !new_keys.is_empty() {
                    let include: Vec<usize> = (0..n_cols).collect();
                    let w = build_write(&columns, &include, &new_keys, None, *data, 0, &mut rng);
                    dataset
                        .append(source_reader(w.batch), Some(write_profile.write_params()))
                        .await
                        .unwrap_or_else(|e| panic!("{ctx}: append failed: {e}"));
                    for (i, k) in new_keys.iter().enumerate() {
                        oracle.insert(*k, full_row(&w.arrays, i));
                    }
                    true
                } else {
                    false
                }
            }

            Op::UpsertFull {
                keys,
                rotate,
                dup_first,
                data,
            } => {
                let keys = dedup_keys(keys);
                if !keys.is_empty() {
                    // FirstSeen dedupe only applies to source rows that match the same *target*
                    // row; duplicate unmatched source rows are inserted twice (documented), which
                    // would break the id-uniqueness invariant — so only duplicate a key that is
                    // already present.
                    let dup = if *dup_first {
                        keys.iter().position(|k| oracle.contains_key(k))
                    } else {
                        None
                    };
                    let include: Vec<usize> = (0..n_cols).collect();
                    let w = build_write(&columns, &include, &keys, dup, *data, *rotate, &mut rng);
                    let updated = keys.iter().filter(|k| oracle.contains_key(k)).count() as u64;
                    let inserted = keys.len() as u64 - updated;
                    let stats = merge(
                        &mut dataset,
                        WhenMatched::UpdateAll,
                        WhenNotMatched::InsertAll,
                        WhenNotMatchedBySource::Keep,
                        dup.is_some(),
                        w.batch,
                    )
                    .await
                    .unwrap_or_else(|e| panic!("{ctx}: merge failed: {e}"));
                    stat_eq!(stats.num_updated_rows, updated, "{ctx}: updated_rows");
                    stat_eq!(stats.num_inserted_rows, inserted, "{ctx}: inserted_rows");
                    if dup.is_some() {
                        stat_eq!(stats.num_skipped_duplicates, 1, "{ctx}: skipped_duplicates");
                    }
                    for (i, k) in keys.iter().enumerate() {
                        oracle.insert(*k, full_row(&w.arrays, i));
                    }
                    true
                } else {
                    false
                }
            }

            Op::UpsertPartial {
                keys,
                mask,
                rotate,
                data,
            } => {
                let keys = dedup_keys(keys);
                if !keys.is_empty() {
                    let mut include: Vec<usize> =
                        (0..n_cols).filter(|j| mask & (1 << (j % 8)) != 0).collect();
                    if include.is_empty() {
                        include.push((*mask as usize) % n_cols);
                    }
                    let w = build_write(&columns, &include, &keys, None, *data, *rotate, &mut rng);
                    let matched: Vec<u64> = keys
                        .iter()
                        .copied()
                        .filter(|k| oracle.contains_key(k))
                        .collect();
                    let stats = merge(
                        &mut dataset,
                        WhenMatched::UpdateAll,
                        WhenNotMatched::DoNothing,
                        WhenNotMatchedBySource::Keep,
                        false,
                        w.batch,
                    )
                    .await
                    .unwrap_or_else(|e| panic!("{ctx}: merge failed: {e}"));
                    stat_eq!(
                        stats.num_updated_rows,
                        matched.len() as u64,
                        "{ctx}: updated_rows"
                    );
                    stat_eq!(stats.num_inserted_rows, 0, "{ctx}: inserted_rows");
                    for k in matched {
                        let i = keys.iter().position(|x| *x == k).expect("key present");
                        let row = oracle.get_mut(&k).expect("matched key");
                        for (slot, &j) in include.iter().enumerate() {
                            row[j] = (w.arrays[slot].clone(), i);
                        }
                    }
                    true
                } else {
                    false
                }
            }

            Op::UpsertIf {
                keys,
                threshold,
                data,
            } => {
                let keys = dedup_keys(keys);
                if !keys.is_empty() {
                    let t = u64::from(*threshold);
                    let include: Vec<usize> = (0..n_cols).collect();
                    let w = build_write(&columns, &include, &keys, None, *data, 0, &mut rng);
                    let matched = WhenMatched::update_if(&dataset, &format!("target.id >= {t}"))
                        .unwrap_or_else(|e| panic!("{ctx}: update_if: {e}"));
                    let updated = keys
                        .iter()
                        .filter(|k| oracle.contains_key(k) && **k >= t)
                        .count() as u64;
                    let inserted = keys.iter().filter(|k| !oracle.contains_key(k)).count() as u64;
                    let stats = merge(
                        &mut dataset,
                        matched,
                        WhenNotMatched::InsertAll,
                        WhenNotMatchedBySource::Keep,
                        false,
                        w.batch,
                    )
                    .await
                    .unwrap_or_else(|e| panic!("{ctx}: merge failed: {e}"));
                    stat_eq!(stats.num_updated_rows, updated, "{ctx}: updated_rows");
                    stat_eq!(stats.num_inserted_rows, inserted, "{ctx}: inserted_rows");
                    for (i, k) in keys.iter().enumerate() {
                        if !oracle.contains_key(k) || *k >= t {
                            oracle.insert(*k, full_row(&w.arrays, i));
                        }
                    }
                    true
                } else {
                    false
                }
            }

            Op::ReplaceAll { keys, data } => {
                // Empty source is allowed here: it deletes everything.
                let keys = dedup_keys(keys);
                let include: Vec<usize> = (0..n_cols).collect();
                let w = build_write(&columns, &include, &keys, None, *data, 0, &mut rng);
                let updated = keys.iter().filter(|k| oracle.contains_key(k)).count() as u64;
                let inserted = keys.len() as u64 - updated;
                let deleted = oracle.keys().filter(|k| !keys.contains(k)).count() as u64;
                let stats = merge(
                    &mut dataset,
                    WhenMatched::UpdateAll,
                    WhenNotMatched::InsertAll,
                    WhenNotMatchedBySource::Delete,
                    false,
                    w.batch,
                )
                .await
                .unwrap_or_else(|e| panic!("{ctx}: merge failed: {e}"));
                stat_eq!(stats.num_updated_rows, updated, "{ctx}: updated_rows");
                stat_eq!(stats.num_inserted_rows, inserted, "{ctx}: inserted_rows");
                stat_eq!(stats.num_deleted_rows, deleted, "{ctx}: deleted_rows");
                oracle.clear();
                for (i, k) in keys.iter().enumerate() {
                    oracle.insert(*k, full_row(&w.arrays, i));
                }
                true
            }

            Op::MergeDelete { keys } => {
                let keys = dedup_keys(keys);
                if !keys.is_empty() {
                    let w = build_write(
                        &columns,
                        &[],
                        &keys,
                        None,
                        DataParams {
                            len: LenClass::Small,
                            nulls: NullClass::None,
                            nans: false,
                        },
                        0,
                        &mut rng,
                    );
                    let deleted = keys.iter().filter(|k| oracle.contains_key(k)).count() as u64;
                    let stats = merge(
                        &mut dataset,
                        WhenMatched::Delete,
                        WhenNotMatched::DoNothing,
                        WhenNotMatchedBySource::Keep,
                        false,
                        w.batch,
                    )
                    .await
                    .unwrap_or_else(|e| panic!("{ctx}: merge failed: {e}"));
                    stat_eq!(stats.num_deleted_rows, deleted, "{ctx}: deleted_rows");
                    for k in keys {
                        oracle.remove(&k);
                    }
                    true
                } else {
                    false
                }
            }

            Op::DeleteEq { key } => {
                let key = u64::from(*key);
                let expected = u64::from(oracle.contains_key(&key));
                let res = dataset
                    .delete(&format!("id = {key}"))
                    .await
                    .unwrap_or_else(|e| panic!("{ctx}: delete failed: {e}"));
                stat_eq!(res.num_deleted_rows, expected, "{ctx}: deleted_rows");
                oracle.remove(&key);
                true
            }

            Op::DeleteRange { lo, hi } => {
                let (lo, hi) = (u64::from(*lo.min(hi)), u64::from(*lo.max(hi)));
                let expected = oracle.range(lo..hi).count() as u64;
                let res = dataset
                    .delete(&format!("id >= {lo} AND id < {hi}"))
                    .await
                    .unwrap_or_else(|e| panic!("{ctx}: delete failed: {e}"));
                stat_eq!(res.num_deleted_rows, expected, "{ctx}: deleted_rows");
                oracle.retain(|k, _| !(*k >= lo && *k < hi));
                true
            }

            Op::DeleteNull { col } => {
                let j = (*col as usize) % n_cols;
                let name = columns[j].name.clone();
                let doomed: Vec<u64> = oracle
                    .iter()
                    .filter(|(_, row)| is_cell_null(&row[j]))
                    .map(|(k, _)| *k)
                    .collect();
                let res = dataset
                    .delete(&format!("{name} IS NULL"))
                    .await
                    .unwrap_or_else(|e| panic!("{ctx}: delete failed: {e}"));
                stat_eq!(
                    res.num_deleted_rows,
                    doomed.len() as u64,
                    "{ctx}: deleted_rows"
                );
                for k in doomed {
                    oracle.remove(&k);
                }
                true
            }

            Op::DeleteAll => {
                let expected = oracle.len() as u64;
                let res = dataset
                    .delete("true")
                    .await
                    .unwrap_or_else(|e| panic!("{ctx}: delete failed: {e}"));
                stat_eq!(res.num_deleted_rows, expected, "{ctx}: deleted_rows");
                oracle.clear();
                true
            }

            Op::UpdateLiteral { col, key } => {
                let friendly: Vec<usize> = (0..n_cols)
                    .filter(|j| lit_friendly(&columns[*j].dt))
                    .collect();
                if !friendly.is_empty() && !oracle.is_empty() {
                    let j = friendly[(*col as usize) % friendly.len()];
                    let k = *oracle
                        .keys()
                        .nth((*key as usize) % oracle.len())
                        .expect("nth key");
                    let g = step as u64;
                    let (lit, expected): (Lit, ArrayRef) = match &columns[j].dt {
                        DataType::Int8 | DataType::Int16 | DataType::Int32 | DataType::Int64 => {
                            let v = (g % 100) as i64;
                            let arr = cast(&Int64Array::from(vec![v]), &columns[j].dt)
                                .expect("cast literal");
                            (Lit::I(v.into()), arr)
                        }
                        DataType::UInt8
                        | DataType::UInt16
                        | DataType::UInt32
                        | DataType::UInt64 => {
                            let v = g % 100;
                            let arr = cast(&UInt64Array::from(vec![v]), &columns[j].dt)
                                .expect("cast literal");
                            (Lit::U(v.into()), arr)
                        }
                        DataType::Boolean => {
                            let v = g.is_multiple_of(2);
                            (Lit::B(v), Arc::new(BooleanArray::from(vec![v])) as ArrayRef)
                        }
                        DataType::Utf8 | DataType::LargeUtf8 => {
                            let v = format!("s{}", g % 997);
                            let arr = cast(&StringArray::from(vec![v.clone()]), &columns[j].dt)
                                .expect("cast literal");
                            (Lit::S(v), arr)
                        }
                        _ => unreachable!("filtered by lit_friendly"),
                    };
                    let res = UpdateBuilder::new(Arc::new(dataset.clone()))
                        .update_where(&format!("id = {k}"))
                        .unwrap_or_else(|e| panic!("{ctx}: update_where: {e}"))
                        .set(&columns[j].name, &lit.to_sql())
                        .unwrap_or_else(|e| panic!("{ctx}: set: {e}"))
                        .build()
                        .unwrap_or_else(|e| panic!("{ctx}: build: {e}"))
                        .execute()
                        .await
                        .unwrap_or_else(|e| panic!("{ctx}: update failed: {e}"));
                    stat_eq!(res.rows_updated, 1, "{ctx}: rows_updated");
                    dataset = (*res.new_dataset).clone();
                    oracle.get_mut(&k).expect("key present")[j] = (expected, 0);
                    true
                } else {
                    false
                }
            }

            Op::UpdateIdentity { col, threshold } => {
                let eligible: Vec<usize> = (0..n_cols)
                    .filter(|j| !matches!(columns[*j].spec, ColSpec::Null))
                    .collect();
                if !eligible.is_empty() {
                    let j = eligible[(*col as usize) % eligible.len()];
                    let t = u64::from(*threshold);
                    let expected = oracle.range(t..).count() as u64;
                    let res = UpdateBuilder::new(Arc::new(dataset.clone()))
                        .update_where(&format!("id >= {t}"))
                        .unwrap_or_else(|e| panic!("{ctx}: update_where: {e}"))
                        .set(&columns[j].name, &columns[j].name)
                        .unwrap_or_else(|e| panic!("{ctx}: set: {e}"))
                        .build()
                        .unwrap_or_else(|e| panic!("{ctx}: build: {e}"))
                        .execute()
                        .await
                        .unwrap_or_else(|e| panic!("{ctx}: update failed: {e}"));
                    stat_eq!(res.rows_updated, expected, "{ctx}: rows_updated");
                    dataset = (*res.new_dataset).clone();
                    // Values are unchanged; the oracle stays as-is.
                    true
                } else {
                    false
                }
            }

            Op::CreateIndex { col, kind } => {
                // Candidate columns: `id` plus every value column with a valid index kind.
                let pick = (*col as usize) % (n_cols + 1);
                let (name, kinds) = if pick == 0 {
                    ("id".to_owned(), scalar_index_kinds(&DataType::UInt64))
                } else {
                    let c = &columns[pick - 1];
                    (c.name.clone(), scalar_index_kinds(&c.dt))
                };
                let mut choices: Vec<IndexKind> =
                    kinds.into_iter().map(IndexKind::Scalar).collect();
                if pick > 0
                    && is_float32_vector(&columns[pick - 1])
                    && oracle.len() >= MIN_ROWS_FOR_VECTOR_INDEX
                    && has_non_null_value(&oracle, &columns, &name)
                {
                    choices.push(IndexKind::Vector);
                }
                if !choices.is_empty() {
                    let choice = choices[(*kind as usize) % choices.len()].clone();
                    let idx_name = format!("idx{next_index}");
                    next_index += 1;
                    let entry = IndexEntry {
                        name: idx_name,
                        col: name,
                        kind: choice,
                    };
                    create_index(&mut dataset, &entry)
                        .await
                        .unwrap_or_else(|e| {
                            panic!(
                                "{ctx}: create_index {} ({:?} on {}) failed: {e}",
                                entry.name, entry.kind, entry.col
                            )
                        });
                    indices.push(entry);
                    // A fresh index covers every current fragment, so this is the one moment
                    // the enabled scan plan is guaranteed to reference the index when the
                    // predicate is indexable — check the plan here (`check_plan`).
                    let entry = indices.last().expect("just pushed");
                    let when = format!("step {step}");
                    verify_index_query(
                        &dataset,
                        &oracle,
                        &columns,
                        &indices,
                        entry,
                        IndexQueryCheck {
                            check_plan: true,
                            salt: step,
                            when: &when,
                        },
                    )
                    .await;
                    true
                } else {
                    false
                }
            }

            Op::DropIndex { sel } => {
                if !indices.is_empty() {
                    let i = (*sel as usize) % indices.len();
                    let entry = indices.remove(i);
                    dataset
                        .drop_index(&entry.name)
                        .await
                        .unwrap_or_else(|e| panic!("{ctx}: drop_index({}): {e}", entry.name));
                    true
                } else {
                    false
                }
            }

            Op::OptimizeIndices { sel } => {
                let mut options = OptimizeOptions::default();
                options.num_indices_to_merge = [None, Some(1), Some(2)][(*sel as usize) % 3];
                dataset
                    .optimize_indices(&options)
                    .await
                    .unwrap_or_else(|e| panic!("{ctx}: optimize_indices failed: {e}"));
                true
            }

            Op::Compact {
                sel,
                materialize,
                defer_remap,
            } => {
                let opts = compaction_profile.options(
                    [8, 64, 512][(*sel as usize) % 3],
                    *materialize,
                    *defer_remap,
                    dataset.manifest().uses_stable_row_ids(),
                );
                compact_files(&mut dataset, opts.clone(), None)
                    .await
                    .map(|_| ())
                    .unwrap_or_else(|e| panic!("{ctx}: compaction failed with {opts:?}: {e}"));
                true
            }

            Op::AddColumn { fill } => {
                if columns.len() < MAX_TOTAL_COLS {
                    let name = format!("v{next_col}");
                    next_col += 1;
                    match fill {
                        AddFill::Nulls(spec) => {
                            let dt = make_generator(spec, LenClass::Small, 0.0)
                                .data_type()
                                .clone();
                            // Sparse sessions keep requesting the sparse layout for columns
                            // added later, or the coverage would silently decay to dense as
                            // the schema evolves. Always safe here: sparse implies the session
                            // is pinned to file version 2.3 (no migration past it, no restore
                            // before it), and eligibility is re-checked per field.
                            let field = Field::new(&name, dt.clone(), true);
                            let field = if sparse {
                                with_sparse_request(field)
                            } else {
                                field
                            };
                            dataset
                                .add_columns(
                                    NewColumnTransform::AllNulls(Arc::new(Schema::new(vec![
                                        field,
                                    ]))),
                                    None,
                                    None,
                                )
                                .await
                                .unwrap_or_else(|e| panic!("{ctx}: add_columns failed: {e}"));
                            let null_cell = all_null_cell(&dt);
                            for row in oracle.values_mut() {
                                row.push(null_cell.clone());
                            }
                            columns.push(Column {
                                name,
                                spec: spec.clone(),
                                dt,
                            });
                        }
                        AddFill::IdDerived => {
                            // Stays dense even in sparse sessions: SqlExpressions carries no
                            // field metadata, so there is nowhere to attach the request.
                            // NULLIF keeps the inferred column nullable; a plain expression over
                            // `id` would be non-null and later null writes would be rejected.
                            dataset
                                .add_columns(
                                    NewColumnTransform::SqlExpressions(vec![(
                                        name.clone(),
                                        "CAST(NULLIF(id * 2 + 7, 7) AS BIGINT)".to_owned(),
                                    )]),
                                    None,
                                    None,
                                )
                                .await
                                .unwrap_or_else(|e| panic!("{ctx}: add_columns failed: {e}"));
                            let arr: ArrayRef = Arc::new(Int64Array::from_iter(
                                oracle
                                    .keys()
                                    .map(|k| (*k != 0).then(|| (*k * 2 + 7) as i64)),
                            ));
                            for (i, row) in oracle.values_mut().enumerate() {
                                row.push((arr.clone(), i));
                            }
                            columns.push(Column {
                                name,
                                spec: ColSpec::Prim(PrimSpec::Int64),
                                dt: DataType::Int64,
                            });
                        }
                    }
                    true
                } else {
                    false
                }
            }

            Op::DropColumn { sel } => {
                let droppable: Vec<usize> = (0..n_cols)
                    .filter(|j| !indices.iter().any(|e| e.col == columns[*j].name))
                    .collect();
                if n_cols >= 2 && !droppable.is_empty() {
                    let j = droppable[(*sel as usize) % droppable.len()];
                    let name = columns[j].name.clone();
                    dataset
                        .drop_columns(&[&name])
                        .await
                        .unwrap_or_else(|e| panic!("{ctx}: drop_columns failed: {e}"));
                    columns.remove(j);
                    for row in oracle.values_mut() {
                        row.remove(j);
                    }
                    true
                } else {
                    false
                }
            }

            Op::Migrate => {
                let current = dataset_file_version(&dataset);
                assert_eq!(
                    current, expected_file_version,
                    "{ctx}: current file version disagrees with oracle"
                );
                // The migration overwrite discards index data, and the rebuild retrains every
                // index — which fails for a vector index whose column has gone all-null since
                // creation (KMeans cannot train on zero vectors). Skip the candidate in that
                // state instead of filing the training error as a migration finding.
                let rebuildable = indices.iter().all(|entry| {
                    !matches!(entry.kind, IndexKind::Vector)
                        || has_non_null_value(&oracle, &columns, &entry.col)
                });
                let target = if oracle.is_empty() || !rebuildable {
                    None
                } else {
                    next_file_version(current)
                };
                if let Some(target) = target {
                    let schema_before = schema_signature(&dataset);
                    dataset = migrate_dataset(&dataset, target, write_profile)
                        .await
                        .unwrap_or_else(|e| {
                            panic!("{ctx}: migration {current} -> {target} failed: {e}")
                        });
                    assert_eq!(
                        dataset_file_version(&dataset),
                        target,
                        "{ctx}: migration did not commit the requested file version"
                    );
                    assert_eq!(
                        schema_signature(&dataset),
                        schema_before,
                        "{ctx}: migration changed the logical schema"
                    );
                    assert_data_file_versions(&dataset, target, &ctx);

                    // The overwrite must discard the old index data, which is why the rebuild
                    // below is a separate step: asserting the discard first is the only way to
                    // notice Lance keeping index metadata that no longer matches the fragments.
                    assert!(
                        load_index_names(&dataset, &ctx).await.is_empty(),
                        "{ctx}: overwrite migration retained index metadata"
                    );
                    rebuild_indices(&mut dataset, &indices, &ctx).await;
                    let mut expected_names: Vec<String> =
                        indices.iter().map(|entry| entry.name.clone()).collect();
                    expected_names.sort_unstable();
                    // Compared as sequences, not sets: a duplicate entry under a live index's
                    // name is a defect the set comparison would swallow.
                    assert_eq!(
                        load_index_names(&dataset, &ctx).await,
                        expected_names,
                        "{ctx}: migration did not rebuild every index exactly once"
                    );

                    let when = format!("{ctx}: rebuilt index");
                    for (i, entry) in indices.iter().enumerate() {
                        verify_index_query(
                            &dataset,
                            &oracle,
                            &columns,
                            &indices,
                            entry,
                            IndexQueryCheck {
                                check_plan: true,
                                salt: step + i,
                                when: &when,
                            },
                        )
                        .await;
                    }
                    expected_file_version = target;
                    dataset
                        .validate()
                        .await
                        .unwrap_or_else(|e| panic!("{ctx}: validate after migration failed: {e}"));
                    true
                } else {
                    false
                }
            }

            Op::Restore { sel } => {
                let target = &snapshots[(*sel as usize) % snapshots.len()];
                if target.version != dataset.version().version {
                    let mut old = dataset
                        .checkout_version(target.version)
                        .await
                        .unwrap_or_else(|e| panic!("{ctx}: checkout_version failed: {e}"));
                    old.restore()
                        .await
                        .unwrap_or_else(|e| panic!("{ctx}: restore failed: {e}"));
                    dataset = old;
                    expected_file_version = target.file_version;
                    oracle = target.oracle.clone();
                    columns = target.columns.clone();
                    indices = target.indices.clone();
                    true
                } else {
                    false
                }
            }

            Op::Read(read) => execute_read(&dataset, &oracle, &columns, read, step, &mut rng).await,
        };

        if !executed {
            continue;
        }

        assert_eq!(
            dataset_file_version(&dataset),
            expected_file_version,
            "{ctx}: file version disagrees with oracle after operation"
        );
        ever_inserted.extend(oracle.keys().copied());

        // Record every commit so `Restore` can roll back to it.
        let v = dataset.version().version;
        if snapshots.last().map(|s| s.version) != Some(v) {
            snapshots.push(Snapshot {
                version: v,
                file_version: expected_file_version,
                oracle: oracle.clone(),
                columns: columns.clone(),
                indices: indices.clone(),
            });
        }

        let when = format!("step {step}");
        verify_full_scan(&dataset, &oracle, &columns, &when).await;
        verify_probes(&dataset, &oracle, &columns, &ever_inserted, &when).await;

        // One rotating index per step: exercises index maintenance after mutations without
        // multiplying every operation by every live index.
        if !indices.is_empty() {
            let entry = &indices[step % indices.len()];
            verify_index_query(
                &dataset,
                &oracle,
                &columns,
                &indices,
                entry,
                IndexQueryCheck {
                    check_plan: false,
                    salt: step,
                    when: &when,
                },
            )
            .await;
        }
        executed_ops += 1;
    }

    // The rotation above only samples one index per step; at session end every live index
    // must still answer its compatible query.
    for (i, entry) in indices.iter().enumerate() {
        verify_index_query(
            &dataset,
            &oracle,
            &columns,
            &indices,
            entry,
            IndexQueryCheck {
                check_plan: false,
                salt: executed_ops + i,
                when: "end",
            },
        )
        .await;
    }

    // Structural invariants (manifest, fragments, index metadata) that the logical row oracle
    // cannot see. Once per session: a per-operation validate would dominate the runtime.
    dataset
        .validate()
        .await
        .unwrap_or_else(|e| panic!("end: validate failed: {e}"));

    // File-backed profiles: reopen the same location through a fresh handle and rerun the
    // final logical verification, so a fresh-open failure and a reopened-state mismatch are
    // reported distinctly from the mutation-time checks above. Memory mode is skipped because
    // a new `memory://` resolution is not guaranteed to share the object store instance.
    if storage != Storage::Memory {
        let reopened = Dataset::open(&location)
            .await
            .unwrap_or_else(|e| panic!("end: reopen of {location} failed: {e}"));
        verify_full_scan(&reopened, &oracle, &columns, "reopen").await;
        verify_probes(&reopened, &oracle, &columns, &ever_inserted, "reopen").await;
    }

    executed_ops
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vector_training_precondition_tracks_non_null_values() {
        let arr: ArrayRef = Arc::new(Int64Array::from(vec![None, Some(7)]));
        let columns = [Column {
            name: "v0".to_owned(),
            spec: ColSpec::Prim(PrimSpec::Int64),
            dt: DataType::Int64,
        }];
        let all_null = Oracle::from([(1, vec![(arr.clone(), 0)])]);
        let mixed = Oracle::from([(1, vec![(arr.clone(), 0)]), (2, vec![(arr, 1)])]);

        assert!(!has_non_null_value(&all_null, &columns, "v0"));
        assert!(has_non_null_value(&mixed, &columns, "v0"));
        assert!(!has_non_null_value(&mixed, &columns, "missing"));
    }

    #[tokio::test]
    async fn ineligible_candidates_do_not_consume_the_operation_limit() {
        let mut ops = vec![Op::DropIndex { sel: 0 }; MAX_OPS];
        ops.push(Op::Append {
            keys: vec![1],
            data: DataParams {
                len: LenClass::Small,
                nulls: NullClass::None,
                nans: false,
            },
        });

        let executed = run_inner(FuzzInput {
            seed: 0,
            schema: vec![ColSpec::Prim(PrimSpec::Int64)],
            ops,
        })
        .await;

        assert_eq!(executed, 1);
    }

    #[tokio::test]
    async fn sparse_sessions_apply_the_request_to_added_columns() {
        // Seed bits 24-25 select file version 2.3, bits 20-21 request sparse encoding, so the
        // added column is marked sparse and the writes after it run through the sparse layout.
        let seed = (3 << 24) | (1 << 20);
        let data = DataParams {
            len: LenClass::Small,
            nulls: NullClass::Some,
            nans: false,
        };

        let executed = run_inner(FuzzInput {
            seed,
            schema: vec![ColSpec::Prim(PrimSpec::Int64)],
            ops: vec![
                Op::Append {
                    keys: vec![1, 2, 3],
                    data,
                },
                Op::AddColumn {
                    fill: AddFill::Nulls(ColSpec::Prim(PrimSpec::Utf8)),
                },
                Op::Append {
                    keys: vec![4, 5],
                    data,
                },
                Op::Compact {
                    sel: 0,
                    materialize: true,
                    defer_remap: false,
                },
            ],
        })
        .await;

        assert_eq!(executed, 4);
    }

    #[tokio::test]
    async fn migration_rebuilds_live_indices() {
        let executed = run_inner(FuzzInput {
            seed: 0,
            schema: vec![ColSpec::Prim(PrimSpec::Int64)],
            ops: vec![
                Op::Append {
                    keys: vec![1, 2],
                    data: DataParams {
                        len: LenClass::Small,
                        nulls: NullClass::None,
                        nans: false,
                    },
                },
                Op::CreateIndex { col: 0, kind: 0 },
                Op::Migrate,
            ],
        })
        .await;

        assert_eq!(executed, 3);
    }
}
