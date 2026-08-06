//! Generated read operations, each checked against the model.

use std::collections::BTreeSet;

use arrow::array::{AsArray as _, UInt32Array};
use arrow::compute::take;
use arrow::datatypes::{DataType, UInt64Type};
use lance::dataset::ProjectionRequest;
use lance::Dataset;
use lance_datagen::{array, RowCount};
use rand_xoshiro::Xoshiro256PlusPlus;

use super::ops::{PredSpec, ReadOp};
use super::oracle::{
    all_names, assert_arrays_eq, cell_lit, concat_or_empty, is_cell_null, is_float_vector,
    lit_friendly, Column, Lit, Oracle,
};
use super::scan::{collect_ids, collect_scan};

/// Resolve a predicate spec to SQL plus the key set the oracle says must match.
fn resolve_pred(pred: &PredSpec, columns: &[Column], oracle: &Oracle) -> (String, BTreeSet<u64>) {
    let null_pred = |col: u8, negate: bool| {
        let j = (col as usize) % columns.len();
        let name = &columns[j].name;
        let sql = if negate {
            format!("{name} IS NOT NULL")
        } else {
            format!("{name} IS NULL")
        };
        let expected = oracle
            .iter()
            .filter(|(_, row)| is_cell_null(&row[j]) != negate)
            .map(|(k, _)| *k)
            .collect();
        (sql, expected)
    };
    match pred {
        PredSpec::IdRange { lo, hi } => {
            let (lo, hi) = (u64::from(*lo.min(hi)), u64::from(*lo.max(hi)));
            let expected = oracle.range(lo..hi).map(|(k, _)| *k).collect();
            (format!("id >= {lo} AND id < {hi}"), expected)
        }
        PredSpec::IsNull { col } => null_pred(*col, false),
        PredSpec::NotNull { col } => null_pred(*col, true),
        PredSpec::Eq { col, key } => {
            let j = (*col as usize) % columns.len();
            let c = &columns[j];
            if lit_friendly(&c.dt) && !oracle.is_empty() {
                let row = oracle
                    .values()
                    .nth((*key as usize) % oracle.len())
                    .expect("nth row");
                if let Some(lit) = cell_lit(&c.dt, &row[j]).filter(Lit::sql_safe) {
                    let expected = oracle
                        .iter()
                        .filter(|(_, row)| cell_lit(&c.dt, &row[j]).as_ref() == Some(&lit))
                        .map(|(k, _)| *k)
                        .collect();
                    return (format!("{} = {}", c.name, lit.to_sql()), expected);
                }
            }
            null_pred(*col, false)
        }
    }
}

/// Runs one read op. Returns `false` if its preconditions were not met, so nothing ran.
pub(super) async fn execute_read(
    dataset: &Dataset,
    oracle: &Oracle,
    columns: &[Column],
    read: &ReadOp,
    step: usize,
    rng: &mut Xoshiro256PlusPlus,
) -> bool {
    let ctx = format!("step {step}: read {read:?}");
    match read {
        ReadOp::Filtered { pred, prefilter } => {
            let (sql, expected) = resolve_pred(pred, columns, oracle);
            for use_si in [true, false] {
                let ids = collect_ids(dataset, &sql, use_si, *prefilter, &ctx).await;
                assert_eq!(
                    ids, expected,
                    "{ctx}: filter `{sql}` (scalar_index={use_si}) disagrees with oracle"
                );
            }
            true
        }

        ReadOp::Count { pred } => {
            let (sql, expected) = resolve_pred(pred, columns, oracle);
            let count = dataset
                .count_rows(Some(sql.clone()))
                .await
                .unwrap_or_else(|e| panic!("{ctx}: count_rows failed: {e}"));
            assert_eq!(count, expected.len(), "{ctx}: count_rows(`{sql}`)");
            true
        }

        ReadOp::Limit { sel, mask } => {
            // Project a column subset (always keeping id), then compare a limit/offset scan
            // against the corresponding slice of the unlimited scan.
            let mut names = vec!["id".to_owned()];
            for (j, c) in columns.iter().enumerate() {
                if mask & (1 << (j % 8)) != 0 {
                    names.push(c.name.clone());
                }
            }
            let full = concat_or_empty(
                &collect_scan(dataset, |scan| scan.project(&names).map(|_| ()), &ctx).await,
                &names,
                columns,
            );
            let n = full.num_rows() as i64;
            let limit = i64::from(*sel) % (n + 2);
            let offset = (i64::from(*sel) / 3) % (n + 1);
            let batch_size = [None, Some(1), Some(7), Some(100)][(*sel as usize) % 4];
            let limited = concat_or_empty(
                &collect_scan(
                    dataset,
                    |scan| {
                        scan.project(&names)?;
                        scan.limit(Some(limit), Some(offset))?;
                        if let Some(b) = batch_size {
                            scan.batch_size(b);
                        }
                        Ok(())
                    },
                    &ctx,
                )
                .await,
                &names,
                columns,
            );
            let expected_len = limit.min(n - offset.min(n)).max(0) as usize;
            assert_eq!(limited.num_rows(), expected_len, "{ctx}: limited row count");
            let expected = full.slice(offset.min(n) as usize, expected_len);
            for (j, name) in names.iter().enumerate().take(expected.num_columns()) {
                assert_arrays_eq(expected.column(j), limited.column(j), &ctx, name);
            }
            true
        }

        ReadOp::TakeOffsets { sel } => {
            if oracle.is_empty() {
                false
            } else {
                let names = all_names(columns);
                let full = concat_or_empty(
                    &collect_scan(dataset, |scan| scan.project(&names).map(|_| ()), &ctx).await,
                    &names,
                    columns,
                );
                let n = full.num_rows();
                let offsets: Vec<u64> = sel.iter().map(|b| u64::from(*b) % n as u64).collect();
                let taken = dataset
                    .take(
                        &offsets,
                        ProjectionRequest::from_columns(&names, dataset.schema()),
                    )
                    .await
                    .unwrap_or_else(|e| panic!("{ctx}: take failed: {e}"));
                let idx = UInt32Array::from(offsets.iter().map(|o| *o as u32).collect::<Vec<_>>());
                for (j, name) in names.iter().enumerate() {
                    let expected =
                        take(full.column(j).as_ref(), &idx, None).expect("take expected");
                    let actual = taken
                        .column_by_name(name)
                        .unwrap_or_else(|| panic!("{ctx}: column {name} missing from take"));
                    assert_arrays_eq(&expected, actual, &ctx, name);
                }
                true
            }
        }

        // TODO: strengthen the KNN oracle with a flat reference search. The checks below are
        // sound but weak: they verify result safety (live, unique, non-null) and an upper
        // cardinality bound, so an empty result passes even when matching rows exist. The
        // looseness is NOT because "IVF is approximate" in general — the only vector index the
        // harness builds is IVF_FLAT with a single partition, so an nprobes=1 probe scans every
        // indexed row. A stronger oracle must instead model what actually makes results diverge:
        // - rows written after the index was built are served by a separate flat-search path
        //   and merged with the indexed results;
        // - tied distances, where any of the tied rows is a correct answer;
        // - null and non-finite (NaN/inf) vector values, whose distance semantics are
        //   engine-specific;
        // and only then compare against a flat (exact) reference over the row model.
        ReadOp::Knn { col, k } => {
            let vectors: Vec<usize> = (0..columns.len())
                .filter(|j| is_float_vector(&columns[*j]))
                .collect();
            if !vectors.is_empty() {
                let j = vectors[(*col as usize) % vectors.len()];
                let c = &columns[j];
                let k = 1 + usize::from(*k % 16);
                let DataType::FixedSizeList(child, dim) = &c.dt else {
                    unreachable!("checked by is_float_vector");
                };
                let query = array::rand_type(child.data_type())
                    .generate(RowCount::from(*dim as u64), rng)
                    .expect("datagen knn query");
                let batches = collect_scan(
                    dataset,
                    |scan| {
                        scan.nearest(&c.name, query.as_ref(), k)?;
                        scan.nprobes(1);
                        scan.project(&["id"])?;
                        Ok(())
                    },
                    &ctx,
                )
                .await;
                let live_non_null = oracle.values().filter(|row| !is_cell_null(&row[j])).count();
                let mut seen = BTreeSet::new();
                for batch in &batches {
                    let ids = batch
                        .column_by_name("id")
                        .unwrap_or_else(|| panic!("{ctx}: id missing from knn output"))
                        .as_primitive::<UInt64Type>();
                    for i in 0..ids.len() {
                        let id = ids.value(i);
                        assert!(seen.insert(id), "{ctx}: duplicate id {id} in knn output");
                        let row = oracle
                            .get(&id)
                            .unwrap_or_else(|| panic!("{ctx}: knn returned dead id {id}"));
                        assert!(
                            !is_cell_null(&row[j]),
                            "{ctx}: knn returned id {id} whose vector is null"
                        );
                    }
                }
                // Upper bound only: a knn search may return *fewer* than min(k, live) rows and
                // still be correct — see the TODO above for what an exact expectation would
                // have to model. The per-row checks above (unique, live, non-null) catch the
                // real defects; requiring an exact count here is a false positive.
                assert!(
                    seen.len() <= k.min(live_non_null),
                    "{ctx}: knn returned {} rows, more than min(k={k}, live={live_non_null})",
                    seen.len()
                );
                true
            } else {
                false
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow::array::{ArrayRef, Int64Array};

    use super::*;
    use crate::dataset_ops::ops::{ColSpec, PrimSpec};

    #[test]
    fn predicates_resolve_to_exact_model_keys() {
        let values: ArrayRef = Arc::new(Int64Array::from(vec![Some(7), None, Some(7)]));
        let oracle = Oracle::from([
            (1, vec![(values.clone(), 0)]),
            (2, vec![(values.clone(), 1)]),
            (3, vec![(values, 2)]),
        ]);
        let columns = [Column {
            name: "v0".to_owned(),
            spec: ColSpec::Prim(PrimSpec::Int64),
            dt: DataType::Int64,
        }];

        assert_eq!(
            resolve_pred(&PredSpec::Eq { col: 0, key: 0 }, &columns, &oracle),
            ("v0 = 7".to_owned(), BTreeSet::from([1, 3]))
        );
        assert_eq!(
            resolve_pred(&PredSpec::IsNull { col: 0 }, &columns, &oracle),
            ("v0 IS NULL".to_owned(), BTreeSet::from([2]))
        );
    }
}
