//! Index creation and the per-kind index query oracle.

use std::collections::BTreeSet;

use arrow::array::AsArray as _;
use arrow::datatypes::{DataType, UInt64Type};
use lance::index::vector::VectorIndexParams;
use lance::index::DatasetIndexExt as _;
use lance::Dataset;
use lance_index::scalar::inverted::tokenizer::InvertedIndexParams;
use lance_index::scalar::{BuiltinIndexType, FullTextSearchQuery, ScalarIndexParams};
use lance_index::IndexType;
use lance_linalg::distance::MetricType;

use super::oracle::{cell_lit, cell_str, is_cell_null, list_cells, probe_lit, Column, Lit, Oracle};
use super::scan::{collect_ids, collect_scan};

/// Index flavors the harness generates.
#[derive(Clone, Debug, PartialEq)]
pub(super) enum IndexKind {
    Scalar(BuiltinIndexType),
    Vector,
}

/// One live index the model tracks, so later checks know what to query and how.
#[derive(Clone)]
pub(super) struct IndexEntry {
    pub(super) name: String,
    pub(super) col: String,
    pub(super) kind: IndexKind,
}

/// Create one generated index on `dataset`, with the fixed parameters the oracle assumes.
pub(super) async fn create_index(
    dataset: &mut Dataset,
    entry: &IndexEntry,
) -> Result<(), lance::Error> {
    match &entry.kind {
        // Inverted indices need tokenizer parameters that a bare scalar configuration omits.
        IndexKind::Scalar(BuiltinIndexType::Inverted) => dataset
            .create_index(
                &[&entry.col],
                IndexType::Inverted,
                Some(entry.name.clone()),
                &inverted_index_params(),
                true,
            )
            .await
            .map(|_| ()),
        IndexKind::Scalar(kind) => dataset
            .create_index(
                &[&entry.col],
                builtin_to_index_type(kind.clone()),
                Some(entry.name.clone()),
                &ScalarIndexParams::for_builtin(kind.clone()),
                true,
            )
            .await
            .map(|_| ()),
        IndexKind::Vector => dataset
            .create_index(
                &[&entry.col],
                IndexType::Vector,
                Some(entry.name.clone()),
                &VectorIndexParams::ivf_flat(1, MetricType::L2),
                true,
            )
            .await
            .map(|_| ()),
    }
}

/// Rebuild every live index, after a migration overwrite discarded the old index data.
/// Reported separately from the migration itself so a rebuild failure is not filed as one.
pub(super) async fn rebuild_indices(dataset: &mut Dataset, indices: &[IndexEntry], ctx: &str) {
    for entry in indices {
        create_index(dataset, entry).await.unwrap_or_else(|e| {
            panic!(
                "{ctx}: rebuilding index {} ({:?} on {}) failed: {e}",
                entry.name, entry.kind, entry.col
            )
        });
    }
}

/// Index kinds Lance accepts for `dt`, in the order the input selector indexes into. Empty when
/// no scalar index applies.
pub(super) fn scalar_index_kinds(dt: &DataType) -> Vec<BuiltinIndexType> {
    use BuiltinIndexType::{Bitmap, BloomFilter, Inverted, LabelList, NGram, ZoneMap};
    let btree = BuiltinIndexType::BTree;
    match dt {
        DataType::Int8
        | DataType::Int16
        | DataType::Int32
        | DataType::Int64
        | DataType::UInt8
        | DataType::UInt16
        | DataType::UInt32
        | DataType::UInt64
        | DataType::Float32
        | DataType::Float64
        | DataType::Date32
        | DataType::Date64
        | DataType::Time32(_)
        | DataType::Time64(_)
        | DataType::Timestamp(_, _) => vec![btree, Bitmap, ZoneMap, BloomFilter],
        DataType::Duration(_) | DataType::Decimal128(_, _) => vec![btree, Bitmap, ZoneMap],
        DataType::Boolean => vec![Bitmap],
        DataType::Utf8 | DataType::LargeUtf8 => {
            vec![btree, Bitmap, ZoneMap, BloomFilter, NGram, Inverted]
        }
        DataType::Binary | DataType::LargeBinary => vec![btree, Bitmap, BloomFilter],
        DataType::List(f) | DataType::LargeList(f) => match f.data_type() {
            DataType::Utf8 | DataType::LargeUtf8 => vec![LabelList, Inverted],
            DataType::Int8
            | DataType::Int16
            | DataType::Int32
            | DataType::Int64
            | DataType::UInt8
            | DataType::UInt16
            | DataType::UInt32
            | DataType::UInt64 => vec![LabelList],
            _ => vec![],
        },
        _ => vec![],
    }
}

/// Inverted-index parameters the model can reproduce exactly: the raw tokenizer emits the
/// whole document as one token and every filter is disabled, so a match query for term `t`
/// returns precisely the rows whose document equals `t`. The default simple tokenizer
/// (lower-casing, stemming, stop words, ascii folding, 40-byte token cap) has no practical
/// independent oracle.
fn inverted_index_params() -> InvertedIndexParams {
    InvertedIndexParams::default()
        .base_tokenizer("raw".to_owned())
        .max_token_length(None)
        .lower_case(false)
        .stem(false)
        .remove_stop_words(false)
        .ascii_folding(false)
}

fn builtin_to_index_type(b: BuiltinIndexType) -> IndexType {
    match b {
        BuiltinIndexType::BTree => IndexType::BTree,
        BuiltinIndexType::Bitmap => IndexType::Bitmap,
        BuiltinIndexType::LabelList => IndexType::LabelList,
        BuiltinIndexType::NGram => IndexType::NGram,
        BuiltinIndexType::ZoneMap => IndexType::ZoneMap,
        BuiltinIndexType::BloomFilter => IndexType::BloomFilter,
        BuiltinIndexType::Inverted => IndexType::Inverted,
        _ => unreachable!("kind not produced by scalar_index_kinds"),
    }
}

enum IndexProbe {
    /// SQL filter, run with scalar-index use enabled and disabled. `indexable` marks
    /// predicates Lance's query parser for this index kind is known to accept; only those
    /// participate in the plan check.
    Filter {
        sql: String,
        expected: BTreeSet<u64>,
        indexable: bool,
    },
    /// Full-text match query (Inverted only). Under the raw tokenizer a document matches
    /// iff it equals the term exactly.
    Fts {
        term: String,
        expected: BTreeSet<u64>,
    },
}

/// Deterministic `contains()` needle drawn from a model string: odd salts probe a middle
/// slice (several rows can share it, and partial matches go through the ngram lookup),
/// even salts and short strings use the whole value.
fn ngram_needle(s: &str, salt: usize) -> String {
    let bounds: Vec<usize> = s.char_indices().map(|(i, _)| i).chain([s.len()]).collect();
    let n = bounds.len() - 1; // chars in s
    if salt.is_multiple_of(2) || n < 2 {
        return s.to_owned();
    }
    s[bounds[n / 4]..bounds[(3 * n).div_ceil(4)]].to_owned()
}

/// Build the kind-compatible probe for one index. `salt` rotates which modeled value the
/// query targets.
fn build_index_probe(
    kind: &BuiltinIndexType,
    col: &str,
    columns: &[Column],
    oracle: &Oracle,
    salt: usize,
) -> IndexProbe {
    use BuiltinIndexType as B;

    // `id` is non-null u64, so it only ever gets equality-family indices.
    if col == "id" {
        return match oracle.keys().nth(salt % oracle.len().max(1)) {
            Some(k) => IndexProbe::Filter {
                sql: format!("id = {k}"),
                expected: BTreeSet::from([*k]),
                indexable: true,
            },
            None => IndexProbe::Filter {
                sql: "id = 1000".to_owned(),
                expected: BTreeSet::new(),
                indexable: false,
            },
        };
    }

    let j = columns
        .iter()
        .position(|c| c.name == col)
        .unwrap_or_else(|| panic!("indexed column {col} not in schema"));
    let dt = &columns[j].dt;

    // Fallback when the model holds no usable literal: a null predicate is valid on every
    // column type, though only the equality-family index parsers accept it.
    let null_probe = |indexable: bool| IndexProbe::Filter {
        sql: format!("{col} IS NULL"),
        expected: oracle
            .iter()
            .filter(|(_, row)| is_cell_null(&row[j]))
            .map(|(k, _)| *k)
            .collect(),
        indexable,
    };

    match kind {
        B::BTree | B::Bitmap | B::ZoneMap | B::BloomFilter => {
            // `probe_lit`, not `cell_lit`: temporal and decimal columns equality-probe here
            // (via constant-folded `arrow_cast`) even though they stay out of `lit_friendly`.
            let lits: Vec<Lit> = oracle
                .values()
                .filter_map(|row| probe_lit(dt, &row[j]).filter(Lit::sql_safe))
                .collect();
            if lits.is_empty() {
                // IsNull is sargable for all four equality-family kinds.
                null_probe(true)
            } else if matches!(lits[0], Lit::Cast { .. }) && !salt.is_multiple_of(2) {
                // Temporal and decimal columns alternate with the IS NULL probe that was their
                // only probe before equality probing existed: equality alone would drop the
                // null-entry staleness coverage that caught real ZoneMap defects.
                null_probe(true)
            } else {
                let lit = lits[salt % lits.len()].clone();
                IndexProbe::Filter {
                    sql: format!("{col} = {}", lit.to_sql()),
                    expected: oracle
                        .iter()
                        .filter(|(_, row)| probe_lit(dt, &row[j]).as_ref() == Some(&lit))
                        .map(|(k, _)| *k)
                        .collect(),
                    // Duration and Decimal128 equality executes correctly but is not sargable:
                    // lance-datafusion's safe_coerce_scalar has no source arm for those scalar
                    // variants, so the planner leaves the predicate as a full filter and the
                    // plan check cannot apply.
                    indexable: !matches!(dt, DataType::Duration(_) | DataType::Decimal128(_, _)),
                }
            }
        }
        B::NGram => {
            let strs: Vec<String> = oracle
                .values()
                .filter_map(|row| cell_str(dt, &row[j]))
                .filter(|s| !s.is_empty())
                .collect();
            if strs.is_empty() {
                null_probe(false)
            } else {
                let needle = ngram_needle(&strs[salt % strs.len()], salt);
                IndexProbe::Filter {
                    sql: format!("contains({col}, {})", Lit::S(needle.clone()).to_sql()),
                    expected: oracle
                        .iter()
                        .filter(|(_, row)| {
                            cell_str(dt, &row[j]).is_some_and(|v| v.contains(&needle))
                        })
                        .map(|(k, _)| *k)
                        .collect(),
                    indexable: true,
                }
            }
        }
        B::LabelList => {
            let child_dt = match dt {
                DataType::List(f) | DataType::LargeList(f) => f.data_type().clone(),
                _ => unreachable!("LabelList only generated on list columns"),
            };
            let lits: Vec<Lit> = oracle
                .values()
                .filter_map(|row| list_cells(&row[j]))
                .flatten()
                .filter_map(|c| cell_lit(&child_dt, &c).filter(Lit::sql_safe))
                .collect();
            if lits.is_empty() {
                null_probe(false)
            } else {
                let lit = lits[salt % lits.len()].clone();
                // Cast the element literal to the child type. A bare integer literal is
                // Int64, and DataFusion coerces by casting the *column* instead, which hides
                // the column reference from the LabelList index parser. `arrow_cast` is
                // constant-folded, so the column stays a plain reference.
                let elem = match &child_dt {
                    DataType::Utf8 | DataType::LargeUtf8 => lit.to_sql(),
                    _ => format!("arrow_cast({}, '{child_dt}')", lit.to_sql()),
                };
                IndexProbe::Filter {
                    sql: format!("array_has({col}, {elem})"),
                    expected: oracle
                        .iter()
                        .filter(|(_, row)| {
                            list_cells(&row[j]).is_some_and(|elems| {
                                elems
                                    .iter()
                                    .any(|c| cell_lit(&child_dt, c).as_ref() == Some(&lit))
                            })
                        })
                        .map(|(k, _)| *k)
                        .collect(),
                    indexable: true,
                }
            }
        }
        B::Inverted => {
            // Exact-match FTS oracle for plain string columns only. List columns have no
            // consistent expectation: the index build joins list elements into one
            // space-separated document before tokenizing, while the flat-match path over
            // not-yet-indexed fragments tokenizes each element separately — under the raw
            // tokenizer the same row matches different terms depending on fragment state.
            let docs: Vec<String> = if matches!(dt, DataType::Utf8 | DataType::LargeUtf8) {
                oracle
                    .values()
                    .filter_map(|row| cell_str(dt, &row[j]))
                    .filter(|s| !s.is_empty())
                    .collect()
            } else {
                Vec::new()
            };
            if docs.is_empty() {
                null_probe(false)
            } else {
                let term = docs[salt % docs.len()].clone();
                IndexProbe::Fts {
                    expected: oracle
                        .iter()
                        .filter(|(_, row)| cell_str(dt, &row[j]).as_deref() == Some(&term))
                        .map(|(k, _)| *k)
                        .collect(),
                    term,
                }
            }
        }
        _ => unreachable!("kind not produced by scalar_index_kinds"),
    }
}

async fn explain_filter_plan(dataset: &Dataset, filter: &str, ctx: &str) -> String {
    let mut scan = dataset.scan();
    scan.project(&["id"])
        .and_then(|s| s.filter(filter))
        .unwrap_or_else(|e| panic!("{ctx}: plan config failed: {e}"));
    scan.use_scalar_index(true);
    scan.explain_plan(true)
        .await
        .unwrap_or_else(|e| panic!("{ctx}: explain_plan failed: {e}"))
}

/// How to run one index check: whether to require the plan to use an index, which modeled value
/// to target (`salt` rotates it), and the session context for failure messages.
pub(super) struct IndexQueryCheck<'a> {
    pub(super) check_plan: bool,
    pub(super) salt: usize,
    pub(super) when: &'a str,
}

/// Query one scalar index through its compatible operator and require the result to equal the
/// model's row set. With `check_plan` (right after creation, when the index covers every
/// fragment) the enabled scan plan must reference a scalar index on the column — otherwise the
/// indexed and unindexed runs could both take the un-indexed path and agree vacuously.
pub(super) async fn verify_index_query(
    dataset: &Dataset,
    oracle: &Oracle,
    columns: &[Column],
    indices: &[IndexEntry],
    entry: &IndexEntry,
    check: IndexQueryCheck<'_>,
) {
    let IndexKind::Scalar(kind) = &entry.kind else {
        return; // KNN oracle strengthening is out of scope; see the Knn read checks.
    };
    let ctx = format!(
        "{}: index query {} ({kind:?} on {})",
        check.when, entry.name, entry.col
    );
    match build_index_probe(kind, &entry.col, columns, oracle, check.salt) {
        IndexProbe::Filter {
            sql,
            expected,
            indexable,
        } => {
            for use_si in [true, false] {
                let ids = collect_ids(dataset, &sql, use_si, false, &ctx).await;
                assert_eq!(
                    ids, expected,
                    "{ctx}: `{sql}` (scalar_index={use_si}) disagrees with oracle"
                );
            }
            // An empty dataset can plan to an empty relation with no index node at all.
            if check.check_plan && indexable && !oracle.is_empty() {
                let plan = explain_filter_plan(dataset, &sql, &ctx).await;
                // Any index on the column satisfies the check: with several indices on one
                // column the planner is free to pick a sibling instead of `entry`.
                assert!(
                    indices
                        .iter()
                        .filter(|e| e.col == entry.col)
                        .any(|e| plan.contains(&format!("@{}(", e.name))),
                    "{ctx}: `{sql}` did not use a scalar index on {}:\n{plan}",
                    entry.col,
                );
            }
        }
        IndexProbe::Fts { term, expected } => {
            let query = FullTextSearchQuery::new(term.clone())
                .with_column(entry.col.clone())
                .unwrap_or_else(|e| panic!("{ctx}: fts column: {e}"));
            let batches = collect_scan(
                dataset,
                |scan| {
                    scan.project(&["id"])?;
                    scan.full_text_search(query)?;
                    Ok(())
                },
                &ctx,
            )
            .await;
            let mut ids = BTreeSet::new();
            for batch in &batches {
                let col = batch
                    .column_by_name("id")
                    .unwrap_or_else(|| panic!("{ctx}: id missing from fts output"))
                    .as_primitive::<UInt64Type>();
                for i in 0..col.len() {
                    assert!(
                        ids.insert(col.value(i)),
                        "{ctx}: duplicate id {} in fts result",
                        col.value(i)
                    );
                }
            }
            assert_eq!(
                ids, expected,
                "{ctx}: match query for {term:?} disagrees with oracle"
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn index_kinds_match_column_type() {
        assert_eq!(
            scalar_index_kinds(&DataType::Boolean),
            vec![BuiltinIndexType::Bitmap]
        );
        assert_eq!(
            scalar_index_kinds(&DataType::Utf8),
            vec![
                BuiltinIndexType::BTree,
                BuiltinIndexType::Bitmap,
                BuiltinIndexType::ZoneMap,
                BuiltinIndexType::BloomFilter,
                BuiltinIndexType::NGram,
                BuiltinIndexType::Inverted,
            ]
        );
    }

    #[test]
    fn ngram_needles_are_unicode_safe() {
        assert_eq!(ngram_needle("abcdef", 0), "abcdef");
        assert_eq!(ngram_needle("abcdef", 1), "bcde");
        assert_eq!(ngram_needle("åßç∂", 1), "ßç");
    }

    #[test]
    fn temporal_and_decimal_cells_probe_equality_via_arrow_cast() {
        use std::sync::Arc;

        use arrow::array::{ArrayRef, Decimal128Array, TimestampMicrosecondArray};

        use crate::dataset_ops::ops::{ColSpec, PrimSpec};

        let ts: ArrayRef = Arc::new(TimestampMicrosecondArray::from(vec![
            Some(5),
            None,
            Some(5),
        ]));
        let dec: ArrayRef = Arc::new(
            Decimal128Array::from(vec![Some(-12345_i128), Some(20_000_000_000), None])
                .with_precision_and_scale(38, 10)
                .expect("decimal scale"),
        );
        let columns = [
            Column {
                name: "v0".to_owned(),
                spec: ColSpec::Prim(PrimSpec::TsUs),
                dt: ts.data_type().clone(),
            },
            Column {
                name: "v1".to_owned(),
                spec: ColSpec::Prim(PrimSpec::Decimal128),
                dt: dec.data_type().clone(),
            },
        ];
        let oracle = Oracle::from([
            (1, vec![(ts.clone(), 0), (dec.clone(), 0)]),
            (2, vec![(ts.clone(), 1), (dec.clone(), 1)]),
            (3, vec![(ts, 2), (dec, 2)]),
        ]);

        let IndexProbe::Filter {
            sql,
            expected,
            indexable,
        } = build_index_probe(&BuiltinIndexType::BTree, "v0", &columns, &oracle, 0)
        else {
            panic!("expected a filter probe for the timestamp column");
        };
        assert_eq!(sql, "v0 = arrow_cast(5, 'Timestamp(Microsecond, None)')");
        assert_eq!(expected, BTreeSet::from([1, 3]));
        assert!(indexable);

        // Odd salts keep the IS NULL probe on arrow_cast types, so null-entry staleness
        // stays covered alongside the new equality probes.
        let IndexProbe::Filter { sql, expected, .. } =
            build_index_probe(&BuiltinIndexType::BTree, "v0", &columns, &oracle, 1)
        else {
            panic!("expected a filter probe for the odd salt");
        };
        assert_eq!(sql, "v0 IS NULL");
        assert_eq!(expected, BTreeSet::from([2]));

        let IndexProbe::Filter { sql, expected, .. } =
            build_index_probe(&BuiltinIndexType::ZoneMap, "v1", &columns, &oracle, 0)
        else {
            panic!("expected a filter probe for the decimal column");
        };
        assert_eq!(
            sql,
            "v1 = arrow_cast('-0.0000012345', 'Decimal128(38, 10)')"
        );
        assert_eq!(expected, BTreeSet::from([1]));
    }

    /// End-to-end: the new arrow_cast probes must be accepted by DataFusion, answered through
    /// the scalar index (check_plan runs right after creation), and agree with the oracle on a
    /// real dataset — for every type family `probe_lit` adds.
    #[tokio::test]
    async fn temporal_and_decimal_equality_probes_run_against_lance() {
        use crate::dataset_ops::ops::{
            ColSpec, DataParams, FuzzInput, LenClass, NullClass, Op, PrimSpec,
        };
        use crate::dataset_ops::runner::run;

        run(FuzzInput {
            seed: 0,
            schema: vec![
                ColSpec::Prim(PrimSpec::DurSec),
                ColSpec::Prim(PrimSpec::Decimal128),
                ColSpec::Prim(PrimSpec::TsMsUtc),
                ColSpec::Prim(PrimSpec::Date32),
                ColSpec::Prim(PrimSpec::Time64Ns),
            ],
            ops: vec![
                Op::Append {
                    keys: vec![1, 2, 3],
                    data: DataParams {
                        len: LenClass::Small,
                        nulls: NullClass::None,
                        nans: false,
                    },
                },
                Op::CreateIndex { col: 1, kind: 0 },
                Op::CreateIndex { col: 2, kind: 0 },
                Op::CreateIndex { col: 3, kind: 0 },
                Op::CreateIndex { col: 4, kind: 0 },
                Op::CreateIndex { col: 5, kind: 0 },
            ],
        })
        .await;
    }
}
