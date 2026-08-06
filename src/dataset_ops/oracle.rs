//! The row model and the checks that compare Lance against it.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use arrow::array::{Array, ArrayRef, AsArray as _, RecordBatch, UInt64Array, new_empty_array};
use arrow::compute::{cast, concat_batches, interleave, sort_to_indices, take};
use arrow::datatypes::{
    DataType, Date32Type, Date64Type, Decimal128Type, DurationMicrosecondType,
    DurationMillisecondType, DurationNanosecondType, DurationSecondType, Field, Int8Type,
    Int16Type, Int32Type, Int64Type, Schema, Time32MillisecondType, Time32SecondType,
    Time64MicrosecondType, Time64NanosecondType, TimeUnit, TimestampMicrosecondType,
    TimestampMillisecondType, TimestampNanosecondType, TimestampSecondType, UInt8Type, UInt16Type,
    UInt32Type, UInt64Type,
};
use lance::Dataset;

use super::ops::{ColSpec, FslChild};
use super::scan::collect_scan;

/// One modeled value, kept as a position in the array that was written rather than a copy of it.
pub(super) type Cell = (ArrayRef, usize);
/// One row's value cells, in dataset column order (excluding `id`).
pub(super) type Row = Vec<Cell>;
/// id → row. The model the dataset must agree with at every step.
pub(super) type Oracle = BTreeMap<u64, Row>;

/// One value column: its dataset name, the spec that generates it, and the Arrow type both agree on.
#[derive(Clone)]
pub(super) struct Column {
    pub(super) name: String,
    pub(super) spec: ColSpec,
    pub(super) dt: DataType,
}

pub(super) fn is_float32_vector(column: &Column) -> bool {
    matches!(
        &column.spec,
        ColSpec::Fsl {
            child: FslChild::Float32,
            ..
        }
    )
}

pub(super) fn is_float_vector(column: &Column) -> bool {
    matches!(&column.spec, ColSpec::Fsl { child, .. } if child.is_float())
}

pub(super) fn all_names(columns: &[Column]) -> Vec<String> {
    std::iter::once("id".to_owned())
        .chain(columns.iter().map(|column| column.name.clone()))
        .collect()
}

/// Concatenate scan batches. An empty scan yields an empty batch with the projected schema, so
/// callers can compare shapes without special-casing it.
pub(super) fn concat_or_empty(
    batches: &[RecordBatch],
    names: &[String],
    columns: &[Column],
) -> RecordBatch {
    if let Some(first) = batches.first() {
        concat_batches(&first.schema(), batches).expect("concat scan batches")
    } else {
        let fields: Vec<Field> = names
            .iter()
            .map(|name| {
                if name == "id" {
                    Field::new("id", DataType::UInt64, false)
                } else {
                    let column = columns
                        .iter()
                        .find(|column| &column.name == name)
                        .expect("known column");
                    Field::new(&column.name, column.dt.clone(), true)
                }
            })
            .collect();
        RecordBatch::new_empty(Arc::new(Schema::new(fields)))
    }
}

pub(super) fn is_cell_null(cell: &Cell) -> bool {
    let (arr, idx) = cell;
    // NullArray has no validity bitmap, so `is_null` alone would report false.
    arr.data_type() == &DataType::Null || arr.is_null(*idx)
}

/// SQL-literal-friendly value of a cell, for equality predicates and update literals.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) enum Lit {
    I(i128),
    U(u128),
    B(bool),
    S(String),
    /// Exact value expressed as `arrow_cast(raw, 'cast_to')` — temporal and decimal types whose
    /// bare literals would make DataFusion coerce the column instead (hiding it from the
    /// scalar-index query parser) or exceed the parser-safe integer range.
    Cast {
        raw: String,
        cast_to: String,
    },
}

impl Lit {
    pub(super) fn to_sql(&self) -> String {
        match self {
            Self::I(v) => v.to_string(),
            Self::U(v) => v.to_string(),
            Self::B(v) => v.to_string(),
            Self::S(v) => format!("'{}'", v.replace('\'', "''")),
            Self::Cast { raw, cast_to } => format!("arrow_cast({raw}, '{cast_to}')"),
        }
    }

    /// Whether the SQL literal round-trips exactly. The SQL parser turns integer literals
    /// outside the i64 range into Float64, which then cannot be converted back to the column
    /// type (so e.g. equality filters on u64 values above `i64::MAX` are unexpressible).
    pub(super) fn sql_safe(&self) -> bool {
        match self {
            Self::I(v) => *v > i128::from(i64::MIN) && *v <= i128::from(i64::MAX),
            Self::U(v) => *v <= u128::try_from(i64::MAX).expect("positive"),
            Self::B(_) | Self::S(_) | Self::Cast { .. } => true,
        }
    }
}

pub(super) fn lit_friendly(dt: &DataType) -> bool {
    matches!(
        dt,
        DataType::Int8
            | DataType::Int16
            | DataType::Int32
            | DataType::Int64
            | DataType::UInt8
            | DataType::UInt16
            | DataType::UInt32
            | DataType::UInt64
            | DataType::Boolean
            | DataType::Utf8
            | DataType::LargeUtf8
    )
}

pub(super) fn cell_lit(dt: &DataType, cell: &Cell) -> Option<Lit> {
    if is_cell_null(cell) {
        return None;
    }
    let (arr, i) = cell;
    let i = *i;
    Some(match dt {
        DataType::Int8 => Lit::I(arr.as_primitive::<Int8Type>().value(i).into()),
        DataType::Int16 => Lit::I(arr.as_primitive::<Int16Type>().value(i).into()),
        DataType::Int32 => Lit::I(arr.as_primitive::<Int32Type>().value(i).into()),
        DataType::Int64 => Lit::I(arr.as_primitive::<Int64Type>().value(i).into()),
        DataType::UInt8 => Lit::U(arr.as_primitive::<UInt8Type>().value(i).into()),
        DataType::UInt16 => Lit::U(arr.as_primitive::<UInt16Type>().value(i).into()),
        DataType::UInt32 => Lit::U(arr.as_primitive::<UInt32Type>().value(i).into()),
        DataType::UInt64 => Lit::U(arr.as_primitive::<UInt64Type>().value(i).into()),
        DataType::Boolean => Lit::B(arr.as_boolean().value(i)),
        DataType::Utf8 => Lit::S(arr.as_string::<i32>().value(i).to_owned()),
        DataType::LargeUtf8 => Lit::S(arr.as_string::<i64>().value(i).to_owned()),
        _ => return None,
    })
}

fn time_unit_name(unit: &TimeUnit) -> &'static str {
    match unit {
        TimeUnit::Second => "Second",
        TimeUnit::Millisecond => "Millisecond",
        TimeUnit::Microsecond => "Microsecond",
        TimeUnit::Nanosecond => "Nanosecond",
    }
}

/// Exact SQL string literal for a raw scaled decimal, e.g. -12345 at scale 10 → '-0.0000012345'.
fn decimal_sql_string(raw: i128, scale: i8) -> Option<String> {
    let scale = u32::try_from(scale).ok()?;
    let mag = raw.unsigned_abs();
    let sign = if raw < 0 { "-" } else { "" };
    if scale == 0 {
        return Some(format!("'{sign}{mag}'"));
    }
    let base = 10_u128.checked_pow(scale)?;
    Some(format!(
        "'{sign}{}.{:0width$}'",
        mag / base,
        mag % base,
        width = scale as usize
    ))
}

/// Literal for index probes. Extends [`cell_lit`] with temporal and decimal values through
/// [`Lit::Cast`]; the wider set stays out of `lit_friendly` so update and read predicates keep
/// their behavior on existing inputs. Float and Decimal256 cells still yield `None`: float
/// literals do not reliably round-trip through SQL, and Decimal256 has no indexable kind.
pub(super) fn probe_lit(dt: &DataType, cell: &Cell) -> Option<Lit> {
    if let Some(lit) = cell_lit(dt, cell) {
        return Some(lit);
    }
    if is_cell_null(cell) {
        return None;
    }
    let (arr, i) = cell;
    let i = *i;
    let (raw, cast_to) = match dt {
        DataType::Date32 => (
            arr.as_primitive::<Date32Type>().value(i).to_string(),
            "Date32".to_owned(),
        ),
        DataType::Date64 => (
            arr.as_primitive::<Date64Type>().value(i).to_string(),
            "Date64".to_owned(),
        ),
        DataType::Time32(unit) => {
            let v = match unit {
                TimeUnit::Second => arr.as_primitive::<Time32SecondType>().value(i),
                TimeUnit::Millisecond => arr.as_primitive::<Time32MillisecondType>().value(i),
                _ => return None, // invalid Time32 unit
            };
            (v.to_string(), format!("Time32({})", time_unit_name(unit)))
        }
        DataType::Time64(unit) => {
            let v = match unit {
                TimeUnit::Microsecond => arr.as_primitive::<Time64MicrosecondType>().value(i),
                TimeUnit::Nanosecond => arr.as_primitive::<Time64NanosecondType>().value(i),
                _ => return None, // invalid Time64 unit
            };
            (v.to_string(), format!("Time64({})", time_unit_name(unit)))
        }
        DataType::Timestamp(unit, tz) => {
            let v = match unit {
                TimeUnit::Second => arr.as_primitive::<TimestampSecondType>().value(i),
                TimeUnit::Millisecond => arr.as_primitive::<TimestampMillisecondType>().value(i),
                TimeUnit::Microsecond => arr.as_primitive::<TimestampMicrosecondType>().value(i),
                TimeUnit::Nanosecond => arr.as_primitive::<TimestampNanosecondType>().value(i),
            };
            let tz = match tz {
                None => "None".to_owned(),
                Some(tz) => format!("Some(\"{tz}\")"),
            };
            (
                v.to_string(),
                format!("Timestamp({}, {tz})", time_unit_name(unit)),
            )
        }
        DataType::Duration(unit) => {
            let v = match unit {
                TimeUnit::Second => arr.as_primitive::<DurationSecondType>().value(i),
                TimeUnit::Millisecond => arr.as_primitive::<DurationMillisecondType>().value(i),
                TimeUnit::Microsecond => arr.as_primitive::<DurationMicrosecondType>().value(i),
                TimeUnit::Nanosecond => arr.as_primitive::<DurationNanosecondType>().value(i),
            };
            (v.to_string(), format!("Duration({})", time_unit_name(unit)))
        }
        DataType::Decimal128(precision, scale) => (
            decimal_sql_string(arr.as_primitive::<Decimal128Type>().value(i), *scale)?,
            format!("Decimal128({precision}, {scale})"),
        ),
        _ => return None,
    };
    Some(Lit::Cast { raw, cast_to })
}

/// Non-null string value of a Utf8/LargeUtf8 cell.
pub(super) fn cell_str(dt: &DataType, cell: &Cell) -> Option<String> {
    match cell_lit(dt, cell) {
        Some(Lit::S(s)) => Some(s),
        _ => None,
    }
}

/// Elements of a list cell as cells into the child array; `None` when the list itself is null.
pub(super) fn list_cells(cell: &Cell) -> Option<Vec<Cell>> {
    if is_cell_null(cell) {
        return None;
    }
    let (arr, idx) = cell;
    let values: ArrayRef = match arr.data_type() {
        DataType::List(_) => arr.as_list::<i32>().value(*idx),
        DataType::LargeList(_) => arr.as_list::<i64>().value(*idx),
        _ => return None,
    };
    Some((0..values.len()).map(|i| (values.clone(), i)).collect())
}

/// The modeled values of column `col`, in oracle key order, as one array.
pub(super) fn expected_column<'a>(
    rows: impl Iterator<Item = &'a Row>,
    col: usize,
    dt: &DataType,
) -> ArrayRef {
    let mut arrays: Vec<ArrayRef> = Vec::new();
    let mut pairs: Vec<(usize, usize)> = Vec::new();
    for row in rows {
        let (arr, idx) = &row[col];
        let ai = arrays
            .iter()
            .position(|a| Arc::ptr_eq(a, arr))
            .unwrap_or_else(|| {
                arrays.push(arr.clone());
                arrays.len() - 1
            });
        pairs.push((ai, *idx));
    }
    if arrays.is_empty() {
        return new_empty_array(dt);
    }
    let refs: Vec<&dyn Array> = arrays.iter().map(|a| a.as_ref()).collect();
    interleave(&refs, &pairs).expect("interleave expected column")
}

/// Dictionary encodings can differ between what we wrote and what a scan returns; compare the
/// logical values instead.
fn canonical(arr: &ArrayRef) -> ArrayRef {
    if let DataType::Dictionary(_, value_type) = arr.data_type() {
        cast(arr, value_type).expect("cast dictionary to values")
    } else {
        arr.clone()
    }
}

pub(super) fn assert_arrays_eq(expected: &ArrayRef, actual: &ArrayRef, ctx: &str, col: &str) {
    let expected = canonical(expected);
    let actual = canonical(actual);
    assert!(
        expected.to_data() == actual.to_data(),
        "{ctx}: column {col} disagrees with oracle\nexpected: {expected:?}\nactual:   {actual:?}"
    );
}

/// Full scan must agree with the oracle exactly (row count, key set, and every cell).
pub(super) async fn verify_full_scan(
    dataset: &Dataset,
    oracle: &Oracle,
    columns: &[Column],
    when: &str,
) {
    let ctx = format!("{when}: full scan");

    // The dataset schema must be exactly id + our columns, in order.
    let ds_names: Vec<&str> = dataset
        .schema()
        .fields
        .iter()
        .map(|f| f.name.as_str())
        .collect();
    let names = all_names(columns);
    assert_eq!(
        ds_names,
        names.iter().map(String::as_str).collect::<Vec<_>>(),
        "{ctx}: schema"
    );

    let batches = collect_scan(dataset, |scan| scan.project(&names).map(|_| ()), &ctx).await;
    let batch = concat_or_empty(&batches, &names, columns);
    assert_eq!(batch.num_rows(), oracle.len(), "{ctx}: row count");
    if batch.num_rows() == 0 {
        return;
    }

    let ids = batch.column(0).as_primitive::<UInt64Type>();
    let mut seen = BTreeSet::new();
    for i in 0..ids.len() {
        assert!(
            seen.insert(ids.value(i)),
            "{ctx}: duplicate id {} in scan output",
            ids.value(i)
        );
    }

    let sort_idx = sort_to_indices(batch.column(0), None, None).expect("sort ids");
    let sorted_ids = take(batch.column(0).as_ref(), &sort_idx, None).expect("take ids");
    let expected_ids: ArrayRef = Arc::new(UInt64Array::from_iter_values(oracle.keys().copied()));
    assert_arrays_eq(&expected_ids, &sorted_ids, &ctx, "id");

    for (j, c) in columns.iter().enumerate() {
        let actual = take(batch.column(j + 1).as_ref(), &sort_idx, None).expect("take column");
        let expected = expected_column(oracle.values(), j, &c.dt);
        assert_arrays_eq(&expected, &actual, &ctx, &c.name);
    }
}

/// Filtered point probes, each run with the scalar index enabled AND disabled: with an index
/// present these go through the scalar-index + prefilter path, which is exactly where
/// lance#7803-class staleness shows up, and the two modes must also agree with each other.
///
/// The absent probes are drawn from `ever_inserted \ oracle` — keys that were once in the
/// dataset and are now deleted. Those are the only keys that can carry a *stale* index entry:
/// a key that never existed has no entry to go stale. A couple of never-seen keys keep the
/// "no index entry at all" path exercised too.
pub(super) async fn verify_probes(
    dataset: &Dataset,
    oracle: &Oracle,
    columns: &[Column],
    ever_inserted: &BTreeSet<u64>,
    when: &str,
) {
    let present: Vec<u64> = oracle.keys().copied().take(2).collect();
    let deleted: Vec<u64> = ever_inserted
        .iter()
        .filter(|k| !oracle.contains_key(k))
        .rev()
        .copied()
        .take(3)
        .collect();
    // Never-seen keys, outside the u8 key domain so they can't collide with real data.
    let never_seen = [1_000_u64, 4_242];

    let names = all_names(columns);
    for key in present
        .iter()
        .chain(deleted.iter())
        .chain(never_seen.iter())
    {
        for use_si in [true, false] {
            let ctx = format!("{when}: probe id={key} scalar_index={use_si}");
            let batches = collect_scan(
                dataset,
                |scan| {
                    scan.project(&names)?;
                    scan.filter(&format!("id = {key}"))?;
                    scan.use_scalar_index(use_si);
                    Ok(())
                },
                &ctx,
            )
            .await;
            let batch = concat_or_empty(&batches, &names, columns);
            match oracle.get(key) {
                Some(row) => {
                    assert_eq!(batch.num_rows(), 1, "{ctx}: expected exactly one row");
                    for (j, c) in columns.iter().enumerate() {
                        let expected = expected_column(std::iter::once(row), j, &c.dt);
                        assert_arrays_eq(&expected, batch.column(j + 1), &ctx, &c.name);
                    }
                }
                None => {
                    assert_eq!(batch.num_rows(), 0, "{ctx}: expected no rows");
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use arrow::array::Int64Array;

    use super::*;

    #[test]
    fn expected_column_preserves_model_order_across_source_arrays() {
        let first: ArrayRef = Arc::new(Int64Array::from(vec![10, 11]));
        let second: ArrayRef = Arc::new(Int64Array::from(vec![20, 21]));
        let rows = [vec![(first, 1)], vec![(second, 0)]];

        let actual = expected_column(rows.iter(), 0, &DataType::Int64);
        let expected: ArrayRef = Arc::new(Int64Array::from(vec![11, 20]));

        assert_eq!(actual.to_data(), expected.to_data());
    }

    #[test]
    fn string_literals_escape_quotes() {
        assert_eq!(Lit::S("it's".to_owned()).to_sql(), "'it''s'");
        assert!(!Lit::U(i64::MAX as u128 + 1).sql_safe());
    }
}
