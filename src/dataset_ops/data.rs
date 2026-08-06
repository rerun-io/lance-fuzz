//! Deterministic Arrow value generation, and the write primitives the operations share.

use std::sync::Arc;

use arrow::array::{
    Array, ArrayRef, RecordBatch, RecordBatchIterator, StructArray, UInt64Array, new_empty_array,
    new_null_array,
};
use arrow::compute::cast;
use arrow::datatypes::{DataType, Field, Fields, Schema, TimeUnit};
use arrow::error::ArrowError;
use lance::Dataset;
use lance::dataset::write::merge_insert::SourceDedupeBehavior;
use lance::dataset::{
    MergeInsertBuilder, MergeStats, WhenMatched, WhenNotMatched, WhenNotMatchedBySource,
    WriteParams,
};
use lance_datagen::{
    ArrayGenerator, ArrayGeneratorExt as _, ByteCount, Dimension, RowCount, array,
};
use rand_xoshiro::Xoshiro256PlusPlus;

use super::ops::{ColSpec, DataParams, LenClass, PrimSpec};
use super::oracle::{Cell, Column, Row};

/// Per-input write layout derived from the shared seed.
#[derive(Clone, Copy, Debug)]
pub(super) struct WriteProfile {
    pub(super) max_rows_per_group: usize,
    pub(super) max_rows_per_file: usize,
    pub(super) enable_stable_row_ids: bool,
}

impl WriteProfile {
    pub(super) fn from_seed(seed: u64) -> Self {
        const ROWS_PER_GROUP: [usize; 3] = [1, 4, 16];
        const ROWS_PER_FILE: [usize; 3] = [2, 4, 8];

        Self {
            max_rows_per_group: ROWS_PER_GROUP[seed as usize % ROWS_PER_GROUP.len()],
            max_rows_per_file: ROWS_PER_FILE[(seed >> 8) as usize % ROWS_PER_FILE.len()],
            enable_stable_row_ids: seed & (1 << 16) != 0,
        }
    }

    pub(super) fn write_params(self) -> WriteParams {
        WriteParams {
            max_rows_per_group: self.max_rows_per_group,
            max_rows_per_file: self.max_rows_per_file,
            enable_stable_row_ids: self.enable_stable_row_ids,
            ..Default::default()
        }
    }
}

#[derive(Debug)]
struct CastGen {
    inner: Box<dyn ArrayGenerator>,
    dt: DataType,
}

impl ArrayGenerator for CastGen {
    fn generate(
        &mut self,
        length: RowCount,
        rng: &mut Xoshiro256PlusPlus,
    ) -> Result<ArrayRef, ArrowError> {
        let arr = self.inner.generate(length, rng)?;
        cast(&arr, &self.dt)
    }

    fn data_type(&self) -> &DataType {
        &self.dt
    }

    fn element_size_bytes(&self) -> Option<ByteCount> {
        self.inner.element_size_bytes()
    }
}

/// Struct generator over caller-provided child generators. datagen's `rand_struct` builds
/// children through `rand_type`, which breaks on tz'd timestamps and samples dates relative to
/// wall-clock `now` — this keeps struct children on our deterministic generators.
#[derive(Debug)]
struct StructGen {
    fields: Fields,
    children: Vec<Box<dyn ArrayGenerator>>,
    dt: DataType,
}

impl ArrayGenerator for StructGen {
    fn generate(
        &mut self,
        length: RowCount,
        rng: &mut Xoshiro256PlusPlus,
    ) -> Result<ArrayRef, ArrowError> {
        let arrays = self
            .children
            .iter_mut()
            .map(|g| g.generate(length, rng))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Arc::new(StructArray::try_new(
            self.fields.clone(),
            arrays,
            None,
        )?))
    }

    fn data_type(&self) -> &DataType {
        &self.dt
    }

    fn element_size_bytes(&self) -> Option<ByteCount> {
        None
    }
}

/// Fixed range for temporal generators: datagen's default samples relative to `Utc::now()`,
/// which would make the generated values differ from run to run.
fn epoch_range() -> (chrono::DateTime<chrono::Utc>, chrono::DateTime<chrono::Utc>) {
    (
        chrono::DateTime::from_timestamp(0, 0).expect("epoch"),
        chrono::DateTime::from_timestamp(1_900_000_000, 0).expect("year 2030"),
    )
}

fn maybe_nulls(g: Box<dyn ArrayGenerator>, null_prob: f64) -> Box<dyn ArrayGenerator> {
    if null_prob > 0.0 {
        g.with_random_nulls(null_prob)
    } else {
        g
    }
}

/// Build the datagen generator for one column. The data type it produces must be stable across
/// `LenClass`es and null probabilities — it defines the dataset schema.
pub(super) fn make_generator(
    spec: &ColSpec,
    len: LenClass,
    null_prob: f64,
) -> Box<dyn ArrayGenerator> {
    match spec {
        ColSpec::Prim(p) => maybe_nulls(prim_generator(*p, len), null_prob),
        ColSpec::Dict { utf8_values } => {
            let inner = if *utf8_values {
                prim_generator(PrimSpec::Utf8, len)
            } else {
                prim_generator(PrimSpec::Int64, len)
            };
            maybe_nulls(array::dict_type(inner, &DataType::Int32), null_prob)
        }
        ColSpec::List { child, large } => maybe_nulls(
            array::rand_list_any(prim_generator(*child, len), *large),
            null_prob,
        ),
        ColSpec::Fsl { child, dim } => maybe_nulls(
            array::cycle_vec(
                array::rand_type(&child.data_type()),
                Dimension::from(ColSpec::fsl_dim(*dim) as u32),
            ),
            null_prob,
        ),
        ColSpec::Struct { a, b } => {
            // Lance cannot encode struct-level validity (lance-encoding physical.rs: "Currently
            // there is no way to encode struct nullability"): a null struct reads back as a
            // *valid* struct with null children. Generate nulls in the children instead so the
            // oracle and the dataset agree.
            let fields = Fields::from(vec![
                Field::new("a", a.data_type(), true),
                Field::new("b", b.data_type(), true),
            ]);
            Box::new(StructGen {
                dt: DataType::Struct(fields.clone()),
                children: vec![
                    maybe_nulls(prim_generator(*a, len), null_prob),
                    maybe_nulls(prim_generator(*b, len), null_prob),
                ],
                fields,
            })
        }
        ColSpec::Null => array::null_type(),
    }
}

fn timestamp_generator(unit: TimeUnit, tz: Option<&str>) -> Box<dyn ArrayGenerator> {
    let (start, end) = epoch_range();
    let inner = array::rand_timestamp_in_range(start, end, &DataType::Timestamp(unit, None));
    match tz {
        None => inner,
        Some(tz) => Box::new(CastGen {
            inner,
            dt: DataType::Timestamp(unit, Some(tz.into())),
        }),
    }
}

fn prim_generator(p: PrimSpec, len: LenClass) -> Box<dyn ArrayGenerator> {
    match p {
        PrimSpec::Utf8 => array::rand_utf8(ByteCount::from(len.bytes()), false),
        PrimSpec::LargeUtf8 => array::rand_utf8(ByteCount::from(len.bytes()), true),
        // Variable length *within* one array, up to the class cap.
        PrimSpec::Binary => {
            array::rand_varbin(ByteCount::from(0), ByteCount::from(len.bytes().max(1)))
        }
        PrimSpec::LargeBinary => array::rand_fixedbin(ByteCount::from(len.bytes()), true),
        PrimSpec::Date32 => {
            let (start, end) = epoch_range();
            array::rand_date32_in_range(start, end)
        }
        PrimSpec::Date64 => {
            let (start, end) = epoch_range();
            array::rand_date64_in_range(start, end)
        }
        PrimSpec::TsSec => timestamp_generator(TimeUnit::Second, None),
        PrimSpec::TsMsUtc => timestamp_generator(TimeUnit::Millisecond, Some("UTC")),
        PrimSpec::TsUs => timestamp_generator(TimeUnit::Microsecond, None),
        PrimSpec::TsNsUtc => timestamp_generator(TimeUnit::Nanosecond, Some("UTC")),
        // datagen's RandomBytesGenerator reinterprets a Vec<u8> as the scalar type, and the
        // allocator doesn't guarantee the 16-byte alignment i128/i256 need — cast from Int64.
        PrimSpec::Decimal128 | PrimSpec::Decimal256 => Box::new(CastGen {
            inner: array::rand_type(&DataType::Int64),
            dt: p.data_type(),
        }),
        _ => array::rand_type(&p.data_type()),
    }
}

/// Source batch for one write, plus the value arrays the model records for it.
pub(super) struct WriteData {
    pub(super) batch: RecordBatch,
    /// Value arrays, parallel to the `include` column indices (in dataset-schema order,
    /// regardless of `rotate`).
    pub(super) arrays: Vec<ArrayRef>,
}

/// Build the source batch for one write: `id` plus the `include` columns, `rotate` positions
/// rotated so the source column order differs from the dataset's.
pub(super) fn build_write(
    columns: &[Column],
    include: &[usize],
    keys: &[u64],
    dup: Option<usize>,
    data: DataParams,
    rotate: u8,
    rng: &mut Xoshiro256PlusPlus,
) -> WriteData {
    let mut ids = keys.to_vec();
    if let Some(i) = dup {
        ids.push(keys[i]);
    }
    let n = ids.len();

    let mut cols: Vec<(Field, ArrayRef)> = Vec::with_capacity(include.len() + 1);
    cols.push((
        Field::new("id", DataType::UInt64, false),
        Arc::new(UInt64Array::from(ids)) as ArrayRef,
    ));
    let mut arrays = Vec::with_capacity(include.len());
    for &j in include {
        let c = &columns[j];
        // lance-datagen panics generating 0-row arrays of wide types (the empty buffer's
        // dangling pointer fails ScalarBuffer's alignment check, e.g. Decimal256/i256).
        let arr = if n == 0 {
            new_empty_array(&c.dt)
        } else {
            let null_prob = if matches!(c.spec, ColSpec::Null) {
                0.0
            } else {
                data.nulls.probability()
            };
            let mut g = make_generator(&c.spec, data.len, null_prob);
            if data.nans && matches!(c.dt, DataType::Float32 | DataType::Float64) {
                g = g.with_nans(&[false, false, true]);
            }
            g.generate(RowCount::from(n as u64), rng)
                .expect("datagen generate")
        };
        assert_eq!(
            arr.data_type(),
            &c.dt,
            "datagen type drift for column {}",
            c.name
        );
        arrays.push(arr.clone());
        cols.push((Field::new(&c.name, c.dt.clone(), true), arr));
    }

    let r = (rotate as usize) % cols.len();
    cols.rotate_left(r);
    let (fields, arrs): (Vec<Field>, Vec<ArrayRef>) = cols.into_iter().unzip();
    let batch =
        RecordBatch::try_new(Arc::new(Schema::new(fields)), arrs).expect("build source batch");
    WriteData { batch, arrays }
}

pub(super) fn full_row(arrays: &[ArrayRef], i: usize) -> Row {
    arrays.iter().map(|a| (a.clone(), i)).collect()
}

/// One all-null cell of the given type, as Lance will reproduce it. Struct-level validity can't
/// be encoded, so an all-null struct column reads back as valid structs with null children.
pub(super) fn all_null_cell(dt: &DataType) -> Cell {
    if let DataType::Struct(fields) = dt {
        let children = fields
            .iter()
            .map(|f| new_null_array(f.data_type(), 1))
            .collect();
        (
            Arc::new(StructArray::new(fields.clone(), children, None)) as ArrayRef,
            0,
        )
    } else {
        (new_null_array(dt, 1), 0)
    }
}

pub(super) fn source_reader(batch: RecordBatch) -> Box<dyn arrow::array::RecordBatchReader + Send> {
    let schema = batch.schema();
    Box::new(RecordBatchIterator::new([Ok(batch)], schema))
}

/// Run one merge-insert and advance `dataset` to the committed version.
pub(super) async fn merge(
    dataset: &mut Dataset,
    matched: WhenMatched,
    not_matched: WhenNotMatched,
    by_source: WhenNotMatchedBySource,
    first_seen: bool,
    batch: RecordBatch,
) -> Result<MergeStats, lance::Error> {
    let mut builder = MergeInsertBuilder::try_new(Arc::new(dataset.clone()), vec!["id".into()])?;
    builder
        .when_matched(matched)
        .when_not_matched(not_matched)
        .when_not_matched_by_source(by_source)
        // Auto-cleanup is time-based and would make old versions vanish nondeterministically,
        // breaking both `Restore` and crash reproducibility.
        .skip_auto_cleanup(true);
    if first_seen {
        builder.source_dedupe_behavior(SourceDedupeBehavior::FirstSeen);
    }
    let job = builder.try_build()?;
    let (new_dataset, stats) = job.execute_reader(source_reader(batch)).await?;
    *dataset = (*new_dataset).clone();
    Ok(stats)
}

#[cfg(test)]
mod tests {
    use rand_xoshiro::rand_core::SeedableRng as _;

    use super::*;

    #[test]
    fn fixed_seed_generates_identical_arrays() {
        let spec = ColSpec::Prim(PrimSpec::Int64);
        let mut first_generator = make_generator(&spec, LenClass::Small, 0.2);
        let mut second_generator = make_generator(&spec, LenClass::Small, 0.2);
        let mut first_rng = Xoshiro256PlusPlus::seed_from_u64(42);
        let mut second_rng = Xoshiro256PlusPlus::seed_from_u64(42);

        let first = first_generator
            .generate(RowCount::from(16), &mut first_rng)
            .expect("generate first array");
        let second = second_generator
            .generate(RowCount::from(16), &mut second_rng)
            .expect("generate second array");

        assert_eq!(first.to_data(), second.to_data());
    }
}
