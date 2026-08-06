//! The fuzz input: column specifications, operations, and their `arbitrary` weighting.

use arbitrary::{Arbitrary, Unstructured};
use arrow::datatypes::{DataType, TimeUnit};

/// One session, decoded from the input bytes.
#[derive(Arbitrary, Debug)]
pub struct FuzzInput {
    /// Seeds the RNG that `lance-datagen` draws values from; the whole run is a pure function
    /// of the input bytes.
    pub(super) seed: u64,
    pub(super) schema: Vec<ColSpec>,
    pub(super) ops: Vec<Op>,
}

/// Non-nested column types. `data_type()` must be stable across ops (it defines the schema).
#[derive(Arbitrary, Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum PrimSpec {
    Bool,
    Int8,
    Int16,
    Int32,
    Int64,
    UInt8,
    UInt16,
    UInt32,
    UInt64,
    Float16,
    Float32,
    Float64,
    Decimal128,
    Decimal256,
    Utf8,
    LargeUtf8,
    Binary,
    LargeBinary,
    FixedSizeBinary(u8),
    Date32,
    Date64,
    Time32Sec,
    Time32Ms,
    Time64Us,
    Time64Ns,
    TsSec,
    TsMsUtc,
    TsUs,
    TsNsUtc,
    DurSec,
    DurMs,
    DurUs,
    DurNs,
}

impl PrimSpec {
    pub(super) fn data_type(self) -> DataType {
        match self {
            Self::Bool => DataType::Boolean,
            Self::Int8 => DataType::Int8,
            Self::Int16 => DataType::Int16,
            Self::Int32 => DataType::Int32,
            Self::Int64 => DataType::Int64,
            Self::UInt8 => DataType::UInt8,
            Self::UInt16 => DataType::UInt16,
            Self::UInt32 => DataType::UInt32,
            Self::UInt64 => DataType::UInt64,
            Self::Float16 => DataType::Float16,
            Self::Float32 => DataType::Float32,
            Self::Float64 => DataType::Float64,
            // Precision 38/76 so any i64 (what the generator draws) fits after scaling.
            Self::Decimal128 => DataType::Decimal128(38, 10),
            Self::Decimal256 => DataType::Decimal256(76, 10),
            Self::Utf8 => DataType::Utf8,
            Self::LargeUtf8 => DataType::LargeUtf8,
            Self::Binary => DataType::Binary,
            Self::LargeBinary => DataType::LargeBinary,
            Self::FixedSizeBinary(n) => DataType::FixedSizeBinary(1 + i32::from(n % 16)),
            Self::Date32 => DataType::Date32,
            Self::Date64 => DataType::Date64,
            Self::Time32Sec => DataType::Time32(TimeUnit::Second),
            Self::Time32Ms => DataType::Time32(TimeUnit::Millisecond),
            Self::Time64Us => DataType::Time64(TimeUnit::Microsecond),
            Self::Time64Ns => DataType::Time64(TimeUnit::Nanosecond),
            Self::TsSec => DataType::Timestamp(TimeUnit::Second, None),
            Self::TsMsUtc => DataType::Timestamp(TimeUnit::Millisecond, Some("UTC".into())),
            Self::TsUs => DataType::Timestamp(TimeUnit::Microsecond, None),
            Self::TsNsUtc => DataType::Timestamp(TimeUnit::Nanosecond, Some("UTC".into())),
            Self::DurSec => DataType::Duration(TimeUnit::Second),
            Self::DurMs => DataType::Duration(TimeUnit::Millisecond),
            Self::DurUs => DataType::Duration(TimeUnit::Microsecond),
            Self::DurNs => DataType::Duration(TimeUnit::Nanosecond),
        }
    }
}

/// Element type of a fixed-size-list ("vector") column.
#[derive(Arbitrary, Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum FslChild {
    Float16,
    Float32,
    Float64,
    Int32,
    UInt8,
}

impl FslChild {
    pub(super) fn data_type(self) -> DataType {
        match self {
            Self::Float16 => DataType::Float16,
            Self::Float32 => DataType::Float32,
            Self::Float64 => DataType::Float64,
            Self::Int32 => DataType::Int32,
            Self::UInt8 => DataType::UInt8,
        }
    }

    pub(super) fn is_float(self) -> bool {
        matches!(self, Self::Float16 | Self::Float32 | Self::Float64)
    }
}

#[derive(Arbitrary, Debug, Clone)]
pub(super) enum ColSpec {
    Prim(PrimSpec),
    /// Dictionary<Int32, Utf8|Int64>.
    Dict {
        utf8_values: bool,
    },
    List {
        child: PrimSpec,
        large: bool,
    },
    /// FixedSizeList<child, 1..=8> — float children are the vector-index / knn candidates.
    Fsl {
        child: FslChild,
        dim: u8,
    },
    Struct {
        a: PrimSpec,
        b: PrimSpec,
    },
    // No `Map`: it needs Lance file format 2.2+, and the schema has to be writable by every
    // initial file version this harness selects (2.0 through 2.3) — on 2.0 and 2.1 a `Map`
    // column fails the write with "Not supported".
    /// The `Null` data type (every value is null).
    Null,
}

impl ColSpec {
    pub(super) fn fsl_dim(dim: u8) -> i32 {
        1 + i32::from(dim % 8)
    }
}

/// Per-write-op payload size for variable-length types.
#[derive(Arbitrary, Debug, Clone, Copy)]
pub(super) enum LenClass {
    Empty,
    One,
    Small,
    Big,
}

impl LenClass {
    pub(super) fn bytes(self) -> u64 {
        match self {
            Self::Empty => 0,
            Self::One => 1,
            Self::Small => 12,
            Self::Big => 1024,
        }
    }
}

/// Per-write-op null density for value columns.
#[derive(Arbitrary, Debug, Clone, Copy)]
pub(super) enum NullClass {
    None,
    Some,
    Heavy,
}

impl NullClass {
    pub(super) fn probability(self) -> f64 {
        match self {
            Self::None => 0.0,
            Self::Some => 0.2,
            Self::Heavy => 0.9,
        }
    }
}

#[derive(Arbitrary, Debug, Clone, Copy)]
pub(super) struct DataParams {
    pub(super) len: LenClass,
    pub(super) nulls: NullClass,
    pub(super) nans: bool,
}

#[derive(Arbitrary, Debug, Clone)]
pub(super) enum AddFill {
    /// `add_columns(AllNulls)` with an arbitrary new column type.
    Nulls(ColSpec),
    /// `add_columns(SqlExpressions)` — value derived from `id`, so the oracle can predict it.
    IdDerived,
}

#[derive(Arbitrary, Debug, Clone, Copy)]
pub(super) enum PredSpec {
    IdRange {
        lo: u8,
        hi: u8,
    },
    IsNull {
        col: u8,
    },
    NotNull {
        col: u8,
    },
    /// Equality on a SQL-literal-friendly value column, using a value known to the oracle.
    Eq {
        col: u8,
        key: u8,
    },
}

#[derive(Arbitrary, Debug, Clone)]
pub(super) enum ReadOp {
    /// Same filter with scalar index on and off; both must agree with the oracle.
    Filtered {
        pred: PredSpec,
        prefilter: bool,
    },
    Count {
        pred: PredSpec,
    },
    /// Projected scan with limit/offset/batch-size vs. the unlimited scan.
    Limit {
        sel: u8,
        mask: u8,
    },
    /// `Dataset::take` by row offset vs. the scan output at those offsets.
    TakeOffsets {
        sel: [u8; 4],
    },
    /// knn on a float vector column; results must be live, non-null, unique, and complete.
    Knn {
        col: u8,
        k: u8,
    },
}

#[derive(Debug, Clone)]
pub(super) enum Op {
    /// Append brand-new rows (keys already present are skipped so ids stay unique).
    Append {
        keys: Vec<u8>,
        data: DataParams,
    },
    /// merge_insert with the full schema: matched rows replaced, unmatched inserted.
    /// `rotate` permutes source columns w.r.t. the dataset schema (this routing detail is what
    /// sent lance#7803 through the in-place partial-update path). `dup_first` appends a
    /// duplicate source row and requires FirstSeen dedupe behavior.
    UpsertFull {
        keys: Vec<u8>,
        rotate: u8,
        dup_first: bool,
        data: DataParams,
    },
    /// merge_insert with a subset of value columns — the partial-update path.
    UpsertPartial {
        keys: Vec<u8>,
        mask: u8,
        rotate: u8,
        data: DataParams,
    },
    /// merge_insert with `WhenMatched::update_if("target.id >= t")`.
    UpsertIf {
        keys: Vec<u8>,
        threshold: u8,
        data: DataParams,
    },
    /// merge_insert with `WhenNotMatchedBySource::Delete` — the source becomes the new content.
    ReplaceAll {
        keys: Vec<u8>,
        data: DataParams,
    },
    /// Bulk delete through merge_insert `WhenMatched::Delete`.
    MergeDelete {
        keys: Vec<u8>,
    },
    /// SQL predicate delete of a single key.
    DeleteEq {
        key: u8,
    },
    /// SQL predicate delete of a key range `[lo, hi)`.
    DeleteRange {
        lo: u8,
        hi: u8,
    },
    /// SQL predicate delete `vN IS NULL` — type-agnostic, works on every column type.
    DeleteNull {
        col: u8,
    },
    /// `delete("true")`.
    DeleteAll,
    /// `UpdateBuilder`: `SET vN = <literal> WHERE id = k` on a SQL-literal-friendly column.
    UpdateLiteral {
        col: u8,
        key: u8,
    },
    /// `UpdateBuilder`: `SET vN = vN WHERE id >= t` — exercises the update path on EVERY type.
    UpdateIdentity {
        col: u8,
        threshold: u8,
    },
    /// Type-appropriate index (btree/bitmap/zonemap/bloomfilter/ngram/inverted/labellist/vector).
    CreateIndex {
        col: u8,
        kind: u8,
    },
    DropIndex {
        sel: u8,
    },
    OptimizeIndices {
        sel: u8,
    },
    Compact {
        sel: u8,
        materialize: bool,
        defer_remap: bool,
    },
    AddColumn {
        fill: AddFill,
    },
    DropColumn {
        sel: u8,
    },
    /// Rewrite the current snapshot using the next concrete Lance file version.
    Migrate,
    /// checkout_version + restore to a previously recorded version.
    Restore {
        sel: u8,
    },
    Read(ReadOp),
}

// Weighted so the dataset stays populated most of the time (the BuzzHouse lesson: uniform op
// choice starves the interesting write/read interleavings because destructive ops win).
impl<'a> Arbitrary<'a> for Op {
    fn arbitrary(u: &mut Unstructured<'a>) -> arbitrary::Result<Self> {
        Ok(match u.int_in_range::<u8>(0..=99)? {
            0..=14 => Self::Append {
                keys: u.arbitrary()?,
                data: u.arbitrary()?,
            },
            15..=24 => Self::UpsertFull {
                keys: u.arbitrary()?,
                rotate: u.arbitrary()?,
                dup_first: u.arbitrary()?,
                data: u.arbitrary()?,
            },
            25..=32 => Self::UpsertPartial {
                keys: u.arbitrary()?,
                mask: u.arbitrary()?,
                rotate: u.arbitrary()?,
                data: u.arbitrary()?,
            },
            33..=37 => Self::UpsertIf {
                keys: u.arbitrary()?,
                threshold: u.arbitrary()?,
                data: u.arbitrary()?,
            },
            38..=40 => Self::ReplaceAll {
                keys: u.arbitrary()?,
                data: u.arbitrary()?,
            },
            41..=44 => Self::MergeDelete {
                keys: u.arbitrary()?,
            },
            45..=47 => Self::DeleteEq {
                key: u.arbitrary()?,
            },
            48..=50 => Self::DeleteRange {
                lo: u.arbitrary()?,
                hi: u.arbitrary()?,
            },
            51..=52 => Self::DeleteNull {
                col: u.arbitrary()?,
            },
            53 => Self::DeleteAll,
            54..=57 => Self::UpdateLiteral {
                col: u.arbitrary()?,
                key: u.arbitrary()?,
            },
            58..=61 => Self::UpdateIdentity {
                col: u.arbitrary()?,
                threshold: u.arbitrary()?,
            },
            62..=69 => Self::CreateIndex {
                col: u.arbitrary()?,
                kind: u.arbitrary()?,
            },
            70..=71 => Self::DropIndex {
                sel: u.arbitrary()?,
            },
            72..=75 => Self::OptimizeIndices {
                sel: u.arbitrary()?,
            },
            76..=80 => Self::Compact {
                sel: u.arbitrary()?,
                materialize: u.arbitrary()?,
                defer_remap: u.arbitrary()?,
            },
            81..=82 => Self::AddColumn {
                fill: u.arbitrary()?,
            },
            83..=84 => Self::DropColumn {
                sel: u.arbitrary()?,
            },
            // One 1% bucket gated at 1-in-10. `ratio` returns true for exhausted input, so
            // exclude that case to keep migration near 0.1% instead of favoring short inputs.
            // The rest of the bucket falls through to `Restore` below — the other history
            // operation — so failing the gate does not inflate the read share.
            87 if !u.is_empty() && u.ratio::<u8>(1, 10)? => Self::Migrate,
            85..=87 => Self::Restore {
                sel: u.arbitrary()?,
            },
            _ => Self::Read(u.arbitrary()?),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fixed_bytes_decode_to_a_fixed_operation() {
        let bytes = [0_u8; 64];
        let op = Op::arbitrary(&mut Unstructured::new(&bytes)).expect("decode operation");

        let Op::Append { keys, data } = op else {
            panic!("zeroed input must select append");
        };
        assert!(keys.is_empty());
        assert!(matches!(data.len, LenClass::Empty));
        assert!(matches!(data.nulls, NullClass::None));
        assert!(!data.nans);
    }
}
