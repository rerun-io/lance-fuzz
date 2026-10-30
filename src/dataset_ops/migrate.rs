//! Lance file version selection and migration, plus the sparse structural-encoding request.

use std::collections::HashMap;
use std::sync::Arc;

use arrow::datatypes::{DataType, Field};
use lance::Dataset;
use lance::dataset::WriteMode;
use lance_encoding::constants::{STRUCTURAL_ENCODING_META_KEY, STRUCTURAL_ENCODING_SPARSE};
use lance_file::version::LanceFileVersion;

use super::data::{WriteProfile, source_reader};

/// Initial file version for one input. Bits 24-25 are read by nothing else per input.
pub(super) fn file_version_from_seed(seed: u64) -> LanceFileVersion {
    match (seed >> 24) & 3 {
        0 => LanceFileVersion::V2_0,
        1 => LanceFileVersion::V2_1,
        2 => LanceFileVersion::V2_2,
        3 => LanceFileVersion::V2_3,
        _ => unreachable!(),
    }
}

/// Whether this input asks its value columns for sparse structural encoding.
///
/// Only file format 2.3 enables the sparse layout: `v2_3` installs the sparse page encoding
/// while earlier versions install `reject_sparse`, which fails the write outright. Bits 20-21
/// read seed bits no other per-input derivation uses, and leaving one of the four values out
/// keeps a quarter of 2.3 inputs on the dense path as controls.
pub(super) fn requests_sparse_encoding(seed: u64, version: LanceFileVersion) -> bool {
    version == LanceFileVersion::V2_3 && (seed >> 20) & 3 != 0
}

/// Whether the sparse writer can encode values of this type.
///
/// It rejects two block shapes outright: dictionary blocks and variable-width packed struct
/// blocks. Automatic selection avoids them by consulting `supports_value_block` and staying
/// dense, but an explicit request reaches the writer and fails the whole write, so the harness
/// must not ask for those types — the same reason `force_binary_copy` is left out of the
/// compaction knobs.
fn sparse_supports_values(dt: &DataType) -> bool {
    fn variable_width(dt: &DataType) -> bool {
        matches!(
            dt,
            DataType::Utf8 | DataType::LargeUtf8 | DataType::Binary | DataType::LargeBinary
        )
    }

    match dt {
        DataType::Dictionary(_, _) => false,
        DataType::Struct(fields) => !fields.iter().any(|f| variable_width(f.data_type())),
        _ => true,
    }
}

/// Whether a field may carry an explicit sparse structural-encoding request.
///
/// Two conditions must hold. The sparse writer plans one layer per Arrow structural layer and
/// errors with "requires at least one Arrow structural layer" when a field has none, so a
/// non-nullable flat primitive (the `id` column) must never be marked; a nullable field always
/// contributes a validity layer. The value type must also be one the sparse writer supports.
fn sparse_eligible(field: &Field) -> bool {
    let has_structural_layer = field.is_nullable()
        || matches!(
            field.data_type(),
            DataType::List(_)
                | DataType::LargeList(_)
                | DataType::FixedSizeList(_, _)
                | DataType::Struct(_)
        );
    has_structural_layer && sparse_supports_values(field.data_type())
}

/// Attach the sparse structural-encoding request to `field` when it is eligible.
pub(super) fn with_sparse_request(field: Field) -> Field {
    if !sparse_eligible(&field) {
        return field;
    }
    let metadata = HashMap::from([(
        STRUCTURAL_ENCODING_META_KEY.to_owned(),
        STRUCTURAL_ENCODING_SPARSE.to_owned(),
    )]);
    field.with_metadata(metadata)
}

/// The version a migration advances to, or `None` at the newest supported version.
pub(super) fn next_file_version(version: LanceFileVersion) -> Option<LanceFileVersion> {
    match version {
        LanceFileVersion::V2_0 => Some(LanceFileVersion::V2_1),
        LanceFileVersion::V2_1 => Some(LanceFileVersion::V2_2),
        LanceFileVersion::V2_2 => Some(LanceFileVersion::V2_3),
        LanceFileVersion::V2_3 => None,
        LanceFileVersion::Legacy | LanceFileVersion::Stable | LanceFileVersion::Next => {
            panic!("expected an exact v2 Lance file version, got {version}")
        }
    }
}

pub(super) fn dataset_file_version(dataset: &Dataset) -> LanceFileVersion {
    dataset
        .manifest()
        .data_storage_format
        .lance_file_format()
        .to_selector()
}

pub(super) type SchemaSignature = Vec<(String, DataType, bool, HashMap<String, String>)>;

/// The logical schema a migration must preserve: per field, name, type, nullability, and metadata
/// (which carries the sparse-encoding request).
pub(super) fn schema_signature(dataset: &Dataset) -> SchemaSignature {
    dataset
        .schema()
        .fields
        .iter()
        .map(|field| {
            (
                field.name.clone(),
                field.data_type(),
                field.nullable,
                field.metadata.clone(),
            )
        })
        .collect()
}

pub(super) fn assert_data_file_versions(dataset: &Dataset, target: LanceFileVersion, ctx: &str) {
    let expected = target.resolve();
    let mut files = 0;
    for fragment in dataset.iter_fragments() {
        for file in &fragment.files {
            files += 1;
            let actual = file
                .file_version()
                .unwrap_or_else(|e| panic!("{ctx}: invalid data file version: {e}"));
            assert_eq!(
                actual, expected,
                "{ctx}: data file was not written in the requested version"
            );
        }
    }
    assert!(files > 0, "{ctx}: migration wrote no data files");
}

/// Rewrite the current snapshot in the `target` file version.
///
/// The overwrite discards the old index data, so the caller must observe that and rebuild every
/// live index; keeping the rebuild out of here is what lets the caller assert the discard happened.
pub(super) async fn migrate_dataset(
    dataset: &Dataset,
    target: LanceFileVersion,
    write_profile: WriteProfile,
) -> Result<Dataset, lance::Error> {
    // The u8 key domain bounds the dataset to 256 live rows, so materializing one batch keeps the
    // migration path simple without exposing DataFusion's stream type in this harness.
    let batch = dataset.scan().try_into_batch().await?;
    let mut params = write_profile.write_params();
    params.mode = WriteMode::Overwrite;
    params.data_storage_version = Some(target);
    Dataset::write(
        source_reader(batch),
        Arc::new(dataset.clone()),
        Some(params),
    )
    .await
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow::array::{ArrayRef, RecordBatch, UInt64Array, new_null_array};
    use arrow::datatypes::{Field, Schema, TimeUnit};
    use lance::index::DatasetIndexExt as _;
    use lance_index::scalar::BuiltinIndexType;
    use rand_xoshiro::Xoshiro256PlusPlus;
    use rand_xoshiro::rand_core::{RngCore as _, SeedableRng as _};

    use super::*;
    use crate::dataset_ops::data::{WriteProfile, make_generator};
    use crate::dataset_ops::index::{IndexEntry, IndexKind, create_index, rebuild_indices};
    use crate::dataset_ops::ops::{ColSpec, FslChild, FuzzInput, LenClass, Op, PrimSpec};
    use crate::dataset_ops::runner::run;

    #[test]
    fn sparse_requests_are_gated_to_2_3_and_eligible_fields() {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(0);
        let mut sparse_at_2_3 = 0usize;
        let mut dense_at_2_3 = 0usize;
        for _ in 0..100_000 {
            let seed = rng.next_u64();
            let version = file_version_from_seed(seed);
            let requested = requests_sparse_encoding(seed, version);
            if version == LanceFileVersion::V2_3 {
                if requested {
                    sparse_at_2_3 += 1;
                } else {
                    dense_at_2_3 += 1;
                }
            } else {
                assert!(!requested, "{version} must never request sparse encoding");
            }
        }
        assert!(sparse_at_2_3 > 0, "2.3 never requests sparse encoding");
        assert!(dense_at_2_3 > 0, "2.3 keeps no dense controls");

        // The non-nullable `id` column has no structural layer and must stay dense.
        assert!(!sparse_eligible(&Field::new("id", DataType::UInt64, false)));
        assert!(
            with_sparse_request(Field::new("id", DataType::UInt64, false))
                .metadata()
                .is_empty()
        );

        // Value columns are nullable, so eligibility turns on whether the sparse writer
        // supports the value type. The writer rejects dictionary blocks and variable-width
        // packed struct blocks, and an explicit request for either fails the write.
        let eligible = [
            ColSpec::Prim(PrimSpec::Int64),
            ColSpec::Prim(PrimSpec::Utf8),
            ColSpec::List {
                child: PrimSpec::Int64,
                large: false,
            },
            ColSpec::Fsl {
                child: FslChild::Float32,
                dim: 3,
            },
            ColSpec::Struct {
                a: PrimSpec::Int64,
                b: PrimSpec::Date32,
            },
            ColSpec::Null,
        ];
        let ineligible = [
            ColSpec::Dict { utf8_values: true },
            ColSpec::Dict { utf8_values: false },
            ColSpec::Struct {
                a: PrimSpec::Int64,
                b: PrimSpec::Utf8,
            },
            ColSpec::Struct {
                a: PrimSpec::Binary,
                b: PrimSpec::Int64,
            },
        ];

        for spec in eligible {
            let field = Field::new("v", spec_data_type(&spec), true);
            assert!(sparse_eligible(&field), "{spec:?} must be sparse-eligible");
            assert_eq!(
                with_sparse_request(field)
                    .metadata()
                    .get(STRUCTURAL_ENCODING_META_KEY),
                Some(&STRUCTURAL_ENCODING_SPARSE.to_owned()),
                "{spec:?} did not receive the sparse request"
            );
        }
        for spec in ineligible {
            let field = Field::new("v", spec_data_type(&spec), true);
            assert!(
                !sparse_eligible(&field),
                "{spec:?} must not be sparse-eligible"
            );
            assert!(
                with_sparse_request(field).metadata().is_empty(),
                "{spec:?} was marked sparse despite being unsupported"
            );
        }
    }

    fn spec_data_type(spec: &ColSpec) -> DataType {
        make_generator(spec, LenClass::Small, 0.0)
            .data_type()
            .clone()
    }

    #[tokio::test]
    async fn migrate_advances_one_file_version() {
        let timestamp_type = DataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into()));
        let timestamp_field = Field::new("ts", timestamp_type.clone(), true)
            .with_metadata(HashMap::from([("source".to_owned(), "fuzz".to_owned())]));
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::UInt64, false),
            timestamp_field,
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(UInt64Array::from(vec![1, 2])) as ArrayRef,
                new_null_array(&timestamp_type, 2),
            ],
        )
        .expect("create batch");
        let write_profile = WriteProfile {
            max_rows_per_group: 1,
            max_rows_per_file: 2,
            enable_stable_row_ids: false,
        };
        let mut params = write_profile.write_params();
        params.mode = WriteMode::Overwrite;
        params.data_storage_version = Some(LanceFileVersion::V2_0);
        let mut dataset = Dataset::write(source_reader(batch.clone()), "memory://", Some(params))
            .await
            .expect("create v2.0 dataset");

        let indices = [IndexEntry {
            name: "id_idx".to_owned(),
            col: "id".to_owned(),
            kind: IndexKind::Scalar(BuiltinIndexType::BTree),
        }];
        create_index(&mut dataset, &indices[0])
            .await
            .expect("create index");
        let indexed_version = dataset.version().version;
        let schema_before = schema_signature(&dataset);
        assert!(!schema_before[0].2);
        assert!(schema_before[1].2);
        assert_eq!(schema_before[1].1, timestamp_type);
        assert_eq!(
            schema_before[1].3.get("source").map(String::as_str),
            Some("fuzz")
        );

        for expected in [
            LanceFileVersion::V2_1,
            LanceFileVersion::V2_2,
            LanceFileVersion::V2_3,
        ] {
            let current = dataset_file_version(&dataset);
            let target = next_file_version(current).expect("next file version");
            assert_eq!(target, expected);
            dataset = migrate_dataset(&dataset, target, write_profile)
                .await
                .expect("migrate dataset");
            assert_eq!(dataset_file_version(&dataset), expected);
            assert_eq!(
                dataset.scan().try_into_batch().await.expect("scan dataset"),
                batch
            );
            assert_eq!(schema_signature(&dataset), schema_before);
            assert_data_file_versions(&dataset, target, "test migration");

            // The overwrite must discard the old index data; only then does the rebuild restore it.
            assert!(
                dataset
                    .load_indices()
                    .await
                    .expect("load indices")
                    .is_empty()
            );
            rebuild_indices(&mut dataset, &indices, "test migration").await;
            let rebuilt = dataset.load_indices().await.expect("load rebuilt indices");
            assert_eq!(rebuilt.len(), 1);
            assert_eq!(rebuilt[0].name, "id_idx");
            dataset.validate().await.expect("validate dataset");
        }

        let version = dataset.version().version;
        assert_eq!(next_file_version(dataset_file_version(&dataset)), None);
        assert_eq!(dataset.version().version, version);

        let mut old = dataset
            .checkout_version(indexed_version)
            .await
            .expect("checkout indexed version");
        old.restore().await.expect("restore indexed version");
        assert_eq!(dataset_file_version(&old), LanceFileVersion::V2_0);
        assert_eq!(schema_signature(&old), schema_before);
        assert_eq!(
            old.load_indices()
                .await
                .expect("load restored indices")
                .len(),
            1
        );
    }

    #[tokio::test]
    async fn migration_skips_empty_dataset() {
        run(FuzzInput {
            seed: 0,
            schema: vec![ColSpec::Prim(PrimSpec::Int64)],
            ops: vec![Op::Migrate],
        })
        .await;
    }
}
