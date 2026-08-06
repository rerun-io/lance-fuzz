//! Scan helpers shared by the oracle, index, and read checks.

use std::collections::BTreeSet;

use arrow::array::{AsArray as _, RecordBatch};
use arrow::datatypes::UInt64Type;
use futures::TryStreamExt as _;
use lance::dataset::scanner::Scanner;
use lance::Dataset;

/// Run a scan configured by `configure`, panicking with `ctx` at whichever stage fails.
pub(super) async fn collect_scan<F>(dataset: &Dataset, configure: F, ctx: &str) -> Vec<RecordBatch>
where
    F: FnOnce(&mut Scanner) -> Result<(), lance::Error>,
{
    let mut scan = dataset.scan();
    configure(&mut scan).unwrap_or_else(|e| panic!("{ctx}: scan config failed: {e}"));
    scan.try_into_stream()
        .await
        .unwrap_or_else(|e| panic!("{ctx}: scan plan failed: {e}"))
        .try_collect::<Vec<_>>()
        .await
        .unwrap_or_else(|e| panic!("{ctx}: scan execution failed: {e}"))
}

/// Key set matching `filter`. Duplicate ids fail here rather than in the caller's comparison.
pub(super) async fn collect_ids(
    dataset: &Dataset,
    filter: &str,
    use_scalar_index: bool,
    prefilter: bool,
    ctx: &str,
) -> BTreeSet<u64> {
    let batches = collect_scan(
        dataset,
        |scan| {
            scan.project(&["id"])?;
            scan.filter(filter)?;
            scan.use_scalar_index(use_scalar_index);
            scan.prefilter(prefilter);
            Ok(())
        },
        ctx,
    )
    .await;
    let mut ids = BTreeSet::new();
    for batch in &batches {
        let column = batch.column(0).as_primitive::<UInt64Type>();
        for row in 0..column.len() {
            assert!(
                ids.insert(column.value(row)),
                "{ctx}: duplicate id {} in result",
                column.value(row)
            );
        }
    }
    ids
}
