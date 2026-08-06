//! Compaction settings, varied per input so one campaign covers many physical layouts.

use lance::dataset::optimize::{CompactionMode, CompactionOptions, IndexRemapMode};

/// Per-input compaction strategy derived from the shared seed.
#[derive(Clone, Copy, Debug)]
pub(super) enum CompactionProfile {
    Default,
    Knobs([CompactionKnob; 2]),
}

#[derive(Clone, Copy, Debug)]
pub(super) enum CompactionKnob {
    MaxRowsPerGroup(usize),
    MaxBytesPerFile(usize),
    DeletionThreshold(f32),
    NumThreads(usize),
    BatchSize(usize),
    IndexRemapCompact,
    BinaryCopy { read_batch_bytes: Option<usize> },
    MaxSourceFragments(usize),
    OverlayLimit(Option<usize>),
}

impl CompactionKnob {
    const COUNT: u64 = 9;

    fn from_kind(kind: u64, selector: u64) -> Self {
        fn pick<T: Copy>(selector: u64, choices: &[T]) -> T {
            choices[selector as usize % choices.len()]
        }

        match kind {
            0 => Self::MaxRowsPerGroup(pick(selector, &[1, 4, 16])),
            1 => Self::MaxBytesPerFile(pick(selector, &[256, 4 * 1024])),
            2 => Self::DeletionThreshold(pick(selector, &[0.0, 0.5, 1.0])),
            3 => Self::NumThreads(pick(selector, &[1, 2])),
            4 => Self::BatchSize(pick(selector, &[1, 3, 16])),
            5 => Self::IndexRemapCompact,
            6 => Self::BinaryCopy {
                read_batch_bytes: pick(selector, &[None, Some(1), Some(64), Some(4 * 1024)]),
            },
            7 => Self::MaxSourceFragments(pick(selector, &[2, 4])),
            8 => Self::OverlayLimit(pick(selector, &[None, Some(0), Some(1), Some(3)])),
            _ => unreachable!("compaction knob kind is selected modulo COUNT"),
        }
    }

    fn apply(self, options: &mut CompactionOptions) {
        match self {
            Self::MaxRowsPerGroup(value) => options.max_rows_per_group = value,
            Self::MaxBytesPerFile(value) => options.max_bytes_per_file = Some(value),
            Self::DeletionThreshold(value) => {
                options.materialize_deletions_threshold = value;
            }
            Self::NumThreads(value) => options.num_threads = Some(value),
            Self::BatchSize(value) => options.batch_size = Some(value),
            Self::IndexRemapCompact => options.index_remap_mode = IndexRemapMode::Compact,
            Self::BinaryCopy { read_batch_bytes } => {
                options.compaction_mode = Some(CompactionMode::TryBinaryCopy);
                options.binary_copy_read_batch_bytes = read_batch_bytes;
            }
            Self::MaxSourceFragments(value) => options.max_source_fragments = Some(value),
            Self::OverlayLimit(value) => options.max_overlays_per_fragment = value,
        }
    }
}

/// splitmix64 finalizer. Knob and selector derivation must not use plain rotations of the seed:
/// a rotation multiplies the value by a power of two modulo `2^64 - 1`, which 3 divides, so
/// every `rotate % 3k` preserved the seed's residue mod 3 — locking knob kinds and 3-way value
/// selectors to the write profile's rows-per-group class (`seed % 3`).
fn mix(seed: u64) -> u64 {
    let mut z = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

impl CompactionProfile {
    pub(super) fn from_seed(seed: u64) -> Self {
        // The gate stays on the raw seed: mod 4 is independent of every other per-input
        // derivation, and keeping it spares a quarter of all inputs any knob selection.
        if seed & 3 == 0 {
            return Self::Default;
        }

        let mixed = mix(seed);
        let first_kind = mixed % CompactionKnob::COUNT;
        let mut second_kind = (mixed >> 32) % (CompactionKnob::COUNT - 1);
        if second_kind >= first_kind {
            second_kind += 1;
        }
        debug_assert_ne!(first_kind, second_kind);

        Self::Knobs([
            CompactionKnob::from_kind(first_kind, mixed >> 8),
            CompactionKnob::from_kind(second_kind, mixed >> 40),
        ])
    }

    pub(super) fn options(
        self,
        target_rows_per_fragment: usize,
        materialize_deletions: bool,
        defer_index_remap: bool,
        uses_stable_row_ids: bool,
    ) -> CompactionOptions {
        let mut options = CompactionOptions {
            target_rows_per_fragment,
            materialize_deletions,
            // Stable row IDs do not need address remapping, and Lance rejects deferring it.
            defer_index_remap: defer_index_remap && !uses_stable_row_ids,
            ..Default::default()
        };
        if let Self::Knobs(knobs) = self {
            for knob in knobs {
                knob.apply(&mut options);
            }
        }
        options
    }
}

#[cfg(test)]
mod tests {
    use rand_xoshiro::rand_core::{RngCore as _, SeedableRng as _};
    use rand_xoshiro::Xoshiro256PlusPlus;

    use super::*;

    #[test]
    fn compaction_knobs_are_reachable_for_every_file_version() {
        use std::collections::HashSet;
        use std::mem::discriminant;

        let mut seen: [[HashSet<_>; 2]; 4] = Default::default();
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(0);
        for _ in 0..100_000 {
            let seed = rng.next_u64();
            if let CompactionProfile::Knobs(knobs) = CompactionProfile::from_seed(seed) {
                let slots = &mut seen[((seed >> 24) & 3) as usize];
                for (slot, knob) in slots.iter_mut().zip(knobs) {
                    slot.insert(discriminant(&knob));
                }
            }
        }

        for (version, slots) in seen.iter().enumerate() {
            for (slot, kinds) in slots.iter().enumerate() {
                assert_eq!(
                    kinds.len(),
                    CompactionKnob::COUNT as usize,
                    "file version {version} slot {slot} reaches only {} knobs",
                    kinds.len()
                );
            }
        }
    }

    /// The compaction gate and knob selectors share seed bits with the write profile, so joint
    /// coverage is asserted rather than argued from bit ranges: every write profile must reach
    /// default compaction and every knob kind in both slots.
    #[test]
    fn knob_selection_is_jointly_reachable_with_the_write_profile() {
        use std::collections::{BTreeMap, BTreeSet, HashSet};
        use std::mem::{discriminant, Discriminant};

        use crate::dataset_ops::data::WriteProfile;

        type Key = (usize, usize, bool);
        let mut defaults: BTreeSet<Key> = BTreeSet::new();
        let mut kinds: BTreeMap<Key, [HashSet<Discriminant<CompactionKnob>>; 2]> = BTreeMap::new();

        let mut rng = Xoshiro256PlusPlus::seed_from_u64(1);
        for _ in 0..200_000 {
            let seed = rng.next_u64();
            let profile = WriteProfile::from_seed(seed);
            let key = (
                profile.max_rows_per_group,
                profile.max_rows_per_file,
                profile.enable_stable_row_ids,
            );
            match CompactionProfile::from_seed(seed) {
                CompactionProfile::Default => {
                    defaults.insert(key);
                }
                CompactionProfile::Knobs(knobs) => {
                    let slots = kinds.entry(key).or_default();
                    for (slot, knob) in slots.iter_mut().zip(knobs) {
                        slot.insert(discriminant(&knob));
                    }
                }
            }
        }

        // 3 rows-per-group × 3 rows-per-file × 2 stable-row-id settings.
        assert_eq!(
            defaults.len(),
            18,
            "a write profile never compacts by default"
        );
        assert_eq!(kinds.len(), 18, "a write profile never selects knobs");
        for (key, slots) in &kinds {
            for (slot, seen) in slots.iter().enumerate() {
                assert_eq!(
                    seen.len(),
                    CompactionKnob::COUNT as usize,
                    "write profile {key:?} slot {slot} reaches only {} knob kinds",
                    seen.len()
                );
            }
        }
    }
}
