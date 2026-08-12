# Fuzz campaign: minimization phase

Follow-on to step 7 of `.claude/skills/fuzz/SKILL.md`. Read this at the moment the fuzz run has
ended and the final `crash-import` sweep is done. It is a separate file on purpose: this phase runs
many hours after the skill was loaded, usually after context compaction — treat this text, not your
memory of the skill, as authoritative.

This is the most judgment-heavy phase. Do not skip straight to spawning agents.

## 1. Consolidate before delegating

For every class in `crashes/dataset_ops/manifest.tsv` that doesn't yet have a matching
`bugs/<slug>/` directory (or whose existing reproducer's panic site no longer matches), replay one
representative artifact per class:

```sh
cargo run --bin replay -- crashes/dataset_ops/<artifact>.bin
REPLAY_DUMP=1 cargo run --bin replay -- crashes/dataset_ops/<artifact>.bin   # decoded op sequence
```

Group classes that hit the **identical file:line** panic site (or, for harness-assertion classes,
whose underlying Lance-side cause you can show is the same) into one real bug — do this the way the
2026-08-11 campaign found four different ops (`Compact`, `UpdateIdentity`, `UpdateLiteral`,
`UpsertPartial`) all panicking at the same `primitive.rs:6631`. One agent per *bug*, not one per
manifest row.

## 2. Shared build scaffold

If `bugs/Cargo.toml` doesn't exist yet, create it: a standalone package (own `[workspace] members
= ["."]`, so it doesn't try to join the fuzz-dir workspace) depending on `lance`, `lance-index`,
`lance-datagen`, `arrow`, `futures`, `tokio`, `tempfile`, pinned to the **same tag** as the root
`Cargo.toml`, with one `[[bin]]` entry per bug pointing at `bugs/<slug>/repro.rs`. `bugs/` is
gitignored in this repo already — it's a local scratch/reporting area, not part of the crate.

Create a placeholder `bugs/<slug>/repro.rs` (`fn main() { todo!() }`) for every new bug before
spawning agents, and kick off a `cargo check` in `bugs/` yourself first to pre-warm the shared
dependency build — 8 agents cold-compiling the same large dependency graph at once is what caused
mass stalls last time; one warm build first avoids that.

## 3. One Opus subagent per bug, in parallel

For each new or changed bug, launch an `Agent` (`model: "opus"`) with a **self-contained** prompt
covering: the repo path; that the machine may be under heavy concurrent load (this run's own fuzz
process, possibly others — don't kill anything); the representative artifact(s) and how to decode
them (`REPLAY_DUMP=1 cargo run --bin replay --`); which harness source files
(`src/dataset_ops/{ops,runner,index,oracle,scalar}.rs`) show how the op maps to real `lance` API
calls; where the exact pinned-version Lance source is checked out for root-cause reading (find it
from a captured panic path, under `~/.cargo` or wherever this machine's checkouts live — do not
guess a path, read it off an actual backtrace first); the requirement to use **only** the public
`lance`/`lance-index`/`lance-datagen` API in `repro.rs` (no `lance_fuzz` imports — verify
afterward with `grep -rn lance_fuzz bugs/*/repro.rs`, expect zero matches); instructions to
minimize aggressively and write both `repro.rs` and a `README.md` (bug description, panic
site/message, minimized case size, how to run); and a requirement to build and run it themselves
before reporting done. Tell each agent not to edit `bugs/Cargo.toml` unless a genuinely new
dependency is needed (siblings share that file).

If the fuzz process (or another heavy job) is still consuming all cores, cap the fan-out at ~4
concurrent agents rather than launching all bugs at once — machine contention is what turns quiet
compiles into apparent hangs.

If two bugs are mechanically related (e.g. a partial- and full-schema variant of the same
underlying defect), say so in both prompts and let the second agent read the first's finished
README once it's done, rather than have both investigate from scratch.

## 4. Handle stalls without assuming failure

A background agent reporting "stalled: no progress for 600s" almost always means it was mid-build
on a contended machine, not stuck. Resume it (`SendMessage` to its agent id) rather than
re-launching from scratch: tell it the likely cause, that the shared dependency build may now be
warmer, and to use a generous Bash timeout (`timeout: 900000`, 15 min) or `run_in_background` for
its next build/run so a quiet compile isn't mistaken for a hang again.

## 5. Verify independently — do not just relay the agent's report

For every finished `bugs/<slug>/`, build and run it yourself:

```sh
cd bugs && cargo run --bin <slug>
```

Confirm it actually panics/asserts with the reported message before treating the bug as done. Only
then update your summary and any tracked crash-suite bug-column entries.

## 6. Cross-check known issues

For every new or updated bug, search upstream before writing it up as novel:

```sh
gh search issues --repo lance-format/lance "<keywords>" --limit 10 --json number,title,state,url
gh issue view <number> --repo lance-format/lance --json number,title,state,body
```

Try a few keyword variants per bug (symptom, function name, panic message fragment) — exact
phrasing rarely matches. Add a short "Known issue" / "Known-issue check" line at the top of each
`README.md` recording what you found either way (exact match and why; adjacent-but-distinct and
why; or "no related issue found after searching for X, Y, Z"). Do not open new upstream issues or
post comments without the user asking — recording the cross-check is the deliverable here, filing
is a separate, explicit request.

## 7. Record the quantitative outcome

If this session has the auto-memory system available, write (or extend) a project memory entry
with the campaign's timeline and per-check-in results — a multi-hour run risks losing this to
context compaction otherwise. Keep bug *content* in the `bugs/*/README.md` files themselves;
memory should index what happened and when, not duplicate the technical writeups.
