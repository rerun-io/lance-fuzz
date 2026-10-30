---
name: fuzz
description: Run a full lance-fuzz campaign in this repo — bump to the latest Lance tag, verify the tracked crash suite still reproduces, clear the corpus for a clean run, fuzz dataset_ops for a caller-given duration while classifying new crashes as they appear, minimize every finding into a standalone bugs/<slug>/ reproducer afterward, and ask before spending time on a coverage report. Use when the user asks to run a fuzz campaign, port Lance to a new version and fuzz it, or do an overnight/timed fuzzing run in lance-fuzz.
---

# Fuzz campaign

Encodes the workflow validated in the 2026-08-11→12 overnight campaign on this repo. Read this
whole file before starting — the phases depend on each other, and several steps exist specifically
because they failed a different way the first time.

## 0. Get the duration or end time — do not guess it

The caller must say how long to fuzz, either as a duration (`2h`, `90m`, `1800s`) or a clock time to
stop at (`until 08:00`, `until tomorrow 6am`). If the skill was invoked without one, ask before doing
anything else. Convert an end time to seconds with `date`, e.g.:

```sh
now=$(date +%s); target=$(date -d "tomorrow 08:00" +%s); secs=$((target - now))
```

Recompute `now` right before actually launching the fuzzer (step 3) — don't reuse a value computed
several steps earlier.

## 1. Check for concurrent work on the machine first

`CARGO_TARGET_DIR` and the object cache on this machine are shared across checkouts and sessions
(see project memory: "Shared cargo target dir hazard"). Before doing anything else:

```sh
pgrep -af "cargo|dataset_ops|fuzz-coverage" | grep -v grep
```

If you find an unrelated build or fuzz run already going, **do not kill it and do not assume it's
stale**. Ask the user how to proceed (leave both running / kill it / scale this run's `-workers` count
down to share cores) before continuing. Heavy concurrent load also means later steps (agent builds
especially) may take much longer than normal and should not be treated as hung — see step 6.

Also preflight what the later phases need — cheap now, painful to discover missing hours in:

```sh
gh auth status          # step 7's known-issue cross-check fails silently without it
mkdir -p logs           # step 5 redirects here before pixi gets a chance to create it
```

If `gh` auth is expired, tell the user now rather than proceeding and hitting it after the run.

## 2. Bump to the latest Lance tag

```sh
git ls-remote --tags https://github.com/lance-format/lance | grep -v '\^{}' | \
  awk -F'refs/tags/' '{print $2}' | sort -V | tail -5
```

Edit `Cargo.toml`: update the tag on **every** `lance-*` git dependency (don't assume a count —
grep for `tag =` and cover them all) and the version mentioned in the comment above them, to the
newest tag. If `bugs/Cargo.toml` exists from a previous campaign, bump its lance tags too —
otherwise step 7's reproducer verification silently runs against the old version. Then:

```sh
cargo update
cargo build --bin replay --bin dataset_ops
cargo fmt --all --check
cargo nextest run --lib
pixi run fuzz-build
```

Fix any break each step surfaces before moving on — don't stack unverified changes. If nothing
needed a source change (the common case for a beta-to-beta bump), say so plainly rather than
padding the report.

## 3. Port crashes: verify the tracked suite against the new version

```sh
pixi run crash-check
```

Report the same/gone/changed breakdown. If the header warns the harness fingerprint (`src/dataset_ops/ops.rs`
hash) changed since the suite was recorded, treat any mass class change as a stale baseline and
re-import (`pixi run crash-import`) instead of reading it as a wave of Lance regressions — this is
called out explicitly in `README.md`'s "Updating source code" section.

Only touch the `bug` column in `crashes/dataset_ops/manifest.tsv` for entries that actually changed
class or disappeared; leave `open`/`harness` entries alone.

## 4. Clear the corpus — this is a clean-run campaign by design

```sh
rm -rf corpus/dataset_ops/*
```

This is intentional, standing behavior for this skill (that's the point of running it — a clean
coverage-guided corpus for the new version, not carried-over state from the last campaign) and
needs no extra confirmation each time. `corpus/` is gitignored and local-only, so this has no git
impact. Do not clear `crashes/` or `artifacts/` — those are the tracked regression suite and raw
findings, respectively.

## 5. Start the fuzz run as a detached background process

```sh
now=$(date +%s); secs=<computed above>
nohup pixi run fuzz "$secs" > logs/campaign-fuzz.out 2>&1 &
disown
```

Use `nohup ... &; disown`, not the Bash tool's `run_in_background`, for the fuzz process itself —
it needs to survive independently of this session's process tree for a run that can span many
hours. Verify it actually started (`pgrep -af dataset_ops`, check the log) before moving on.

Note the `pixi run fuzz` task does its own redirect: the real fuzzer output goes to the directory
`logs/dataset_ops-<timestamp>/` (one `fuzz-<n>.log` per worker, plus `run.log`), and
`campaign-fuzz.out` captures little more than the one-line "logging to <path>" message. Read that
path out of `campaign-fuzz.out` once, and monitor the worker logs there for stats and crash counts
— tailing `campaign-fuzz.out` tells you nothing.

**Tell the user now, once, clearly:** the fuzz process itself will keep running even if this
session closes, but the periodic classification (step 6) and the automatic wrap-up (step 7) are
tied to this session and stop if it closes. A multi-hour run needs the session kept open for those
parts to happen on schedule; if it closes early, crashes still land in `artifacts/dataset_ops/` and
can be classified/minimized later by asking again.

## 6. Classify continuously while it runs

Set up a recurring check (`CronCreate`, interval ~30 minutes — cron jobs here are session-only and
fire while idle, which fits this workflow) with a **self-contained** prompt, since it may fire long
after this conversation's start and may run under context compaction. It must state: the repo path,
that a `pixi run fuzz` background process is running toward a known end time, and each firing
should:

1. Check the current time.
2. Before the end time: run `pixi run crash-import`. If it added inputs, briefly name the new
   class(es) (site + message, one line each). If it added 0, say so in one line — do not pad a
   no-op tick with detail.
3. At/after the end time (the fuzz process should have exited on its own by then — if `pgrep -af
   dataset_ops` still shows the real binary, not just this poll loop's own command line, wait for
   it before continuing): run a final `crash-import` sweep, then **read
   `<repo>/.claude/skills/fuzz/minimize.md` and follow it** (updating `bugs/` — the cron prompt
   must include this file path verbatim, since by then the session has likely been compacted and
   this skill text is no longer reliably in context), then delete this cron job (`CronDelete`) once
   the minimization phase and the step 8 question are both done.

A new crash class is not automatically a new *bug* — several tracked classes can share one root
cause (op name differs, panic site doesn't). Defer grouping to step 7, where you'll replay
representative artifacts anyway; don't try to dedupe mid-campaign.

## 7. After the fuzz run ends: update `bugs/`

This is the most judgment-heavy phase, and it runs many hours after this file was read — usually
after context compaction. Do not work from memory of these instructions: **read
`.claude/skills/fuzz/minimize.md` (next to this file) and follow it.** It covers consolidating
crash classes into real bugs, the shared `bugs/` build scaffold, per-bug Opus subagents, stall
handling, independent verification of every reproducer, upstream known-issue cross-checks, and
recording the outcome to project memory.

## 8. Ask before computing coverage — never run it automatically

`pixi run fuzz-coverage` rebuilds with coverage instrumentation and replays the entire corpus; it
is slow and not everyone wants it every time. Ask explicitly once steps 1–7 are done:

> Calculate coverage now? (`pixi run fuzz-coverage` + `fuzz-coverage-report`, replays the full
> corpus, can take several minutes to tens of minutes depending on machine load)

If yes, run both tasks and report total line coverage, the per-crate table, and the fully-dark
(0%, large) file list. If no, skip it and say so in the final summary.

## 9. Final summary

Report, in this order: what version was bumped to and whether anything broke; the crash-check
result before fuzzing; corpus/coverage counters and elapsed time from the fuzz run itself; how many
crash-suite classes/inputs were added; the list of new/updated bugs with one line each (symptom +
known-issue cross-reference result, not full root-cause detail — that's what the READMEs are for);
and the coverage result if computed. Keep it scannable; link to `bugs/*/README.md` and
`crashes/dataset_ops/manifest.tsv` rather than re-deriving their content in the chat.
