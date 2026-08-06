#!/usr/bin/env python3
"""Re-run the preserved crash inputs and report what a code change did to them.

The point of a fuzz campaign is not the artifacts it drops, it is knowing whether the bugs are still
there next month. This keeps a small tracked set of crash inputs plus the class each one *used* to
produce, replays them against the current build, and sorts the results into three buckets:

    same class      still broken, nothing to do
    no crash        candidate fix — go update the bug report
    class changed   behaviour moved; could be a partial fix, a reworded error, or a new bug

The third bucket is the one that matters and the reason a plain "did it crash?" check is not enough:
`cargo fuzz tmin` once silently turned a bug-018 artifact into a bug-001 one, so "still crashes" is
not the same claim as "still this bug".

Runs on the stable `replay` binary, so no nightly toolchain, no cargo-fuzz, and no instrumented
build are needed — a Lance version bump plus `cargo build --bin replay` is the whole setup.

Usage:
    hack/crash-check.py [target] [name ...]        replay the suite (default: all of it)
    hack/crash-check.py [target] --import [dir]    add artifacts to the suite
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from triage import (
    ARTIFACT_PREFIXES,
    FAILED_TO_START,
    NO_PANIC,
    REPO,
    TIMED_OUT,
    classify,
    find_replay,
    panic_block,
)

MANIFEST_NAME = "manifest.tsv"
UNASSIGNED = "?"

# Words that carry no information about *which* bug a class is, so they are dropped from the
# generated filename. The full class stays in the manifest either way.
SLUG_STOPWORDS = {
    "a",
    "and",
    "assertion",
    "at",
    "be",
    "ctx",
    "error",
    "failed",
    "for",
    "in",
    "is",
    "left",
    "n",
    "not",
    "of",
    "op",
    "right",
    "the",
    "to",
    "was",
    "with",
}
WORD = re.compile(r"[A-Za-z]+")


# --- suite layout ------------------------------------------------------------------------------


def suite_dir(target: str) -> Path:
    return REPO / "crashes" / target


def content_id(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:8]


def harness_source(target: str) -> Path:
    """The harness source file that defines how stored input bytes decode.

    For module targets that is `ops.rs` — it holds the `Arbitrary` input types (`FuzzInput`,
    `Op`, `ColSpec`) and their weighting; `mod.rs` is just module declarations and would never
    change when the layout does.
    """
    source = REPO / "src" / f"{target}.rs"
    if not source.is_file():
        source = REPO / "src" / target / "ops.rs"
    return source


def harness_fingerprint(target: str) -> str:
    """Fingerprint the harness source that decides how these bytes decode.

    Every stored input is an `arbitrary`-decoded session, so editing the input types re-interprets
    the whole suite: the inputs still run and still crash, but they no longer describe the sessions
    that were recorded. That failure is silent, so it gets a loud warning.
    """
    source = harness_source(target)
    if not source.is_file():
        return "unknown"
    return hashlib.sha256(source.read_bytes()).hexdigest()[:12]


def lance_tag() -> str:
    """The Lance tag every lance-* dependency is pinned to, for the baseline record."""
    match = re.search(r'tag\s*=\s*"([^"]+)"', (REPO / "Cargo.toml").read_text())
    return match.group(1) if match else "unknown"


def slug(cls: str) -> str:
    """Readable filename stem for a class, so `ls crashes/` is informative on its own."""
    site, _, message = cls.partition(" | ")
    stem = Path(site).stem or "crash"
    words = [w.lower() for w in WORD.findall(message)]
    words = [w for w in words if w not in SLUG_STOPWORDS][:8]
    tail = "-".join(words)[:48].strip("-")
    return f"{stem}-{tail}" if tail else stem


# --- manifest ----------------------------------------------------------------------------------


class Entry:
    """One suite member: the input file, the class it is expected to produce, and its bug number."""

    def __init__(self, name: str, bug: str, cls: str) -> None:
        self.name = name
        self.bug = bug
        self.cls = cls


def read_manifest(target: str) -> list[Entry]:
    path = suite_dir(target) / MANIFEST_NAME
    if not path.is_file():
        return []
    entries = []
    for line in path.read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        fields = line.split("\t")
        if len(fields) >= 3:
            entries.append(Entry(fields[0], fields[1], fields[2]))
    return entries


def write_manifest(target: str, entries: list[Entry]) -> None:
    """Rewrite the manifest, sorted by bug then name so diffs stay readable."""
    lines = [
        f"# Crash regression suite for {target}. Replay it with: pixi run crash-check",
        f"# baseline: lance {lance_tag()}, harness"
        f" {harness_source(target).relative_to(REPO)} sha {harness_fingerprint(target)}",
        "# columns: input<TAB>bug<TAB>expected class     (the bug column is hand-maintained)",
    ]
    for entry in sorted(entries, key=lambda e: (e.bug, e.name)):
        lines.append(f"{entry.name}\t{entry.bug}\t{entry.cls}")
    path = suite_dir(target) / MANIFEST_NAME
    path.write_text("\n".join(lines) + "\n")


def baseline_fingerprint(target: str) -> str | None:
    path = suite_dir(target) / MANIFEST_NAME
    if not path.is_file():
        return None
    for line in path.read_text().splitlines():
        match = re.search(r"harness src/\S+ sha (\w+)", line)
        if match:
            return match.group(1)
    return None


# --- import ------------------------------------------------------------------------------------


def do_import(target: str, dirs: list[Path], per_class: int, jobs: int, picked: bool, build: bool) -> int:
    """Add artifacts to the suite.

    `picked` means the directories were named on the command line rather than swept by default. Those
    files were chosen by hand — typically the reproducer an existing bug report points at — so they
    bypass both the `crash-` name filter and the per-class quota. Content hashing keeps re-imports
    idempotent either way.
    """
    suite = suite_dir(target)
    suite.mkdir(parents=True, exist_ok=True)

    existing = read_manifest(target)
    # A class already at its quota is left alone: the suite is maintained by hand from here on, and
    # re-importing should add coverage, never churn the files a bug report points at.
    by_class: dict[str, list[Entry]] = {}
    for entry in existing:
        by_class.setdefault(entry.cls, []).append(entry)
    known_bug = {entry.cls: entry.bug for entry in existing if entry.bug != UNASSIGNED}

    candidates = []
    for path in dirs:
        if path.is_file():  # a single named artifact, e.g. the repro behind one bug report
            candidates.append(path)
            continue
        if not path.is_dir():
            print(f"skipping {path}: not a file or directory", file=sys.stderr)
            continue
        candidates += [
            artifact
            for artifact in sorted(path.iterdir())
            if artifact.is_file() and (picked or artifact.name.startswith(ARTIFACT_PREFIXES))
        ]
    if not candidates:
        print(f"no artifacts found in: {', '.join(str(d) for d in dirs)}", file=sys.stderr)
        return 1

    # Every artifact is replayed here rather than reusing the triage daemon's log. The log is keyed by
    # artifact name only, so it silently mixes classes recorded by different builds — which is how the
    # first attempt at this suite ended up with one crash filed under two different panic sites.
    replay = require_replay(target, build)
    print(f"replaying {len(candidates)} artifacts across {jobs} jobs", flush=True)
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        blocks = list(pool.map(lambda a: panic_block(replay, a, target), candidates))
    cache = {a.name: classify(b) for a, b in zip(candidates, blocks)}

    # Smallest input first: a shorter session is a shorter story when the bug report gets written.
    grouped: dict[str, list[Path]] = {}
    for artifact in candidates:
        cls = cache[artifact.name]
        if cls.endswith(NO_PANIC) or cls.endswith(TIMED_OUT) or FAILED_TO_START in cls:
            continue
        grouped.setdefault(cls, []).append(artifact)

    added = 0
    for cls, artifacts in sorted(grouped.items()):
        held = by_class.get(cls, [])
        held_ids = {entry.name.rsplit("-", 1)[-1].removesuffix(".bin") for entry in held}
        artifacts.sort(key=lambda a: (a.stat().st_size, a.name))
        for artifact in artifacts:
            if not picked and len(held) >= per_class:
                break
            data = artifact.read_bytes()
            ident = content_id(data)
            if ident in held_ids:
                continue
            name = f"{slug(cls)}-{ident}.bin"
            shutil.copyfile(artifact, suite / name)
            entry = Entry(name, known_bug.get(cls, UNASSIGNED), cls)
            held.append(entry)
            held_ids.add(ident)
            existing.append(entry)
            added += 1

    write_manifest(target, existing)
    classes = len({entry.cls for entry in existing})
    print(f"added {added} inputs; suite now {len(existing)} inputs over {classes} classes")
    print(f"suite: {suite.relative_to(REPO)}/")
    unassigned = sum(1 for e in existing if e.bug == UNASSIGNED)
    if unassigned:
        print(f"{unassigned} inputs have no bug number — fill the bug column in {MANIFEST_NAME}")
    return 0


# --- check -------------------------------------------------------------------------------------


class Result:
    def __init__(self, entry: Entry, cls: str, flaky: bool) -> None:
        self.entry = entry
        self.cls = cls
        self.flaky = flaky

    @property
    def bucket(self) -> str:
        if self.cls == self.entry.cls:
            return "same"
        if self.cls.endswith(NO_PANIC):
            return "gone"
        if self.cls.endswith(TIMED_OUT) or FAILED_TO_START in self.cls:
            return "error"
        return "changed"


def check_one(replay: Path, target: str, suite: Path, entry: Entry, retries: int) -> Result:
    """Replay one input; on a mismatch, try again before believing it.

    The retry is not paranoia: bug 012 is wall-clock nondeterminism in Lance's data generator, and
    two artifacts from an earlier campaign stopped reproducing for that reason alone. A one-shot
    check would report those as fixes.
    """
    cls = classify(panic_block(replay, suite / entry.name, target))
    flaky = False
    for _ in range(retries):
        if cls == entry.cls:
            break
        cls = classify(panic_block(replay, suite / entry.name, target))
        flaky = cls == entry.cls
    return Result(entry, cls, flaky)


def do_check(target: str, names: list[str], retries: int, jobs: int, build: bool) -> int:
    suite = suite_dir(target)
    entries = read_manifest(target)
    if not entries:
        print(
            f"no crash suite for {target} — build one with: pixi run crash-import",
            file=sys.stderr,
        )
        return 1
    if names:
        wanted = set(names)
        entries = [e for e in entries if e.name in wanted or any(n in e.name for n in wanted)]
        if not entries:
            print(f"no suite inputs match: {' '.join(names)}", file=sys.stderr)
            return 1

    missing = [e.name for e in entries if not (suite / e.name).is_file()]
    if missing:
        print(f"manifest lists {len(missing)} missing inputs: {missing[0]} ...", file=sys.stderr)
        return 1

    replay = require_replay(target, build)
    recorded, current = baseline_fingerprint(target), harness_fingerprint(target)
    print(f"=== {target}: {len(entries)} crash inputs vs lance {lance_tag()} ===")
    if recorded and recorded != current:
        print(
            f"\nWARNING: harness {harness_source(target).relative_to(REPO)} changed since this"
            " suite was recorded"
            f" ({recorded} -> {current}).\n"
            "         These bytes decode through the harness input types, so a layout change"
            " silently\n"
            "         re-interprets every input. Mass class changes below are a stale baseline,"
            " not\n"
            "         Lance regressions — re-import instead of filing bugs.\n"
        )

    with ThreadPoolExecutor(max_workers=jobs) as pool:
        results = list(pool.map(lambda e: check_one(replay, target, suite, e, retries), entries))

    buckets: dict[str, list[Result]] = {"same": [], "gone": [], "changed": [], "error": []}
    for result in results:
        buckets[result.bucket].append(result)

    same = buckets["same"]
    print(f"\nstill reproducing   {len(same):>4}", end="")
    if same:
        counts: dict[str, int] = {}
        for result in same:
            counts[result.entry.bug] = counts.get(result.entry.bug, 0) + 1
        listed = ", ".join(f"{bug}x{n}" for bug, n in sorted(counts.items()))
        print(f"   ({listed})")
        flaky = [r for r in same if r.flaky]
        if flaky:
            print(f"{'':>20}   {len(flaky)} needed a retry: {', '.join(r.entry.name for r in flaky)}")
    else:
        print()

    print(f"NO LONGER CRASHING {len(buckets['gone']):>5}", end="")
    print("   <- candidate fixes" if buckets["gone"] else "")
    for result in buckets["gone"]:
        print(f"    {result.entry.name}\n      was bug {result.entry.bug}: {result.entry.cls[:150]}")

    print(f"CLASS CHANGED      {len(buckets['changed']):>5}", end="")
    print("   <- inspect: partial fix, reworded error, or new bug" if buckets["changed"] else "")
    for result in buckets["changed"]:
        print(f"    {result.entry.name}  (bug {result.entry.bug})")
        print(f"      was: {result.entry.cls[:150]}")
        print(f"      now: {result.cls[:150]}")

    if buckets["error"]:
        print(f"REPLAY ERRORS      {len(buckets['error']):>5}")
        for result in buckets["error"]:
            print(f"    {result.entry.name}: {result.cls}")

    changed = len(buckets["gone"]) + len(buckets["changed"]) + len(buckets["error"])
    if changed:
        print(
            f"\n{changed} input(s) behave differently — review their bug column in"
            f" {suite.relative_to(REPO)}/{MANIFEST_NAME}"
        )
    return 1 if changed else 0


# --- entry point -------------------------------------------------------------------------------


def foreign_sources(replay: Path) -> list[str]:
    """Source files of the current `replay` binary that do not belong to this checkout.

    `CARGO_TARGET_DIR` here is a shared directory, so another lance-fuzz checkout can uplink *its*
    binary over `debug/replay` — and cargo then reports the build as fresh, because its own unit is
    fresh and the clobbered output is newer than the object it came from. That happened while this
    suite was first being built, and every class it recorded named source files this tree does not
    contain. Cargo's dep-info file lists the sources of whatever is currently uplinked, which is the
    cheap way to notice.
    """
    dep_info = replay.with_suffix(".d")
    if not dep_info.is_file():
        return []
    _, _, sources = dep_info.read_text().partition(": ")
    return [s for s in sources.split() if s.endswith(".rs") and not s.startswith(f"{REPO}/")]


def build_replay(force: bool) -> None:
    if force:  # cargo will not redo a "fresh" unit, so invalidate the crate it belongs to
        (REPO / "src" / "lib.rs").touch()
    completed = subprocess.run(["cargo", "build", "--bin", "replay"], cwd=REPO)
    if completed.returncode != 0:
        print("cargo build --bin replay failed", file=sys.stderr)
        raise SystemExit(1)


def require_replay(target: str, build: bool = True) -> Path:
    """Locate `replay`, and guarantee it was built from this source tree.

    A stale or foreign binary does not fail loudly — it produces perfectly plausible crash classes
    for the wrong code — so this verifies rather than assumes, and forces a rebuild if it has to.
    """
    if build and not os.environ.get("LANCE_FUZZ_REPLAY"):
        build_replay(force=False)
        if foreign_sources(find_replay()):
            print("replay binary was built from another checkout — rebuilding", file=sys.stderr)
            build_replay(force=True)
            intruders = foreign_sources(find_replay())
            if intruders:
                print(f"replay still built from foreign sources: {intruders[0]}", file=sys.stderr)
                print("another checkout shares this CARGO_TARGET_DIR", file=sys.stderr)
                raise SystemExit(1)

    replay = find_replay()
    if not replay.is_file():
        print(f"replay binary not found at {replay}", file=sys.stderr)
        print("build it with: cargo build --bin replay", file=sys.stderr)
        print("or point LANCE_FUZZ_REPLAY at it", file=sys.stderr)
        raise SystemExit(1)
    return replay


def main() -> int:
    default_jobs = max(1, (os.cpu_count() or 4) // 2)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("target", nargs="?", default="dataset_ops", help="fuzz target to check")
    parser.add_argument("names", nargs="*", help="check only suite inputs matching these substrings")
    parser.add_argument(
        "--import",
        dest="import_dirs",
        nargs="*",
        metavar="DIR",
        help="add artifacts to the suite instead of checking it (default dirs: artifacts/<target>*)",
    )
    parser.add_argument(
        "--per-class",
        type=int,
        default=3,
        help="inputs to keep per crash class (default: 3). More than one because a partial fix can "
        "kill one route to a bug and leave another, and because some crashes are flaky",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=1,
        help="re-replay a mismatching input this many times before believing it (default: 1)",
    )
    parser.add_argument("--jobs", type=int, default=default_jobs, help="parallel replays")
    parser.add_argument("--no-build", action="store_true", help="use the existing replay binary as-is")
    args = parser.parse_args()

    if args.import_dirs is not None:
        dirs = [Path(d) for d in args.import_dirs] or sorted(
            d for d in (REPO / "artifacts").glob(f"{args.target}*") if d.is_dir()
        )
        return do_import(args.target, dirs, args.per_class, args.jobs, bool(args.import_dirs), not args.no_build)
    return do_check(args.target, args.names, args.retries, args.jobs, not args.no_build)


if __name__ == "__main__":
    sys.exit(main())
