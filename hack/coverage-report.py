#!/usr/bin/env python3
"""Render a Lance coverage report from the profdata `cargo fuzz coverage` produced.

Answers "which parts of Lance does the corpus never reach", which libFuzzer's edge counters
cannot. Kept separate from collection because collection builds and replays the whole corpus
(minutes) while rendering takes a second, so a report can be re-read or re-thresholded for free.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# Only these count towards the report: the Lance crates, taken from the pinned git checkout.
LANCE_MARKER = "checkouts/lance-"


class ReportError(RuntimeError):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "target",
        nargs="?",
        default="dataset_ops",
        help="fuzz target whose profdata to render (default: dataset_ops)",
    )
    parser.add_argument(
        "--min-dark",
        type=int,
        default=400,
        help="smallest fully-uncovered file to list, in lines (default: 400)",
    )
    parser.add_argument(
        "--collect",
        action="store_true",
        help="build with coverage and replay the corpus instead of reusing profdata",
    )
    return parser.parse_args()


def collect(target: str) -> None:
    """Build with coverage and replay every corpus unit to produce fresh profdata."""
    corpus = REPO / "corpus" / target
    if not corpus.is_dir():
        raise ReportError(f"no corpus for {target}: expected corpus/{target}")

    corpus = corpus.relative_to(REPO)
    print(f"collecting coverage over: {corpus}", flush=True)
    completed = subprocess.run(
        [
            "cargo",
            "+nightly",
            "fuzz",
            "coverage",
            "--fuzz-dir",
            ".",
            target,
            str(corpus),
        ],
        cwd=REPO,
    )
    if completed.returncode != 0:
        raise ReportError(f"cargo fuzz coverage failed with exit code {completed.returncode}")


def find_llvm_cov() -> Path:
    """Locate llvm-cov inside the nightly toolchain.

    Derived from the toolchain rather than hardcoded: `target-libdir` is
    `<sysroot>/lib/rustlib/<triple>/lib` and the LLVM binaries sit in the sibling `bin/`. This is
    also why `cargo-binutils` and `rustfilt` are not needed.
    """
    try:
        libdir = subprocess.run(
            ["rustc", "+nightly", "--print", "target-libdir"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ReportError(f"could not ask rustc +nightly for its target-libdir: {exc}") from exc

    llvm_cov = Path(libdir).parent / "bin" / "llvm-cov"
    if not llvm_cov.is_file():
        raise ReportError(
            f"llvm-cov not found at {llvm_cov}\ninstall it with: rustup +nightly component add llvm-tools-preview"
        )
    return llvm_cov


def find_coverage_binary(target: str) -> Path:
    """Find the coverage-instrumented binary.

    cargo-fuzz nests it as `target/<triple>/coverage/<triple>/release/<target>`, which is easy to
    get wrong by hand, so glob for it instead of assuming the layout.
    """
    matches = sorted(REPO.glob(f"target/*/coverage/*/release/{target}"))
    matches = [m for m in matches if m.is_file()]
    if not matches:
        raise ReportError(
            f"no coverage-instrumented binary for {target!r} under target/\nrun 'pixi run fuzz-coverage {target}' first"
        )
    return matches[0]


def crate_of(path: str) -> str | None:
    """Crate name for a file inside the Lance checkout.

    The name is the component after the *last* `rust` in the path. Matching the first is wrong: the
    dependency path starts `/<...>/rust/cargo/git/checkouts/lance-<hash>/<rev>/rust/<crate>/...`, so
    a naive match reports every Lance crate as "cargo".
    """
    parts = path.split("/")
    for i in range(len(parts) - 1, 0, -1):
        if parts[i] == "rust":
            return parts[i + 1] if i + 1 < len(parts) else None
    return None


def relative_to_rust(path: str) -> str:
    """Trim a checkout path down to `<crate>/src/...` for display."""
    parts = path.split("/")
    for i in range(len(parts) - 1, 0, -1):
        if parts[i] == "rust":
            return "/".join(parts[i + 1 :])
    return path


class Totals:
    def __init__(self) -> None:
        self.lines = 0
        self.covered = 0

    def add(self, lines: int, missed: int) -> None:
        self.lines += lines
        self.covered += lines - missed

    @property
    def percent(self) -> float:
        return self.covered / self.lines * 100 if self.lines else 0.0


def parse_report(text: str, min_dark: int) -> tuple[dict[str, Totals], Totals, list[tuple[int, str]]]:
    """Fold `llvm-cov report` output into per-crate totals and a fully-dark file list.

    Columns are: Filename Regions Missed Cover Functions Missed Executed Lines Missed Cover
    [Branches Missed Cover] — so total lines is field 7 and missed lines field 8, 0-indexed.
    """
    per_crate: dict[str, Totals] = {}
    overall = Totals()
    dark: list[tuple[int, str]] = []

    for line in text.splitlines():
        fields = line.split()
        if len(fields) < 10 or LANCE_MARKER not in fields[0]:
            continue
        crate = crate_of(fields[0])
        if crate is None:
            continue
        try:
            lines, missed = int(fields[7]), int(fields[8])
        except ValueError:
            continue

        per_crate.setdefault(crate, Totals()).add(lines, missed)
        overall.add(lines, missed)
        if missed == lines and lines >= min_dark:
            dark.append((lines, relative_to_rust(fields[0])))

    return per_crate, overall, dark


def main() -> int:
    args = parse_args()
    target = args.target

    if args.collect:
        try:
            collect(target)
        except ReportError as exc:
            print(exc, file=sys.stderr)
            return 1

    profdata = REPO / "coverage" / target / "coverage.profdata"
    if not profdata.is_file():
        print(
            f"no profdata at {profdata.relative_to(REPO)} — run 'pixi run fuzz-coverage {target}' first",
            file=sys.stderr,
        )
        return 1

    try:
        llvm_cov = find_llvm_cov()
        covbin = find_coverage_binary(target)
    except ReportError as exc:
        print(exc, file=sys.stderr)
        return 1

    completed = subprocess.run(
        [str(llvm_cov), "report", str(covbin), f"-instr-profile={profdata}"],
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        print(f"llvm-cov report failed:\n{completed.stderr}", file=sys.stderr)
        return 1

    per_file = REPO / "coverage" / target / "per-file.txt"
    per_file.write_text(completed.stdout)

    per_crate, overall, dark = parse_report(completed.stdout, args.min_dark)
    if not overall.lines:
        print(
            "no Lance files in the report — is the profdata stale or from another target?",
            file=sys.stderr,
        )
        return 1

    print(f"=== {target}: Lance line coverage from corpus ===")
    print(f"binary:   {covbin.relative_to(REPO)}")
    print(f"profdata: {profdata.relative_to(REPO)}")
    print()
    print(f"TOTAL {overall.lines} lines, {overall.covered} covered ({overall.percent:.1f}%)")
    print()
    print(f"{'crate':<26} {'lines':>9} {'covered':>9} {'cover%':>8}")
    for crate, totals in sorted(per_crate.items(), key=lambda kv: kv[1].percent):
        print(f"{crate:<26} {totals.lines:>9} {totals.covered:>9} {totals.percent:>7.1f}%")
    print()
    print(f"=== fully dark: 0% covered, >= {args.min_dark} lines ===")
    if dark:
        for lines, path in sorted(dark, reverse=True):
            print(f"{lines:>7}  {path}")
    else:
        print("(none)")
    print()
    print(f"full per-file report: {per_file.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
