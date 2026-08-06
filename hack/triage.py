#!/usr/bin/env python3
"""Shared crash-signature logic: replay an artifact, reduce its panic to a stable class.

Used by both `triage-daemon.py` (is this crash new?) and `crash-check.py` (does this crash still
happen, and is it still the *same* crash?). One definition of "same crash" for both, because the two
answers have to agree: a class that shifts meaning between them would make the regression suite
report phantom changes.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
ARTIFACT_PREFIXES = ("crash-", "oom-", "timeout-")

# Sentinels returned by panic_block() in place of a panic. Callers distinguish them from a real
# panic, so they are values rather than exceptions.
NO_PANIC = "NO PANIC REPRODUCED"
TIMED_OUT = "REPLAY TIMED OUT"

# --- signature normalisation -------------------------------------------------------------------
#
# Four things vary between artifacts that share one underlying defect. Each was found the hard way
# by a filed bug reporting itself as "new" on every recurrence, so all four must be erased:
#
#   1. generated op payloads      `keys: [1, 1]` vs `keys: [1, 1, 1, 1]`
#   2. per-run data file names    `…/0010abcd….lance`
#   3. dependency checkout paths  `/…/checkouts/lance-<hash>/<rev>/…`
#   4. where it was noticed       `step 7:` vs `end:`
#
# Panic *line numbers* are deliberately excluded from the class too: editing the harness shifts
# every line and would re-report every known family as new. Upgrading Lance shifts them as well,
# which is what makes a class comparable across Lance versions — see crash-check.py.

PANIC_LOCATION = re.compile(r"panicked at (\S+?):\d+:\d+:\s*")
BRACE_GROUP = re.compile(r"\{[^{}]*\}")
BRACKET_GROUP = re.compile(r"\[[^\[\]]*\]")
LANCE_FILE = re.compile(r"[\w/.-]*/[0-9a-f]{16,}\w*\.lance")
CHECKOUT_PATH = re.compile(r"/[\w./-]*/(?:lance-[0-9a-f]+|index\.crates\.io-[0-9a-f]+)/")
NEGATIVE = re.compile(r"-(\d)")
CONTEXT = re.compile(r"(^|failed: )(?:step \d+|end)(?:, candidate \d+)?: ")
PARENTHESISED_NUM = re.compile(r"\(\d+\)")
DIGITS = re.compile(r"\d+")
SPACES = re.compile(r"\s+")


def collapse_groups(text: str) -> str:
    """Collapse innermost {...} and [...] repeatedly, so nested payloads reduce to a marker."""
    for _ in range(6):  # far more nesting than the harness ever prints
        collapsed = BRACKET_GROUP.sub("[..]", BRACE_GROUP.sub("{..}", text))
        if collapsed == text:
            break
        text = collapsed
    return text


def classify(raw: str) -> str:
    """Reduce a panic block to a stable signature class: source file plus normalised message."""
    location = PANIC_LOCATION.search(raw)
    site = Path(location.group(1)).name if location else ""
    message = raw[location.end() :] if location else raw

    message = collapse_groups(message)
    message = LANCE_FILE.sub("FILE.lance", message)
    message = CHECKOUT_PATH.sub("PATH/", message)
    message = NEGATIVE.sub(r"\1", message)
    message = CONTEXT.sub(r"\1CTX: ", message)
    message = PARENTHESISED_NUM.sub("", message)
    message = DIGITS.sub("N", message)
    message = SPACES.sub(" ", message).strip()[:200]
    return f"{site} | {message}"


# --- replay ------------------------------------------------------------------------------------


def find_replay() -> Path:
    """Locate the stable `replay` binary.

    Not hardcoded: this repository is often built with `CARGO_TARGET_DIR` pointing outside the
    working tree, so ask cargo where its target directory is and fall back to the default layout.
    `LANCE_FUZZ_REPLAY` overrides everything.
    """
    override = os.environ.get("LANCE_FUZZ_REPLAY")
    if override:
        return Path(override)

    target_dir: Path | None = None
    env_target = os.environ.get("CARGO_TARGET_DIR")
    if env_target:
        target_dir = Path(env_target)
    else:
        try:
            metadata = subprocess.run(
                ["cargo", "metadata", "--no-deps", "--format-version", "1"],
                cwd=REPO,
                capture_output=True,
                text=True,
                check=True,
            )
            target_dir = Path(json.loads(metadata.stdout)["target_directory"])
        except (OSError, subprocess.CalledProcessError, ValueError, KeyError):
            target_dir = REPO / "target"

    return target_dir / "debug" / "replay"


def panic_block(replay: Path, artifact: Path, target: str = "dataset_ops") -> str:
    """Replay one artifact and return its panic line plus the two following lines."""
    try:
        completed = subprocess.run(
            ["nice", "-n", "10", str(replay), str(artifact)],
            capture_output=True,
            text=True,
            timeout=180,
        )
    except subprocess.TimeoutExpired:
        return TIMED_OUT
    except OSError as exc:
        return f"REPLAY FAILED TO START: {exc}"

    output = (completed.stdout + completed.stderr).splitlines()
    for i, line in enumerate(output):
        if "panicked at" in line:
            return " ".join(output[i : i + 3])
    return NO_PANIC
