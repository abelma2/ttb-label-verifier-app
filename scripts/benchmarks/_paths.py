"""Repo-root / sys.path resolution for the benchmark scripts.

Walks UP from this file until it finds the repo root -- the directory that holds the project's
`config.py` and `extraction.py` (the sentinel) -- instead of hard-coding a count of
`os.path.dirname()` calls. That makes the benchmark scripts move-invariant: drop them a directory
deeper (as happened when `scripts/` -> `scripts/benchmarks/`) and ROOT still resolves, with no
per-file edits and no `ModuleNotFoundError` from a stale sys.path depth.

Usage in a benchmark script (replaces the old hand-rolled `ROOT = dirname(dirname(...))` +
`sys.path.insert(...)` block):

    import _paths
    _paths.ensure_paths()          # ROOT, scripts/, and this dir onto sys.path
    ROOT = _paths.ROOT             # for os.path.join(ROOT, "test_labels" | "output" | ...)

`_paths` is itself a sibling module: when a script runs as `python scripts/benchmarks/foo.py`,
Python puts `scripts/benchmarks/` on sys.path[0], so `import _paths` resolves with no setup.
"""
import os
import sys

# The repo root is identified by these files living side-by-side (not by directory depth).
_SENTINELS = ("config.py", "extraction.py")


def _find_root(start):
    """Walk upward from `start` until a directory contains all _SENTINELS; that is the repo root."""
    d = os.path.abspath(start)
    while True:
        if all(os.path.exists(os.path.join(d, s)) for s in _SENTINELS):
            return d
        parent = os.path.dirname(d)
        if parent == d:   # reached the filesystem root without finding the sentinels
            raise RuntimeError(
                f"repo root not found: none of {_SENTINELS} while walking up from {start!r}")
        d = parent


BENCHMARKS_DIR = os.path.dirname(os.path.abspath(__file__))   # scripts/benchmarks/
ROOT = _find_root(BENCHMARKS_DIR)                             # repo root (holds config.py/extraction.py)
SCRIPTS_DIR = os.path.join(ROOT, "scripts")                  # where smoke_test.py / generate_adversarial.py live


def ensure_paths():
    """Put the three import roots on sys.path so a benchmark script's imports resolve wherever it
    sits: ROOT (for `extraction`/`config`), SCRIPTS_DIR (for `smoke_test`), and BENCHMARKS_DIR
    (for sibling benchmark modules like `model_benchmark`)."""
    for p in (BENCHMARKS_DIR, SCRIPTS_DIR, ROOT):
        if p not in sys.path:
            sys.path.insert(0, p)
