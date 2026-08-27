"""Discover and run correctness tests for the rewritten modules."""

from __future__ import annotations

from pathlib import Path
import unittest


_REWRITE_DIRECTORY = Path(__file__).resolve().parent
_PROJECT_DIRECTORY = _REWRITE_DIRECTORY.parent
_TEST_DIRECTORY = _REWRITE_DIRECTORY / "tests"


def main() -> int:
    """Run all rewrite tests and return a process-compatible status code."""
    suite = unittest.defaultTestLoader.discover(
        start_dir=str(_TEST_DIRECTORY),
        pattern="test_*.py",
        top_level_dir=str(_PROJECT_DIRECTORY),
    )
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
