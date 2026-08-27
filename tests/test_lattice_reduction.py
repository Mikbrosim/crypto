"""Correctness tests for the lattice-reduction adapter."""

from __future__ import annotations

from pathlib import Path
import shutil
import stat
import tempfile
import unittest
from unittest.mock import patch

from sage.all import Matrix, ZZ

from rewrite.lattice_reduction import FlatterError, flatter


class FlatterTests(unittest.TestCase):
    """Verify that flatter returns a short basis for the same lattice."""

    @staticmethod
    def _small_basis():
        # This unimodular basis is deliberately poor: its first vector has
        # squared length 10,001, while the lattice has a unit-vector basis.
        return Matrix(ZZ, [[100, 1], [1, 0]])

    def assert_unit_lattice_reduction(self, original, reduced) -> None:
        """Assert that ``reduced`` is a unit basis equivalent to ``original``."""
        self.assertEqual(reduced.dimensions(), original.dimensions())

        change_of_basis = reduced * original.inverse()
        self.assertTrue(all(entry in ZZ for entry in change_of_basis.list()))
        self.assertEqual(abs(change_of_basis.det()), 1)

        squared_lengths = sorted(row.dot_product(row) for row in reduced.rows())
        self.assertEqual(squared_lengths, [1, 1])

    def test_sage_fallback_solves_small_reduction_problem(self) -> None:
        """A missing flatter executable should still produce a valid basis."""
        original = self._small_basis()

        with (
            patch("rewrite.lattice_reduction.shutil.which", return_value=None),
            self.assertLogs("rewrite.lattice_reduction", level="WARNING"),
        ):
            reduced = flatter(original)

        self.assert_unit_lattice_reduction(original, reduced)

    def test_external_flatter_protocol_returns_valid_reduction(self) -> None:
        """The adapter should send and parse flatter's matrix representation."""
        original = self._small_basis()

        with tempfile.TemporaryDirectory() as temporary_directory:
            executable = Path(temporary_directory) / "flatter"
            executable.write_text(
                "#!/bin/sh\n"
                "cat >/dev/null\n"
                "printf '[[1 0]\\n[0 1]]\\n'\n",
                encoding="utf-8",
            )
            executable.chmod(executable.stat().st_mode | stat.S_IXUSR)

            reduced = flatter(
                original,
                executable=str(executable),
                fallback_to_lll=False,
                timeout=5.0,
            )

        self.assert_unit_lattice_reduction(original, reduced)

    def test_strict_mode_reports_a_missing_executable(self) -> None:
        """Strict mode should report failures instead of silently using LLL."""
        with patch("rewrite.lattice_reduction.shutil.which", return_value=None):
            with self.assertRaisesRegex(FlatterError, "was not found"):
                flatter(self._small_basis(), fallback_to_lll=False)

    @unittest.skipUnless(shutil.which("flatter"), "real flatter is not installed")
    def test_real_flatter_solves_small_reduction_problem(self) -> None:
        """Exercise the real executable when the development host provides it."""
        original = self._small_basis()
        reduced = flatter(original, fallback_to_lll=False, timeout=5.0)
        self.assert_unit_lattice_reduction(original, reduced)


if __name__ == "__main__":
    unittest.main()
