"""Correctness tests for approximate-common-divisor recovery."""

from __future__ import annotations

import unittest

from rewrite.agcd import agcd, recover_agcd


class ApproximateCommonDivisorTests(unittest.TestCase):
    """Solve a small deterministic AGCD instance in each supported mode."""

    DIVISOR = 1009
    QUOTIENTS = (3001, 3253, 3527, 4001)
    ERRORS = (3, -2, 1, -3)
    ERROR_BITS = 3
    SAMPLES = (3028012, 3282275, 3558744, 4037006)

    def assert_correct_solution(self, result) -> None:
        """Verify the recovered values reconstruct every public sample."""
        self.assertEqual(result.divisor, self.DIVISOR)
        self.assertEqual(result.quotients, self.QUOTIENTS)
        self.assertEqual(result.errors, self.ERRORS)

        error_bound = 1 << self.ERROR_BITS
        for sample, quotient, error in zip(
            self.SAMPLES,
            result.quotients,
            result.errors,
            strict=True,
        ):
            self.assertEqual(sample, result.divisor * quotient + error)
            self.assertLess(abs(error), error_bound)

    def test_recovers_full_solution_with_rounded_lattice(self) -> None:
        result = recover_agcd(self.SAMPLES, self.ERROR_BITS)
        self.assert_correct_solution(result)

    def test_compatibility_wrapper_returns_quotient_vector(self) -> None:
        quotients = agcd(self.SAMPLES, self.ERROR_BITS)
        self.assertEqual(tuple(int(entry) for entry in quotients), self.QUOTIENTS)

    def test_strongest_effort_recovers_full_solution(self) -> None:
        result = recover_agcd(
            self.SAMPLES,
            self.ERROR_BITS,
            effort=len(self.SAMPLES),
        )
        self.assert_correct_solution(result)

    def test_original_scaled_lattice_remains_available(self) -> None:
        result = recover_agcd(
            self.SAMPLES,
            self.ERROR_BITS,
            rounded=False,
        )
        self.assert_correct_solution(result)

    def test_custom_lattice_reduction_callback_is_supported(self) -> None:
        reduction_calls = []

        def reduce_with_lll(basis):
            reduction_calls.append(basis.dimensions())
            return basis.LLL()

        result = recover_agcd(
            self.SAMPLES,
            self.ERROR_BITS,
            lattice_reduction=reduce_with_lll,
        )

        self.assertEqual(reduction_calls, [(4, 5)])
        self.assert_correct_solution(result)


if __name__ == "__main__":
    unittest.main()
