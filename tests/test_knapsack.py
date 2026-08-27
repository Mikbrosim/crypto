"""Correctness tests for exact and lattice subset-sum strategies."""

from __future__ import annotations

import unittest

from rewrite.knapsack import subset_sum


class SubsetSumTests(unittest.TestCase):
    """Verify returned bits against small independently checkable systems."""

    @staticmethod
    def assert_solution(weights, targets, solution, modulus=None) -> None:
        """Check the bit domain and all exact or modular target equations."""
        if isinstance(weights[0], int):
            rows = [weights]
            expected_targets = [targets]
        else:
            rows = weights
            expected_targets = targets

        if solution is None:
            raise AssertionError("subset_sum returned no solution")
        if any(bit not in (0, 1) for bit in solution):
            raise AssertionError(f"non-binary solution: {solution}")
        for row, target in zip(rows, expected_targets, strict=True):
            total = sum(bit * weight for bit, weight in zip(solution, row, strict=True))
            if modulus is None:
                if total != target:
                    raise AssertionError(f"{total} != {target}")
            elif (total - target) % modulus:
                raise AssertionError(f"{total} != {target} modulo {modulus}")

    def test_exact_solver_handles_non_modular_single_sum(self) -> None:
        weights = [2, 5, 11, 23]
        target = 28
        solution = subset_sum(weights, target, method="exact")

        self.assertEqual(solution, [0, 1, 0, 1])
        self.assert_solution(weights, target, solution)

    def test_exact_solver_handles_multiple_modular_sums(self) -> None:
        weights = [[2, 5, 11, 23], [7, 3, 13, 17]]
        expected = [0, 1, 0, 1]
        modulus = 31
        targets = [
            sum(bit * value for bit, value in zip(expected, row, strict=True))
            % modulus
            for row in weights
        ]

        solution = subset_sum(
            weights,
            targets,
            modulus=modulus,
            method="exact",
        )
        self.assert_solution(weights, targets, solution, modulus)

    def test_direct_embedding_solves_a_small_low_density_sum(self) -> None:
        weights = [3, 17, 41, 109, 257, 631]
        expected = [1, 0, 1, 0, 0, 1]
        target = sum(
            bit * weight for bit, weight in zip(expected, weights, strict=True)
        )

        solution = subset_sum(weights, target, method="direct")
        self.assert_solution(weights, target, solution)

    def test_kernel_embedding_solves_multiple_exact_sums(self) -> None:
        weights = [
            [2, 5, 11, 23],
            [7, 3, 13, 17],
            [19, 29, 31, 37],
        ]
        expected = [0, 1, 0, 1]
        targets = [
            sum(bit * value for bit, value in zip(expected, row, strict=True))
            for row in weights
        ]

        solution = subset_sum(weights, targets, method="kernel")
        self.assert_solution(weights, targets, solution)

    def test_auto_uses_exact_solver_for_small_instances(self) -> None:
        weights = [2, 5, 11, 23]
        solution = subset_sum(weights, 28)
        self.assertEqual(solution, [0, 1, 0, 1])


if __name__ == "__main__":
    unittest.main()
