"""Correctness tests for the rewritten LWE recovery helpers."""

from __future__ import annotations

import unittest

from rewrite.lwe import center, recover_low_bits, recover_secret


class LWERecoveryTests(unittest.TestCase):
    """Recover a small deterministic secret and its unique closest error."""

    MODULUS = 101
    MATRIX = [
        [2, 5],
        [7, 3],
        [11, 13],
        [17, 19],
        [23, 29],
    ]
    SECRET = [3, -2]
    ERROR = [1, -1, 2, 0, -2]
    TARGET = [98, 14, 9, 13, 9]

    def assert_lwe_solution(self, secret, error) -> None:
        """Verify every modular LWE equation and declared bound."""
        self.assertEqual(secret, self.SECRET)
        self.assertEqual(error, self.ERROR)
        for row, target, error_value in zip(
            self.MATRIX,
            self.TARGET,
            error,
            strict=True,
        ):
            self.assertEqual(
                (
                    sum(
                        coefficient * value
                        for coefficient, value in zip(
                            row,
                            secret,
                            strict=True,
                        )
                    )
                    + error_value
                    - target
                )
                % self.MODULUS,
                0,
            )
            self.assertLessEqual(abs(error_value), 2)

    def test_center_returns_canonical_residues(self) -> None:
        self.assertEqual(center(100, 101), -1)
        self.assertEqual(center(51, 101), -50)
        self.assertEqual(center(5, 10), 5)
        self.assertEqual(center(6, 10), -4)

    def test_error_lattice_recovers_low_bits_with_babai_and_kannan(self) -> None:
        for method in ("babai", "kannan"):
            with self.subTest(method=method):
                error = recover_low_bits(
                    self.MATRIX,
                    self.TARGET,
                    self.MODULUS,
                    method=method,
                    error_bound=2,
                )
                self.assertEqual(error, self.ERROR)

    def test_known_error_uses_exact_smith_recovery(self) -> None:
        secret = recover_secret(
            self.MATRIX,
            self.TARGET,
            self.MODULUS,
            e=self.ERROR,
        )
        self.assert_lwe_solution(secret, self.ERROR)

    def test_augmented_kannan_recovers_secret_and_error_together(self) -> None:
        secret = recover_secret(
            self.MATRIX,
            self.TARGET,
            self.MODULUS,
            s_bits=3,
            formulation="augmented",
            method="kannan",
            error_bound=2,
        )
        error = recover_low_bits(
            self.MATRIX,
            self.TARGET,
            self.MODULUS,
            s_bits=3,
            formulation="augmented",
            method="kannan",
            error_bound=2,
        )
        self.assert_lwe_solution(secret, error)

    def test_non_modular_known_error_system(self) -> None:
        matrix = [[2, 1], [1, -1], [3, 2]]
        error = [1, 0, -1]
        target = [6, 7, 5]

        secret = recover_secret(matrix, target, e=error)

        self.assertEqual(secret, [4, -3])
        for row, target_value, error_value in zip(
            matrix,
            target,
            error,
            strict=True,
        ):
            self.assertEqual(
                sum(
                    coefficient * value
                    for coefficient, value in zip(row, secret, strict=True)
                )
                + error_value,
                target_value,
            )

    def test_incorrect_error_bound_rejects_the_candidate(self) -> None:
        self.assertIsNone(
            recover_low_bits(
                self.MATRIX,
                self.TARGET,
                self.MODULUS,
                method="kannan",
                error_bound=1,
            )
        )


if __name__ == "__main__":
    unittest.main()
