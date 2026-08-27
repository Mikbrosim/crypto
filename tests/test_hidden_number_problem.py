"""Correctness tests for HNP, EHNP, and Babai CVP."""

from __future__ import annotations

import unittest

from sage.all import Matrix, QQ, ZZ, vector

from rewrite.hidden_number_problem import babai_cvp, ehnp, hnp


class HiddenNumberProblemTests(unittest.TestCase):
    """Use small instances whose complete solution sets are easy to verify."""

    MODULUS = 1009
    SECRET = 321
    MULTIPLIERS = [17, 83, 211, 407]

    def test_babai_returns_a_nearby_lattice_point(self) -> None:
        basis = Matrix(ZZ, [[2, 0], [0, 2]])
        target = vector(QQ, [QQ(3) / 2, QQ(7) / 5])

        point = babai_cvp(basis, target)

        self.assertEqual(point, vector(QQ, [2, 2]))
        self.assertEqual(point * basis.inverse(), vector(QQ, [1, 1]))

    def test_exact_hnp_recovers_unsigned_residuals(self) -> None:
        residuals = [3, 11, 7, 13]
        offsets = [
            (multiplier * self.SECRET - residual) % self.MODULUS
            for multiplier, residual in zip(
                self.MULTIPLIERS,
                residuals,
                strict=True,
            )
        ]

        recovered = hnp(
            self.MODULUS,
            self.MULTIPLIERS,
            offsets,
            16,
            method="exact",
        )
        self.assertEqual(recovered, self.SECRET)

    def test_exact_hnp_supports_signed_residuals(self) -> None:
        residuals = [-3, 7, -11, 4]
        offsets = [
            (multiplier * self.SECRET - residual) % self.MODULUS
            for multiplier, residual in zip(
                self.MULTIPLIERS,
                residuals,
                strict=True,
            )
        ]

        recovered = hnp(
            self.MODULUS,
            self.MULTIPLIERS,
            offsets,
            16,
            method="exact",
            signed=True,
        )
        self.assertEqual(recovered, self.SECRET)

    def test_kannan_hnp_recovers_the_same_unique_secret(self) -> None:
        residuals = [3, 11, 7, 13]
        offsets = [
            (multiplier * self.SECRET - residual) % self.MODULUS
            for multiplier, residual in zip(
                self.MULTIPLIERS,
                residuals,
                strict=True,
            )
        ]

        recovered = hnp(
            self.MODULUS,
            self.MULTIPLIERS,
            offsets,
            16,
            method="kannan",
        )
        self.assertEqual(recovered, self.SECRET)

    def test_exact_ehnp_recovers_fragmented_secret(self) -> None:
        recovered = ehnp(
            xbar=1,
            p=101,
            Pi=[1, 4],
            Nu=[2, 2],
            Alpha=[7, 13, 29],
            Rho=[[5], [11], [17]],
            Mu=[[2], [2], [2]],
            Beta=[61, 82, 37],
            method="exact",
        )

        self.assertEqual(recovered, 21)

    def test_ehnp_babai_and_kannan_decode_a_lattice_target(self) -> None:
        # The hidden fragments and multipliers are all interval midpoints, so
        # the valid lattice point is exactly the constructed CVP target.
        for method in ("babai", "kannan"):
            with self.subTest(method=method):
                recovered = ehnp(
                    xbar=4,
                    p=101,
                    Pi=[0],
                    Nu=[2],
                    Alpha=[7, 13],
                    Rho=[[5], [11]],
                    Mu=[[2], [2]],
                    Beta=[52, 100],
                    delta=QQ(1) / 1000,
                    method=method,
                )
                self.assertEqual(recovered, 6)


if __name__ == "__main__":
    unittest.main()
