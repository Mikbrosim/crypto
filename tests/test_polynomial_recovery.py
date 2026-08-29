"""Small correctness tests for modular polynomial and modulus recovery."""

from __future__ import annotations

import unittest

from rewrite.polynomial_recovery import (
    AmbiguousModulusError,
    PolynomialRecoveryError,
    polynomial_relation_multiple,
    recover_modular_polynomial,
    recover_modulus_multiple,
    recover_modulus_of_polynomial_fast,
    recover_modulus_of_polynomial_slow,
    recover_prime_modulus,
)


def evaluate(coefficients, x_value, modulus):
    """Evaluate ascending coefficients with Horner's method."""

    result = 0
    for coefficient in reversed(coefficients):
        result = (result * x_value + coefficient) % modulus
    return result


def make_samples(coefficients, modulus, x_values):
    return tuple(
        (x_value, evaluate(coefficients, x_value, modulus))
        for x_value in x_values
    )


class PolynomialRelationTests(unittest.TestCase):
    """Check both exact Vandermonde-kernel relation strategies."""

    def test_finite_difference_relation_is_an_exact_modulus_multiple(self) -> None:
        samples = make_samples((7, 4, 9), 101, (2, 5, 8, 11))

        relation = polynomial_relation_multiple(
            samples,
            method="finite_difference",
        )

        self.assertEqual(relation, 101)
        self.assertEqual(relation % 101, 0)

    def test_barycentric_relation_handles_arbitrary_distinct_inputs(self) -> None:
        samples = make_samples((7, 4, 9), 101, (1, 3, 4, 8))

        relation = polynomial_relation_multiple(
            samples,
            method="barycentric",
        )

        self.assertEqual(relation, 17 * 101)
        self.assertEqual(relation % 101, 0)

    def test_auto_selects_the_applicable_relation_and_validates_inputs(self) -> None:
        equally_spaced = make_samples((7, 4, 9), 101, (2, 5, 8, 11))
        self.assertEqual(
            polynomial_relation_multiple(equally_spaced),
            polynomial_relation_multiple(
                equally_spaced,
                method="finite_difference",
            ),
        )

        arbitrary = make_samples((7, 4, 9), 101, (1, 3, 4, 8))
        with self.assertRaises(ValueError):
            polynomial_relation_multiple(
                arbitrary,
                method="finite_difference",
            )
        with self.assertRaises(ValueError):
            polynomial_relation_multiple(((1, 2), (1, 2)))


class ModulusRecoveryTests(unittest.TestCase):
    """Recover a small prime while preserving uncertainty information."""

    COEFFICIENTS = (7, 4, 9)
    MODULUS = 101
    SAMPLES = make_samples(COEFFICIENTS, MODULUS, (1, 3, 4, 8, 10))

    def test_extra_relations_remove_an_accidental_cofactor(self) -> None:
        self.assertEqual(
            recover_modulus_multiple(
                self.SAMPLES,
                degree=2,
                max_relations=1,
            ),
            17 * self.MODULUS,
        )
        self.assertEqual(
            recover_modulus_multiple(self.SAMPLES, degree=2),
            self.MODULUS,
        )

    def test_recovers_a_unique_prime_and_the_polynomial(self) -> None:
        recovered_modulus = recover_prime_modulus(self.SAMPLES, degree=2)
        coefficients = recover_modular_polynomial(
            self.SAMPLES,
            recovered_modulus,
            degree=2,
        )

        self.assertEqual(recovered_modulus, self.MODULUS)
        self.assertEqual(coefficients, list(self.COEFFICIENTS))
        for x_value, y_value in self.SAMPLES:
            self.assertEqual(
                evaluate(coefficients, x_value, recovered_modulus),
                y_value,
            )

    def test_ambiguous_prime_factors_are_reported_instead_of_guessed(self) -> None:
        # The values fit 3*x^2 + 3*x modulo 5 and 4*x^2 + 5*x + 1
        # modulo 7.  With no bound, both prime explanations are valid.
        samples = (
            (20, 0),
            (28, 1),
            (54, 0),
            (127, 3),
            (151, 1),
        )

        self.assertEqual(recover_modulus_multiple(samples, degree=2), 35)
        with self.assertRaises(AmbiguousModulusError) as context:
            recover_prime_modulus(samples, degree=2)

        self.assertEqual(context.exception.multiple, 35)
        self.assertEqual(context.exception.candidates, (5, 7))
        self.assertEqual(
            recover_prime_modulus(samples, degree=2, max_modulus=5),
            5,
        )

    def test_zero_and_inconsistent_relations_fail_clearly(self) -> None:
        no_wrap = make_samples((1, 1), 101, (0, 1, 2, 3))
        with self.assertRaisesRegex(PolynomialRecoveryError, "all.*zero"):
            recover_modulus_multiple(no_wrap, degree=1)

        corrupted = list(self.SAMPLES)
        corrupted[-1] = (corrupted[-1][0], corrupted[-1][1] + 1)
        with self.assertRaisesRegex(PolynomialRecoveryError, "GCD 1"):
            recover_modulus_multiple(corrupted, degree=2)

    def test_prime_recovery_requires_two_informative_relations(self) -> None:
        one_window = self.SAMPLES[:4]
        self.assertEqual(
            recover_modulus_multiple(one_window, degree=2),
            17 * self.MODULUS,
        )
        with self.assertRaisesRegex(PolynomialRecoveryError, "two nonzero"):
            recover_prime_modulus(one_window, degree=2)

    def test_legacy_fast_and_slow_entry_points_remain_usable(self) -> None:
        self.assertEqual(
            recover_modulus_of_polynomial_fast(self.SAMPLES, degree=2),
            self.MODULUS,
        )
        self.assertEqual(
            recover_modulus_of_polynomial_fast(
                self.SAMPLES,
                degree=2,
                extra_gcds=False,
            ),
            self.MODULUS,
        )
        self.assertEqual(
            recover_modulus_of_polynomial_slow(self.SAMPLES),
            self.MODULUS,
        )


class ModularInterpolationTests(unittest.TestCase):
    """Reconstruct and verify polynomials over prime and composite moduli."""

    def test_composite_modulus_uses_integer_residue_ring_semantics(self) -> None:
        coefficients = (12, 7, 4)
        samples = make_samples(coefficients, 35, (0, 1, 2, 3))

        recovered = recover_modular_polynomial(samples, 35, degree=2)

        self.assertEqual(recovered, list(coefficients))
        for x_value, y_value in samples:
            self.assertEqual(evaluate(recovered, x_value, 35), y_value)

    def test_redundant_sample_detects_an_incorrect_degree_or_corruption(self) -> None:
        samples = make_samples((7, 4, 9), 101, (1, 3, 4, 8))
        with self.assertRaisesRegex(PolynomialRecoveryError, "matches every"):
            recover_modular_polynomial(samples, 101, degree=1)

        corrupted = list(samples)
        corrupted[-1] = (corrupted[-1][0], corrupted[-1][1] + 1)
        with self.assertRaisesRegex(PolynomialRecoveryError, "matches every"):
            recover_modular_polynomial(corrupted, 101, degree=2)

    def test_modular_duplicates_are_coalesced_or_rejected(self) -> None:
        consistent = ((0, 1), (15, 1), (1, 2))
        self.assertEqual(
            recover_modular_polynomial(consistent, 15, degree=1),
            [1, 1],
        )

        conflicting = ((0, 1), (15, 2), (1, 2))
        with self.assertRaisesRegex(PolynomialRecoveryError, "conflicting"):
            recover_modular_polynomial(conflicting, 15, degree=1)

    def test_composite_interpolation_rejects_nonunit_differences(self) -> None:
        with self.assertRaisesRegex(PolynomialRecoveryError, "invertible"):
            recover_modular_polynomial(
                ((0, 1), (3, 4)),
                15,
                degree=1,
            )

    def test_invalid_sizes_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            recover_modulus_multiple(((0, 1), (1, 2)), degree=2)
        with self.assertRaises(ValueError):
            recover_modular_polynomial(((0, 1),), 101, degree=1)
        with self.assertRaises(ValueError):
            recover_modular_polynomial(((0, 1),), 1)


if __name__ == "__main__":
    unittest.main()
