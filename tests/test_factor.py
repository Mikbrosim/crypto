"""Deterministic correctness tests for structured factor recovery."""

from __future__ import annotations

import unittest

from rewrite.factor import (
    CandidateLimitError,
    factorize_1,
    factorize_2,
    factorize_3,
    factorize_4,
    naive_factor,
    powerset,
    prod_powerset,
    trial_division,
)


class StructuredFactorTests(unittest.TestCase):
    """Exercise every attack on small instances with exact factor oracles."""

    def assert_factorization(self, result, modulus, expected) -> None:
        self.assertIsNotNone(result)
        factor1, factor2 = result
        self.assertEqual(factor1 * factor2, modulus)
        self.assertEqual({factor1, factor2}, set(expected))

    def test_trial_division_reports_complete_and_partial_results(self) -> None:
        value = 2**3 * 3 * 101
        complete = trial_division(value)
        self.assertEqual(complete.factors, (2, 2, 2, 3, 101))
        self.assertEqual(complete.remainder, 1)
        self.assertTrue(complete.complete)
        self.assertEqual(complete.value, value)

        partial_value = 2 * 101 * 103
        partial = trial_division(partial_value, 10)
        self.assertEqual(partial.factors, (2,))
        self.assertEqual(partial.remainder, 101 * 103)
        self.assertFalse(partial.complete)
        self.assertEqual(partial.value, partial_value)
        self.assertEqual(naive_factor(partial_value, 10), [2])

    def test_powersets_preserve_subsets_but_deduplicate_products(self) -> None:
        self.assertEqual(
            list(powerset([1, 2, 3])),
            [
                (),
                (1,),
                (2,),
                (3,),
                (1, 2),
                (1, 3),
                (2, 3),
                (1, 2, 3),
            ],
        )
        self.assertEqual(
            list(prod_powerset([2, 2, 3])),
            [1, 2, 3, 4, 6, 12],
        )

    def test_separated_products_support_strict_and_carry_searches(self) -> None:
        strict_modulus = 11 * 17
        self.assert_factorization(
            factorize_1(strict_modulus, 3),
            strict_modulus,
            (11, 17),
        )

        carry_modulus = 53 * 67
        self.assertIsNone(factorize_1(carry_modulus, 4))
        self.assert_factorization(
            factorize_1(carry_modulus, 4, carry_bound=1),
            carry_modulus,
            (53, 67),
        )

        low_carry_modulus = 53 * 103
        self.assertIsNone(
            factorize_1(low_carry_modulus, 4, carry_bound=1)
        )
        self.assert_factorization(
            factorize_1(low_carry_modulus, 4, carry_bound=3),
            low_carry_modulus,
            (53, 103),
        )

        with self.assertRaises(CandidateLimitError) as context:
            factorize_1(strict_modulus, 3, max_candidates=1)
        self.assertEqual(context.exception.attack, "factorize_1")

    def test_partial_digit_recovery_in_multiple_bases(self) -> None:
        decimal_modulus = 137 * 149
        self.assert_factorization(
            factorize_2(
                decimal_modulus,
                "1?7",
                "?49",
                max_posib=1000,
            ),
            decimal_modulus,
            (137, 149),
        )

        hexadecimal_modulus = 0x35 * 0x43
        self.assert_factorization(
            factorize_2(
                hexadecimal_modulus,
                "3?",
                "?3",
                base=16,
                max_posib=1000,
            ),
            hexadecimal_modulus,
            (0x35, 0x43),
        )

        binary_modulus = 11 * 13
        self.assert_factorization(
            factorize_2(
                binary_modulus,
                "1?11",
                "11?1",
                base=2,
                max_posib=100,
            ),
            binary_modulus,
            (11, 13),
        )

    def test_digit_recovery_limit_does_not_silently_prune(self) -> None:
        with self.assertRaises(CandidateLimitError) as context:
            factorize_2(11 * 13, "??", "??", max_posib=1)
        self.assertEqual(context.exception.attack, "factorize_2")
        self.assertGreater(context.exception.candidates, context.exception.limit)

    def test_biased_binary_recovery_has_exact_and_beam_modes(self) -> None:
        many_ones_modulus = 23 * 31
        self.assert_factorization(
            factorize_3(
                many_ones_modulus,
                True,
                max_posib=32,
                strategy="exact",
            ),
            many_ones_modulus,
            (23, 31),
        )
        self.assert_factorization(
            factorize_3(
                many_ones_modulus,
                True,
                max_posib=1,
                strategy="beam",
            ),
            many_ones_modulus,
            (23, 31),
        )

        many_zeros_modulus = 17 * 19
        self.assert_factorization(
            factorize_3(
                many_zeros_modulus,
                False,
                max_posib=8,
                strategy="exact",
            ),
            many_zeros_modulus,
            (17, 19),
        )

        with self.assertRaises(CandidateLimitError) as context:
            factorize_3(
                many_ones_modulus,
                True,
                max_posib=1,
                strategy="exact",
            )
        self.assertEqual(context.exception.attack, "factorize_3")

    def test_partial_rsa_components_recover_and_verify_factors(self) -> None:
        p = 11
        q = 13
        modulus = p * q
        exponent = 7
        totient = (p - 1) * (q - 1)
        d = pow(exponent, -1, totient)
        dp = pow(exponent, -1, p - 1)
        dq = pow(exponent, -1, q - 1)
        k = (exponent * d - 1) // totient
        kp = (exponent * dp - 1) // (p - 1)
        kq = (exponent * dq - 1) // (q - 1)

        result = factorize_4(
            exponent,
            n=modulus,
            d=d,
            d_MASK=0b1111,
            kd=k,
            dp=dp,
            dp_MASK=0b11,
            kdp=kp,
            p=p,
            p_MASK=0b0101,
            dq=dq,
            dq_MASK=0b111,
            kdq=kq,
            q=q,
            q_MASK=0b1001,
            max_candidates=100,
        )
        self.assertEqual(result, (p, q))
        self.assertEqual(result[0] * result[1], modulus)
        self.assertEqual(exponent * d, 1 + k * totient)
        self.assertEqual(exponent * dp, 1 + kp * (p - 1))
        self.assertEqual(exponent * dq, 1 + kq * (q - 1))

        lambda_divisor = 2
        reduced_totient = totient // lambda_divisor
        lambda_d = pow(exponent, -1, reduced_totient)
        lambda_k = (exponent * lambda_d - 1) // reduced_totient
        self.assertEqual(
            factorize_4(
                exponent,
                n=modulus,
                d=lambda_d,
                d_MASK=(1 << lambda_d.bit_length()) - 1,
                kd=lambda_k,
                p=p,
                p_MASK=0b1111,
                q=q,
                q_MASK=0b1111,
                f=lambda_divisor,
            ),
            (p, q),
        )

    def test_partial_rsa_limit_and_contradictory_leakage(self) -> None:
        with self.assertRaises(CandidateLimitError) as context:
            factorize_4(7, n=11 * 13, max_candidates=1)
        self.assertEqual(context.exception.attack, "factorize_4")

        self.assertIsNone(
            factorize_4(
                7,
                n=11 * 13,
                p=10,
                p_MASK=0b1111,
                max_candidates=100,
            )
        )

    def test_invalid_problem_descriptions_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "positive"):
            trial_division(0)
        with self.assertRaisesRegex(ValueError, "positive"):
            list(prod_powerset([2, 0, 3]))
        with self.assertRaisesRegex(ValueError, "base"):
            factorize_2(143, "1011", "1101", base=1)
        with self.assertRaisesRegex(ValueError, "invalid for base"):
            factorize_2(143, "1021", "1101", base=2)
        with self.assertRaisesRegex(TypeError, "many_ones"):
            factorize_3(143, 1)
        with self.assertRaisesRegex(ValueError, "strategy"):
            factorize_3(143, True, strategy="guess")
        with self.assertRaisesRegex(ValueError, "odd"):
            factorize_4(4, n=143)
        with self.assertRaisesRegex(ValueError, "kd must be positive"):
            factorize_4(7, n=143, d=3, d_MASK=1, kd=0)


if __name__ == "__main__":
    unittest.main()
