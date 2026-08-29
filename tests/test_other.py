"""Small correctness tests for every rewritten miscellaneous helper."""

from __future__ import annotations

import unittest

from sage.all import GF, PolynomialRing, QQ, ZZ, var, vector

from rewrite.other import (
    Polynomial,
    construct_GF2_matrix,
    crt,
    ideal_groebner,
    legendre,
    modular_square_roots,
    pack_words,
    pohlig_hellman,
    recover_modular_polynomial,
    recover_modulus_of_polynomial_fast,
    recover_modulus_of_polynomial_slow,
    tonelli_shanks,
    unpack_words,
    wrapper_maker,
)


class NumberTheoryTests(unittest.TestCase):
    """Check exact congruence, square-root, and discrete-log recovery."""

    def test_legendre_uses_the_conventional_three_values(self) -> None:
        self.assertEqual(legendre(3, 13), 1)
        self.assertEqual(legendre(2, 13), -1)
        self.assertEqual(legendre(0, 13), 0)
        with self.assertRaises(ValueError):
            legendre(1, 15)

    def test_crt_consumes_generators_and_supports_shared_factors(self) -> None:
        residues = (value for value in (2, 3, 2))
        moduli = (value for value in (3, 5, 7))
        self.assertEqual(crt(residues, moduli), 23)

        recovered = crt((2, 6), (4, 8))
        self.assertEqual(recovered, 6)
        self.assertEqual(recovered % 4, 2)
        self.assertEqual(recovered % 8, 6)
        with self.assertRaisesRegex(ValueError, "inconsistent"):
            crt((1, 2), (4, 6))

    def test_tonelli_shanks_returns_every_distinct_prime_root(self) -> None:
        self.assertEqual(tonelli_shanks(10, 13), (6, 7))
        self.assertEqual(tonelli_shanks(0, 13), (0,))
        self.assertEqual(tonelli_shanks(7, 2), (1,))
        self.assertEqual(tonelli_shanks(2, 13), ())
        with self.assertRaises(ValueError):
            tonelli_shanks(4, 15)

    def test_factored_composite_square_roots_are_complete_and_verified(self) -> None:
        expected = tuple(value for value in range(77) if value * value % 77 == 64)
        roots = modular_square_roots(64, (7, 11))

        self.assertEqual(roots, expected)
        self.assertEqual(tonelli_shanks(64, [7, 11]), expected)
        self.assertTrue(all(root * root % 77 == 64 for root in roots))
        with self.assertRaisesRegex(ValueError, "pairwise"):
            modular_square_roots(1, (7, 7))

    def test_pohlig_hellman_merges_repeated_prime_factors(self) -> None:
        prime = 97
        generator = 5
        secret = 37
        target = pow(generator, secret, prime)
        repeated_factors = (2, 2, 2, 2, 2, 3)

        for method in ("auto", "bsgs", "lambda"):
            with self.subTest(method=method):
                recovered = pohlig_hellman(
                    generator,
                    target,
                    prime,
                    repeated_factors,
                    method=method,
                )
                self.assertEqual(recovered, secret)
                self.assertEqual(pow(generator, recovered, prime), target)

        with self.assertRaisesRegex(ValueError, "product"):
            pohlig_hellman(generator, target, prime, repeated_factors, q=48)


class CoefficientPolynomialTests(unittest.TestCase):
    """Verify arithmetic and reductions without operand or input mutation."""

    def test_arithmetic_is_immutable_and_trims_trailing_zeroes(self) -> None:
        left = Polynomial([1, 2])
        right = Polynomial([3, 4, 5])

        self.assertEqual((left + right).coefficients, (4, 6, 5))
        self.assertEqual((left - right).coefficients, (-2, -2, -5))
        self.assertEqual((left * right).coefficients, (3, 10, 13, 10))
        self.assertEqual(left.coefficients, (1, 2))
        self.assertEqual(right.coefficients, (3, 4, 5))

        compatibility_copy = left.coef
        compatibility_copy.append(99)
        self.assertEqual(left.coefficients, (1, 2))
        self.assertEqual(Polynomial([1, 0, 0]), Polynomial([1]))

    def test_concrete_order_and_polynomial_reduction(self) -> None:
        reduced_coefficients = Polynomial([15, -1, 3]).reduce_by_order(
            7,
            3,
            [],
        )
        self.assertEqual(reduced_coefficients.coefficients, (1, 6, 3))

        modulus = Polynomial([1, 1, 1])
        reduced_polynomial = Polynomial([0, 0, 0, 1]).reduce_by_polynomial(
            modulus,
            2,
            [],
        )
        self.assertEqual(reduced_polynomial, Polynomial([1]))
        with self.assertRaisesRegex(ValueError, "monic"):
            Polynomial([1, 2, 3]).reduce_by_polynomial(
                Polynomial([1, 1, 2]),
                2,
                [],
            )

    def test_symbolic_reduction_uses_quotients_without_consuming_them(self) -> None:
        coefficient, quotient = var("other_coefficient other_quotient")
        quotients = [quotient]

        reduced = Polynomial([coefficient]).reduce_by_order(7, 1, quotients)

        self.assertEqual(reduced.coefficients, (coefficient - 7 * quotient,))
        self.assertEqual(quotients, [quotient])

    def test_construction_from_a_sage_finite_field_polynomial(self) -> None:
        polynomial_ring = PolynomialRing(GF(7), "other_finite_x")
        finite_polynomial = polynomial_ring([3, 5, 0])
        recovered = Polynomial.from_gf(finite_polynomial)
        self.assertEqual(recovered.coefficients, (3, 5))
        self.assertEqual(Polynomial.from_list([3, 5]), recovered)


class PackedAffineFunctionTests(unittest.TestCase):
    """Exercise unequal word widths and exact affine reconstruction."""

    def test_pack_and_unpack_are_msb_first_and_reject_overflow(self) -> None:
        packed = pack_words((5, 0x123), (3, 12))
        self.assertEqual(packed, (5 << 12) | 0x123)
        self.assertEqual(unpack_words(packed, (3, 12)), (5, 0x123))
        with self.assertRaisesRegex(ValueError, "fit"):
            pack_words((8,), (3,))
        with self.assertRaisesRegex(ValueError, "fit"):
            unpack_words(1 << 8, (8,))

    def test_wrapper_preserves_unequal_argument_and_result_boundaries(self) -> None:
        def transform(left, right):
            return left ^ 3, (right + 1) & 0x1F

        wrapper = wrapper_maker(transform, (3, 5), (3, 5))
        packed_input = pack_words((5, 17), (3, 5))
        packed_output = wrapper(packed_input)

        self.assertEqual(unpack_words(packed_output, (3, 5)), (6, 18))
        scalar_wrapper = wrapper_maker(lambda value: value ^ 7, (4,), (4,))
        self.assertEqual(scalar_wrapper(9), 14)

        unordered_wrapper = wrapper_maker(lambda _value: {1, 2}, (2,), (2, 2))
        with self.assertRaisesRegex(TypeError, "list or tuple"):
            unordered_wrapper(0)

    @staticmethod
    def affine_function(value):
        bit3 = value >> 3 & 1
        bit2 = value >> 2 & 1
        bit1 = value >> 1 & 1
        bit0 = value & 1
        return (bit3 << 2) | ((bit2 ^ bit0) << 1) | (bit1 ^ 1)

    def test_gf2_matrix_reproduces_every_small_input(self) -> None:
        transformation, offset = construct_GF2_matrix(
            self.affine_function,
            4,
            output_dimension=3,
        )

        self.assertEqual(transformation.dimensions(), (4, 3))
        self.assertEqual(tuple(map(int, offset)), (0, 0, 1))
        field = GF(2)
        for value in range(16):
            input_bits = vector(field, [int(bit) for bit in f"{value:04b}"])
            output = input_bits * transformation + offset
            expected = tuple(int(bit) for bit in f"{self.affine_function(value):03b}")
            self.assertEqual(tuple(map(int, output)), expected)

    def test_gf2_matrix_rejects_a_nonlinear_function(self) -> None:
        nonlinear = lambda value: (value & 1) & (value >> 1 & 1)
        with self.assertRaisesRegex(ValueError, "not affine"):
            construct_GF2_matrix(nonlinear, 2, output_dimension=1)


class GroebnerExtractionTests(unittest.TestCase):
    """Extract only independent affine solutions and reject wider systems."""

    def test_rational_affine_basis_is_solved_and_verified(self) -> None:
        ring = PolynomialRing(QQ, names=("other_x", "other_y"), order="lex")
        x_value, y_value = ring.gens()

        solution = ideal_groebner((2 * x_value - 6, 3 * y_value + 12))

        self.assertEqual(solution[x_value], 3)
        self.assertEqual(solution[y_value], -4)

    def test_base_elements_remain_symbolic_parameters(self) -> None:
        ring = PolynomialRing(QQ, names=("other_u", "other_t"), order="lex")
        unknown, parameter = ring.gens()

        solution = ideal_groebner(
            (unknown - parameter - 1,),
            base_elements=(parameter,),
        )

        self.assertEqual(solution[unknown], parameter + 1)

    def test_integer_basis_recovers_modular_affine_values(self) -> None:
        ring = PolynomialRing(ZZ, names=("other_mx", "other_my"), order="lex")
        x_value, y_value = ring.gens()

        solution = ideal_groebner((101, 3 * x_value - 7, y_value - 5))

        self.assertEqual(solution["mod"], 101)
        self.assertEqual(solution[x_value], 36)
        self.assertEqual(solution[y_value], 5)
        self.assertEqual((3 * solution[x_value] - 7) % 101, 0)

    def test_nonlinear_and_inconsistent_bases_are_rejected(self) -> None:
        ring = PolynomialRing(QQ, "other_z")
        variable = ring.gen()
        with self.assertRaisesRegex(ValueError, "not affine"):
            ideal_groebner((variable**2 - 1,))
        with self.assertRaisesRegex(ValueError, "inconsistent"):
            ideal_groebner((ring(1),))


class PolynomialRecoveryCompatibilityTests(unittest.TestCase):
    """Ensure the former appended helpers use the dedicated rewrite."""

    @staticmethod
    def samples():
        coefficients = (7, 4, 9)
        modulus = 101
        return tuple(
            (
                x_value,
                sum(
                    coefficient * x_value**position
                    for position, coefficient in enumerate(coefficients)
                )
                % modulus,
            )
            for x_value in (1, 3, 4, 8, 10)
        )

    def test_polynomial_recovery_compatibility_exports(self) -> None:
        samples = self.samples()
        self.assertEqual(recover_modulus_of_polynomial_fast(samples, 2), 101)
        self.assertEqual(recover_modulus_of_polynomial_slow(samples), 101)
        self.assertEqual(
            recover_modular_polynomial(samples, 101, degree=2),
            [7, 4, 9],
        )


if __name__ == "__main__":
    unittest.main()
