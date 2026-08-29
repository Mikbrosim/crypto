"""Correctness tests for RSA factor recovery from known high bits."""

from __future__ import annotations

from contextlib import redirect_stdout
from io import StringIO
import unittest

from sage.all import Matrix, PolynomialRing, ZZ, is_prime

from rewrite.rsahighbits import (
    attack_full,
    construct_lattice,
    gen_rsakey,
    generate_problem_instance,
    get_kt,
    main,
    recover_factor_from_high_bits,
    roots_from_reduced_lattice,
    solve_from_reduced_lattice,
)


class RSAHighBitsTests(unittest.TestCase):
    """Use a balanced 17-bit modulus with a three-bit factor suffix."""

    P = 251
    Q = 269
    MODULUS = P * Q
    UNKNOWN_BITS = 2
    BOUND = 1 << UNKNOWN_BITS
    KNOWN_HIGH = P - P % BOUND
    ROOT = P - KNOWN_HIGH
    PROBLEM = (MODULUS, KNOWN_HIGH, UNKNOWN_BITS)

    def assert_factor(self, factor) -> None:
        self.assertEqual(factor, self.P)
        self.assertEqual(self.MODULUS % factor, 0)
        self.assertLess(self.ROOT, self.BOUND)
        self.assertEqual(self.KNOWN_HIGH + self.ROOT, factor)

    def test_key_and_problem_generation_preserve_declared_bits(self) -> None:
        modulus, (p, q) = gen_rsakey(32)
        self.assertEqual(modulus, p * q)
        self.assertEqual(modulus.bit_length(), 32)
        self.assertTrue(is_prime(p))
        self.assertTrue(is_prime(q))
        self.assertNotEqual(p, q)
        self.assertEqual(p >> 14, 0b11)
        self.assertEqual(q >> 14, 0b11)

        answer, problem = generate_problem_instance(5, 32)
        generated_modulus, known_high, unknown_bits = problem
        generated_p, generated_q = answer
        self.assertEqual(generated_modulus, generated_p * generated_q)
        self.assertEqual(unknown_bits, 5)
        self.assertEqual(known_high % (1 << unknown_bits), 0)
        self.assertEqual(generated_p - known_high, generated_p % 32)

    def test_parameter_selection_is_general_and_resource_bounded(self) -> None:
        self.assertEqual(
            get_kt(3, 32, factor_bits=16, max_dimension=64),
            (1, 3),
        )
        with self.assertRaisesRegex(ValueError, "no conservative"):
            get_kt(8, 32, factor_bits=16, max_dimension=32)

    def test_constructed_lattice_has_the_required_modular_root(self) -> None:
        lattice = construct_lattice(
            self.PROBLEM,
            (self.P, self.Q),
            k=1,
            t=2,
            max_dimension=3,
        )

        self.assertEqual(lattice.dimensions(), (3, 3))
        ring = PolynomialRing(ZZ, "rsa_high_bits_test_x")
        for row in lattice.rows():
            coefficients = []
            for degree, entry in enumerate(row):
                scale = self.BOUND**degree
                self.assertEqual(entry % scale, 0)
                coefficients.append(entry // scale)
            polynomial = ring(coefficients)
            self.assertEqual(polynomial(self.ROOT) % self.P, 0)

    def test_root_extraction_checks_all_rows_and_includes_zero(self) -> None:
        extraction_bits = self.UNKNOWN_BITS + 1
        extraction_bound = 1 << extraction_bits
        reduced = Matrix(
            ZZ,
            [
                [-6, -extraction_bound, extraction_bound**2],
                [0, extraction_bound, 0],
                [1, 0, 0],
            ],
        )

        roots = roots_from_reduced_lattice(extraction_bits, reduced)

        self.assertEqual(roots, [0, 3])
        self.assertEqual(
            solve_from_reduced_lattice(extraction_bits, reduced),
            0,
        )
        self.assertIsNone(
            solve_from_reduced_lattice(1, Matrix(ZZ, [[1, 0]]))
        )

    def test_bruteforce_recovers_and_verifies_the_factor(self) -> None:
        factor = recover_factor_from_high_bits(
            self.MODULUS,
            self.KNOWN_HIGH,
            self.UNKNOWN_BITS,
            method="bruteforce",
            brute_force_bits=self.UNKNOWN_BITS,
        )
        self.assert_factor(factor)

        self.assertIsNone(
            recover_factor_from_high_bits(
                self.MODULUS,
                self.KNOWN_HIGH - self.BOUND,
                self.UNKNOWN_BITS,
                method="bruteforce",
                brute_force_bits=self.UNKNOWN_BITS,
            )
        )

    def test_zero_unknown_bits_take_the_exact_gcd_path(self) -> None:
        self.assert_factor(
            recover_factor_from_high_bits(
                self.MODULUS,
                self.P,
                0,
                method="auto",
            )
        )

    def test_sage_coppersmith_recovers_the_factor(self) -> None:
        factor = recover_factor_from_high_bits(
            self.MODULUS,
            self.KNOWN_HIGH,
            self.UNKNOWN_BITS,
            method="sage",
            max_dimension=64,
        )
        self.assert_factor(factor)

    def test_explicit_lattice_recovers_the_factor(self) -> None:
        factor = recover_factor_from_high_bits(
            self.MODULUS,
            self.KNOWN_HIGH,
            self.UNKNOWN_BITS,
            method="lattice",
            k=2,
            t=3,
            max_dimension=5,
        )
        self.assert_factor(factor)

    def test_custom_reducer_cannot_change_lattice_dimensions(self) -> None:
        def drop_a_row(basis):
            return Matrix(ZZ, list(basis.rows())[:-1])

        with self.assertRaisesRegex(ValueError, "dimensions"):
            recover_factor_from_high_bits(
                self.MODULUS,
                self.KNOWN_HIGH,
                self.UNKNOWN_BITS,
                method="lattice",
                k=1,
                t=2,
                max_dimension=3,
                lattice_reduction=drop_a_row,
            )

    def test_attack_wrapper_and_cli_solve_public_problem(self) -> None:
        factor = attack_full(
            self.UNKNOWN_BITS,
            self.PROBLEM,
            method="bruteforce",
            brute_force_bits=self.UNKNOWN_BITS,
        )
        self.assert_factor(factor)

        output = StringIO()
        with redirect_stdout(output):
            status = main(
                [
                    "--unknown-bits",
                    str(self.UNKNOWN_BITS),
                    "--N",
                    str(self.MODULUS),
                    "--known-high",
                    str(self.KNOWN_HIGH),
                    "--method",
                    "bruteforce",
                    "--brute-force-bits",
                    str(self.UNKNOWN_BITS),
                ]
            )
        self.assertEqual(status, 0)
        self.assertEqual(output.getvalue().strip(), hex(self.P))


if __name__ == "__main__":
    unittest.main()
