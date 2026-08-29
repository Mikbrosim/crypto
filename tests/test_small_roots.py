"""Correctness tests for small-root recovery and relation solvers."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from sage.all import Matrix, PolynomialRing, QQ, ZZ, Zmod

from rewrite.small_roots import (
    IIter,
    NonInvertibleLeadingCoefficient,
    small_roots,
    small_roots_jacobian,
    solve_bounded_polynomial_system,
    solve_system_with_gb,
    solve_system_with_jacobian,
    solve_system_with_resultants,
)


class PolynomialSystemSolverTests(unittest.TestCase):
    """Solve tiny exact systems and verify every returned coordinate."""

    def setUp(self) -> None:
        ring = PolynomialRing(
            ZZ,
            names=("small_roots_system_x", "small_roots_system_y"),
        )
        self.x, self.y = ring.gens()

    def assert_system_root(self, equations, root) -> None:
        self.assertEqual(root, (0, 3))
        self.assertTrue(all(equation(*root) == 0 for equation in equations))

    def test_resultants_preserve_zero_roots_and_do_not_mutate_variables(self) -> None:
        equations = (self.x, self.y - 3)
        variables = [self.x, self.y]

        roots = solve_system_with_resultants(equations, variables)

        self.assertEqual(variables, [self.x, self.y])
        self.assertEqual(roots, [(0, 3)])
        self.assert_system_root(equations, roots[0])

    def test_groebner_solver_returns_only_integral_full_system_roots(self) -> None:
        equations = (self.x, self.y - 3, self.x + self.y - 3)

        roots = solve_system_with_gb(equations, (self.x, self.y))

        self.assertEqual(roots, [(0, 3)])
        self.assert_system_root(equations, roots[0])

    def test_jacobian_solver_continues_past_singular_subsets(self) -> None:
        equations = (
            self.x + self.y - 3,
            2 * self.x + 2 * self.y - 6,
            self.x,
            self.y - 3,
        )

        roots = solve_system_with_jacobian(
            equations,
            self.x + self.y,
            (5, 5),
            12,
            100,
        )

        self.assertEqual(roots, [(0, 3)])
        self.assert_system_root(equations, roots[0])


class BoundedPolynomialSystemTests(unittest.TestCase):
    """Solve small nonlinear systems and verify every original equation."""

    def assert_product_system_solutions(
        self,
        equations,
        variables,
        solutions,
        modulus=None,
    ) -> None:
        x, y = variables
        self.assertEqual(
            solutions,
            [{x: 2, y: 3}, {x: 3, y: 2}],
        )
        for solution in solutions:
            values = (solution[x], solution[y])
            self.assertTrue(all(abs(value) < 5 for value in values))
            for equation in equations:
                evaluated = int(equation(*values))
                if modulus is None:
                    self.assertEqual(evaluated, 0)
                else:
                    self.assertEqual(evaluated % modulus, 0)

    def test_exhaustive_integer_system_supports_variable_products(self) -> None:
        ring = PolynomialRing(
            ZZ,
            names=("bounded_bruteforce_x", "bounded_bruteforce_y"),
        )
        x, y = ring.gens()
        equations = (x * y - 6, x + y - 5)

        solutions = solve_bounded_polynomial_system(
            equations,
            (5, 5),
            method="bruteforce",
        )

        self.assert_product_system_solutions(
            equations,
            (x, y),
            solutions,
        )

    def test_groebner_system_filters_to_strict_bounds(self) -> None:
        ring = PolynomialRing(
            ZZ,
            names=("bounded_groebner_x", "bounded_groebner_y"),
        )
        x, y = ring.gens()
        equations = (x * y - 6, x + y - 5)

        solutions = solve_bounded_polynomial_system(
            equations,
            {x: 5, y: 5},
            method="groebner",
            brute_force_limit=0,
        )

        self.assert_product_system_solutions(
            equations,
            (x, y),
            solutions,
        )

    def test_prime_field_groebner_system_returns_bounded_lifts(self) -> None:
        ring = PolynomialRing(
            Zmod(101),
            names=("bounded_prime_x", "bounded_prime_y"),
        )
        x, y = ring.gens()
        equations = (x * y - 6, x + y - 5)

        solutions = solve_bounded_polynomial_system(
            equations,
            (5, 5),
            method="groebner",
            brute_force_limit=0,
        )

        self.assert_product_system_solutions(
            equations,
            (x, y),
            solutions,
            modulus=101,
        )

    def test_composite_modulus_uses_correct_exhaustive_semantics(self) -> None:
        ring = PolynomialRing(
            Zmod(35),
            names=("bounded_composite_x", "bounded_composite_y"),
        )
        x, y = ring.gens()
        equations = (x * y - 6, x + y - 5)

        solutions = solve_bounded_polynomial_system(
            equations,
            (5, 5),
            method="bruteforce",
        )

        self.assert_product_system_solutions(
            equations,
            (x, y),
            solutions,
            modulus=35,
        )

    def test_composite_system_refuses_an_invalid_field_groebner_solve(self) -> None:
        ring = PolynomialRing(
            Zmod(35),
            names=("bounded_refusal_x", "bounded_refusal_y"),
        )
        x, y = ring.gens()

        with self.assertRaisesRegex(ValueError, "prime modulus"):
            solve_bounded_polynomial_system(
                (x * y - 6, x + y - 5),
                (20, 20),
                method="groebner",
                brute_force_limit=100,
            )

    def test_coppersmith_builds_mixed_shifts_for_underdetermined_system(self) -> None:
        modulus = 10007
        ring = PolynomialRing(
            Zmod(modulus),
            names=(
                "bounded_coppersmith_x",
                "bounded_coppersmith_y",
                "bounded_coppersmith_z",
            ),
        )
        x, y, z = ring.gens()
        secret = (2, 3, 1)
        equations = (
            x * y + 1000 * x + 1000 * y + 1000 * z + 4001,
            x * z + 1200 * x + 800 * y + 900 * z + 4305,
        )
        self.assertEqual(
            tuple(int(equation(*secret)) for equation in equations),
            (0, 0),
        )

        with patch(
            "rewrite.small_roots._solve_relation_system",
            return_value=[secret, (0, 0, 0)],
        ) as solve_relations:
            solutions = solve_bounded_polynomial_system(
                equations,
                (5, 5, 5),
                method="coppersmith",
                m=2,
                shift_degree=0,
                lattice_reduction=lambda basis: basis,
                brute_force_limit=0,
            )

        self.assertEqual(solutions, [{x: 2, y: 3, z: 1}])
        relations = solve_relations.call_args.args[0]
        self.assertGreaterEqual(len(relations), 3)
        self.assertTrue(
            all(int(relation(*secret)) % modulus**2 == 0 for relation in relations)
        )

    def test_coppersmith_recovers_more_variables_than_equations(self) -> None:
        modulus = 10001
        ring = PolynomialRing(
            Zmod(modulus),
            names=(
                "bounded_recovery_x",
                "bounded_recovery_y",
                "bounded_recovery_z",
            ),
        )
        x, y, z = ring.gens()
        equations = (
            x * y + 1000 * x + 1000 * y + 1000 * z + 3995,
            x * z + 1200 * x + 800 * y + 900 * z + 4299,
        )

        solutions = solve_bounded_polynomial_system(
            equations,
            (5, 5, 5),
            method="coppersmith",
            m=1,
            shift_degree=0,
            brute_force_limit=0,
        )

        self.assertEqual(solutions, [{x: 2, y: 3, z: 1}])
        for equation in equations:
            self.assertEqual(int(equation(2, 3, 1)), 0)


class SmallRootTests(unittest.TestCase):
    """Exercise exact, proven-univariate, and heuristic lattice paths."""

    def test_iiter_enumerates_every_bounded_composition_once(self) -> None:
        self.assertEqual(
            list(IIter(2, 2)),
            [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (2, 0)],
        )

    def test_exact_integer_polynomial_returns_all_bounded_roots(self) -> None:
        ring = PolynomialRing(ZZ, "small_roots_exact_x")
        x = ring.gen()

        roots = small_roots((x - 3) * (x + 4) * (x - 8), (5,))

        self.assertEqual(roots, [(-4,), (3,)])
        self.assertTrue(
            all(
                ((x - 3) * (x + 4) * (x - 8))(root[0]) == 0
                for root in roots
            )
        )

    def test_sage_univariate_coppersmith_finds_a_small_modular_root(self) -> None:
        ring = PolynomialRing(Zmod(101), "small_roots_modular_x")
        x = ring.gen()

        roots = small_roots(x - 7, (10,), method="sage")

        self.assertIn((7,), roots)
        for root, in roots:
            self.assertLess(abs(root), 10)
            self.assertEqual((x - 7)(root), 0)

    def test_jochemsz_may_path_rejects_false_and_out_of_bound_candidates(self) -> None:
        ring = PolynomialRing(
            Zmod(101),
            names=("small_roots_lattice_x", "small_roots_lattice_y"),
        )
        x, y = ring.gens()
        polynomial = x + 10 * y - 32

        with patch(
            "rewrite.small_roots._solve_relation_system",
            return_value=[(4, 4), (5, 3), (2, 3), (2, 3)],
        ):
            roots = small_roots(
                polynomial,
                (5, 5),
                m=1,
                d=1,
                method="jochemsz_may",
                lattice_reduction=lambda basis: basis,
                brute_force_limit=0,
            )

        self.assertEqual(roots, [(2, 3)])
        self.assertEqual(polynomial(*roots[0]), 0)

    def test_multivariate_path_solves_a_complete_tiny_problem(self) -> None:
        ring = PolynomialRing(
            Zmod(101),
            names=("small_roots_complete_x", "small_roots_complete_y"),
        )
        x, y = ring.gens()
        polynomial = x + 10 * y - 32

        roots = small_roots(
            polynomial,
            (5, 5),
            m=1,
            d=1,
            method="jochemsz_may",
        )

        self.assertEqual(roots, [(2, 3)])
        self.assertEqual(polynomial(*roots[0]), 0)

    def test_beta_verification_uses_the_unknown_factor_bound(self) -> None:
        ring = PolynomialRing(Zmod(15), "small_roots_beta_x")
        x = ring.gen()
        polynomial = x + 3
        options = {
            "method": "jochemsz_may",
            "lattice_reduction": lambda basis: basis,
            "brute_force_limit": 0,
        }

        with patch(
            "rewrite.small_roots._solve_relation_system",
            return_value=[(2,)],
        ):
            factor_root = small_roots(polynomial, (4,), beta=QQ(1) / 2, **options)
            full_root = small_roots(polynomial, (4,), beta=1, **options)

        self.assertEqual(factor_root, [(2,)])
        self.assertEqual(full_root, [])
        self.assertGreaterEqual(5**2, 15)

    def test_noninvertible_leading_coefficient_exposes_a_factor(self) -> None:
        ring = PolynomialRing(Zmod(15), "small_roots_factor_x")
        x = ring.gen()

        with self.assertRaises(NonInvertibleLeadingCoefficient) as raised:
            small_roots(
                5 * x + 1,
                (4,),
                method="jochemsz_may",
                brute_force_limit=0,
            )

        self.assertEqual(raised.exception.factor, 5)
        self.assertEqual(raised.exception.modulus, 15)

    def test_specialized_jacobian_lattice_solves_a_tiny_affine_instance(self) -> None:
        ring = PolynomialRing(
            Zmod(101),
            names=("small_roots_jacobian_x", "small_roots_jacobian_y"),
        )
        x, y = ring.gens()
        polynomial = x + 10 * y - 32

        roots = small_roots_jacobian(
            polynomial,
            (5, 5),
            m=2,
            t=1,
        )

        self.assertEqual(roots, [(2, 3)])
        self.assertEqual(polynomial(*roots[0]), 0)

    def test_custom_reducer_cannot_change_the_lattice_dimensions(self) -> None:
        ring = PolynomialRing(Zmod(101), "small_roots_reducer_x")
        x = ring.gen()

        def drop_a_row(basis):
            return Matrix(ZZ, list(basis.rows())[:-1])

        with self.assertRaisesRegex(ValueError, "dimensions"):
            small_roots(
                x - 2,
                (5,),
                method="jochemsz_may",
                lattice_reduction=drop_a_row,
                brute_force_limit=0,
            )


if __name__ == "__main__":
    unittest.main()
