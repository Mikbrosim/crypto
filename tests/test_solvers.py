"""Correctness tests for the rewritten small-root solvers."""

from __future__ import annotations

import unittest

from sage.all import ZZ, var

from rewrite.solvers import (
    solve_linear,
    solve_linear_expressions,
    solve_linear_expressions_bounds,
    solve_quadratic_mod_power_2,
)


class QuadraticSolverTests(unittest.TestCase):
    """Compare root lifting with exhaustive search on small moduli."""

    def test_returns_every_root_for_representative_quadratics(self) -> None:
        polynomial_ring = ZZ["x"]
        x = polynomial_ring.gen()
        cases = (
            (x**2 - 1, 16),
            (2 * x**2 + 4 * x + 8, 8),
            (2 * x**2 + x + 1, 8),
            (x**2 + x + 1, 8),
            (2 * x + 2, 8),
        )

        for polynomial, modulus in cases:
            with self.subTest(polynomial=polynomial, modulus=modulus):
                expected = [
                    value
                    for value in range(modulus)
                    if polynomial(value) % modulus == 0
                ]
                self.assertEqual(
                    solve_quadratic_mod_power_2(polynomial, modulus),
                    expected,
                )


class LinearSolverTests(unittest.TestCase):
    """Exercise both lattice strategies on small exact-answer systems."""

    def test_pair_solver_recovers_a_bounded_modular_root(self) -> None:
        x = var("pair_solver_x")
        forms = [[(37, x), (-10, 1)]]
        bounds = {1: 1, x: 8}

        for method in ("direct", "kernel"):
            with self.subTest(method=method):
                solutions = solve_linear(
                    forms,
                    bounds,
                    101,
                    method=method,
                )
                self.assertIn({1: 1, x: 3}, solutions)
                for solution in solutions:
                    self.assertLessEqual(abs(solution[x]), bounds[x])
                    self.assertEqual(
                        (37 * solution[x] - 10) % 101,
                        0,
                    )

    def test_expression_solver_recovers_an_integer_solution(self) -> None:
        x, y = var("expression_solver_x expression_solver_y")
        expressions = [3 * x + y - 20, x - y]
        bounds = {1: 1, x: 8, y: 8}

        for method in ("direct", "kernel"):
            with self.subTest(method=method):
                results = solve_linear_expressions(
                    expressions,
                    bounds,
                    [None, None],
                    method=method,
                )
                self.assertIn(({1: 1, x: 5, y: 5}, [0, 0]), results)
                for solution, errors in results:
                    self.assertEqual(errors, [0, 0])
                    self.assertEqual(3 * solution[x] + solution[y], 20)
                    self.assertEqual(solution[x], solution[y])

    def test_interval_solver_handles_nonzero_inclusive_bounds(self) -> None:
        x, y = var("interval_solver_x interval_solver_y")
        results = solve_linear_expressions_bounds(
            [x + y - 19, x - y - 5],
            {1: 1, x: (10, 12), y: (7, 9)},
            [None, None],
        )

        self.assertIn(({1: 1, x: 12, y: 7}, [0, 0]), results)
        for solution, errors in results:
            self.assertEqual(errors, [0, 0])
            self.assertLessEqual(10, solution[x])
            self.assertLessEqual(solution[x], 12)
            self.assertLessEqual(7, solution[y])
            self.assertLessEqual(solution[y], 9)
            self.assertEqual(solution[x] + solution[y], 19)
            self.assertEqual(solution[x] - solution[y], 5)

    def test_expression_solver_rejects_nonlinear_input(self) -> None:
        x, y = var("nonlinear_solver_x nonlinear_solver_y")
        with self.assertRaisesRegex(ValueError, "not affine-linear"):
            solve_linear_expressions(
                [x * y - 1],
                {1: 1, x: 4, y: 4},
                None,
            )


if __name__ == "__main__":
    unittest.main()
