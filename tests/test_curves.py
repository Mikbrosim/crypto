"""Small correctness tests for every rewritten elliptic-curve helper."""

from __future__ import annotations

from contextlib import redirect_stdout
from io import StringIO
import unittest

from sage.all import EllipticCurve, GF, Zmod, set_random_seed

from rewrite.curves import (
    CurveAnalysis,
    CurveAttackError,
    NonceReuseCandidate,
    analyze_curve,
    embedding_degree,
    frey_ruck_attack,
    get_embedding_degree,
    is_anomalous,
    mov_attack,
    padic_ecdlp,
    parameter_recovery,
    recover_nonce,
    recover_reused_nonce,
    singular_curve_attack,
    smart_attack,
    solve_linear_congruence,
    test as diagnose_curve,
)


def all_extension_points(curve):
    """Give pairing tests a deterministic source on their tiny curve."""

    return curve.points()


class CurveDiagnosticTests(unittest.TestCase):
    """Distinguish field, subgroup, and residue-ring properties."""

    def test_embedding_degree_uses_the_selected_subgroup(self) -> None:
        curve = EllipticCurve(GF(5), [0, 1])
        point = curve(0, 1)
        self.assertEqual(point.order(), 3)
        self.assertEqual(embedding_degree(5, 3, max_k=4), 2)
        self.assertEqual(
            get_embedding_degree(curve, max_k=4, subgroup_order=point.order()),
            2,
        )
        self.assertIsNone(embedding_degree(5, 3, max_k=1))

    def test_structured_analysis_and_compatibility_printer(self) -> None:
        curve = EllipticCurve(GF(5), [0, 1])
        point = curve(0, 1)
        analysis = analyze_curve(curve, point=point, max_embedding_degree=4)

        self.assertIsInstance(analysis, CurveAnalysis)
        self.assertEqual(analysis.base_kind, "finite_field")
        self.assertEqual(analysis.field_order, 5)
        self.assertEqual(analysis.subgroup_order, 3)
        self.assertEqual(analysis.embedding_degree, 2)
        self.assertFalse(analysis.anomalous)
        self.assertTrue(analysis.might_be_vulnerable)
        self.assertFalse(is_anomalous(curve))

        output = StringIO()
        with redirect_stdout(output):
            vulnerable = diagnose_curve(curve, point=point)
        self.assertTrue(vulnerable)
        self.assertIn("embedding degree 2", output.getvalue())

    def test_finite_extensions_are_not_prime_power_residue_rings(self) -> None:
        field_curve = EllipticCurve(GF(5**3), [1, 1])
        ring_curve = EllipticCurve(Zmod(5**3), [1, 1])

        field_analysis = analyze_curve(field_curve)
        ring_analysis = analyze_curve(ring_curve)

        self.assertEqual(field_analysis.base_kind, "finite_field")
        self.assertIsNone(field_analysis.residue_prime)
        self.assertEqual(ring_analysis.base_kind, "prime_power_residue_ring")
        self.assertEqual(ring_analysis.residue_prime, 5)
        self.assertEqual(ring_analysis.residue_exponent, 3)


class PairingAttackTests(unittest.TestCase):
    """Reduce a three-element subgroup to GF(5^2)^*."""

    def setUp(self) -> None:
        self.curve = EllipticCurve(GF(5), [0, 1])
        self.base = self.curve(0, 1)
        self.target = 2 * self.base

    def test_mov_attack_returns_a_verified_logarithm(self) -> None:
        recovered = mov_attack(
            self.base,
            self.target,
            max_k=2,
            max_tries=100,
            auxiliary_points=all_extension_points,
        )
        self.assertEqual(recovered, 2)
        self.assertEqual(recovered * self.base, self.target)

    def test_frey_ruck_attack_returns_a_verified_logarithm(self) -> None:
        recovered = frey_ruck_attack(
            self.base,
            self.target,
            max_k=2,
            max_tries=100,
            auxiliary_points=all_extension_points,
        )
        self.assertEqual(recovered, 2)
        self.assertEqual(recovered * self.base, self.target)

    def test_pairing_attacks_report_an_insufficient_degree_bound(self) -> None:
        self.assertIsNone(mov_attack(self.base, self.target, max_k=1))


class PadicAttackTests(unittest.TestCase):
    """Exercise anomalous-field and prime-power-ring recovery separately."""

    def test_smart_attack_over_an_extension_field(self) -> None:
        # Small published-style anomalous example: #E(GF(11^2)) = 11^2.
        field = GF(11**2, names="smart_test_generator")
        generator = field.gen()
        curve = EllipticCurve(field, [generator, 8 * generator])
        base = curve.gen(0)
        secret = 57
        target = secret * base

        self.assertTrue(is_anomalous(curve))
        self.assertEqual(base.order(), field.order())
        set_random_seed(20260829)
        recovered = smart_attack(base, target, max_tries=8)

        self.assertEqual(recovered, secret)
        self.assertEqual(recovered * base, target)

    def test_prime_power_ring_attack_recovers_a_small_exact_secret(self) -> None:
        ring = Zmod(5**4)
        curve = EllipticCurve(ring, [1, -1])
        base = curve(1, 1)
        secret = 37
        target = secret * base

        recovered = padic_ecdlp(base, target)

        self.assertEqual(recovered, secret)
        self.assertEqual(recovered * base, target)
        self.assertEqual(padic_ecdlp(base, target, digits_per_step=1), secret)

        # The function deliberately recovers only modulo 5^(4-1).
        shifted_target = target + 2 * 5**3 * base
        self.assertEqual(padic_ecdlp(base, shifted_target), secret)

    def test_padic_attack_rejects_a_finite_extension_field(self) -> None:
        curve = EllipticCurve(GF(5**2), [1, 1])
        base = curve.gen(0)
        with self.assertRaises(ValueError):
            padic_ecdlp(base, 2 * base)


class ParameterAndNonceTests(unittest.TestCase):
    """Recover short-Weierstrass parameters and reused ECDSA scalars."""

    def test_parameter_recovery_fits_both_points(self) -> None:
        field = GF(101)
        curve = EllipticCurve(field, [2, 3])
        points = [point for point in curve.points() if not point.is_zero()]
        first = points[0]
        second = next(point for point in points if point[0] != first[0])

        recovered = parameter_recovery(
            101,
            int(first[0]),
            int(first[1]),
            int(second[0]),
            int(second[1]),
        )

        self.assertEqual(recovered, (2, 3))
        with self.assertRaises(ValueError):
            parameter_recovery(101, 1, 2, 1, 99)

    def test_linear_congruence_checks_divisibility_and_returns_all_roots(self) -> None:
        self.assertEqual(solve_linear_congruence(6, 8, 14), (6, 13))
        self.assertEqual(solve_linear_congruence(6, 5, 14), ())
        with self.assertRaises(CurveAttackError):
            solve_linear_congruence(0, 0, 101, max_solutions=10)

    def test_prime_order_nonce_reuse_recovers_the_unique_pair(self) -> None:
        order = 101
        private_key = 37
        nonce = 23
        r_value = 17
        messages = (11, 42)
        signatures = tuple(
            (message + r_value * private_key) * pow(nonce, -1, order) % order
            for message in messages
        )

        candidates = recover_reused_nonce(
            order,
            messages[0],
            r_value,
            signatures[0],
            messages[1],
            r_value,
            signatures[1],
        )

        self.assertEqual(candidates, (NonceReuseCandidate(nonce, private_key),))
        self.assertEqual(
            list(
                recover_nonce(
                    order,
                    messages[0],
                    r_value,
                    signatures[0],
                    messages[1],
                    r_value,
                    signatures[1],
                )
            ),
            [(nonce, private_key)],
        )

    def test_composite_order_keeps_every_verified_candidate(self) -> None:
        candidates = recover_reused_nonce(15, 1, 6, 5, 4, 6, 14)
        self.assertEqual(
            candidates,
            (
                NonceReuseCandidate(2, 4),
                NonceReuseCandidate(2, 9),
                NonceReuseCandidate(2, 14),
            ),
        )
        with self.assertRaises(ValueError):
            recover_reused_nonce(101, 1, 4, 5, 2, 7, 8)


class SingularCurveAttackTests(unittest.TestCase):
    """Solve the additive cusp and both forms of multiplicative node."""

    def test_cuspidal_curve_uses_the_additive_parameterization(self) -> None:
        field = GF(101)
        base_parameter = field(2)
        secret = 7
        target_parameter = secret * base_parameter

        base_x = 1 / base_parameter**2
        base_y = 1 / base_parameter**3
        target_x = 1 / target_parameter**2
        target_y = 1 / target_parameter**3

        recovered = singular_curve_attack(
            101,
            0,
            0,
            0,
            int(base_x),
            int(base_y),
            int(target_x),
            int(target_y),
        )
        self.assertEqual(recovered, secret)

    def _nodal_instance(self, prime, secret):
        field = GF(prime)
        tangent_square = field(-1)
        if tangent_square.is_square():
            parameter_field = field
            tangent = tangent_square.sqrt()
        else:
            parameter_field = GF(prime**2, names=f"node_test_{prime}")
            tangent = parameter_field(tangent_square).sqrt()

        # On y^2 = x^2(x-1), writing c=y/x gives x=c^2+1.
        base_c = field(2)
        base_x = base_c**2 + 1
        base_y = base_c * base_x
        base_parameter = (
            parameter_field(base_y) + tangent * parameter_field(base_x)
        ) / (
            parameter_field(base_y) - tangent * parameter_field(base_x)
        )
        target_parameter = base_parameter**secret
        target_c = tangent * (target_parameter + 1) / (target_parameter - 1)
        target_c = field(int(target_c))
        target_x = target_c**2 + 1
        target_y = target_c * target_x
        return tuple(map(int, (base_x, base_y, target_x, target_y)))

    def test_split_node_uses_the_base_field_multiplicative_group(self) -> None:
        coordinates = self._nodal_instance(101, 7)
        recovered = singular_curve_attack(101, -1, 0, 0, *coordinates)
        self.assertEqual(recovered, 7)

    def test_nonsplit_node_uses_the_quadratic_extension(self) -> None:
        coordinates = self._nodal_instance(103, 7)
        recovered = singular_curve_attack(103, -1, 0, 0, *coordinates)
        self.assertEqual(recovered, 7)

    def test_singular_attack_rejects_points_off_the_cubic(self) -> None:
        with self.assertRaises(ValueError):
            singular_curve_attack(101, 0, 0, 0, 1, 1, 2, 2)


if __name__ == "__main__":
    unittest.main()
