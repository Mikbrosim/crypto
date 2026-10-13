"""Independent small-field checks for the hyperelliptic helpers."""

from itertools import product
import unittest

from sage.all import CyclotomicField, GF, HyperellipticCurve, PolynomialRing, ZZ

from rewrite.hyperelliptic import (
    jacobian_order_from_point_counts,
    jacobian_order_y2_hy_xn,
    poly_crt,
    poly_sqrt_mod,
    recover_mumford_v,
    recover_plaintext_points,
    verify_jacobian_order,
)


def representatives(ring, degree):
    return [ring(coefficients) for coefficients in product(ring.base_ring(), repeat=degree)]


def point_counts(p, n, h, genus):
    """Count the smooth odd-degree projective model, including infinity."""
    counts = []
    for m in range(1, genus + 1):
        field = GF(p**m, name=f'count_a_{m}')
        count = 1
        for x in field:
            discriminant = field(h)**2 + 4*x**n
            count += 1 if not discriminant else (2 if discriminant.is_square() else 0)
        counts.append(count)
    return counts


def direct_jacobi_sum(p, n, root):
    """Definition sum_t chi(t)*chi(1-t); no ideals or normalization."""
    field = GF(p)
    cyclotomic = CyclotomicField(n)
    zeta = cyclotomic.gen()
    lookup = {root**j: zeta**j for j in range(n)}
    def character(value):
        return cyclotomic.zero() if not value else lookup[value**((p-1)//n)]
    return sum((character(t)*character(1-t) for t in field), cyclotomic.zero())


class PolynomialRootTests(unittest.TestCase):
    def test_crt_handles_generators_and_canonical_remainders(self):
        ring = PolynomialRing(GF(5), 'x')
        x = ring.gen()
        moduli = (x, x+1, x**2+2)
        residues = (x+3, 4, x+2)
        result = poly_crt(iter(residues), iter(moduli))
        self.assertLess(result.degree(), 4)
        for residue, modulus in zip(residues, moduli):
            self.assertEqual(result % modulus, ring(residue) % modulus)
        with self.assertRaisesRegex(ValueError, 'coprime'):
            poly_crt((1, 2), (x, x**2))
        with self.assertRaises(ValueError):
            poly_crt((1,), (x, x+1))
        with self.assertRaises(ValueError):
            poly_crt((), ())
        with self.assertRaises(ValueError):
            poly_crt((1,), (ring(0),))

    def test_repeated_linear_factors_match_exhaustive_enumeration(self):
        ring = PolynomialRing(GF(3), 'x')
        x = ring.gen()
        for modulus in (x**3, x**2*(x+1), x*(x+1)*(x+2)):
            values = representatives(ring, int(modulus.degree()))
            squares = {}
            for root in values:
                squares.setdefault(root**2 % modulus, set()).add(root)
            for value in values:
                with self.subTest(modulus=modulus, value=value):
                    expected = squares.get(value, set())
                    self.assertEqual(set(poly_sqrt_mod(value, modulus)), expected)
                    one = poly_sqrt_mod(value, modulus, all_roots=False)
                    if expected:
                        self.assertIn(one, expected)
                    else:
                        self.assertIsNone(one)

    def test_irreducible_prime_powers_and_mixed_crt(self):
        ring = PolynomialRing(GF(5), 'x')
        x = ring.gen()
        factor = x**2+2
        modulus = factor**3 * (x+1)**2
        original = x**3+2*x+1
        roots = poly_sqrt_mod(original**2, modulus)
        self.assertEqual(len(roots), 4)
        self.assertIn(original % modulus, roots)
        for root in roots:
            self.assertFalse((root**2-original**2) % modulus)

    def test_singular_roots_and_limits(self):
        ring = PolynomialRing(GF(3), 'x')
        x = ring.gen()
        values = representatives(ring, 5)
        expected = {root for root in values if (root**2-x**2) % x**5 == 0}
        self.assertEqual(set(poly_sqrt_mod(x**2, x**5)), expected)
        zero_roots = poly_sqrt_mod(0, x**5)
        self.assertEqual(len(zero_roots), 9)
        self.assertTrue(all(root % x**3 == 0 for root in zero_roots))
        self.assertEqual(poly_sqrt_mod(x, x**5), [])
        with self.assertRaisesRegex(ValueError, 'max_roots'):
            poly_sqrt_mod(0, x**5, max_roots=8)
        with self.assertRaisesRegex(ValueError, 'max_roots'):
            poly_sqrt_mod(1, x*(x+1)*(x+2), max_roots=4)
        self.assertEqual(poly_sqrt_mod(0, x**5, all_roots=False, max_roots=1), 0)

    def test_finite_extension_base_field(self):
        field = GF(9, 'a')
        ring = PolynomialRing(field, 'x')
        x = ring.gen()
        factor = next(x**2+b*x+c for b in field for c in field
                      if (x**2+b*x+c).is_irreducible())
        root = field.gen()*x+1
        expected = {r for r in representatives(ring, 2) if (r**2-root**2) % factor == 0}
        self.assertEqual(set(poly_sqrt_mod(root**2, factor)), expected)
        lifted = poly_sqrt_mod(root**2, factor**2)
        self.assertEqual(len(lifted), 2)
        self.assertTrue(all((r**2-root**2) % factor**2 == 0 for r in lifted))

    def test_input_validation_and_nonmonic_modulus(self):
        ring = PolynomialRing(GF(5), 'x')
        x = ring.gen()
        self.assertEqual(set(poly_sqrt_mod(1, 3*x)), {ring(1), ring(4)})
        with self.assertRaises(ValueError):
            poly_sqrt_mod(1, ring(1))
        with self.assertRaises(ValueError):
            poly_sqrt_mod(1, x, max_roots=0)
        binary = PolynomialRing(GF(2), 'x')
        with self.assertRaises(NotImplementedError):
            poly_sqrt_mod(1, binary.gen())

    def test_mumford_recovery_for_polynomial_h_and_identity(self):
        ring = PolynomialRing(GF(5), 'x')
        x = ring.gen()
        u = x**2*(x+1)
        original = x**2+3
        for h in (ring(0), ring(1), x+3):
            f = original**2 + h*original + u*(x**2+1)
            expected = {v for v in representatives(ring, 3) if (f-h*v-v**2) % u == 0}
            self.assertEqual(set(recover_mumford_v(f, h, u)), expected)
            self.assertIn(recover_mumford_v(f, h, u, all_roots=False), expected)
        self.assertEqual(recover_mumford_v(x**5, 1, ring(1)), [ring(0)])


class JacobianOrderTests(unittest.TestCase):
    def test_newton_reconstruction_against_extension_counts(self):
        for p, n, h in ((7, 3, 1), (11, 5, 1), (3, 7, 1)):
            genus = (n-1)//2
            counts = point_counts(p, n, h, 2*genus+1)
            order, polynomial = jacobian_order_from_point_counts(p, genus, counts[:genus])
            checked, _ = jacobian_order_from_point_counts(p, genus, counts)
            self.assertEqual(order, checked)
            self.assertEqual(polynomial[0], 1)
            self.assertEqual(polynomial[2*genus], p**genus)
            if n in (3, 5):
                self.assertEqual(order, jacobian_order_y2_hy_xn(p, n, h))

    def test_bad_counts_are_rejected(self):
        with self.assertRaisesRegex(ValueError, 'at least'):
            jacobian_order_from_point_counts(5, 2, [6])
        with self.assertRaisesRegex(ValueError, 'prime power'):
            jacobian_order_from_point_counts(6, 1, [7])
        with self.assertRaisesRegex(ValueError, 'Weil'):
            jacobian_order_from_point_counts(5, 1, [100])
        with self.assertRaisesRegex(ValueError, 'nonintegral'):
            jacobian_order_from_point_counts(5, 2, [6, 25])
        with self.assertRaisesRegex(ValueError, 'extra'):
            jacobian_order_from_point_counts(5, 1, [6, 26])
        self.assertEqual(jacobian_order_from_point_counts(5, 0, [6])[0], 1)

    def test_fast_jacobi_sums_match_definition_and_all_twists(self):
        for p, n in ((7, 3), (13, 3), (11, 5), (31, 5), (29, 7), (23, 11), (67, 11)):
            details = jacobian_order_y2_hy_xn(p, n, return_details=True)
            direct = direct_jacobi_sum(p, n, details.primitive_root)
            self.assertEqual(details.jacobi_sum, direct)
            zeta = details.cyclotomic_root
            # Constant h represents every twist since n is odd.
            for j in range(n):
                h = next(h for h in GF(p) if h and (h**-2)**((p-1)//n) == details.primitive_root**j)
                expected = ZZ((direct+zeta**j).norm())
                self.assertEqual(jacobian_order_y2_hy_xn(p, n, h), expected)

    def test_fast_orders_against_projective_counts(self):
        for p, n in ((7, 3), (11, 5), (29, 7)):
            for h in (1, 2, 3):
                expected, _ = jacobian_order_from_point_counts(p, (n-1)//2,
                                  point_counts(p, n, h, (n-1)//2))
                self.assertEqual(jacobian_order_y2_hy_xn(p, n, h), expected)

    def test_special_family_assumptions_and_singular_normalization(self):
        for p, n in ((9, 3), (11, 9), (13, 5), (2, 3)):
            with self.assertRaises(ValueError):
                jacobian_order_y2_hy_xn(p, n)
        with self.assertRaisesRegex(ValueError, 'singular'):
            jacobian_order_y2_hy_xn(7, 3, 0)
        details = jacobian_order_y2_hy_xn(7, 3, 0, normalize_h0=True, return_details=True)
        self.assertEqual(details.order, 1)
        self.assertTrue(details.singular)

    def test_small_scalar_leak_recovers_and_verifies_plaintext(self):
        ring = PolynomialRing(GF(31), 'x')
        x = ring.gen()
        curve = HyperellipticCurve(x**5, 1)
        jacobian = curve.jacobian()
        J = jacobian(jacobian.base_ring())
        order = jacobian_order_y2_hy_xn(31, 5)
        point = next(J(curve.lift_x(a)) for a in GF(31)
                     if a and (1+4*a**5).is_square())
        leak = next(multiplier*point for multiplier in range(1, 100)
                    if gcd_scalar(int((multiplier*point)[1][0]), int(order)))
        scalar = int(leak[1][0])
        ciphertext = scalar*point
        points = recover_plaintext_points(x**5, 1, leak[0], ciphertext[0], ciphertext[1], order)
        expected_x = int(-point[0][0])
        self.assertTrue(any(candidate.scalar == scalar and candidate.x == expected_x for candidate in points))
        self.assertTrue(verify_jacobian_order(J, order, [point, ciphertext]))
        with self.assertRaises(ValueError):
            verify_jacobian_order(J, 1, [point])


def gcd_scalar(scalar, order):
    from math import gcd
    return scalar != 0 and gcd(scalar, order) == 1


if __name__ == '__main__':
    unittest.main()
