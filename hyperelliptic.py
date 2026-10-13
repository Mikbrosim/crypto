"""SageMath helpers for hyperelliptic CTFs, using y^2 + h(x)y = f(x).

Polynomial roots (including repeated factors), missing Mumford coordinates,
and point-count reconstruction work over finite fields of odd characteristic.
The fast cyclotomic order routine is specific to y^2 + h*y = x^n, with
constant nonzero h, odd primes p and n, and p = 1 mod n. It requires a
principal prime ideal in Q(zeta_n); it is not a general point counter.

Example, in Sage's Python interpreter::

    R = PolynomialRing(GF(67), 'x')
    x = R.gen()
    candidates = recover_mumford_v(x**11, 1, R([1, 2, 1]))
    order = jacobian_order_y2_hy_xn(67, 11)

To solve the supplied challenge without executing its generator::

    python hyperelliptic.py hyper/random_ctf_chal.py

Use a Python environment with SageMath installed. On traditional Sage
installations, invoke ``sage -python`` instead of ``python``.

No network access or external programs are used at runtime.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from math import gcd, prod
from operator import index
from typing import Any

from sage.all import CyclotomicField, GF, HyperellipticCurve, PolynomialRing, ZZ


__all__ = [
    'JacobiOrderResult', 'PlaintextPoint', 'poly_crt', 'poly_sqrt_mod',
    'recover_mumford_v', 'jacobian_order_from_point_counts',
    'jacobian_order_y2_hy_xn', 'verify_jacobian_order',
    'recover_plaintext_points',
]


def _integer(value: Any, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool):
        raise TypeError(f'{name} must be an integer, not bool')
    result = index(value)
    if result < minimum:
        raise ValueError(f'{name} must be at least {minimum}')
    return result


def _polynomial_modulus(modulus: Any, *, odd: bool = True):
    ring = modulus.parent()
    if ring.ngens() != 1 or not hasattr(modulus, 'degree'):
        raise TypeError('modulus must be a univariate polynomial')
    field = ring.base_ring()
    if not field.is_field() or not field.is_finite():
        raise ValueError('the polynomial base ring must be a finite field')
    if odd and field.characteristic() == 2:
        raise NotImplementedError('square-root recovery requires odd characteristic')
    if modulus.degree() < 1:
        raise ValueError('modulus must have positive degree')
    return ring, modulus.monic()


def poly_crt(values, moduli, *, verify: bool = True):
    """Canonical solution to polynomial congruences with coprime moduli.

    Moduli must share a univariate polynomial ring over a finite field.
    Iterables are consumed once. Constant and noncoprime moduli are rejected.
    """
    values, moduli = tuple(values), tuple(moduli)
    if not moduli or len(values) != len(moduli):
        raise ValueError('values and moduli must have equal, nonzero lengths')
    ring, _ = _polynomial_modulus(moduli[0], odd=False)
    normalized = []
    for modulus in moduli:
        if modulus.parent() != ring:
            raise ValueError('all moduli must have the same polynomial parent')
        _, modulus = _polynomial_modulus(modulus, odd=False)
        if any(modulus.gcd(previous).degree() > 0 for previous in normalized):
            raise ValueError('moduli must be pairwise coprime')
        normalized.append(modulus)
    values = tuple(ring(value) for value in values)
    combined = prod(normalized)
    result = ring.zero()
    for value, modulus in zip(values, normalized):
        partial = combined // modulus
        result += value * partial * partial.inverse_mod(modulus)
    result %= combined
    if verify and any(result % m != v % m for v, m in zip(values, normalized)):
        raise ArithmeticError('polynomial CRT verification failed')
    return result


def _pow_mod(base, exponent: int, modulus):
    """Binary exponentiation without constructing a huge polynomial first."""
    result = modulus.parent().one()
    base %= modulus
    while exponent:
        if exponent & 1:
            result = result * base % modulus
        base = base * base % modulus
        exponent >>= 1
    return result


def _sqrt_mod_irreducible(value, factor):
    # Tonelli--Shanks in F[x]/(factor). Using polynomial representatives avoids
    # Sage's differing absolute/relative finite-field extension conversions.
    ring = factor.parent()
    value %= factor
    if not value:
        return [ring.zero()]
    size = int(ring.base_ring().order()) ** int(factor.degree())
    if _pow_mod(value, (size - 1) // 2, factor) != 1:
        return []
    odd_part, exponent = size - 1, 0
    while odd_part % 2 == 0:
        odd_part //= 2
        exponent += 1
    if exponent == 1:
        root = _pow_mod(value, (size + 1) // 4, factor)
        return [root, -root % factor]
    nonresidue = ring.zero()
    while not nonresidue or _pow_mod(nonresidue, (size - 1) // 2, factor) == 1:
        nonresidue = ring.random_element(degree=int(factor.degree()) - 1)
    correction = _pow_mod(nonresidue, odd_part, factor)
    root = _pow_mod(value, (odd_part + 1) // 2, factor)
    remainder = _pow_mod(value, odd_part, factor)
    while remainder != 1:
        squared, position = remainder, 0
        while squared != 1 and position < exponent:
            squared = squared * squared % factor
            position += 1
        if position == exponent:
            raise ArithmeticError('residue-field square-root iteration failed')
        adjustment = _pow_mod(correction, 1 << (exponent - position - 1), factor)
        root = root * adjustment % factor
        correction = adjustment * adjustment % factor
        remainder = remainder * correction % factor
        exponent = position
    return [root, -root % factor]


def _residue_representatives(factor, exponent: int):
    ring = factor.parent()
    degree = int(factor.degree()) * exponent
    if degree == 0:
        yield ring.zero()
        return
    # Called only after the number of representatives has been bounded.
    for coefficients in product(ring.base_ring(), repeat=degree):
        yield ring(coefficients)


def _limit_roots(count: int, maximum: int):
    if count > maximum:
        raise ValueError(f'{count} roots exceed max_roots={maximum}')


def _sqrt_mod_prime_power(value, factor, exponent, all_roots, maximum):
    ring = factor.parent()
    modulus = factor**exponent
    unit = value % modulus
    valuation = 0
    while valuation < exponent and unit % factor == 0:
        unit //= factor
        valuation += 1
    size = int(ring.base_ring().order()) ** int(factor.degree())
    if valuation == exponent:
        if not all_roots:
            return [ring.zero()]
        _limit_roots(size ** (exponent // 2), maximum)
        return [factor ** ((exponent + 1) // 2) * free
                for free in _residue_representatives(factor, exponent // 2)]
    if valuation % 2:
        return []
    shift = valuation // 2
    precision = exponent - valuation
    roots = _sqrt_mod_irreducible(unit, factor)
    if not roots:
        return []
    if all_roots:
        _limit_roots(len(roots) * size**shift, maximum)
    else:
        roots = roots[:1]
    lifted = []
    for root in roots:
        power = factor
        for _ in range(1, precision):
            error, residual = (unit - root**2).quo_rem(power)
            if residual:
                raise ArithmeticError('invalid Hensel square-root seed')
            digit = error * (2 * root).inverse_mod(factor) % factor
            root += power * digit
            power *= factor
        free_values = (_residue_representatives(factor, shift)
                       if all_roots else (ring.zero(),))
        lifted.extend(factor**shift * (root + power * free) % modulus
                      for free in free_values)
    return lifted


def poly_sqrt_mod(f, M, *, all_roots: bool = True,
                  max_roots: int = 100_000, verify: bool = True):
    """Solve w^2 = f modulo M over a finite field of odd characteristic.

    Repeated irreducible factors and singular roots are supported. Return a
    list of canonical roots, or one root/None when all_roots=False. Enumeration
    is bounded by max_roots, including local singular families and CRT products.
    """
    ring, modulus = _polynomial_modulus(M)
    maximum = _integer(max_roots, 'max_roots', 1)
    value = ring(f) % modulus
    moduli, local_roots = [], []
    for factor, exponent in modulus.factor():
        exponent = int(exponent)
        roots = _sqrt_mod_prime_power(value, factor, exponent, all_roots, maximum)
        if not roots:
            return [] if all_roots else None
        moduli.append(factor**exponent)
        local_roots.append(roots)
    _limit_roots(prod(map(len, local_roots)), maximum)
    results = [poly_crt(roots, moduli, verify=verify)
               for roots in product(*local_roots)]
    if verify and any((root**2 - value) % modulus for root in results):
        raise ArithmeticError('polynomial square-root verification failed')
    return results if all_roots else results[0]


def recover_mumford_v(f, h, u, *, all_roots: bool = True,
                      max_roots: int = 100_000, verify: bool = True):
    """Recover all v of degree < deg(u) with u dividing f - h*v - v^2.

    Uses (2*v+h)^2 = h^2+4*f modulo u; h may be a polynomial. This checks
    the Mumford congruence, not curve smoothness or reduced divisor degree.
    The identity u=1 has the single coordinate v=0.
    """
    ring = u.parent()
    if u == 1:
        return [ring.zero()] if all_roots else ring.zero()
    ring, modulus = _polynomial_modulus(u)
    f, h = ring(f), ring(h)
    roots = poly_sqrt_mod(h**2 + 4*f, modulus, all_roots=all_roots,
                          max_roots=max_roots, verify=verify)
    if not all_roots:
        roots = [] if roots is None else [roots]
    inv2 = ring.base_ring()(2)**-1
    values = [((root - h) * inv2) % modulus for root in roots]
    if verify and any((f - h*v - v**2) % modulus for v in values):
        raise ArithmeticError('Mumford coordinate verification failed')
    return values if all_roots else (values[0] if values else None)


def jacobian_order_from_point_counts(q, genus, counts):
    """Return (L(1), L(T)) from counts[m-1] = #C(F_(q^m)).

    At least genus counts of the smooth projective curve are required,
    including points at infinity. Newton identities and the functional
    equation determine L(T)=prod(1-alpha_i*T). Supplied extra counts are
    checked; integrality and Weil bounds are necessary consistency checks,
    not a proof that arbitrary input counts come from a curve.
    """
    q = _integer(q, 'q', 2)
    genus = _integer(genus, 'genus')
    if not ZZ(q).is_prime_power():
        raise ValueError('q must be a prime power')
    counts = tuple(_integer(count, 'point count') for count in counts)
    if len(counts) < genus:
        raise ValueError('at least genus point counts are required')
    sums = [q**m + 1 - count for m, count in enumerate(counts, 1)]
    if any(total**2 > 4 * genus**2 * q**m
           for m, total in enumerate(sums, 1)):
        raise ValueError('point counts violate the Weil bounds')
    coefficients = [1] + [0] * (2 * genus)
    for k in range(1, genus + 1):
        numerator = -sum(coefficients[k-i] * sums[i-1] for i in range(1, k+1))
        coefficients[k], remainder = divmod(numerator, k)
        if remainder:
            raise ValueError('point counts give nonintegral Newton coefficients')
    for k in range(genus):
        coefficients[2*genus-k] = q**(genus-k) * coefficients[k]
    for m in range(genus + 1, len(sums) + 1):
        expected = -sum(coefficients[k] * sums[m-k-1]
                        for k in range(1, min(m, 2*genus + 1)))
        if m <= 2*genus:
            expected -= m * coefficients[m]
        if expected != sums[m-1]:
            raise ValueError('extra point counts disagree with the L-polynomial')
    polynomial = PolynomialRing(ZZ, 'T')(coefficients)
    order = polynomial(1)
    if order < 1:
        raise ValueError('point counts give a nonpositive Jacobian order')
    return order, polynomial


@dataclass(frozen=True)
class JacobiOrderResult:
    """Order and cyclotomic data; singular=True refers to normalization."""
    order: Any
    jacobi_sum: Any = None
    twist: int | None = None
    primitive_root: Any = None
    cyclotomic_root: Any = None
    singular: bool = False


def _cyclotomic_sigma(value, exponent):
    field = value.parent()
    return field(value.polynomial()(field.gen()**int(exponent)))


def jacobian_order_y2_hy_xn(p, n, h=1, *, proof: bool = True,
                           normalize_h0: bool = False,
                           return_details: bool = False):
    """Order for y^2+h*y=x^n, constant h, odd primes p,n, p=1 mod n.

    Uses the Buhler--Koblitz cyclotomic/Jacobi-sum method (1998):
    https://doi.org/10.1017/S000497270003207X . A principal generator beta
    of (p,zeta-r) yields J=unit*prod(sigma_(i^-1)(beta), i=1..(n-1)/2).
    Normalize by J=-1 mod (zeta-1)^2; for A=h^-2 and A^((p-1)/n)=r^j,
    the order is Norm(J+zeta^j). The relevant ideal must be principal.

    h=0 is singular and rejected unless normalize_h0=True, in which case
    its rational normalization has trivial Jacobian. proof=False permits
    Sage/PARI's conditional number-field computations.
    """
    p, n = _integer(p, 'p', 3), _integer(n, 'n', 3)
    if p % 2 == 0 or n % 2 == 0 or not ZZ(p).is_prime(proof=proof) or not ZZ(n).is_prime(proof=proof):
        raise ValueError('p and n must be odd primes')
    if p % n != 1:
        raise ValueError('p must be 1 modulo n')
    finite_field = GF(p)
    h = finite_field(h)
    if not h:
        if not normalize_h0:
            raise ValueError('h=0 defines a singular curve; request its normalization explicitly')
        details = JacobiOrderResult(ZZ(1), singular=True)
        return details if return_details else details.order
    exponent = (p-1) // n
    candidate = 2
    root = finite_field(candidate)**exponent
    while root == 1:
        candidate += 1
        root = finite_field(candidate)**exponent
    if root**n != 1 or root == 0:
        raise ArithmeticError('invalid nth root of unity')
    field = CyclotomicField(n)
    zeta = field.gen()
    ideal = field.ideal(p, zeta - ZZ(root))
    if ideal.norm() != p:
        raise ArithmeticError('cyclotomic prime ideal has incorrect norm')
    generators = ideal.gens_reduced(proof=proof)
    if len(generators) != 1:
        raise NotImplementedError('the cyclotomic prime ideal must be principal')
    beta = field(generators[0])
    if field.ideal(beta) != ideal or beta.norm() != p:
        raise ArithmeticError('principal generator verification failed')
    jacobi_seed = field.one()
    for i in range(1, (n-1)//2 + 1):
        jacobi_seed *= _cyclotomic_sigma(beta, pow(i, -1, n))
    if jacobi_seed * _cyclotomic_sigma(jacobi_seed, n-1) != p:
        raise ArithmeticError('Jacobi sum does not satisfy J*conjugate(J)=p')
    normalization_ideal = field.ideal((zeta-1)**2)
    candidates = [sign*zeta**k*jacobi_seed for sign in (1, -1) for k in range(n)
                  if sign*zeta**k*jacobi_seed + 1 in normalization_ideal]
    if len(candidates) != 1:
        raise ArithmeticError('Jacobi-sum normalization is not unique')
    jacobi_sum = candidates[0]
    target = (h**-2)**exponent
    twist = next((j for j in range(n) if root**j == target), None)
    if twist is None:
        raise ArithmeticError('constant h has no identified twist class')
    order = ZZ((jacobi_sum + zeta**twist).norm())
    if order < 1:
        raise ArithmeticError('computed Jacobian order is not positive')
    details = JacobiOrderResult(order, jacobi_sum, twist, root, zeta)
    return details if return_details else order


def verify_jacobian_order(J, order, divisors):
    """Check order*D=0 for supplied divisors; this does not prove exact order."""
    order = _integer(order, 'order', 1)
    if any(order * divisor != J(0) for divisor in divisors):
        raise ValueError('the supplied order does not annihilate a divisor')
    return True


@dataclass(frozen=True)
class PlaintextPoint:
    """One verified inverse-scalar point with an integer x-coordinate."""
    scalar: int
    x: int
    y: int

    def as_bytes(self) -> bytes:
        """Minimal unsigned big-endian x; leading zero bytes are not encoded."""
        return self.x.to_bytes(max(1, (self.x.bit_length()+7)//8), 'big')


def recover_plaintext_points(f, h, leaked_u, ciphertext_u, ciphertext_v, order,
                             *, max_roots: int = 100_000):
    """Undo ct=v_D(0)*P for candidates reconstructed from leaked u_D.

    Return verified degree-one plaintext points for invertible scalar
    candidates only. Noninvertible scalars require a separate division attack.
    Assumes a smooth odd-degree model with a single point at infinity.
    """
    ring, _ = _polynomial_modulus(leaked_u)
    if ring.base_ring().degree() != 1:
        raise NotImplementedError('integer-coordinate plaintext recovery requires a prime field')
    f, h = ring(f), ring(h)
    discriminant = h**2 + 4*f
    if discriminant.degree() < 3 or discriminant.degree() % 2 != 1 or not discriminant.is_squarefree():
        raise ValueError('a smooth odd-degree hyperelliptic model is required')
    genus = (int(discriminant.degree())-1)//2
    if leaked_u.degree() > genus:
        raise ValueError('leaked_u must be a reduced Mumford coordinate')
    order = _integer(order, 'order', 1)
    curve = HyperellipticCurve(f, h)
    jacobian = curve.jacobian()
    J = jacobian(jacobian.base_ring())
    ciphertext = J([ring(ciphertext_u), ring(ciphertext_v)])
    verify_jacobian_order(J, order, [ciphertext])
    points = []
    for v in recover_mumford_v(f, h, leaked_u, max_roots=max_roots):
        scalar = int(v[0])
        if gcd(scalar, order) != 1:
            continue
        plaintext = pow(scalar, -1, order) * ciphertext
        if plaintext[0].degree() != 1:
            continue
        coordinate = -plaintext[0][0] / plaintext[0][1]
        ordinate = plaintext[1](coordinate)
        if ordinate**2 + h(coordinate)*ordinate != f(coordinate) or scalar * plaintext != ciphertext:
            raise ArithmeticError('plaintext verification failed')
        points.append(PlaintextPoint(scalar, int(coordinate), int(ordinate)))
    return points


def main() -> int:
    """Read only literal challenge parameters; never execute the input file."""
    import argparse
    import ast
    from pathlib import Path

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('challenge', type=Path, help='Python file containing p, n, D_u, u, v literals')
    parser.add_argument('--max-roots', type=int, default=100_000)
    args = parser.parse_args()
    parameters = {}
    for node in ast.parse(args.challenge.read_text()).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name in ('p', 'n', 'D_u', 'u', 'v'):
                try:
                    parameters[name] = ast.literal_eval(node.value)
                except (ValueError, TypeError):
                    pass
    missing = {'p', 'n', 'D_u', 'u', 'v'} - parameters.keys()
    if missing:
        parser.error(f'missing literal parameters: {sorted(missing)}')
    p, n = parameters['p'], parameters['n']
    ring = PolynomialRing(GF(p), 'x')
    print('Computing Jacobian order...', flush=True)
    order = jacobian_order_y2_hy_xn(p, n)
    print(f'Jacobian order = {order}', flush=True)
    points = recover_plaintext_points(ring.gen()**n, 1, ring(parameters['D_u']),
                                     ring(parameters['u']), ring(parameters['v']), order,
                                     max_roots=args.max_roots)
    for point in points:
        print(f'scalar = {point.scalar}\nx = {point.x}\nplaintext bytes = {point.as_bytes()!r}')
    if not points:
        print('No degree-one plaintext from the invertible scalar candidates.')
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
