"""Elliptic-curve diagnostics and small, verified cryptanalytic helpers.

The attacks in this module intentionally return a result only after checking it
against the original problem.  Pairing reductions use the order of the supplied
base point (not the usually larger curve order), Smart's attack supports finite
extension fields through an unramified p-adic lift, and the prime-power-ring
attack is kept separate from finite fields of the same cardinality.

These routines are intended for CTFs, research examples, and deliberately weak
curves.  Generic discrete logarithms in the target finite fields can still be
expensive even when an elliptic-curve reduction itself succeeds.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from math import gcd
from operator import index as integer_index
from typing import Any

from sage.all import EllipticCurve, GF, PolynomialRing, Qp, Qq, ZZ


__all__ = [
    "CurveAnalysis",
    "CurveAttackError",
    "NonceReuseCandidate",
    "analyze_curve",
    "embedding_degree",
    "frey_ruck_attack",
    "get_embedding_degree",
    "is_anomalous",
    "mov_attack",
    "padic_ecdlp",
    "parameter_recovery",
    "recover_nonce",
    "recover_reused_nonce",
    "singular_curve_attack",
    "smart_attack",
    "solve_linear_congruence",
    "test",
]


class CurveAttackError(RuntimeError):
    """Raised when an applicable attack fails to produce a verified result."""


@dataclass(frozen=True)
class CurveAnalysis:
    """A structured summary of the inexpensive curve checks."""

    base_kind: str
    field_order: int | None
    curve_order: int | None
    subgroup_order: int | None
    anomalous: bool
    embedding_degree: int | None
    residue_prime: int | None
    residue_exponent: int | None

    @property
    def might_be_vulnerable(self) -> bool:
        """Whether at least one implemented special-case attack may apply."""

        return (
            self.anomalous
            or self.embedding_degree is not None
            or (
                self.residue_prime is not None
                and self.residue_exponent is not None
                and self.residue_exponent > 1
            )
        )


@dataclass(frozen=True, order=True)
class NonceReuseCandidate:
    """One nonce/private-key pair satisfying both ECDSA scalar equations."""

    nonce: int
    private_key: int


def _exact_integer(value: object, name: str) -> int:
    """Return an exact Python integer without accepting truncation or bools."""

    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer, not bool")
    try:
        return integer_index(value)
    except TypeError as error:
        raise TypeError(f"{name} must be an integer") from error


def _positive_integer(value: object, name: str) -> int:
    normalized = _exact_integer(value, name)
    if normalized <= 0:
        raise ValueError(f"{name} must be positive")
    return normalized


def _finite_field_order(curve: Any) -> int | None:
    """Return the base-field order, or ``None`` for a non-finite-field curve."""

    try:
        base_ring = curve.base_ring()
        if not bool(base_ring.is_field()) or not bool(base_ring.is_finite()):
            return None
        return int(base_ring.order())
    except (AttributeError, NotImplementedError, TypeError, ValueError):
        return None


def _prime_power_data(value: int) -> tuple[int, int] | None:
    """Recognize an exact prime power without mistaking GF(p^k) for Z/p^kZ."""

    candidate = ZZ(value)
    if candidate < 2:
        return None
    if candidate.is_prime():
        return int(candidate), 1

    # Trying exact roots is particularly cheap for the intended p^k moduli and
    # avoids factoring an otherwise arbitrary large integer.
    for exponent in range(candidate.nbits(), 1, -1):
        try:
            root = candidate.nth_root(exponent)
        except ValueError:
            continue
        if root.is_prime():
            return int(root), exponent
    return None


def _residue_ring_data(curve: Any) -> tuple[int, int] | None:
    """Return ``(p, k)`` only when the curve is over a non-field of order p^k."""

    try:
        base_ring = curve.base_ring()
        if bool(base_ring.is_field()) or not bool(base_ring.is_finite()):
            return None
        cardinality = int(base_ring.cardinality())
    except (AttributeError, NotImplementedError, TypeError, ValueError):
        return None
    data = _prime_power_data(cardinality)
    if data is None or data[1] < 2:
        return None
    return data


def _curve_order(curve: Any) -> int | None:
    try:
        return int(curve.order())
    except (AttributeError, ArithmeticError, NotImplementedError, TypeError, ValueError):
        return None


def _same_curve(base_point: Any, target_point: Any) -> Any:
    try:
        curve = base_point.curve()
        target_curve = target_point.curve()
    except AttributeError as error:
        raise TypeError("base and target must be elliptic-curve points") from error
    if curve != target_curve:
        raise ValueError("base and target must lie on the same curve")
    return curve


def _is_zero(point: Any) -> bool:
    try:
        return bool(point.is_zero())
    except AttributeError:
        return bool(point == point.curve()(0))


def embedding_degree(
    field_order: int,
    subgroup_order: int,
    max_k: int = 20,
) -> int | None:
    """Return the least ``k <= max_k`` with q^k = 1 modulo the subgroup order.

    The subgroup order is the order of the DLP base point.  Using the full
    curve order can hide a perfectly usable MOV or Frey--Rück reduction.
    """

    q = _positive_integer(field_order, "field_order")
    order = _positive_integer(subgroup_order, "subgroup_order")
    limit = _positive_integer(max_k, "max_k")
    if order == 1:
        raise ValueError("subgroup_order must be greater than one")
    if gcd(q, order) != 1:
        return None
    for degree in range(1, limit + 1):
        if pow(q, degree, order) == 1:
            return degree
    return None


def get_embedding_degree(
    curve: Any,
    max_k: int = 20,
    subgroup_order: int | None = None,
) -> int | None:
    """Compatibility wrapper around :func:`embedding_degree` for a curve."""

    field_order = _finite_field_order(curve)
    if field_order is None:
        return None
    if subgroup_order is None:
        subgroup_order = _curve_order(curve)
        if subgroup_order is None:
            return None
    return embedding_degree(field_order, subgroup_order, max_k)


def is_anomalous(curve: Any) -> bool:
    """Return whether a finite-field curve has order equal to its field size."""

    field_order = _finite_field_order(curve)
    if field_order is None:
        return False
    curve_order = _curve_order(curve)
    return (
        curve_order is not None
        and curve_order == field_order
    )


def analyze_curve(
    curve: Any,
    *,
    point: Any | None = None,
    subgroup_order: int | None = None,
    max_embedding_degree: int = 20,
) -> CurveAnalysis:
    """Inspect a curve for anomalous, pairing, and residue-ring weaknesses.

    Pass ``point`` or ``subgroup_order`` for a meaningful pairing diagnostic.
    Without either, the curve order is retained as a compatibility fallback.
    """

    limit = _positive_integer(max_embedding_degree, "max_embedding_degree")
    field_order = _finite_field_order(curve)
    curve_order = _curve_order(curve) if field_order is not None else None
    residue_data = _residue_ring_data(curve)

    if point is not None:
        try:
            if point.curve() != curve:
                raise ValueError("point must lie on curve")
            selected_order = int(point.order())
        except AttributeError as error:
            raise TypeError("point must be an elliptic-curve point") from error
        if subgroup_order is not None:
            supplied_order = _positive_integer(subgroup_order, "subgroup_order")
            if supplied_order != selected_order:
                raise ValueError("subgroup_order does not match point.order()")
    elif subgroup_order is not None:
        selected_order = _positive_integer(subgroup_order, "subgroup_order")
    else:
        selected_order = curve_order

    degree = None
    if field_order is not None and selected_order is not None and selected_order > 1:
        degree = embedding_degree(field_order, selected_order, limit)

    if field_order is not None:
        base_kind = "finite_field"
    elif residue_data is not None:
        base_kind = "prime_power_residue_ring"
    else:
        base_kind = "unsupported"

    return CurveAnalysis(
        base_kind=base_kind,
        field_order=field_order,
        curve_order=curve_order,
        subgroup_order=selected_order,
        anomalous=(
            field_order is not None
            and curve_order is not None
            and field_order == curve_order
        ),
        embedding_degree=degree,
        residue_prime=None if residue_data is None else residue_data[0],
        residue_exponent=None if residue_data is None else residue_data[1],
    )


def test(
    curve: Any,
    *,
    point: Any | None = None,
    subgroup_order: int | None = None,
    max_embedding_degree: int = 20,
) -> bool:
    """Print a compatibility diagnostic and return its vulnerability flag."""

    result = analyze_curve(
        curve,
        point=point,
        subgroup_order=subgroup_order,
        max_embedding_degree=max_embedding_degree,
    )
    if result.anomalous:
        print("[+] Curve is anomalous; Smart's attack may apply")
    else:
        print("[-] Curve is not anomalous")

    if result.residue_prime is not None:
        print(
            "[+] Curve is over Z/(p^k) with "
            f"p={result.residue_prime}, k={result.residue_exponent}; "
            "partial p-adic recovery may apply"
        )
    else:
        print("[-] Curve is not over a supported prime-power residue ring")

    if result.embedding_degree is None:
        print(
            "[-] No embedding degree at most "
            f"{max_embedding_degree} was found for the selected subgroup"
        )
    else:
        print(
            f"[+] Selected subgroup has embedding degree "
            f"{result.embedding_degree}; MOV or Frey--Rück may apply"
        )
    return result.might_be_vulnerable


AuxiliaryPointSource = Callable[[Any], Iterable[Any]]


def _auxiliary_points(
    extension_curve: Any,
    max_tries: int,
    source: AuxiliaryPointSource | None,
) -> Iterator[Any]:
    if source is None:
        for _ in range(max_tries):
            yield extension_curve.random_point()
        return

    try:
        iterator = iter(source(extension_curve))
    except TypeError as error:
        raise TypeError(
            "auxiliary_points must return an iterable of curve points"
        ) from error
    for attempt, point in enumerate(iterator):
        if attempt >= max_tries:
            break
        yield point


def _pairing_problem(
    base_point: Any,
    target_point: Any,
    max_k: int,
) -> tuple[Any, Any, Any, int, int] | None:
    curve = _same_curve(base_point, target_point)
    if _is_zero(base_point):
        raise ValueError("base point must be nonzero")
    order = int(base_point.order())
    field_order = _finite_field_order(curve)
    if field_order is None:
        raise ValueError("pairing attacks require a finite-field curve")
    if gcd(order, field_order) != 1:
        raise ValueError(
            "the base-point order must be coprime to the field characteristic"
        )
    degree = embedding_degree(field_order, order, max_k)
    if degree is None:
        return None

    extension_curve = curve.base_extend(GF(field_order**degree))
    return (
        extension_curve,
        extension_curve(base_point),
        extension_curve(target_point),
        order,
        degree,
    )


def _project_to_subgroup(candidate: Any, subgroup_order: int) -> Any | None:
    try:
        candidate_order = int(candidate.order())
        divisor = gcd(candidate_order, subgroup_order)
        projected = (candidate_order // divisor) * candidate
        if int(projected.order()) != subgroup_order:
            return None
        return projected
    except (ArithmeticError, TypeError, ValueError):
        return None


def mov_attack(
    base_point: Any,
    target_point: Any,
    max_k: int = 6,
    max_tries: int = 10,
    *,
    auxiliary_points: AuxiliaryPointSource | None = None,
) -> int | None:
    """Solve ``target_point = l * base_point`` using the MOV reduction.

    ``auxiliary_points`` is primarily useful for deterministic tests; it is a
    callable receiving the extension curve and returning candidate points.
    """

    limit = _positive_integer(max_k, "max_k")
    attempts = _positive_integer(max_tries, "max_tries")
    _same_curve(base_point, target_point)
    if _is_zero(base_point):
        raise ValueError("base point must be nonzero")
    if _is_zero(target_point):
        return 0
    problem = _pairing_problem(base_point, target_point, limit)
    if problem is None:
        return None
    extension_curve, extended_base, extended_target, order, _degree = problem

    for candidate in _auxiliary_points(extension_curve, attempts, auxiliary_points):
        auxiliary = _project_to_subgroup(candidate, order)
        if auxiliary is None:
            continue
        try:
            base_pairing = extended_base.weil_pairing(auxiliary, order)
            if int(base_pairing.multiplicative_order()) != order:
                continue
            target_pairing = extended_target.weil_pairing(auxiliary, order)
            logarithm = int(target_pairing.log(base_pairing)) % order
        except (ArithmeticError, TypeError, ValueError, ZeroDivisionError):
            continue
        if logarithm * base_point == target_point:
            return logarithm
    return None


def frey_ruck_attack(
    base_point: Any,
    target_point: Any,
    max_k: int = 6,
    max_tries: int = 10,
    *,
    auxiliary_points: AuxiliaryPointSource | None = None,
) -> int | None:
    """Solve ``target_point = l * base_point`` using a reduced Tate pairing."""

    limit = _positive_integer(max_k, "max_k")
    attempts = _positive_integer(max_tries, "max_tries")
    _same_curve(base_point, target_point)
    if _is_zero(base_point):
        raise ValueError("base point must be nonzero")
    if _is_zero(target_point):
        return 0
    problem = _pairing_problem(base_point, target_point, limit)
    if problem is None:
        return None
    extension_curve, extended_base, extended_target, order, degree = problem

    for candidate in _auxiliary_points(extension_curve, attempts, auxiliary_points):
        auxiliary = _project_to_subgroup(candidate, order)
        if auxiliary is None:
            continue
        try:
            base_pairing = extended_base.tate_pairing(auxiliary, order, degree)
            if int(base_pairing.multiplicative_order()) != order:
                continue
            target_pairing = extended_target.tate_pairing(
                auxiliary,
                order,
                degree,
            )
            logarithm = int(target_pairing.log(base_pairing)) % order
        except (ArithmeticError, TypeError, ValueError, ZeroDivisionError):
            continue
        if logarithm * base_point == target_point:
            return logarithm
    return None


def _field_to_qadic(value: Any, qadic_field: Any, degree: int) -> Any:
    if degree == 1:
        return qadic_field(ZZ(value))
    coefficients = [ZZ(coefficient) for coefficient in value.polynomial().list()]
    return qadic_field(coefficients)


def _lift_finite_field_point(
    lifted_curve: Any,
    point: Any,
    qadic_field: Any,
    degree: int,
) -> Any:
    if _is_zero(point):
        return lifted_curve(0)
    lifted_x = _field_to_qadic(point[0], qadic_field, degree)
    expected_y = _field_to_qadic(point[1], qadic_field, degree)
    for candidate in lifted_curve.lift_x(lifted_x, all=True):
        try:
            if (candidate[1] - expected_y).valuation() > 0:
                return candidate
        except (AttributeError, ValueError):
            continue
    raise CurveAttackError("could not lift the finite-field point p-adically")


def _qadic_integer_mod(value: Any, prime: int, exponent: int) -> int | None:
    """Extract an ordinary integer modulo p^exponent from a q-adic value."""

    if exponent == 1:
        try:
            return int(ZZ(value % prime)) % prime
        except (ArithmeticError, TypeError, ValueError):
            pass

    try:
        coefficients = value.polynomial().list()
    except AttributeError:
        coefficients = [value]
    if not coefficients:
        return 0

    # A genuine scalar is in Q_p, so every nonconstant unramified-extension
    # coefficient must vanish to the precision needed for this result.
    for coefficient in coefficients[1:]:
        try:
            if not coefficient.is_zero() and coefficient.valuation() < exponent:
                return None
        except AttributeError:
            if coefficient != 0:
                return None

    constant = coefficients[0]
    result = 0
    for position in range(exponent):
        try:
            digit = int(constant.residue()) % prime
        except (AttributeError, ValueError, TypeError):
            return None
        result += digit * prime**position
        constant = (constant - digit) / prime
    return result


def smart_attack(
    base_point: Any,
    target_point: Any,
    *,
    max_tries: int = 8,
    precision: int | None = None,
) -> int | None:
    """Solve a DLP on an anomalous curve using Smart's p-adic attack.

    Prime fields and finite extension fields are supported.  For an extension
    GF(p^n), the base point must have the full anomalous order p^n so that the
    recovered scalar is unambiguous modulo the curve order.
    """

    attempts = _positive_integer(max_tries, "max_tries")
    curve = _same_curve(base_point, target_point)
    if _is_zero(base_point):
        raise ValueError("base point must be nonzero")
    if _is_zero(target_point):
        return 0
    field_order = _finite_field_order(curve)
    if field_order is None:
        raise ValueError("Smart's attack requires a finite-field curve")
    curve_order = _curve_order(curve)
    if curve_order != field_order:
        raise ValueError("Smart's attack requires an anomalous curve")
    if int(base_point.order()) != curve_order:
        raise ValueError("base point must generate the anomalous curve group")

    finite_field = curve.base_ring()
    prime = int(finite_field.characteristic())
    degree = int(finite_field.degree())
    selected_precision = (
        max(20, 2 * degree + 8)
        if precision is None
        else _positive_integer(precision, "precision")
    )
    if selected_precision < 2 * degree + 2:
        raise ValueError(
            f"precision must be at least {2 * degree + 2} for GF(p^{degree})"
        )

    qadic_field = Qq(
        ZZ(field_order),
        prec=selected_precision,
        names="smart_lift_generator",
    )
    original_invariants = tuple(curve.a_invariants())

    for _ in range(attempts):
        lifted_invariants = [
            _field_to_qadic(value, qadic_field, degree)
            + field_order * ZZ.random_element(1, field_order)
            for value in original_invariants
        ]
        try:
            lifted_curve = EllipticCurve(qadic_field, lifted_invariants)
            lifted_base_point = _lift_finite_field_point(
                lifted_curve,
                base_point,
                qadic_field,
                degree,
            )
            lifted_target_point = _lift_finite_field_point(
                lifted_curve,
                target_point,
                qadic_field,
                degree,
            )
            lifted_base = field_order * lifted_base_point
            lifted_target = field_order * lifted_target_point
            if _is_zero(lifted_base) or _is_zero(lifted_target):
                continue
            base_x, base_y = lifted_base.xy()
            target_x, target_y = lifted_target.xy()
            quotient = (target_x / target_y) / (base_x / base_y)
            logarithm = _qadic_integer_mod(quotient, prime, 1)
            if logarithm is None:
                continue

            # Over GF(p^n), the first quotient determines only l modulo p.
            # Recover the remaining base-p digits in the order-p subgroup.
            if degree > 1:
                order_p_base = prime ** (degree - 1) * base_point
                lifted_order_p_base = field_order * _lift_finite_field_point(
                    lifted_curve,
                    order_p_base,
                    qadic_field,
                    degree,
                )
                if _is_zero(lifted_order_p_base):
                    continue
                order_p_x, order_p_y = lifted_order_p_base.xy()
                order_p_parameter = order_p_x / order_p_y

                for position in range(1, degree):
                    correction_point = prime ** (degree - position - 1) * (
                        target_point - logarithm * base_point
                    )
                    if _is_zero(correction_point):
                        continue
                    lifted_correction = (
                        field_order
                        * _lift_finite_field_point(
                            lifted_curve,
                            correction_point,
                            qadic_field,
                            degree,
                        )
                    )
                    correction_x, correction_y = lifted_correction.xy()
                    digit = _qadic_integer_mod(
                        (correction_x / correction_y) / order_p_parameter,
                        prime,
                        1,
                    )
                    if digit is None:
                        raise ArithmeticError("non-scalar q-adic digit")
                    logarithm += prime**position * digit
        except (ArithmeticError, TypeError, ValueError, ZeroDivisionError):
            continue
        if logarithm is not None and logarithm * base_point == target_point:
            return logarithm
    return None


def _lift_residue_point(curve: Any, point: Any) -> Any:
    if _is_zero(point):
        return curve(0)
    try:
        return curve(point)
    except (TypeError, ValueError):
        return curve(ZZ(point[0]), ZZ(point[1]))


def padic_ecdlp(
    base_point: Any,
    target_point: Any,
    *,
    digits_per_step: int = 4,
) -> int:
    """Recover ``l mod p^(k-1)`` on a curve over ``Z/(p^k)``.

    This is not an attack on curves over ``GF(p^k)``.  The returned residue is
    checked by killing the reduction-order component on the original curve;
    it need not be a full discrete logarithm when the curve has an additional
    prime-to-p component.  ``digits_per_step`` trades more, smaller p-adic
    updates for fewer, wider updates without changing the recovered modulus.
    """

    digit_chunk = _positive_integer(digits_per_step, "digits_per_step")
    curve = _same_curve(base_point, target_point)
    if _is_zero(base_point):
        raise ValueError("base point must be nonzero")
    if _is_zero(target_point):
        return 0
    residue_data = _residue_ring_data(curve)
    if residue_data is None:
        raise ValueError("curve must be defined over Z/(p^k) with k >= 2")
    prime, exponent = residue_data

    reduced_curve = curve.change_ring(GF(prime))
    reduction_order = int(reduced_curve.order())
    p_adics = Qp(prime, prec=exponent)
    lifted_curve = curve.change_ring(p_adics)
    lifted_base = reduction_order * _lift_residue_point(lifted_curve, base_point)
    lifted_target = reduction_order * _lift_residue_point(
        lifted_curve,
        target_point,
    )

    logarithm = 0
    place_value = 1
    remaining_digits = exponent - 1
    while remaining_digits:
        if _is_zero(lifted_target):
            break
        if _is_zero(lifted_base):
            raise CurveAttackError("p-adic precision was exhausted too early")
        width = min(digit_chunk, remaining_digits)
        radix = prime**width
        try:
            base_parameter = lifted_base[0] / lifted_base[1]
            target_parameter = lifted_target[0] / lifted_target[1]
            digit = int(ZZ(target_parameter / base_parameter) % radix)
        except (ArithmeticError, TypeError, ValueError, ZeroDivisionError) as error:
            raise CurveAttackError("could not extract the next p-adic digit") from error
        logarithm += place_value * digit
        lifted_target -= digit * lifted_base
        lifted_base *= radix
        place_value *= radix
        remaining_digits -= width
    residual = reduction_order * (target_point - logarithm * base_point)
    if not _is_zero(residual):
        raise CurveAttackError(
            "the recovered residue failed its p-primary subgroup check"
        )
    return logarithm


def parameter_recovery(
    p: int,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
) -> tuple[int, int]:
    """Recover ``a, b`` in ``y^2 = x^3 + a*x + b`` from two points."""

    prime = _positive_integer(p, "p")
    if prime <= 3 or not ZZ(prime).is_prime():
        raise ValueError("p must be a prime greater than three")
    coordinates = [
        _exact_integer(value, name)
        for value, name in (
            (x1, "x1"),
            (y1, "y1"),
            (x2, "x2"),
            (y2, "y2"),
        )
    ]
    x1_value, y1_value, x2_value, y2_value = (
        value % prime for value in coordinates
    )
    difference = (x1_value - x2_value) % prime
    if difference == 0:
        raise ValueError("the two x-coordinates must be distinct modulo p")

    a = (
        (y1_value**2 - y2_value**2 - x1_value**3 + x2_value**3)
        * pow(difference, -1, prime)
    ) % prime
    b = (y1_value**2 - x1_value**3 - a * x1_value) % prime
    for x_value, y_value in ((x1_value, y1_value), (x2_value, y2_value)):
        if (y_value**2 - x_value**3 - a * x_value - b) % prime:
            raise CurveAttackError("recovered parameters do not fit both points")
    return a, b


def solve_linear_congruence(
    a: int,
    b: int,
    modulus: int,
    *,
    max_solutions: int = 100_000,
) -> tuple[int, ...]:
    """Return every canonical solution of ``a*x = b (mod modulus)``."""

    coefficient = _exact_integer(a, "a")
    target = _exact_integer(b, "b")
    normalized_modulus = _positive_integer(modulus, "modulus")
    limit = _positive_integer(max_solutions, "max_solutions")
    divisor = gcd(coefficient, normalized_modulus)
    if target % divisor:
        return ()
    if divisor > limit:
        raise CurveAttackError(
            f"congruence has {divisor} solutions, exceeding max_solutions={limit}"
        )

    reduced_modulus = normalized_modulus // divisor
    if reduced_modulus == 1:
        first = 0
    else:
        first = (
            (target // divisor)
            * pow((coefficient // divisor) % reduced_modulus, -1, reduced_modulus)
        ) % reduced_modulus
    return tuple(first + offset * reduced_modulus for offset in range(divisor))


def recover_reused_nonce(
    n: int,
    m1: int,
    r1: int,
    s1: int,
    m2: int,
    r2: int,
    s2: int,
    *,
    max_candidates: int = 100_000,
) -> tuple[NonceReuseCandidate, ...]:
    """Return all nonce/key pairs consistent with two reused-nonce signatures."""

    order = _positive_integer(n, "n")
    if order <= 2:
        raise ValueError("n must be greater than two")
    limit = _positive_integer(max_candidates, "max_candidates")
    message1, signature_r1, signature_s1, message2, signature_r2, signature_s2 = (
        _exact_integer(value, name) % order
        for value, name in (
            (m1, "m1"),
            (r1, "r1"),
            (s1, "s1"),
            (m2, "m2"),
            (r2, "r2"),
            (s2, "s2"),
        )
    )
    if signature_r1 == 0 or signature_r2 == 0:
        raise ValueError("r values must be nonzero modulo n")
    if signature_r1 != signature_r2:
        raise ValueError("reused ECDSA nonces must have equal r values")
    if signature_s1 == 0 or signature_s2 == 0:
        raise ValueError("s values must be nonzero modulo n")

    nonces = solve_linear_congruence(
        signature_s1 - signature_s2,
        message1 - message2,
        order,
        max_solutions=limit,
    )
    candidates: set[NonceReuseCandidate] = set()
    examined = 0
    for nonce in nonces:
        keys = solve_linear_congruence(
            signature_r1,
            nonce * signature_s1 - message1,
            order,
            max_solutions=limit,
        )
        for private_key in keys:
            examined += 1
            if examined > limit:
                raise CurveAttackError(
                    f"candidate search exceeded max_candidates={limit}"
                )
            if nonce == 0 or private_key == 0:
                continue
            if (
                (signature_s1 * nonce - message1 - signature_r1 * private_key)
                % order
                == 0
                and (
                    signature_s2 * nonce
                    - message2
                    - signature_r2 * private_key
                )
                % order
                == 0
            ):
                candidates.add(NonceReuseCandidate(nonce, private_key))
    return tuple(sorted(candidates))


def recover_nonce(
    n: int,
    m1: int,
    r1: int,
    s1: int,
    m2: int,
    r2: int,
    s2: int,
    *,
    max_candidates: int = 100_000,
) -> Iterator[tuple[int, int]]:
    """Compatibility generator yielding ``(nonce, private_key)`` tuples."""

    for candidate in recover_reused_nonce(
        n,
        m1,
        r1,
        s1,
        m2,
        r2,
        s2,
        max_candidates=max_candidates,
    ):
        yield candidate.nonce, candidate.private_key


def singular_curve_attack(
    p: int,
    a2: int,
    a4: int,
    a6: int,
    Gx: int,
    Gy: int,
    Px: int,
    Py: int,
) -> int:
    """Solve a DLP on a cuspidal or nodal singular Weierstrass cubic.

    Nonsplit nodes are handled in ``GF(p^2)``; split nodes stay in ``GF(p)``.
    The result is verified in the corresponding additive or multiplicative
    parameter group before being returned.
    """

    prime = _positive_integer(p, "p")
    if prime <= 3 or not ZZ(prime).is_prime():
        raise ValueError("p must be a prime greater than three")
    field = GF(prime)
    coefficient_a2, coefficient_a4, coefficient_a6 = (
        field(_exact_integer(value, name))
        for value, name in ((a2, "a2"), (a4, "a4"), (a6, "a6"))
    )
    base_x, base_y, target_x, target_y = (
        field(_exact_integer(value, name))
        for value, name in (
            (Gx, "Gx"),
            (Gy, "Gy"),
            (Px, "Px"),
            (Py, "Py"),
        )
    )

    def cubic(x_value: Any) -> Any:
        return (
            x_value**3
            + coefficient_a2 * x_value**2
            + coefficient_a4 * x_value
            + coefficient_a6
        )

    if base_y**2 != cubic(base_x):
        raise ValueError("base point is not on the singular cubic")
    if target_y**2 != cubic(target_x):
        raise ValueError("target point is not on the singular cubic")

    polynomial_ring = PolynomialRing(field, "singular_x")
    variable = polynomial_ring.gen()
    polynomial = (
        variable**3
        + coefficient_a2 * variable**2
        + coefficient_a4 * variable
        + coefficient_a6
    )
    roots = polynomial.roots()
    if sum(multiplicity for _root, multiplicity in roots) != 3:
        raise ValueError("the supplied cubic is not singular over GF(p)")

    repeated = [
        (root, multiplicity)
        for root, multiplicity in roots
        if multiplicity >= 2
    ]
    if len(repeated) != 1:
        raise ValueError("expected exactly one repeated root")
    alpha, multiplicity = repeated[0]
    if (base_x == alpha and base_y == 0) or (
        target_x == alpha and target_y == 0
    ):
        raise ValueError("the singular point is not part of the curve group")

    if multiplicity == 3:
        try:
            base_parameter = (base_x - alpha) / base_y
            target_parameter = (target_x - alpha) / target_y
        except ZeroDivisionError as error:
            raise ValueError("cuspidal parameters require nonzero y-coordinates") from error
        if base_parameter == 0:
            raise ValueError("base point maps to the additive identity")
        logarithm = int(target_parameter / base_parameter)
        if field(logarithm) * base_parameter != target_parameter:
            raise CurveAttackError("the additive singular-curve result failed verification")
        return logarithm

    if multiplicity != 2 or len(roots) != 2:
        raise ValueError("expected a cusp or an ordinary node")
    beta = next(root for root, _root_multiplicity in roots if root != alpha)
    tangent_square = alpha - beta
    if tangent_square.is_square():
        parameter_field = field
        tangent = tangent_square.sqrt()
    else:
        parameter_field = GF(prime**2, names="node_extension_generator")
        tangent = parameter_field(tangent_square).sqrt()

    alpha_extended = parameter_field(alpha)

    def node_parameter(x_value: Any, y_value: Any) -> Any:
        x_extended = parameter_field(x_value)
        y_extended = parameter_field(y_value)
        numerator = y_extended + tangent * (x_extended - alpha_extended)
        denominator = y_extended - tangent * (x_extended - alpha_extended)
        if numerator == 0 or denominator == 0:
            raise ValueError("point lies on an excluded tangent through the node")
        return numerator / denominator

    base_parameter = node_parameter(base_x, base_y)
    target_parameter = node_parameter(target_x, target_y)
    try:
        logarithm = int(target_parameter.log(base_parameter)) % int(
            base_parameter.multiplicative_order()
        )
    except (ArithmeticError, ValueError) as error:
        raise CurveAttackError(
            "target is not in the multiplicative subgroup generated by base"
        ) from error
    if base_parameter**logarithm != target_parameter:
        raise CurveAttackError("the nodal singular-curve result failed verification")
    return logarithm
