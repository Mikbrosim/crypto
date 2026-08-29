"""Recover an unknown modulus and polynomial from exact modular samples.

For samples ``y_i = f(x_i) mod m`` with ``deg(f) <= d``, every primitive
integer relation that annihilates the columns ``1, x, ..., x**d`` produces
an integer multiple of ``m`` when applied to the sample values.  Consecutive
windows of ``d + 2`` samples provide such relations; their GCD is usually the
hidden modulus, but can retain an accidental cofactor.

The distinction between a *modulus multiple* and a uniquely identified prime
modulus is deliberate.  Prime recovery validates every candidate and raises
on ambiguity instead of guessing that the largest factor is the right one.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from math import comb, gcd, lcm
from operator import index as integer_index
from typing import Literal

from ._number_theory import is_probable_prime


__all__ = [
    "AmbiguousModulusError",
    "PolynomialRecoveryError",
    "polynomial_relation_multiple",
    "recover_modular_polynomial",
    "recover_modulus_multiple",
    "recover_modulus_of_polynomial_fast",
    "recover_modulus_of_polynomial_slow",
    "recover_prime_modulus",
]


RelationMethod = Literal["auto", "barycentric", "finite_difference"]


class PolynomialRecoveryError(RuntimeError):
    """Raised when the supplied samples do not determine a requested result."""


class AmbiguousModulusError(PolynomialRecoveryError):
    """Raised when more than one prime modulus is consistent with the data."""

    def __init__(self, multiple: int, candidates: Sequence[int]) -> None:
        normalized_candidates = tuple(sorted({int(value) for value in candidates}))
        super().__init__(
            f"relation GCD {multiple} admits multiple prime moduli: "
            f"{normalized_candidates}; supply tighter modulus bounds or more samples"
        )
        self.multiple = int(multiple)
        self.candidates = normalized_candidates


def _exact_integer(value: object, name: str) -> int:
    """Return an exact Python integer without silently truncating values."""

    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer, not bool")
    try:
        return integer_index(value)
    except TypeError as error:
        raise TypeError(f"{name} must be an integer") from error


def _nonnegative_integer(value: object, name: str) -> int:
    normalized = _exact_integer(value, name)
    if normalized < 0:
        raise ValueError(f"{name} must be non-negative")
    return normalized


def _positive_integer(value: object, name: str) -> int:
    normalized = _exact_integer(value, name)
    if normalized <= 0:
        raise ValueError(f"{name} must be positive")
    return normalized


def _normalize_samples(
    samples: Iterable[tuple[int, int]],
) -> tuple[tuple[int, int], ...]:
    """Normalize an iterable of coordinate pairs to exact Python integers."""

    try:
        raw_samples = tuple(samples)
    except TypeError as error:
        raise TypeError("samples must be an iterable of (x, y) pairs") from error

    normalized: list[tuple[int, int]] = []
    for sample_index, sample in enumerate(raw_samples):
        try:
            x_value, y_value = sample
        except (TypeError, ValueError) as error:
            raise TypeError(
                f"samples[{sample_index}] must be a pair of integers"
            ) from error
        normalized.append(
            (
                _exact_integer(x_value, f"samples[{sample_index}][0]"),
                _exact_integer(y_value, f"samples[{sample_index}][1]"),
            )
        )
    return tuple(normalized)


def _validate_relation_method(method: object) -> RelationMethod:
    if method not in ("auto", "barycentric", "finite_difference"):
        raise ValueError(
            "method must be 'auto', 'barycentric', or 'finite_difference'"
        )
    return method  # type: ignore[return-value]


def _has_equal_spacing(x_values: Sequence[int]) -> bool:
    if len(x_values) < 2:
        return False
    step = x_values[1] - x_values[0]
    return step != 0 and all(
        right - left == step
        for left, right in zip(x_values[:-1], x_values[1:], strict=True)
    )


def _finite_difference_multiple(y_values: Sequence[int]) -> int:
    """Return the highest finite difference of the supplied values."""

    order = len(y_values) - 1
    relation = sum(
        (-1) ** (order - position) * comb(order, position) * value
        for position, value in enumerate(y_values)
    )
    return abs(relation)


def _barycentric_multiple(samples: Sequence[tuple[int, int]]) -> int:
    """Apply a primitive integral barycentric kernel vector to samples."""

    x_values = [x_value for x_value, _ in samples]
    denominators: list[int] = []
    for position, x_value in enumerate(x_values):
        denominator = 1
        for other_position, other_x in enumerate(x_values):
            if position != other_position:
                denominator *= x_value - other_x
        if denominator == 0:
            raise ValueError("relation windows require distinct x-values")
        denominators.append(denominator)

    common_denominator = lcm(*(abs(value) for value in denominators))
    weights = [common_denominator // value for value in denominators]

    # The LCM construction is primitive in exact arithmetic.  Normalize again
    # defensively so future changes cannot introduce a systematic GCD cofactor.
    weight_gcd = 0
    for weight in weights:
        weight_gcd = gcd(weight_gcd, abs(weight))
    weights = [weight // weight_gcd for weight in weights]

    relation = sum(
        weight * y_value
        for weight, (_x_value, y_value) in zip(weights, samples, strict=True)
    )
    return abs(relation)


def polynomial_relation_multiple(
    window: Iterable[tuple[int, int]],
    *,
    method: RelationMethod = "auto",
) -> int:
    """Return an exact modulus multiple from one polynomial sample window.

    A window of ``d + 2`` samples annihilates every polynomial of degree at
    most ``d``.  This function does not need ``d`` explicitly because it is
    determined by the window length.  A zero result is valid but contains no
    information about the modulus.

    Equally spaced inputs use a compact finite-difference relation.  Arbitrary
    distinct inputs use the equivalent primitive barycentric relation.
    """

    normalized = _normalize_samples(window)
    if len(normalized) < 2:
        raise ValueError("a relation window must contain at least two samples")

    x_values = [x_value for x_value, _ in normalized]
    if len(set(x_values)) != len(x_values):
        raise ValueError("relation windows require distinct x-values")

    normalized_method = _validate_relation_method(method)
    equal_spacing = _has_equal_spacing(x_values)
    if normalized_method == "finite_difference" and not equal_spacing:
        raise ValueError(
            "finite_difference requires equally spaced x-values in sample order"
        )
    if normalized_method == "finite_difference" or (
        normalized_method == "auto" and equal_spacing
    ):
        return _finite_difference_multiple(
            [y_value for _x_value, y_value in normalized]
        )
    return _barycentric_multiple(normalized)


def _relation_multiples(
    samples: Iterable[tuple[int, int]],
    degree: int,
    *,
    max_relations: int | None,
    method: RelationMethod,
) -> tuple[tuple[tuple[int, int], ...], tuple[int, ...]]:
    normalized_samples = _normalize_samples(samples)
    normalized_degree = _nonnegative_integer(degree, "degree")
    normalized_method = _validate_relation_method(method)

    x_values = [x_value for x_value, _ in normalized_samples]
    if len(set(x_values)) != len(x_values):
        raise ValueError("modulus recovery requires distinct integer x-values")

    window_length = normalized_degree + 2
    available_relations = len(normalized_samples) - window_length + 1
    if available_relations < 1:
        raise ValueError(
            f"degree {normalized_degree} requires at least "
            f"{window_length} samples for one modulus multiple"
        )

    if max_relations is None:
        relation_count = available_relations
    else:
        relation_count = _positive_integer(max_relations, "max_relations")
        relation_count = min(relation_count, available_relations)

    multiples = tuple(
        polynomial_relation_multiple(
            normalized_samples[offset : offset + window_length],
            method=normalized_method,
        )
        for offset in range(relation_count)
    )
    return normalized_samples, multiples


def _gcd_of_relations(multiples: Sequence[int]) -> int:
    common_multiple = 0
    for multiple in multiples:
        common_multiple = gcd(common_multiple, multiple)

    if common_multiple == 0:
        raise PolynomialRecoveryError(
            "all polynomial relations are zero; the samples reveal no modulus"
        )
    if common_multiple == 1:
        raise PolynomialRecoveryError(
            "the polynomial relations have GCD 1; check the degree and samples"
        )
    return common_multiple


def recover_modulus_multiple(
    samples: Iterable[tuple[int, int]],
    degree: int,
    *,
    max_relations: int | None = None,
    method: RelationMethod = "auto",
) -> int:
    """Return the GCD of available relation multiples.

    The result is guaranteed to be divisible by any modulus compatible with
    the supplied degree bound.  It equals the hidden modulus only when the
    unrelated integer quotients of the relations are coprime.  By default all
    available consecutive windows are used; ``max_relations`` is the explicit
    effort/reliability tradeoff.
    """

    _normalized_samples, multiples = _relation_multiples(
        samples,
        degree,
        max_relations=max_relations,
        method=method,
    )
    return _gcd_of_relations(multiples)


def _coalesce_modular_samples(
    samples: Sequence[tuple[int, int]],
    modulus: int,
) -> tuple[tuple[int, int], ...]:
    """Merge repeated modular x-values and reject contradictory values."""

    by_x: dict[int, int] = {}
    order: list[int] = []
    for x_value, y_value in samples:
        reduced_x = x_value % modulus
        reduced_y = y_value % modulus
        if reduced_x in by_x:
            if by_x[reduced_x] != reduced_y:
                raise PolynomialRecoveryError(
                    f"x-values congruent to {reduced_x} modulo {modulus} "
                    "have conflicting outputs"
                )
            continue
        by_x[reduced_x] = reduced_y
        order.append(reduced_x)
    return tuple((x_value, by_x[x_value]) for x_value in order)


def _newton_interpolation(
    samples: Sequence[tuple[int, int]],
    modulus: int,
) -> list[int]:
    """Interpolate over Z/mZ when all divided-difference denominators are units."""

    x_values = [x_value for x_value, _ in samples]
    differences = [y_value for _x_value, y_value in samples]
    newton_coefficients = [differences[0]]

    for order in range(1, len(samples)):
        next_differences: list[int] = []
        for start in range(len(samples) - order):
            denominator = (x_values[start + order] - x_values[start]) % modulus
            try:
                inverse = pow(denominator, -1, modulus)
            except ValueError as error:
                raise PolynomialRecoveryError(
                    "interpolation requires every selected x-value difference "
                    f"to be invertible modulo {modulus}"
                ) from error
            next_differences.append(
                (differences[start + 1] - differences[start])
                * inverse
                % modulus
            )
        differences = next_differences
        newton_coefficients.append(differences[0])

    coefficients = [0]
    basis = [1]
    for position, newton_coefficient in enumerate(newton_coefficients):
        if len(coefficients) < len(basis):
            coefficients.extend([0] * (len(basis) - len(coefficients)))
        for exponent, basis_coefficient in enumerate(basis):
            coefficients[exponent] = (
                coefficients[exponent]
                + newton_coefficient * basis_coefficient
            ) % modulus

        if position + 1 < len(newton_coefficients):
            root = x_values[position]
            expanded_basis = [0] * (len(basis) + 1)
            for exponent, basis_coefficient in enumerate(basis):
                expanded_basis[exponent] = (
                    expanded_basis[exponent] - root * basis_coefficient
                ) % modulus
                expanded_basis[exponent + 1] = (
                    expanded_basis[exponent + 1] + basis_coefficient
                ) % modulus
            basis = expanded_basis

    while len(coefficients) > 1 and coefficients[-1] == 0:
        coefficients.pop()
    return coefficients


def _evaluate_modular_polynomial(
    coefficients: Sequence[int],
    x_value: int,
    modulus: int,
) -> int:
    result = 0
    for coefficient in reversed(coefficients):
        result = (result * x_value + coefficient) % modulus
    return result


def recover_modular_polynomial(
    samples: Iterable[tuple[int, int]],
    modulus: int,
    degree: int | None = None,
    *,
    verify: bool = True,
) -> list[int]:
    """Recover ascending polynomial coefficients modulo ``modulus``.

    When ``degree`` is supplied, exactly ``degree + 1`` distinct modular
    inputs determine the polynomial and every remaining sample is used only
    for verification.  This is the recommended correctness-checking mode.

    Composite moduli are supported when the relevant x-value differences are
    units.  Consistent repeated inputs are coalesced; contradictory repeats
    are rejected.
    """

    normalized_samples = _normalize_samples(samples)
    normalized_modulus = _positive_integer(modulus, "modulus")
    if normalized_modulus < 2:
        raise ValueError("modulus must be at least 2")
    if not isinstance(verify, bool):
        raise TypeError("verify must be a bool")
    if not normalized_samples:
        raise ValueError("at least one sample is required")

    modular_samples = _coalesce_modular_samples(
        normalized_samples,
        normalized_modulus,
    )
    if degree is None:
        interpolation_count = len(modular_samples)
        normalized_degree = interpolation_count - 1
    else:
        normalized_degree = _nonnegative_integer(degree, "degree")
        interpolation_count = normalized_degree + 1
        if len(modular_samples) < interpolation_count:
            raise ValueError(
                f"degree {normalized_degree} requires at least "
                f"{interpolation_count} distinct inputs modulo the modulus"
            )

    interpolation_samples = modular_samples[:interpolation_count]
    coefficients = _newton_interpolation(
        interpolation_samples,
        normalized_modulus,
    )
    if len(coefficients) - 1 > normalized_degree:
        raise PolynomialRecoveryError(
            "interpolation produced a polynomial above the requested degree"
        )

    if verify:
        for x_value, y_value in modular_samples:
            if (
                _evaluate_modular_polynomial(
                    coefficients,
                    x_value,
                    normalized_modulus,
                )
                != y_value
            ):
                raise PolynomialRecoveryError(
                    f"no polynomial of degree at most {normalized_degree} "
                    "matches every sample"
                )
    return coefficients


def _fallback_prime_divisors(value: int) -> tuple[int, ...]:
    """Factor small values when the module is exercised outside Sage."""

    if is_probable_prime(value):
        return (value,)
    if value.bit_length() > 48:
        raise RuntimeError(
            "Sage is required to factor a large composite modulus multiple"
        )

    remaining = value
    divisors: list[int] = []
    candidate = 2
    while candidate * candidate <= remaining:
        if remaining % candidate == 0:
            divisors.append(candidate)
            while remaining % candidate == 0:
                remaining //= candidate
        candidate = 3 if candidate == 2 else candidate + 2
    if remaining > 1:
        divisors.append(remaining)
    return tuple(divisors)


def _prime_divisors(value: int) -> tuple[int, ...]:
    """Return proven prime divisors with Sage and a small local fallback."""

    try:
        from sage.all import ZZ
    except ModuleNotFoundError:
        return _fallback_prime_divisors(value)

    integer = ZZ(value)
    if integer.is_prime():
        return (value,)
    return tuple(int(prime) for prime, _exponent in integer.factor())


def recover_prime_modulus(
    samples: Iterable[tuple[int, int]],
    degree: int,
    *,
    min_modulus: int | None = None,
    max_modulus: int | None = None,
    assume_reduced: bool = True,
    max_relations: int | None = None,
    method: RelationMethod = "auto",
) -> int:
    """Recover a uniquely determined prime modulus and verify every sample.

    At least two nonzero relations are required.  ``assume_reduced=True`` uses
    the common challenge assumption ``0 <= y_i < p`` to reject impossible
    prime factors.  Explicit inclusive bounds can encode known modulus size.
    If multiple factors remain valid, :class:`AmbiguousModulusError` is raised.
    """

    if not isinstance(assume_reduced, bool):
        raise TypeError("assume_reduced must be a bool")

    normalized_samples, multiples = _relation_multiples(
        samples,
        degree,
        max_relations=max_relations,
        method=method,
    )
    nonzero_relations = tuple(value for value in multiples if value != 0)
    if len(nonzero_relations) < 2:
        raise PolynomialRecoveryError(
            "prime recovery requires at least two nonzero relations; "
            "supply more samples or use recover_modulus_multiple"
        )
    relation_gcd = _gcd_of_relations(nonzero_relations)

    lower_bound = 2
    if min_modulus is not None:
        lower_bound = max(
            lower_bound,
            _positive_integer(min_modulus, "min_modulus"),
        )
    if assume_reduced:
        y_values = [y_value for _x_value, y_value in normalized_samples]
        if any(y_value < 0 for y_value in y_values):
            raise ValueError(
                "assume_reduced=True requires non-negative sample outputs"
            )
        lower_bound = max(lower_bound, max(y_values) + 1)

    if max_modulus is None:
        upper_bound = relation_gcd
    else:
        upper_bound = _positive_integer(max_modulus, "max_modulus")
    if upper_bound < lower_bound:
        raise ValueError("max_modulus is smaller than the effective lower bound")

    valid_candidates: list[int] = []
    for candidate in _prime_divisors(relation_gcd):
        if not lower_bound <= candidate <= upper_bound:
            continue
        try:
            recover_modular_polynomial(
                normalized_samples,
                candidate,
                degree,
                verify=True,
            )
        except (PolynomialRecoveryError, ValueError):
            continue
        valid_candidates.append(candidate)

    if not valid_candidates:
        raise PolynomialRecoveryError(
            f"no prime factor of relation GCD {relation_gcd} satisfies the "
            "bounds and all polynomial samples"
        )
    if len(valid_candidates) > 1:
        raise AmbiguousModulusError(relation_gcd, valid_candidates)
    return valid_candidates[0]


def recover_modulus_of_polynomial_fast(
    samples: Iterable[tuple[int, int]],
    degree: int,
    extra_gcds: bool = True,
    *,
    min_modulus: int | None = None,
    max_modulus: int | None = None,
) -> int:
    """Compatibility wrapper for the original fast prime-modulus API."""

    if not isinstance(extra_gcds, bool):
        raise TypeError("extra_gcds must be a bool")
    return recover_prime_modulus(
        samples,
        degree,
        min_modulus=min_modulus,
        max_modulus=max_modulus,
        max_relations=None if extra_gcds else 2,
    )


def recover_modulus_of_polynomial_slow(
    samples: Iterable[tuple[int, int]],
    verbose: bool = False,
    *,
    min_modulus: int | None = None,
    max_modulus: int | None = None,
) -> int:
    """Compatibility wrapper that infers the degree from ``d + 3`` samples.

    The former rational-matrix implementation computed the same Vandermonde
    kernel relations less directly.  This wrapper now uses the exact primitive
    relation implementation while preserving the old call shape.
    """

    if not isinstance(verbose, bool):
        raise TypeError("verbose must be a bool")
    normalized_samples = _normalize_samples(samples)
    if len(normalized_samples) < 3:
        raise ValueError("at least three samples are required to infer a degree")
    degree = len(normalized_samples) - 3
    if verbose:
        print(
            f"[polynomial_recovery] using degree {degree} and two exact relations"
        )
    return recover_prime_modulus(
        normalized_samples,
        degree,
        min_modulus=min_modulus,
        max_modulus=max_modulus,
        max_relations=2,
    )
