"""Validated miscellaneous algebra and number-theory helpers.

This module collects the genuinely reusable parts of the original ``other.py``:
generalized CRT, modular square roots, Pohlig--Hellman, a small coefficient-list
polynomial type, affine maps over GF(2), and strict extraction from simple
Groebner bases.  Polynomial-modulus recovery is re-exported from its dedicated
module instead of being implemented twice.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Sequence
from math import gcd, isqrt
from operator import index as integer_index
from random import Random
from typing import Any, Literal

from sage.all import GF, Ideal, ZZ, discrete_log_lambda, matrix, vector

from ._number_theory import discrete_log_bsgs
from .polynomial_recovery import (
    recover_modular_polynomial,
    recover_modulus_of_polynomial_fast,
    recover_modulus_of_polynomial_slow,
)


__all__ = [
    "Polynomial",
    "construct_GF2_matrix",
    "crt",
    "ideal_groebner",
    "legendre",
    "modular_square_roots",
    "pack_words",
    "pohlig_hellman",
    "recover_modular_polynomial",
    "recover_modulus_of_polynomial_fast",
    "recover_modulus_of_polynomial_slow",
    "tonelli_shanks",
    "unpack_words",
    "wrapper_maker",
]


DiscreteLogMethod = Literal["auto", "bsgs", "lambda"]


def _exact_integer(value: object, name: str) -> int:
    """Return an exact integer without silently truncating values."""

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


def _nonnegative_integer(value: object, name: str) -> int:
    normalized = _exact_integer(value, name)
    if normalized < 0:
        raise ValueError(f"{name} must be non-negative")
    return normalized


def _integer_tuple(values: Iterable[int], name: str) -> tuple[int, ...]:
    try:
        raw_values = tuple(values)
    except TypeError as error:
        raise TypeError(f"{name} must be an iterable of integers") from error
    return tuple(
        _exact_integer(value, f"{name}[{position}]")
        for position, value in enumerate(raw_values)
    )


def _legendre_prime(value: int, prime: int) -> int:
    residue = pow(value % prime, (prime - 1) // 2, prime)
    return -1 if residue == prime - 1 else residue


def legendre(a: int, p: int) -> int:
    """Return the Legendre symbol ``(a | p)`` as ``-1``, ``0``, or ``1``."""

    value = _exact_integer(a, "a")
    prime = _positive_integer(p, "p")
    if prime == 2 or not ZZ(prime).is_prime():
        raise ValueError("p must be an odd prime")
    return _legendre_prime(value, prime)


def crt(a: Iterable[int], n: Iterable[int]) -> int:
    """Solve a compatible system of integer congruences.

    Moduli need not be coprime.  The least non-negative solution modulo their
    least common multiple is returned.  Inconsistent systems raise rather than
    producing a value that satisfies only part of the input.
    """

    residues = _integer_tuple(a, "a")
    moduli = _integer_tuple(n, "n")
    if not residues or len(residues) != len(moduli):
        raise ValueError("a and n must be non-empty and have the same length")
    if any(modulus <= 0 for modulus in moduli):
        raise ValueError("CRT moduli must be positive")

    result = residues[0] % moduli[0]
    combined_modulus = moduli[0]
    for residue, modulus in zip(residues[1:], moduli[1:], strict=True):
        common_divisor = gcd(combined_modulus, modulus)
        difference = residue - result
        if difference % common_divisor:
            raise ValueError("the congruences are inconsistent")

        left = combined_modulus // common_divisor
        right = modulus // common_divisor
        if right == 1:
            adjustment = 0
        else:
            adjustment = (
                difference // common_divisor * pow(left, -1, right)
            ) % right
        result += combined_modulus * adjustment
        combined_modulus *= right
        result %= combined_modulus
    return result


def _tonelli_shanks_prime(value: int, prime: int) -> tuple[int, ...]:
    """Return every square root modulo a validated prime."""

    if prime == 2:
        return (value % 2,)
    residue = value % prime
    if residue == 0:
        return (0,)
    if _legendre_prime(residue, prime) != 1:
        return ()
    if prime % 4 == 3:
        root = pow(residue, (prime + 1) // 4, prime)
        return tuple(sorted({root, (-root) % prime}))

    odd_part = prime - 1
    power_of_two = 0
    while odd_part % 2 == 0:
        odd_part //= 2
        power_of_two += 1

    nonresidue = 2
    while _legendre_prime(nonresidue, prime) != -1:
        nonresidue += 1
    multiplier = pow(nonresidue, odd_part, prime)
    root = pow(residue, (odd_part + 1) // 2, prime)
    remainder = pow(residue, odd_part, prime)
    exponent = power_of_two

    while remainder != 1:
        squared = remainder
        for position in range(1, exponent):
            squared = squared * squared % prime
            if squared == 1:
                break
        else:
            raise ArithmeticError("Tonelli--Shanks invariant was violated")
        correction = pow(multiplier, 1 << (exponent - position - 1), prime)
        root = root * correction % prime
        multiplier = correction * correction % prime
        remainder = remainder * multiplier % prime
        exponent = position
    return tuple(sorted({root, (-root) % prime}))


def modular_square_roots(
    value: int,
    prime_factors: Iterable[int],
) -> tuple[int, ...]:
    """Return all roots modulo a product of distinct prime factors."""

    normalized_value = _exact_integer(value, "value")
    factors = _integer_tuple(prime_factors, "prime_factors")
    if not factors:
        raise ValueError("prime_factors must not be empty")
    for position, factor in enumerate(factors):
        if factor <= 1 or not ZZ(factor).is_prime():
            raise ValueError(f"prime_factors[{position}] must be prime")
        if any(gcd(factor, previous) != 1 for previous in factors[:position]):
            raise ValueError("prime_factors must be pairwise coprime")

    roots_by_factor = [
        _tonelli_shanks_prime(normalized_value, factor) for factor in factors
    ]
    if any(not roots for roots in roots_by_factor):
        return ()

    combinations: list[tuple[int, ...]] = [()]
    for roots in roots_by_factor:
        combinations = [
            prefix + (root,)
            for prefix in combinations
            for root in roots
        ]
    return tuple(
        sorted({crt(root_tuple, factors) for root_tuple in combinations})
    )


def tonelli_shanks(
    n: int,
    p: int | Iterable[int],
) -> tuple[int, ...]:
    """Return modular square roots for a prime or factored squarefree modulus.

    Passing an integer invokes Tonelli--Shanks for a proven prime.  Passing an
    iterable of pairwise-coprime primes preserves the original composite-mode
    API and combines each prime's roots with CRT.
    """

    value = _exact_integer(n, "n")
    try:
        prime = _exact_integer(p, "p")
    except TypeError:
        return modular_square_roots(value, p)  # type: ignore[arg-type]
    if prime <= 1 or not ZZ(prime).is_prime():
        raise ValueError("integer p must be prime")
    return _tonelli_shanks_prime(value, prime)


def _prime_power_components(factors: Sequence[int]) -> tuple[int, ...]:
    exponents: dict[int, int] = {}
    for factor in factors:
        for prime, exponent in ZZ(factor).factor():
            prime_value = int(prime)
            exponents[prime_value] = exponents.get(prime_value, 0) + int(exponent)
    return tuple(
        prime**exponents[prime] for prime in sorted(exponents)
    )


def pohlig_hellman(
    g: int,
    h: int,
    p: int,
    phi_facs: Iterable[int],
    q: int | None = None,
    *,
    method: DiscreteLogMethod = "auto",
    max_bsgs_table: int = 1_000_000,
) -> int:
    """Solve ``g^x = h mod p`` for a fully factored smooth order.

    Repeated prime factors are merged into coprime prime-power components, and
    each component is recovered one base-prime digit at a time.  ``auto`` uses
    BSGS while its digit table fits ``max_bsgs_table`` and Sage's Pollard lambda
    implementation otherwise.
    """

    base = _exact_integer(g, "g")
    target = _exact_integer(h, "h")
    prime = _positive_integer(p, "p")
    if not ZZ(prime).is_prime():
        raise ValueError("p must be prime")
    supplied_factors = _integer_tuple(phi_facs, "phi_facs")
    if not supplied_factors or any(factor <= 1 for factor in supplied_factors):
        raise ValueError("phi_facs must contain integers greater than one")
    factor_product = 1
    for factor in supplied_factors:
        factor_product *= factor
    order = factor_product if q is None else _positive_integer(q, "q")
    if factor_product != order:
        raise ValueError("the product of phi_facs must equal q")
    if method not in ("auto", "bsgs", "lambda"):
        raise ValueError("method must be 'auto', 'bsgs', or 'lambda'")
    table_limit = _positive_integer(max_bsgs_table, "max_bsgs_table")

    base %= prime
    target %= prime
    if base == 0 or target == 0:
        raise ValueError("g and h must be nonzero modulo p")
    components = _prime_power_components(supplied_factors)
    component_data = []
    for component in components:
        factorization = tuple(ZZ(component).factor())
        if len(factorization) != 1:
            raise ArithmeticError("internal factor merging did not yield prime powers")
        factor_prime, exponent = factorization[0]
        component_data.append((int(factor_prime), int(exponent), component))
    if pow(base, order, prime) != 1 or any(
        pow(base, order // factor_prime, prime) == 1
        for factor_prime, _exponent, _component in component_data
    ):
        raise ValueError("g does not have the declared order q")
    if pow(target, order, prime) != 1:
        raise ValueError("h is not in the declared subgroup")

    residues: list[int] = []
    finite_field = GF(prime)
    for factor_prime, exponent, component in component_data:
        digit_base = pow(base, order // factor_prime, prime)
        component_logarithm = 0
        selected_method = method
        if selected_method == "auto":
            selected_method = (
                "bsgs"
                if isqrt(factor_prime) + 1 <= table_limit
                else "lambda"
            )

        for position in range(exponent):
            correction = (
                target * pow(pow(base, component_logarithm, prime), -1, prime)
            ) % prime
            digit_target = pow(
                correction,
                order // factor_prime ** (position + 1),
                prime,
            )
            if digit_target == 1:
                digit = 0
            elif selected_method == "bsgs":
                digit = discrete_log_bsgs(
                    digit_base,
                    digit_target,
                    prime,
                    factor_prime,
                )
            else:
                digit = int(
                    discrete_log_lambda(
                        finite_field(digit_target),
                        finite_field(digit_base),
                        (0, factor_prime - 1),
                    )
                )
            digit %= factor_prime
            if pow(digit_base, digit, prime) != digit_target:
                raise ArithmeticError("a Pohlig--Hellman digit failed verification")
            component_logarithm += digit * factor_prime**position

        component_base = pow(base, order // component, prime)
        component_target = pow(target, order // component, prime)
        if pow(component_base, component_logarithm, prime) != component_target:
            raise ArithmeticError(
                "a prime-power discrete logarithm failed verification"
            )
        residues.append(component_logarithm)

    result = crt(residues, components) % order
    if pow(base, result, prime) != target:
        #raise ValueError("h is not generated by g")
        print("[!] WARNING: h is not generated by g")
    return result


def _trim_coefficients(coefficients: Sequence[Any]) -> tuple[Any, ...]:
    normalized = list(coefficients)
    while len(normalized) > 1:
        try:
            is_zero = bool(normalized[-1] == 0)
        except (TypeError, ValueError):
            is_zero = False
        if not is_zero:
            break
        normalized.pop()
    return tuple(normalized or [0])


def _concrete_integer(value: Any) -> int | None:
    try:
        return _exact_integer(value, "coefficient")
    except TypeError:
        return None


class Polynomial:
    """A small immutable ascending-coefficient polynomial container."""

    __slots__ = ("_coefficients",)

    def __init__(self, coefficients: Iterable[Any]) -> None:
        try:
            values = tuple(coefficients)
        except TypeError as error:
            raise TypeError("coefficients must be iterable") from error
        self._coefficients = _trim_coefficients(values)

    @property
    def coefficients(self) -> tuple[Any, ...]:
        """Return immutable ascending coefficients."""

        return self._coefficients

    @property
    def coef(self) -> list[Any]:
        """Compatibility copy of the original mutable ``coef`` attribute."""

        return list(self._coefficients)

    def __iter__(self) -> Iterator[Any]:
        return iter(self._coefficients)

    def __len__(self) -> int:
        return len(self._coefficients)

    def __repr__(self) -> str:
        return f"Polynomial({list(self._coefficients)!r})"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Polynomial) and self.coefficients == other.coefficients

    def _binary_coefficients(self, other: object) -> tuple[list[Any], list[Any]]:
        if not isinstance(other, Polynomial):
            raise TypeError("polynomial arithmetic requires another Polynomial")
        length = max(len(self), len(other))
        left = list(self.coefficients) + [0] * (length - len(self))
        right = list(other.coefficients) + [0] * (length - len(other))
        return left, right

    def __add__(self, other: object) -> "Polynomial":
        left, right = self._binary_coefficients(other)
        return type(self)(a + b for a, b in zip(left, right, strict=True))

    def __sub__(self, other: object) -> "Polynomial":
        left, right = self._binary_coefficients(other)
        return type(self)(a - b for a, b in zip(left, right, strict=True))

    def __mul__(self, other: object) -> "Polynomial":
        if not isinstance(other, Polynomial):
            raise TypeError("polynomial arithmetic requires another Polynomial")
        result = [0] * (len(self) + len(other) - 1)
        for left_position, left in enumerate(self.coefficients):
            for right_position, right in enumerate(other.coefficients):
                result[left_position + right_position] += left * right
        return type(self)(result)

    @classmethod
    def from_list(cls, coefficients: Iterable[Any]) -> "Polynomial":
        return cls(coefficients)

    @classmethod
    def from_gf(cls, polynomial: Iterable[Any]) -> "Polynomial":
        return cls(int(coefficient) for coefficient in polynomial)

    @staticmethod
    def _symbolic_quotient(quotients: list[Any], operation: str) -> Any:
        if not quotients:
            raise ValueError(f"not enough symbolic quotients for {operation}")
        return quotients.pop()

    @staticmethod
    def _check_quotients_consumed(quotients: list[Any], operation: str) -> None:
        if quotients:
            raise ValueError(f"unused symbolic quotients for {operation}")

    def reduce_by_order(
        self,
        field_order: int,
        field_degree: int,
        k: Iterable[Any],
    ) -> "Polynomial":
        """Reduce the first ``field_degree`` coefficients modulo field order.

        Symbolic coefficients use quotient variables taken from the end of a
        copied ``k`` sequence, preserving the legacy ordering without mutating
        the caller's list.
        """

        order = _positive_integer(field_order, "field_order")
        degree = _positive_integer(field_degree, "field_degree")
        result = list(self.coefficients)
        if len(result) < degree:
            result.extend([0] * (degree - len(result)))
        quotients = list(k)
        for position in range(degree):
            concrete = _concrete_integer(result[position])
            quotient = (
                concrete // order
                if concrete is not None
                else self._symbolic_quotient(quotients, "order reduction")
            )
            result[position] -= order * quotient
        self._check_quotients_consumed(quotients, "order reduction")
        return type(self)(result)

    def reduce_by_polynomial(
        self,
        field_modulus: "Polynomial",
        field_degree: int,
        k: Iterable[Any],
    ) -> "Polynomial":
        """Reduce by a monic ascending-coefficient modulus polynomial."""

        if not isinstance(field_modulus, Polynomial):
            raise TypeError("field_modulus must be a Polynomial")
        degree = _positive_integer(field_degree, "field_degree")
        if len(field_modulus) != degree + 1 or field_modulus.coefficients[-1] != 1:
            raise ValueError(
                "field_modulus must be monic with degree field_degree"
            )

        result = list(self.coefficients)
        quotients = list(k)
        for offset in range(len(result) - degree - 1, -1, -1):
            leading_position = offset + degree
            concrete = _concrete_integer(result[leading_position])
            quotient = (
                concrete
                if concrete is not None
                else self._symbolic_quotient(quotients, "polynomial reduction")
            )
            for position, modulus_coefficient in enumerate(
                field_modulus.coefficients
            ):
                result[offset + position] -= modulus_coefficient * quotient
        self._check_quotients_consumed(quotients, "polynomial reduction")
        return type(self)(result)


def _normalize_widths(widths: Iterable[int], name: str) -> tuple[int, ...]:
    normalized = _integer_tuple(widths, name)
    if not normalized or any(width <= 0 for width in normalized):
        raise ValueError(f"{name} must contain positive widths")
    return normalized


def pack_words(values: Iterable[int], widths: Iterable[int]) -> int:
    """Pack unsigned words in MSB-first order."""

    normalized_values = _integer_tuple(values, "values")
    normalized_widths = _normalize_widths(widths, "widths")
    if len(normalized_values) != len(normalized_widths):
        raise ValueError("values and widths must have the same length")
    packed = 0
    for position, (value, width) in enumerate(
        zip(normalized_values, normalized_widths, strict=True)
    ):
        if not 0 <= value < 1 << width:
            raise ValueError(f"values[{position}] does not fit width {width}")
        packed = (packed << width) | value
    return packed


def unpack_words(value: int, widths: Iterable[int]) -> tuple[int, ...]:
    """Unpack an unsigned integer into MSB-first words."""

    packed = _nonnegative_integer(value, "value")
    normalized_widths = _normalize_widths(widths, "widths")
    total_width = sum(normalized_widths)
    if packed >= 1 << total_width:
        raise ValueError("value does not fit the declared widths")
    words_reversed: list[int] = []
    remaining = packed
    for width in reversed(normalized_widths):
        words_reversed.append(remaining & ((1 << width) - 1))
        remaining >>= width
    return tuple(reversed(words_reversed))


def wrapper_maker(
    func: Callable[..., Any],
    in_bitsize: Iterable[int],
    out_bitsize: Iterable[int],
) -> Callable[[int], int]:
    """Adapt a fixed-width multi-input/output function to packed integers."""

    if not callable(func):
        raise TypeError("func must be callable")
    input_widths = _normalize_widths(in_bitsize, "in_bitsize")
    output_widths = _normalize_widths(out_bitsize, "out_bitsize")

    def wrapper(value: int) -> int:
        arguments = unpack_words(value, input_widths)
        raw_result = func(*arguments)
        if len(output_widths) == 1 and not isinstance(raw_result, (list, tuple)):
            results = (raw_result,)
        elif isinstance(raw_result, (list, tuple)):
            results = tuple(raw_result)
        else:
            raise TypeError("multiple outputs must be returned as a list or tuple")
        return pack_words(results, output_widths)

    wrapper.__name__ = getattr(func, "__name__", "packed_wrapper")
    wrapper.__doc__ = f"Packed fixed-width adapter for {func!r}."
    return wrapper


def _bits(value: int, width: int) -> list[int]:
    return [int(bool(value & (1 << position))) for position in range(width - 1, -1, -1)]


def construct_GF2_matrix(
    func_wrapper: Callable[[int], int],
    dimension: int,
    *,
    output_dimension: int | None = None,
    verify: bool = True,
    exhaustive_limit: int = 12,
    verification_trials: int = 100,
    seed: int = 0,
) -> tuple[Any, Any]:
    """Recover ``A, b`` for an affine map ``f(x) = x*A + b`` over GF(2)."""

    if not callable(func_wrapper):
        raise TypeError("func_wrapper must be callable")
    input_width = _positive_integer(dimension, "dimension")
    output_width = (
        input_width
        if output_dimension is None
        else _positive_integer(output_dimension, "output_dimension")
    )
    if not isinstance(verify, bool):
        raise TypeError("verify must be a bool")
    exhaustive_bound = _nonnegative_integer(exhaustive_limit, "exhaustive_limit")
    trials = _nonnegative_integer(verification_trials, "verification_trials")
    random_seed = _exact_integer(seed, "seed")

    def evaluate(value: int) -> int:
        result = _nonnegative_integer(func_wrapper(value), "func_wrapper result")
        if result >= 1 << output_width:
            raise ValueError("func_wrapper result exceeds output_dimension")
        return result

    field = GF(2)
    offset_value = evaluate(0)
    offset = vector(field, _bits(offset_value, output_width))
    rows = []
    for position in range(input_width):
        basis_value = 1 << (input_width - position - 1)
        rows.append(_bits(evaluate(basis_value) ^ offset_value, output_width))
    transformation = matrix(field, rows)

    if verify:
        if input_width <= exhaustive_bound:
            candidates: Iterable[int] = range(1 << input_width)
        else:
            random = Random(random_seed)
            candidates = (
                random.randrange(1 << input_width) for _ in range(trials)
            )
        for candidate in candidates:
            input_vector = vector(field, _bits(candidate, input_width))
            expected = vector(field, _bits(evaluate(candidate), output_width))
            if input_vector * transformation + offset != expected:
                raise ValueError("func_wrapper is not affine over GF(2)")
    return transformation, offset


def _base_variables(base_elements: Iterable[Any] | None) -> set[Any]:
    variables: set[Any] = set()
    if base_elements is None:
        return variables
    try:
        elements = tuple(base_elements)
    except TypeError as error:
        raise TypeError("base_elements must be iterable") from error
    for element in elements:
        if element == 1:
            continue
        try:
            variables.update(element.variables())
        except AttributeError:
            variables.add(element)
    return variables


def _constant_integer(polynomial: Any) -> int | None:
    try:
        if polynomial.variables():
            return None
    except AttributeError:
        pass
    try:
        return int(ZZ(polynomial))
    except (TypeError, ValueError):
        return None


def _degree_in(polynomial: Any, variable: Any) -> int:
    """Return a variable degree for both uni- and multivariate Sage types."""

    try:
        return int(polynomial.degree(variable))
    except TypeError:
        try:
            generators = tuple(polynomial.parent().gens())
        except AttributeError as error:
            raise TypeError("Groebner rows must be polynomials") from error
        if len(generators) != 1 or generators[0] != variable:
            raise
        return int(polynomial.degree())


def _linear_coefficient(polynomial: Any, variable: Any) -> Any:
    """Extract the coefficient of a degree-one variable across Sage backends."""

    generators = tuple(polynomial.parent().gens())
    if len(generators) == 1 and generators[0] == variable:
        return polynomial[1]
    return polynomial.coefficient({variable: 1})


def ideal_groebner(
    f: Iterable[Any],
    base_elements: Iterable[Any] | None = None,
) -> dict[Any, Any]:
    """Extract a unique solution from independent affine Groebner equations.

    A constant integer in an integer-polynomial basis is interpreted as the
    modulus.  Bases containing nonlinear or coupled unresolved equations are
    rejected; this helper deliberately does not pretend to be a general
    polynomial-system solver.
    """

    try:
        equations = tuple(f)
    except TypeError as error:
        raise TypeError("f must be an iterable of polynomials") from error
    if not equations:
        raise ValueError("f must not be empty")
    excluded = _base_variables(base_elements)
    basis = tuple(Ideal(equations).groebner_basis())
    if not basis:
        raise ValueError("the ideal has an empty Groebner basis")

    constant_rows = [
        value
        for row in basis
        if (value := _constant_integer(row)) is not None and value != 0
    ]
    modulus = 0
    for value in constant_rows:
        modulus = gcd(modulus, abs(value))
    if modulus == 1:
        raise ValueError("the equations generate the unit ideal and are inconsistent")

    solution: dict[Any, Any] = {}
    if modulus > 1:
        solution["mod"] = modulus

    solved_rows = 0
    for row in basis:
        if _constant_integer(row) is not None:
            continue
        active_variables = set(row.variables()) - excluded
        if not active_variables:
            continue
        if len(active_variables) != 1:
            raise ValueError("Groebner basis contains coupled unresolved variables")
        variable = next(iter(active_variables))
        if _degree_in(row, variable) != 1:
            raise ValueError(f"Groebner equation for {variable} is not affine")
        coefficient = _linear_coefficient(row, variable)
        remainder = row - coefficient * variable
        if variable in remainder.variables():
            raise ValueError(f"could not isolate {variable}")

        if modulus > 1:
            coefficient_integer = _constant_integer(coefficient)
            if coefficient_integer is None:
                raise ValueError("modular leading coefficients must be integers")
            if gcd(coefficient_integer, modulus) != 1:
                raise ValueError(
                    f"leading coefficient for {variable} is not invertible"
                )
            inverse = pow(coefficient_integer % modulus, -1, modulus)
            remainder_integer = _constant_integer(remainder)
            if remainder_integer is None:
                raise ValueError(
                    "modular affine extraction does not support symbolic parameters"
                )
            value = (-remainder_integer * inverse) % modulus
        else:
            value = -remainder / coefficient

        if variable in solution and solution[variable] != value:
            raise ValueError(f"conflicting values were derived for {variable}")
        solution[variable] = value
        solved_rows += 1

    if solved_rows == 0:
        raise ValueError("Groebner basis contains no extractable affine equations")

    substitutions = {key: value for key, value in solution.items() if key != "mod"}
    for equation in equations:
        try:
            evaluated = equation.subs(substitutions)
        except AttributeError:
            evaluated = equation
        try:
            if evaluated.variables():
                continue
        except AttributeError:
            pass
        if modulus > 1:
            if int(ZZ(evaluated)) % modulus:
                raise ArithmeticError("extracted modular solution failed verification")
        elif evaluated != 0:
            raise ArithmeticError("extracted solution failed verification")
    return solution
