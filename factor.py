"""Correctness-first helpers for structured integer factorization.

The routines in this module solve distinct leakage models: bounded trial
division, separated high/low products, partially known radix digits, biased
binary factors, and partially known RSA private-key components.  Every
candidate returned by an attack is verified by exact multiplication.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from itertools import combinations
from math import prod
from operator import index as integer_index
from typing import TypeVar


__all__ = [
    "CandidateLimitError",
    "TrialDivisionResult",
    "factorize_1",
    "factorize_2",
    "factorize_3",
    "factorize_4",
    "naive_factor",
    "powerset",
    "prod_powerset",
    "trial_division",
]


_DIGITS = "0123456789abcdefghijklmnopqrstuvwxyz"
_T = TypeVar("_T")


class CandidateLimitError(ValueError):
    """Raised when an exact search would exceed its declared resource cap."""

    def __init__(
        self,
        stage: int,
        candidates: int,
        limit: int,
        *,
        attack: str,
    ) -> None:
        super().__init__(
            f"{attack} produced {candidates} candidates at stage {stage}, "
            f"exceeding the limit of {limit}"
        )
        self.stage = stage
        self.candidates = candidates
        self.limit = limit
        self.attack = attack


@dataclass(frozen=True)
class TrialDivisionResult:
    """Prime factors found by trial division and any unresolved cofactor."""

    factors: tuple[int, ...]
    remainder: int

    @property
    def complete(self) -> bool:
        """Return whether the result is a complete prime factorization."""
        return self.remainder == 1

    @property
    def value(self) -> int:
        """Reconstruct the original positive integer."""
        return prod(self.factors) * self.remainder


def _exact_integer(value: object, name: str) -> int:
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


def _candidate_limit(value: object | None, name: str) -> int | None:
    if value is None:
        return None
    return _positive_integer(value, name)


def _verbose_printer(enabled: bool, attack: str):
    if not isinstance(enabled, bool):
        raise TypeError("verbose must be a bool")
    if enabled:
        return lambda *values: print(f"[{attack}]", *values)
    return lambda *values: None


def trial_division(
    n: int,
    limit: int | None = None,
) -> TrialDivisionResult:
    """Factor a positive integer using candidate divisors below ``limit``.

    With ``limit=None`` the division continues until the factorization is
    proven complete.  A finite limit is exclusive, matching ``range(2,
    limit)`` from the original helper.  An unresolved composite is retained
    in ``remainder`` rather than silently discarded.
    """
    normalized_n = _positive_integer(n, "n")
    if limit is None:
        normalized_limit = None
    else:
        normalized_limit = _positive_integer(limit, "limit")
        if normalized_limit < 2:
            raise ValueError("limit must be at least two")

    factors: list[int] = []
    remainder = normalized_n
    divisor = 2
    while remainder > 1:
        if divisor * divisor > remainder:
            factors.append(remainder)
            remainder = 1
            break
        if normalized_limit is not None and divisor >= normalized_limit:
            break
        while remainder % divisor == 0:
            factors.append(divisor)
            remainder //= divisor
        divisor = 3 if divisor == 2 else divisor + 2
    return TrialDivisionResult(tuple(factors), remainder)


def naive_factor(n: int, limit: int) -> list[int]:
    """Compatibility wrapper returning factors found below ``limit``."""
    return list(trial_division(n, limit).factors)


def powerset(iterable: Iterable[_T]) -> Iterator[tuple[_T, ...]]:
    """Yield every positional subset of ``iterable`` in increasing size."""
    values = tuple(iterable)
    for subset_size in range(len(values) + 1):
        yield from combinations(values, subset_size)


def prod_powerset(iterable: Iterable[int]) -> Iterator[int]:
    """Yield the distinct products of every subset in sorted order."""
    products = {1}
    for position, value in enumerate(iterable):
        normalized = _positive_integer(value, f"iterable[{position}]")
        products.update(product * normalized for product in tuple(products))
    yield from sorted(products)


def _fallback_factorization(value: int) -> tuple[tuple[int, int], ...]:
    """Factor small test values when this module is imported outside Sage."""
    if value.bit_length() > 48:
        raise RuntimeError(
            "SageMath is required to factor separated products above 48 bits"
        )
    result = trial_division(value)
    return tuple(sorted(Counter(result.factors).items()))


def _factor_exponents(value: int) -> tuple[tuple[int, int], ...]:
    if value == 1:
        return ()
    try:
        from sage.all import factor as sage_factor
    except ModuleNotFoundError:
        return _fallback_factorization(value)
    return tuple((int(prime), int(exponent)) for prime, exponent in sage_factor(value))


def _positive_divisors(value: int) -> tuple[int, ...]:
    if value <= 0:
        raise ValueError("divisor generation requires a positive integer")
    divisors = [1]
    for prime, exponent in _factor_exponents(value):
        previous = tuple(divisors)
        power = 1
        for _ in range(exponent):
            power *= prime
            divisors.extend(divisor * power for divisor in previous)
    return tuple(sorted(divisors))


def factorize_1(
    N: int,
    bits: int,
    verbose: bool = False,
    *,
    carry_bound: int = 0,
    max_candidates: int | None = 100_000,
) -> tuple[int, int] | None:
    """Recover factors whose high and low limbs have small product carries.

    Write ``p = a1*B + u1`` and ``q = a2*B + u2`` for ``B = 2**bits``.
    ``carry_bound=0`` is the original strict model.  Larger values enumerate
    low-product and middle-product carries up to the supplied inclusive bound.
    """
    modulus = _positive_integer(N, "N")
    limb_bits = _positive_integer(bits, "bits")
    bound = _nonnegative_integer(carry_bound, "carry_bound")
    limit = _candidate_limit(max_candidates, "max_candidates")
    emit = _verbose_printer(verbose, "factorize_1")
    if modulus <= 3:
        raise ValueError("N must be greater than three")

    base = 1 << limb_bits
    high_residue = modulus // (base * base)
    low_residue = modulus % base
    divisor_cache: dict[int, tuple[int, ...]] = {}

    def divisors(value: int) -> tuple[int, ...]:
        if value not in divisor_cache:
            divisor_cache[value] = _positive_divisors(value)
        return divisor_cache[value]

    tested: set[tuple[int, int]] = set()
    candidate_count = 0
    for low_carry in range(bound + 1):
        low_product = low_residue + low_carry * base
        if not 0 < low_product < base * base:
            continue
        low_pairs = tuple(
            (low, low_product // low)
            for low in divisors(low_product)
            if low < base and low_product // low < base
        )
        for high_carry in range(bound + 1):
            high_product = high_residue - high_carry
            if high_product <= 0:
                continue
            emit(
                f"low carry {low_carry}, high carry {high_carry}",
                f"products {low_product}, {high_product}",
            )
            for high in divisors(high_product):
                other_high = high_product // high
                for low, other_low in low_pairs:
                    factor1 = high * base + low
                    factor2 = other_high * base + other_low
                    pair = (factor1, factor2)
                    if pair in tested:
                        continue
                    tested.add(pair)
                    candidate_count += 1
                    if limit is not None and candidate_count > limit:
                        raise CandidateLimitError(
                            low_carry * (bound + 1) + high_carry,
                            candidate_count,
                            limit,
                            attack="factorize_1",
                        )
                    if 1 < factor1 < modulus and factor1 * factor2 == modulus:
                        return tuple(sorted((factor1, factor2)))
    return None


def _normalize_pattern(pattern: object, name: str, base: int) -> str:
    if not isinstance(pattern, str):
        raise TypeError(f"{name} must be a string")
    if not pattern:
        raise ValueError(f"{name} must not be empty")
    normalized = pattern.lower()
    for character in normalized:
        if character == "?":
            continue
        if character not in _DIGITS:
            raise ValueError(f"{name} contains invalid digit {character!r}")
        if _DIGITS.index(character) >= base:
            raise ValueError(
                f"digit {character!r} in {name} is invalid for base {base}"
            )
    return normalized


def _digit_options(pattern: str, position: int, base: int):
    if position >= len(pattern):
        return (0,)
    character = pattern[-position - 1]
    if character == "?":
        return range(base)
    return (_DIGITS.index(character),)


def factorize_2(
    N: int,
    partial_p: str,
    partial_q: str,
    base: int = 10,
    max_posib: int | None = 100,
    *,
    verbose: bool = False,
) -> tuple[int, int] | None:
    """Recover two factors from radix strings whose unknown digits are ``?``.

    Candidates are constructed from least to most significant digit and are
    retained only when their product is correct modulo the newly completed
    power of ``base``.  ``max_posib`` is a hard limit, not a pruning beam.
    """
    modulus = _positive_integer(N, "N")
    radix = _positive_integer(base, "base")
    if not 2 <= radix <= len(_DIGITS):
        raise ValueError(f"base must be between 2 and {len(_DIGITS)}")
    pattern_p = _normalize_pattern(partial_p, "partial_p", radix)
    pattern_q = _normalize_pattern(partial_q, "partial_q", radix)
    limit = _candidate_limit(max_posib, "max_posib")
    emit = _verbose_printer(verbose, "factorize_2")

    candidates = {(0, 0)}
    place = 1
    symmetric = pattern_p == pattern_q
    positions = max(len(pattern_p), len(pattern_q))
    for position in range(positions):
        modulus_at_position = place * radix
        next_candidates: set[tuple[int, int]] = set()
        digits_p = _digit_options(pattern_p, position, radix)
        digits_q = _digit_options(pattern_q, position, radix)
        for candidate_p, candidate_q in candidates:
            for digit_p in digits_p:
                next_p = candidate_p + digit_p * place
                for digit_q in digits_q:
                    next_q = candidate_q + digit_q * place
                    if next_p * next_q % modulus_at_position != (
                        modulus % modulus_at_position
                    ):
                        continue
                    pair = (next_p, next_q)
                    if symmetric and next_p > next_q:
                        pair = (next_q, next_p)
                    next_candidates.add(pair)

        emit(f"digit {position}: {len(next_candidates)} candidates")
        if not next_candidates:
            return None
        if limit is not None and len(next_candidates) > limit:
            raise CandidateLimitError(
                position,
                len(next_candidates),
                limit,
                attack="factorize_2",
            )
        candidates = next_candidates
        place = modulus_at_position

    for factor1, factor2 in sorted(candidates):
        if factor1 > 1 and factor2 > 1 and factor1 * factor2 == modulus:
            return factor1, factor2
    return None


def _bias_score(pair: tuple[int, int], width: int, many_ones: bool) -> int:
    ones = pair[0].bit_count() + pair[1].bit_count()
    if many_ones:
        return 2 * width - ones
    return ones


def factorize_3(
    n: int,
    many_ones: bool,
    max_posib: int | None = 1024,
    *,
    strategy: str = "exact",
    factor_bits: int | None = None,
    verbose: bool = False,
) -> tuple[int, int] | None:
    """Recover balanced binary factors, optionally prioritizing a bit bias.

    ``strategy='exact'`` raises before dropping candidates.  The opt-in
    ``'beam'`` strategy retains the ``max_posib`` candidates with the fewest
    minority bits and can therefore miss the correct factorization.
    """
    modulus = _positive_integer(n, "n")
    if modulus <= 3:
        raise ValueError("n must be greater than three")
    if not isinstance(many_ones, bool):
        raise TypeError("many_ones must be a bool")
    if strategy not in {"exact", "beam"}:
        raise ValueError("strategy must be 'exact' or 'beam'")
    limit = _candidate_limit(max_posib, "max_posib")
    if factor_bits is None:
        width = (modulus.bit_length() + 1) // 2
    else:
        width = _positive_integer(factor_bits, "factor_bits")
    if 2 * width < modulus.bit_length():
        raise ValueError("factor_bits is too small for n")
    emit = _verbose_printer(verbose, "factorize_3")

    candidates = {(0, 0)}
    for position in range(width):
        bit = 1 << position
        low_mask = (bit << 1) - 1
        target = modulus & low_mask
        next_candidates: set[tuple[int, int]] = set()
        for factor1, factor2 in candidates:
            for bit1 in (0, bit):
                for bit2 in (0, bit):
                    next_factor1 = factor1 | bit1
                    next_factor2 = factor2 | bit2
                    if next_factor1 * next_factor2 & low_mask != target:
                        continue
                    next_candidates.add(
                        tuple(sorted((next_factor1, next_factor2)))
                    )

        emit(f"bit {position}: {len(next_candidates)} candidates")
        if not next_candidates:
            return None
        if limit is not None and len(next_candidates) > limit:
            if strategy == "exact":
                raise CandidateLimitError(
                    position,
                    len(next_candidates),
                    limit,
                    attack="factorize_3",
                )
            ordered = sorted(
                next_candidates,
                key=lambda pair: (
                    _bias_score(pair, position + 1, many_ones),
                    pair,
                ),
            )
            next_candidates = set(ordered[:limit])
        candidates = next_candidates

    for factor1, factor2 in sorted(candidates):
        if factor1 > 1 and factor2 > 1 and factor1 * factor2 == modulus:
            return factor1, factor2
    return None


def _matches_known_bits(candidate: int, value: int, mask: int, bits: int) -> bool:
    low_mask = (1 << bits) - 1
    return ((candidate ^ value) & mask & low_mask) == 0


def _two_adic_valuation(value: int) -> int:
    return (value & -value).bit_length() - 1


def _partial_rsa_constraints_hold(
    factor_p: int,
    factor_q: int,
    precision: int,
    *,
    modulus: int,
    exponent: int,
    p_value: int,
    p_mask: int,
    q_value: int,
    q_mask: int,
    d_value: int,
    d_mask: int,
    d_multiplier: int,
    dp_value: int,
    dp_mask: int,
    dp_multiplier: int,
    dq_value: int,
    dq_mask: int,
    dq_multiplier: int,
    totient_divisor: int,
) -> bool:
    low_mask = (1 << precision) - 1
    if factor_p * factor_q & low_mask != modulus & low_mask:
        return False
    if not _matches_known_bits(factor_p, p_value, p_mask, precision):
        return False
    if not _matches_known_bits(factor_q, q_value, q_mask, precision):
        return False

    local_modulus = 1 << precision
    inverse_exponent = pow(exponent, -1, local_modulus)
    if dp_mask:
        candidate_dp = (
            inverse_exponent * (dp_multiplier * (factor_p - 1) + 1)
        ) % local_modulus
        if not _matches_known_bits(candidate_dp, dp_value, dp_mask, precision):
            return False
    if dq_mask:
        candidate_dq = (
            inverse_exponent * (dq_multiplier * (factor_q - 1) + 1)
        ) % local_modulus
        if not _matches_known_bits(candidate_dq, dq_value, dq_mask, precision):
            return False

    if d_mask:
        removed_bits = _two_adic_valuation(totient_divisor)
        available_bits = precision - removed_bits
        if available_bits > 0:
            power_of_two = 1 << removed_bits
            partial_totient = (factor_p - 1) * (factor_q - 1)
            if partial_totient % power_of_two:
                return False
            relation_modulus = 1 << available_bits
            odd_divisor = totient_divisor >> removed_bits
            divided_totient = (
                (partial_totient >> removed_bits)
                * pow(odd_divisor, -1, relation_modulus)
            ) % relation_modulus
            candidate_d = (
                pow(exponent, -1, relation_modulus)
                * (d_multiplier * divided_totient + 1)
            ) % relation_modulus
            if not _matches_known_bits(
                candidate_d,
                d_value,
                d_mask,
                available_bits,
            ):
                return False
    return True


def _complete_rsa_constraints_hold(
    factor_p: int,
    factor_q: int,
    *,
    modulus: int,
    exponent: int,
    p_value: int,
    p_mask: int,
    q_value: int,
    q_mask: int,
    d_value: int,
    d_mask: int,
    d_multiplier: int,
    dp_value: int,
    dp_mask: int,
    dp_multiplier: int,
    dq_value: int,
    dq_mask: int,
    dq_multiplier: int,
    totient_divisor: int,
) -> bool:
    if factor_p <= 1 or factor_q <= 1 or factor_p * factor_q != modulus:
        return False
    if (factor_p ^ p_value) & p_mask:
        return False
    if (factor_q ^ q_value) & q_mask:
        return False

    relations = (
        (dp_mask, dp_value, dp_multiplier * (factor_p - 1) + 1),
        (dq_mask, dq_value, dq_multiplier * (factor_q - 1) + 1),
    )
    for mask, leaked_value, numerator in relations:
        if not mask:
            continue
        if numerator % exponent:
            return False
        if ((numerator // exponent) ^ leaked_value) & mask:
            return False

    if d_mask:
        totient = (factor_p - 1) * (factor_q - 1)
        if totient % totient_divisor:
            return False
        numerator = d_multiplier * (totient // totient_divisor) + 1
        if numerator % exponent:
            return False
        if ((numerator // exponent) ^ d_value) & d_mask:
            return False
    return True


def factorize_4(
    e: int,
    n: int = 0,
    n_MASK: int = 0,
    d: int = 0,
    d_MASK: int = 0,
    kd: int = 0,
    dp: int = 0,
    dp_MASK: int = 0,
    kdp: int = 0,
    p: int = 0,
    p_MASK: int = 0,
    dq: int = 0,
    dq_MASK: int = 0,
    kdq: int = 0,
    q: int = 0,
    q_MASK: int = 0,
    f: int = 1,
    *,
    factor_bits: int | None = None,
    max_candidates: int | None = 100_000,
    verbose: bool = False,
) -> tuple[int, int] | None:
    """Recover balanced RSA factors from known bits of private components.

    The supplied multipliers describe ``e*d = 1 + kd*phi/f``,
    ``e*dp = 1 + kdp*(p-1)``, and ``e*dq = 1 + kdq*(q-1)``.  A component is
    ignored when its mask is zero.  ``n_MASK`` is retained for source
    compatibility; the complete public modulus ``n`` is always used because
    exact product bits provide strictly stronger valid pruning.
    """
    exponent = _positive_integer(e, "e")
    modulus = _positive_integer(n, "n")
    if exponent % 2 == 0:
        raise ValueError("e must be odd for bitwise reconstruction")
    if modulus <= 3 or modulus % 2 == 0:
        raise ValueError("n must be an odd RSA modulus greater than three")

    normalized_n_mask = _nonnegative_integer(n_MASK, "n_MASK")
    del normalized_n_mask
    d_value = _nonnegative_integer(d, "d")
    d_mask = _nonnegative_integer(d_MASK, "d_MASK")
    d_multiplier = _nonnegative_integer(kd, "kd")
    dp_value = _nonnegative_integer(dp, "dp")
    dp_mask = _nonnegative_integer(dp_MASK, "dp_MASK")
    dp_multiplier = _nonnegative_integer(kdp, "kdp")
    p_value = _nonnegative_integer(p, "p")
    p_mask = _nonnegative_integer(p_MASK, "p_MASK")
    dq_value = _nonnegative_integer(dq, "dq")
    dq_mask = _nonnegative_integer(dq_MASK, "dq_MASK")
    dq_multiplier = _nonnegative_integer(kdq, "kdq")
    q_value = _nonnegative_integer(q, "q")
    q_mask = _nonnegative_integer(q_MASK, "q_MASK")
    totient_divisor = _positive_integer(f, "f")
    limit = _candidate_limit(max_candidates, "max_candidates")
    emit = _verbose_printer(verbose, "factorize_4")

    for mask, multiplier, name in (
        (d_mask, d_multiplier, "kd"),
        (dp_mask, dp_multiplier, "kdp"),
        (dq_mask, dq_multiplier, "kdq"),
    ):
        if mask and multiplier == 0:
            raise ValueError(f"{name} must be positive when its leakage mask is set")

    if factor_bits is None:
        width = (modulus.bit_length() + 1) // 2
    else:
        width = _positive_integer(factor_bits, "factor_bits")
    if 2 * width < modulus.bit_length():
        raise ValueError("factor_bits is too small for n")

    candidates = {(0, 0)}
    for position in range(width):
        bit = 1 << position
        precision = position + 1
        next_candidates: set[tuple[int, int]] = set()
        for factor_p, factor_q in candidates:
            for p_bit in (0, bit):
                for q_bit in (0, bit):
                    next_p = factor_p | p_bit
                    next_q = factor_q | q_bit
                    if _partial_rsa_constraints_hold(
                        next_p,
                        next_q,
                        precision,
                        modulus=modulus,
                        exponent=exponent,
                        p_value=p_value,
                        p_mask=p_mask,
                        q_value=q_value,
                        q_mask=q_mask,
                        d_value=d_value,
                        d_mask=d_mask,
                        d_multiplier=d_multiplier,
                        dp_value=dp_value,
                        dp_mask=dp_mask,
                        dp_multiplier=dp_multiplier,
                        dq_value=dq_value,
                        dq_mask=dq_mask,
                        dq_multiplier=dq_multiplier,
                        totient_divisor=totient_divisor,
                    ):
                        next_candidates.add((next_p, next_q))

        emit(f"bit {position}: {len(next_candidates)} candidates")
        if not next_candidates:
            return None
        if limit is not None and len(next_candidates) > limit:
            raise CandidateLimitError(
                position,
                len(next_candidates),
                limit,
                attack="factorize_4",
            )
        candidates = next_candidates

    for factor_p, factor_q in sorted(candidates):
        if _complete_rsa_constraints_hold(
            factor_p,
            factor_q,
            modulus=modulus,
            exponent=exponent,
            p_value=p_value,
            p_mask=p_mask,
            q_value=q_value,
            q_mask=q_mask,
            d_value=d_value,
            d_mask=d_mask,
            d_multiplier=d_multiplier,
            dp_value=dp_value,
            dp_mask=dp_mask,
            dp_multiplier=dp_multiplier,
            dq_value=dq_value,
            dq_mask=dq_mask,
            dq_multiplier=dq_multiplier,
            totient_divisor=totient_divisor,
        ):
            return factor_p, factor_q
    return None
