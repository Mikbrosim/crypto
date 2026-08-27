"""Recover ECDSA private keys from structured nonce leakage."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from math import gcd
from operator import index as integer_index
from typing import Any

from rewrite.hidden_number_problem import CVPMethod, hnp
from rewrite.knapsack import KnapsackMethod, subset_sum


LatticeReduction = Callable[[Any], Any]


def _exact_integer(value: Any, name: str) -> int:
    """Return an exact integer while rejecting booleans and truncation."""
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer, not bool")
    try:
        return integer_index(value)
    except TypeError as error:
        raise TypeError(f"{name} must be an integer") from error


def _integer_tuple(values: Iterable[Any], name: str) -> tuple[int, ...]:
    """Normalize an iterable of exact integers."""
    try:
        raw_values = tuple(values)
    except TypeError as error:
        raise TypeError(f"{name} must be an iterable of integers") from error
    return tuple(
        _exact_integer(value, f"{name}[{index}]")
        for index, value in enumerate(raw_values)
    )


def _normalize_signatures(
    Z: Iterable[int],
    R: Iterable[int],
    S: Iterable[int],
    n: int,
    *,
    minimum_samples: int = 1,
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...], int]:
    """Validate signature scalars and normalize them modulo the group order."""
    order = _exact_integer(n, "n")
    if order <= 2:
        raise ValueError("n must be greater than two")
    hashes = _integer_tuple(Z, "Z")
    r_values = _integer_tuple(R, "R")
    s_values = _integer_tuple(S, "S")
    if not (len(hashes) == len(r_values) == len(s_values)):
        raise ValueError("Z, R, and S must have the same length")
    if len(hashes) < minimum_samples:
        raise ValueError(f"at least {minimum_samples} signatures are required")

    normalized_hashes = tuple(value % order for value in hashes)
    normalized_r: list[int] = []
    normalized_s: list[int] = []
    for index, (r_value, s_value) in enumerate(
        zip(r_values, s_values, strict=True)
    ):
        r_value %= order
        s_value %= order
        if r_value == 0:
            raise ValueError(f"R[{index}] must be nonzero modulo n")
        if s_value == 0 or gcd(s_value, order) != 1:
            raise ValueError(f"S[{index}] must be invertible modulo n")
        normalized_r.append(r_value)
        normalized_s.append(s_value)
    return normalized_hashes, tuple(normalized_r), tuple(normalized_s), order


def _normalize_leakage(
    values: Iterable[int],
    expected_length: int,
    upper_bound: int,
    name: str,
) -> tuple[int, ...]:
    """Validate per-signature known-bit values."""
    normalized = _integer_tuple(values, name)
    if len(normalized) != expected_length:
        raise ValueError(f"{name} must have one value per signature")
    if any(not 0 <= value < upper_bound for value in normalized):
        raise ValueError(f"{name} entries must be in [0, {upper_bound})")
    return normalized


def _nonce_bit_parameters(order: int, known_bits: int) -> tuple[int, int]:
    """Return the order bit length and unknown-bit bound."""
    bit_count = _exact_integer(known_bits, "l")
    order_bits = order.bit_length()
    if not 0 < bit_count < order_bits:
        raise ValueError(f"l must be between 1 and {order_bits - 1}")
    return bit_count, 1 << (order_bits - bit_count)


def _private_key_nonces(
    private_key: int | None,
    hashes: Sequence[int],
    r_values: Sequence[int],
    s_values: Sequence[int],
    order: int,
) -> tuple[int, tuple[int, ...]] | None:
    """Reconstruct all nonces and validate the ECDSA scalar equations."""
    if private_key is None:
        return None
    try:
        candidate = _exact_integer(private_key, "private key") % order
    except TypeError:
        return None
    if candidate == 0:
        return None

    nonces: list[int] = []
    for message_hash, r_value, s_value in zip(
        hashes,
        r_values,
        s_values,
        strict=True,
    ):
        nonce = (
            pow(s_value, -1, order)
            * (message_hash + r_value * candidate)
        ) % order
        if nonce == 0:
            return None
        if (s_value * nonce - message_hash - r_value * candidate) % order:
            return None
        nonces.append(nonce)
    return candidate, tuple(nonces)


def _solve_hnp_private_key(
    hashes: Sequence[int],
    r_values: Sequence[int],
    s_values: Sequence[int],
    order: int,
    multipliers: Sequence[int],
    offsets: Sequence[int],
    bound: int,
    *,
    signed: bool,
    lattice_reduction: LatticeReduction | None,
    verbose: bool,
    method: CVPMethod,
    effort: int,
    exact_threshold: int,
) -> tuple[int, tuple[int, ...]] | None:
    """Run HNP and reconstruct the corresponding ECDSA nonces."""
    private_key = hnp(
        order,
        multipliers,
        offsets,
        bound,
        lattice_reduction=lattice_reduction,
        verbose=verbose,
        method=method,
        effort=effort,
        signed=signed,
        exact_threshold=exact_threshold,
    )
    return _private_key_nonces(
        private_key,
        hashes,
        r_values,
        s_values,
        order,
    )


def ecdsa_biased_nonce_zero_msb(
    Z: Iterable[int],
    R: Iterable[int],
    S: Iterable[int],
    n: int,
    l: int,
    lattice_reduction: LatticeReduction | None = None,
    *,
    method: CVPMethod = "auto",
    effort: int = 1,
    verbose: bool = False,
    exact_threshold: int = 1 << 16,
) -> int | None:
    """Recover a key when each nonce has ``l`` zero most-significant bits."""
    hashes, r_values, s_values, order = _normalize_signatures(Z, R, S, n)
    _, bound = _nonce_bit_parameters(order, l)
    inverses = [pow(s_value, -1, order) for s_value in s_values]
    multipliers = [
        inverse * r_value % order
        for inverse, r_value in zip(inverses, r_values, strict=True)
    ]
    offsets = [
        -inverse * message_hash % order
        for inverse, message_hash in zip(inverses, hashes, strict=True)
    ]
    recovered = _solve_hnp_private_key(
        hashes,
        r_values,
        s_values,
        order,
        multipliers,
        offsets,
        bound,
        signed=False,
        lattice_reduction=lattice_reduction,
        verbose=verbose,
        method=method,
        effort=effort,
        exact_threshold=exact_threshold,
    )
    if recovered is None:
        return None
    private_key, nonces = recovered
    return private_key if all(nonce < bound for nonce in nonces) else None


def ecdsa_biased_nonce_zero_lsb(
    Z: Iterable[int],
    R: Iterable[int],
    S: Iterable[int],
    n: int,
    l: int,
    lattice_reduction: LatticeReduction | None = None,
    *,
    method: CVPMethod = "auto",
    effort: int = 1,
    verbose: bool = False,
    exact_threshold: int = 1 << 16,
) -> int | None:
    """Recover a key when each nonce has ``l`` zero least-significant bits."""
    hashes, r_values, s_values, order = _normalize_signatures(Z, R, S, n)
    known_bits, bound = _nonce_bit_parameters(order, l)
    power = 1 << known_bits
    if gcd(power, order) != 1:
        raise ValueError("2**l must be invertible modulo n")
    inverse_power = pow(power, -1, order)
    inverses = [pow(s_value, -1, order) for s_value in s_values]
    multipliers = [
        inverse_power * inverse * r_value % order
        for inverse, r_value in zip(inverses, r_values, strict=True)
    ]
    offsets = [
        -inverse_power * inverse * message_hash % order
        for inverse, message_hash in zip(inverses, hashes, strict=True)
    ]
    recovered = _solve_hnp_private_key(
        hashes,
        r_values,
        s_values,
        order,
        multipliers,
        offsets,
        bound,
        signed=False,
        lattice_reduction=lattice_reduction,
        verbose=verbose,
        method=method,
        effort=effort,
        exact_threshold=exact_threshold,
    )
    if recovered is None:
        return None
    private_key, nonces = recovered
    return private_key if all(nonce % power == 0 for nonce in nonces) else None


def ecdsa_biased_nonce_known_msb(
    Z: Iterable[int],
    R: Iterable[int],
    S: Iterable[int],
    T: Iterable[int],
    n: int,
    l: int,
    lattice_reduction: LatticeReduction | None = None,
    *,
    method: CVPMethod = "auto",
    effort: int = 1,
    verbose: bool = False,
    exact_threshold: int = 1 << 16,
) -> int | None:
    """Recover a key from the known ``l`` most-significant nonce bits."""
    hashes, r_values, s_values, order = _normalize_signatures(Z, R, S, n)
    known_bits, bound = _nonce_bit_parameters(order, l)
    known_values = _normalize_leakage(
        T,
        len(hashes),
        1 << known_bits,
        "T",
    )
    inverses = [pow(s_value, -1, order) for s_value in s_values]
    multipliers = [
        inverse * r_value % order
        for inverse, r_value in zip(inverses, r_values, strict=True)
    ]
    offsets = [
        (bound * known - inverse * message_hash) % order
        for known, inverse, message_hash in zip(
            known_values,
            inverses,
            hashes,
            strict=True,
        )
    ]
    recovered = _solve_hnp_private_key(
        hashes,
        r_values,
        s_values,
        order,
        multipliers,
        offsets,
        bound,
        signed=False,
        lattice_reduction=lattice_reduction,
        verbose=verbose,
        method=method,
        effort=effort,
        exact_threshold=exact_threshold,
    )
    if recovered is None:
        return None
    private_key, nonces = recovered
    return (
        private_key
        if all(
            nonce // bound == known
            for nonce, known in zip(nonces, known_values, strict=True)
        )
        else None
    )


def ecdsa_biased_nonce_known_lsb(
    Z: Iterable[int],
    R: Iterable[int],
    S: Iterable[int],
    T: Iterable[int],
    n: int,
    l: int,
    lattice_reduction: LatticeReduction | None = None,
    *,
    method: CVPMethod = "auto",
    effort: int = 1,
    verbose: bool = False,
    exact_threshold: int = 1 << 16,
) -> int | None:
    """Recover a key from the known ``l`` least-significant nonce bits."""
    hashes, r_values, s_values, order = _normalize_signatures(Z, R, S, n)
    known_bits, bound = _nonce_bit_parameters(order, l)
    power = 1 << known_bits
    if gcd(power, order) != 1:
        raise ValueError("2**l must be invertible modulo n")
    known_values = _normalize_leakage(T, len(hashes), power, "T")
    inverse_power = pow(power, -1, order)
    inverses = [pow(s_value, -1, order) for s_value in s_values]
    multipliers = [
        inverse_power * inverse * r_value % order
        for inverse, r_value in zip(inverses, r_values, strict=True)
    ]
    offsets = [
        inverse_power * (known - inverse * message_hash) % order
        for known, inverse, message_hash in zip(
            known_values,
            inverses,
            hashes,
            strict=True,
        )
    ]
    recovered = _solve_hnp_private_key(
        hashes,
        r_values,
        s_values,
        order,
        multipliers,
        offsets,
        bound,
        signed=False,
        lattice_reduction=lattice_reduction,
        verbose=verbose,
        method=method,
        effort=effort,
        exact_threshold=exact_threshold,
    )
    if recovered is None:
        return None
    private_key, nonces = recovered
    return (
        private_key
        if all(
            nonce % power == known
            for nonce, known in zip(nonces, known_values, strict=True)
        )
        else None
    )


def ecdsa_biased_nonce_shared_msb(
    Z: Iterable[int],
    R: Iterable[int],
    S: Iterable[int],
    n: int,
    l: int,
    lattice_reduction: LatticeReduction | None = None,
    *,
    method: CVPMethod = "auto",
    effort: int = 1,
    verbose: bool = False,
    exact_threshold: int = 1 << 16,
) -> int | None:
    """Recover a key when all nonces share ``l`` most-significant bits."""
    hashes, r_values, s_values, order = _normalize_signatures(
        Z,
        R,
        S,
        n,
        minimum_samples=2,
    )
    _, bound = _nonce_bit_parameters(order, l)
    inverses = [pow(s_value, -1, order) for s_value in s_values]
    reference_r = inverses[0] * r_values[0] % order
    reference_hash = inverses[0] * hashes[0] % order
    multipliers = [
        (inverse * r_value - reference_r) % order
        for inverse, r_value in zip(
            inverses[1:],
            r_values[1:],
            strict=True,
        )
    ]
    offsets = [
        (reference_hash - inverse * message_hash) % order
        for inverse, message_hash in zip(
            inverses[1:],
            hashes[1:],
            strict=True,
        )
    ]
    recovered = _solve_hnp_private_key(
        hashes,
        r_values,
        s_values,
        order,
        multipliers,
        offsets,
        bound,
        signed=True,
        lattice_reduction=lattice_reduction,
        verbose=verbose,
        method=method,
        effort=effort,
        exact_threshold=exact_threshold,
    )
    if recovered is None:
        return None
    private_key, nonces = recovered
    shared_prefix = nonces[0] // bound
    return (
        private_key
        if all(nonce // bound == shared_prefix for nonce in nonces)
        else None
    )


def ecdsa_nonce_hash_xor_privkey(
    Z1: Iterable[int],
    Z2: Iterable[int],
    R: Iterable[int],
    S: Iterable[int],
    n: int,
    lattice_reduction: LatticeReduction | None = None,
    *,
    method: KnapsackMethod = "auto",
    effort: int = 1,
    verbose: bool = False,
    exact_threshold: int = 32,
) -> int | None:
    """Recover a key when each nonce is ``nonce_hash XOR private_key``."""
    hashes, r_values, s_values, order = _normalize_signatures(Z2, R, S, n)
    order_bits = order.bit_length()
    nonce_hashes = _normalize_leakage(
        Z1,
        len(hashes),
        1 << order_bits,
        "Z1",
    )

    weights: list[list[int]] = []
    targets: list[int] = []
    for nonce_hash, message_hash, r_value, s_value in zip(
        nonce_hashes,
        hashes,
        r_values,
        s_values,
        strict=True,
    ):
        row = []
        for bit_index in range(order_bits):
            nonce_hash_bit = (nonce_hash >> bit_index) & 1
            coefficient = (1 << bit_index) * (
                s_value * (1 - 2 * nonce_hash_bit) - r_value
            )
            row.append(coefficient % order)
        weights.append(row)
        targets.append((message_hash - s_value * nonce_hash) % order)

    bits = subset_sum(
        weights,
        targets,
        modulus=order,
        lattice_reduction=lattice_reduction,
        verbose=verbose,
        method=method,
        effort=effort,
        exact_threshold=exact_threshold,
    )
    if bits is None:
        return None
    private_key = sum(bit << index for index, bit in enumerate(bits))
    if not 0 < private_key < order:
        return None

    for nonce_hash, message_hash, r_value, s_value in zip(
        nonce_hashes,
        hashes,
        r_values,
        s_values,
        strict=True,
    ):
        nonce = nonce_hash ^ private_key
        if not 0 < nonce < order:
            return None
        if (s_value * nonce - message_hash - r_value * private_key) % order:
            return None
    return private_key
