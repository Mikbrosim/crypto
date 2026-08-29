"""Small pure-Python number-theory helpers for rewritten adapters."""

from __future__ import annotations

from math import isqrt


def is_probable_prime(value: int) -> bool:
    """Use deterministic Miller-Rabin below 2**64 and strong tests above it."""
    if value < 2:
        return False
    small_primes = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37)
    if value in small_primes:
        return True
    if any(value % prime == 0 for prime in small_primes):
        return False

    odd_part = value - 1
    power_of_two = 0
    while odd_part % 2 == 0:
        power_of_two += 1
        odd_part //= 2
    if value < 1 << 64:
        bases = (2, 325, 9375, 28178, 450775, 9780504, 1795265022)
    else:
        bases = small_primes
    for base in bases:
        if base % value == 0:
            continue
        witness = pow(base, odd_part, value)
        if witness in (1, value - 1):
            continue
        for _ in range(power_of_two - 1):
            witness = witness * witness % value
            if witness == value - 1:
                break
        else:
            return False
    return True


def discrete_log_bsgs(base: int, target: int, modulus: int, order: int) -> int:
    """Solve a verified discrete log in a group of known finite order."""
    if order <= 0:
        raise ValueError("order must be positive")
    base %= modulus
    target %= modulus
    if pow(base, order, modulus) != 1 or pow(target, order, modulus) != 1:
        raise ValueError("base and target must lie in the declared subgroup")

    step = isqrt(order) + 1
    table: dict[int, int] = {}
    value = 1
    for exponent in range(step):
        table.setdefault(value, exponent)
        value = value * base % modulus
    giant_multiplier = pow(pow(base, step, modulus), -1, modulus)
    value = target
    for giant_step in range(step + 1):
        if value in table:
            exponent = giant_step * step + table[value]
            if exponent < order and pow(base, exponent, modulus) == target:
                return exponent
        value = value * giant_multiplier % modulus
    raise ValueError("target has no discrete logarithm to the supplied base")


def crt_coprime(residues: list[int], moduli: list[int]) -> int:
    """Combine residues with pairwise-coprime positive moduli."""
    if not residues or len(residues) != len(moduli):
        raise ValueError("residues and moduli must be non-empty and equal length")
    result = 0
    combined_modulus = 1
    for residue, modulus in zip(residues, moduli, strict=True):
        if modulus <= 0:
            raise ValueError("CRT moduli must be positive")
        adjustment = (
            (residue - result) * pow(combined_modulus, -1, modulus)
        ) % modulus
        result += combined_modulus * adjustment
        combined_modulus *= modulus
    return result % combined_modulus
