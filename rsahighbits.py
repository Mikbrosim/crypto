"""Factor balanced RSA moduli from known high bits of one prime.

For a factor ``p = known_high + low`` with ``0 <= low < 2**unknown_bits``,
the polynomial ``known_high + x`` has the small root ``low`` modulo ``p``.
The default recovery path uses SageMath's maintained univariate Coppersmith
implementation.  An explicit Howgrave--Graham lattice is retained for custom
reduction and parameter experiments.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Iterable, Sequence
from math import gcd
from operator import index as integer_index
from typing import Any, Literal

from sage.all import Matrix, PolynomialRing, QQ, ZZ, Zmod, random_prime

try:
    from .small_roots import small_roots as find_small_roots
except ImportError:  # Permit ``python rewrite/rsahighbits.py``.
    from small_roots import small_roots as find_small_roots


BITLENGTH = 1024
RecoveryMethod = Literal["auto", "bruteforce", "sage", "lattice"]
LatticeReduction = Callable[[Any], Any]


def _exact_integer(value: Any, name: str) -> int:
    """Return an exact integer without accepting booleans or truncation."""
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer, not bool")
    try:
        return integer_index(value)
    except TypeError:
        try:
            rational = QQ(value)
        except (TypeError, ValueError) as error:
            raise TypeError(f"{name} must be an integer") from error
        if rational.denominator() != 1:
            raise TypeError(f"{name} must be an integer")
        return int(rational)


def _positive_integer(value: Any, name: str) -> int:
    """Validate a strictly positive integer."""
    normalized = _exact_integer(value, name)
    if normalized <= 0:
        raise ValueError(f"{name} must be positive")
    return normalized


def _nonnegative_integer(value: Any, name: str) -> int:
    """Validate a non-negative integer."""
    normalized = _exact_integer(value, name)
    if normalized < 0:
        raise ValueError(f"{name} must be non-negative")
    return normalized


def _verbose_printer(enabled: bool) -> Callable[..., None]:
    """Create a small compatibility verbosity callback."""
    if not isinstance(enabled, bool):
        raise TypeError("verbose must be a bool")
    if enabled:
        return lambda *values: print("[rsahighbits]", *values)
    return lambda *values: None


def _normalize_problem(
    modulus: Any,
    known_high: Any,
    unknown_bits: Any,
) -> tuple[int, int, int, int]:
    """Validate a high-bit factorization problem and return its root bound."""
    normalized_modulus = _positive_integer(modulus, "N")
    if normalized_modulus <= 3:
        raise ValueError("N must be greater than three")
    normalized_high = _positive_integer(known_high, "p_msb")
    if normalized_high >= normalized_modulus:
        raise ValueError("p_msb must be smaller than N")
    normalized_unknown = _nonnegative_integer(unknown_bits, "unknown_bits")
    bound = 1 << normalized_unknown
    if normalized_high % bound != 0:
        raise ValueError("p_msb must be aligned to 2**unknown_bits")
    if normalized_high + bound > normalized_modulus:
        raise ValueError("the complete factor interval must lie below N")
    return normalized_modulus, normalized_high, normalized_unknown, bound


def _normalize_problem_tuple(problem: Sequence[Any]) -> tuple[int, int, int, int]:
    """Validate the legacy ``(N, p_msb, unknown_bits)`` representation."""
    try:
        values = tuple(problem)
    except TypeError as error:
        raise TypeError("problem must be a three-item sequence") from error
    if len(values) != 3:
        raise ValueError("problem must contain (N, p_msb, unknown_bits)")
    return _normalize_problem(*values)


def _normalize_effort(
    effort: Any,
    lattice_reduction: LatticeReduction | None,
) -> int:
    """Validate the shared LLL/BKZ effort convention."""
    normalized = _positive_integer(effort, "effort")
    if lattice_reduction is not None and not callable(lattice_reduction):
        raise TypeError("lattice_reduction must be callable or None")
    if lattice_reduction is not None and normalized != 1:
        raise ValueError("effort must be 1 with a custom lattice reduction")
    return normalized


def gen_rsakey(bitlength: int = BITLENGTH) -> tuple[int, tuple[int, int]]:
    """Generate a balanced RSA modulus with the top two prime bits set.

    The optional ``bitlength`` generalizes the original global-only helper and
    makes small correctness instances inexpensive to generate.
    """
    modulus_bits = _positive_integer(bitlength, "bitlength")
    if modulus_bits < 10:
        raise ValueError("bitlength must be at least ten")
    p_bits = modulus_bits // 2
    q_bits = modulus_bits - p_bits

    def generate_prime(bits: int) -> int:
        lower = 3 << (bits - 2)
        upper = (1 << bits) - 1
        return int(random_prime(upper, lbound=lower))

    p = generate_prime(p_bits)
    q = generate_prime(q_bits)
    while q == p:
        q = generate_prime(q_bits)
    modulus = p * q
    if modulus.bit_length() != modulus_bits:
        raise RuntimeError("generated primes did not produce the requested modulus size")
    return modulus, (p, q)


def generate_problem_instance(
    unknown_bits: int,
    bitlength: int = BITLENGTH,
) -> tuple[tuple[int, int], tuple[int, int, int]]:
    """Generate a key and reveal all but the requested low bits of ``p``."""
    normalized_unknown = _nonnegative_integer(unknown_bits, "unknown_bits")
    modulus_bits = _positive_integer(bitlength, "bitlength")
    if normalized_unknown >= modulus_bits // 2:
        raise ValueError("unknown_bits must leave at least one known bit of p")
    modulus, (p, q) = gen_rsakey(modulus_bits)
    bound = 1 << normalized_unknown
    known_high = p - p % bound
    return (p, q), (modulus, known_high, normalized_unknown)


def _parameter_condition(
    modulus_bits: int,
    factor_lower_exponent: int,
    root_bits: int,
    k: int,
    t: int,
) -> bool:
    """Check a conservative Howgrave--Graham/LLL norm inequality.

    We use ``N < 2**modulus_bits``, ``p >= 2**factor_lower_exponent``,
    ``X = 2**root_bits``, the standard LLL first-vector factor, and the
    Euclidean-to-coefficient norm conversion.  The integer inequality avoids
    unstable floating-point decisions near the theoretical boundary.
    """
    dimension = k + t
    left = (
        dimension * (dimension - 1)
        + 2 * k * (k + 1) * modulus_bits
        + 2 * dimension * (dimension - 1) * root_bits
        + 2 * dimension * dimension.bit_length()
    )
    right = 4 * k * dimension * factor_lower_exponent
    return left < right


def get_kt(
    unknown_bits: int,
    bitlength: int = BITLENGTH,
    *,
    factor_bits: int | None = None,
    safety_bits: int = 0,
    max_dimension: int = 256,
) -> tuple[int, int]:
    """Select the smallest conservative explicit-lattice parameters.

    ``factor_bits`` is the expected exact bit length of the leaked factor.  If
    omitted, balanced factors are assumed.  Unlike the old lookup table, this
    works for arbitrary modulus sizes and fails before allocating an
    impractically large lattice.
    """
    root_bits = _nonnegative_integer(unknown_bits, "unknown_bits")
    modulus_bits = _positive_integer(bitlength, "bitlength")
    normalized_safety = _nonnegative_integer(safety_bits, "safety_bits")
    dimension_limit = _positive_integer(max_dimension, "max_dimension")
    if dimension_limit < 2:
        raise ValueError("max_dimension must be at least two")

    if factor_bits is None:
        normalized_factor_bits = modulus_bits // 2
    else:
        normalized_factor_bits = _positive_integer(factor_bits, "factor_bits")
    if normalized_factor_bits >= modulus_bits:
        raise ValueError("factor_bits must be smaller than bitlength")
    factor_lower_exponent = normalized_factor_bits - 1
    if factor_lower_exponent <= 0:
        raise ValueError("factor_bits is too small for lattice parameter selection")

    effective_root_bits = root_bits + normalized_safety
    for dimension in range(2, dimension_limit + 1):
        for k in range(1, dimension):
            t = dimension - k
            if _parameter_condition(
                modulus_bits,
                factor_lower_exponent,
                effective_root_bits,
                k,
                t,
            ):
                return k, t
    raise ValueError(
        "no conservative (k, t) parameters fit max_dimension; "
        "reveal more factor bits or raise the resource limit"
    )


def _normalize_lattice_parameters(
    k: int | None,
    t: int | None,
    *,
    modulus: int,
    known_high: int,
    unknown_bits: int,
    safety_bits: int,
    max_dimension: int,
) -> tuple[int, int]:
    """Select automatic parameters or validate an explicit pair."""
    dimension_limit = _positive_integer(max_dimension, "max_dimension")
    if (k is None) != (t is None):
        raise ValueError("k and t must either both be supplied or both be omitted")
    if k is None:
        selected_k, selected_t = get_kt(
            unknown_bits,
            modulus.bit_length(),
            factor_bits=known_high.bit_length(),
            safety_bits=safety_bits,
            max_dimension=dimension_limit,
        )
    else:
        selected_k = _positive_integer(k, "k")
        selected_t = _positive_integer(t, "t")
    if selected_k + selected_t > dimension_limit:
        raise ValueError(
            f"lattice dimension {selected_k + selected_t} exceeds "
            f"max_dimension={dimension_limit}"
        )
    return selected_k, selected_t


def _auxiliary_polynomials(
    modulus: int,
    known_high: int,
    k: int,
    t: int,
) -> tuple[Any, ...]:
    """Construct the unscaled Howgrave--Graham auxiliary polynomials."""
    polynomial_ring = PolynomialRing(ZZ, "rsa_high_bits_x")
    x = polynomial_ring.gen()
    f = known_high + x
    shifts = [modulus ** (k - i) * f**i for i in range(k)]
    shifts.extend(x**i * f**k for i in range(t))
    return tuple(shifts)


def construct_lattice(
    prob: Sequence[Any],
    answer: Sequence[Any] | None = None,
    *,
    k: int | None = None,
    t: int | None = None,
    safety_bits: int = 0,
    max_dimension: int = 256,
) -> Any:
    """Construct the explicit scaled row lattice for a legacy problem tuple."""
    modulus, known_high, unknown_bits, bound = _normalize_problem_tuple(prob)
    selected_k, selected_t = _normalize_lattice_parameters(
        k,
        t,
        modulus=modulus,
        known_high=known_high,
        unknown_bits=unknown_bits,
        safety_bits=safety_bits,
        max_dimension=max_dimension,
    )
    shifts = _auxiliary_polynomials(
        modulus,
        known_high,
        selected_k,
        selected_t,
    )
    dimension = selected_k + selected_t
    lattice = Matrix(ZZ, dimension, dimension)
    for row_index, shift in enumerate(shifts):
        scaled = shift(bound * shift.parent().gen())
        for column_index, coefficient in enumerate(scaled.list()):
            lattice[row_index, column_index] = coefficient

    if lattice.rank() != dimension:
        raise RuntimeError("constructed lattice is unexpectedly rank deficient")

    if answer is not None:
        try:
            p, q = tuple(answer)
        except (TypeError, ValueError) as error:
            raise ValueError("answer must contain exactly (p, q)") from error
        p = _positive_integer(p, "answer[0]")
        q = _positive_integer(q, "answer[1]")
        if p * q != modulus:
            raise ValueError("answer does not factor N")
        if p - p % bound != known_high:
            raise ValueError("answer[0] does not match the supplied high bits")
        root = p - known_high
        prime_power = p**selected_k
        if any(shift(root) % prime_power != 0 for shift in shifts):
            raise RuntimeError("an auxiliary polynomial lost the required root")
    return lattice


def _reduce_lattice(
    basis: Any,
    effort: int,
    lattice_reduction: LatticeReduction | None,
) -> Any:
    """Reduce a full-rank integer lattice and validate lattice preservation."""
    try:
        integer_basis = Matrix(ZZ, basis)
    except (TypeError, ValueError) as error:
        raise TypeError("lattice must be an integer matrix") from error
    if integer_basis.nrows() == 0 or integer_basis.ncols() == 0:
        raise ValueError("lattice must be non-empty")
    if integer_basis.rank() != integer_basis.nrows():
        raise ValueError("lattice must have full row rank")
    if effort > integer_basis.nrows():
        raise ValueError(
            f"effort cannot exceed the lattice row count ({integer_basis.nrows()})"
        )

    if lattice_reduction is not None:
        reduced = lattice_reduction(integer_basis)
    elif effort == 1:
        reduced = integer_basis.LLL()
    else:
        reduced = integer_basis.BKZ(block_size=effort)
    try:
        reduced = Matrix(ZZ, reduced)
    except (TypeError, ValueError) as error:
        raise TypeError("lattice reduction must return an integer matrix") from error
    if reduced.dimensions() != integer_basis.dimensions():
        raise ValueError("lattice reduction changed the matrix dimensions")
    if reduced.rank() != integer_basis.nrows():
        raise ValueError("lattice reduction returned a rank-deficient basis")

    coordinates: list[list[int]] = []
    for row in reduced.rows():
        try:
            representation = integer_basis.solve_left(row)
        except ValueError as error:
            raise ValueError(
                "lattice reduction returned a row outside the input span"
            ) from error
        coordinate_row: list[int] = []
        for coefficient in representation:
            rational = QQ(coefficient)
            if rational.denominator() != 1:
                raise ValueError(
                    "lattice reduction returned a row outside the input lattice"
                )
            coordinate_row.append(int(rational))
        coordinates.append(coordinate_row)
    transformation = Matrix(ZZ, coordinates)
    if transformation.nrows() == transformation.ncols():
        if abs(int(transformation.det())) != 1:
            raise ValueError("lattice reduction changed the generated lattice")
    return reduced


def roots_from_reduced_lattice(
    unknown_bits: int,
    reduced_lattice: Any,
    *,
    rows: int | None = None,
) -> list[int]:
    """Extract every non-negative bounded integral root from reduced rows."""
    normalized_unknown = _nonnegative_integer(unknown_bits, "unknown_bits")
    bound = 1 << normalized_unknown
    try:
        lattice = Matrix(ZZ, reduced_lattice)
    except (TypeError, ValueError) as error:
        raise TypeError("L_red must be an integer matrix") from error
    if lattice.nrows() == 0 or lattice.ncols() == 0:
        raise ValueError("L_red must be non-empty")
    if rows is None:
        row_count = lattice.nrows()
    else:
        row_count = min(_positive_integer(rows, "rows"), lattice.nrows())

    polynomial_ring = PolynomialRing(ZZ, "rsa_high_bits_x")
    roots: set[int] = set()
    factors = [bound**degree for degree in range(lattice.ncols())]
    for row_index in range(row_count):
        coefficients: list[int] = []
        for entry, factor in zip(lattice[row_index], factors, strict=True):
            if entry % factor != 0:
                raise ValueError(
                    "a reduced coefficient is not exactly divisible by its "
                    "column scale"
                )
            coefficients.append(int(entry // factor))
        polynomial = polynomial_ring(coefficients)
        if polynomial == 0 or polynomial.degree() <= 0:
            continue
        for root, _multiplicity in polynomial.roots(ring=ZZ):
            integer_root = int(root)
            if 0 <= integer_root < bound and polynomial(integer_root) == 0:
                roots.add(integer_root)
    return sorted(roots)


def solve_from_reduced_lattice(
    unknown_bits: int,
    L_red: Any,
) -> int | None:
    """Compatibility wrapper returning the first root, or ``None`` on failure."""
    roots = roots_from_reduced_lattice(unknown_bits, L_red)
    return roots[0] if roots else None


def _factor_from_roots(
    modulus: int,
    known_high: int,
    roots: Iterable[int],
) -> int | None:
    """Return the first candidate that is an actual proper divisor."""
    for root in sorted(set(map(int, roots))):
        candidate = known_high + root
        if 1 < candidate < modulus and modulus % candidate == 0:
            return candidate
    return None


def _bruteforce_factor(
    modulus: int,
    known_high: int,
    bound: int,
) -> int | None:
    """Exhaust a deliberately tiny low-bit interval."""
    if modulus % 2:
        first_root = (1 - known_high) % 2
        roots = range(first_root, bound, 2)
    else:
        roots = range(bound)
    for root in roots:
        candidate = known_high + root
        if 1 < candidate < modulus and modulus % candidate == 0:
            return candidate
    return None


def _factor_beta(modulus: int, factor_lower_bound: int) -> Any:
    """Compute a close rational beta proven below the factor's log ratio."""
    denominator = min(2048, max(64, modulus.bit_length()))
    lower_power = factor_lower_bound**denominator
    low = 0
    high = denominator
    while low < high:
        middle = (low + high + 1) // 2
        if modulus**middle <= lower_power:
            low = middle
        else:
            high = middle - 1
    if low == 0:
        raise ValueError("the known factor lower bound is too small")
    return QQ(low) / denominator


def _sage_epsilon_and_dimension(
    modulus: int,
    bound: int,
    beta: Any,
    epsilon: Any | None,
) -> tuple[Any, int]:
    """Select a Sage epsilon that covers ``bound`` and estimate its dimension."""
    modulus_log_lower = modulus.bit_length() - 1
    bound_bits_with_half_factor = bound.bit_length()
    exponent_needed = QQ(bound_bits_with_half_factor) / modulus_log_lower
    gap = beta**2 - exponent_needed
    if gap <= 0:
        raise ValueError(
            "the unknown-bit bound is outside the conservative Coppersmith "
            "range for the known factor size"
        )
    if epsilon is None:
        normalized_epsilon = gap / 2
    else:
        try:
            normalized_epsilon = QQ(epsilon)
        except (TypeError, ValueError) as error:
            raise TypeError("epsilon must be rational or None") from error
        if normalized_epsilon <= 0:
            raise ValueError("epsilon must be positive")
        if normalized_epsilon >= gap:
            raise ValueError(
                "epsilon is too large for the requested root bound"
            )

    m = max(beta**2 / normalized_epsilon, 7 * beta).ceil()
    t = (m * (1 / beta - 1)).floor()
    return normalized_epsilon, int(m + t)


def recover_factor_from_high_bits(
    N: int,
    p_msb: int,
    unknown_bits: int,
    *,
    method: RecoveryMethod = "auto",
    effort: int = 1,
    k: int | None = None,
    t: int | None = None,
    epsilon: Any | None = None,
    safety_bits: int = 0,
    max_dimension: int = 256,
    brute_force_bits: int = 18,
    lattice_reduction: LatticeReduction | None = None,
    rows: int | None = None,
    verbose: bool = False,
) -> int | None:
    """Recover and verify an RSA factor from its known most-significant bits.

    ``method='auto'`` exhausts very small intervals, otherwise uses Sage's
    univariate Coppersmith implementation.  Supplying ``k``, ``t``, a custom
    reducer, or ``effort > 1`` makes ``auto`` select the explicit lattice path.
    """
    modulus, known_high, normalized_unknown, bound = _normalize_problem(
        N,
        p_msb,
        unknown_bits,
    )
    report = _verbose_printer(verbose)
    normalized_effort = _normalize_effort(effort, lattice_reduction)
    dimension_limit = _positive_integer(max_dimension, "max_dimension")
    brute_limit = _nonnegative_integer(brute_force_bits, "brute_force_bits")
    if method not in {"auto", "bruteforce", "sage", "lattice"}:
        raise ValueError(
            "method must be 'auto', 'bruteforce', 'sage', or 'lattice'"
        )

    immediate = gcd(modulus, known_high)
    if 1 < immediate < modulus:
        report("The known portion already reveals a factor by gcd")
        return immediate

    selected_method = method
    if selected_method == "auto":
        if (
            k is not None
            or t is not None
            or lattice_reduction is not None
            or normalized_effort > 1
        ):
            selected_method = "lattice"
        elif normalized_unknown <= brute_limit:
            selected_method = "bruteforce"
        else:
            selected_method = "sage"
    report(f"Using {selected_method} recovery")

    if selected_method == "bruteforce":
        if normalized_unknown > brute_limit:
            raise ValueError(
                "the requested brute-force interval exceeds brute_force_bits"
            )
        return _bruteforce_factor(modulus, known_high, bound)

    if selected_method == "sage":
        if k is not None or t is not None:
            raise ValueError("k and t are only used by method='lattice'")
        if lattice_reduction is not None or normalized_effort != 1:
            raise ValueError(
                "custom reduction and effort are only used by method='lattice'"
            )
        beta = _factor_beta(modulus, known_high)
        selected_epsilon, estimated_dimension = _sage_epsilon_and_dimension(
            modulus,
            bound,
            beta,
            epsilon,
        )
        if estimated_dimension > dimension_limit:
            raise ValueError(
                f"estimated Sage lattice dimension {estimated_dimension} "
                f"exceeds max_dimension={dimension_limit}"
            )
        report(
            f"beta={beta}, epsilon={selected_epsilon}, "
            f"estimated dimension={estimated_dimension}"
        )
        polynomial_ring = PolynomialRing(
            Zmod(modulus),
            "rsa_high_bits_x",
            implementation="NTL",
        )
        x = polynomial_ring.gen()
        roots = find_small_roots(
            x + known_high,
            (bound,),
            method="sage",
            beta=beta,
            epsilon=selected_epsilon,
            brute_force_limit=0,
        )
        return _factor_from_roots(
            modulus,
            known_high,
            (root[0] for root in roots),
        )

    if epsilon is not None:
        raise ValueError("epsilon is only used by method='sage'")
    problem = (modulus, known_high, normalized_unknown)
    lattice = construct_lattice(
        problem,
        k=k,
        t=t,
        safety_bits=safety_bits,
        max_dimension=dimension_limit,
    )
    report(f"Explicit lattice dimensions: {lattice.dimensions()}")
    reduced = _reduce_lattice(
        lattice,
        normalized_effort,
        lattice_reduction,
    )
    roots = roots_from_reduced_lattice(
        normalized_unknown,
        reduced,
        rows=rows,
    )
    return _factor_from_roots(modulus, known_high, roots)


def attack_full(
    unknown_bits: int,
    problem: Sequence[Any] | None = None,
    verbose: bool = False,
    *,
    bitlength: int = BITLENGTH,
    **recovery_options: Any,
) -> int | None:
    """Compatibility wrapper that generates or accepts a legacy problem."""
    normalized_unknown = _nonnegative_integer(unknown_bits, "unknown_bits")
    report = _verbose_printer(verbose)
    if problem is None:
        answer, generated_problem = generate_problem_instance(
            normalized_unknown,
            bitlength,
        )
        modulus, known_high, problem_unknown = generated_problem
        report(f"Generated a modulus with secret factor {hex(answer[0])}")
    else:
        modulus, known_high, problem_unknown, _bound = _normalize_problem_tuple(
            problem
        )
        if problem_unknown != normalized_unknown:
            raise ValueError(
                "unknown_bits disagrees with the value stored in problem"
            )

    factor = recover_factor_from_high_bits(
        modulus,
        known_high,
        problem_unknown,
        verbose=verbose,
        **recovery_options,
    )
    if factor is None:
        report("Recovery failed")
    else:
        report(f"Recovered factor {hex(factor)}")
    return factor


def _argument_integer(value: str) -> int:
    """Parse decimal, hexadecimal, octal, or binary CLI integers."""
    try:
        return int(value, 0)
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"invalid integer: {value!r}") from error


def main(argv: Sequence[str] | None = None) -> int:
    """Run a generated demonstration or attack a supplied public instance."""
    parser = argparse.ArgumentParser(
        description="Recover an RSA factor from known most-significant bits",
    )
    parser.add_argument(
        "--unknown-bits",
        type=int,
        default=200,
        help="number of unknown low factor bits (default: 200)",
    )
    parser.add_argument(
        "--bitlength",
        type=int,
        default=BITLENGTH,
        help="generated modulus size (default: 1024)",
    )
    parser.add_argument("--N", dest="modulus", type=_argument_integer)
    parser.add_argument("--known-high", type=_argument_integer)
    parser.add_argument(
        "--method",
        choices=("auto", "bruteforce", "sage", "lattice"),
        default="auto",
    )
    parser.add_argument("--effort", type=int, default=1)
    parser.add_argument("--max-dimension", type=int, default=256)
    parser.add_argument("--brute-force-bits", type=int, default=18)
    parser.add_argument("--verbose", action="store_true")
    arguments = parser.parse_args(argv)

    if (arguments.modulus is None) != (arguments.known_high is None):
        parser.error("--N and --known-high must be supplied together")
    if arguments.modulus is None:
        problem = None
    else:
        problem = (
            arguments.modulus,
            arguments.known_high,
            arguments.unknown_bits,
        )

    factor = attack_full(
        arguments.unknown_bits,
        problem,
        arguments.verbose,
        bitlength=arguments.bitlength,
        method=arguments.method,
        effort=arguments.effort,
        max_dimension=arguments.max_dimension,
        brute_force_bits=arguments.brute_force_bits,
    )
    if factor is None:
        print("factor recovery failed")
        return 1
    print(hex(factor))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "BITLENGTH",
    "attack_full",
    "construct_lattice",
    "gen_rsakey",
    "generate_problem_instance",
    "get_kt",
    "main",
    "recover_factor_from_high_bits",
    "roots_from_reduced_lattice",
    "solve_from_reduced_lattice",
]
