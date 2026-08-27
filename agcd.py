"""Recover approximate common divisors with an orthogonal-lattice attack."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from itertools import combinations
from math import gcd, isqrt
from operator import index as integer_index
from typing import Any

from sage.all import AA, Matrix, ZZ, vector


LatticeReduction = Callable[[Any], Any]


class AGCDRecoveryError(RuntimeError):
    """Raised when lattice reduction does not reveal a valid AGCD solution."""


@dataclass(frozen=True, slots=True)
class AGCDResult:
    """A validated approximate-common-divisor solution."""

    divisor: int
    quotients: tuple[int, ...]
    errors: tuple[int, ...]


def _integer_parameter(value: Any, name: str) -> int:
    """Return an exact Python integer parameter, rejecting implicit coercions."""
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer, not bool")

    try:
        return integer_index(value)
    except TypeError as error:
        raise TypeError(f"{name} must be an integer") from error


def _validate_inputs(
    samples: Iterable[int],
    error_bits: int,
    effort: int,
    rounded: bool,
    lattice_reduction: LatticeReduction | None,
) -> tuple[tuple[Any, ...], int, int]:
    """Validate and normalize public attack parameters."""
    try:
        raw_samples = tuple(samples)
    except TypeError as error:
        raise TypeError("samples must be an iterable of positive integers") from error

    if len(raw_samples) < 2:
        raise ValueError("at least two samples are required")

    integer_samples: list[Any] = []
    for index, sample in enumerate(raw_samples):
        if isinstance(sample, bool):
            raise TypeError(f"samples[{index}] must be an integer, not bool")
        try:
            integer = ZZ(integer_index(sample))
        except TypeError as error:
            raise TypeError(f"samples[{index}] must be an integer") from error
        if integer <= 0:
            raise ValueError(f"samples[{index}] must be positive")
        integer_samples.append(integer)

    normalized_error_bits = _integer_parameter(error_bits, "error_bits")
    if normalized_error_bits < 0:
        raise ValueError("error_bits must be non-negative")

    normalized_effort = _integer_parameter(effort, "effort")
    if not 1 <= normalized_effort <= len(integer_samples):
        raise ValueError(
            f"effort must be between 1 and the sample count ({len(integer_samples)})"
        )

    if not isinstance(rounded, bool):
        raise TypeError("rounded must be a bool")
    if lattice_reduction is not None and not callable(lattice_reduction):
        raise TypeError("lattice_reduction must be callable or None")
    if lattice_reduction is not None and normalized_effort != 1:
        raise ValueError("effort must be 1 when lattice_reduction is supplied")

    return tuple(integer_samples), normalized_error_bits, normalized_effort


def _nearest_scaled_square_root(numerator: int, denominator: int) -> int:
    """Compute ``round(sqrt(numerator) / denominator)`` exactly."""
    lower = isqrt(numerator) // denominator
    midpoint_comparison = (2 * lower + 1) ** 2 * denominator**2
    if 4 * numerator >= midpoint_comparison:
        return lower + 1
    return lower


def _scaled_lattice(samples: tuple[Any, ...], error_bits: int) -> Any:
    """Construct the original scaled orthogonal lattice."""
    sample_count = len(samples)
    radicand = sample_count << (2 * error_bits)
    alpha = max(
        1,
        _nearest_scaled_square_root(radicand, sample_count - 1),
    )

    basis = Matrix(ZZ, sample_count, sample_count + 1)
    for index, sample in enumerate(samples):
        basis[index, 0] = sample
        basis[index, index + 1] = alpha
    return basis


def _rounded_lattice(samples: tuple[Any, ...], error_bits: int) -> Any:
    """Construct the smaller rounded lattice from Xu, Sarkar, and Hu."""
    sample_count = len(samples)
    square_root = AA(sample_count).sqrt()
    error_bound = ZZ(1) << error_bits
    alpha = (
        square_root
        * error_bound
        / ((sample_count - 1) * (square_root + 1))
    )

    rows = []
    half = AA(1) / 2
    for row_index, sample in enumerate(samples):
        scaled_sample = ZZ((AA(sample) / alpha + half).floor())
        row = [scaled_sample]
        row.extend(ZZ(row_index == column) for column in range(sample_count))
        rows.append(row)
    return Matrix(ZZ, rows)


def _reduce_lattice(
    basis: Any,
    effort: int,
    lattice_reduction: LatticeReduction | None,
) -> Any:
    """Reduce a lattice with a custom reducer, LLL, or effort-sized BKZ."""
    if lattice_reduction is not None:
        reduced = lattice_reduction(basis)
    elif effort == 1:
        reduced = basis.LLL()
    else:
        reduced = basis.BKZ(block_size=effort)

    try:
        reduced = Matrix(ZZ, reduced)
    except (TypeError, ValueError) as error:
        raise AGCDRecoveryError(
            "lattice reduction did not return an integer matrix"
        ) from error

    if reduced.dimensions() != basis.dimensions() or reduced.rank() != basis.nrows():
        raise AGCDRecoveryError(
            "lattice reduction returned a matrix with an invalid shape or rank"
        )
    return reduced


def _primitive_positive_kernel(relation_matrix: Any) -> tuple[int, ...] | None:
    """Return the primitive positive generator of a one-dimensional kernel."""
    kernel = relation_matrix.right_kernel()
    if kernel.dimension() != 1:
        return None

    candidate = [int(entry) for entry in kernel.basis()[0]]
    common_factor = 0
    for entry in candidate:
        common_factor = gcd(common_factor, abs(entry))
    if common_factor == 0:
        return None
    candidate = [entry // common_factor for entry in candidate]

    if all(entry < 0 for entry in candidate):
        candidate = [-entry for entry in candidate]
    if not all(entry > 0 for entry in candidate):
        return None
    return tuple(candidate)


def _ceil_div(numerator: int, denominator: int) -> int:
    """Return the mathematical ceiling of an integer ratio."""
    return -((-numerator) // denominator)


def _fit_divisor(
    samples: tuple[Any, ...],
    quotients: tuple[int, ...],
    error_bits: int,
) -> AGCDResult | None:
    """Find and validate a divisor compatible with a quotient candidate."""
    maximum_error = (1 << error_bits) - 1
    lower_bound = 2
    upper_bound: int | None = None

    for sample, quotient in zip(samples, quotients, strict=True):
        sample = int(sample)
        lower_bound = max(
            lower_bound,
            _ceil_div(sample - maximum_error, quotient),
        )
        sample_upper_bound = (sample + maximum_error) // quotient
        upper_bound = (
            sample_upper_bound
            if upper_bound is None
            else min(upper_bound, sample_upper_bound)
        )

    if upper_bound is None or lower_bound > upper_bound:
        return None

    divisor_candidates = {lower_bound, upper_bound}
    for sample, quotient in zip(samples, quotients, strict=True):
        approximate_divisor, remainder = divmod(int(sample), quotient)
        if 2 * remainder >= quotient:
            approximate_divisor += 1
        divisor_candidates.add(
            min(upper_bound, max(lower_bound, approximate_divisor))
        )

    def residual_score(divisor: int) -> tuple[int, int, int]:
        residuals = [
            abs(int(sample) - divisor * quotient)
            for sample, quotient in zip(samples, quotients, strict=True)
        ]
        return max(residuals), sum(residuals), -divisor

    divisor = min(divisor_candidates, key=residual_score)
    errors = tuple(
        int(sample) - divisor * quotient
        for sample, quotient in zip(samples, quotients, strict=True)
    )
    if any(abs(error) > maximum_error for error in errors):
        return None

    return AGCDResult(divisor, quotients, errors)


def _candidate_results(
    reduced_basis: Any,
    samples: tuple[Any, ...],
    error_bits: int,
) -> list[AGCDResult]:
    """Extract and validate quotient candidates from reduced-basis relations."""
    sample_count = len(samples)
    coefficient_matrix = reduced_basis.matrix_from_columns(
        list(range(1, sample_count + 1))
    )

    results: list[AGCDResult] = []
    seen_quotients: set[tuple[int, ...]] = set()
    for row_indices in combinations(range(sample_count), sample_count - 1):
        relations = coefficient_matrix.matrix_from_rows(row_indices)
        quotients = _primitive_positive_kernel(relations)
        if quotients is None or quotients in seen_quotients:
            continue
        seen_quotients.add(quotients)

        result = _fit_divisor(samples, quotients, error_bits)
        if result is not None:
            results.append(result)
    return results


def recover_agcd(
    samples: Iterable[int],
    error_bits: int,
    *,
    effort: int = 1,
    rounded: bool = True,
    lattice_reduction: LatticeReduction | None = None,
) -> AGCDResult:
    """Recover an approximate common divisor and its quotient decomposition.

    The samples must satisfy ``a_i = p*q_i + r_i`` with ``p > 1``, positive
    ``q_i``, and ``abs(r_i) < 2**error_bits``. Recovery is heuristic and
    requires enough samples relative to the divisor, sample, and error sizes.

    By default, the attack uses the rounded lattice proposed by Xu, Sarkar, and
    Hu, which has smaller coefficients than the original scaled construction.
    Set ``rounded=False`` to use the original construction.

    ``effort=1`` selects LLL. Values from 2 through the number of samples select
    BKZ with that block size; larger values usually improve reduction quality at
    a rapidly increasing cost. A custom one-argument reduction callback may be
    supplied instead, in which case ``effort`` must remain 1.

    Args:
        samples: Approximate multiples of the unknown common divisor.
        error_bits: Strict bit bound on each signed error.
        effort: Lattice-reduction strength, from 1 to the sample count.
        rounded: Whether to use the faster rounded lattice construction.
        lattice_reduction: Optional custom lattice-reduction callback.

    Returns:
        A validated divisor, quotient tuple, and signed-error tuple.

    Raises:
        TypeError: If an argument has an invalid type.
        ValueError: If the parameters are outside their supported ranges.
        AGCDRecoveryError: If reduction produces no valid candidate.
    """
    normalized_samples, normalized_error_bits, normalized_effort = (
        _validate_inputs(
            samples,
            error_bits,
            effort,
            rounded,
            lattice_reduction,
        )
    )

    basis = (
        _rounded_lattice(normalized_samples, normalized_error_bits)
        if rounded
        else _scaled_lattice(normalized_samples, normalized_error_bits)
    )
    reduced_basis = _reduce_lattice(
        basis,
        normalized_effort,
        lattice_reduction,
    )
    results = _candidate_results(
        reduced_basis,
        normalized_samples,
        normalized_error_bits,
    )
    if not results:
        raise AGCDRecoveryError(
            "lattice reduction did not reveal quotients consistent with the "
            "declared error bound; try more samples or a higher effort"
        )

    return min(
        results,
        key=lambda result: (
            max(abs(error) for error in result.errors),
            sum(abs(error) for error in result.errors),
            -result.divisor,
        ),
    )


def agcd(
    samples: Iterable[int],
    rho: int,
    *,
    effort: int = 1,
    rounded: bool = True,
    lattice_reduction: LatticeReduction | None = None,
) -> Any:
    """Recover the quotient vector, preserving the original function's API."""
    result = recover_agcd(
        samples,
        rho,
        effort=effort,
        rounded=rounded,
        lattice_reduction=lattice_reduction,
    )
    return vector(ZZ, result.quotients)
