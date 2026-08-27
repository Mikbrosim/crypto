"""Recover secrets and errors from small Learning With Errors instances."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from math import lcm
from operator import index as integer_index
from typing import Any, Literal

from sage.all import Matrix, QQ, ZZ, vector


LWEFormulation = Literal["auto", "error", "augmented"]
LWEMethod = Literal["auto", "babai", "kannan"]
LatticeReduction = Callable[[Any], Any]


@dataclass(frozen=True, slots=True)
class _LWECandidate:
    secret: tuple[int, ...]
    error: tuple[int, ...]


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


def center(c: int, q: int) -> int:
    """Return the canonical centered representative of ``c`` modulo ``q``."""
    modulus = _exact_integer(q, "q")
    if modulus <= 0:
        raise ValueError("q must be positive")
    residue = _exact_integer(c, "c") % modulus
    if 2 * residue > modulus:
        residue -= modulus
    return residue


def _normalize_problem(
    a: Iterable[Iterable[int]],
    b: Iterable[int],
    N: int | None,
) -> tuple[tuple[tuple[int, ...], ...], tuple[int, ...], int | None]:
    """Validate an LWE matrix, target vector, and optional modulus."""
    try:
        raw_rows = tuple(a)
    except TypeError as error:
        raise TypeError("a must be a non-empty matrix of integers") from error
    if not raw_rows:
        raise ValueError("a must contain at least one sample")
    rows = tuple(
        _integer_tuple(row, f"a[{index}]")
        for index, row in enumerate(raw_rows)
    )
    secret_dimension = len(rows[0])
    if secret_dimension == 0:
        raise ValueError("a must contain at least one secret column")
    if any(len(row) != secret_dimension for row in rows):
        raise ValueError("all rows of a must have the same length")

    target = _integer_tuple(b, "b")
    if len(target) != len(rows):
        raise ValueError("a and b must contain the same number of samples")

    if N is None:
        modulus = None
    else:
        raw_modulus = _exact_integer(N, "N")
        if raw_modulus == 0:
            modulus = None
        elif raw_modulus <= 1:
            raise ValueError("N must be zero, None, or greater than one")
        else:
            modulus = raw_modulus

    if modulus is not None:
        rows = tuple(
            tuple(value % modulus for value in row)
            for row in rows
        )
        target = tuple(value % modulus for value in target)
    return rows, target, modulus


def _normalize_error_bounds(
    error_bound: int | Sequence[int] | None,
    sample_count: int,
) -> tuple[int, ...] | None:
    """Expand an optional scalar or per-sample absolute error bound."""
    if error_bound is None:
        return None
    try:
        scalar = _exact_integer(error_bound, "error_bound")
    except TypeError:
        bounds = _integer_tuple(error_bound, "error_bound")
        if len(bounds) != sample_count:
            raise ValueError("error_bound must have one value per sample")
    else:
        bounds = (scalar,) * sample_count
    if any(bound < 0 for bound in bounds):
        raise ValueError("error bounds must be non-negative")
    return bounds


def _normalize_secret_bits(s_bits: int | None) -> int | None:
    """Validate an optional strict bit bound on secret coordinates."""
    if s_bits is None:
        return None
    normalized = _exact_integer(s_bits, "s_bits")
    if normalized < 0:
        raise ValueError("s_bits must be non-negative")
    return normalized


def _normalize_reduction(
    effort: int,
    lattice_reduction: LatticeReduction | None,
) -> int:
    """Validate LLL/BKZ and custom-reducer controls."""
    normalized = _exact_integer(effort, "effort")
    if normalized < 1:
        raise ValueError("effort must be at least 1")
    if lattice_reduction is not None and not callable(lattice_reduction):
        raise TypeError("lattice_reduction must be callable or None")
    if lattice_reduction is not None and normalized != 1:
        raise ValueError("effort must be 1 with a custom lattice reduction")
    return normalized


def _positive_rational(value: Any, name: str) -> Any:
    """Return a strictly positive Sage rational."""
    try:
        rational = QQ(value)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{name} must be rational") from error
    if rational <= 0:
        raise ValueError(f"{name} must be positive")
    return rational


def _reduce_basis(
    basis: Any,
    effort: int,
    lattice_reduction: LatticeReduction | None,
) -> Any:
    """Reduce and validate a full-row-rank integer basis."""
    if basis.rank() != basis.nrows():
        raise ValueError("lattice basis must have full row rank")
    if effort > basis.nrows():
        raise ValueError(
            f"effort cannot exceed the lattice row count ({basis.nrows()})"
        )
    if lattice_reduction is not None:
        reduced = lattice_reduction(basis)
    elif effort == 1:
        reduced = basis.LLL()
    else:
        reduced = basis.BKZ(block_size=effort)

    try:
        reduced = Matrix(ZZ, reduced)
    except (TypeError, ValueError) as error:
        raise TypeError("lattice reduction must return an integer matrix") from error
    if reduced.dimensions() != basis.dimensions():
        raise ValueError("lattice reduction changed the basis dimensions")
    if reduced.rank() != basis.nrows():
        raise ValueError("lattice reduction returned a rank-deficient basis")
    return reduced


def _integralize_lattice(basis: Any, target: Any) -> tuple[Any, Any, int]:
    """Clear all basis and target denominators with one global scale."""
    rational_basis = Matrix(QQ, basis)
    rational_target = vector(QQ, target)
    denominators = [
        int(QQ(entry).denominator())
        for entry in rational_basis.list() + list(rational_target)
    ]
    denominator = lcm(*denominators) if denominators else 1
    return (
        Matrix(ZZ, rational_basis * denominator),
        vector(ZZ, rational_target * denominator),
        denominator,
    )


def _babai_reduced(basis: Any, target: Any) -> Any:
    """Apply nearest plane to an already reduced full-rank row basis."""
    gram_schmidt = basis.gram_schmidt()[0]
    residual = vector(basis.base_ring(), target)
    for index in reversed(range(basis.nrows())):
        orthogonal = gram_schmidt[index]
        squared_norm = orthogonal * orthogonal
        if squared_norm == 0:
            raise ValueError("Babai requires a full-row-rank basis")
        coefficient = ((residual * orthogonal) / squared_norm).round()
        residual -= coefficient * basis[index]
    return target - residual


def _babai_point(
    basis: Any,
    target: Any,
    effort: int,
    lattice_reduction: LatticeReduction | None,
) -> Any:
    """Reduce an exact lattice and return Babai's lattice point."""
    integer_basis, integer_target, denominator = _integralize_lattice(
        basis,
        target,
    )
    reduced = _reduce_basis(integer_basis, effort, lattice_reduction)
    point = _babai_reduced(reduced, integer_target)
    return vector(QQ, [QQ(entry) / denominator for entry in point])


def _kannan_points(
    basis: Any,
    target: Any,
    embedding_factor: Any,
    effort: int,
    lattice_reduction: LatticeReduction | None,
    report: Callable[..., None],
) -> list[Any]:
    """Return lattice points exposed by Kannan's marked embedding."""
    factor = _positive_rational(embedding_factor, "embedding_factor")
    rational_basis = Matrix(QQ, basis)
    rational_target = vector(QQ, target)
    denominators = [
        int(QQ(entry).denominator())
        for entry in rational_basis.list() + list(rational_target) + [factor]
    ]
    denominator = lcm(*denominators) if denominators else 1
    integer_basis = Matrix(ZZ, rational_basis * denominator)
    integer_target = vector(ZZ, rational_target * denominator)
    integer_factor = int(factor * denominator)

    rows = [list(row) + [0] for row in integer_basis.rows()]
    rows.append(list(-integer_target) + [integer_factor])
    embedded_basis = Matrix(ZZ, rows)
    report(f"Kannan lattice dimensions: {embedded_basis.dimensions()}")
    reduced = _reduce_basis(embedded_basis, effort, lattice_reduction)

    points: list[Any] = []
    seen: set[tuple[Any, ...]] = set()
    for row in reduced.rows():
        marker = int(row[-1])
        if abs(marker) != integer_factor:
            continue
        sign = 1 if marker > 0 else -1
        point_integer = [
            sign * int(entry) + integer_target[index]
            for index, entry in enumerate(row[:-1])
        ]
        point = vector(
            QQ,
            [QQ(entry) / denominator for entry in point_integer],
        )
        key = tuple(point)
        if key not in seen:
            seen.add(key)
            points.append(point)
    return points


def _cvp_points(
    basis: Any,
    target: Any,
    method: LWEMethod,
    embedding_factor: Any,
    effort: int,
    lattice_reduction: LatticeReduction | None,
    report: Callable[..., None],
) -> list[Any]:
    """Dispatch a CVP instance to Babai, Kannan, or Kannan with fallback."""
    if basis.nrows() == 0:
        return [vector(QQ, [0] * basis.ncols())]
    if method == "babai":
        return [_babai_point(basis, target, effort, lattice_reduction)]
    points = _kannan_points(
        basis,
        target,
        embedding_factor,
        effort,
        lattice_reduction,
        report,
    )
    if points or method == "kannan":
        return points
    report("Kannan exposed no marked row; falling back to Babai")
    return [_babai_point(basis, target, effort, lattice_reduction)]


def _row_lattice_basis(rows: Sequence[Sequence[int]], ambient: int) -> Any:
    """Replace a possibly dependent generating set with a row-lattice basis."""
    if not rows or not any(any(row) for row in rows):
        return Matrix(ZZ, 0, ambient)
    generators = Matrix(ZZ, rows)
    return Matrix(ZZ, generators.row_module().basis_matrix())


def _error_lattice(
    a: Sequence[Sequence[int]],
    modulus: int | None,
) -> Any:
    """Construct the q-ary lattice containing every ``A*s + q*k``."""
    sample_count = len(a)
    secret_dimension = len(a[0])
    rows = [
        [a[sample][secret] for sample in range(sample_count)]
        for secret in range(secret_dimension)
    ]
    if modulus is not None:
        rows.extend(
            [modulus * int(row == column) for column in range(sample_count)]
            for row in range(sample_count)
        )
    return _row_lattice_basis(rows, sample_count)


def _secret_interval(
    s_bits: int,
    signed_secret: bool,
) -> tuple[int, int, Any, Any]:
    """Return inclusive bounds, center, and radius for secret coordinates."""
    limit = 1 << s_bits
    if signed_secret:
        lower, upper = -(limit - 1), limit - 1
    else:
        lower, upper = 0, limit - 1
    interval_center = QQ(lower + upper) / 2
    interval_radius = max(QQ(1), QQ(upper - lower) / 2)
    return lower, upper, interval_center, interval_radius


def _augmented_lattice(
    a: Sequence[Sequence[int]],
    b: Sequence[int],
    modulus: int | None,
    s_bits: int,
    signed_secret: bool,
    error_bounds: Sequence[int] | None,
) -> tuple[Any, Any, Any]:
    """Construct a small-secret primal lattice targeting ``(b, center(s))``."""
    sample_count = len(a)
    secret_dimension = len(a[0])
    _, _, secret_center, secret_radius = _secret_interval(
        s_bits,
        signed_secret,
    )
    error_scale = max(error_bounds) if error_bounds else 1
    error_scale = max(1, error_scale)
    secret_scale = QQ(error_scale) / secret_radius
    ambient = sample_count + secret_dimension
    rows: list[list[Any]] = []

    if modulus is not None:
        for sample in range(sample_count):
            row = [0] * ambient
            row[sample] = modulus
            rows.append(row)
    for secret in range(secret_dimension):
        row = [a[sample][secret] for sample in range(sample_count)]
        row.extend(
            secret_scale * int(secret == coordinate)
            for coordinate in range(secret_dimension)
        )
        rows.append(row)

    basis = Matrix(QQ, rows)
    target = vector(
        QQ,
        list(b)
        + [secret_center * secret_scale] * secret_dimension,
    )
    return basis, target, secret_scale


def _integer_solution_coset(
    a: Sequence[Sequence[int]],
    right_hand_side: Sequence[int],
    modulus: int | None,
) -> tuple[Any, Any] | None:
    """Compute a particular secret and its homogeneous solution lattice."""
    sample_count = len(a)
    secret_dimension = len(a[0])
    unknown_count = secret_dimension + (sample_count if modulus is not None else 0)
    equations = Matrix(ZZ, sample_count, unknown_count)
    rhs = vector(ZZ, right_hand_side)
    for row_index, row in enumerate(a):
        for column_index, coefficient in enumerate(row):
            equations[row_index, column_index] = coefficient
        if modulus is not None:
            equations[row_index, secret_dimension + row_index] = modulus

    diagonal, left_transform, right_transform = equations.smith_form()
    transformed_rhs = left_transform * rhs
    rank = equations.rank()
    transformed_solution = vector(ZZ, unknown_count)
    for index in range(rank):
        diagonal_value = diagonal[index, index]
        if transformed_rhs[index] % diagonal_value:
            return None
        transformed_solution[index] = transformed_rhs[index] // diagonal_value
    if any(transformed_rhs[index] for index in range(rank, sample_count)):
        return None

    full_particular = right_transform * transformed_solution
    particular = vector(ZZ, full_particular[:secret_dimension])
    generators = [
        list(right_transform.column(index)[:secret_dimension])
        for index in range(rank, unknown_count)
    ]
    generators = [generator for generator in generators if any(generator)]
    if generators:
        kernel_basis = Matrix(ZZ, generators).row_module().basis_matrix()
    else:
        kernel_basis = Matrix(ZZ, 0, secret_dimension)
    return particular, kernel_basis


def _error_within_bounds(
    error: Sequence[int],
    bounds: Sequence[int] | None,
) -> bool:
    """Check optional coordinate-wise absolute error bounds."""
    return bounds is None or all(
        abs(value) <= bound
        for value, bound in zip(error, bounds, strict=True)
    )


def _secret_within_bounds(
    secret: Sequence[int],
    s_bits: int | None,
    signed_secret: bool,
) -> bool:
    """Check the optional strict secret bit bound."""
    if s_bits is None:
        return True
    lower, upper, _, _ = _secret_interval(s_bits, signed_secret)
    return all(lower <= value <= upper for value in secret)


def _valid_equations(
    a: Sequence[Sequence[int]],
    b: Sequence[int],
    secret: Sequence[int],
    error: Sequence[int],
    modulus: int | None,
) -> bool:
    """Verify every recovered LWE equation exactly."""
    for row, target, error_value in zip(a, b, error, strict=True):
        residual = (
            sum(
                coefficient * value
                for coefficient, value in zip(row, secret, strict=True)
            )
            + error_value
            - target
        )
        if modulus is None:
            if residual != 0:
                return False
        elif residual % modulus:
            return False
    return True


def _integral_coordinates(values: Sequence[Any]) -> tuple[int, ...] | None:
    """Convert exact rational coordinates to integers without truncation."""
    result: list[int] = []
    for value in values:
        rational = QQ(value)
        if rational.denominator() != 1:
            return None
        result.append(int(rational))
    return tuple(result)


def _select_secret(
    coset: tuple[Any, Any],
    a: Sequence[Sequence[int]],
    b: Sequence[int],
    error: Sequence[int],
    modulus: int | None,
    s_bits: int | None,
    signed_secret: bool,
    method: LWEMethod,
    effort: int,
    lattice_reduction: LatticeReduction | None,
    report: Callable[..., None],
) -> tuple[int, ...] | None:
    """Select and validate a canonical or bounded secret from a solution coset."""
    particular, kernel_basis = coset
    secret_dimension = len(a[0])

    if modulus is not None and s_bits is None:
        if signed_secret:
            candidate = tuple(center(int(value), modulus) for value in particular)
        else:
            candidate = tuple(int(value) % modulus for value in particular)
        return (
            candidate
            if _valid_equations(a, b, candidate, error, modulus)
            else None
        )

    if s_bits is None:
        secret_center = QQ(0)
        secret_radius = QQ(1)
    else:
        _, _, secret_center, secret_radius = _secret_interval(
            s_bits,
            signed_secret,
        )

    candidates = [tuple(map(int, particular))]
    if kernel_basis.nrows() > 0:
        target = vector(
            QQ,
            [secret_center - value for value in particular],
        )
        points = _cvp_points(
            kernel_basis,
            target,
            method,
            secret_radius,
            effort,
            lattice_reduction,
            report,
        )
        for point in points:
            offset = _integral_coordinates(point)
            if offset is not None:
                candidates.append(
                    tuple(
                        int(particular[index]) + offset[index]
                        for index in range(secret_dimension)
                    )
                )

    valid = [
        candidate
        for candidate in candidates
        if _secret_within_bounds(candidate, s_bits, signed_secret)
        and _valid_equations(a, b, candidate, error, modulus)
    ]
    if not valid:
        return None
    return min(
        valid,
        key=lambda candidate: sum(
            (QQ(value) - secret_center) ** 2 for value in candidate
        ),
    )


def _candidate_from_error_point(
    point: Any,
    a: Sequence[Sequence[int]],
    b: Sequence[int],
    modulus: int | None,
    error_bounds: Sequence[int] | None,
    s_bits: int | None,
    signed_secret: bool,
    method: LWEMethod,
    effort: int,
    lattice_reduction: LatticeReduction | None,
    report: Callable[..., None],
) -> _LWECandidate | None:
    """Decode a standard error-lattice point into a complete LWE solution."""
    lattice_point = _integral_coordinates(point)
    if lattice_point is None:
        return None
    error = tuple(
        target - value
        for target, value in zip(b, lattice_point, strict=True)
    )
    if not _error_within_bounds(error, error_bounds):
        return None
    coset = _integer_solution_coset(
        a,
        [target - value for target, value in zip(b, error, strict=True)],
        modulus,
    )
    if coset is None:
        return None
    secret = _select_secret(
        coset,
        a,
        b,
        error,
        modulus,
        s_bits,
        signed_secret,
        method,
        effort,
        lattice_reduction,
        report,
    )
    if secret is None:
        return None
    return _LWECandidate(secret, error)


def _candidate_from_augmented_point(
    point: Any,
    secret_scale: Any,
    a: Sequence[Sequence[int]],
    b: Sequence[int],
    modulus: int | None,
    error_bounds: Sequence[int] | None,
    s_bits: int,
    signed_secret: bool,
) -> _LWECandidate | None:
    """Decode an augmented primal-lattice point into secret and error vectors."""
    sample_count = len(a)
    lattice_point = _integral_coordinates(point[:sample_count])
    if lattice_point is None:
        return None
    secret = _integral_coordinates(
        [coordinate / secret_scale for coordinate in point[sample_count:]]
    )
    if secret is None:
        return None
    error = tuple(
        target - value
        for target, value in zip(b, lattice_point, strict=True)
    )
    if not _error_within_bounds(error, error_bounds):
        return None
    if not _secret_within_bounds(secret, s_bits, signed_secret):
        return None
    if not _valid_equations(a, b, secret, error, modulus):
        return None
    return _LWECandidate(secret, error)


def _recover_candidates(
    a: Sequence[Sequence[int]],
    b: Sequence[int],
    modulus: int | None,
    formulation: LWEFormulation,
    method: LWEMethod,
    s_bits: int | None,
    signed_secret: bool,
    error_bounds: Sequence[int] | None,
    embedding_factor: Any | None,
    effort: int,
    lattice_reduction: LatticeReduction | None,
    report: Callable[..., None],
) -> list[_LWECandidate]:
    """Construct, decode, and validate an LWE BDD formulation."""
    selected_formulation = formulation
    if selected_formulation == "auto":
        selected_formulation = "augmented" if s_bits is not None else "error"
    report(f"Formulation: {selected_formulation}; method: {method}")

    default_factor = max(error_bounds) if error_bounds else 1
    factor = (
        max(1, default_factor)
        if embedding_factor is None
        else _positive_rational(embedding_factor, "embedding_factor")
    )
    candidates: list[_LWECandidate] = []
    if selected_formulation == "error":
        basis = _error_lattice(a, modulus)
        target = vector(QQ, b)
        report(f"Error lattice dimensions: {basis.dimensions()}")
        points = _cvp_points(
            basis,
            target,
            method,
            factor,
            effort,
            lattice_reduction,
            report,
        )
        for point in points:
            candidate = _candidate_from_error_point(
                point,
                a,
                b,
                modulus,
                error_bounds,
                s_bits,
                signed_secret,
                method,
                effort,
                lattice_reduction,
                report,
            )
            if candidate is not None:
                candidates.append(candidate)
    else:
        if s_bits is None:
            raise ValueError("augmented formulation requires s_bits")
        basis, target, secret_scale = _augmented_lattice(
            a,
            b,
            modulus,
            s_bits,
            signed_secret,
            error_bounds,
        )
        report(f"Augmented lattice dimensions: {basis.dimensions()}")
        points = _cvp_points(
            basis,
            target,
            method,
            factor,
            effort,
            lattice_reduction,
            report,
        )
        for point in points:
            candidate = _candidate_from_augmented_point(
                point,
                secret_scale,
                a,
                b,
                modulus,
                error_bounds,
                s_bits,
                signed_secret,
            )
            if candidate is not None:
                candidates.append(candidate)

    unique: dict[tuple[tuple[int, ...], tuple[int, ...]], _LWECandidate] = {}
    for candidate in candidates:
        unique[(candidate.secret, candidate.error)] = candidate
    return sorted(
        unique.values(),
        key=lambda candidate: (
            max(map(abs, candidate.error), default=0),
            sum(value * value for value in candidate.error),
            sum(value * value for value in candidate.secret),
        ),
    )


def _normalize_controls(
    formulation: LWEFormulation,
    method: LWEMethod,
    signed_secret: bool,
    verbose: bool,
) -> tuple[LWEFormulation, LWEMethod, Callable[..., None]]:
    """Validate shared formulation, decoder, and reporting controls."""
    if formulation not in ("auto", "error", "augmented"):
        raise ValueError("formulation must be 'auto', 'error', or 'augmented'")
    if method not in ("auto", "babai", "kannan"):
        raise ValueError("method must be 'auto', 'babai', or 'kannan'")
    if not isinstance(signed_secret, bool):
        raise TypeError("signed_secret must be a bool")
    if not isinstance(verbose, bool):
        raise TypeError("verbose must be a bool")
    report = (lambda *args: print("[lwe]", *args)) if verbose else lambda *_: None
    return formulation, method, report


def recover_low_bits(
    a: Iterable[Iterable[int]],
    b: Iterable[int],
    N: int | None = 0,
    lattice_reduction: LatticeReduction | None = None,
    verbose: bool = False,
    *,
    formulation: LWEFormulation = "error",
    method: LWEMethod = "auto",
    effort: int = 1,
    error_bound: int | Sequence[int] | None = None,
    s_bits: int | None = None,
    signed_secret: bool = True,
    embedding_factor: Any | None = None,
) -> list[int] | None:
    """Recover the small error vector in ``b = A*s + e (mod N)``.

    The standard ``error`` formulation decodes ``b`` in the q-ary lattice
    generated by ``A``. The ``augmented`` formulation additionally embeds a
    bounded secret and therefore requires ``s_bits``.
    """
    matrix, target, modulus = _normalize_problem(a, b, N)
    bounds = _normalize_error_bounds(error_bound, len(matrix))
    secret_bits = _normalize_secret_bits(s_bits)
    normalized_effort = _normalize_reduction(effort, lattice_reduction)
    formulation, method, report = _normalize_controls(
        formulation,
        method,
        signed_secret,
        verbose,
    )
    candidates = _recover_candidates(
        matrix,
        target,
        modulus,
        formulation,
        method,
        secret_bits,
        signed_secret,
        bounds,
        embedding_factor,
        normalized_effort,
        lattice_reduction,
        report,
    )
    return list(candidates[0].error) if candidates else None


def recover_secret(
    a: Iterable[Iterable[int]],
    b: Iterable[int],
    N: int | None = 0,
    e: Iterable[int] | None = None,
    s_bits: int | None = None,
    lattice_reduction: LatticeReduction | None = None,
    verbose: bool = False,
    *,
    formulation: LWEFormulation = "auto",
    method: LWEMethod = "auto",
    effort: int = 1,
    error_bound: int | Sequence[int] | None = None,
    signed_secret: bool = True,
    embedding_factor: Any | None = None,
) -> list[int] | None:
    """Recover ``s`` from ``b = A*s + e (mod N)`` and verify all equations.

    A supplied error vector bypasses bounded-distance decoding and solves the
    resulting integer or modular system through Smith form. Without ``e``, the
    selected error or augmented primal formulation first performs BDD.
    """
    matrix, target, modulus = _normalize_problem(a, b, N)
    bounds = _normalize_error_bounds(error_bound, len(matrix))
    secret_bits = _normalize_secret_bits(s_bits)
    normalized_effort = _normalize_reduction(effort, lattice_reduction)
    formulation, method, report = _normalize_controls(
        formulation,
        method,
        signed_secret,
        verbose,
    )

    if e is not None:
        error = _integer_tuple(e, "e")
        if len(error) != len(matrix):
            raise ValueError("e must contain one value per sample")
        if not _error_within_bounds(error, bounds):
            return None
        right_hand_side = [
            target_value - error_value
            for target_value, error_value in zip(target, error, strict=True)
        ]
        coset = _integer_solution_coset(matrix, right_hand_side, modulus)
        if coset is None:
            return None
        secret = _select_secret(
            coset,
            matrix,
            target,
            error,
            modulus,
            secret_bits,
            signed_secret,
            method,
            normalized_effort,
            lattice_reduction,
            report,
        )
        return list(secret) if secret is not None else None

    candidates = _recover_candidates(
        matrix,
        target,
        modulus,
        formulation,
        method,
        secret_bits,
        signed_secret,
        bounds,
        embedding_factor,
        normalized_effort,
        lattice_reduction,
        report,
    )
    return list(candidates[0].secret) if candidates else None
