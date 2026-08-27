"""Validated solvers for hidden-number and extended hidden-number problems."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from itertools import product
from math import lcm
from operator import index as integer_index
from typing import Any, Literal

from sage.all import Matrix, QQ, ZZ, vector


CVPMethod = Literal["auto", "exact", "babai", "kannan"]
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


def _positive_rational(value: Any, name: str) -> Any:
    """Return a strictly positive Sage rational."""
    try:
        rational = QQ(value)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{name} must be rational") from error
    if rational <= 0:
        raise ValueError(f"{name} must be positive")
    return rational


def _normalize_reduction(
    effort: int,
    lattice_reduction: LatticeReduction | None,
) -> int:
    """Validate LLL/BKZ and custom-reducer controls."""
    normalized_effort = _exact_integer(effort, "effort")
    if normalized_effort < 1:
        raise ValueError("effort must be at least 1")
    if lattice_reduction is not None and not callable(lattice_reduction):
        raise TypeError("lattice_reduction must be callable or None")
    if lattice_reduction is not None and normalized_effort != 1:
        raise ValueError("effort must be 1 with a custom lattice reduction")
    return normalized_effort


def _reduce_basis(
    basis: Any,
    effort: int,
    lattice_reduction: LatticeReduction | None,
) -> Any:
    """Reduce and validate a full-row-rank integer lattice basis."""
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
    """Clear all denominators from a basis and target with one global scale."""
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
    """Return Babai's nearest-plane lattice point for an already reduced basis."""
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


def _babai_lattice_point(
    basis: Any,
    target: Any,
    effort: int,
    lattice_reduction: LatticeReduction | None,
) -> Any:
    """Reduce an exact lattice and apply Babai nearest plane."""
    integer_basis, integer_target, denominator = _integralize_lattice(
        basis,
        target,
    )
    reduced = _reduce_basis(integer_basis, effort, lattice_reduction)
    point = _babai_reduced(reduced, integer_target)
    return vector(QQ, [QQ(entry) / denominator for entry in point])


def _kannan_lattice_points(
    basis: Any,
    target: Any,
    embedding_factor: Any,
    effort: int,
    lattice_reduction: LatticeReduction | None,
    report: Callable[..., None],
) -> list[Any]:
    """Return CVP candidates exposed by Kannan's embedding."""
    factor = _positive_rational(embedding_factor, "embedding_factor")
    rational_basis = Matrix(QQ, basis)
    rational_target = vector(QQ, target)
    denominators = [
        int(QQ(entry).denominator())
        for entry in (
            rational_basis.list()
            + list(rational_target)
            + [factor]
        )
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

    candidates: list[Any] = []
    seen: set[tuple[Any, ...]] = set()
    for row in reduced.rows():
        marker = int(row[-1])
        if abs(marker) != integer_factor:
            continue
        sign = 1 if marker > 0 else -1
        point_integer = vector(ZZ, [
            sign * int(entry) + integer_target[index]
            for index, entry in enumerate(row[:-1])
        ])
        point = vector(
            QQ,
            [QQ(entry) / denominator for entry in point_integer],
        )
        key = tuple(point)
        if key not in seen:
            seen.add(key)
            candidates.append(point)
    return candidates


def babai_cvp(
    B: Any,
    t: Any,
    perform_reduction: bool = True,
    *,
    effort: int = 1,
    lattice_reduction: LatticeReduction | None = None,
) -> Any:
    """Return Babai's approximate closest lattice vector to ``t``.

    The basis vectors are interpreted as rows. Rational inputs are scaled to a
    common integer lattice before reduction and mapped back exactly afterwards.
    """
    if not isinstance(perform_reduction, bool):
        raise TypeError("perform_reduction must be a bool")
    normalized_effort = _normalize_reduction(effort, lattice_reduction)
    try:
        basis = Matrix(QQ, B)
        target = vector(QQ, t)
    except (TypeError, ValueError) as error:
        raise TypeError("B and t must define a rational lattice problem") from error
    if basis.ncols() != len(target):
        raise ValueError("target dimension must equal the lattice ambient dimension")
    if basis.rank() != basis.nrows():
        raise ValueError("Babai requires a full-row-rank basis")

    if perform_reduction:
        return _babai_lattice_point(
            basis,
            target,
            normalized_effort,
            lattice_reduction,
        )
    if lattice_reduction is not None or normalized_effort != 1:
        raise ValueError("reduction controls require perform_reduction=True")
    return _babai_reduced(basis, target)


def _normalize_hnp(
    p: int,
    T: Iterable[int],
    A: Iterable[int],
    B: int,
) -> tuple[int, tuple[int, ...], tuple[int, ...], int]:
    """Validate a standard hidden-number instance."""
    modulus = _exact_integer(p, "p")
    if modulus <= 2:
        raise ValueError("p must be greater than two")
    multipliers = _integer_tuple(T, "T")
    offsets = _integer_tuple(A, "A")
    if not multipliers:
        raise ValueError("at least one HNP sample is required")
    if len(multipliers) != len(offsets):
        raise ValueError("T and A must have the same length")
    bound = _exact_integer(B, "B")
    if bound <= 0:
        raise ValueError("B must be positive")
    if bound > modulus:
        raise ValueError("B must not exceed p")
    return (
        modulus,
        tuple(value % modulus for value in multipliers),
        tuple(value % modulus for value in offsets),
        bound,
    )


def _hnp_residuals(
    alpha: int,
    modulus: int,
    multipliers: Sequence[int],
    offsets: Sequence[int],
    bound: int,
    signed: bool,
) -> tuple[int, ...] | None:
    """Return validated HNP residuals or reject the secret candidate."""
    alpha %= modulus
    if alpha == 0:
        return None
    residuals: list[int] = []
    for multiplier, offset in zip(multipliers, offsets, strict=True):
        residual = (multiplier * alpha - offset) % modulus
        if signed and 2 * residual > modulus:
            residual -= modulus
        if signed:
            if not -bound < residual < bound:
                return None
        elif not 0 <= residual < bound:
            return None
        residuals.append(residual)
    return tuple(residuals)


def _hnp_lattice(
    modulus: int,
    multipliers: Sequence[int],
    offsets: Sequence[int],
    bound: int,
    signed: bool,
) -> tuple[Any, Any, Any]:
    """Construct a centered HNP CVP lattice, target, and distance scale."""
    sample_count = len(multipliers)
    if signed:
        residual_center = QQ(0)
        residual_radius = QQ(max(1, bound - 1))
    else:
        residual_center = QQ(bound - 1) / 2
        residual_radius = max(QQ(1), residual_center)

    alpha_center = QQ(modulus) / 2
    alpha_radius = QQ(modulus) / 2
    alpha_scale = residual_radius / alpha_radius

    basis = Matrix(QQ, sample_count + 1, sample_count + 1)
    for index in range(sample_count):
        basis[index, index] = modulus
    for index, multiplier in enumerate(multipliers):
        basis[sample_count, index] = multiplier
    basis[sample_count, sample_count] = alpha_scale

    target = vector(
        QQ,
        [offset + residual_center for offset in offsets]
        + [alpha_center * alpha_scale],
    )
    return basis, target, residual_radius


def _hnp_alpha_from_point(point: Any, alpha_scale: Any, modulus: int) -> int | None:
    """Extract the hidden multiplier coefficient from a lattice point."""
    alpha_value = QQ(point[-1]) / alpha_scale
    if alpha_value.denominator() != 1:
        return None
    alpha = int(alpha_value) % modulus
    return alpha if alpha else None


def hnp(
    p: int,
    T: Iterable[int],
    A: Iterable[int],
    B: int,
    lattice_reduction: LatticeReduction | None = None,
    verbose: bool = False,
    *,
    method: CVPMethod = "auto",
    effort: int = 1,
    signed: bool = False,
    exact_threshold: int = 1 << 16,
) -> int | None:
    r"""Recover ``alpha`` from ``beta_i - T_i*alpha + A_i = 0 (mod p)``.

    Unsigned residuals satisfy ``0 <= beta_i < B``. Set ``signed=True`` for
    ``-B < beta_i < B``. Small instances can be solved exhaustively; larger
    instances use Babai CVP or Kannan's CVP-to-SVP embedding.
    """
    modulus, multipliers, offsets, bound = _normalize_hnp(p, T, A, B)
    if not isinstance(verbose, bool):
        raise TypeError("verbose must be a bool")
    if not isinstance(signed, bool):
        raise TypeError("signed must be a bool")
    report = (lambda *args: print("[hnp]", *args)) if verbose else lambda *_: None
    if method not in ("auto", "exact", "babai", "kannan"):
        raise ValueError("method must be 'auto', 'exact', 'babai', or 'kannan'")
    normalized_effort = _normalize_reduction(effort, lattice_reduction)
    normalized_threshold = _exact_integer(exact_threshold, "exact_threshold")
    if normalized_threshold < 0:
        raise ValueError("exact_threshold must be non-negative")

    selected_method = method
    if selected_method == "auto":
        selected_method = (
            "exact"
            if lattice_reduction is None and modulus <= normalized_threshold
            else "kannan"
        )
    report(f"Method: {selected_method}")

    if selected_method == "exact":
        for alpha in range(1, modulus):
            if _hnp_residuals(
                alpha,
                modulus,
                multipliers,
                offsets,
                bound,
                signed,
            ) is not None:
                return alpha
        return None

    basis, target, embedding_factor = _hnp_lattice(
        modulus,
        multipliers,
        offsets,
        bound,
        signed,
    )
    alpha_scale = basis[basis.nrows() - 1, basis.ncols() - 1]
    report(f"CVP lattice dimensions: {basis.dimensions()}")
    if selected_method == "babai":
        points = [
            _babai_lattice_point(
                basis,
                target,
                normalized_effort,
                lattice_reduction,
            )
        ]
    else:
        points = _kannan_lattice_points(
            basis,
            target,
            embedding_factor,
            normalized_effort,
            lattice_reduction,
            report,
        )

    seen: set[int] = set()
    for point in points:
        alpha = _hnp_alpha_from_point(point, alpha_scale, modulus)
        if alpha is None or alpha in seen:
            continue
        seen.add(alpha)
        if _hnp_residuals(
            alpha,
            modulus,
            multipliers,
            offsets,
            bound,
            signed,
        ) is not None:
            return alpha
    return None


def _normalize_nested_rows(
    rows: Iterable[Iterable[int]],
    name: str,
) -> tuple[tuple[int, ...], ...]:
    """Normalize a nested integer sequence."""
    try:
        raw_rows = tuple(rows)
    except TypeError as error:
        raise TypeError(f"{name} must be an iterable of integer iterables") from error
    return tuple(
        _integer_tuple(row, f"{name}[{index}]")
        for index, row in enumerate(raw_rows)
    )


def _normalize_ehnp(
    xbar: int,
    p: int,
    Pi: Iterable[int],
    Nu: Iterable[int],
    Alpha: Iterable[int],
    Rho: Iterable[Iterable[int]],
    Mu: Iterable[Iterable[int]],
    Beta: Iterable[int],
) -> tuple[
    int,
    int,
    tuple[int, ...],
    tuple[int, ...],
    tuple[int, ...],
    tuple[tuple[int, ...], ...],
    tuple[tuple[int, ...], ...],
    tuple[int, ...],
]:
    """Validate and normalize an extended hidden-number instance."""
    known_part = _exact_integer(xbar, "xbar")
    modulus = _exact_integer(p, "p")
    if modulus <= 2:
        raise ValueError("p must be greater than two")
    positions = _integer_tuple(Pi, "Pi")
    widths = _integer_tuple(Nu, "Nu")
    alphas = _integer_tuple(Alpha, "Alpha")
    rhos = _normalize_nested_rows(Rho, "Rho")
    multiplier_widths = _normalize_nested_rows(Mu, "Mu")
    betas = _integer_tuple(Beta, "Beta")

    if len(positions) != len(widths):
        raise ValueError("Pi and Nu must have the same length")
    if not positions:
        raise ValueError("at least one unknown x fragment is required")
    if any(position < 0 for position in positions):
        raise ValueError("Pi entries must be non-negative")
    if any(width <= 0 for width in widths):
        raise ValueError("Nu entries must be positive")
    fragment_intervals = sorted(
        (position, position + width)
        for position, width in zip(positions, widths, strict=True)
    )
    if any(
        left_end > right_start
        for (_, left_end), (right_start, _) in zip(
            fragment_intervals,
            fragment_intervals[1:],
        )
    ):
        raise ValueError("the bit ranges described by Pi and Nu must not overlap")
    if not alphas:
        raise ValueError("at least one EHNP equation is required")
    if not (len(alphas) == len(rhos) == len(multiplier_widths) == len(betas)):
        raise ValueError("Alpha, Rho, Mu, and Beta must have the same length")
    for index, (rho_row, width_row) in enumerate(
        zip(rhos, multiplier_widths, strict=True)
    ):
        if len(rho_row) != len(width_row):
            raise ValueError(f"Rho[{index}] and Mu[{index}] must have equal length")
        if any(width <= 0 for width in width_row):
            raise ValueError(f"Mu[{index}] entries must be positive")

    return (
        known_part,
        modulus,
        positions,
        widths,
        tuple(value % modulus for value in alphas),
        tuple(tuple(value % modulus for value in row) for row in rhos),
        multiplier_widths,
        tuple(value % modulus for value in betas),
    )


def _kappa_upper_bound(dimension: int, bounded_count: int) -> int:
    """Bound the EHNP kappa expression above with an exact integer."""
    power_bound = 1 << ((dimension + 3) // 4)
    square_root_bound = 1
    while square_root_bound * square_root_bound < bounded_count:
        square_root_bound += 1
    return max(1, (power_bound * square_root_bound + 2) // 2)


def _default_delta(dimension: int, bounded_count: int) -> Any:
    """Choose a conservative exact rational satisfying the EHNP condition."""
    return QQ(1) / (
        10**8 * _kappa_upper_bound(dimension, bounded_count)
    )


def _ehnp_lattice(
    modulus: int,
    positions: Sequence[int],
    widths: Sequence[int],
    alphas: Sequence[int],
    rhos: Sequence[Sequence[int]],
    multiplier_widths: Sequence[Sequence[int]],
    betas: Sequence[int],
    known_part: int,
    delta: Any,
) -> tuple[Any, Any, tuple[Any, ...], tuple[tuple[Any, ...], ...]]:
    """Construct the exact EHNP CVP lattice and coordinate scales."""
    equation_count = len(alphas)
    fragment_count = len(positions)
    multiplier_count = sum(len(row) for row in rhos)
    dimension = equation_count + fragment_count + multiplier_count
    basis = Matrix(QQ, dimension, dimension)

    for index in range(equation_count):
        basis[index, index] = modulus

    fragment_scales: list[Any] = []
    for fragment_index, (position, width) in enumerate(
        zip(positions, widths, strict=True)
    ):
        row_index = equation_count + fragment_index
        for equation_index, alpha in enumerate(alphas):
            basis[row_index, equation_index] = alpha * (1 << position)
        scale = delta / (1 << width)
        fragment_scales.append(scale)
        basis[row_index, equation_count + fragment_index] = scale

    multiplier_scales: list[tuple[Any, ...]] = []
    flat_multiplier_index = 0
    for equation_index, (rho_row, width_row) in enumerate(
        zip(rhos, multiplier_widths, strict=True)
    ):
        row_scales: list[Any] = []
        for rho, width in zip(rho_row, width_row, strict=True):
            row_index = equation_count + fragment_count + flat_multiplier_index
            basis[row_index, equation_index] = rho
            scale = delta / (1 << width)
            row_scales.append(scale)
            basis[
                row_index,
                equation_count + fragment_count + flat_multiplier_index,
            ] = scale
            flat_multiplier_index += 1
        multiplier_scales.append(tuple(row_scales))

    target = [
        beta - alpha * known_part
        for alpha, beta in zip(alphas, betas, strict=True)
    ]
    target.extend([delta / 2] * (fragment_count + multiplier_count))
    return (
        basis,
        vector(QQ, target),
        tuple(fragment_scales),
        tuple(multiplier_scales),
    )


def _integral_coordinate(value: Any, scale: Any) -> int | None:
    """Decode an integral bounded variable from a scaled lattice coordinate."""
    coordinate = QQ(value) / scale
    if coordinate.denominator() != 1:
        return None
    return int(coordinate)


def _validate_ehnp_point(
    point: Any,
    known_part: int,
    modulus: int,
    positions: Sequence[int],
    widths: Sequence[int],
    alphas: Sequence[int],
    rhos: Sequence[Sequence[int]],
    multiplier_widths: Sequence[Sequence[int]],
    betas: Sequence[int],
    fragment_scales: Sequence[Any],
    multiplier_scales: Sequence[Sequence[Any]],
) -> int | None:
    """Decode and fully validate an EHNP lattice-point candidate."""
    equation_count = len(alphas)
    fragments: list[int] = []
    for index, (width, scale) in enumerate(
        zip(widths, fragment_scales, strict=True)
    ):
        value = _integral_coordinate(point[equation_count + index], scale)
        if value is None or not 0 <= value < 1 << width:
            return None
        fragments.append(value)

    multipliers: list[tuple[int, ...]] = []
    coordinate_index = equation_count + len(fragments)
    for width_row, scale_row in zip(
        multiplier_widths,
        multiplier_scales,
        strict=True,
    ):
        decoded_row: list[int] = []
        for width, scale in zip(width_row, scale_row, strict=True):
            value = _integral_coordinate(point[coordinate_index], scale)
            coordinate_index += 1
            if value is None or not 0 <= value < 1 << width:
                return None
            decoded_row.append(value)
        multipliers.append(tuple(decoded_row))

    recovered = known_part + sum(
        fragment * (1 << position)
        for fragment, position in zip(fragments, positions, strict=True)
    )
    recovered_modulus = recovered % modulus
    if recovered_modulus == 0:
        return None
    for alpha, rho_row, multiplier_row, beta in zip(
        alphas,
        rhos,
        multipliers,
        betas,
        strict=True,
    ):
        value = alpha * recovered
        value += sum(
            rho * multiplier
            for rho, multiplier in zip(rho_row, multiplier_row, strict=True)
        )
        if (value - beta) % modulus:
            return None
    return recovered_modulus


def _ehnp_exact_work(
    widths: Sequence[int],
    multiplier_widths: Sequence[Sequence[int]],
) -> int:
    """Bound the exhaustive work for an EHNP instance."""
    fragment_space = 1 << sum(widths)
    multiplier_work = sum(
        1 << sum(width_row) for width_row in multiplier_widths
    )
    return fragment_space * max(1, multiplier_work)


def _exact_ehnp(
    known_part: int,
    modulus: int,
    positions: Sequence[int],
    widths: Sequence[int],
    alphas: Sequence[int],
    rhos: Sequence[Sequence[int]],
    multiplier_widths: Sequence[Sequence[int]],
    betas: Sequence[int],
) -> int | None:
    """Exhaustively solve a deliberately small EHNP instance."""
    fragment_ranges = [range(1 << width) for width in widths]
    for fragments in product(*fragment_ranges):
        recovered = known_part + sum(
            fragment * (1 << position)
            for fragment, position in zip(fragments, positions, strict=True)
        )
        if recovered % modulus == 0:
            continue

        equations_hold = True
        for alpha, rho_row, width_row, beta in zip(
            alphas,
            rhos,
            multiplier_widths,
            betas,
            strict=True,
        ):
            multiplier_ranges = [range(1 << width) for width in width_row]
            if not any(
                (
                    alpha * recovered
                    + sum(
                        rho * multiplier
                        for rho, multiplier in zip(
                            rho_row,
                            multiplier_values,
                            strict=True,
                        )
                    )
                    - beta
                )
                % modulus
                == 0
                for multiplier_values in product(*multiplier_ranges)
            ):
                equations_hold = False
                break
        if equations_hold:
            return recovered % modulus
    return None


def ehnp(
    xbar: int,
    p: int,
    Pi: Iterable[int],
    Nu: Iterable[int],
    Alpha: Iterable[int],
    Rho: Iterable[Iterable[int]],
    Mu: Iterable[Iterable[int]],
    Beta: Iterable[int],
    delta: Any | None = None,
    lattice_reduction: LatticeReduction | None = None,
    verbose: bool = False,
    *,
    method: CVPMethod = "auto",
    effort: int = 1,
    exact_threshold: int = 1 << 16,
) -> int | None:
    """Recover the hidden value in an extended hidden-number instance."""
    (
        known_part,
        modulus,
        positions,
        widths,
        alphas,
        rhos,
        multiplier_widths,
        betas,
    ) = _normalize_ehnp(xbar, p, Pi, Nu, Alpha, Rho, Mu, Beta)
    if not isinstance(verbose, bool):
        raise TypeError("verbose must be a bool")
    report = (lambda *args: print("[ehnp]", *args)) if verbose else lambda *_: None
    if method not in ("auto", "exact", "babai", "kannan"):
        raise ValueError("method must be 'auto', 'exact', 'babai', or 'kannan'")
    normalized_effort = _normalize_reduction(effort, lattice_reduction)
    normalized_threshold = _exact_integer(exact_threshold, "exact_threshold")
    if normalized_threshold < 0:
        raise ValueError("exact_threshold must be non-negative")

    multiplier_count = sum(len(row) for row in rhos)
    dimension = len(alphas) + len(positions) + multiplier_count
    bounded_count = len(positions) + multiplier_count
    normalized_delta = (
        _default_delta(dimension, bounded_count)
        if delta is None
        else _positive_rational(delta, "delta")
    )
    kappa_bound = _kappa_upper_bound(dimension, bounded_count)
    if normalized_delta * kappa_bound >= 1:
        raise ValueError(
            "delta is too large for the conservative EHNP lattice condition"
        )

    exact_work = _ehnp_exact_work(widths, multiplier_widths)
    selected_method = method
    if selected_method == "auto" and lattice_reduction is None:
        if exact_work <= normalized_threshold:
            selected_method = "exact"
    report(f"Method: {selected_method}")

    if selected_method == "exact":
        if exact_work > normalized_threshold:
            raise ValueError(
                "exact EHNP search exceeds exact_threshold; increase the "
                "threshold explicitly or choose a lattice method"
            )
        return _exact_ehnp(
            known_part,
            modulus,
            positions,
            widths,
            alphas,
            rhos,
            multiplier_widths,
            betas,
        )

    basis, target, fragment_scales, multiplier_scales = _ehnp_lattice(
        modulus,
        positions,
        widths,
        alphas,
        rhos,
        multiplier_widths,
        betas,
        known_part,
        normalized_delta,
    )
    report(f"CVP lattice dimensions: {basis.dimensions()}")

    if selected_method in ("auto", "babai"):
        point = _babai_lattice_point(
            basis,
            target,
            normalized_effort,
            lattice_reduction,
        )
        recovered = _validate_ehnp_point(
            point,
            known_part,
            modulus,
            positions,
            widths,
            alphas,
            rhos,
            multiplier_widths,
            betas,
            fragment_scales,
            multiplier_scales,
        )
        if recovered is not None or selected_method == "babai":
            return recovered

    points = _kannan_lattice_points(
        basis,
        target,
        normalized_delta,
        normalized_effort,
        lattice_reduction,
        report,
    )
    for point in points:
        recovered = _validate_ehnp_point(
            point,
            known_part,
            modulus,
            positions,
            widths,
            alphas,
            rhos,
            multiplier_widths,
            betas,
            fragment_scales,
            multiplier_scales,
        )
        if recovered is not None:
            return recovered
    return None
