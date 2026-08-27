"""Solve binary subset-sum systems with exact and lattice strategies."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from math import isqrt, log2
from operator import index as integer_index
from typing import Any, Literal

from sage.all import Matrix, ZZ, vector


KnapsackMethod = Literal["auto", "exact", "direct", "kernel"]
LatticeReduction = Callable[[Any], Any]


def _exact_integer(value: Any, name: str) -> int:
    """Return an exact integer while rejecting booleans and lossy coercions."""
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


def _normalize_problem(
    weights: Sequence[int] | Sequence[Sequence[int]],
    targets: int | Sequence[int],
    modulus: int | None,
) -> tuple[tuple[tuple[int, ...], ...], tuple[int, ...], int | None]:
    """Normalize single and multiple subset-sum representations."""
    try:
        raw_weights = tuple(weights)
    except TypeError as error:
        raise TypeError("weights must be a non-empty sequence") from error
    if not raw_weights:
        raise ValueError("weights must not be empty")

    try:
        _exact_integer(raw_weights[0], "weights[0]")
    except TypeError:
        rows = tuple(
            _integer_tuple(row, f"weights[{index}]")
            for index, row in enumerate(raw_weights)
        )
    else:
        rows = (_integer_tuple(raw_weights, "weights"),)

    item_count = len(rows[0])
    if item_count == 0:
        raise ValueError("weight rows must not be empty")
    if any(len(row) != item_count for row in rows):
        raise ValueError("all weight rows must have the same length")

    try:
        scalar_target = _exact_integer(targets, "targets")
    except TypeError:
        normalized_targets = _integer_tuple(targets, "targets")
    else:
        normalized_targets = (scalar_target,)
    if len(normalized_targets) != len(rows):
        raise ValueError("there must be one target per weight row")

    if modulus is None:
        normalized_modulus = None
    else:
        normalized_modulus = _exact_integer(modulus, "modulus")
        if normalized_modulus <= 1:
            raise ValueError("modulus must be greater than one")

    return rows, normalized_targets, normalized_modulus


def _normalize_reduction(
    effort: int,
    lattice_reduction: LatticeReduction | None,
) -> int:
    """Validate shared lattice-reduction controls."""
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
    """Reduce a full-row-rank integer lattice basis."""
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
    return reduced


def _is_solution(
    bits: Sequence[int],
    weights: Sequence[Sequence[int]],
    targets: Sequence[int],
    modulus: int | None,
) -> bool:
    """Check the binary domain and every target equation exactly."""
    if len(bits) != len(weights[0]) or any(bit not in (0, 1) for bit in bits):
        return False
    for row, target in zip(weights, targets, strict=True):
        total = sum(bit * weight for bit, weight in zip(bits, row, strict=True))
        if modulus is None:
            if total != target:
                return False
        elif (total - target) % modulus:
            return False
    return True


def _subset_signatures(
    weights: Sequence[Sequence[int]],
    start: int,
    stop: int,
    modulus: int | None,
) -> list[tuple[tuple[int, ...], int]]:
    """Enumerate vector sums for one half of a meet-in-the-middle search."""
    signatures: list[tuple[tuple[int, ...], int]] = [
        ((0,) * len(weights), 0)
    ]
    for item_index in range(start, stop):
        bit = 1 << (item_index - start)
        additions = tuple(row[item_index] for row in weights)
        new_entries = []
        for signature, mask in signatures:
            new_signature = tuple(
                value + addition
                for value, addition in zip(signature, additions, strict=True)
            )
            if modulus is not None:
                new_signature = tuple(value % modulus for value in new_signature)
            new_entries.append((new_signature, mask | bit))
        signatures.extend(new_entries)
    return signatures


def _decode_masks(left_mask: int, right_mask: int, split: int, n: int) -> list[int]:
    """Decode two subset masks into one bit vector."""
    return [
        (left_mask >> index) & 1
        if index < split
        else (right_mask >> (index - split)) & 1
        for index in range(n)
    ]


def _exact_subset_sum(
    weights: Sequence[Sequence[int]],
    targets: Sequence[int],
    modulus: int | None,
) -> list[int] | None:
    """Solve exactly with a meet-in-the-middle search."""
    item_count = len(weights[0])
    split = item_count // 2
    left_entries = _subset_signatures(weights, 0, split, modulus)
    right_entries = _subset_signatures(weights, split, item_count, modulus)

    left_by_signature: dict[tuple[int, ...], int] = {}
    for signature, mask in left_entries:
        left_by_signature.setdefault(signature, mask)

    normalized_targets = (
        tuple(target % modulus for target in targets)
        if modulus is not None
        else tuple(targets)
    )
    for right_signature, right_mask in right_entries:
        required = tuple(
            target - value
            for target, value in zip(
                normalized_targets,
                right_signature,
                strict=True,
            )
        )
        if modulus is not None:
            required = tuple(value % modulus for value in required)
        left_mask = left_by_signature.get(required)
        if left_mask is None:
            continue
        bits = _decode_masks(left_mask, right_mask, split, item_count)
        if _is_solution(bits, weights, targets, modulus):
            return bits
    return None


def _default_embedding_scale(item_count: int) -> int:
    """Return ``ceil(sqrt((n + 1) / 4))`` using integer arithmetic."""
    root = isqrt(item_count + 1)
    if root * root < item_count + 1:
        root += 1
    return max(1, (root + 1) // 2)


def _direct_subset_sum(
    weights: Sequence[Sequence[int]],
    targets: Sequence[int],
    modulus: int | None,
    embedding_scale: int,
    effort: int,
    lattice_reduction: LatticeReduction | None,
    report: Callable[..., None],
) -> list[int] | None:
    """Use the specialized Pan--Zhang binary SVP embedding."""
    item_count = len(weights[0])
    equation_count = len(weights)
    column_count = item_count + 1 + equation_count
    rows: list[list[int]] = []

    for item_index in range(item_count):
        row = [0] * column_count
        row[item_index] = 2
        for equation_index in range(equation_count):
            row[item_count + 1 + equation_index] = (
                embedding_scale * weights[equation_index][item_index]
            )
        rows.append(row)

    if modulus is not None:
        for equation_index in range(equation_count):
            row = [0] * column_count
            row[item_count + 1 + equation_index] = embedding_scale * modulus
            rows.append(row)

    target_row = [1] * item_count + [1]
    target_row.extend(embedding_scale * target for target in targets)
    rows.append(target_row)

    basis = Matrix(ZZ, rows)
    report(f"Direct knapsack lattice dimensions: {basis.dimensions()}")
    reduced = _reduce_basis(basis, effort, lattice_reduction)

    for row in reduced.rows():
        marker = int(row[item_count])
        if marker not in (-1, 1):
            continue
        numerators = [1 - marker * int(entry) for entry in row[:item_count]]
        if any(numerator % 2 for numerator in numerators):
            continue
        bits = [numerator // 2 for numerator in numerators]
        if _is_solution(bits, weights, targets, modulus):
            return bits
    return None


def _integer_solution_coset(
    coefficients: Sequence[Sequence[int]],
    right_hand_side: Sequence[int],
    moduli: Sequence[int | None],
    variable_count: int,
) -> tuple[Any, Any] | None:
    """Compute a particular integer solution and projected kernel lattice."""
    modular_count = sum(modulus is not None for modulus in moduli)
    unknown_count = variable_count + modular_count
    equations = Matrix(ZZ, len(coefficients), unknown_count)
    rhs = vector(ZZ, right_hand_side)

    modulus_column = variable_count
    for row_index, (row, modulus) in enumerate(
        zip(coefficients, moduli, strict=True)
    ):
        for column_index, coefficient in enumerate(row):
            equations[row_index, column_index] = coefficient
        if modulus is not None:
            equations[row_index, modulus_column] = modulus
            modulus_column += 1

    diagonal, left_transform, right_transform = equations.smith_form()
    transformed_rhs = left_transform * rhs
    rank = equations.rank()
    transformed_solution = vector(ZZ, unknown_count)
    for index in range(rank):
        diagonal_value = diagonal[index, index]
        if transformed_rhs[index] % diagonal_value:
            return None
        transformed_solution[index] = transformed_rhs[index] // diagonal_value
    if any(transformed_rhs[index] for index in range(rank, len(coefficients))):
        return None

    full_particular = right_transform * transformed_solution
    particular = vector(ZZ, full_particular[:variable_count])
    generators = [
        list(right_transform.column(index)[:variable_count])
        for index in range(rank, unknown_count)
    ]
    generators = [generator for generator in generators if any(generator)]
    if generators:
        kernel_basis = Matrix(ZZ, generators).row_module().basis_matrix()
    else:
        kernel_basis = Matrix(ZZ, 0, variable_count)
    return particular, kernel_basis


def _signed_bit_coset(
    weights: Sequence[Sequence[int]],
    targets: Sequence[int],
    modulus: int | None,
) -> tuple[Any, Any] | None:
    """Build the affine solution space for signed, odd bit coordinates."""
    item_count = len(weights[0])
    coefficients: list[tuple[int, ...]] = []
    right_hand_side: list[int] = []
    moduli: list[int | None] = []

    for row, target in zip(weights, targets, strict=True):
        coefficients.append(tuple(row))
        right_hand_side.append(2 * target - sum(row))
        moduli.append(None if modulus is None else 2 * modulus)

    for item_index in range(item_count):
        parity_row = [0] * item_count
        parity_row[item_index] = 1
        coefficients.append(tuple(parity_row))
        right_hand_side.append(1)
        moduli.append(2)

    return _integer_solution_coset(
        coefficients,
        right_hand_side,
        moduli,
        item_count,
    )


def _kernel_subset_sum(
    weights: Sequence[Sequence[int]],
    targets: Sequence[int],
    modulus: int | None,
    effort: int,
    lattice_reduction: LatticeReduction | None,
    report: Callable[..., None],
) -> list[int] | None:
    """Solve the signed-bit affine coset using Kannan embedding."""
    coset = _signed_bit_coset(weights, targets, modulus)
    if coset is None:
        return None
    particular, kernel_basis = coset

    candidates: list[tuple[int, ...]] = [tuple(map(int, particular))]
    if kernel_basis.nrows() > 0:
        embedded_rows = [list(row) + [0] for row in kernel_basis.rows()]
        embedded_rows.append(list(particular) + [1])
        embedded_basis = Matrix(ZZ, embedded_rows)
        report(f"Kernel/Kannan lattice dimensions: {embedded_basis.dimensions()}")
        reduced = _reduce_basis(embedded_basis, effort, lattice_reduction)
        for row in reduced.rows():
            marker = int(row[-1])
            if marker not in (-1, 1):
                continue
            candidates.append(tuple(marker * int(entry) for entry in row[:-1]))

    seen: set[tuple[int, ...]] = set()
    for signed_bits in candidates:
        if signed_bits in seen:
            continue
        seen.add(signed_bits)
        if any(value not in (-1, 1) for value in signed_bits):
            continue
        bits = [(value + 1) // 2 for value in signed_bits]
        if _is_solution(bits, weights, targets, modulus):
            return bits
    return None


def subset_sum(
    weights: Sequence[int] | Sequence[Sequence[int]],
    targets: int | Sequence[int],
    modulus: int | None = None,
    N: int | None = None,
    lattice_reduction: LatticeReduction | None = None,
    verbose: bool = False,
    *,
    method: KnapsackMethod = "auto",
    effort: int = 1,
    exact_threshold: int = 32,
) -> list[int] | None:
    """Return binary coefficients satisfying every supplied subset sum.

    ``method="exact"`` uses meet-in-the-middle search. ``"direct"`` uses the
    specialized binary SVP embedding, while ``"kernel"`` computes a signed-bit
    affine kernel and applies Kannan embedding. ``"auto"`` uses exact search up
    to ``exact_threshold`` items and the direct lattice otherwise.
    """
    normalized_weights, normalized_targets, normalized_modulus = (
        _normalize_problem(weights, targets, modulus)
    )
    if not isinstance(verbose, bool):
        raise TypeError("verbose must be a bool")
    report = (
        (lambda *args: print("[subset_sum]", *args))
        if verbose
        else lambda *_: None
    )

    if method not in ("auto", "exact", "direct", "kernel"):
        raise ValueError("method must be 'auto', 'exact', 'direct', or 'kernel'")
    normalized_effort = _normalize_reduction(effort, lattice_reduction)
    normalized_threshold = _exact_integer(exact_threshold, "exact_threshold")
    if normalized_threshold < 0:
        raise ValueError("exact_threshold must be non-negative")

    if N is None:
        embedding_scale = _default_embedding_scale(len(normalized_weights[0]))
    else:
        embedding_scale = _exact_integer(N, "N")
        if embedding_scale <= 0:
            raise ValueError("N must be positive")

    if normalized_modulus is not None:
        density = len(normalized_weights[0]) / (
            len(normalized_weights) * log2(normalized_modulus)
        )
        report(f"Density: {density:.4f}")
    else:
        maximum_weight = max(abs(value) for row in normalized_weights for value in row)
        if maximum_weight > 1:
            density = len(normalized_weights[0]) / (
                len(normalized_weights) * log2(maximum_weight)
            )
            report(f"Density: {density:.4f}")

    selected_method = method
    if selected_method == "auto":
        selected_method = (
            "exact"
            if lattice_reduction is None
            and len(normalized_weights[0]) <= normalized_threshold
            else "direct"
        )
    report(f"Method: {selected_method}")

    if selected_method == "exact":
        return _exact_subset_sum(
            normalized_weights,
            normalized_targets,
            normalized_modulus,
        )
    if selected_method == "kernel":
        return _kernel_subset_sum(
            normalized_weights,
            normalized_targets,
            normalized_modulus,
            normalized_effort,
            lattice_reduction,
            report,
        )
    return _direct_subset_sum(
        normalized_weights,
        normalized_targets,
        normalized_modulus,
        embedding_scale,
        normalized_effort,
        lattice_reduction,
        report,
    )
