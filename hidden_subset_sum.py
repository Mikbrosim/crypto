"""Recover hidden weights and binary vectors from modular subset sums.

The module implements both the Nguyen--Stern attack and the polynomial-time
multivariate attack of Coron and Gini.  Recovered candidates are always checked
against every public sample before they are returned.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from math import ceil, gcd, log2
from operator import index as integer_index
from typing import Any, Literal

from sage.all import GF, Integers, Matrix, RealField, ZZ, random_prime, vector

from rewrite.lattice_reduction import flatter


HiddenSubsetSumMethod = Literal["auto", "nguyen_stern", "multivariate"]
KernelMethod = Literal["auto", "exact", "lll"]
LatticeReduction = Callable[[Any], Any]


@dataclass(frozen=True, slots=True)
class HiddenSubsetSumSolution:
    """A verified solution, with the hidden binary vectors stored as rows."""

    binary_vectors: Any
    weights: tuple[int, ...]
    method: str

    def as_tuple(self) -> tuple[Any, Any]:
        """Return the matrix/vector pair used by the legacy API."""
        return self.binary_vectors, vector(ZZ, self.weights)


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
        _exact_integer(value, f"{name}[{position}]")
        for position, value in enumerate(raw_values)
    )


def _positive_integer(value: Any, name: str, minimum: int = 1) -> int:
    """Validate an integer lower bound."""
    normalized = _exact_integer(value, name)
    if normalized < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return normalized


def _normalize_modulus(modulus: Any) -> int:
    """Validate a non-trivial modular ring size."""
    normalized = _exact_integer(modulus, "modulus")
    if normalized <= 1:
        raise ValueError("modulus must be greater than one")
    return normalized


def _normalize_samples(
    samples: Iterable[Any],
    modulus: int,
) -> tuple[int, ...]:
    """Normalize public samples into canonical modular representatives."""
    normalized = _integer_tuple(samples, "samples")
    if not normalized:
        raise ValueError("samples must not be empty")
    return tuple(value % modulus for value in normalized)


def _normalize_attack_problem(
    dimension: Any,
    modulus: Any,
    samples: Iterable[Any],
) -> tuple[int, int, tuple[int, ...]]:
    """Validate the common hidden-subset-sum attack inputs."""
    normalized_dimension = _positive_integer(dimension, "dimension")
    normalized_modulus = _normalize_modulus(modulus)
    normalized_samples = _normalize_samples(samples, normalized_modulus)
    if len(normalized_samples) < normalized_dimension:
        raise ValueError("the sample count must be at least the hidden dimension")
    return normalized_dimension, normalized_modulus, normalized_samples


def _normalize_reduction(
    effort: Any,
    lattice_reduction: LatticeReduction | None,
) -> int:
    """Validate the common LLL/BKZ controls."""
    normalized = _positive_integer(effort, "effort")
    if lattice_reduction is not None and not callable(lattice_reduction):
        raise TypeError("lattice_reduction must be callable or None")
    if lattice_reduction is not None and normalized != 1:
        raise ValueError("effort must be 1 with a custom lattice reduction")
    return normalized


def _reduce_basis(
    basis: Any,
    effort: int,
    lattice_reduction: LatticeReduction | None,
) -> Any:
    """Reduce and validate a full-row-rank integer lattice basis."""
    integer_basis = Matrix(ZZ, basis)
    if integer_basis.nrows() == 0:
        return integer_basis
    if integer_basis.rank() != integer_basis.nrows():
        raise ValueError("lattice basis must have full row rank")
    if effort > integer_basis.nrows():
        raise ValueError(
            f"effort cannot exceed the lattice row count "
            f"({integer_basis.nrows()})"
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
        raise ValueError("lattice reduction changed the basis dimensions")
    if reduced.rank() != integer_basis.nrows():
        raise ValueError("lattice reduction returned a rank-deficient basis")
    return reduced


def generate_pseudoprime(bits: int, minimum_factor_bits: int = 211) -> int:
    """Generate a prime or product of primes with approximately ``bits`` bits."""
    normalized_bits = _positive_integer(bits, "bits", 2)
    factor_bits = _positive_integer(
        minimum_factor_bits,
        "minimum_factor_bits",
        2,
    )
    if normalized_bits <= 2 * factor_bits:
        return int(
            random_prime(
                1 << normalized_bits,
                proof=False,
                lbound=1 << (normalized_bits - 1),
            )
        )
    factor = random_prime(
        1 << factor_bits,
        proof=False,
        lbound=1 << (factor_bits - 1),
    )
    return int(factor) * generate_pseudoprime(
        normalized_bits - factor_bits,
        factor_bits,
    )


def _random_binary_matrix(
    sample_count: int,
    dimension: int,
    modulus: int,
) -> Any:
    """Generate a full-rank binary matrix with a unit leading minor."""
    while True:
        binary_matrix = Matrix(
            ZZ,
            sample_count,
            dimension,
            [ZZ.random_element(2) for _ in range(sample_count * dimension)],
        )
        leading_minor = binary_matrix[:dimension, :]
        if (
            binary_matrix.rank() == dimension
            and gcd(abs(int(leading_minor.det())), modulus) == 1
        ):
            return binary_matrix


def generate_instance(
    dimension: int = 10,
    sample_count: int = 20,
    modulus_bits: int = 100,
    *,
    modulus: int | None = None,
) -> tuple[int, Any, Any, Any]:
    """Generate ``b = X*a (mod modulus)`` with a full-rank binary ``X``."""
    normalized_dimension = _positive_integer(dimension, "dimension")
    normalized_count = _positive_integer(sample_count, "sample_count")
    if normalized_count < normalized_dimension:
        raise ValueError("sample_count must be at least dimension")
    if modulus is None:
        normalized_modulus = generate_pseudoprime(modulus_bits)
    else:
        normalized_modulus = _normalize_modulus(modulus)

    hidden_weights = vector(
        ZZ,
        [
            ZZ.random_element(normalized_modulus)
            for _ in range(normalized_dimension)
        ],
    )
    binary_matrix = _random_binary_matrix(
        normalized_count,
        normalized_dimension,
        normalized_modulus,
    )
    samples = vector(
        ZZ,
        [
            int(value) % normalized_modulus
            for value in binary_matrix * hidden_weights
        ],
    )
    return normalized_modulus, hidden_weights, binary_matrix, samples


def _smith_orthogonal_lattice(samples: Sequence[int], modulus: int) -> Any:
    """Construct the modular orthogonal lattice without requiring a unit."""
    sample_count = len(samples)
    equation = Matrix(ZZ, [list(samples) + [modulus]])
    _, _, right_transform = equation.smith_form()
    generators = [
        list(right_transform.column(index)[:sample_count])
        for index in range(1, sample_count + 1)
    ]
    return Matrix(ZZ, generators).row_module().basis_matrix()


def orthogonal_lattice(samples: Iterable[int], modulus: int) -> Any:
    """Return ``{u in ZZ^m : <u, samples> = 0 (mod modulus)}``."""
    normalized_modulus = _normalize_modulus(modulus)
    normalized_samples = _normalize_samples(samples, normalized_modulus)
    sample_count = len(normalized_samples)

    pivot = next(
        (
            index
            for index, value in enumerate(normalized_samples)
            if gcd(value, normalized_modulus) == 1
        ),
        None,
    )
    if pivot is None:
        basis = _smith_orthogonal_lattice(
            normalized_samples,
            normalized_modulus,
        )
    else:
        inverse = pow(normalized_samples[pivot], -1, normalized_modulus)
        rows: list[list[int]] = []
        for row_index in range(sample_count):
            row = [0] * sample_count
            if row_index == pivot:
                row[pivot] = normalized_modulus
            else:
                row[row_index] = 1
                row[pivot] = (
                    -normalized_samples[row_index] * inverse
                ) % normalized_modulus
            rows.append(row)
        basis = Matrix(ZZ, rows)

    if basis.rank() != sample_count:
        raise RuntimeError("failed to construct a full-rank orthogonal lattice")
    if any(
        sum(int(entry) * sample for entry, sample in zip(row, normalized_samples))
        % normalized_modulus
        for row in basis.rows()
    ):
        raise RuntimeError("constructed basis is not modularly orthogonal")
    return basis


def _exact_integer_right_kernel(integer_matrix: Any) -> Any:
    """Compute a saturated right kernel with Smith normal form."""
    column_count = integer_matrix.ncols()
    rank = integer_matrix.rank()
    if rank == column_count:
        return Matrix(ZZ, 0, column_count)
    _, _, right_transform = integer_matrix.smith_form()
    generators = [
        list(right_transform.column(index))
        for index in range(rank, column_count)
    ]
    return Matrix(ZZ, generators).row_module().basis_matrix()


def _lll_integer_right_kernel(
    integer_matrix: Any,
    effort: int,
    lattice_reduction: LatticeReduction | None,
) -> Any | None:
    """Use the scaled Nguyen--Stern embedding to expose a right kernel."""
    row_count = integer_matrix.nrows()
    column_count = integer_matrix.ncols()
    expected_dimension = column_count - integer_matrix.rank()
    if expected_dimension == 0:
        return Matrix(ZZ, 0, column_count)

    height = max(
        [abs(int(entry)) for entry in integer_matrix.list()],
        default=0,
    )
    scale = (1 << (column_count // 2)) * max(1, height)
    embedded_rows = []
    for column in range(column_count):
        row = [scale * integer_matrix[index, column] for index in range(row_count)]
        row.extend(int(column == index) for index in range(column_count))
        embedded_rows.append(row)
    embedded = Matrix(ZZ, embedded_rows)
    reduced = _reduce_basis(embedded, effort, lattice_reduction)

    candidates = [
        list(row[row_count:])
        for row in reduced.rows()
        if not any(row[:row_count]) and any(row[row_count:])
    ]
    if not candidates:
        return None
    kernel = Matrix(ZZ, candidates).row_module().basis_matrix()
    if kernel.rank() != expected_dimension:
        return None
    if integer_matrix * kernel.T != 0:
        return None
    return kernel


def integer_right_kernel(
    integer_matrix: Any,
    *,
    method: KernelMethod = "auto",
    effort: int = 1,
    lattice_reduction: LatticeReduction | None = None,
) -> Any:
    """Compute and verify an integer right-kernel basis."""
    try:
        normalized_matrix = Matrix(ZZ, integer_matrix)
    except (TypeError, ValueError) as error:
        raise TypeError("integer_matrix must be an integer matrix") from error
    if normalized_matrix.ncols() == 0:
        raise ValueError("integer_matrix must contain at least one column")
    if method not in ("auto", "exact", "lll"):
        raise ValueError("method must be 'auto', 'exact', or 'lll'")
    normalized_effort = _normalize_reduction(effort, lattice_reduction)

    use_exact = method == "exact" or (
        method == "auto"
        and normalized_matrix.ncols() < 2 * normalized_matrix.nrows()
    )
    if use_exact:
        kernel = _exact_integer_right_kernel(normalized_matrix)
    else:
        kernel = _lll_integer_right_kernel(
            normalized_matrix,
            normalized_effort,
            lattice_reduction,
        )
        if kernel is None:
            if method == "lll":
                raise RuntimeError("LLL did not expose the full integer kernel")
            kernel = _exact_integer_right_kernel(normalized_matrix)

    expected_dimension = normalized_matrix.ncols() - normalized_matrix.rank()
    if kernel.dimensions() != (expected_dimension, normalized_matrix.ncols()):
        raise RuntimeError("integer kernel has unexpected dimensions")
    if normalized_matrix * kernel.T != 0:
        raise RuntimeError("integer kernel verification failed")
    return kernel


def _canonical_binary(values: Iterable[Any]) -> tuple[int, ...] | None:
    """Canonicalize a {0,1} or {0,-1} vector."""
    normalized = _integer_tuple(values, "values")
    if all(value in (0, 1) for value in normalized):
        return normalized
    if all(value in (0, -1) for value in normalized):
        return tuple(-value for value in normalized)
    return None


def _select_independent_rows(
    rows: Sequence[tuple[int, ...]],
    expected_count: int,
) -> Any | None:
    """Select the first ``expected_count`` independent binary rows."""
    selected: list[tuple[int, ...]] = []
    current_rank = 0
    for row in rows:
        trial = Matrix(ZZ, selected + [row])
        trial_rank = trial.rank()
        if trial_rank == current_rank:
            continue
        selected.append(row)
        current_rank = trial_rank
        if len(selected) == expected_count:
            return Matrix(ZZ, selected)
    return None


def recover_binary_vectors(
    reduced_basis: Any,
    expected_count: int | None = None,
    *,
    max_candidates: int = 10_000,
) -> Any | None:
    """Recover binary vectors from a basis containing vectors and differences."""
    try:
        basis = Matrix(ZZ, reduced_basis)
    except (TypeError, ValueError) as error:
        raise TypeError("reduced_basis must be an integer matrix") from error
    if basis.ncols() == 0:
        raise ValueError("reduced_basis must contain at least one column")
    candidate_limit = _positive_integer(max_candidates, "max_candidates")
    if expected_count is None:
        normalized_count = None
    else:
        normalized_count = _positive_integer(expected_count, "expected_count")
        if normalized_count > basis.nrows():
            raise ValueError("expected_count cannot exceed the basis row count")

    original_rows = [tuple(map(int, row)) for row in basis.rows()]
    candidates: list[tuple[int, ...]] = []
    seen: set[tuple[int, ...]] = set()
    pending: deque[tuple[int, ...]] = deque()

    def add_candidate(values: Iterable[Any]) -> None:
        candidate = _canonical_binary(values)
        if candidate is None or not any(candidate) or candidate in seen:
            return
        seen.add(candidate)
        candidates.append(candidate)
        pending.append(candidate)

    for row in original_rows:
        add_candidate(row)
    if normalized_count is not None:
        selected = _select_independent_rows(candidates, normalized_count)
        if selected is not None:
            return selected

    while pending:
        recovered = pending.popleft()
        for basis_row in original_rows:
            add_candidate(
                left - right
                for left, right in zip(basis_row, recovered, strict=True)
            )
            add_candidate(
                left + right
                for left, right in zip(basis_row, recovered, strict=True)
            )
            if len(candidates) >= candidate_limit:
                raise RuntimeError("binary recovery exceeded max_candidates")
        if normalized_count is not None:
            selected = _select_independent_rows(candidates, normalized_count)
            if selected is not None:
                return selected

    if normalized_count is not None:
        return None
    if not candidates:
        return Matrix(ZZ, 0, basis.ncols())
    return Matrix(ZZ, candidates)


def _is_verified_solution(
    binary_vectors: Any,
    weights: Sequence[int],
    samples: Sequence[int],
    modulus: int,
    dimension: int,
) -> bool:
    """Verify rank, binary coordinates, and every modular sample."""
    if binary_vectors.dimensions() != (dimension, len(samples)):
        return False
    if binary_vectors.rank() != dimension:
        return False
    if any(int(entry) not in (0, 1) for entry in binary_vectors.list()):
        return False
    if len(weights) != dimension:
        return False
    reconstructed = binary_vectors.T * vector(ZZ, weights)
    return all(
        (int(value) - sample) % modulus == 0
        for value, sample in zip(reconstructed, samples, strict=True)
    )


def _candidate_minors(coefficient_matrix: Any) -> list[tuple[int, ...]]:
    """Return deterministic square minors likely to be invertible modulo M."""
    sample_count = coefficient_matrix.nrows()
    dimension = coefficient_matrix.ncols()
    candidates: list[tuple[int, ...]] = []

    def add(indices: Iterable[int]) -> None:
        candidate = tuple(indices)
        if len(candidate) == dimension and candidate not in candidates:
            candidates.append(candidate)

    add(range(dimension))
    if coefficient_matrix.rank() == dimension:
        add(coefficient_matrix.T.pivots())
    for start in range(0, sample_count - dimension + 1, dimension):
        add(range(start, start + dimension))
        if len(candidates) >= 34:
            break
    return candidates


def _recover_weights(
    binary_vectors: Any,
    samples: Sequence[int],
    modulus: int,
) -> tuple[int, ...] | None:
    """Solve ``binary_vectors.T * weights = samples (mod modulus)``."""
    coefficients = binary_vectors.T
    modular_ring = Integers(modulus)
    for indices in _candidate_minors(coefficients):
        minor = coefficients.matrix_from_rows(indices)
        if minor.rank() != binary_vectors.nrows():
            continue
        if gcd(abs(int(minor.det())), modulus) != 1:
            continue
        modular_minor = Matrix(modular_ring, minor)
        right_hand_side = vector(
            modular_ring,
            [samples[index] for index in indices],
        )
        candidate = modular_minor.inverse() * right_hand_side
        weights = tuple(int(value) % modulus for value in candidate)
        if _is_verified_solution(
            binary_vectors,
            weights,
            samples,
            modulus,
            binary_vectors.nrows(),
        ):
            return weights

    try:
        candidate = Matrix(modular_ring, coefficients).solve_right(
            vector(modular_ring, samples)
        )
    except (ArithmeticError, TypeError, ValueError):
        return None
    weights = tuple(int(value) % modulus for value in candidate)
    return (
        weights
        if _is_verified_solution(
            binary_vectors,
            weights,
            samples,
            modulus,
            binary_vectors.nrows(),
        )
        else None
    )


def _make_solution(
    binary_vectors: Any | None,
    samples: Sequence[int],
    modulus: int,
    dimension: int,
    method: str,
) -> HiddenSubsetSumSolution | None:
    """Recover weights and construct a fully verified result."""
    if binary_vectors is None:
        return None
    normalized_vectors = Matrix(ZZ, binary_vectors)
    weights = _recover_weights(normalized_vectors, samples, modulus)
    if weights is None:
        return None
    if not _is_verified_solution(
        normalized_vectors,
        weights,
        samples,
        modulus,
        dimension,
    ):
        return None
    return HiddenSubsetSumSolution(normalized_vectors, weights, method)


def nguyen_stern_attack(
    dimension: int,
    modulus: int,
    samples: Iterable[int],
    *,
    effort: int = 1,
    kernel_method: KernelMethod = "auto",
    lattice_reduction: LatticeReduction | None = None,
    max_candidates: int = 10_000,
    verbose: bool = False,
) -> HiddenSubsetSumSolution | None:
    """Run the lower-sample Nguyen--Stern orthogonal-lattice attack."""
    normalized_dimension, normalized_modulus, normalized_samples = (
        _normalize_attack_problem(dimension, modulus, samples)
    )
    sample_count = len(normalized_samples)
    if sample_count < 2 * normalized_dimension:
        raise ValueError("Nguyen--Stern requires at least 2*dimension samples")
    normalized_effort = _normalize_reduction(effort, lattice_reduction)
    if kernel_method not in ("auto", "exact", "lll"):
        raise ValueError("kernel_method must be 'auto', 'exact', or 'lll'")
    if not isinstance(verbose, bool):
        raise TypeError("verbose must be a bool")
    report = (
        (lambda *args: print("[hidden_subset_sum]", *args))
        if verbose
        else lambda *_: None
    )

    modular_orthogonal = orthogonal_lattice(
        normalized_samples,
        normalized_modulus,
    )
    report(f"Orthogonal lattice: {modular_orthogonal.dimensions()}")
    reduced_orthogonal = _reduce_basis(
        modular_orthogonal,
        1,
        lattice_reduction,
    )
    hidden_orthogonal = reduced_orthogonal[: sample_count - normalized_dimension]
    if hidden_orthogonal.rank() != sample_count - normalized_dimension:
        return None

    hidden_lattice = integer_right_kernel(
        hidden_orthogonal,
        method=kernel_method,
        effort=1,
        lattice_reduction=lattice_reduction,
    )
    if hidden_lattice.nrows() != normalized_dimension:
        return None
    report(f"Recovered hidden lattice: {hidden_lattice.dimensions()}")
    reduced_hidden = _reduce_basis(
        hidden_lattice,
        normalized_effort,
        lattice_reduction,
    )
    binary_vectors = recover_binary_vectors(
        reduced_hidden,
        normalized_dimension,
        max_candidates=max_candidates,
    )
    return _make_solution(
        binary_vectors,
        normalized_samples,
        normalized_modulus,
        normalized_dimension,
        "nguyen_stern",
    )


def minimum_multivariate_samples(dimension: int) -> int:
    """Return the paper implementation's sample count rounded to a block."""
    normalized_dimension = _positive_integer(dimension, "dimension", 2)
    block_count = (
        (normalized_dimension + 3) // 2
        if normalized_dimension % 2
        else (normalized_dimension + 4) // 2
    )
    return normalized_dimension * block_count


def matrix_bit_length(integer_matrix: Any) -> int:
    """Return the largest absolute entry bit length, or zero for a zero matrix."""
    normalized_matrix = Matrix(ZZ, integer_matrix)
    return max(
        (abs(int(entry)).bit_length() for entry in normalized_matrix.list()),
        default=0,
    )


def round_matrix(real_matrix: Any) -> Any:
    """Round every real matrix entry to its nearest integer."""
    return Matrix(
        ZZ,
        real_matrix.nrows(),
        real_matrix.ncols(),
        [entry.round() for entry in real_matrix.list()],
    )


def _unit_pivot(samples: Sequence[int], modulus: int) -> int | None:
    """Find a public sample invertible modulo the modulus."""
    return next(
        (
            index
            for index, value in enumerate(samples)
            if gcd(value, modulus) == 1
        ),
        None,
    )


def _structured_orthogonal_lattice(
    samples: Sequence[int],
    dimension: int,
    modulus: int,
    *,
    allow_permutation: bool,
) -> tuple[Any, tuple[int, ...]]:
    """Build the compressed orthogonal matrix used by the fast first step."""
    sample_count = len(samples)
    if sample_count < 3 * dimension:
        raise ValueError("multivariate attack requires at least 3*dimension samples")
    if sample_count % dimension:
        raise ValueError("the multivariate sample count must be divisible by dimension")

    pivot = _unit_pivot(samples, modulus)
    if pivot is None:
        raise ValueError("at least one sample must be invertible modulo modulus")
    if pivot and not allow_permutation:
        raise ValueError("samples[0] must be invertible modulo modulus")
    order = list(range(sample_count))
    if pivot:
        order[0], order[pivot] = order[pivot], order[0]
    permuted_samples = tuple(samples[index] for index in order)

    compressed = Matrix(ZZ, sample_count, 3 * dimension)
    for index in range(2 * dimension):
        compressed[index, index] = 1
    for block in range(2, sample_count // dimension):
        start = block * dimension
        for index in range(dimension):
            compressed[start + index, 2 * dimension + index] = 1

    inverse = pow(permuted_samples[0], -1, modulus)
    compressed[0, 0] = modulus
    for row in range(1, sample_count):
        compressed[row, 0] = (-permuted_samples[row] * inverse) % modulus
    return compressed, tuple(order)


def _size_reduce_blocks(
    work_matrix: Any,
    first_row: int,
    block_size: int,
    column_count: int,
    reducer_rows: Any,
    inverse: Any,
    max_rounds: int,
) -> None:
    """Size-reduce lower row blocks and stop only at a fixed point."""
    for _ in range(max_rounds):
        changed = False
        for start in range(first_row, work_matrix.nrows(), block_size):
            stop = min(work_matrix.nrows(), start + block_size)
            multiplier = round_matrix(
                work_matrix[start:stop, :column_count] * inverse
            )
            if multiplier == 0:
                continue
            work_matrix[start:stop, :] -= multiplier * reducer_rows
            changed = True
        if not changed:
            return
    raise RuntimeError("numerical size reduction did not converge")


def _multivariate_hidden_basis(
    dimension: int,
    modulus: int,
    samples: Sequence[int],
    effort: int,
    lattice_reduction: LatticeReduction | None,
    kernel_reduction: LatticeReduction | None,
    kernel_method: KernelMethod,
    chunk_count: int,
    precision_bits: int | None,
    max_rounds: int,
    report: Callable[..., None],
) -> tuple[Any, tuple[int, ...]]:
    """Recover a hidden-lattice basis over GF(3) with structured reduction."""
    compressed, order = _structured_orthogonal_lattice(
        samples,
        dimension,
        modulus,
        allow_permutation=True,
    )
    sample_count = len(samples)
    normalized_chunks = min(
        dimension,
        _positive_integer(chunk_count, "chunk_count"),
    )
    chunk_size = max(1, dimension // normalized_chunks)

    seed = _reduce_basis(
        compressed[:chunk_size, :chunk_size],
        1,
        lattice_reduction,
    )
    compressed[:chunk_size, :chunk_size] = seed
    leading = _reduce_basis(
        compressed[: 2 * dimension, : 2 * dimension],
        effort,
        lattice_reduction,
    )

    requested_precision = (
        None
        if precision_bits is None
        else _positive_integer(precision_bits, "precision_bits", 53)
    )
    precision = requested_precision or max(
        128,
        2 * matrix_bit_length(compressed) + 64,
    )
    real_field = RealField(precision)
    seed_inverse = Matrix(
        real_field,
        compressed[:chunk_size, :chunk_size],
    ).inverse()
    _size_reduce_blocks(
        compressed,
        2 * dimension,
        chunk_size,
        chunk_size,
        compressed[:chunk_size, :],
        seed_inverse,
        max_rounds,
    )

    compressed[: 2 * dimension, : 2 * dimension] = leading
    leading_inverse = Matrix(real_field, leading).inverse()
    _size_reduce_blocks(
        compressed,
        2 * dimension,
        dimension,
        2 * dimension,
        compressed[: 2 * dimension, :],
        leading_inverse,
        max_rounds,
    )

    first_kernel = integer_right_kernel(
        compressed[:dimension, : 2 * dimension],
        method=kernel_method,
        effort=effort,
        lattice_reduction=kernel_reduction,
    )
    if first_kernel.nrows() != dimension:
        raise RuntimeError("first-stage kernel has unexpected dimension")
    field = GF(3)
    hidden_basis = Matrix(field, dimension, sample_count)
    hidden_basis[:, : 2 * dimension] = Matrix(field, first_kernel)
    for start in range(2 * dimension, sample_count, dimension):
        stop = start + dimension
        block = compressed[start:stop, : 2 * dimension]
        hidden_basis[:, start:stop] = -(
            Matrix(field, block) * hidden_basis[:, : 2 * dimension].T
        ).T
    if hidden_basis.rank() != dimension:
        raise RuntimeError("recovered hidden basis is rank deficient over GF(3)")
    report(f"Hidden GF(3) basis: {hidden_basis.dimensions()}")
    return hidden_basis, order


def recover_multivariate_binary_vectors(
    hidden_basis: Any,
) -> Any | None:
    """Recover binary generators using quadratic linearization over GF(3)."""
    field = GF(3)
    basis = Matrix(field, hidden_basis)
    dimension = basis.nrows()
    if dimension == 0 or basis.ncols() == 0:
        raise ValueError("hidden_basis must be non-empty")
    if basis.rank() != dimension:
        raise ValueError("hidden_basis must have full row rank")

    quadratic_pairs = [
        (left, right)
        for left in range(dimension)
        for right in range(left, dimension)
    ]
    equations = []
    for column in basis.columns():
        row = [-entry for entry in column]
        row.extend(
            column[left]
            * column[right]
            * (1 if left == right else 2)
            for left, right in quadratic_pairs
        )
        equations.append(row)
    linearized = Matrix(field, equations)
    kernel = linearized.right_kernel().basis_matrix()
    if kernel.nrows() != dimension:
        return None

    linear_part = kernel[:, :dimension]
    if linear_part.rank() != dimension:
        return None
    normalized_kernel = linear_part.inverse() * kernel

    multiplication_data = Matrix(field, dimension, dimension * dimension)
    for offset, (left, right) in enumerate(quadratic_pairs):
        coefficient = normalized_kernel.column(dimension + offset)
        multiplication_data[:, left * dimension + right] = coefficient
        multiplication_data[:, right * dimension + left] = coefficient

    subspaces = [Matrix.identity(field, dimension)]
    for coordinate in range(dimension):
        multiplication = multiplication_data[
            :,
            coordinate * dimension : (coordinate + 1) * dimension,
        ]
        refined = []
        for subspace in subspaces:
            if subspace.nrows() == 1:
                refined.append(subspace)
                continue
            restriction = subspace.solve_left(subspace * multiplication)
            for eigenspace_data in restriction.eigenspaces_left():
                eigenspace = eigenspace_data[1].basis_matrix()
                refined.append(eigenspace * subspace)
        subspaces = refined

    if len(subspaces) != dimension or any(
        subspace.nrows() != 1 for subspace in subspaces
    ):
        return None
    candidates: list[tuple[int, ...]] = []
    for subspace in subspaces:
        candidate = tuple(int(value) for value in subspace[0] * basis)
        if all(value in (0, 1) for value in candidate):
            normalized = candidate
        elif all(value in (0, 2) for value in candidate):
            normalized = tuple((2 * value) % 3 for value in candidate)
        else:
            return None
        if not any(normalized) or normalized in candidates:
            return None
        candidates.append(normalized)

    binary_vectors = Matrix(ZZ, candidates)
    return binary_vectors if binary_vectors.rank() == dimension else None


def _restore_sample_order(binary_vectors: Any, order: Sequence[int]) -> Any:
    """Undo the sample permutation used to obtain a unit leading sample."""
    restored = Matrix(ZZ, binary_vectors.nrows(), binary_vectors.ncols())
    for permuted_index, original_index in enumerate(order):
        restored[:, original_index] = binary_vectors[:, permuted_index]
    return restored


def multivariate_attack(
    dimension: int,
    modulus: int,
    samples: Iterable[int],
    *,
    effort: int = 1,
    kernel_method: KernelMethod = "auto",
    lattice_reduction: LatticeReduction | None = None,
    kernel_reduction: LatticeReduction | None = None,
    chunk_count: int = 4,
    precision_bits: int | None = None,
    max_rounds: int = 64,
    verbose: bool = False,
) -> HiddenSubsetSumSolution | None:
    """Run the structured polynomial-time multivariate HSSP attack."""
    normalized_dimension, normalized_modulus, normalized_samples = (
        _normalize_attack_problem(dimension, modulus, samples)
    )
    sample_count = len(normalized_samples)
    if sample_count < minimum_multivariate_samples(normalized_dimension):
        raise ValueError("insufficient samples for the multivariate attack")
    if sample_count % normalized_dimension:
        raise ValueError("the multivariate sample count must be divisible by dimension")
    normalized_effort = _normalize_reduction(effort, lattice_reduction)
    _normalize_reduction(effort, kernel_reduction)
    normalized_rounds = _positive_integer(max_rounds, "max_rounds")
    if kernel_method not in ("auto", "exact", "lll"):
        raise ValueError("kernel_method must be 'auto', 'exact', or 'lll'")
    if not isinstance(verbose, bool):
        raise TypeError("verbose must be a bool")
    report = (
        (lambda *args: print("[hidden_subset_sum]", *args))
        if verbose
        else lambda *_: None
    )

    hidden_basis, order = _multivariate_hidden_basis(
        normalized_dimension,
        normalized_modulus,
        normalized_samples,
        normalized_effort,
        lattice_reduction,
        kernel_reduction,
        kernel_method,
        chunk_count,
        precision_bits,
        normalized_rounds,
        report,
    )
    permuted_binary = recover_multivariate_binary_vectors(hidden_basis)
    binary_vectors = (
        None
        if permuted_binary is None
        else _restore_sample_order(permuted_binary, order)
    )
    return _make_solution(
        binary_vectors,
        normalized_samples,
        normalized_modulus,
        normalized_dimension,
        "multivariate",
    )


def recover_hidden_subset_sum(
    dimension: int,
    modulus: int,
    samples: Iterable[int],
    *,
    method: HiddenSubsetSumMethod = "auto",
    effort: int = 1,
    kernel_method: KernelMethod = "auto",
    lattice_reduction: LatticeReduction | None = None,
    kernel_reduction: LatticeReduction | None = None,
    max_candidates: int = 10_000,
    chunk_count: int = 4,
    precision_bits: int | None = None,
    max_rounds: int = 64,
    verbose: bool = False,
) -> HiddenSubsetSumSolution | None:
    """Recover a hidden subset-sum instance with a selected attack."""
    normalized_dimension, normalized_modulus, normalized_samples = (
        _normalize_attack_problem(dimension, modulus, samples)
    )
    if method not in ("auto", "nguyen_stern", "multivariate"):
        raise ValueError(
            "method must be 'auto', 'nguyen_stern', or 'multivariate'"
        )
    selected_method = method
    if selected_method == "auto":
        enough_samples = len(normalized_samples) >= minimum_multivariate_samples(
            normalized_dimension
        )
        selected_method = (
            "multivariate"
            if enough_samples and len(normalized_samples) % normalized_dimension == 0
            else "nguyen_stern"
        )

    if selected_method == "multivariate":
        return multivariate_attack(
            normalized_dimension,
            normalized_modulus,
            normalized_samples,
            effort=effort,
            kernel_method=kernel_method,
            lattice_reduction=lattice_reduction,
            kernel_reduction=kernel_reduction,
            chunk_count=chunk_count,
            precision_bits=precision_bits,
            max_rounds=max_rounds,
            verbose=verbose,
        )
    return nguyen_stern_attack(
        normalized_dimension,
        normalized_modulus,
        normalized_samples,
        effort=effort,
        kernel_method=kernel_method,
        lattice_reduction=lattice_reduction,
        max_candidates=max_candidates,
        verbose=verbose,
    )


# Legacy compatibility wrappers -------------------------------------------------


def genpseudoprime(eta: int, etamin: int = 211) -> int:
    """Compatibility wrapper for :func:`generate_pseudoprime`."""
    return generate_pseudoprime(eta, etamin)


def genParams(
    n: int = 10,
    m: int = 20,
    nx0: int = 100,
) -> tuple[int, Any, Any, Any]:
    """Compatibility wrapper for :func:`generate_instance`."""
    return generate_instance(n, m, nx0)


def orthoLattice(b: Iterable[int], x0: int) -> Any:
    """Compatibility wrapper for :func:`orthogonal_lattice`."""
    return orthogonal_lattice(b, x0)


def allones(v: Iterable[int]) -> Any | None:
    """Return the canonical binary Sage vector used by the original code."""
    candidate = _canonical_binary(v)
    return None if candidate is None else vector(ZZ, candidate)


def recoverBinary(M5: Any) -> Any:
    """Compatibility wrapper for binary-basis recovery."""
    basis = Matrix(ZZ, M5)
    recovered = recover_binary_vectors(basis, basis.nrows())
    return (
        Matrix(ZZ, 0, basis.ncols())
        if recovered is None
        else recovered
    )


def allpmones(v: Iterable[int]) -> bool:
    """Return whether every coordinate belongs to {-1, 0, 1}."""
    return all(value in (-1, 0, 1) for value in _integer_tuple(v, "v"))


def kernelLLL(M: Any, use_flatter: bool = False) -> Any:
    """Compatibility wrapper for the LLL-based integer kernel."""
    if not isinstance(use_flatter, bool):
        raise TypeError("use_flatter must be a bool")
    normalized = Matrix(ZZ, M)
    method: KernelMethod = (
        "exact"
        if normalized.ncols() < 2 * normalized.nrows()
        else "auto"
    )
    return integer_right_kernel(
        normalized,
        method=method,
        lattice_reduction=flatter if use_flatter and method != "exact" else None,
    )


def NSattack(
    n: int = 60,
    *,
    instance: tuple[int, Any, Any, Any] | None = None,
    effort: int = 1,
    verbose: bool = True,
) -> tuple[Any, Any] | None:
    """Generate or accept an instance and run the Nguyen--Stern attack."""
    normalized_dimension = _positive_integer(n, "n", 2)
    if instance is None:
        sample_count = max(
            2 * normalized_dimension,
            ceil(16 * log2(normalized_dimension) + 24),
        )
        modulus_bits = max(
            2,
            int(
                2 * 0.035 * normalized_dimension**2
                + normalized_dimension * log2(normalized_dimension)
            ),
        )
        modulus, _, _, samples = generate_instance(
            normalized_dimension,
            sample_count,
            modulus_bits,
        )
    else:
        modulus, _, _, samples = instance
    solution = nguyen_stern_attack(
        normalized_dimension,
        modulus,
        samples,
        effort=effort,
        verbose=verbose,
    )
    return None if solution is None else solution.as_tuple()


def statNS(
    dimensions: Iterable[int] | None = None,
    *,
    effort: int = 1,
) -> list[tuple[int, tuple[Any, Any] | None]]:
    """Run explicitly requested Nguyen--Stern benchmark dimensions."""
    if dimensions is None:
        raise ValueError("dimensions must be supplied explicitly")
    results = []
    for dimension in dimensions:
        normalized = _positive_integer(dimension, "dimension", 2)
        results.append((normalized, NSattack(normalized, effort=effort)))
    return results


def matNbits(M: Any) -> int:
    """Compatibility wrapper for :func:`matrix_bit_length`."""
    return matrix_bit_length(M)


def roundM(M: Any) -> Any:
    """Compatibility wrapper for :func:`round_matrix`."""
    return round_matrix(M)


def orthoLatticeMod(b: Iterable[int], n: int, x0: int) -> Any:
    """Return the legacy compressed orthogonal matrix without permutation."""
    normalized_dimension, normalized_modulus, normalized_samples = (
        _normalize_attack_problem(n, x0, b)
    )
    compressed, _ = _structured_orthogonal_lattice(
        normalized_samples,
        normalized_dimension,
        normalized_modulus,
        allow_permutation=False,
    )
    return compressed


def NZeroVectors(M: Iterable[Any]) -> int:
    """Count zero vectors or scalar zero entries."""
    count = 0
    for value in M:
        try:
            is_zero = bool(value.is_zero())
        except AttributeError:
            try:
                is_zero = not any(value)
            except TypeError:
                is_zero = value == 0
        count += is_zero
    return count


def multiAttack_gen(
    n: int = 16,
    *,
    nx0: int | None = None,
) -> tuple[int, Any, Any, Any]:
    """Generate a multivariate instance in the original return order."""
    normalized_dimension = _positive_integer(n, "n", 2)
    modulus_bits = (
        max(
            2,
            int(
                2 * 0.035 * normalized_dimension**2
                + normalized_dimension * log2(normalized_dimension)
            ),
        )
        if nx0 is None
        else _positive_integer(nx0, "nx0", 2)
    )
    modulus, weights, binary_matrix, samples = generate_instance(
        normalized_dimension,
        minimum_multivariate_samples(normalized_dimension),
        modulus_bits,
    )
    return modulus, samples, weights, binary_matrix


def multiAttack_attack(
    n: int,
    x0: int,
    b: Iterable[int],
    use_flatter1: bool = True,
    use_flatter2: bool = True,
) -> tuple[Any, Any] | None:
    """Compatibility wrapper for :func:`multivariate_attack`."""
    if not isinstance(use_flatter1, bool) or not isinstance(use_flatter2, bool):
        raise TypeError("use_flatter flags must be bools")
    solution = multivariate_attack(
        n,
        x0,
        b,
        lattice_reduction=flatter if use_flatter1 else None,
        kernel_reduction=flatter if use_flatter2 else None,
        verbose=True,
    )
    return None if solution is None else solution.as_tuple()
