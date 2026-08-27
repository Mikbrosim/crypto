"""Lattice-reduction helpers backed by flatter and SageMath."""

from __future__ import annotations

import logging
import re
import shutil
import subprocess
from typing import Any

from sage.all import Matrix, ZZ


LOGGER = logging.getLogger(__name__)
_ROW_PATTERN = re.compile(r"\[([^\[\]]*)\]")


class FlatterError(RuntimeError):
    """Raised when flatter cannot produce a valid reduced basis."""


def _serialize_matrix(matrix: Any) -> str:
    """Serialize a non-empty integer matrix in flatter's input format."""
    try:
        row_count = matrix.nrows()
        column_count = matrix.ncols()
        base_ring = matrix.base_ring()
    except AttributeError as error:
        raise TypeError("matrix must be a SageMath matrix") from error

    if row_count == 0 or column_count == 0:
        raise ValueError("matrix must have at least one row and one column")
    if base_ring != ZZ:
        raise TypeError(f"matrix must be defined over ZZ, not {base_ring}")

    rows = [" ".join(str(entry) for entry in row) for row in matrix]
    return "[[" + "]\n[".join(rows) + "]]"


def _parse_matrix(output: str, row_count: int, column_count: int) -> Any:
    """Parse and validate a matrix emitted by flatter."""
    text = output.strip()
    if not (text.startswith("[") and text.endswith("]")):
        raise FlatterError("flatter returned output that is not a matrix")

    body = text[1:-1].strip()
    row_texts = _ROW_PATTERN.findall(body)
    unmatched_text = _ROW_PATTERN.sub("", body)
    if unmatched_text.strip():
        raise FlatterError("flatter returned unexpected text with its matrix")
    if len(row_texts) != row_count:
        raise FlatterError(
            f"flatter returned {len(row_texts)} rows; expected {row_count}"
        )

    rows: list[list[int]] = []
    for row_number, row_text in enumerate(row_texts, start=1):
        tokens = row_text.split()
        if len(tokens) != column_count:
            raise FlatterError(
                f"flatter returned {len(tokens)} columns in row {row_number}; "
                f"expected {column_count}"
            )
        try:
            rows.append([int(token) for token in tokens])
        except ValueError as error:
            raise FlatterError(
                f"flatter returned a non-integer value in row {row_number}"
            ) from error

    return Matrix(ZZ, rows)


def _failure_result(matrix: Any, message: str, fallback_to_lll: bool) -> Any:
    """Either reduce with Sage LLL or raise a descriptive flatter error."""
    if not fallback_to_lll:
        raise FlatterError(message)

    LOGGER.warning("%s; falling back to SageMath LLL", message)
    return matrix.LLL()


def flatter(
    matrix: Any,
    *,
    executable: str = "flatter",
    fallback_to_lll: bool = True,
    timeout: float | None = None,
) -> Any:
    """Reduce an integer lattice basis using flatter.

    The one-argument call remains suitable for use as a lattice-reduction
    callback. If flatter is unavailable or fails, SageMath's ``LLL`` method is
    used by default. Set ``fallback_to_lll`` to ``False`` to receive a
    :class:`FlatterError` instead.

    Args:
        matrix: A non-empty SageMath matrix defined over ``ZZ``.
        executable: The flatter executable name or path.
        fallback_to_lll: Whether to use SageMath LLL after a flatter failure.
        timeout: Optional maximum number of seconds to allow flatter to run.

    Returns:
        A reduced SageMath integer matrix with the same dimensions as ``matrix``.

    Raises:
        TypeError: If ``matrix`` is not a SageMath integer matrix.
        ValueError: If ``matrix`` is empty or ``timeout`` is not positive.
        FlatterError: If flatter fails and fallback is disabled.
    """
    serialized_matrix = _serialize_matrix(matrix)

    if timeout is not None and timeout <= 0:
        raise ValueError("timeout must be positive or None")

    resolved_executable = shutil.which(executable)
    if resolved_executable is None:
        return _failure_result(
            matrix,
            f"flatter executable {executable!r} was not found",
            fallback_to_lll,
        )

    try:
        process = subprocess.run(
            [resolved_executable],
            input=serialized_matrix,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return _failure_result(
            matrix,
            f"flatter exceeded its {timeout}-second timeout",
            fallback_to_lll,
        )
    except OSError as error:
        return _failure_result(
            matrix,
            f"flatter could not be started: {error}",
            fallback_to_lll,
        )

    if process.returncode != 0:
        message = f"flatter exited with status {process.returncode}"
        if stderr := process.stderr.strip():
            message = f"{message}: {stderr}"
        return _failure_result(matrix, message, fallback_to_lll)

    try:
        return _parse_matrix(process.stdout, matrix.nrows(), matrix.ncols())
    except FlatterError as error:
        return _failure_result(matrix, str(error), fallback_to_lll)
