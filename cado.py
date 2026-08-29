"""Correctness-first adapter for CADO-NFS factoring and discrete logs."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from contextlib import contextmanager
import math
import os
from operator import index as integer_index
from pathlib import Path
from math import prod
import re
import shutil
from tempfile import mkdtemp
from typing import Iterator
from uuid import uuid4

from rewrite._external_process import ProcessExecutionError, run_process
from rewrite._number_theory import (
    crt_coprime,
    discrete_log_bsgs,
    is_probable_prime,
)


__all__ = [
    "CadoExecutionError",
    "CadoOutputError",
    "dlp",
    "factor",
    "pohlig_hellman",
]


DEFAULT_CADO_PATH = "cado-nfs.py"
DEFAULT_THREADS = os.cpu_count() or 1
# Original module-level configuration names retained for challenge scripts.
cado_path = DEFAULT_CADO_PATH
num_cpu_threads = DEFAULT_THREADS
_FACTOR_LINE_RE = re.compile(r"^\s*\d+(?:\s+\d+)*\s*$")
_DLP_LINE_RE = re.compile(r"^\s*-?\d+(?:\s*,\s*-?\d+)+\s*$")


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


def _positive_timeout(value: object | None) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise TypeError("timeout must be a real number or None, not bool")
    try:
        normalized = float(value)
    except (TypeError, ValueError) as error:
        raise TypeError("timeout must be a real number or None") from error
    if not math.isfinite(normalized) or normalized <= 0:
        raise ValueError("timeout must be finite and positive")
    return normalized


def _boolean(value: object, name: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a bool")
    return value


def _backend(value: object) -> str:
    if not isinstance(value, str):
        raise TypeError("backend must be a string")
    normalized = value.lower()
    if normalized not in {"auto", "cado", "sage"}:
        raise ValueError("backend must be 'auto', 'cado', or 'sage'")
    return normalized


def _resolve_executable(value: str | os.PathLike[str]) -> Path:
    try:
        text = os.fspath(value)
    except TypeError as error:
        raise TypeError("cado_path must be a string or path-like object") from error
    found = shutil.which(text)
    candidate = Path(found if found is not None else text).expanduser()
    try:
        candidate = candidate.resolve()
    except OSError as error:
        raise FileNotFoundError(f"cannot resolve CADO-NFS executable: {text}") from error
    if not candidate.is_file():
        raise FileNotFoundError(f"CADO-NFS executable does not exist: {candidate}")
    if not os.access(candidate, os.X_OK):
        raise PermissionError(f"CADO-NFS executable is not executable: {candidate}")
    return candidate


@contextmanager
def _workspace(
    workdir: str | os.PathLike[str] | None,
    *,
    prefix: str,
) -> Iterator[Path]:
    if workdir is None:
        directory = Path(mkdtemp(prefix=prefix))
        owned = True
    else:
        try:
            directory = Path(workdir).expanduser().resolve()
        except TypeError as error:
            raise TypeError("workdir must be a string or path-like object") from error
        directory.mkdir(parents=True, exist_ok=True)
        owned = False
    try:
        yield directory
    finally:
        if owned:
            shutil.rmtree(directory, ignore_errors=True)


class CadoExecutionError(RuntimeError):
    """Report a launch, timeout, or nonzero-exit failure from CADO-NFS."""

    def __init__(
        self,
        message: str,
        *,
        command: tuple[str, ...] = (),
        returncode: int | None = None,
        output: str = "",
        working_directory: Path | None = None,
        duration: float = 0.0,
        timed_out: bool = False,
    ) -> None:
        super().__init__(message)
        self.command = command
        self.returncode = returncode
        self.output = output
        self.working_directory = working_directory
        self.duration = duration
        self.timed_out = timed_out


class CadoOutputError(CadoExecutionError):
    """Report successful execution that produced invalid mathematical data."""


def _run_cado(
    arguments: Sequence[str],
    *,
    cado_path: str | os.PathLike[str],
    workdir: Path,
    timeout: float | None,
    verbose: bool,
) -> tuple[str, tuple[str, ...], float]:
    try:
        executable = _resolve_executable(cado_path)
    except (FileNotFoundError, PermissionError, TypeError) as error:
        raise CadoExecutionError(str(error), working_directory=workdir) from error
    identifier = uuid4().hex
    command = (str(executable), *map(str, arguments))
    try:
        result = run_process(
            command,
            working_directory=workdir,
            timeout=timeout,
            log_filename=f"cado-wrapper-{identifier}.log",
        )
    except ProcessExecutionError as error:
        raise CadoExecutionError(
            str(error),
            command=error.command,
            returncode=error.returncode,
            output=error.output,
            working_directory=error.working_directory,
            duration=error.duration,
            timed_out=error.timed_out,
        ) from error
    if verbose:
        print(result.output, end="" if result.output.endswith("\n") else "\n")
    return result.output, command, result.duration


def _parse_factor_output(value: int, output: str) -> list[int]:
    for line in reversed(output.splitlines()):
        if not _FACTOR_LINE_RE.fullmatch(line):
            continue
        factors = sorted(map(int, line.split()))
        if factors and prod(factors) == value:
            if not all(is_probable_prime(component) for component in factors):
                raise CadoOutputError(
                    "CADO-NFS returned a composite factor component",
                    output=output,
                )
            return factors
    raise CadoOutputError(
        "CADO-NFS output did not contain a valid complete factorization",
        output=output,
    )


def _python_factor(value: int) -> list[int] | None:
    if value.bit_length() > 40:
        return None
    from rewrite.factor import trial_division

    result = trial_division(value)
    if not result.complete:
        return None
    return list(result.factors)


def _sage_factor(value: int) -> list[int]:
    try:
        from sage.all import factor as sage_factor
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "SageMath is required for the selected factorization fallback"
        ) from error
    factors = [
        int(prime)
        for prime, exponent in sage_factor(value)
        for _ in range(int(exponent))
    ]
    if not factors or prod(factors) != value:
        raise RuntimeError("SageMath returned an invalid factorization")
    return sorted(factors)


def _local_factor(value: int, *, require_sage: bool) -> list[int]:
    if not require_sage:
        factors = _python_factor(value)
        if factors is not None:
            return factors
    return _sage_factor(value)


def factor(
    N: int,
    cado_path: str | os.PathLike[str] = DEFAULT_CADO_PATH,
    threads: int | None = None,
    verbose: bool = False,
    force_sage: bool = False,
    *,
    backend: str = "auto",
    timeout: float | None = None,
    workdir: str | os.PathLike[str] | None = None,
    cado_min_digits: int = 60,
    extra_args: Sequence[str] = (),
) -> list[int]:
    """Return a verified complete probable-prime factorization of ``N``.

    ``backend='auto'`` uses a local method below ``cado_min_digits`` and
    CADO-NFS above it.  An execution failure in auto mode falls back to Sage;
    malformed CADO output is never hidden by a fallback.
    """
    value = _positive_integer(N, "N")
    if value == 1:
        raise ValueError("N must be greater than one")
    selected_backend = _backend(backend)
    normalized_force_sage = _boolean(force_sage, "force_sage")
    normalized_verbose = _boolean(verbose, "verbose")
    if normalized_force_sage:
        if selected_backend != "auto":
            raise TypeError("force_sage cannot be combined with backend")
        selected_backend = "sage"
    minimum_digits = _positive_integer(cado_min_digits, "cado_min_digits")

    if selected_backend == "sage":
        return _local_factor(value, require_sage=True)
    if selected_backend == "auto" and len(str(value)) < minimum_digits:
        return _local_factor(value, require_sage=False)

    normalized_threads = _positive_integer(
        (os.cpu_count() or 1) if threads is None else threads,
        "threads",
    )
    normalized_timeout = _positive_timeout(timeout)
    normalized_args = tuple(str(argument) for argument in extra_args)
    with _workspace(workdir, prefix="cado-factor-") as directory:
        arguments = (
            str(value),
            f"workdir={directory}",
            "--screenlog",
            "warning",
            "-t",
            str(normalized_threads),
            *normalized_args,
        )
        try:
            output, command, duration = _run_cado(
                arguments,
                cado_path=cado_path,
                workdir=directory,
                timeout=normalized_timeout,
                verbose=normalized_verbose,
            )
        except CadoExecutionError:
            if selected_backend == "auto":
                return _local_factor(value, require_sage=True)
            raise
        try:
            return _parse_factor_output(value, output)
        except CadoOutputError as error:
            error.command = command
            error.working_directory = directory
            error.duration = duration
            raise


def _subgroup_problem(
    generator: int,
    target: int,
    modulus: int,
    order_factor: int,
) -> tuple[int, int]:
    if pow(generator, order_factor, modulus) == 1 and generator != 1:
        projected_generator = generator
        projected_target = target
    else:
        cofactor = (modulus - 1) // order_factor
        projected_generator = pow(generator, cofactor, modulus)
        projected_target = pow(target, cofactor, modulus)
    if projected_generator == 1:
        raise ValueError(
            "generator has no component of the requested subgroup order"
        )
    if pow(projected_generator, order_factor, modulus) != 1:
        raise ValueError("generator projection has an invalid order")
    if pow(projected_target, order_factor, modulus) != 1:
        raise ValueError("target is outside the requested subgroup")
    return projected_generator, projected_target


def _parse_dlp_output(output: str) -> tuple[int, int]:
    for line in reversed(output.splitlines()):
        if not _DLP_LINE_RE.fullmatch(line):
            continue
        values = tuple(int(value.strip()) for value in line.split(","))
        if len(values) == 2:
            return values
    raise CadoOutputError(
        "CADO-NFS output did not contain exactly two target logarithms",
        output=output,
    )


def _sage_dlp(generator: int, target: int, modulus: int, order: int) -> int:
    try:
        from sage.all import Zmod, discrete_log
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "SageMath is required for the selected discrete-log fallback"
        ) from error
    ring = Zmod(modulus)
    result = int(
        discrete_log(
            ring(target),
            ring(generator),
            ord=order,
            operation="*",
        )
    )
    if pow(generator, result, modulus) != target:
        raise RuntimeError("SageMath returned an invalid discrete logarithm")
    return result


def dlp(
    g: int,
    h: int,
    p: int,
    phi_factor: int,
    workdir: str | os.PathLike[str] | None = None,
    cado_path: str | os.PathLike[str] = DEFAULT_CADO_PATH,
    threads: int | None = None,
    verbose: bool = False,
    *,
    backend: str = "auto",
    timeout: float | None = None,
    local_dlp_limit: int = 1_000_000,
    cado_min_digits: int = 30,
    extra_args: Sequence[str] = (),
) -> int:
    """Recover the logarithm of ``h`` to base ``g`` modulo ``phi_factor``.

    If ``g`` already has order ``phi_factor``, the subgroup equation is solved
    directly.  Otherwise both values are projected from GF(p)* into that
    prime-order component.  Every returned result is verified there.
    """
    modulus = _positive_integer(p, "p")
    if not is_probable_prime(modulus):
        raise ValueError("p must be prime")
    order_factor = _positive_integer(phi_factor, "phi_factor")
    if not is_probable_prime(order_factor):
        raise ValueError("phi_factor must be prime")
    if (modulus - 1) % order_factor != 0:
        raise ValueError("phi_factor must divide p - 1")
    generator = _exact_integer(g, "g") % modulus
    target = _exact_integer(h, "h") % modulus
    if generator == 0 or target == 0:
        raise ValueError("g and h must be nonzero modulo p")
    projected_generator, projected_target = _subgroup_problem(
        generator,
        target,
        modulus,
        order_factor,
    )

    selected_backend = _backend(backend)
    normalized_verbose = _boolean(verbose, "verbose")
    local_limit = _positive_integer(local_dlp_limit, "local_dlp_limit")
    minimum_digits = _positive_integer(cado_min_digits, "cado_min_digits")
    if selected_backend == "sage":
        return _sage_dlp(
            projected_generator,
            projected_target,
            modulus,
            order_factor,
        )
    if selected_backend == "auto" and order_factor <= local_limit:
        return discrete_log_bsgs(
            projected_generator,
            projected_target,
            modulus,
            order_factor,
        )
    if selected_backend == "auto" and len(str(modulus)) < minimum_digits:
        return _sage_dlp(
            projected_generator,
            projected_target,
            modulus,
            order_factor,
        )

    normalized_threads = _positive_integer(
        (os.cpu_count() or 1) if threads is None else threads,
        "threads",
    )
    normalized_timeout = _positive_timeout(timeout)
    normalized_args = tuple(str(argument) for argument in extra_args)
    with _workspace(workdir, prefix="cado-dlp-") as directory:
        arguments = (
            "-dlp",
            "-ell",
            str(order_factor),
            f"target={generator},{target}",
            str(modulus),
            f"workdir={directory}",
            "--screenlog",
            "warning",
            "-t",
            str(normalized_threads),
            *normalized_args,
        )
        try:
            output, command, duration = _run_cado(
                arguments,
                cado_path=cado_path,
                workdir=directory,
                timeout=normalized_timeout,
                verbose=normalized_verbose,
            )
        except CadoExecutionError:
            if selected_backend == "auto":
                return _sage_dlp(
                    projected_generator,
                    projected_target,
                    modulus,
                    order_factor,
                )
            raise
        try:
            log_generator, log_target = _parse_dlp_output(output)
            try:
                logarithm = (
                    log_target * pow(log_generator, -1, order_factor)
                ) % order_factor
            except ValueError as error:
                raise CadoOutputError(
                    "CADO-NFS returned a non-invertible generator logarithm",
                    command=command,
                    output=output,
                    working_directory=directory,
                    duration=duration,
                ) from error
            if pow(projected_generator, logarithm, modulus) != projected_target:
                raise CadoOutputError(
                    "CADO-NFS logarithms fail subgroup verification",
                    command=command,
                    output=output,
                    working_directory=directory,
                    duration=duration,
                )
            return logarithm
        except CadoOutputError as error:
            if not error.command:
                error.command = command
                error.working_directory = directory
                error.duration = duration
            raise


def pohlig_hellman(
    g: int,
    h: int,
    p: int,
    phi_facs: Sequence[int],
    workdir: str | os.PathLike[str] | None = None,
    cado_path: str | os.PathLike[str] = DEFAULT_CADO_PATH,
    threads: int | None = None,
    verbose: bool = False,
    *,
    backend: str = "auto",
    timeout: float | None = None,
    local_dlp_limit: int = 1_000_000,
    cado_min_digits: int = 30,
) -> int:
    """Solve ``g**x = h (mod p)`` from the complete factored order of ``g``."""
    modulus = _positive_integer(p, "p")
    if not is_probable_prime(modulus):
        raise ValueError("p must be prime")
    generator = _exact_integer(g, "g") % modulus
    target = _exact_integer(h, "h") % modulus
    if generator == 0 or target == 0:
        raise ValueError("g and h must be nonzero modulo p")
    try:
        factors = tuple(
            _positive_integer(factor_value, f"phi_facs[{position}]")
            for position, factor_value in enumerate(phi_facs)
        )
    except TypeError as error:
        if "not iterable" in str(error):
            raise TypeError("phi_facs must be an iterable of prime factors") from error
        raise
    if not factors:
        raise ValueError("phi_facs must not be empty")
    if not all(is_probable_prime(factor_value) for factor_value in factors):
        raise ValueError("phi_facs must contain prime factors")

    order = prod(factors)
    if (modulus - 1) % order != 0:
        raise ValueError("the supplied order must divide p - 1")
    if pow(generator, order, modulus) != 1:
        raise ValueError("the supplied factors are not a multiple of g's order")
    multiplicities = Counter(factors)
    for prime in multiplicities:
        if pow(generator, order // prime, modulus) == 1:
            raise ValueError("phi_facs do not describe the exact order of g")
    if pow(target, order, modulus) != 1:
        raise ValueError("h is outside the subgroup generated by g")

    normalized_verbose = _boolean(verbose, "verbose")
    selected_backend = _backend(backend)
    normalized_timeout = _positive_timeout(timeout)
    residues: list[int] = []
    moduli: list[int] = []
    with _workspace(workdir, prefix="cado-pohlig-") as base_directory:
        for prime, exponent in sorted(multiplicities.items()):
            digit_base = pow(generator, order // prime, modulus)
            residue = 0
            prime_power = 1
            prime_workdir = base_directory / f"p{modulus}-ell{prime}"
            for digit_index in range(exponent):
                corrected_target = (
                    target * pow(pow(generator, residue, modulus), -1, modulus)
                ) % modulus
                digit_target = pow(
                    corrected_target,
                    order // (prime ** (digit_index + 1)),
                    modulus,
                )
                digit = dlp(
                    digit_base,
                    digit_target,
                    modulus,
                    prime,
                    workdir=prime_workdir,
                    cado_path=cado_path,
                    threads=threads,
                    verbose=normalized_verbose,
                    backend=selected_backend,
                    timeout=normalized_timeout,
                    local_dlp_limit=local_dlp_limit,
                    cado_min_digits=cado_min_digits,
                )
                if not 0 <= digit < prime:
                    raise RuntimeError("digit discrete log is outside its range")
                residue += digit * prime_power
                prime_power *= prime
            residues.append(residue)
            moduli.append(prime_power)

    logarithm = crt_coprime(residues, moduli)
    if pow(generator, logarithm, modulus) != target:
        raise ValueError("h has no logarithm to base g in the supplied subgroup")
    return logarithm
