"""Verified subprocess adapter for the external YAFU factorization tool."""

from __future__ import annotations

from collections.abc import Sequence
from contextlib import contextmanager
from dataclasses import dataclass
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
from rewrite._number_theory import is_probable_prime


__all__ = [
    "IncompleteFactorizationError",
    "YafuExecutionError",
    "YafuOutputError",
    "YafuResult",
    "factor",
    "factor_cmd",
    "rho",
    "run_yafu",
]


DEFAULT_YAFU_PATH = "/opt/yafu/yafu"
_EXPRESSION_RE = re.compile(r"^\s*(factor|rho)\s*\(\s*(\d+)\s*\)\s*$", re.I)
_LABELED_FACTOR_RE = re.compile(
    r"^\s*(PRP|P|C)\s*\d*\s*=\s*(\d+)\s*$",
    re.I | re.M,
)
_LIVE_FACTOR_RE = re.compile(r"\bfactor\s*=\s*(\d+)\b", re.I)


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


def _resolve_executable(value: str | os.PathLike[str]) -> Path:
    try:
        text = os.fspath(value)
    except TypeError as error:
        raise TypeError("yafu_path must be a string or path-like object") from error
    found = shutil.which(text)
    candidate = Path(found if found is not None else text).expanduser()
    try:
        candidate = candidate.resolve()
    except OSError as error:
        raise FileNotFoundError(f"cannot resolve YAFU executable: {text}") from error
    if not candidate.is_file():
        raise FileNotFoundError(f"YAFU executable does not exist: {candidate}")
    if not os.access(candidate, os.X_OK):
        raise PermissionError(f"YAFU executable is not executable: {candidate}")
    return candidate


@dataclass(frozen=True)
class YafuResult:
    """A multiplication-verified decomposition reported by YAFU."""

    input_value: int
    components: tuple[int, ...]
    method: str
    command: tuple[str, ...]
    duration: float
    output: str
    working_directory: Path

    def __post_init__(self) -> None:
        input_value = _positive_integer(self.input_value, "input_value")
        if input_value == 1:
            raise ValueError("input_value must be greater than one")
        normalized_components = tuple(
            _positive_integer(component, f"components[{position}]")
            for position, component in enumerate(self.components)
        )
        if any(component == 1 for component in normalized_components):
            raise ValueError("factorization components must be greater than one")
        if not normalized_components or prod(normalized_components) != input_value:
            raise ValueError("components do not multiply to input_value")
        object.__setattr__(self, "input_value", input_value)
        object.__setattr__(self, "components", tuple(sorted(normalized_components)))

    @property
    def complete(self) -> bool:
        """Return whether every reported component is probably prime."""
        return all(is_probable_prime(component) for component in self.components)

    @property
    def unresolved(self) -> tuple[int, ...]:
        """Return components that are still composite."""
        return tuple(
            component
            for component in self.components
            if not is_probable_prime(component)
        )

    @property
    def factors(self) -> tuple[int, ...]:
        """Compatibility-oriented name for all decomposition components."""
        return self.components


class YafuExecutionError(RuntimeError):
    """Report a launch, timeout, or nonzero-exit failure from YAFU."""

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


class YafuOutputError(YafuExecutionError):
    """Report successful execution that produced invalid factor data."""


class IncompleteFactorizationError(YafuOutputError):
    """Report a verified decomposition that still contains composites."""

    def __init__(self, result: YafuResult) -> None:
        super().__init__(
            f"YAFU left composite components: {result.unresolved}",
            command=result.command,
            output=result.output,
            working_directory=result.working_directory,
            duration=result.duration,
        )
        self.result = result


@contextmanager
def _workspace(
    workdir: str | os.PathLike[str] | None,
    cleanup: bool,
) -> Iterator[Path]:
    if workdir is None:
        directory = Path(mkdtemp(prefix="yafu-wrapper-"))
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
        if owned and cleanup:
            shutil.rmtree(directory, ignore_errors=True)


def _parse_components(input_value: int, output: str) -> tuple[int, ...]:
    segment = output.rsplit("***factors found***", 1)[-1]
    labeled = _LABELED_FACTOR_RE.findall(segment)
    if labeled:
        components = tuple(int(value) for _, value in labeled)
        if components and prod(components) == input_value:
            return tuple(sorted(components))
        raise YafuOutputError(
            "YAFU's labeled components do not multiply to the input",
            output=output,
        )

    remainder = input_value
    components: list[int] = []
    for text in _LIVE_FACTOR_RE.findall(output):
        factor_value = int(text)
        if 1 < factor_value < remainder and remainder % factor_value == 0:
            components.append(factor_value)
            remainder //= factor_value
    if not components:
        raise YafuOutputError(
            "YAFU output did not contain a factor decomposition",
            output=output,
        )
    if remainder > 1:
        components.append(remainder)
    if not components or prod(components) != input_value:
        raise YafuOutputError("could not parse a valid YAFU decomposition", output=output)
    return tuple(sorted(components))


def _select_path(
    legacy_path: str | os.PathLike[str],
    modern_path: str | os.PathLike[str] | None,
) -> str | os.PathLike[str]:
    if modern_path is not None and os.fspath(legacy_path) != DEFAULT_YAFU_PATH:
        raise TypeError("specify yafuPath or yafu_path, not both")
    return legacy_path if modern_path is None else modern_path


def _execute_yafu(
    input_value: int,
    expression: str,
    method: str,
    *,
    threads: int,
    timeout: float | None,
    yafu_path: str | os.PathLike[str],
    cleanup: bool,
    verbose: bool,
    workdir: str | os.PathLike[str] | None,
    extra_args: Sequence[str],
) -> YafuResult:
    executable = _resolve_executable(yafu_path)
    normalized_threads = _positive_integer(threads, "threads")
    normalized_timeout = _positive_timeout(timeout)
    normalized_cleanup = _boolean(cleanup, "cleanup")
    normalized_verbose = _boolean(verbose, "verbose")
    normalized_args = tuple(str(argument) for argument in extra_args)

    with _workspace(workdir, normalized_cleanup) as directory:
        identifier = uuid4().hex
        command = (
            str(executable),
            expression,
            "-threads",
            str(normalized_threads),
            "-logfile",
            str(directory / f"factor-{identifier}.log"),
            "-session",
            str(directory / f"session-{identifier}.log"),
            *normalized_args,
        )
        try:
            process_result = run_process(
                command,
                working_directory=directory,
                timeout=normalized_timeout,
                log_filename=f"wrapper-{identifier}.log",
            )
        except ProcessExecutionError as error:
            raise YafuExecutionError(
                str(error),
                command=error.command,
                returncode=error.returncode,
                output=error.output,
                working_directory=error.working_directory,
                duration=error.duration,
                timed_out=error.timed_out,
            ) from error
        if normalized_verbose:
            print(process_result.output, end="" if process_result.output.endswith("\n") else "\n")
        try:
            components = _parse_components(input_value, process_result.output)
        except YafuOutputError as error:
            error.command = command
            error.working_directory = directory
            error.duration = process_result.duration
            raise
        return YafuResult(
            input_value=input_value,
            components=components,
            method=method,
            command=command,
            duration=process_result.duration,
            output=process_result.output,
            working_directory=directory,
        )


def run_yafu(
    composite: int,
    *,
    method: str = "factor",
    threads: int = 1,
    timeout: float | None = None,
    yafu_path: str | os.PathLike[str] = DEFAULT_YAFU_PATH,
    cleanup: bool = True,
    verbose: bool = False,
    workdir: str | os.PathLike[str] | None = None,
    extra_args: Sequence[str] = (),
) -> YafuResult:
    """Run ``factor`` or ``rho`` and return a verified decomposition."""
    input_value = _positive_integer(composite, "composite")
    if input_value == 1:
        raise ValueError("composite must be greater than one")
    if not isinstance(method, str):
        raise TypeError("method must be a string")
    normalized_method = method.lower()
    if normalized_method not in {"factor", "rho"}:
        raise ValueError("method must be 'factor' or 'rho'")
    return _execute_yafu(
        input_value,
        f"{normalized_method}({input_value})",
        normalized_method,
        threads=threads,
        timeout=timeout,
        yafu_path=yafu_path,
        cleanup=cleanup,
        verbose=verbose,
        workdir=workdir,
        extra_args=extra_args,
    )


def factor(
    composite: int,
    threads: int = 1,
    timeout: float | None = None,
    yafuPath: str | os.PathLike[str] = DEFAULT_YAFU_PATH,
    cleanup: bool = True,
    verbose: bool = False,
    *,
    yafu_path: str | os.PathLike[str] | None = None,
    workdir: str | os.PathLike[str] | None = None,
) -> list[int]:
    """Return a verified complete probable-prime factorization from YAFU."""
    result = run_yafu(
        composite,
        method="factor",
        threads=threads,
        timeout=timeout,
        yafu_path=_select_path(yafuPath, yafu_path),
        cleanup=cleanup,
        verbose=verbose,
        workdir=workdir,
    )
    if not result.complete:
        raise IncompleteFactorizationError(result)
    return list(result.components)


def rho(
    composite: int,
    threads: int = 1,
    timeout: float | None = None,
    yafuPath: str | os.PathLike[str] = DEFAULT_YAFU_PATH,
    cleanup: bool = True,
    verbose: bool = False,
    *,
    yafu_path: str | os.PathLike[str] | None = None,
    workdir: str | os.PathLike[str] | None = None,
) -> list[int]:
    """Return a multiplication-verified decomposition from one rho run."""
    result = run_yafu(
        composite,
        method="rho",
        threads=threads,
        timeout=timeout,
        yafu_path=_select_path(yafuPath, yafu_path),
        cleanup=cleanup,
        verbose=verbose,
        workdir=workdir,
    )
    return list(result.components)


def factor_cmd(
    command: str | bytes,
    threads: int = 1,
    timeout: float | None = None,
    yafuPath: str | os.PathLike[str] = DEFAULT_YAFU_PATH,
    cleanup: bool = True,
    verbose: bool = False,
    *,
    composite: int | None = None,
    yafu_path: str | os.PathLike[str] | None = None,
    workdir: str | os.PathLike[str] | None = None,
) -> list[int]:
    """Compatibility wrapper for a YAFU expression with a known input value."""
    if isinstance(command, bytes):
        try:
            expression = command.decode("ascii")
        except UnicodeDecodeError as error:
            raise ValueError("command bytes must contain ASCII YAFU syntax") from error
    elif isinstance(command, str):
        expression = command
    else:
        raise TypeError("command must be str or bytes")

    match = _EXPRESSION_RE.fullmatch(expression)
    if match is not None:
        method = match.group(1).lower()
        parsed_input = int(match.group(2))
        if composite is not None and _positive_integer(composite, "composite") != parsed_input:
            raise ValueError("composite disagrees with the command input")
        input_value = parsed_input
    else:
        if composite is None:
            raise ValueError(
                "composite is required when command is not factor(N) or rho(N)"
            )
        input_value = _positive_integer(composite, "composite")
        method = "command"

    result = _execute_yafu(
        input_value,
        expression,
        method,
        threads=threads,
        timeout=timeout,
        yafu_path=_select_path(yafuPath, yafu_path),
        cleanup=cleanup,
        verbose=verbose,
        workdir=workdir,
        extra_args=(),
    )
    return list(result.components)
