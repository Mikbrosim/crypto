"""Internal subprocess helpers shared by external cryptanalytic adapters."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import signal
import subprocess
import time


_LOG_TAIL_BYTES = 4 * 1024 * 1024


@dataclass(frozen=True)
class ProcessResult:
    """Captured outcome of one successful external command."""

    command: tuple[str, ...]
    working_directory: Path
    returncode: int
    output: str
    duration: float


class ProcessExecutionError(RuntimeError):
    """Internal description of a launch, timeout, or exit failure."""

    def __init__(
        self,
        message: str,
        *,
        command: tuple[str, ...],
        working_directory: Path,
        returncode: int | None = None,
        output: str = "",
        duration: float = 0.0,
        timed_out: bool = False,
    ) -> None:
        super().__init__(message)
        self.command = command
        self.working_directory = working_directory
        self.returncode = returncode
        self.output = output
        self.duration = duration
        self.timed_out = timed_out


def _read_log_tail(path: Path) -> str:
    try:
        with path.open("rb") as log_file:
            log_file.seek(0, os.SEEK_END)
            size = log_file.tell()
            log_file.seek(max(0, size - _LOG_TAIL_BYTES))
            return log_file.read().decode("utf-8", errors="replace")
    except OSError:
        return ""


def _terminate_process_tree(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    if os.name == "posix":
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
    else:
        process.terminate()
    try:
        process.wait(timeout=1.0)
        return
    except subprocess.TimeoutExpired:
        pass

    if os.name == "posix":
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            return
    else:
        process.kill()
    process.wait()


def run_process(
    command: tuple[str, ...],
    *,
    working_directory: Path,
    timeout: float | None,
    log_filename: str,
) -> ProcessResult:
    """Run a command without a shell and capture a bounded output tail."""
    log_path = working_directory / log_filename
    started = time.monotonic()
    try:
        with log_path.open("wb") as log_file:
            process = subprocess.Popen(
                command,
                cwd=working_directory,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                start_new_session=os.name == "posix",
            )
            try:
                process.wait(timeout=timeout)
            except subprocess.TimeoutExpired as error:
                _terminate_process_tree(process)
                duration = time.monotonic() - started
                raise ProcessExecutionError(
                    f"external command timed out after {timeout:g} seconds",
                    command=command,
                    working_directory=working_directory,
                    returncode=process.returncode,
                    output=_read_log_tail(log_path),
                    duration=duration,
                    timed_out=True,
                ) from error
    except ProcessExecutionError:
        raise
    except OSError as error:
        duration = time.monotonic() - started
        raise ProcessExecutionError(
            f"could not execute external command: {error}",
            command=command,
            working_directory=working_directory,
            output=_read_log_tail(log_path),
            duration=duration,
        ) from error

    duration = time.monotonic() - started
    output = _read_log_tail(log_path)
    if process.returncode != 0:
        raise ProcessExecutionError(
            f"external command exited with status {process.returncode}",
            command=command,
            working_directory=working_directory,
            returncode=process.returncode,
            output=output,
            duration=duration,
        )
    return ProcessResult(
        command=command,
        working_directory=working_directory,
        returncode=process.returncode,
        output=output,
        duration=duration,
    )
