"""Safe process adapter for the external HashClash MD5 collision tools.

HashClash performs the expensive cryptanalysis.  This module gives its shell
scripts isolated input/output directories, checks process failures, and
cryptographically verifies every result before returning it.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import os
from operator import index as integer_index
from pathlib import Path
import signal
import subprocess
from tempfile import TemporaryDirectory
from typing import Iterator


__all__ = [
    "CollisionGenerationError",
    "CollisionPair",
    "HashClash",
    "Hashclash",
    "is_md5_collision",
]


_LOG_TAIL_BYTES = 16_384


def _as_bytes(value: object, name: str) -> bytes:
    if isinstance(value, bytes):
        return value
    if isinstance(value, (bytearray, memoryview)):
        return bytes(value)
    raise TypeError(f"{name} must be bytes-like")


def _md5_digest(message: bytes) -> bytes:
    try:
        return hashlib.md5(message, usedforsecurity=False).digest()
    except TypeError:  # Compatibility with older Python builds.
        return hashlib.md5(message).digest()


def is_md5_collision(
    message1: bytes | bytearray | memoryview,
    message2: bytes | bytearray | memoryview,
) -> bool:
    """Return whether two distinct byte strings have the same MD5 digest."""
    normalized_first = _as_bytes(message1, "message1")
    normalized_second = _as_bytes(message2, "message2")
    return (
        normalized_first != normalized_second
        and _md5_digest(normalized_first) == _md5_digest(normalized_second)
    )


@dataclass(frozen=True)
class CollisionPair:
    """A verified pair of complete colliding messages and their prefixes."""

    message1: bytes
    message2: bytes
    prefix1: bytes = b""
    prefix2: bytes = b""

    def __post_init__(self) -> None:
        normalized_message1 = _as_bytes(self.message1, "message1")
        normalized_message2 = _as_bytes(self.message2, "message2")
        normalized_prefix1 = _as_bytes(self.prefix1, "prefix1")
        normalized_prefix2 = _as_bytes(self.prefix2, "prefix2")
        object.__setattr__(self, "message1", normalized_message1)
        object.__setattr__(self, "message2", normalized_message2)
        object.__setattr__(self, "prefix1", normalized_prefix1)
        object.__setattr__(self, "prefix2", normalized_prefix2)

        if not normalized_message1.startswith(normalized_prefix1):
            raise ValueError("message1 does not preserve prefix1")
        if not normalized_message2.startswith(normalized_prefix2):
            raise ValueError("message2 does not preserve prefix2")
        if not is_md5_collision(normalized_message1, normalized_message2):
            raise ValueError(
                "messages must be distinct and have the same MD5 digest"
            )

    @property
    def digest(self) -> bytes:
        """Return the common raw MD5 digest."""
        return _md5_digest(self.message1)

    @property
    def hexdigest(self) -> str:
        """Return the common MD5 digest in hexadecimal form."""
        return self.digest.hex()

    @property
    def suffix1(self) -> bytes:
        """Return the first message after its requested prefix."""
        return self.message1[len(self.prefix1) :]

    @property
    def suffix2(self) -> bytes:
        """Return the second message after its requested prefix."""
        return self.message2[len(self.prefix2) :]

    def __iter__(self) -> Iterator[bytes]:
        """Preserve tuple unpacking used by the original wrapper."""
        yield self.message1
        yield self.message2


class CollisionGenerationError(RuntimeError):
    """Report a failed, incomplete, or invalid HashClash invocation."""

    def __init__(
        self,
        message: str,
        *,
        command: tuple[str, ...] = (),
        returncode: int | None = None,
        output: str = "",
    ) -> None:
        super().__init__(message)
        self.command = command
        self.returncode = returncode
        self.output = output


def _positive_integer(value: object, name: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer, not bool")
    try:
        normalized = integer_index(value)
    except TypeError as error:
        raise TypeError(f"{name} must be an integer") from error
    if normalized <= 0:
        raise ValueError(f"{name} must be positive")
    return normalized


def _positive_timeout(value: object | None) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise TypeError("timeout must be a real number, not bool")
    try:
        normalized = float(value)
    except (TypeError, ValueError) as error:
        raise TypeError("timeout must be a real number or None") from error
    if not math.isfinite(normalized) or normalized <= 0:
        raise ValueError("timeout must be finite and positive")
    return normalized


def _log_tail(log_path: Path) -> str:
    try:
        with log_path.open("rb") as log_file:
            log_file.seek(0, os.SEEK_END)
            size = log_file.tell()
            log_file.seek(max(0, size - _LOG_TAIL_BYTES))
            return log_file.read().decode("utf-8", errors="replace")
    except OSError:
        return ""


def _terminate_process_group(process: subprocess.Popen[bytes]) -> None:
    """Terminate a Bash script and the worker processes it started."""
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=1.0)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        return
    process.wait()


class HashClash:
    """Run an installed HashClash checkout and verify its MD5 collisions.

    ``timeout`` limits each complete shell-script invocation. ``max_cpus`` is
    exported as ``MAXCPUS``; the upstream chosen-prefix script recognizes it.
    """

    def __init__(
        self,
        path: str | os.PathLike[str],
        *,
        timeout: float | None = None,
        max_cpus: int | None = None,
    ) -> None:
        try:
            normalized_path = Path(path).expanduser().resolve()
        except TypeError as error:
            raise TypeError("path must be a string or path-like object") from error
        scripts_path = normalized_path / "scripts"
        if not normalized_path.is_dir():
            raise FileNotFoundError(
                f"HashClash directory does not exist: {normalized_path}"
            )
        if not scripts_path.is_dir():
            raise FileNotFoundError(
                f"HashClash scripts directory does not exist: {scripts_path}"
            )

        self._hashclash_path = normalized_path
        self._scripts_path = scripts_path
        self._timeout = _positive_timeout(timeout)
        self._max_cpus = (
            None if max_cpus is None else _positive_integer(max_cpus, "max_cpus")
        )

    @property
    def hashclash_path(self) -> Path:
        """Return the validated HashClash installation directory."""
        return self._hashclash_path

    @property
    def scripts_path(self) -> Path:
        """Return the validated upstream scripts directory."""
        return self._scripts_path

    @property
    def timeout(self) -> float | None:
        """Return the per-invocation timeout in seconds, if configured."""
        return self._timeout

    @property
    def max_cpus(self) -> int | None:
        """Return the configured upstream CPU limit, if any."""
        return self._max_cpus

    def _script(self, filename: str) -> Path:
        script = self.scripts_path / filename
        if not script.is_file():
            raise FileNotFoundError(f"HashClash script does not exist: {script}")
        return script

    def _run_script(
        self,
        script: Path,
        arguments: tuple[str, ...],
        working_directory: Path,
    ) -> None:
        command = ("bash", str(script), *arguments)
        environment = os.environ.copy()
        if self.max_cpus is not None:
            environment["MAXCPUS"] = str(self.max_cpus)

        log_path = working_directory / "hashclash-wrapper.log"
        try:
            with log_path.open("wb") as log_file:
                process = subprocess.Popen(
                    command,
                    cwd=working_directory,
                    env=environment,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                try:
                    process.wait(timeout=self.timeout)
                except subprocess.TimeoutExpired as error:
                    _terminate_process_group(process)
                    raise CollisionGenerationError(
                        f"HashClash timed out after {self.timeout:g} seconds",
                        command=command,
                        returncode=process.returncode,
                        output=_log_tail(log_path),
                    ) from error
        except CollisionGenerationError:
            raise
        except OSError as error:
            raise CollisionGenerationError(
                f"could not execute HashClash: {error}",
                command=command,
                output=_log_tail(log_path),
            ) from error

        if process.returncode != 0:
            raise CollisionGenerationError(
                f"HashClash exited with status {process.returncode}",
                command=command,
                returncode=process.returncode,
                output=_log_tail(log_path),
            )

    @staticmethod
    def _read_output(working_directory: Path, filename: str) -> bytes:
        output_path = working_directory / filename
        if not output_path.is_file():
            raise CollisionGenerationError(
                f"HashClash did not create expected output: {filename}"
            )
        try:
            return output_path.read_bytes()
        except OSError as error:
            raise CollisionGenerationError(
                f"could not read HashClash output {filename}: {error}"
            ) from error

    def _generate(
        self,
        *,
        kind: str,
        script_name: str,
        inputs: tuple[tuple[str, bytes], ...],
        arguments: tuple[str, ...],
        output_names: tuple[str, str],
        required_prefixes: tuple[bytes, bytes],
        reported_prefixes: tuple[bytes, bytes],
    ) -> CollisionPair:
        script = self._script(script_name)
        with TemporaryDirectory(prefix=f"hashclash-{kind}-") as temporary:
            working_directory = Path(temporary)
            for filename, contents in inputs:
                (working_directory / filename).write_bytes(contents)

            self._run_script(script, arguments, working_directory)
            message1 = self._read_output(working_directory, output_names[0])
            message2 = self._read_output(working_directory, output_names[1])

        if not message1.startswith(required_prefixes[0]):
            raise CollisionGenerationError(
                "first HashClash output does not preserve its input prefix"
            )
        if not message2.startswith(required_prefixes[1]):
            raise CollisionGenerationError(
                "second HashClash output does not preserve its input prefix"
            )
        try:
            return CollisionPair(
                message1,
                message2,
                reported_prefixes[0],
                reported_prefixes[1],
            )
        except ValueError as error:
            raise CollisionGenerationError(
                f"HashClash produced an invalid collision: {error}"
            ) from error

    def chosen_prefix_collision(
        self,
        prefix1: bytes | bytearray | memoryview,
        prefix2: bytes | bytearray | memoryview,
    ) -> CollisionPair:
        """Generate complete colliding messages for two chosen prefixes."""
        normalized_prefix1 = _as_bytes(prefix1, "prefix1")
        normalized_prefix2 = _as_bytes(prefix2, "prefix2")
        filenames = ("prefix1.bin", "prefix2.bin")
        return self._generate(
            kind="chosen-prefix",
            script_name="cpc.sh",
            inputs=(
                (filenames[0], normalized_prefix1),
                (filenames[1], normalized_prefix2),
            ),
            arguments=filenames,
            output_names=(f"{filenames[0]}.coll", f"{filenames[1]}.coll"),
            required_prefixes=(normalized_prefix1, normalized_prefix2),
            reported_prefixes=(normalized_prefix1, normalized_prefix2),
        )

    def identical_prefix_collision(
        self,
        prefix: bytes | bytearray | memoryview,
        *,
        pad_byte: bytes | bytearray | memoryview = b" ",
    ) -> CollisionPair:
        """Generate complete colliding messages sharing one prefix.

        HashClash's generic identical-prefix script consumes full 32-bit words
        after complete MD5 blocks.  A one-byte ``pad_byte`` therefore extends
        the supplied prefix to a multiple of four bytes.  These explicit pad
        bytes are included at the beginning of both returned suffixes.
        """
        normalized_prefix = _as_bytes(prefix, "prefix")
        normalized_pad = _as_bytes(pad_byte, "pad_byte")
        if len(normalized_pad) != 1:
            raise ValueError("pad_byte must contain exactly one byte")
        padding_length = -len(normalized_prefix) % 4
        effective_prefix = normalized_prefix + normalized_pad * padding_length
        filename = "prefix.bin"
        return self._generate(
            kind="identical-prefix",
            script_name="generic_ipc.sh",
            inputs=((filename, effective_prefix),),
            arguments=(filename,),
            output_names=("collision1.bin", "collision2.bin"),
            required_prefixes=(effective_prefix, effective_prefix),
            reported_prefixes=(normalized_prefix, normalized_prefix),
        )

    def text_collision(
        self,
        prefix: bytes | bytearray | memoryview = b"",
    ) -> CollisionPair:
        """Generate the textual collision configured by ``textcoll.sh``."""
        normalized_prefix = _as_bytes(prefix, "prefix")
        filename = "prefix.bin"
        return self._generate(
            kind="text",
            script_name="textcoll.sh",
            inputs=((filename, normalized_prefix),),
            arguments=(filename,),
            output_names=(
                "final_collision1.txt",
                "final_collision2.txt",
            ),
            required_prefixes=(normalized_prefix, normalized_prefix),
            reported_prefixes=(normalized_prefix, normalized_prefix),
        )


# Preserve the original project's class spelling.
Hashclash = HashClash
