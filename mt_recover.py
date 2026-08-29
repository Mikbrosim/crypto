"""Recover CPython's MT19937 state from observed ``getrandbits`` output.

MT19937 is a linear, non-cryptographic generator.  Its tempering transform is
invertible, so 624 consecutive complete 32-bit outputs are enough to clone the
generator and predict subsequent output.  This module also handles CPython's
packing of variable-width ``getrandbits`` results and can exhaustively fill a
small number of missing low bits.

The returned :class:`random.Random` is always positioned *after* every supplied
observation.  Dense truncated-output recovery is intentionally out of scope;
that problem is better represented as a large linear system over GF(2).
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from operator import index as integer_index
import random


__all__ = [
    "AmbiguousStateError",
    "CandidateLimitError",
    "MT19937Recover",
    "ObservationMismatchError",
    "PartialWord32",
    "Recover",
    "RecoveryError",
    "clone_from_outputs",
    "temper",
    "untemper",
]


WORD_BITS = 32
STATE_WORDS = 624
WORD_MASK = (1 << WORD_BITS) - 1
_TEMPER_B = 0x9D2C5680
_TEMPER_C = 0xEFC60000


class RecoveryError(ValueError):
    """Base class for failures caused by inconsistent recovery data."""


class ObservationMismatchError(RecoveryError):
    """Raised when an output after the recovered state does not match."""

    def __init__(self, index: int, observed: int, predicted: int) -> None:
        super().__init__(
            f"output {index} is inconsistent: observed 0x{observed:08x}, "
            f"predicted 0x{predicted:08x}"
        )
        self.index = index
        self.observed = observed
        self.predicted = predicted


class CandidateLimitError(RecoveryError):
    """Raised when partial observations require too much exhaustive search."""

    def __init__(self, unknown_bits: int, limit: int) -> None:
        candidates = 1 << unknown_bits
        super().__init__(
            f"recovery requires {unknown_bits} unknown bits "
            f"({candidates} candidates), exceeding the limit of {limit} bits"
        )
        self.unknown_bits = unknown_bits
        self.candidates = candidates
        self.limit = limit


class AmbiguousStateError(RecoveryError):
    """Raised when the observations admit different future output streams."""


def _exact_integer(value: object, name: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer, not bool")
    try:
        return integer_index(value)
    except TypeError as error:
        raise TypeError(f"{name} must be an integer") from error


def _nonnegative_integer(value: object, name: str) -> int:
    normalized = _exact_integer(value, name)
    if normalized < 0:
        raise ValueError(f"{name} must be non-negative")
    return normalized


def _word(value: object, name: str) -> int:
    normalized = _nonnegative_integer(value, name)
    if normalized > WORD_MASK:
        raise ValueError(f"{name} must fit in 32 bits")
    return normalized


def _shift(value: object) -> int:
    normalized = _exact_integer(value, "shift")
    if not 1 <= normalized < WORD_BITS:
        raise ValueError("shift must be between 1 and 31")
    return normalized


def _undo_right_shift_xor(value: int, shift: int) -> int:
    """Invert ``x ^ (x >> shift)`` on an unsigned 32-bit word."""
    original = _word(value, "value")
    distance = _shift(shift)
    result = original
    for _ in range((WORD_BITS + distance - 1) // distance):
        result = original ^ (result >> distance)
    return result & WORD_MASK


def _undo_left_shift_xor_mask(value: int, shift: int, mask: int) -> int:
    """Invert ``x ^ ((x << shift) & mask)`` on a 32-bit word."""
    original = _word(value, "value")
    distance = _shift(shift)
    normalized_mask = _word(mask, "mask")
    result = original
    for _ in range((WORD_BITS + distance - 1) // distance):
        result = original ^ ((result << distance) & normalized_mask)
    return result & WORD_MASK


def temper(value: int) -> int:
    """Apply MT19937's output tempering transform to one state word."""
    result = _word(value, "value")
    result ^= result >> 11
    result ^= (result << 7) & _TEMPER_B
    result ^= (result << 15) & _TEMPER_C
    result ^= result >> 18
    return result & WORD_MASK


def untemper(value: int) -> int:
    """Invert MT19937's tempering transform for one complete output word."""
    result = _word(value, "value")
    result = _undo_right_shift_xor(result, 18)
    result = _undo_left_shift_xor_mask(result, 15, _TEMPER_C)
    result = _undo_left_shift_xor_mask(result, 7, _TEMPER_B)
    result = _undo_right_shift_xor(result, 11)
    return result


@dataclass(frozen=True)
class PartialWord32:
    """The known high bits of a 32-bit output and its unknown low-bit count.

    For example, ``PartialWord32(0b101, 29)`` represents every 32-bit word
    whose three most significant bits are ``101``.
    """

    known_msb: int
    unknown_lsb_bits: int

    def __post_init__(self) -> None:
        known = _nonnegative_integer(self.known_msb, "known_msb")
        unknown = _nonnegative_integer(
            self.unknown_lsb_bits,
            "unknown_lsb_bits",
        )
        if unknown >= WORD_BITS:
            raise ValueError("unknown_lsb_bits must be between 0 and 31")
        known_bits = WORD_BITS - unknown
        if known >= 1 << known_bits:
            raise ValueError(
                f"known_msb must fit in the remaining {known_bits} bits"
            )
        object.__setattr__(self, "known_msb", known)
        object.__setattr__(self, "unknown_lsb_bits", unknown)

    @property
    def v(self) -> int:
        """Compatibility name for :attr:`known_msb`."""
        return self.known_msb

    @property
    def n(self) -> int:
        """Compatibility name for :attr:`unknown_lsb_bits`."""
        return self.unknown_lsb_bits

    @property
    def values(self) -> Iterator[int]:
        """Iterate over every complete word represented by this observation."""
        prefix = self.known_msb << self.unknown_lsb_bits
        for suffix in range(1 << self.unknown_lsb_bits):
            yield prefix | suffix

    def matches(self, value: int) -> bool:
        """Return whether a complete output word matches the known prefix."""
        normalized = _word(value, "value")
        return normalized >> self.unknown_lsb_bits == self.known_msb

    def __repr__(self) -> str:
        known_bits = WORD_BITS - self.unknown_lsb_bits
        prefix = format(self.known_msb, f"0{known_bits}b")
        return f"0b{prefix}{'x' * self.unknown_lsb_bits}"


Observation = int | PartialWord32


def _random_from_words(words: Sequence[int]) -> random.Random:
    if len(words) != STATE_WORDS:
        raise ValueError(f"state must contain exactly {STATE_WORDS} words")
    clone = random.Random()
    clone.setstate((3, tuple(words) + (STATE_WORDS,), None))
    return clone


def clone_from_outputs(outputs: Iterable[int]) -> random.Random:
    """Clone CPython's MT19937 from at least 624 complete 32-bit outputs.

    All outputs must be consecutive results of ``getrandbits(32)``.  Any
    observations after the first 624 are verified and consumed, so the
    returned generator predicts the output immediately following the final
    supplied observation.
    """
    normalized = tuple(
        _word(value, f"outputs[{position}]")
        for position, value in enumerate(outputs)
    )
    if len(normalized) < STATE_WORDS:
        raise ValueError(
            f"at least {STATE_WORDS} complete outputs are required; "
            f"received {len(normalized)}"
        )

    clone = _random_from_words(tuple(map(untemper, normalized[:STATE_WORDS])))
    for position, observed in enumerate(
        normalized[STATE_WORDS:],
        start=STATE_WORDS,
    ):
        predicted = clone.getrandbits(WORD_BITS)
        if predicted != observed:
            raise ObservationMismatchError(position, observed, predicted)
    return clone


class MT19937Recover:
    """Compatibility-oriented interface for complete-output recovery."""

    @staticmethod
    def unshift_right(value: int, shift: int) -> int:
        """Invert a right-shift XOR on one 32-bit word."""
        return _undo_right_shift_xor(value, shift)

    @staticmethod
    def unshift_left(value: int, shift: int, mask: int) -> int:
        """Invert a masked left-shift XOR on one 32-bit word."""
        return _undo_left_shift_xor_mask(value, shift, mask)

    @staticmethod
    def untemper(value: int) -> int:
        """Invert MT19937 tempering for one output word."""
        return untemper(value)

    def go(self, outputs: Iterable[int]) -> random.Random:
        """Recover a clone positioned after every supplied output."""
        return clone_from_outputs(outputs)

    # Original camelCase spellings retained for existing challenge scripts.
    unshiftRight = unshift_right
    unshiftLeft = unshift_left


class Recover(MT19937Recover):
    """Collect variable-width CPython ``getrandbits`` observations.

    CPython generates a multi-word integer from least-significant word to
    most-significant word.  If the width is not divisible by 32, the final
    generator output contributes only its most significant bits; its unused
    low bits are represented by :class:`PartialWord32`.
    """

    PartialInt32 = PartialWord32

    def __init__(self) -> None:
        self.chunks: list[Observation] = []

    def feed_random_bits(self, width: int, value: int) -> None:
        """Record ``value`` returned by CPython ``getrandbits(width)``."""
        remaining = _nonnegative_integer(width, "width")
        normalized_value = _nonnegative_integer(value, "value")
        if normalized_value.bit_length() > remaining:
            raise ValueError("value does not fit in the declared width")
        if remaining == 0:
            return

        while remaining >= WORD_BITS:
            self.chunks.append(normalized_value & WORD_MASK)
            normalized_value >>= WORD_BITS
            remaining -= WORD_BITS

        if remaining:
            self.chunks.append(
                PartialWord32(normalized_value, WORD_BITS - remaining)
            )

    def feedRandomBits(self, n: int, v: int) -> None:
        """Compatibility alias for :meth:`feed_random_bits`."""
        self.feed_random_bits(n, v)

    @staticmethod
    def _normalize_chunks(chunks: Iterable[Observation]) -> tuple[Observation, ...]:
        normalized: list[Observation] = []
        for position, chunk in enumerate(chunks):
            if isinstance(chunk, PartialWord32):
                normalized.append(chunk)
            else:
                normalized.append(_word(chunk, f"chunks[{position}]"))
        return tuple(normalized)

    @staticmethod
    def _matches(chunk: Observation, predicted: int) -> bool:
        if isinstance(chunk, PartialWord32):
            return chunk.matches(predicted)
        return chunk == predicted

    @classmethod
    def _candidate_clone(
        cls,
        state_words: Sequence[int],
        verification: Sequence[Observation],
    ) -> random.Random | None:
        clone = _random_from_words(state_words)
        for chunk in verification:
            predicted = clone.getrandbits(WORD_BITS)
            if not cls._matches(chunk, predicted):
                return None
        return clone

    @staticmethod
    def _future_signature(
        clone: random.Random,
    ) -> tuple[tuple[int, ...], tuple[int, ...]]:
        """Return enough future output to determine the predictive state.

        CPython stores 624 words (19,968 bits), while MT19937 has 19,937 bits
        of effective state.  Distinct stored arrays can therefore represent
        the same future stream.  This advances through the next twist and its
        first output.  Equality of the intervening output and resulting state
        proves that every later output is also equal.
        """
        probe = random.Random()
        probe.setstate(clone.getstate())
        index = probe.getstate()[1][-1]
        steps = 1 if index == STATE_WORDS else STATE_WORDS - index + 1
        outputs = tuple(probe.getrandbits(WORD_BITS) for _ in range(steps))
        return outputs, probe.getstate()[1]

    def recover(
        self,
        max_unknown_bits: int = 16,
        chunks: Iterable[Observation] | None = None,
        *,
        bitBruteLimit: int | None = None,
    ) -> random.Random | None:
        """Recover a unique clone from the collected output-word observations.

        ``None`` means that fewer than 624 words were supplied or that no
        candidate agrees with the verification observations.  Candidates
        representing the same future stream are equivalent; candidates with
        different futures raise :class:`AmbiguousStateError`.  The method
        never silently chooses between distinct predictions.

        ``max_unknown_bits`` bounds exhaustive work to ``2**n`` candidates.
        ``bitBruteLimit`` is the deprecated spelling from the original module.
        """
        if bitBruteLimit is not None:
            compatibility_limit = _nonnegative_integer(
                bitBruteLimit,
                "bitBruteLimit",
            )
            if max_unknown_bits != 16:
                raise TypeError(
                    "specify max_unknown_bits or bitBruteLimit, not both"
                )
            max_unknown_bits = compatibility_limit
        limit = _nonnegative_integer(max_unknown_bits, "max_unknown_bits")
        observations = self._normalize_chunks(
            self.chunks if chunks is None else chunks
        )
        if len(observations) < STATE_WORDS:
            return None

        first = observations[:STATE_WORDS]
        verification = observations[STATE_WORDS:]
        partials = [
            (position, chunk)
            for position, chunk in enumerate(first)
            if isinstance(chunk, PartialWord32)
        ]
        unknown_bits = sum(chunk.unknown_lsb_bits for _, chunk in partials)
        if unknown_bits > limit:
            raise CandidateLimitError(unknown_bits, limit)

        base_words = [
            chunk.known_msb << chunk.unknown_lsb_bits
            if isinstance(chunk, PartialWord32)
            else chunk
            for chunk in first
        ]
        base_state = list(map(untemper, base_words))
        recovered: random.Random | None = None
        recovered_signature: tuple[tuple[int, ...], tuple[int, ...]] | None = None
        for candidate_bits in range(1 << unknown_bits):
            candidate_state = base_state.copy()
            remaining = candidate_bits
            for position, chunk in partials:
                suffix_mask = (1 << chunk.unknown_lsb_bits) - 1
                complete_word = base_words[position] | (remaining & suffix_mask)
                candidate_state[position] = untemper(complete_word)
                remaining >>= chunk.unknown_lsb_bits

            candidate = self._candidate_clone(candidate_state, verification)
            if candidate is None:
                continue
            candidate_signature = self._future_signature(candidate)
            if recovered is None:
                recovered = candidate
                recovered_signature = candidate_signature
                continue
            if candidate_signature != recovered_signature:
                raise AmbiguousStateError(
                    "observations admit multiple MT19937 output streams; "
                    "supply more output for verification"
                )
        return recovered
