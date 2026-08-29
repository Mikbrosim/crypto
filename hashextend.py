"""Length-extension helpers for Merkle--Damgard prefix hashes.

This module demonstrates attacks against constructions of the form
``Hash(secret || message)``.  Given the digest and a guess for the secret's
length, an attacker can append data without learning the secret.  HMAC and
SHA-3 do not have this property and are deliberately unsupported.

MD5, SHA-1, SHA-256, and SHA-512 are implemented without optional packages.
Only their compression functions are needed: the supplied digest is decoded
as the chaining state after the original message and its glue padding.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from operator import index as integer_index
from typing import Callable


__all__ = [
    "HashExtension",
    "extend_hash",
    "extend_hash_for_lengths",
    "glue_padding",
    "md5_extend",
]


_MASK32 = (1 << 32) - 1
_MASK64 = (1 << 64) - 1
_HEX_DIGITS = frozenset("0123456789abcdefABCDEF")


@dataclass(frozen=True)
class HashExtension:
    """One candidate forged from a particular secret-length guess.

    ``message`` is the complete attacker-visible message to send to the
    verifier: the known original message, glue padding, and suffix.  The
    unknown secret is intentionally not part of it.
    """

    algorithm: str
    digest: bytes
    message: bytes
    glue_padding: bytes
    secret_length: int
    suffix: bytes

    @property
    def hexdigest(self) -> str:
        """Return the candidate digest in hexadecimal form."""
        return self.digest.hex()

    @property
    def appended_data(self) -> bytes:
        """Return the bytes appended to the known original message."""
        return self.glue_padding + self.suffix

    @property
    def forged_message(self) -> bytes:
        """Return an explicit alias for :attr:`message`."""
        return self.message


CompressionFunction = Callable[[tuple[int, ...], bytes], tuple[int, ...]]


@dataclass(frozen=True)
class _HashSpecification:
    name: str
    digest_size: int
    block_size: int
    length_size: int
    byteorder: str
    word_size: int
    compress: CompressionFunction


def _rotate_left(value: int, amount: int, bits: int) -> int:
    mask = (1 << bits) - 1
    return ((value << amount) | (value >> (bits - amount))) & mask


def _rotate_right(value: int, amount: int, bits: int) -> int:
    mask = (1 << bits) - 1
    return ((value >> amount) | (value << (bits - amount))) & mask


_MD5_SHIFTS = (
    7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22,
    5, 9, 14, 20, 5, 9, 14, 20, 5, 9, 14, 20, 5, 9, 14, 20,
    4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23,
    6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21,
)

_MD5_CONSTANTS = (
    0xD76AA478, 0xE8C7B756, 0x242070DB, 0xC1BDCEEE,
    0xF57C0FAF, 0x4787C62A, 0xA8304613, 0xFD469501,
    0x698098D8, 0x8B44F7AF, 0xFFFF5BB1, 0x895CD7BE,
    0x6B901122, 0xFD987193, 0xA679438E, 0x49B40821,
    0xF61E2562, 0xC040B340, 0x265E5A51, 0xE9B6C7AA,
    0xD62F105D, 0x02441453, 0xD8A1E681, 0xE7D3FBC8,
    0x21E1CDE6, 0xC33707D6, 0xF4D50D87, 0x455A14ED,
    0xA9E3E905, 0xFCEFA3F8, 0x676F02D9, 0x8D2A4C8A,
    0xFFFA3942, 0x8771F681, 0x6D9D6122, 0xFDE5380C,
    0xA4BEEA44, 0x4BDECFA9, 0xF6BB4B60, 0xBEBFBC70,
    0x289B7EC6, 0xEAA127FA, 0xD4EF3085, 0x04881D05,
    0xD9D4D039, 0xE6DB99E5, 0x1FA27CF8, 0xC4AC5665,
    0xF4292244, 0x432AFF97, 0xAB9423A7, 0xFC93A039,
    0x655B59C3, 0x8F0CCC92, 0xFFEFF47D, 0x85845DD1,
    0x6FA87E4F, 0xFE2CE6E0, 0xA3014314, 0x4E0811A1,
    0xF7537E82, 0xBD3AF235, 0x2AD7D2BB, 0xEB86D391,
)


def _compress_md5(state: tuple[int, ...], block: bytes) -> tuple[int, ...]:
    words = tuple(
        int.from_bytes(block[offset : offset + 4], "little")
        for offset in range(0, 64, 4)
    )
    a, b, c, d = state
    initial = state

    for round_index in range(64):
        if round_index < 16:
            nonlinear = (b & c) | ((~b) & d)
            word_index = round_index
        elif round_index < 32:
            nonlinear = (d & b) | ((~d) & c)
            word_index = (5 * round_index + 1) % 16
        elif round_index < 48:
            nonlinear = b ^ c ^ d
            word_index = (3 * round_index + 5) % 16
        else:
            nonlinear = c ^ (b | (~d))
            word_index = (7 * round_index) % 16

        mixed = (
            a
            + nonlinear
            + _MD5_CONSTANTS[round_index]
            + words[word_index]
        ) & _MASK32
        a, d, c, b = (
            d,
            c,
            b,
            (b + _rotate_left(mixed, _MD5_SHIFTS[round_index], 32))
            & _MASK32,
        )

    return tuple(
        (old + new) & _MASK32
        for old, new in zip(initial, (a, b, c, d), strict=True)
    )


def _compress_sha1(state: tuple[int, ...], block: bytes) -> tuple[int, ...]:
    words = [
        int.from_bytes(block[offset : offset + 4], "big")
        for offset in range(0, 64, 4)
    ]
    for word_index in range(16, 80):
        words.append(
            _rotate_left(
                words[word_index - 3]
                ^ words[word_index - 8]
                ^ words[word_index - 14]
                ^ words[word_index - 16],
                1,
                32,
            )
        )

    a, b, c, d, e = state
    initial = state
    for round_index in range(80):
        if round_index < 20:
            nonlinear = (b & c) | ((~b) & d)
            constant = 0x5A827999
        elif round_index < 40:
            nonlinear = b ^ c ^ d
            constant = 0x6ED9EBA1
        elif round_index < 60:
            nonlinear = (b & c) | (b & d) | (c & d)
            constant = 0x8F1BBCDC
        else:
            nonlinear = b ^ c ^ d
            constant = 0xCA62C1D6

        temporary = (
            _rotate_left(a, 5, 32)
            + nonlinear
            + e
            + constant
            + words[round_index]
        ) & _MASK32
        e, d, c, b, a = d, c, _rotate_left(b, 30, 32), a, temporary

    return tuple(
        (old + new) & _MASK32
        for old, new in zip(initial, (a, b, c, d, e), strict=True)
    )


_SHA256_CONSTANTS = (
    0x428A2F98, 0x71374491, 0xB5C0FBCF, 0xE9B5DBA5,
    0x3956C25B, 0x59F111F1, 0x923F82A4, 0xAB1C5ED5,
    0xD807AA98, 0x12835B01, 0x243185BE, 0x550C7DC3,
    0x72BE5D74, 0x80DEB1FE, 0x9BDC06A7, 0xC19BF174,
    0xE49B69C1, 0xEFBE4786, 0x0FC19DC6, 0x240CA1CC,
    0x2DE92C6F, 0x4A7484AA, 0x5CB0A9DC, 0x76F988DA,
    0x983E5152, 0xA831C66D, 0xB00327C8, 0xBF597FC7,
    0xC6E00BF3, 0xD5A79147, 0x06CA6351, 0x14292967,
    0x27B70A85, 0x2E1B2138, 0x4D2C6DFC, 0x53380D13,
    0x650A7354, 0x766A0ABB, 0x81C2C92E, 0x92722C85,
    0xA2BFE8A1, 0xA81A664B, 0xC24B8B70, 0xC76C51A3,
    0xD192E819, 0xD6990624, 0xF40E3585, 0x106AA070,
    0x19A4C116, 0x1E376C08, 0x2748774C, 0x34B0BCB5,
    0x391C0CB3, 0x4ED8AA4A, 0x5B9CCA4F, 0x682E6FF3,
    0x748F82EE, 0x78A5636F, 0x84C87814, 0x8CC70208,
    0x90BEFFFA, 0xA4506CEB, 0xBEF9A3F7, 0xC67178F2,
)


def _compress_sha256(state: tuple[int, ...], block: bytes) -> tuple[int, ...]:
    words = [
        int.from_bytes(block[offset : offset + 4], "big")
        for offset in range(0, 64, 4)
    ]
    for word_index in range(16, 64):
        previous_15 = words[word_index - 15]
        previous_2 = words[word_index - 2]
        sigma_0 = (
            _rotate_right(previous_15, 7, 32)
            ^ _rotate_right(previous_15, 18, 32)
            ^ (previous_15 >> 3)
        )
        sigma_1 = (
            _rotate_right(previous_2, 17, 32)
            ^ _rotate_right(previous_2, 19, 32)
            ^ (previous_2 >> 10)
        )
        words.append(
            (words[word_index - 16] + sigma_0 + words[word_index - 7] + sigma_1)
            & _MASK32
        )

    a, b, c, d, e, f, g, h = state
    initial = state
    for round_index in range(64):
        sum_1 = (
            _rotate_right(e, 6, 32)
            ^ _rotate_right(e, 11, 32)
            ^ _rotate_right(e, 25, 32)
        )
        choice = (e & f) ^ ((~e) & g)
        temporary_1 = (
            h
            + sum_1
            + choice
            + _SHA256_CONSTANTS[round_index]
            + words[round_index]
        ) & _MASK32
        sum_0 = (
            _rotate_right(a, 2, 32)
            ^ _rotate_right(a, 13, 32)
            ^ _rotate_right(a, 22, 32)
        )
        majority = (a & b) ^ (a & c) ^ (b & c)
        temporary_2 = (sum_0 + majority) & _MASK32
        h, g, f, e, d, c, b, a = (
            g,
            f,
            e,
            (d + temporary_1) & _MASK32,
            c,
            b,
            a,
            (temporary_1 + temporary_2) & _MASK32,
        )

    return tuple(
        (old + new) & _MASK32
        for old, new in zip(initial, (a, b, c, d, e, f, g, h), strict=True)
    )


_SHA512_CONSTANTS = (
    0x428A2F98D728AE22, 0x7137449123EF65CD,
    0xB5C0FBCFEC4D3B2F, 0xE9B5DBA58189DBBC,
    0x3956C25BF348B538, 0x59F111F1B605D019,
    0x923F82A4AF194F9B, 0xAB1C5ED5DA6D8118,
    0xD807AA98A3030242, 0x12835B0145706FBE,
    0x243185BE4EE4B28C, 0x550C7DC3D5FFB4E2,
    0x72BE5D74F27B896F, 0x80DEB1FE3B1696B1,
    0x9BDC06A725C71235, 0xC19BF174CF692694,
    0xE49B69C19EF14AD2, 0xEFBE4786384F25E3,
    0x0FC19DC68B8CD5B5, 0x240CA1CC77AC9C65,
    0x2DE92C6F592B0275, 0x4A7484AA6EA6E483,
    0x5CB0A9DCBD41FBD4, 0x76F988DA831153B5,
    0x983E5152EE66DFAB, 0xA831C66D2DB43210,
    0xB00327C898FB213F, 0xBF597FC7BEEF0EE4,
    0xC6E00BF33DA88FC2, 0xD5A79147930AA725,
    0x06CA6351E003826F, 0x142929670A0E6E70,
    0x27B70A8546D22FFC, 0x2E1B21385C26C926,
    0x4D2C6DFC5AC42AED, 0x53380D139D95B3DF,
    0x650A73548BAF63DE, 0x766A0ABB3C77B2A8,
    0x81C2C92E47EDAEE6, 0x92722C851482353B,
    0xA2BFE8A14CF10364, 0xA81A664BBC423001,
    0xC24B8B70D0F89791, 0xC76C51A30654BE30,
    0xD192E819D6EF5218, 0xD69906245565A910,
    0xF40E35855771202A, 0x106AA07032BBD1B8,
    0x19A4C116B8D2D0C8, 0x1E376C085141AB53,
    0x2748774CDF8EEB99, 0x34B0BCB5E19B48A8,
    0x391C0CB3C5C95A63, 0x4ED8AA4AE3418ACB,
    0x5B9CCA4F7763E373, 0x682E6FF3D6B2B8A3,
    0x748F82EE5DEFB2FC, 0x78A5636F43172F60,
    0x84C87814A1F0AB72, 0x8CC702081A6439EC,
    0x90BEFFFA23631E28, 0xA4506CEBDE82BDE9,
    0xBEF9A3F7B2C67915, 0xC67178F2E372532B,
    0xCA273ECEEA26619C, 0xD186B8C721C0C207,
    0xEADA7DD6CDE0EB1E, 0xF57D4F7FEE6ED178,
    0x06F067AA72176FBA, 0x0A637DC5A2C898A6,
    0x113F9804BEF90DAE, 0x1B710B35131C471B,
    0x28DB77F523047D84, 0x32CAAB7B40C72493,
    0x3C9EBE0A15C9BEBC, 0x431D67C49C100D4C,
    0x4CC5D4BECB3E42B6, 0x597F299CFC657E2A,
    0x5FCB6FAB3AD6FAEC, 0x6C44198C4A475817,
)


def _compress_sha512(state: tuple[int, ...], block: bytes) -> tuple[int, ...]:
    words = [
        int.from_bytes(block[offset : offset + 8], "big")
        for offset in range(0, 128, 8)
    ]
    for word_index in range(16, 80):
        previous_15 = words[word_index - 15]
        previous_2 = words[word_index - 2]
        sigma_0 = (
            _rotate_right(previous_15, 1, 64)
            ^ _rotate_right(previous_15, 8, 64)
            ^ (previous_15 >> 7)
        )
        sigma_1 = (
            _rotate_right(previous_2, 19, 64)
            ^ _rotate_right(previous_2, 61, 64)
            ^ (previous_2 >> 6)
        )
        words.append(
            (words[word_index - 16] + sigma_0 + words[word_index - 7] + sigma_1)
            & _MASK64
        )

    a, b, c, d, e, f, g, h = state
    initial = state
    for round_index in range(80):
        sum_1 = (
            _rotate_right(e, 14, 64)
            ^ _rotate_right(e, 18, 64)
            ^ _rotate_right(e, 41, 64)
        )
        choice = (e & f) ^ ((~e) & g)
        temporary_1 = (
            h
            + sum_1
            + choice
            + _SHA512_CONSTANTS[round_index]
            + words[round_index]
        ) & _MASK64
        sum_0 = (
            _rotate_right(a, 28, 64)
            ^ _rotate_right(a, 34, 64)
            ^ _rotate_right(a, 39, 64)
        )
        majority = (a & b) ^ (a & c) ^ (b & c)
        temporary_2 = (sum_0 + majority) & _MASK64
        h, g, f, e, d, c, b, a = (
            g,
            f,
            e,
            (d + temporary_1) & _MASK64,
            c,
            b,
            a,
            (temporary_1 + temporary_2) & _MASK64,
        )

    return tuple(
        (old + new) & _MASK64
        for old, new in zip(initial, (a, b, c, d, e, f, g, h), strict=True)
    )


_SPECIFICATIONS = {
    "md5": _HashSpecification("md5", 16, 64, 8, "little", 4, _compress_md5),
    "sha1": _HashSpecification("sha1", 20, 64, 8, "big", 4, _compress_sha1),
    "sha256": _HashSpecification(
        "sha256", 32, 64, 8, "big", 4, _compress_sha256
    ),
    "sha512": _HashSpecification(
        "sha512", 64, 128, 16, "big", 8, _compress_sha512
    ),
}


def _normalize_algorithm(algorithm: str) -> _HashSpecification:
    if not isinstance(algorithm, str):
        raise TypeError("algorithm must be a string")
    canonical = algorithm.lower().replace("-", "").replace("_", "")
    try:
        return _SPECIFICATIONS[canonical]
    except KeyError as error:
        raise ValueError(
            "unsupported algorithm; choose md5, sha1, sha256, or sha512 "
            "(SHA-3 and truncated SHA variants cannot use this interface)"
        ) from error


def _nonnegative_integer(value: object, name: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer, not bool")
    try:
        normalized = integer_index(value)
    except TypeError as error:
        raise TypeError(f"{name} must be an integer") from error
    if normalized < 0:
        raise ValueError(f"{name} must be non-negative")
    return normalized


def _bytes(value: object, name: str) -> bytes:
    if isinstance(value, bytes):
        return value
    if isinstance(value, (bytearray, memoryview)):
        return bytes(value)
    raise TypeError(f"{name} must be bytes-like")


def _digest_bytes(digest: object, specification: _HashSpecification) -> bytes:
    if isinstance(digest, str):
        expected_hex_length = 2 * specification.digest_size
        if (
            len(digest) != expected_hex_length
            or not all(character in _HEX_DIGITS for character in digest)
        ):
            raise ValueError(
                f"{specification.name} digest must contain exactly "
                f"{expected_hex_length} hexadecimal characters"
            )
        normalized = bytes.fromhex(digest)
    else:
        normalized = _bytes(digest, "digest")

    if len(normalized) != specification.digest_size:
        raise ValueError(
            f"{specification.name} digest must be exactly "
            f"{specification.digest_size} bytes"
        )
    return normalized


def _validate_encodable_length(
    message_length: int,
    specification: _HashSpecification,
) -> None:
    length_bits = 8 * specification.length_size
    maximum_bytes = ((1 << length_bits) - 1) // 8
    if message_length > maximum_bytes:
        raise ValueError(
            f"message is too long for {specification.name}'s "
            f"{length_bits}-bit length field"
        )


def _glue_padding(
    message_length: int,
    specification: _HashSpecification,
) -> bytes:
    _validate_encodable_length(message_length, specification)
    zero_count = -(
        message_length + 1 + specification.length_size
    ) % specification.block_size
    encoded_length = (message_length * 8).to_bytes(
        specification.length_size,
        specification.byteorder,
    )
    return b"\x80" + b"\x00" * zero_count + encoded_length


def glue_padding(message_length: int, algorithm: str = "md5") -> bytes:
    """Return the exact padding after a byte-length message.

    MD5 encodes the bit length little-endian.  SHA-1 and SHA-256 use an
    eight-byte big-endian field, while SHA-512 uses a sixteen-byte one.
    """
    specification = _normalize_algorithm(algorithm)
    normalized_length = _nonnegative_integer(message_length, "message_length")
    return _glue_padding(normalized_length, specification)


def _decode_state(
    digest: bytes,
    specification: _HashSpecification,
) -> tuple[int, ...]:
    return tuple(
        int.from_bytes(
            digest[offset : offset + specification.word_size],
            specification.byteorder,
        )
        for offset in range(0, len(digest), specification.word_size)
    )


def _encode_state(
    state: tuple[int, ...],
    specification: _HashSpecification,
) -> bytes:
    return b"".join(
        word.to_bytes(specification.word_size, specification.byteorder)
        for word in state
    )


def _resume_hash(
    digest: bytes,
    processed_length: int,
    suffix: bytes,
    specification: _HashSpecification,
) -> bytes:
    if processed_length % specification.block_size:
        raise ValueError("processed_length must end on a hash-block boundary")

    final_message_length = processed_length + len(suffix)
    tail = suffix + _glue_padding(final_message_length, specification)
    state = _decode_state(digest, specification)
    for offset in range(0, len(tail), specification.block_size):
        state = specification.compress(
            state,
            tail[offset : offset + specification.block_size],
        )
    return _encode_state(state, specification)


def _forge_extension(
    digest: bytes,
    original_message: bytes,
    suffix: bytes,
    secret_length: int,
    specification: _HashSpecification,
) -> HashExtension:
    original_length = secret_length + len(original_message)
    padding = _glue_padding(original_length, specification)
    processed_length = original_length + len(padding)
    forged_digest = _resume_hash(
        digest,
        processed_length,
        suffix,
        specification,
    )
    return HashExtension(
        algorithm=specification.name,
        digest=forged_digest,
        message=original_message + padding + suffix,
        glue_padding=padding,
        secret_length=secret_length,
        suffix=suffix,
    )


def extend_hash(
    digest: bytes | bytearray | memoryview | str,
    original_message: bytes | bytearray | memoryview,
    suffix: bytes | bytearray | memoryview,
    secret_length: int,
    algorithm: str = "md5",
) -> HashExtension:
    """Forge one candidate for ``Hash(secret || original_message)``.

    ``secret_length`` is a guess in bytes.  The returned ``message`` omits
    the unknown secret and is ready to send to a verifier that prepends it.
    A hexadecimal digest string or raw digest bytes may be supplied.
    """
    specification = _normalize_algorithm(algorithm)
    normalized_digest = _digest_bytes(digest, specification)
    normalized_message = _bytes(original_message, "original_message")
    normalized_suffix = _bytes(suffix, "suffix")
    normalized_secret_length = _nonnegative_integer(
        secret_length,
        "secret_length",
    )
    return _forge_extension(
        normalized_digest,
        normalized_message,
        normalized_suffix,
        normalized_secret_length,
        specification,
    )


def extend_hash_for_lengths(
    digest: bytes | bytearray | memoryview | str,
    original_message: bytes | bytearray | memoryview,
    suffix: bytes | bytearray | memoryview,
    secret_lengths: Iterable[int],
    algorithm: str = "md5",
) -> list[HashExtension]:
    """Forge candidates for each possible secret length in ``secret_lengths``."""
    try:
        lengths = iter(secret_lengths)
    except TypeError as error:
        raise TypeError("secret_lengths must be iterable") from error

    specification = _normalize_algorithm(algorithm)
    normalized_digest = _digest_bytes(digest, specification)
    normalized_message = _bytes(original_message, "original_message")
    normalized_suffix = _bytes(suffix, "suffix")
    return [
        _forge_extension(
            normalized_digest,
            normalized_message,
            normalized_suffix,
            _nonnegative_integer(secret_length, "secret_length"),
            specification,
        )
        for secret_length in lengths
    ]


def md5_extend(
    cur_hash: bytes | bytearray | memoryview | str,
    mlen: int,
    to_add: bytes | bytearray | memoryview,
) -> tuple[bytes, bytes]:
    """Compatibility wrapper for the original ``hash_extend.md5_extend``.

    Unlike :func:`extend_hash`, ``mlen`` is the total number of bytes already
    hashed before MD5 padding, including any unknown prefix.  The second
    return value is ``glue_padding + to_add``.
    """
    specification = _SPECIFICATIONS["md5"]
    normalized_digest = _digest_bytes(cur_hash, specification)
    normalized_length = _nonnegative_integer(mlen, "mlen")
    normalized_suffix = _bytes(to_add, "to_add")
    padding = _glue_padding(normalized_length, specification)
    forged_digest = _resume_hash(
        normalized_digest,
        normalized_length + len(padding),
        normalized_suffix,
        specification,
    )
    return forged_digest, padding + normalized_suffix
