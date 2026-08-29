"""Correctness tests for dependency-free hash length extension."""

from __future__ import annotations

import hashlib
import unittest

from rewrite.hashextend import (
    extend_hash,
    extend_hash_for_lengths,
    glue_padding,
    md5_extend,
)


class HashExtensionTests(unittest.TestCase):
    """Verify forged prefix hashes against the standard-library hashes."""

    SECRET = b"key"
    MESSAGE = b"a=1"
    SUFFIX = b"&admin=1"

    @staticmethod
    def hash_bytes(algorithm: str, data: bytes) -> bytes:
        """Hash test data even when MD5 is disabled for security use."""
        try:
            return hashlib.new(
                algorithm,
                data,
                usedforsecurity=False,
            ).digest()
        except TypeError:  # Compatibility with older Python builds.
            return hashlib.new(algorithm, data).digest()

    def original_digest(self, algorithm: str) -> bytes:
        return self.hash_bytes(algorithm, self.SECRET + self.MESSAGE)

    def assert_valid_extension(self, algorithm: str, result) -> None:
        self.assertEqual(result.algorithm, algorithm)
        self.assertEqual(result.secret_length, len(self.SECRET))
        self.assertEqual(result.suffix, self.SUFFIX)
        self.assertEqual(
            result.digest,
            self.hash_bytes(algorithm, self.SECRET + result.message),
        )

    def test_glue_padding_boundaries_and_length_encoding(self) -> None:
        cases = {
            "md5": (64, 8, "little", (0, 55, 56, 63, 64)),
            "sha1": (64, 8, "big", (0, 55, 56, 63, 64)),
            "sha256": (64, 8, "big", (0, 55, 56, 63, 64)),
            "sha512": (128, 16, "big", (0, 111, 112, 127, 128)),
        }
        for algorithm, (block_size, length_size, byteorder, lengths) in cases.items():
            for message_length in lengths:
                with self.subTest(
                    algorithm=algorithm,
                    message_length=message_length,
                ):
                    padding = glue_padding(message_length, algorithm)
                    self.assertEqual(padding[0], 0x80)
                    self.assertEqual(
                        (message_length + len(padding)) % block_size,
                        0,
                    )
                    self.assertEqual(
                        int.from_bytes(padding[-length_size:], byteorder),
                        message_length * 8,
                    )
                    self.assertEqual(
                        padding[1:-length_size],
                        b"\x00" * (len(padding) - length_size - 1),
                    )

        self.assertEqual(
            glue_padding(3, "SHA-256"),
            glue_padding(3, "sha_256"),
        )

    def test_every_supported_algorithm_forges_a_valid_prefix_hash(self) -> None:
        for algorithm in ("md5", "sha1", "sha256", "sha512"):
            with self.subTest(algorithm=algorithm):
                result = extend_hash(
                    self.original_digest(algorithm),
                    self.MESSAGE,
                    self.SUFFIX,
                    len(self.SECRET),
                    algorithm,
                )
                self.assert_valid_extension(algorithm, result)
                self.assertEqual(
                    result.message,
                    self.MESSAGE + result.glue_padding + self.SUFFIX,
                )

    def test_long_suffix_crosses_hash_block_boundaries(self) -> None:
        suffix = bytes(range(140))
        for algorithm in ("md5", "sha1", "sha256", "sha512"):
            with self.subTest(algorithm=algorithm):
                original = self.original_digest(algorithm)
                result = extend_hash(
                    original,
                    bytearray(self.MESSAGE),
                    memoryview(suffix),
                    len(self.SECRET),
                    algorithm,
                )
                self.assertEqual(
                    result.digest,
                    self.hash_bytes(algorithm, self.SECRET + result.message),
                )

    def test_hexadecimal_digest_and_result_properties(self) -> None:
        result = extend_hash(
            self.original_digest("sha256").hex().upper(),
            self.MESSAGE,
            self.SUFFIX,
            len(self.SECRET),
            "SHA-256",
        )

        self.assert_valid_extension("sha256", result)
        self.assertEqual(result.hexdigest, result.digest.hex())
        self.assertEqual(
            result.appended_data,
            result.glue_padding + self.SUFFIX,
        )
        self.assertEqual(result.forged_message, result.message)

    def test_secret_length_enumeration_contains_the_valid_guess(self) -> None:
        secret = b"hidden!"
        message = b"file=report"
        suffix = b"&access=all"
        original_digest = self.hash_bytes("sha1", secret + message)

        candidates = extend_hash_for_lengths(
            original_digest,
            message,
            suffix,
            range(3, 11),
            "sha1",
        )
        valid_lengths = [
            candidate.secret_length
            for candidate in candidates
            if candidate.digest
            == self.hash_bytes("sha1", secret + candidate.message)
        ]

        self.assertEqual(len(candidates), 8)
        self.assertEqual(valid_lengths, [len(secret)])
        self.assertEqual(
            extend_hash_for_lengths(
                original_digest,
                message,
                suffix,
                [],
                "sha1",
            ),
            [],
        )

    def test_legacy_md5_wrapper_preserves_total_length_semantics(self) -> None:
        original = self.SECRET + self.MESSAGE
        old_digest = self.hash_bytes("md5", original)

        new_digest, appended_data = md5_extend(
            old_digest.hex(),
            len(original),
            self.SUFFIX,
        )

        self.assertEqual(
            new_digest,
            self.hash_bytes("md5", original + appended_data),
        )
        self.assertEqual(
            appended_data,
            glue_padding(len(original), "md5") + self.SUFFIX,
        )

    def test_invalid_types_and_lengths_are_rejected(self) -> None:
        digest = self.original_digest("md5")
        with self.assertRaisesRegex(TypeError, "original_message"):
            extend_hash(digest, "message", self.SUFFIX, 3, "md5")
        with self.assertRaisesRegex(TypeError, "suffix"):
            extend_hash(digest, self.MESSAGE, "suffix", 3, "md5")
        with self.assertRaisesRegex(TypeError, "not bool"):
            extend_hash(digest, self.MESSAGE, self.SUFFIX, True, "md5")
        with self.assertRaisesRegex(ValueError, "non-negative"):
            glue_padding(-1, "md5")
        with self.assertRaisesRegex(TypeError, "algorithm"):
            glue_padding(1, b"md5")
        with self.assertRaisesRegex(TypeError, "iterable"):
            extend_hash_for_lengths(
                digest,
                self.MESSAGE,
                self.SUFFIX,
                3,
                "md5",
            )
        with self.assertRaisesRegex(ValueError, "too long"):
            glue_padding(1 << 61, "sha256")

    def test_malformed_and_unsupported_digests_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "exactly 16 bytes"):
            extend_hash(b"x" * 15, self.MESSAGE, self.SUFFIX, 3, "md5")
        with self.assertRaisesRegex(ValueError, "exactly 16 bytes"):
            extend_hash_for_lengths(
                b"x" * 15,
                self.MESSAGE,
                self.SUFFIX,
                [],
                "md5",
            )
        with self.assertRaisesRegex(ValueError, "hexadecimal"):
            extend_hash("g" * 32, self.MESSAGE, self.SUFFIX, 3, "md5")

        digest = self.original_digest("sha256")
        for algorithm in ("sha224", "sha384", "sha3-512"):
            with self.subTest(algorithm=algorithm):
                with self.assertRaisesRegex(ValueError, "unsupported"):
                    extend_hash(
                        digest,
                        self.MESSAGE,
                        self.SUFFIX,
                        3,
                        algorithm,
                    )


if __name__ == "__main__":
    unittest.main()
