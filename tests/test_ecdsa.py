"""Deterministic correctness tests for every rewritten ECDSA attack."""

from __future__ import annotations

import unittest

from rewrite.ecdsa import (
    ecdsa_biased_nonce_known_lsb,
    ecdsa_biased_nonce_known_msb,
    ecdsa_biased_nonce_shared_msb,
    ecdsa_biased_nonce_zero_lsb,
    ecdsa_biased_nonce_zero_msb,
    ecdsa_nonce_hash_xor_privkey,
)


class ECDSANonceAttackTests(unittest.TestCase):
    """Generate small valid scalar equations with a unique private key."""

    ORDER = 1009
    PRIVATE_KEY = 321
    HASHES = [101, 202, 303, 404, 505, 606]
    R_VALUES = [37, 89, 157, 263, 401, 503]
    LEAKED_BITS = 5
    LOW_PART_BOUND = 32

    def signatures_for_nonces(self, nonces):
        """Create valid deterministic ECDSA scalar equations."""
        s_values = [
            (message_hash + r_value * self.PRIVATE_KEY)
            * pow(nonce, -1, self.ORDER)
            % self.ORDER
            for message_hash, r_value, nonce in zip(
                self.HASHES,
                self.R_VALUES,
                nonces,
                strict=True,
            )
        ]
        self.assertTrue(all(s_values))
        return self.HASHES, self.R_VALUES, s_values

    def test_zero_msb_attack(self) -> None:
        nonces = [3, 7, 12, 19, 25, 29]
        hashes, r_values, s_values = self.signatures_for_nonces(nonces)

        recovered = ecdsa_biased_nonce_zero_msb(
            hashes,
            r_values,
            s_values,
            self.ORDER,
            self.LEAKED_BITS,
            method="exact",
        )
        self.assertEqual(recovered, self.PRIVATE_KEY)

    def test_zero_lsb_attack(self) -> None:
        nonces = [self.LOW_PART_BOUND * value for value in (3, 7, 11, 17, 23, 29)]
        hashes, r_values, s_values = self.signatures_for_nonces(nonces)

        recovered = ecdsa_biased_nonce_zero_lsb(
            hashes,
            r_values,
            s_values,
            self.ORDER,
            self.LEAKED_BITS,
            method="exact",
        )
        self.assertEqual(recovered, self.PRIVATE_KEY)

    def test_known_msb_attack(self) -> None:
        nonces = [99, 231, 364, 563, 761, 957]
        known_parts = [nonce // self.LOW_PART_BOUND for nonce in nonces]
        hashes, r_values, s_values = self.signatures_for_nonces(nonces)

        recovered = ecdsa_biased_nonce_known_msb(
            hashes,
            r_values,
            s_values,
            known_parts,
            self.ORDER,
            self.LEAKED_BITS,
            method="exact",
        )
        self.assertEqual(recovered, self.PRIVATE_KEY)

    def test_known_lsb_attack(self) -> None:
        low_parts = [3, 7, 12, 19, 25, 29]
        high_parts = [2, 5, 9, 13, 17, 21]
        nonces = [
            low + self.LOW_PART_BOUND * high
            for low, high in zip(low_parts, high_parts, strict=True)
        ]
        hashes, r_values, s_values = self.signatures_for_nonces(nonces)

        recovered = ecdsa_biased_nonce_known_lsb(
            hashes,
            r_values,
            s_values,
            low_parts,
            self.ORDER,
            self.LEAKED_BITS,
            method="exact",
        )
        self.assertEqual(recovered, self.PRIVATE_KEY)

    def test_shared_msb_attack_uses_signed_nonce_differences(self) -> None:
        nonces = [11 * self.LOW_PART_BOUND + low for low in (3, 7, 12, 19, 25, 29)]
        hashes, r_values, s_values = self.signatures_for_nonces(nonces)

        recovered = ecdsa_biased_nonce_shared_msb(
            hashes,
            r_values,
            s_values,
            self.ORDER,
            self.LEAKED_BITS,
            method="exact",
        )
        self.assertEqual(recovered, self.PRIVATE_KEY)

    def test_nonce_hash_xor_private_key_attack(self) -> None:
        nonce_hashes = [17, 88, 145, 277, 511, 650]
        nonces = [value ^ self.PRIVATE_KEY for value in nonce_hashes]
        hashes, r_values, s_values = self.signatures_for_nonces(nonces)

        recovered = ecdsa_nonce_hash_xor_privkey(
            nonce_hashes,
            hashes,
            r_values,
            s_values,
            self.ORDER,
            method="exact",
        )
        self.assertEqual(recovered, self.PRIVATE_KEY)


if __name__ == "__main__":
    unittest.main()
