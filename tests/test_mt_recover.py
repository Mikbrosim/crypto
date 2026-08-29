"""Deterministic correctness tests for MT19937 state recovery."""

from __future__ import annotations

import random
import unittest

from rewrite.mt_recover import (
    AmbiguousStateError,
    CandidateLimitError,
    MT19937Recover,
    ObservationMismatchError,
    PartialWord32,
    Recover,
    clone_from_outputs,
    temper,
    untemper,
)


class MT19937RecoveryTests(unittest.TestCase):
    def assert_future_matches(
        self,
        source: random.Random,
        clone: random.Random,
    ) -> None:
        widths = (32, 1, 64, 127, 0, 17, 96, 31)
        for width in widths:
            self.assertEqual(source.getrandbits(width), clone.getrandbits(width))
        self.assertEqual(source.random(), clone.random())
        for _ in range(20):
            self.assertEqual(source.getrandbits(32), clone.getrandbits(32))

    def test_temper_and_untemper_are_exact_inverses(self) -> None:
        recoverer = MT19937Recover()
        values = (0, 1, 0x12345678, 0x80000000, 0xFFFFFFFF)
        for value in values:
            self.assertEqual(untemper(temper(value)), value)
            self.assertEqual(recoverer.untemper(temper(value)), value)

            right = value ^ (value >> 11)
            self.assertEqual(recoverer.unshift_right(right, 11), value)
            self.assertEqual(recoverer.unshiftRight(right, 11), value)

            left = value ^ ((value << 7) & 0x9D2C5680)
            self.assertEqual(
                recoverer.unshift_left(left, 7, 0x9D2C5680),
                value,
            )
            self.assertEqual(
                recoverer.unshiftLeft(left, 7, 0x9D2C5680),
                value,
            )

    def test_clone_predicts_from_arbitrarily_aligned_state_windows(self) -> None:
        for burned_words in (0, 1, 137, 623, 1000):
            with self.subTest(burned_words=burned_words):
                source = random.Random(0xC0FFEE)
                for _ in range(burned_words):
                    source.getrandbits(32)
                outputs = [source.getrandbits(32) for _ in range(624)]
                clone = clone_from_outputs(outputs)
                self.assert_future_matches(source, clone)

    def test_extra_outputs_are_verified_and_consumed(self) -> None:
        source = random.Random(20230809)
        outputs = [source.getrandbits(32) for _ in range(640)]
        clone = clone_from_outputs(outputs)
        self.assert_future_matches(source, clone)

        corrupt = outputs.copy()
        corrupt[631] ^= 1
        with self.assertRaises(ObservationMismatchError) as context:
            clone_from_outputs(corrupt)
        self.assertEqual(context.exception.index, 631)
        self.assertEqual(context.exception.observed, corrupt[631])

    def test_go_compatibility_interface_clones_complete_outputs(self) -> None:
        source = random.Random(55)
        outputs = [source.getrandbits(32) for _ in range(625)]
        clone = MT19937Recover().go(outputs)
        self.assert_future_matches(source, clone)

    def test_variable_width_collector_matches_cpython_word_packing(self) -> None:
        source = random.Random(8675309)
        collector = Recover()
        widths = (0, 32, 64, 96, 1024) * 17
        for position, width in enumerate(widths):
            value = source.getrandbits(width)
            if position % 2:
                collector.feedRandomBits(width, value)
            else:
                collector.feed_random_bits(width, value)

        self.assertGreater(len(collector.chunks), 624)
        clone = collector.recover()
        self.assertIsNotNone(clone)
        self.assert_future_matches(source, clone)

    def test_partial_word_packing_matches_individual_cpython_outputs(self) -> None:
        for width in (1, 17, 31, 33, 63, 65, 95, 127):
            with self.subTest(width=width):
                source = random.Random(12345)
                reference = random.Random(12345)
                collector = Recover()
                collector.feed_random_bits(width, source.getrandbits(width))

                expected = []
                remaining = width
                while remaining >= 32:
                    expected.append(reference.getrandbits(32))
                    remaining -= 32
                if remaining:
                    complete_word = reference.getrandbits(32)
                    expected.append(
                        PartialWord32(
                            complete_word >> (32 - remaining),
                            32 - remaining,
                        )
                    )
                self.assertEqual(collector.chunks, expected)

    def test_sparse_partial_output_is_recovered_uniquely(self) -> None:
        source = random.Random(424242)
        collector = Recover()
        collector.feed_random_bits(63, source.getrandbits(63))
        for _ in range(622 + 8):
            collector.feed_random_bits(32, source.getrandbits(32))

        self.assertIsInstance(collector.chunks[1], PartialWord32)
        clone = collector.recover(max_unknown_bits=1)
        self.assertIsNotNone(clone)
        self.assert_future_matches(source, clone)

    def test_partial_verification_does_not_increase_search_effort(self) -> None:
        source = random.Random(5150)
        collector = Recover()
        for _ in range(624):
            collector.feed_random_bits(32, source.getrandbits(32))
        collector.feed_random_bits(1, source.getrandbits(1))

        clone = collector.recover(max_unknown_bits=0)
        self.assertIsNotNone(clone)
        self.assert_future_matches(source, clone)

    def test_partial_word_representation_and_compatibility_properties(self) -> None:
        partial = PartialWord32(0b101, 29)
        self.assertEqual(partial.v, 0b101)
        self.assertEqual(partial.n, 29)
        self.assertEqual(repr(partial), "0b101" + "x" * 29)
        self.assertTrue(partial.matches(0b101 << 29))
        self.assertTrue(partial.matches((0b101 << 29) | 12345))
        self.assertFalse(partial.matches(0b001 << 29))

        tiny = PartialWord32(0b1010101010101010101010101010101, 1)
        self.assertEqual(
            list(tiny.values),
            [tiny.known_msb << 1, (tiny.known_msb << 1) | 1],
        )
        self.assertIs(Recover.PartialInt32, PartialWord32)

    def test_insufficient_and_ambiguous_observations_do_not_guess(self) -> None:
        source = random.Random(101)
        insufficient = Recover()
        for _ in range(623):
            insufficient.feed_random_bits(32, source.getrandbits(32))
        self.assertIsNone(insufficient.recover())

        source = random.Random(202)
        ambiguous = Recover()
        ambiguous.feed_random_bits(63, source.getrandbits(63))
        for _ in range(622):
            ambiguous.feed_random_bits(32, source.getrandbits(32))
        with self.assertRaises(AmbiguousStateError):
            ambiguous.recover(max_unknown_bits=1)

    def test_equivalent_stored_states_return_the_same_future(self) -> None:
        source = random.Random(404)
        collector = Recover()
        collector.feed_random_bits(31, source.getrandbits(31))
        for _ in range(623):
            collector.feed_random_bits(32, source.getrandbits(32))

        clone = collector.recover(max_unknown_bits=1)
        self.assertIsNotNone(clone)
        self.assert_future_matches(source, clone)

    def test_effort_limit_and_inconsistent_data_are_explicit(self) -> None:
        source = random.Random(303)
        collector = Recover()
        collector.feed_random_bits(29, source.getrandbits(29))
        for _ in range(623 + 2):
            collector.feed_random_bits(32, source.getrandbits(32))

        with self.assertRaises(CandidateLimitError) as context:
            collector.recover(max_unknown_bits=2)
        self.assertEqual(context.exception.unknown_bits, 3)
        self.assertEqual(context.exception.candidates, 8)
        self.assertEqual(context.exception.limit, 2)

        corrupted = list(collector.chunks)
        corrupted[-1] ^= 1
        self.assertIsNone(
            collector.recover(bitBruteLimit=3, chunks=corrupted)
        )

    def test_invalid_inputs_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            clone_from_outputs([0] * 623)
        with self.assertRaises(ValueError):
            clone_from_outputs([0] * 623 + [1 << 32])
        with self.assertRaises(TypeError):
            temper(True)
        with self.assertRaises(ValueError):
            untemper(-1)
        with self.assertRaises(ValueError):
            PartialWord32(8, 29)
        with self.assertRaises(ValueError):
            PartialWord32(0, 32)

        collector = Recover()
        with self.assertRaises(ValueError):
            collector.feed_random_bits(-1, 0)
        with self.assertRaises(ValueError):
            collector.feed_random_bits(3, 8)
        with self.assertRaises(ValueError):
            collector.feed_random_bits(0, 1)
        with self.assertRaises(TypeError):
            collector.feed_random_bits(True, 0)
        collector.feed_random_bits(0, 0)
        self.assertEqual(collector.chunks, [])

        with self.assertRaises(TypeError):
            collector.recover(1, bitBruteLimit=1)


if __name__ == "__main__":
    unittest.main()
