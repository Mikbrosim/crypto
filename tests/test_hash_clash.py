"""Correctness tests for the HashClash process adapter."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import time
import unittest

from rewrite.hash_clash import (
    CollisionGenerationError,
    CollisionPair,
    HashClash,
    Hashclash,
    is_md5_collision,
)


class HashClashTests(unittest.TestCase):
    """Use the genuine short text collision published by HashClash."""

    MESSAGE1 = (
        b"TEXTCOLLBYfGiJUETHQ4hAcKSMd5zYpgqf1YRDhkmxHkhPWptrkoyz28wnI9V0aHeAuaKnak"
    )
    MESSAGE2 = (
        b"TEXTCOLLBYfGiJUETHQ4hEcKSMd5zYpgqf1YRDhkmxHkhPWptrkoyz28wnI9V0aHeAuaKnak"
    )
    DIGEST_HEX = "faad49866e9498fc1719f5289e7a0269"

    def setUp(self) -> None:
        self.temporary = TemporaryDirectory(prefix="hashclash-tests-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name) / "hashclash"
        self.scripts = self.root / "scripts"
        self.scripts.mkdir(parents=True)
        self.install_success_scripts()

    def write_script(self, name: str, body: str) -> None:
        (self.scripts / name).write_text(
            "#!/usr/bin/env bash\nset -eu\n" + body,
            encoding="utf-8",
        )

    def install_success_scripts(self) -> None:
        message1 = self.MESSAGE1.decode("ascii")
        message2 = self.MESSAGE2.decode("ascii")
        self.write_script(
            "cpc.sh",
            """
if [ "${MAXCPUS:-}" != "2" ]; then
    echo "MAXCPUS was not exported" >&2
    exit 9
fi
printf '%s' 'MESSAGE1' > "${1}.coll"
printf '%s' 'MESSAGE2' > "${2}.coll"
""".replace("MESSAGE1", message1).replace("MESSAGE2", message2),
        )
        self.write_script(
            "generic_ipc.sh",
            """
printf '%s' 'MESSAGE1' > collision1.bin
printf '%s' 'MESSAGE2' > collision2.bin
""".replace("MESSAGE1", message1).replace("MESSAGE2", message2),
        )
        self.write_script(
            "textcoll.sh",
            """
printf '%s' 'MESSAGE1' > final_collision1.txt
printf '%s' 'MESSAGE2' > final_collision2.txt
""".replace("MESSAGE1", message1).replace("MESSAGE2", message2),
        )

    def test_collision_fixture_and_result_properties(self) -> None:
        self.assertTrue(is_md5_collision(self.MESSAGE1, self.MESSAGE2))
        self.assertFalse(is_md5_collision(self.MESSAGE1, self.MESSAGE1))
        self.assertFalse(
            is_md5_collision(self.MESSAGE1, self.MESSAGE2[:-1] + b"b")
        )

        common_prefix = self.MESSAGE1[:20]
        pair = CollisionPair(
            bytearray(self.MESSAGE1),
            memoryview(self.MESSAGE2),
            common_prefix,
            common_prefix,
        )
        self.assertEqual(pair.hexdigest, self.DIGEST_HEX)
        self.assertEqual(pair.digest.hex(), self.DIGEST_HEX)
        self.assertEqual(pair.suffix1, self.MESSAGE1[len(common_prefix) :])
        self.assertEqual(pair.suffix2, self.MESSAGE2[len(common_prefix) :])
        first, second = pair
        self.assertEqual((first, second), (self.MESSAGE1, self.MESSAGE2))

        with self.assertRaisesRegex(ValueError, "same MD5"):
            CollisionPair(b"first", b"second")

    def test_chosen_prefix_workflow_and_cpu_limit(self) -> None:
        prefix1 = self.MESSAGE1[:22]
        prefix2 = self.MESSAGE2[:22]
        adapter = HashClash(self.root, max_cpus=2)

        pair = adapter.chosen_prefix_collision(prefix1, prefix2)

        self.assertEqual(pair.message1, self.MESSAGE1)
        self.assertEqual(pair.message2, self.MESSAGE2)
        self.assertEqual(pair.prefix1, prefix1)
        self.assertEqual(pair.prefix2, prefix2)
        self.assertEqual(pair.hexdigest, self.DIGEST_HEX)
        self.assertEqual(adapter.max_cpus, 2)
        self.assertIsNone(adapter.timeout)
        self.assertFalse((self.scripts / "prefix1.bin.coll").exists())
        self.assertFalse((self.scripts / "prefix2.bin.coll").exists())

    def test_identical_prefix_workflow_makes_padding_explicit(self) -> None:
        prefix = self.MESSAGE1[:19]
        pad_byte = self.MESSAGE1[19:20]
        adapter = HashClash(self.root)

        pair = adapter.identical_prefix_collision(
            prefix,
            pad_byte=pad_byte,
        )

        self.assertEqual(pair.message1, self.MESSAGE1)
        self.assertEqual(pair.message2, self.MESSAGE2)
        self.assertEqual(pair.prefix1, prefix)
        self.assertTrue(pair.suffix1.startswith(pad_byte))
        self.assertTrue(pair.suffix2.startswith(pad_byte))

    def test_text_collision_workflow(self) -> None:
        prefix = self.MESSAGE1[:20]
        adapter = HashClash(self.root)

        pair = adapter.text_collision(prefix)

        self.assertEqual(pair.message1, self.MESSAGE1)
        self.assertEqual(pair.message2, self.MESSAGE2)
        self.assertEqual(pair.prefix1, prefix)
        self.assertEqual(pair.hexdigest, self.DIGEST_HEX)

    def test_process_failure_includes_status_and_log_tail(self) -> None:
        self.write_script(
            "generic_ipc.sh",
            "echo 'deliberate tool failure'\nexit 7\n",
        )
        adapter = HashClash(self.root)

        with self.assertRaisesRegex(
            CollisionGenerationError,
            "status 7",
        ) as context:
            adapter.identical_prefix_collision(b"")

        self.assertEqual(context.exception.returncode, 7)
        self.assertIn("deliberate tool failure", context.exception.output)
        self.assertEqual(context.exception.command[0], "bash")

    def test_missing_invalid_and_prefix_losing_outputs_are_rejected(self) -> None:
        adapter = HashClash(self.root, max_cpus=2)
        self.write_script("textcoll.sh", ":\n")
        with self.assertRaisesRegex(
            CollisionGenerationError,
            "expected output",
        ):
            adapter.text_collision()

        self.write_script(
            "generic_ipc.sh",
            "printf '%s' one > collision1.bin\n"
            "printf '%s' two > collision2.bin\n",
        )
        with self.assertRaisesRegex(
            CollisionGenerationError,
            "invalid collision",
        ):
            adapter.identical_prefix_collision(b"")

        self.install_success_scripts()
        with self.assertRaisesRegex(
            CollisionGenerationError,
            "preserve",
        ):
            adapter.chosen_prefix_collision(b"wrong-one", b"wrong-two")

    def test_timeout_terminates_the_script(self) -> None:
        self.write_script("textcoll.sh", "sleep 5\n")
        adapter = HashClash(self.root, timeout=0.05)
        started = time.monotonic()

        with self.assertRaisesRegex(
            CollisionGenerationError,
            "timed out",
        ) as context:
            adapter.text_collision()

        self.assertLess(time.monotonic() - started, 2.0)
        self.assertIsNotNone(context.exception.returncode)

    def test_installation_configuration_and_input_validation(self) -> None:
        self.assertIs(Hashclash, HashClash)
        adapter = Hashclash(self.root)
        self.assertEqual(adapter.hashclash_path, self.root.resolve())
        self.assertEqual(adapter.scripts_path, self.scripts.resolve())

        with self.assertRaisesRegex(FileNotFoundError, "does not exist"):
            HashClash(self.root / "missing")
        empty_root = Path(self.temporary.name) / "empty"
        empty_root.mkdir()
        with self.assertRaisesRegex(FileNotFoundError, "scripts"):
            HashClash(empty_root)

        for timeout in (0, -1, float("inf")):
            with self.subTest(timeout=timeout):
                with self.assertRaises(ValueError):
                    HashClash(self.root, timeout=timeout)
        with self.assertRaisesRegex(TypeError, "not bool"):
            HashClash(self.root, max_cpus=True)
        with self.assertRaisesRegex(ValueError, "positive"):
            HashClash(self.root, max_cpus=0)
        with self.assertRaisesRegex(TypeError, "prefix"):
            adapter.identical_prefix_collision("not bytes")
        with self.assertRaisesRegex(ValueError, "exactly one"):
            adapter.identical_prefix_collision(b"prefix", pad_byte=b"xx")
        with self.assertRaisesRegex(TypeError, "message1"):
            is_md5_collision("not bytes", b"bytes")

        (self.scripts / "textcoll.sh").unlink()
        with self.assertRaisesRegex(FileNotFoundError, "textcoll.sh"):
            adapter.text_collision()


if __name__ == "__main__":
    unittest.main()
