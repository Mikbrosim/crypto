"""Deterministic tests for the verified YAFU subprocess adapter."""

from __future__ import annotations

import os
from pathlib import Path
from tempfile import TemporaryDirectory
import time
import unittest

from rewrite.yafu import (
    IncompleteFactorizationError,
    YafuExecutionError,
    YafuOutputError,
    factor,
    factor_cmd,
    rho,
    run_yafu,
)


_FAKE_YAFU = r'''#!/usr/bin/env python3
import os
from pathlib import Path
import subprocess
import sys
import time

expression = sys.argv[1]
Path("factor.json").write_text("fake artifact")
Path("arguments.txt").write_text("\n".join(sys.argv[1:]))

if "315" in expression:
    print("factor = 3")
    print("***factors found***")
    print("P1 = 3")
    print("P1 = 3")
    print("P1 = 5")
    print("P1 = 7")
elif expression == "rho(105)":
    print("rho: factor = 3")
elif expression == "factor(105)":
    print("***factors found***")
    print("P1 = 3")
    print("C2 = 35")
elif expression == "factor(143)":
    print("***factors found***")
    print("P2 = 11")
    print("P2 = 17")
elif expression == "factor(221)":
    print("nothing useful here")
elif expression == "factor(299)":
    print("deliberate failure")
    raise SystemExit(7)
elif expression == "factor(323)":
    child = subprocess.Popen(["sleep", "30"])
    Path("child.pid").write_text(str(child.pid))
    time.sleep(30)
else:
    print("rho: factor = 3")
'''


class YafuAdapterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.executable = self.root / "fake-yafu"
        self.executable.write_text(_FAKE_YAFU)
        self.executable.chmod(0o755)

    def test_complete_factorization_and_result_properties(self) -> None:
        workdir = self.root / "complete-work"
        result = run_yafu(
            315,
            yafu_path=self.executable,
            threads=3,
            workdir=workdir,
        )
        self.assertEqual(result.components, (3, 3, 5, 7))
        self.assertEqual(result.factors, result.components)
        self.assertTrue(result.complete)
        self.assertEqual(result.unresolved, ())
        self.assertEqual(result.input_value, 315)
        self.assertEqual(result.method, "factor")
        self.assertEqual(result.working_directory, workdir.resolve())
        self.assertIn("-threads", result.command)
        self.assertIn("3", result.command)
        self.assertEqual(factor(315, yafu_path=self.executable), [3, 3, 5, 7])

    def test_rho_preserves_a_verified_composite_cofactor(self) -> None:
        result = run_yafu(105, method="rho", yafu_path=self.executable)
        self.assertEqual(result.components, (3, 35))
        self.assertFalse(result.complete)
        self.assertEqual(result.unresolved, (35,))
        self.assertEqual(rho(105, yafu_path=self.executable), [3, 35])

    def test_incomplete_factor_command_fails_explicitly(self) -> None:
        with self.assertRaises(IncompleteFactorizationError) as context:
            factor(105, yafu_path=self.executable)
        self.assertEqual(context.exception.result.components, (3, 35))
        self.assertFalse(context.exception.result.complete)

    def test_factor_cmd_supports_legacy_bytes_and_explicit_inputs(self) -> None:
        self.assertEqual(
            factor_cmd(b"factor(315)", yafu_path=self.executable),
            [3, 3, 5, 7],
        )
        self.assertEqual(
            factor_cmd(
                "ecm(315)",
                composite=315,
                yafu_path=self.executable,
            ),
            [3, 3, 5, 7],
        )
        with self.assertRaises(ValueError):
            factor_cmd("ecm(315)", yafu_path=self.executable)
        with self.assertRaises(ValueError):
            factor_cmd(
                "factor(315)",
                composite=105,
                yafu_path=self.executable,
            )

    def test_dishonest_and_malformed_output_are_rejected(self) -> None:
        with self.assertRaises(YafuOutputError) as dishonest:
            run_yafu(143, yafu_path=self.executable)
        self.assertIn("do not multiply", str(dishonest.exception))
        self.assertTrue(dishonest.exception.command)

        with self.assertRaises(YafuOutputError):
            run_yafu(221, yafu_path=self.executable)

    def test_nonzero_exit_and_timeout_include_diagnostics(self) -> None:
        with self.assertRaises(YafuExecutionError) as failed:
            run_yafu(299, yafu_path=self.executable)
        self.assertEqual(failed.exception.returncode, 7)
        self.assertIn("deliberate failure", failed.exception.output)
        self.assertFalse(failed.exception.timed_out)

        timeout_workdir = self.root / "timeout-work"
        with self.assertRaises(YafuExecutionError) as timed_out:
            run_yafu(
                323,
                yafu_path=self.executable,
                timeout=0.1,
                workdir=timeout_workdir,
            )
        self.assertTrue(timed_out.exception.timed_out)
        child_pid = int((timeout_workdir / "child.pid").read_text())
        for _ in range(20):
            try:
                os.kill(child_pid, 0)
            except ProcessLookupError:
                break
            time.sleep(0.01)
        else:
            self.fail("YAFU child process survived the timeout")

    def test_workspace_cleanup_never_deletes_caller_files(self) -> None:
        automatic = run_yafu(315, yafu_path=self.executable)
        self.assertFalse(automatic.working_directory.exists())

        explicit = self.root / "caller-work"
        explicit.mkdir()
        sentinel = explicit / "factor.log"
        sentinel.write_text("keep me")
        run_yafu(315, yafu_path=self.executable, workdir=explicit)
        self.assertEqual(sentinel.read_text(), "keep me")
        self.assertTrue((explicit / "factor.json").is_file())

    def test_invalid_configuration_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            run_yafu(1, yafu_path=self.executable)
        with self.assertRaises(ValueError):
            run_yafu(15, method="ecm", yafu_path=self.executable)
        with self.assertRaises(ValueError):
            run_yafu(15, threads=0, yafu_path=self.executable)
        with self.assertRaises(ValueError):
            run_yafu(15, timeout=0, yafu_path=self.executable)
        with self.assertRaises(TypeError):
            run_yafu(15, cleanup=1, yafu_path=self.executable)
        with self.assertRaises(FileNotFoundError):
            run_yafu(15, yafu_path=self.root / "missing")
        with self.assertRaises(TypeError):
            factor(
                315,
                yafuPath=self.executable,
                yafu_path=self.executable,
            )


if __name__ == "__main__":
    unittest.main()
