"""Deterministic tests for CADO-NFS factoring and discrete-log adapters."""

from __future__ import annotations

import os
from pathlib import Path
from tempfile import TemporaryDirectory
import time
import unittest

from rewrite.cado import (
    CadoExecutionError,
    CadoOutputError,
    dlp,
    factor,
    pohlig_hellman,
)
from rewrite import cado as cado_module


_FAKE_CADO = r'''#!/usr/bin/env python3
from pathlib import Path
import subprocess
import sys
import time

Path("arguments.txt").write_text("\n".join(sys.argv[1:]))

if "-dlp" in sys.argv:
    target_arg = next(arg for arg in sys.argv if arg.startswith("target="))
    generator, target = map(int, target_arg.split("=", 1)[1].split(","))
    if target == 8:
        print("CADO diagnostic")
        print("4,1")
    elif target == 16:
        print("0,1")
    else:
        print("4,1")
    raise SystemExit(0)

value = int(sys.argv[1])
if value == 315:
    print("CADO diagnostic")
    print("3  3\t5 7")
elif value == 45:
    print("3 15")
elif value == 143:
    print("11 17")
elif value == 221:
    print("malformed result")
elif value == 299:
    print("deliberate CADO failure")
    raise SystemExit(9)
elif value == 323:
    child = subprocess.Popen(["sleep", "30"])
    Path("child.pid").write_text(str(child.pid))
    time.sleep(30)
else:
    print(str(value))
'''


class CadoAdapterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.executable = self.root / "fake-cado-nfs.py"
        self.executable.write_text(_FAKE_CADO)
        self.executable.chmod(0o755)

    def test_small_local_factorization_preserves_multiplicity(self) -> None:
        self.assertEqual(factor(315), [3, 3, 5, 7])
        self.assertEqual(factor(101), [101])

    def test_cado_factor_output_and_workdir_are_verified(self) -> None:
        workdir = self.root / "factor-work"
        factors = factor(
            315,
            cado_path=self.executable,
            threads=4,
            backend="cado",
            workdir=workdir,
        )
        self.assertEqual(factors, [3, 3, 5, 7])
        arguments = (workdir / "arguments.txt").read_text().splitlines()
        self.assertIn(f"workdir={workdir.resolve()}", arguments)
        self.assertNotIn(f"--workdir={workdir.resolve()}", arguments)
        self.assertIn("-t", arguments)
        self.assertIn("4", arguments)

    def test_invalid_factor_output_and_process_failures_are_explicit(self) -> None:
        with self.assertRaises(CadoOutputError):
            factor(143, cado_path=self.executable, backend="cado")
        with self.assertRaises(CadoOutputError) as composite:
            factor(45, cado_path=self.executable, backend="cado")
        self.assertIn("composite factor", str(composite.exception))
        with self.assertRaises(CadoOutputError):
            factor(221, cado_path=self.executable, backend="cado")

        with self.assertRaises(CadoExecutionError) as failed:
            factor(299, cado_path=self.executable, backend="cado")
        self.assertEqual(failed.exception.returncode, 9)
        self.assertIn("deliberate CADO failure", failed.exception.output)

    def test_timeout_terminates_cado_workers(self) -> None:
        workdir = self.root / "timeout-work"
        with self.assertRaises(CadoExecutionError) as context:
            factor(
                323,
                cado_path=self.executable,
                backend="cado",
                timeout=0.1,
                workdir=workdir,
            )
        self.assertTrue(context.exception.timed_out)
        child_pid = int((workdir / "child.pid").read_text())
        for _ in range(20):
            try:
                os.kill(child_pid, 0)
            except ProcessLookupError:
                break
            time.sleep(0.01)
        else:
            self.fail("CADO worker survived the timeout")

    def test_local_dlp_verifies_direct_and_projected_subgroups(self) -> None:
        self.assertEqual(dlp(2, pow(2, 7, 23), 23, 11), 7)

        generator = 5
        exponent = 9
        target = pow(generator, exponent, 23)
        self.assertEqual(dlp(generator, target, 23, 11), exponent % 11)

    def test_cado_arbitrary_base_logs_are_converted_and_verified(self) -> None:
        workdir = self.root / "dlp-work"
        logarithm = dlp(
            2,
            8,
            23,
            11,
            workdir=workdir,
            cado_path=self.executable,
            backend="cado",
            threads=2,
        )
        self.assertEqual(logarithm, 3)
        arguments = (workdir / "arguments.txt").read_text().splitlines()
        self.assertIn("target=2,8", arguments)
        self.assertIn(f"workdir={workdir.resolve()}", arguments)

        with self.assertRaises(CadoOutputError) as noninvertible:
            dlp(
                2,
                16,
                23,
                11,
                cado_path=self.executable,
                backend="cado",
            )
        self.assertIn("non-invertible", str(noninvertible.exception))

        with self.assertRaises(CadoOutputError) as incorrect:
            dlp(
                2,
                4,
                23,
                11,
                cado_path=self.executable,
                backend="cado",
            )
        self.assertIn("verification", str(incorrect.exception))

    def test_pohlig_hellman_handles_repeated_prime_factors(self) -> None:
        modulus = 97
        generator = 5
        order_factors = [2, 2, 2, 2, 2, 3]
        exponent = 37
        target = pow(generator, exponent, modulus)
        recovered = pohlig_hellman(
            generator,
            target,
            modulus,
            order_factors,
        )
        self.assertEqual(recovered, exponent)
        self.assertEqual(pow(generator, recovered, modulus), target)

    def test_invalid_group_descriptions_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            dlp(2, 8, 21, 5)
        with self.assertRaises(ValueError):
            dlp(2, 8, 23, 10)
        with self.assertRaises(ValueError):
            dlp(1, 1, 23, 11)
        with self.assertRaises(ValueError):
            pohlig_hellman(5, 1, 97, [])
        with self.assertRaises(ValueError):
            pohlig_hellman(5, 1, 97, [4, 3])
        with self.assertRaises(ValueError):
            pohlig_hellman(pow(5, 2, 97), 1, 97, [2, 2, 2, 2, 2, 3])

    def test_invalid_backend_and_resource_configuration(self) -> None:
        self.assertEqual(cado_module.cado_path, "cado-nfs.py")
        self.assertGreaterEqual(cado_module.num_cpu_threads, 1)
        with self.assertRaises(ValueError):
            factor(1)
        with self.assertRaises(ValueError):
            factor(15, backend="guess")
        with self.assertRaises(TypeError):
            factor(15, force_sage=True, backend="cado")
        with self.assertRaises(ValueError):
            factor(
                15,
                cado_path=self.executable,
                backend="cado",
                threads=0,
            )
        with self.assertRaises(ValueError):
            dlp(2, 8, 23, 11, backend="cado", timeout=0)
        with self.assertRaises(CadoExecutionError):
            factor(
                15,
                cado_path=self.root / "missing",
                backend="cado",
            )


if __name__ == "__main__":
    unittest.main()
