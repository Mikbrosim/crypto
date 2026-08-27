"""Correctness tests for hidden subset-sum recovery."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from sage.all import GF, Matrix, RealField, ZZ, vector

from rewrite.hidden_subset_sum import (
    NSattack,
    NZeroVectors,
    allones,
    allpmones,
    genParams,
    genpseudoprime,
    generate_instance,
    generate_pseudoprime,
    integer_right_kernel,
    kernelLLL,
    matNbits,
    matrix_bit_length,
    minimum_multivariate_samples,
    multiAttack_attack,
    multiAttack_gen,
    multivariate_attack,
    nguyen_stern_attack,
    orthoLattice,
    orthoLatticeMod,
    orthogonal_lattice,
    recoverBinary,
    recover_binary_vectors,
    recover_hidden_subset_sum,
    recover_multivariate_binary_vectors,
    roundM,
    round_matrix,
    statNS,
)


class HiddenSubsetSumTests(unittest.TestCase):
    """Exercise small instances where every returned equation is checkable."""

    MODULUS = 1_000_003
    BINARY_VECTORS = Matrix(
        ZZ,
        [
            [1, 0, 1, 0, 1, 0],
            [0, 1, 1, 0, 0, 1],
        ],
    )
    WEIGHTS = (12_345, 67_890)
    SAMPLES = (12_345, 67_890, 80_235, 0, 12_345, 67_890)

    def assert_solution(self, solution, method=None) -> None:
        """Verify rank, binary entries, and all public modular samples."""
        self.assertIsNotNone(solution)
        if method is not None:
            self.assertEqual(solution.method, method)
        self.assertEqual(solution.binary_vectors.dimensions(), (2, 6))
        self.assertEqual(solution.binary_vectors.rank(), 2)
        self.assertTrue(
            all(entry in (0, 1) for entry in solution.binary_vectors.list())
        )
        reconstructed = solution.binary_vectors.T * vector(ZZ, solution.weights)
        self.assertEqual(
            [int(value) % self.MODULUS for value in reconstructed],
            list(self.SAMPLES),
        )

    def test_generation_helpers_produce_valid_instances(self) -> None:
        for generator in (generate_pseudoprime, genpseudoprime):
            pseudoprime = generator(13, 6)
            self.assertGreaterEqual(pseudoprime.bit_length(), 11)

        for generator in (
            lambda: generate_instance(2, 4, 16),
            lambda: genParams(2, 4, 16),
        ):
            modulus, weights, binary_matrix, samples = generator()
            self.assertEqual(binary_matrix.dimensions(), (4, 2))
            self.assertEqual(binary_matrix.rank(), 2)
            self.assertEqual(
                [int(value) % modulus for value in binary_matrix * weights],
                list(samples),
            )

        modulus, samples, weights, binary_matrix = multiAttack_gen(2, nx0=16)
        self.assertEqual(len(samples), minimum_multivariate_samples(2))
        self.assertEqual(
            [int(value) % modulus for value in binary_matrix * weights],
            list(samples),
        )

    def test_orthogonal_lattices_satisfy_modular_equations(self) -> None:
        samples = [3, 5, 7]
        for constructor in (orthogonal_lattice, orthoLattice):
            basis = constructor(samples, 35)
            self.assertEqual(basis.dimensions(), (3, 3))
            self.assertEqual(abs(int(basis.det())), 35)
            self.assertTrue(
                all(
                    sum(int(x) * y for x, y in zip(row, samples, strict=True))
                    % 35
                    == 0
                    for row in basis.rows()
                )
            )

        no_unit_basis = orthogonal_lattice([6, 10], 15)
        self.assertEqual(abs(int(no_unit_basis.det())), 15)
        compressed = orthoLatticeMod(self.SAMPLES, 2, self.MODULUS)
        self.assertEqual(compressed.dimensions(), (6, 6))

    def test_integer_kernel_paths_return_the_exact_kernel(self) -> None:
        coefficients = Matrix(ZZ, [[1, 1, 1]])
        exact = integer_right_kernel(coefficients, method="exact")
        embedded = integer_right_kernel(coefficients, method="lll")
        legacy = kernelLLL(coefficients)
        for kernel in (exact, embedded, legacy):
            self.assertEqual(kernel.dimensions(), (2, 3))
            self.assertEqual(coefficients * kernel.T, 0)

    def test_binary_recovery_and_legacy_vector_helpers(self) -> None:
        difference_basis = Matrix(
            ZZ,
            [
                [1, -1, 1, -1],
                [0, 1, 0, 1],
            ],
        )
        for recovered in (
            recover_binary_vectors(difference_basis, 2),
            recoverBinary(difference_basis),
        ):
            self.assertEqual(recovered.rank(), 2)
            self.assertTrue(all(entry in (0, 1) for entry in recovered.list()))
        self.assertEqual(list(allones([0, -1, 0, -1])), [0, 1, 0, 1])
        self.assertIsNone(allones([0, 2]))
        self.assertTrue(allpmones([-1, 0, 1]))
        self.assertFalse(allpmones([-2, 0, 1]))

    def test_matrix_helpers_round_and_measure_entries(self) -> None:
        integer_matrix = Matrix(ZZ, [[0, -17], [3, 1]])
        self.assertEqual(matrix_bit_length(integer_matrix), 5)
        self.assertEqual(matNbits(integer_matrix), 5)
        real_field = RealField(100)
        real_matrix = Matrix(
            real_field,
            [[real_field("1.49"), real_field("-2.51")]],
        )
        expected = Matrix(ZZ, [[1, -3]])
        self.assertEqual(round_matrix(real_matrix), expected)
        self.assertEqual(roundM(real_matrix), expected)
        self.assertEqual(NZeroVectors([0, vector(ZZ, [0, 0]), 1]), 2)

    def test_multivariate_algebra_recovers_scrambled_binary_basis(self) -> None:
        binary_vectors = Matrix(
            ZZ,
            [[0, 0, 1, 1], [0, 1, 0, 1]],
        )
        change_of_basis = Matrix(GF(3), [[1, 1], [1, 2]])
        hidden_basis = change_of_basis * Matrix(GF(3), binary_vectors)

        recovered = recover_multivariate_binary_vectors(hidden_basis)

        self.assertIsNotNone(recovered)
        self.assertEqual(
            {tuple(map(int, row)) for row in recovered.rows()},
            {tuple(map(int, row)) for row in binary_vectors.rows()},
        )

    def test_nguyen_stern_and_dispatcher_solve_small_instance(self) -> None:
        solution = nguyen_stern_attack(
            2,
            self.MODULUS,
            self.SAMPLES,
            kernel_method="exact",
        )
        self.assert_solution(solution, "nguyen_stern")

        dispatched = recover_hidden_subset_sum(
            2,
            self.MODULUS,
            self.SAMPLES[:4],
            method="nguyen_stern",
            kernel_method="exact",
        )
        self.assertIsNotNone(dispatched)
        reconstructed = dispatched.binary_vectors.T * vector(
            ZZ,
            dispatched.weights,
        )
        self.assertEqual(
            [int(value) % self.MODULUS for value in reconstructed],
            list(self.SAMPLES[:4]),
        )

    def test_multivariate_attack_and_legacy_wrapper_solve_small_instance(self) -> None:
        solution = multivariate_attack(
            2,
            self.MODULUS,
            self.SAMPLES,
            kernel_method="exact",
        )
        self.assert_solution(solution, "multivariate")

        automatic = recover_hidden_subset_sum(
            2,
            self.MODULUS,
            self.SAMPLES,
        )
        self.assert_solution(automatic, "multivariate")

        legacy = multiAttack_attack(
            2,
            self.MODULUS,
            self.SAMPLES,
            False,
            False,
        )
        self.assertIsNotNone(legacy)
        binary_vectors, weights = legacy
        reconstructed = binary_vectors.T * weights
        self.assertEqual(
            [int(value) % self.MODULUS for value in reconstructed],
            list(self.SAMPLES),
        )

    def test_legacy_benchmark_wrappers_are_safe_and_return_results(self) -> None:
        instance = (
            self.MODULUS,
            vector(ZZ, self.WEIGHTS),
            self.BINARY_VECTORS.T,
            vector(ZZ, self.SAMPLES),
        )
        result = NSattack(2, instance=instance, verbose=False)
        self.assertIsNotNone(result)
        with self.assertRaisesRegex(ValueError, "supplied explicitly"):
            statNS()

        with patch(
            "rewrite.hidden_subset_sum.NSattack",
            return_value=result,
        ) as attack:
            benchmark = statNS([2])
        self.assertEqual(benchmark, [(2, result)])
        attack.assert_called_once_with(2, effort=1)


if __name__ == "__main__":
    unittest.main()
