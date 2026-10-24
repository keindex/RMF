import json
import unittest
from pathlib import Path

import numpy as np

from crmf.math_utils.eigensolver import sdiag


class EigensolverTests(unittest.TestCase):
    def test_original_fortran_reference(self):
        path = Path(__file__).resolve().parent / "fixtures" / "sdiag_reference.json"
        reference = json.loads(path.read_text())
        values, vectors = sdiag(reference["matrix"])
        np.testing.assert_allclose(values, reference["eigenvalues"], atol=2e-11, rtol=0)
        np.testing.assert_allclose(vectors, reference["eigenvectors"], atol=2e-10, rtol=0)

    def test_orthogonality_residual_and_order(self):
        rng = np.random.default_rng(818)
        for n in (1, 2, 10, 44):
            with self.subTest(n=n):
                matrix = rng.normal(size=(n, n))
                matrix = matrix + matrix.T
                values, vectors = sdiag(matrix)
                np.testing.assert_allclose(vectors.T @ vectors, np.eye(n), atol=2e-13, rtol=0)
                np.testing.assert_allclose(matrix @ vectors, vectors * values, atol=1e-8, rtol=0)
                np.testing.assert_allclose(values, np.linalg.eigvalsh(matrix), atol=1e-10, rtol=0)
                maxima = vectors[np.abs(vectors).argmax(axis=0), np.arange(n)]
                self.assertTrue(np.all(maxima >= 0))
                self.assertTrue(np.all(np.diff(values) >= 0))

    def test_degenerate_zero_and_diagonal_matrices(self):
        for matrix in (np.zeros((8, 8)), np.eye(8), np.diag([3.0, -2.0, -2.0, 0.0])):
            values, vectors = sdiag(matrix)
            np.testing.assert_allclose(matrix @ vectors, vectors * values, atol=1e-14)
            np.testing.assert_allclose(vectors.T @ vectors, np.eye(matrix.shape[0]), atol=1e-14)

    def test_invalid_input(self):
        with self.assertRaises(ValueError):
            sdiag([[1, 2]])
        with self.assertRaises(ValueError):
            sdiag([[np.nan]])


if __name__ == "__main__":
    unittest.main()
