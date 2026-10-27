import unittest

import numpy as np

from crmf.finite_nucleus.fock import (
    reconstruct_yukawa_kernel,
    yukawa_multipole_coefficients,
)


class YukawaMultipoleTests(unittest.TestCase):
    def test_multipoles_reconstruct_direct_yukawa_kernel(self):
        radii = np.array([0.35, 0.8, 1.2])
        cosine = np.array([-0.7, 0.0, 0.65])
        mass = 0.72
        coefficients = yukawa_multipole_coefficients(mass, radii, 80)
        reconstructed = reconstruct_yukawa_kernel(coefficients, cosine)
        separation = np.sqrt(
            radii[:, None, None] ** 2
            + radii[None, :, None] ** 2
            - 2 * radii[:, None, None] * radii[None, :, None] * cosine
        )
        direct = np.exp(-mass * separation) / (4 * np.pi * separation)
        # Equal radii converge slowly at the angular singularity. The
        # off-diagonal radial blocks have geometric convergence.
        off_diagonal = ~np.eye(radii.size, dtype=bool)
        np.testing.assert_allclose(
            reconstructed[off_diagonal], direct[off_diagonal],
            rtol=2e-9, atol=2e-11,
        )

    def test_zero_mass_limit_reconstructs_coulomb_kernel(self):
        radii = np.array([0.25, 0.6, 1.1])
        cosine = np.array([-0.4, 0.2, 0.9])
        coefficients = yukawa_multipole_coefficients(0.0, radii, 80)
        reconstructed = reconstruct_yukawa_kernel(coefficients, cosine)
        separation = np.sqrt(
            radii[:, None, None] ** 2
            + radii[None, :, None] ** 2
            - 2 * radii[:, None, None] * radii[None, :, None] * cosine
        )
        direct = 1.0 / (4 * np.pi * separation)
        off_diagonal = ~np.eye(radii.size, dtype=bool)
        np.testing.assert_allclose(
            reconstructed[off_diagonal], direct[off_diagonal],
            rtol=2e-11, atol=2e-12,
        )

    def test_radial_symmetry_and_validation(self):
        radii = np.array([0.2, 0.5, 1.0])
        result = yukawa_multipole_coefficients(0.4, radii, 4)
        np.testing.assert_allclose(result, result.transpose(0, 2, 1), atol=0.0)
        with self.assertRaises(ValueError):
            yukawa_multipole_coefficients(-1.0, radii, 4)
        with self.assertRaises(ValueError):
            yukawa_multipole_coefficients(0.4, np.array([0.0]), 1)


if __name__ == "__main__":
    unittest.main()
