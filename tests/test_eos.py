"""Numerical regressions against the supplied Fortran calculation."""
import unittest
from pathlib import Path
from dataclasses import replace

import numpy as np

from crmf.nuclear_matter.eos import (
    default_parameters, equation_of_state, solve_fields, field_residuals,
    symmetry_energy, symmetry_slope, incompressibility, energy_per_particle,
    beta_equilibrium,
    scalar_densities, chemical_potentials, energy_density, pressure,
    MesonFields, calibrate_isoscalar, skewness, calibrate_with_skewness,
    funcv, RMFParameters,
)


class EOSTests(unittest.TestCase):
    def test_fortran_reference_rows(self):
        table = equation_of_state()
        self.assertEqual(table.shape, (2001, 8))
        reference = np.array([
            [0.100000, -14.8729, 0.301148, 0.179783, 0., -0.0148729, 0.0255162, 0.0548073],
            [0.300000, -3.90332, 0.639094, 0.418581, 0., -0.00390334, 0.0310662, -0.0991102],
        ])
        np.testing.assert_allclose(table[[0, -1]], reference, rtol=5e-6, atol=5e-8)

    def test_entire_fortran_eos_table(self):
        lines = (Path(__file__).parent/'fixtures'/'EOS.OUT').read_text().splitlines()
        reference = np.array([[float(line[i:i+13]) for i in range(0, 104, 13)] for line in lines])
        actual = equation_of_state()
        np.testing.assert_allclose(actual, reference, rtol=5e-6, atol=1e-8)

    def test_independent_fortran_physics_values(self):
        # Original source compiled with a test driver printing ES25.16;
        # these test asymmetric fields without depending on either solver.
        p = default_parameters()
        _, symmetric = calibrate_isoscalar(p)
        f = MesonFields(symmetric.sigma, symmetric.omega, -.02, -.01)
        n = p.saturation_density
        actual = [*scalar_densities(n, f, .6, p), *chemical_potentials(n, f, .6, p),
                  energy_density(n, f, .6, p), pressure(n, f, .6, p), skewness(n, symmetric, p)]
        np.testing.assert_allclose(actual,
            [0.10974316662046563, 0.028703516457518789,
             0.94910038892621440, 0.89120677566524997,
             0.13726506430022867, 0.00068185233453358762, -0.32211377884392034],
            atol=2e-12, rtol=2e-12)

    def test_skewness_calibration(self):
        for method in ('nonlinear', 'analytic'):
            p = calibrate_with_skewness(default_parameters(), -300.0, method=method)
            fields = solve_fields(p.saturation_density, parameters=p)
            self.assertAlmostEqual(skewness(p.saturation_density, fields, p)*1000, -300.0, places=3)
            self.assertAlmostEqual(incompressibility(p.saturation_density, fields, p)*1000,
                                   p.incompressibility, places=3)

    def test_original_residual_modes(self):
        p = default_parameters()
        n = p.saturation_density
        f = solve_fields(n, parameters=p)
        np.testing.assert_allclose(funcv(3, [f.sigma], n, parameters=p), 0, atol=1e-9)
        np.testing.assert_allclose(funcv(4, [f.omega, f.rho], n, parameters=p), 0, atol=1e-9)
        np.testing.assert_allclose(funcv(6, [p.g_w*f.omega], n, parameters=p), 0, atol=1e-9)
        with self.assertRaises(ValueError):
            funcv(8, [f.sigma, f.omega, .5], n, parameters=p)

    def test_saturation_targets(self):
        p = default_parameters()
        n = p.saturation_density
        f = solve_fields(n, parameters=p)
        self.assertAlmostEqual(energy_per_particle(n, f, parameters=p)*1000, p.binding_energy, places=4)
        self.assertAlmostEqual(incompressibility(n, f, p)*1000, p.incompressibility, places=4)
        self.assertAlmostEqual(symmetry_energy(n, f, p)*1000, p.symmetry_energy, places=4)
        self.assertAlmostEqual(symmetry_slope(n, f, p)*1000, p.symmetry_slope, places=4)

    def test_asymmetric_field_equations_and_isospin(self):
        p = default_parameters()
        for n in (0.05, 0.148, 0.3):
            fields = solve_fields(n, 0.6, p)
            np.testing.assert_allclose(field_residuals(fields, n, 0.6, p), 0, atol=2e-9)
            mirror = solve_fields(n, -0.6, p)
            np.testing.assert_allclose(tuple(mirror), np.array(tuple(fields))*[1, 1, -1, -1], atol=1e-9)

    def test_beta_equilibrium_charge_and_chemical_balance(self):
        result = beta_equilibrium(0.148)
        self.assertAlmostEqual((1-result.asymmetry)/2,
                               result.electron_fraction+result.muon_fraction, places=9)
        self.assertGreater(result.energy_density, 0)
        self.assertTrue(np.isfinite(result.pressure))

    def test_beta_equilibrium_brackets_physical_high_density_branch(self):
        parameters = replace(
            RMFParameters(), g_s=10.2143, g_w=13.4353, g_r=6.98518,
            g_d=6.2, b_s=0.00159524/0.9390, c_s=0.000540269,
            c_w=0.00528340, xl_v=0.0200112, c_ds=0.054,
        )
        result = beta_equilibrium(0.8, parameters)
        self.assertAlmostEqual(
            (1-result.asymmetry)/2,
            result.electron_fraction+result.muon_fraction,
            places=8,
        )
        self.assertTrue(np.isfinite(result.energy_density))

    def test_invalid_density(self):
        for density in (-1, 0, np.nan):
            with self.assertRaises(ValueError):
                solve_fields(density)

    def test_invalid_physics_inputs(self):
        p = default_parameters()
        for alpha in (-1.01, 1.01, np.nan):
            with self.assertRaises(ValueError):
                solve_fields(.148, alpha, p)
        with self.assertRaises(ValueError):
            replace(p, xms=-.5)
        with self.assertRaises(ValueError):
            replace(p, effective_mass_ratio=0)
        with self.assertRaises(ValueError):
            symmetry_energy(.148, MesonFields(np.nan, 0), p)


if __name__ == "__main__":
    unittest.main()
