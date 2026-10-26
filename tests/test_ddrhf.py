import unittest

import numpy as np

from crmf.nuclear_matter.ddrhf import (
    PKO_PARAMETERS,
    density_dependent_couplings,
    nuclear_matter_properties,
    solve_nuclear_matter,
)
from crmf.nuclear_matter.ddrhf import _solve_matter


class DDRHFParameterTests(unittest.TestCase):
    def test_pko_sets_reproduce_the_published_vertex_values_at_saturation(self):
        expected = {
            "PKO1": (8.833239, 10.729933, 2.629 * __import__("math").exp(-0.076760),
                     __import__("math").exp(-1.231976)),
            "PKO2": (8.920597, 10.550553, 4.068299 * __import__("math").exp(-0.631605), 0.0),
            "PKO3": (8.895635, 10.802690, 3.832480 * __import__("math").exp(-0.635336),
                     __import__("math").exp(-0.934122)),
        }
        for name, values in expected.items():
            couplings = density_dependent_couplings(
                PKO_PARAMETERS[name], PKO_PARAMETERS[name].saturation_density_fm3
            )
            self.assertAlmostEqual(couplings.g_sigma, values[0], places=5)
            self.assertAlmostEqual(couplings.g_omega, values[1], places=5)
            self.assertAlmostEqual(couplings.g_rho, values[2], places=5)
            self.assertAlmostEqual(couplings.f_pi, values[3], places=5)

    def test_invalid_density_and_unknown_parameter_set_are_rejected(self):
        parameters = PKO_PARAMETERS["PKO1"]
        for density in (0.0, -0.1, float("nan")):
            with self.assertRaises(ValueError):
                solve_nuclear_matter(density, parameters=parameters)
        with self.assertRaises(KeyError):
            PKO_PARAMETERS["not-a-pko-set"]


class DDRHFNuclearMatterTests(unittest.TestCase):
    def test_pko_sets_reproduce_symmetric_matter_table_ii(self):
        expected = {
            "PKO1": (250.24, 0.1520, 34.371, -15.996, 0.5900, 0.7459, 0.7272),
            "PKO2": (249.60, 0.1510, 32.492, -16.027, 0.6025, 0.7636, 0.7447),
            "PKO3": (262.47, 0.1530, 32.987, -16.041, 0.5862, 0.7416, 0.7229),
        }
        for name, reference in expected.items():
            with self.subTest(parameter_set=name):
                result = nuclear_matter_properties(PKO_PARAMETERS[name])
                observed = (
                    result.incompressibility_MeV,
                    result.saturation_density_fm3,
                    result.symmetry_energy_MeV,
                    result.binding_energy_per_particle_MeV,
                    result.dirac_mass_ratio,
                    result.nonrelativistic_mass_ratio,
                    result.relativistic_mass_ratio,
                )
                tolerances = (1.0, 0.0005, 0.25, 0.05, 0.005, 0.006, 0.003)
                for value, target, tolerance in zip(observed, reference, tolerances):
                    self.assertAlmostEqual(value, target, delta=tolerance)
                self.assertAlmostEqual(result.pressure_at_saturation_MeV_fm3, 0.0, delta=1e-5)

    def test_asymmetric_energy_is_even_in_isospin_asymmetry(self):
        parameters = PKO_PARAMETERS["PKO1"]
        positive = solve_nuclear_matter(parameters.saturation_density_fm3, 0.2, parameters)
        negative = solve_nuclear_matter(parameters.saturation_density_fm3, -0.2, parameters)
        self.assertAlmostEqual(
            positive.binding_energy_per_particle_MeV,
            negative.binding_energy_per_particle_MeV,
            places=7,
        )

    def test_fermi_energy_satisfies_thermodynamic_derivative(self):
        parameters = PKO_PARAMETERS["PKO1"]
        density = 0.14
        step = 1e-5
        state = _solve_matter(density, 0.0, parameters)
        above = solve_nuclear_matter(density + step, parameters=parameters)
        below = solve_nuclear_matter(density - step, parameters=parameters)
        binding = state.result.binding_energy_per_particle_MeV
        derivative = (
            above.binding_energy_per_particle_MeV
            - below.binding_energy_per_particle_MeV
        ) / (2 * step)
        thermodynamic_mu = 938.9 + binding + density * derivative
        self.assertTrue(np.allclose(state.fermi_energies_MeV, thermodynamic_mu, atol=2e-5))


if __name__ == "__main__":
    unittest.main()
