"""Analytic invariants that do not depend on the Fortran reference outputs."""
import math
import json
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

from crmf.finite_nucleus import finite_setup as setup
from crmf.finite_nucleus import finite_solver as solver
from crmf.finite_nucleus.state import NucleusConfig, apply_config, new_state

FIXTURES = Path(__file__).resolve().parent / "fixtures"


class FiniteSolverTests(unittest.TestCase):
    def test_single_level_bcs_solution_and_particle_count(self):
        s = new_state()
        s.itx = 1
        s.nb = 1
        s.id[1, 1:3] = [1, 0]
        s.kb[1], s.mb[1] = -1, 2
        s.hh[1, 1], s.de[1, 1] = -5.0, 2.0
        s.tz[1], s.ala[1] = 0.6, -7.0
        solver.dirhb(s, 1)
        occupation = 0.3
        expected_lambda = -5 - 2 * (1 - 2 * occupation) / (2 * math.sqrt(occupation * (1 - occupation)))
        self.assertAlmostEqual(s.ala[1], expected_lambda, places=9)
        solver.denssh(s, 1)
        self.assertAlmostEqual(s.rosh[1, 1], 0.6, delta=2e-9)
        self.assertAlmostEqual(s.spk[1], math.sqrt(occupation * (1 - occupation)), delta=2e-9)
        uv = s.fguv[1:3, 1, 1]
        self.assertAlmostEqual(float(uv @ uv), 1.0, places=12)

    def test_radial_density_normalization_and_laplacian_integral(self):
        s = new_state(NucleusConfig(n0f=4, n0b=6))
        setup.default(s)
        apply_config(s, s.config)
        setup.forces(s, "new1")
        setup.gaush(s)
        setup.prep(s)
        setup.base(s)
        setup.gaupol(s)
        # One normalized occupied neutron 1s oscillator wave function.
        ib = next(i for i in range(1, s.nb + 1) if s.kb[i] == -1)
        s.rosh[1, ib] = 2.0
        solver.densit(s)
        self.assertAlmostEqual(float(s.rv[1:, 1] @ s.wdcor[1:]), 2.0, places=12)
        self.assertAlmostEqual(float(s.dro[1:, 2] @ s.wdcor[1:]), 0.0, places=12)
        np.testing.assert_allclose(s.rs[:, 1], s.rv[:, 1], atol=1e-14)

    def test_constraint_shifts_both_dirac_potentials(self):
        s = new_state()
        s.ipc = 0
        s.hbc = 197.328284
        s.rb[:] = np.linspace(0, 12, s.ngh + 1)
        s.xlambda = 0.002
        solver.poten(s)
        for it in (1, 2):
            np.testing.assert_allclose(s.vps[:, it], -s.xlambda * s.rb ** 2)
            np.testing.assert_allclose(s.vms[:, it], -s.xlambda * s.rb ** 2)

    def test_point_coupling_derivative_matches_finite_difference(self):
        s = new_state()
        s.ipc, s.dsat = 1, 0.152
        s.a_m[1:], s.b_m[1:], s.c_m[1:], s.d_m[1:] = 0.3, -0.7, 1.2, 0.8
        s.ro[:, 2] = np.linspace(0.01, 0.18, s.ngh + 1)
        solver.gdd(s)
        derivative = s.ff[:, 1, 2].copy()
        step = 1e-6
        s.ro[:, 2] += step
        solver.gdd(s)
        plus = s.ff[:, 1, 1].copy()
        s.ro[:, 2] -= 2 * step
        solver.gdd(s)
        minus = s.ff[:, 1, 1].copy()
        np.testing.assert_allclose(derivative, (plus - minus) / (2 * step), rtol=1e-8, atol=1e-9)

    def test_canonical_degeneracy_resolved_by_hamiltonian(self):
        angle = 0.43
        rotation = np.array([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]])
        h = np.array([[1.0, 0.8], [0.8, -2.0]])
        vectors = solver.degen(np.array([0.0, 0.0]), rotation, h)
        transformed = vectors.T @ h @ vectors
        self.assertAlmostEqual(transformed[0, 1], 0.0, places=12)
        np.testing.assert_allclose(np.diag(transformed), np.linalg.eigvalsh(h), atol=1e-12)

    def test_changed_nucleus_and_basis_fortran_regression(self):
        from crmf.finite_nucleus.nucleus import solve_nucleus
        config = NucleusConfig.from_file(FIXTURES / "ca40_n8_b12.dat")
        reference = np.loadtxt(FIXTURES / "ca40_n8_b12_results.txt", skiprows=1, usecols=range(1, 7))[0]
        s = solve_nucleus(config, "new1")
        values = [s.ea, s.radius_charge, s.radius_neutron - s.radius_proton,
                  s.els_2d_p, s.els_3p_n, s.els_2f_n]
        np.testing.assert_allclose(values, reference, atol=6e-6, rtol=0)

    def test_constrained_point_coupling_and_microscopic_com_fortran_regression(self):
        from crmf.finite_nucleus.nucleus import solve_nucleus
        fixture = json.loads((FIXTURES / "ca40_ddpc1_constrained.json").read_text())
        config = replace(NucleusConfig.from_file(FIXTURES / "ca40_n8_b12.dat"), constraint=0.01)
        s = solve_nucleus(config, "DD-PC1")
        values = np.r_[s.ea, s.radius_charge, s.radius_neutron, s.radius_proton,
                       s.radius_mass, s.els_2d_p, s.els_3p_n, s.els_2f_n,
                       s.ecmd[1:], s.ecmn[1:], s.ecm[1:]]
        np.testing.assert_allclose(values, list(fixture["observables"].values()), atol=2e-7, rtol=0)
        self.assertEqual(s.icm, 2)


if __name__ == "__main__":
    unittest.main()
