"""Check the documented formulas against the current Python implementation.

Run from any directory: python docs/verify_derivation.py
Only the documentation check report is written; calculation output is untouched.
"""
from __future__ import annotations

import argparse
import json
from math import factorial, gamma, pi, sqrt
from pathlib import Path
import sys

import numpy as np
from scipy.integrate import quad
from scipy.special import eval_genlaguerre

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from crmf.nuclear_matter import eos
from crmf.finite_nucleus import finite_setup, pairing
from crmf.math_utils.eigensolver import sdiag
from crmf.finite_nucleus.state import NucleusConfig, apply_config, new_state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    saved_results = ROOT / "output" / "results.json"
    if not saved_results.exists():
        saved_results = ROOT / "python_output" / "results.json"
    parser.add_argument("--results", type=Path, default=saved_results)
    args = parser.parse_args()
    checks = []

    def check(name, actual, expected, *, atol=1e-10, rtol=1e-10):
        actual, expected = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
        if not np.isfinite(actual).all() or not np.isfinite(expected).all():
            raise AssertionError(f"{name}: nonfinite value")
        np.testing.assert_allclose(actual, expected, atol=atol, rtol=rtol, err_msg=name)
        checks.append(dict(name=name, passed=True, actual=actual.tolist(),
                           expected=expected.tolist(),
                           max_abs_error=float(np.max(np.abs(actual - expected))),
                           absolute_tolerance=atol, relative_tolerance=rtol))

    p = eos.default_parameters()
    n = p.saturation_density
    fields = eos.solve_fields(n, parameters=p)
    mass, _ = eos.effective_masses(fields, p)
    k = eos.fermi_momentum(n / 2)
    ef = sqrt(k * k + mass * mass)
    factor = 1 / (eos.PI2 * eos.HBAR3)

    # Direct quadrature provides a check independent of the closed-form integrals.
    scalar_integral = factor * quad(lambda q: mass * q * q / sqrt(q * q + mass * mass),
                                    0, k, epsabs=1e-13)[0]
    kinetic_integral = factor * quad(lambda q: q * q * sqrt(q * q + mass * mass),
                                     0, k, epsabs=1e-13)[0]
    pressure_integral = factor / 3 * quad(lambda q: q**4 / sqrt(q * q + mass * mass),
                                         0, k, epsabs=1e-13)[0]
    check("scalar_density_integral", eos._scalar(k, mass), scalar_integral)
    check("kinetic_energy_integral", eos._kinetic_energy(k, mass), kinetic_integral)
    check("kinetic_pressure_integral", eos._kinetic_pressure(k, mass), pressure_integral)
    n_from_k = k**3 / (3 * eos.PI2 * eos.HBAR3)
    check("kinetic_pressure_identity", pressure_integral, n_from_k * ef - kinetic_integral)
    check("kinetic_energy_scalar_identity", kinetic_integral,
          (3 * n_from_k * ef + mass * scalar_integral) / 4)
    check("symmetric_field_equations", eos.field_residuals(fields, n, parameters=p),
          np.zeros(4), atol=1e-8)

    values = np.array([eos.energy_per_particle(n, fields, parameters=p),
                       eos.incompressibility(n, fields, p),
                       eos.symmetry_energy(n, fields, p),
                       eos.symmetry_slope(n, fields, p)]) * 1000
    check("saturation_targets_MeV", values,
          [p.binding_energy, p.incompressibility, p.symmetry_energy, p.symmetry_slope],
          atol=2e-5, rtol=0)

    dn = 1e-5
    low, high = (eos.solve_fields(x, parameters=p) for x in (n - dn, n + dn))
    slope_fd = 3 * n * (eos.symmetry_energy(n + dn, high, p)
                         - eos.symmetry_energy(n - dn, low, p)) / (2 * dn)
    k_fd = 9 * (eos.pressure(n + dn, high, parameters=p)
                 - eos.pressure(n - dn, low, parameters=p)) / (2 * dn)
    check("symmetry_slope_density_derivative_MeV", values[3], 1000 * slope_fd, atol=1e-4, rtol=0)
    check("incompressibility_pressure_derivative_MeV", values[1], 1000 * k_fd, atol=1e-3, rtol=0)
    da = 1e-3
    plus, minus = (eos.solve_fields(n, a, p) for a in (da, -da))
    symmetry_fd = (eos.energy_per_particle(n, plus, da, p)
                   + eos.energy_per_particle(n, minus, -da, p)
                   - 2 * eos.energy_per_particle(n, fields, parameters=p)) / (2 * da**2)
    check("symmetry_energy_asymmetry_derivative_MeV", values[2], 1000 * symmetry_fd,
          atol=2e-4, rtol=0)

    alpha = 0.6
    asymmetric_fields = eos.solve_fields(n, alpha, p)
    mu_n, mu_p = eos.chemical_potentials(n, asymmetric_fields, alpha, p)
    thermo_pressure = n * ((1 + alpha) * mu_n + (1 - alpha) * mu_p) / 2
    thermo_pressure -= eos.energy_density(n, asymmetric_fields, alpha, p)
    pressure_code = eos.pressure(n, asymmetric_fields, alpha, p)
    s = p.g_s * asymmetric_fields.sigma * eos.HBAR15
    d = p.g_d * asymmetric_fields.delta * eos.HBAR15
    check("documented_asymmetric_pressure_difference_MeV_fm3",
          1000 * (pressure_code - thermo_pressure),
          -1000 * 3 * p.c_ds * s * s * d * d / (4 * eos.HBAR3), atol=2e-5, rtol=0)

    table = eos.equation_of_state(p, [n])
    check("EOS_energy_columns_offset_MeV", table[0, 1] - 1000 * table[0, 5],
          1000 * eos.XM - 939)
    table_fields = eos.MesonFields(*table[0, 2:5], 0.0)
    s = p.g_s * table_fields.sigma * eos.HBAR15
    w = p.g_w * table_fields.omega * eos.HBAR15
    table_mass = eos.XM - s
    scalar_potential = p.xms**2 * s*s / (2 * p.g_s**2)
    scalar_potential += p.b_s * eos.XM * s**3 / 3 + p.c_s * s**4 / 4
    vector_potential = p.xmw**2 * w*w / (2 * p.g_w**2) + p.c_w * w**4 / 4
    energy_before_elimination = 2 * eos._kinetic_energy(k, table_mass)
    energy_before_elimination += (scalar_potential - vector_potential) / eos.HBAR3 + n * w
    vector_residual = p.xmw**2 * w / p.g_w**2 + p.c_w * w**3 - eos.HBAR3 * n
    check("approximate_field_energy_elimination_identity_MeV_fm3",
          1000 * (eos.energy_density(n, table_fields, parameters=p) - energy_before_elimination),
          1000 * w * vector_residual / eos.HBAR3)
    beta = eos.beta_equilibrium(n, p)
    check("beta_equilibrium_charge_neutrality", (1 - beta.asymmetry) / 2,
          beta.electron_fraction + beta.muon_fraction, atol=1e-9)

    state = new_state(NucleusConfig(n0f=4, n0b=6))
    finite_setup.default(state)
    apply_config(state, state.config)
    finite_setup.forces(state, "new1")
    finite_setup.gaush(state)
    finite_setup.prep(state)
    finite_setup.base(state)
    finite_setup.gaupol(state)
    # Check the first 1s oscillator function, independent of basis array bounds.
    check("oscillator_1s_normalization", np.sum(state.rnl[1, 0, 1:]**2), 1.0)
    check("Talmi_Moshinsky_0s_transform", pairing.talmos(0, 0, 0, 0, 0, 0, 0, 0, 0), 1.0)
    overlaps = pairing.vn0(state)
    oscillator_b = state.b0
    a2 = state.gal
    for radial_n in (0, 1):
        normalizer = sqrt(2 * factorial(radial_n)
                          / (oscillator_b**3 * gamma(radial_n + 1.5)))

        def integrand(x):
            radial = normalizer * eval_genlaguerre(radial_n, 0.5, x*x / oscillator_b**2)
            radial *= np.exp(-x*x / (2 * oscillator_b**2))
            gaussian = np.exp(-x*x / (2 * a2)) / (4 * pi * a2)**1.5
            return sqrt(4 * pi) * x*x * radial * gaussian

        direct_overlap = 2**0.75 * quad(integrand, 0, np.inf, epsabs=1e-12)[0]
        check(f"Gaussian_pairing_overlap_n{radial_n}", overlaps[radial_n], direct_overlap)

    eigenvalues, eigenvectors = sdiag(np.array([[-1.2, 0.8], [0.8, 1.2]]))
    quasiparticle_energy = sqrt(1.2**2 + 0.8**2)
    check("one_level_HFB_energy", eigenvalues, [-quasiparticle_energy, quasiparticle_energy])
    check("one_level_HFB_occupation", eigenvectors[1, 1]**2,
          (1 + 1.2 / quasiparticle_energy) / 2)

    if args.results.exists():
        for record in json.loads(args.results.read_text(encoding="utf-8")):
            rn, rp, rm = record["rms_radii"]
            config = NucleusConfig(**record["config"])
            label = record["model"]
            expected_rm = sqrt(((config.mass_number - config.proton_number) * rn**2
                                + config.proton_number * rp**2) / config.mass_number)
            check(f"{label}_matter_radius_fm", rm, expected_rm)
            check(f"{label}_charge_radius_fm", record["values"][1], sqrt(rp**2 + 0.64))
            check(f"{label}_neutron_skin_fm", record["values"][2], rn - rp)

    report = dict(passed=True, check_count=len(checks), checks=checks,
                  saturation_values_MeV=values.tolist(),
                  EOS_table_energy_at_n0_MeV=float(table[0, 1]),
                  finite_nucleus_results=str(args.results) if args.results.exists() else None,
                  scope="Formula checks; finite-nucleus identities use saved Python results.")
    path = ROOT / "output" / "derivation_checks.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"PASS: {len(checks)} documented formula checks")
    print(f"Saturation e, K, symmetry energy, L [MeV]: {values}")
    print(f"Report: {path}")


if __name__ == "__main__":
    main()
