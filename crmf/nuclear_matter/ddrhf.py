"""Density-dependent relativistic Hartree-Fock nuclear matter with PKO forces.

The implementation follows the PKO parameterization and homogeneous-matter
equations documented in ``full.md`` and the DDRHF equations of state in
arXiv:0910.4236. Public densities use fm**-3; momenta, masses, self-energies,
and energies use MeV. Internal momentum integrals use natural units.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from math import exp, log, pi, sqrt
from types import MappingProxyType

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.optimize import minimize_scalar


HBAR_C_MEV_FM = 197.328284
NUCLEON_MASS_MEV = 938.9
OMEGA_MASS_MEV = 783.0
RHO_MASS_MEV = 769.0
PION_MASS_MEV = 138.0


@dataclass(frozen=True)
class PKOParameters:
    """One PKO density-dependent meson-nucleon parameterization."""

    name: str
    sigma_mass_MeV: float
    g_sigma_at_saturation: float
    g_omega_at_saturation: float
    g_rho_at_zero: float
    f_pi_at_zero: float | None
    a_rho: float
    a_pi: float | None
    saturation_density_fm3: float
    sigma_density: tuple[float, float, float, float]
    omega_density: tuple[float, float, float, float]

    def __post_init__(self):
        numeric = (
            self.sigma_mass_MeV,
            self.g_sigma_at_saturation,
            self.g_omega_at_saturation,
            self.g_rho_at_zero,
            self.a_rho,
            self.saturation_density_fm3,
            *self.sigma_density,
            *self.omega_density,
        )
        if not all(np.isfinite(value) for value in numeric):
            raise ValueError("PKO parameters must be finite")
        if self.f_pi_at_zero is not None and not np.isfinite(self.f_pi_at_zero):
            raise ValueError("pion coupling must be finite when present")
        if self.a_pi is not None and not np.isfinite(self.a_pi):
            raise ValueError("pion density parameter must be finite when present")
        if self.sigma_mass_MeV <= 0 or self.saturation_density_fm3 <= 0:
            raise ValueError("meson mass and saturation density must be positive")
        if (self.f_pi_at_zero is None) != (self.a_pi is None):
            raise ValueError("pion coupling and density parameter must both be present or absent")


PKO_PARAMETERS = MappingProxyType({
    "PKO1": PKOParameters(
        "PKO1", 525.769084, 8.833239, 10.729933, 2.629000, 1.0,
        0.076760, 1.231976, 0.151989,
        (1.384494, 1.513190, 2.296615, 0.380974),
        (1.403347, 2.008719, 3.046686, 0.330770),
    ),
    "PKO2": PKOParameters(
        "PKO2", 534.461766, 8.920597, 10.550553, 4.068299, None,
        0.631605, None, 0.151021,
        (1.375772, 2.064391, 3.052417, 0.330459),
        (1.451420, 3.574373, 5.478373, 0.246668),
    ),
    "PKO3": PKOParameters(
        "PKO3", 525.667686, 8.895635, 10.802690, 3.832480, 1.0,
        0.635336, 0.934122, 0.153006,
        (1.244635, 1.566659, 2.074581, 0.400843),
        (1.245714, 1.645754, 2.177077, 0.391293),
    ),
})


@dataclass(frozen=True)
class DensityDependentCouplings:
    g_sigma: float
    dg_sigma_drho: float
    g_omega: float
    dg_omega_drho: float
    g_rho: float
    dg_rho_drho: float
    f_pi: float
    df_pi_drho: float


@dataclass(frozen=True)
class NuclearMatterState:
    density_fm3: float
    asymmetry: float
    energy_density_MeV_fm3: float
    binding_energy_per_particle_MeV: float
    scalar_density_fm3: float
    rearrangement_MeV: float
    neutron_fermi_momentum_MeV: float
    proton_fermi_momentum_MeV: float


@dataclass(frozen=True)
class NuclearMatterProperties:
    parameter_set: str
    saturation_density_fm3: float
    incompressibility_MeV: float
    symmetry_energy_MeV: float
    binding_energy_per_particle_MeV: float
    dirac_mass_ratio: float
    nonrelativistic_mass_ratio: float
    relativistic_mass_ratio: float
    pressure_at_saturation_MeV_fm3: float


@dataclass(frozen=True)
class _SpeciesGrid:
    isospin: int
    fermi_momentum: float
    momenta: np.ndarray
    weights: np.ndarray


@dataclass(frozen=True)
class _SolvedMatter:
    result: NuclearMatterState
    parameters: PKOParameters
    couplings: DensityDependentCouplings
    grids: tuple[_SpeciesGrid, ...]
    hat_mass: tuple[np.ndarray, ...]
    hat_momentum: tuple[np.ndarray, ...]
    channel_fields: tuple[dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]], ...]
    rearrangement: float
    dirac_mass_ratio: float
    relativistic_mass_MeV: float
    nonrelativistic_mass_MeV: float
    fermi_energies_MeV: tuple[float, ...]


def _density_function(x: float, parameters: tuple[float, float, float, float]):
    a, b, c, d = parameters
    y = (x + d) ** 2
    denominator = 1.0 + c * y
    value = a * (1.0 + b * y) / denominator
    derivative = 2.0 * a * (b - c) * (x + d) / denominator**2
    return value, derivative


def density_dependent_couplings(
    parameters: PKOParameters, density_fm3: float
) -> DensityDependentCouplings:
    """Evaluate PKO vertices and their derivatives with respect to natural density.

    The derivatives are with respect to density in MeV**3, as required by the
    rearrangement self-energy in natural units.
    """
    if not np.isfinite(density_fm3) or density_fm3 <= 0:
        raise ValueError("baryon density must be finite and positive")
    x = density_fm3 / parameters.saturation_density_fm3
    g_sigma_shape, dg_sigma_shape = _density_function(x, parameters.sigma_density)
    g_omega_shape, dg_omega_shape = _density_function(x, parameters.omega_density)
    density_scale = parameters.saturation_density_fm3 * HBAR_C_MEV_FM**3
    g_sigma = parameters.g_sigma_at_saturation * g_sigma_shape
    g_omega = parameters.g_omega_at_saturation * g_omega_shape
    g_rho = parameters.g_rho_at_zero * exp(-parameters.a_rho * x)
    if parameters.f_pi_at_zero is None:
        f_pi = 0.0
        df_pi = 0.0
    else:
        f_pi = parameters.f_pi_at_zero * exp(-parameters.a_pi * x)
        df_pi = -parameters.a_pi * f_pi / density_scale
    return DensityDependentCouplings(
        g_sigma,
        parameters.g_sigma_at_saturation * dg_sigma_shape / density_scale,
        g_omega,
        parameters.g_omega_at_saturation * dg_omega_shape / density_scale,
        g_rho,
        -parameters.a_rho * g_rho / density_scale,
        f_pi,
        df_pi,
    )


@lru_cache(maxsize=8)
def _quadrature(order: int):
    nodes, weights = leggauss(order)
    return nodes, weights


def _species_grids(density_fm3: float, asymmetry: float, order: int):
    density_neutron = density_fm3 * (1.0 + asymmetry) / 2.0
    density_proton = density_fm3 * (1.0 - asymmetry) / 2.0
    nodes, weights = _quadrature(order)
    grids = []
    for isospin, density in ((1, density_neutron), (-1, density_proton)):
        if density <= 0:
            continue
        fermi_momentum = HBAR_C_MEV_FM * (3.0 * pi**2 * density) ** (1.0 / 3.0)
        momenta = 0.5 * fermi_momentum * (nodes + 1.0)
        quadrature_weights = 0.5 * fermi_momentum * weights
        grids.append(_SpeciesGrid(isospin, fermi_momentum, momenta, quadrature_weights))
    return tuple(grids)


def _propagator_angles(p, q, meson_mass):
    product = p * q
    denominator_minus = meson_mass**2 + (p - q) ** 2
    theta = np.log1p(4.0 * product / denominator_minus)
    phi = 0.5 * (
        (p**2 + q**2 + meson_mass**2) * theta / (2.0 * product) - 2.0
    )
    return theta, phi


def _meson_kernels(p, q, parameters, couplings):
    p_grid, q_grid = np.broadcast_arrays(p, q)
    theta_sigma, phi_sigma = _propagator_angles(
        p_grid, q_grid, parameters.sigma_mass_MeV
    )
    theta_omega, phi_omega = _propagator_angles(
        p_grid, q_grid, OMEGA_MASS_MEV
    )
    theta_rho, phi_rho = _propagator_angles(p_grid, q_grid, RHO_MASS_MEV)
    theta_pi, phi_pi = _propagator_angles(p_grid, q_grid, PION_MASS_MEV)
    gs2 = couplings.g_sigma**2
    gw2 = couplings.g_omega**2
    gr2 = couplings.g_rho**2
    fp2 = couplings.f_pi**2
    return {
        "sigma": (gs2 * theta_sigma, gs2 * theta_sigma, -2.0 * gs2 * phi_sigma),
        "omega": (2.0 * gw2 * theta_omega, -4.0 * gw2 * theta_omega,
                  -4.0 * gw2 * phi_omega),
        "rho": (2.0 * gr2 * theta_rho, -4.0 * gr2 * theta_rho,
                -4.0 * gr2 * phi_rho),
        "pi": (-fp2 * theta_pi, -fp2 * theta_pi,
               2.0 * fp2 / PION_MASS_MEV**2
               * ((p_grid**2 + q_grid**2) * phi_pi - p_grid * q_grid * theta_pi)),
    }


def _exchange_fields(target_momenta, target_grid, grids, hat_mass, hat_momentum,
                     parameters, couplings):
    total = {}
    target_count = target_momenta.size
    for meson in ("sigma", "omega", "rho", "pi"):
        total[meson] = tuple(np.zeros(target_count) for _ in range(3))
    for source_index, source_grid in enumerate(grids):
        p = target_momenta[:, None]
        q = source_grid.momenta[None, :]
        kernels = _meson_kernels(p, q, parameters, couplings)
        radial_weights = source_grid.weights * source_grid.momenta
        isospin_delta = float(target_grid.isospin == source_grid.isospin)
        for meson, (kernel_a, kernel_b, kernel_c) in kernels.items():
            isospin_weight = isospin_delta if meson in ("sigma", "omega") else 2.0 - isospin_delta
            scalar, time, vector = total[meson]
            scalar += isospin_weight * (kernel_b @ (radial_weights * hat_mass[source_index]))
            time += isospin_weight * (kernel_a @ radial_weights)
            vector += isospin_weight * (kernel_c @ (radial_weights * hat_momentum[source_index]))
    prefactor = 1.0 / ((4.0 * pi) ** 2 * target_momenta)
    for meson in total:
        total[meson] = tuple(component * prefactor for component in total[meson])
    return total


def _field_integral(grids, values):
    return sum(
        np.dot(grid.weights, grid.momenta**2 * value)
        for grid, value in zip(grids, values)
    ) / pi**2


def _solve_matter(density_fm3, asymmetry, parameters, order=24, tolerance=2e-7,
                  max_iterations=160, mixing=0.25):
    couplings = density_dependent_couplings(parameters, density_fm3)
    grids = _species_grids(density_fm3, asymmetry, order)
    if not grids:
        raise ValueError("at least one nucleon species must have positive density")
    density_nat = density_fm3 * HBAR_C_MEV_FM**3
    density_neutron = density_nat * (1.0 + asymmetry) / 2.0
    density_proton = density_nat * (1.0 - asymmetry) / 2.0
    density_isovector = density_neutron - density_proton
    momenta = tuple(grid.momenta for grid in grids)
    sigma = tuple(np.full(grid.momenta.size, -250.0) for grid in grids)
    vector = tuple(np.zeros(grid.momenta.size) for grid in grids)
    rearrangement = 0.0

    for _ in range(max_iterations):
        hat_mass = []
        hat_momentum = []
        for grid, scalar_field, vector_field in zip(grids, sigma, vector):
            effective_mass = NUCLEON_MASS_MEV + scalar_field
            effective_momentum = grid.momenta + vector_field
            effective_energy = np.sqrt(effective_mass**2 + effective_momentum**2)
            if np.any(effective_energy <= 0) or np.any(effective_mass <= 0):
                raise RuntimeError("DDRHF iteration entered a nonphysical effective-mass branch")
            hat_mass.append(effective_mass / effective_energy)
            hat_momentum.append(effective_momentum / effective_energy)
        hat_mass = tuple(hat_mass)
        hat_momentum = tuple(hat_momentum)
        scalar_density = _field_integral(grids, hat_mass)
        direct_sigma = -couplings.g_sigma**2 * scalar_density / parameters.sigma_mass_MeV**2
        direct_omega = couplings.g_omega**2 * density_nat / OMEGA_MASS_MEV**2
        direct_rho = couplings.g_rho**2 * density_isovector / RHO_MASS_MEV**2
        channel_fields = []
        new_sigma = []
        new_vector = []
        for species_index, grid in enumerate(grids):
            exchange = _exchange_fields(
                grid.momenta, grid, grids, hat_mass, hat_momentum,
                parameters, couplings,
            )
            fields = {
                meson: tuple(component.copy() for component in exchange[meson])
                for meson in exchange
            }
            fields["sigma"] = (
                fields["sigma"][0] + direct_sigma,
                fields["sigma"][1],
                fields["sigma"][2],
            )
            fields["omega"] = (
                fields["omega"][0],
                fields["omega"][1] + direct_omega,
                fields["omega"][2],
            )
            fields["rho"] = (
                fields["rho"][0],
                fields["rho"][1] + grid.isospin * direct_rho,
                fields["rho"][2],
            )
            channel_fields.append(fields)
            new_sigma.append(sum(fields[meson][0] for meson in fields))
            new_vector.append(sum(fields[meson][2] for meson in fields))

        rearrangement = 0.0
        for meson, coupling, derivative in (
            ("sigma", couplings.g_sigma, couplings.dg_sigma_drho),
            ("omega", couplings.g_omega, couplings.dg_omega_drho),
            ("rho", couplings.g_rho, couplings.dg_rho_drho),
            ("pi", couplings.f_pi, couplings.df_pi_drho),
        ):
            if coupling == 0.0:
                continue
            rearrangement += derivative / coupling * _field_integral(
                grids,
                tuple(
                    hat_mass[index] * channel_fields[index][meson][0]
                    + channel_fields[index][meson][1]
                    + hat_momentum[index] * channel_fields[index][meson][2]
                    for index in range(len(grids))
                ),
            )
        residual = max(
            max(float(np.max(np.abs(new - old))) for new, old in zip(new_sigma, sigma)),
            max(float(np.max(np.abs(new - old))) for new, old in zip(new_vector, vector)),
        )
        sigma = tuple((1.0 - mixing) * old + mixing * new for old, new in zip(sigma, new_sigma))
        vector = tuple((1.0 - mixing) * old + mixing * new for old, new in zip(vector, new_vector))
        if residual < tolerance:
            break
    else:
        raise RuntimeError(
            f"{parameters.name} DDRHF matter iteration failed at rho={density_fm3:g}, "
            f"alpha={asymmetry:g}; residual={residual:.3g} MeV"
        )

    # Re-evaluate fields and the rearrangement term with the converged spinors.
    hat_mass = []
    hat_momentum = []
    for grid, scalar_field, vector_field in zip(grids, sigma, vector):
        effective_mass = NUCLEON_MASS_MEV + scalar_field
        effective_momentum = grid.momenta + vector_field
        effective_energy = np.sqrt(effective_mass**2 + effective_momentum**2)
        hat_mass.append(effective_mass / effective_energy)
        hat_momentum.append(effective_momentum / effective_energy)
    hat_mass = tuple(hat_mass)
    hat_momentum = tuple(hat_momentum)
    scalar_density = _field_integral(grids, hat_mass)
    direct_sigma = -couplings.g_sigma**2 * scalar_density / parameters.sigma_mass_MeV**2
    direct_omega = couplings.g_omega**2 * density_nat / OMEGA_MASS_MEV**2
    direct_rho = couplings.g_rho**2 * density_isovector / RHO_MASS_MEV**2
    channel_fields = []
    for grid in grids:
        exchange = _exchange_fields(
            grid.momenta, grid, grids, hat_mass, hat_momentum,
            parameters, couplings,
        )
        fields = {meson: tuple(component.copy() for component in exchange[meson])
                  for meson in exchange}
        fields["sigma"] = (fields["sigma"][0] + direct_sigma,
                           fields["sigma"][1], fields["sigma"][2])
        fields["omega"] = (fields["omega"][0],
                           fields["omega"][1] + direct_omega, fields["omega"][2])
        fields["rho"] = (fields["rho"][0],
                         fields["rho"][1] + grid.isospin * direct_rho,
                         fields["rho"][2])
        channel_fields.append(fields)
    channel_fields = tuple(channel_fields)
    rearrangement = sum(
        derivative / coupling * _field_integral(
            grids,
            tuple(
                hat_mass[index] * channel_fields[index][meson][0]
                + channel_fields[index][meson][1]
                + hat_momentum[index] * channel_fields[index][meson][2]
                for index in range(len(grids))
            ),
        )
        for meson, coupling, derivative in (
            ("sigma", couplings.g_sigma, couplings.dg_sigma_drho),
            ("omega", couplings.g_omega, couplings.dg_omega_drho),
            ("rho", couplings.g_rho, couplings.dg_rho_drho),
            ("pi", couplings.f_pi, couplings.df_pi_drho),
        )
        if coupling != 0.0
    )

    kinetic_energy = sum(
        np.dot(
            grid.weights,
            grid.momenta**2
            * (grid.momenta * hat_momentum[index]
               + NUCLEON_MASS_MEV * hat_mass[index]),
        )
        for index, grid in enumerate(grids)
    ) / pi**2
    energy_direct = (
        -0.5 * couplings.g_sigma**2 * scalar_density**2 / parameters.sigma_mass_MeV**2
        + 0.5 * couplings.g_omega**2 * density_nat**2 / OMEGA_MASS_MEV**2
        + 0.5 * couplings.g_rho**2 * density_isovector**2 / RHO_MASS_MEV**2
    )
    energy_exchange = 0.0
    for target_index, target_grid in enumerate(grids):
        p = target_grid.momenta[:, None]
        for source_index, source_grid in enumerate(grids):
            q = source_grid.momenta[None, :]
            kernels = _meson_kernels(p, q, parameters, couplings)
            isospin_delta = float(target_grid.isospin == source_grid.isospin)
            radial_weights_target = target_grid.weights * target_grid.momenta
            radial_weights_source = source_grid.weights * source_grid.momenta
            pair_weights = radial_weights_target[:, None] * radial_weights_source[None, :]
            for meson, (kernel_a, kernel_b, kernel_c) in kernels.items():
                isospin_weight = isospin_delta if meson in ("sigma", "omega") else 2.0 - isospin_delta
                integrand = (
                    kernel_a
                    + hat_mass[target_index][:, None] * hat_mass[source_index][None, :] * kernel_b
                    + hat_momentum[target_index][:, None] * hat_momentum[source_index][None, :] * kernel_c
                )
                energy_exchange += isospin_weight * float(np.sum(pair_weights * integrand))
    energy_exchange *= 0.5 / (2.0 * pi) ** 4
    energy_density_nat = kinetic_energy + energy_direct + energy_exchange
    binding_energy = energy_density_nat / density_nat - NUCLEON_MASS_MEV

    result = NuclearMatterState(
        density_fm3=float(density_fm3),
        asymmetry=float(asymmetry),
        energy_density_MeV_fm3=float(energy_density_nat / HBAR_C_MEV_FM**3),
        binding_energy_per_particle_MeV=float(binding_energy),
        scalar_density_fm3=float(scalar_density / HBAR_C_MEV_FM**3),
        rearrangement_MeV=float(rearrangement),
        neutron_fermi_momentum_MeV=next(
            (grid.fermi_momentum for grid in grids if grid.isospin == 1), 0.0
        ),
        proton_fermi_momentum_MeV=next(
            (grid.fermi_momentum for grid in grids if grid.isospin == -1), 0.0
        ),
    )

    mass_ratios = []
    relativistic_masses = []
    nonrelativistic_masses = []
    fermi_energies = []
    for index, grid in enumerate(grids):
        target_momenta = np.array([grid.fermi_momentum])
        fermi_exchange = _exchange_fields(
            target_momenta, grid, grids, hat_mass, hat_momentum,
            parameters, couplings,
        )
        fermi_sigma = direct_sigma + sum(fermi_exchange[m][0][0] for m in fermi_exchange)
        isovector_time = grid.isospin * direct_rho
        fermi_vector = (
            direct_omega + isovector_time + rearrangement
            + sum(fermi_exchange[m][1][0] for m in fermi_exchange)
        )
        mass_ratios.append((NUCLEON_MASS_MEV + fermi_sigma) / NUCLEON_MASS_MEV)

        def single_particle_energy(momentum):
            exchange = _exchange_fields(
                np.array([momentum]), grid, grids, hat_mass, hat_momentum,
                parameters, couplings,
            )
            scalar = direct_sigma + sum(exchange[m][0][0] for m in exchange)
            time = (direct_omega + isovector_time + rearrangement
                    + sum(exchange[m][1][0] for m in exchange))
            space = sum(exchange[m][2][0] for m in exchange)
            effective_mass = NUCLEON_MASS_MEV + scalar
            effective_momentum = momentum + space
            return time + sqrt(effective_mass**2 + effective_momentum**2)

        derivative_step = max(0.2, grid.fermi_momentum * 1e-4)
        group_velocity = (
            single_particle_energy(grid.fermi_momentum + derivative_step)
            - single_particle_energy(grid.fermi_momentum - derivative_step)
        ) / (2.0 * derivative_step)
        relativistic_mass = grid.fermi_momentum / group_velocity
        fermi_energy = single_particle_energy(grid.fermi_momentum)
        relativistic_masses.append(relativistic_mass)
        nonrelativistic_masses.append(relativistic_mass - (fermi_energy - NUCLEON_MASS_MEV))
        fermi_energies.append(fermi_energy)

    return _SolvedMatter(
        result=result,
        parameters=parameters,
        couplings=couplings,
        grids=grids,
        hat_mass=hat_mass,
        hat_momentum=hat_momentum,
        channel_fields=channel_fields,
        rearrangement=float(rearrangement),
        dirac_mass_ratio=float(np.mean(mass_ratios)),
        relativistic_mass_MeV=float(np.mean(relativistic_masses)),
        nonrelativistic_mass_MeV=float(np.mean(nonrelativistic_masses)),
        fermi_energies_MeV=tuple(float(value) for value in fermi_energies),
    )


def solve_nuclear_matter(
    density_fm3: float,
    asymmetry: float = 0.0,
    parameters: PKOParameters | str = "PKO1",
    *,
    quadrature_order: int = 24,
    tolerance_MeV: float = 2e-7,
) -> NuclearMatterState:
    """Solve homogeneous DDRHF matter at density and isospin asymmetry.

    ``asymmetry`` is :math:`(rho_n-rho_p)/rho_b`. The no-sea approximation
    and the self-consistent momentum-dependent sigma, omega, rho, and pion
    Fock self-energies are used. PKO2 has no pion vertex.
    """
    if isinstance(parameters, str):
        try:
            parameters = PKO_PARAMETERS[parameters.upper()]
        except KeyError as error:
            raise ValueError(f"unknown PKO parameter set: {parameters}") from error
    if not isinstance(parameters, PKOParameters):
        raise TypeError("parameters must be a PKOParameters instance or PKO name")
    if not np.isfinite(density_fm3) or density_fm3 <= 0:
        raise ValueError("baryon density must be finite and positive")
    if not np.isfinite(asymmetry) or abs(asymmetry) > 1:
        raise ValueError("isospin asymmetry must be finite and between -1 and 1")
    if quadrature_order < 8:
        raise ValueError("quadrature_order must be at least 8")
    if not np.isfinite(tolerance_MeV) or tolerance_MeV <= 0:
        raise ValueError("tolerance_MeV must be finite and positive")
    return _solve_matter(
        density_fm3,
        asymmetry,
        parameters,
        order=quadrature_order,
        tolerance=tolerance_MeV,
    ).result


def nuclear_matter_properties(
    parameters: PKOParameters | str = "PKO1",
    *,
    quadrature_order: int = 24,
) -> NuclearMatterProperties:
    """Calculate saturation observables shown in Table II of ``full.md``."""
    if isinstance(parameters, str):
        try:
            parameters = PKO_PARAMETERS[parameters.upper()]
        except KeyError as error:
            raise ValueError(f"unknown PKO parameter set: {parameters}") from error
    if not isinstance(parameters, PKOParameters):
        raise TypeError("parameters must be a PKOParameters instance or PKO name")
    rho_reference = parameters.saturation_density_fm3

    def energy_per_particle(density, asymmetry=0.0):
        return _solve_matter(
            density, asymmetry, parameters, order=quadrature_order
        ).result.binding_energy_per_particle_MeV

    minimum = minimize_scalar(
        energy_per_particle,
        bounds=(0.82 * rho_reference, 1.18 * rho_reference),
        method="bounded",
        options={"xatol": 2e-7, "maxiter": 24},
    )
    saturation_density = float(minimum.x)
    symmetric = _solve_matter(
        saturation_density, 0.0, parameters, order=quadrature_order
    )
    h = min(0.00025, saturation_density / 100.0)
    energies = [
        energy_per_particle(saturation_density + 2 * h),
        energy_per_particle(saturation_density + h),
        symmetric.result.binding_energy_per_particle_MeV,
        energy_per_particle(saturation_density - h),
        energy_per_particle(saturation_density - 2 * h),
    ]
    second_density_derivative = (
        -energies[0] + 16 * energies[1] - 30 * energies[2]
        + 16 * energies[3] - energies[4]
    ) / (12.0 * h**2)
    incompressibility = 9.0 * saturation_density**2 * second_density_derivative
    asymmetry_step = 0.02
    neutron_rich = energy_per_particle(saturation_density, asymmetry_step)
    proton_rich = energy_per_particle(saturation_density, -asymmetry_step)
    symmetry_energy = (
        neutron_rich + proton_rich - 2.0 * symmetric.result.binding_energy_per_particle_MeV
    ) / (2.0 * asymmetry_step**2)
    pressure_step = 0.0002
    pressure_derivative = (
        energy_per_particle(saturation_density + pressure_step)
        - energy_per_particle(saturation_density - pressure_step)
    ) / (2.0 * pressure_step)
    pressure = saturation_density**2 * pressure_derivative
    return NuclearMatterProperties(
        parameter_set=parameters.name,
        saturation_density_fm3=saturation_density,
        incompressibility_MeV=float(incompressibility),
        symmetry_energy_MeV=float(symmetry_energy),
        binding_energy_per_particle_MeV=symmetric.result.binding_energy_per_particle_MeV,
        dirac_mass_ratio=symmetric.dirac_mass_ratio,
        nonrelativistic_mass_ratio=symmetric.nonrelativistic_mass_MeV / NUCLEON_MASS_MEV,
        relativistic_mass_ratio=symmetric.relativistic_mass_MeV / NUCLEON_MASS_MEV,
        pressure_at_saturation_MeV_fm3=float(pressure),
    )


__all__ = [
    "DensityDependentCouplings",
    "NuclearMatterProperties",
    "NuclearMatterState",
    "PKOParameters",
    "PKO_PARAMETERS",
    "density_dependent_couplings",
    "nuclear_matter_properties",
    "solve_nuclear_matter",
]
