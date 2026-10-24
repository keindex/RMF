"""Beta-equilibrium matter and Tolman-Oppenheimer-Volkoff stellar structure."""

import math

import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import PchipInterpolator

from ..nuclear_matter.eos import (
    RMFParameters,
    beta_equilibrium,
    chemical_potentials,
)


MSUN_LENGTH_KM = 1.47662504
MEV_FM3_TO_KM2 = 1.3238333135663825e-6


def build_beta_equilibrium_eos(
        parameters: RMFParameters, transition_density: float,
        max_density: float = 1.55, density_points: int = 150):
    r"""构造单调、电中性的 beta 平衡中子星核心 EOS。

    密度单位为 $\mathrm{fm}^{-3}$，压强和能量密度单位为
    $\mathrm{MeV}\,\mathrm{fm}^{-3}$。

    Pressure is reconstructed from the zero-temperature identity
    $P=n\mu_n-\epsilon$, including leptons in $\epsilon$.

    Args:
        parameters (RMFParameters): RMF interaction parameters.
        transition_density (float): Core-crust matching density in
            $\mathrm{fm}^{-3}$.
        max_density (float): Maximum central density in $\mathrm{fm}^{-3}$.
        density_points (int): Number of points above the matching density.

    Returns:
        tuple: Density ($\mathrm{fm}^{-3}$), pressure and energy density
        arrays (both $\mathrm{MeV}\,\mathrm{fm}^{-3}$).

    Raises:
        RuntimeError: If the selected equilibrium branch is nonmonotone or
            has nonpositive pressure at the matching density.
    """
    densities = np.unique(np.r_[
        transition_density,
        np.geomspace(transition_density * 1.005, max_density, density_points),
    ])
    equilibrium_states = [
        beta_equilibrium(float(density), parameters) for density in densities
    ]
    # The first law gives P = n*mu_n - epsilon for neutral beta-stable matter.
    pressure_core = np.array([
        density * chemical_potentials(
            density, equilibrium.fields, equilibrium.asymmetry, parameters
        )[0] * 1000 - equilibrium.energy_density
        for density, equilibrium in zip(densities, equilibrium_states)
    ])
    energy_core = np.array([
        equilibrium.energy_density for equilibrium in equilibrium_states
    ])
    if np.any(np.diff(pressure_core) <= 0):
        first_bad_index = int(np.flatnonzero(np.diff(pressure_core) <= 0)[0])
        raise RuntimeError(
            "beta-equilibrium EOS is nonmonotone near "
            f"n={densities[first_bad_index]:.4g} fm^-3; "
            "a phase construction is required"
        )
    if pressure_core[0] <= 0:
        raise RuntimeError("nonpositive pressure at crust matching density")
    return densities, pressure_core, energy_core


def make_crust(density: float, pressure_value: float, energy: float,
               gamma: float = 4 / 3):
    r"""将多方状态方程 crust 连续匹配到核心 EOS。

    The crust uses $P=K n^\gamma$ and
    $\epsilon=a n+P/(\gamma-1)$. The coefficient $a$ is chosen
    so pressure and energy density match at the transition.

    Args:
        density (float): Matching density in $\mathrm{fm}^{-3}$.
        pressure_value (float): Matching pressure in $\mathrm{MeV}\,\mathrm{fm}^{-3}$.
        energy (float): Matching energy density in $\mathrm{MeV}\,\mathrm{fm}^{-3}$.
        gamma (float): Polytropic exponent.

    Returns:
        dict: Matching density, exponent, and linear energy coefficient.
    """
    return {
        "n_transition": density,
        "gamma": gamma,
        "linear_energy": (energy - pressure_value / (gamma - 1)) / density,
    }


def integrate_star(n_c: float, eos):
    r"""给定中心重子密度，积分求解一颗静态球对称中子星。

    In geometrized units $G=c=1$, the structure equations are

    $$
        \frac{dm}{dr}=4\pi r^2\epsilon,\qquad
        \frac{dP}{dr}=-\frac{(\epsilon+P)(m+4\pi r^3P)}{r(r-2m)}.
    $$

    Args:
        n_c (float): Central density in $\mathrm{fm}^{-3}$.
        eos (tuple): Core density, pressure, energy arrays and crust match
            dictionary returned/constructed by this module.

    Returns:
        tuple[float, float]: Radius in km and gravitational mass in solar
        masses. Returns ``(nan, nan)`` if the surface event is not reached.
    """
    densities, p_core, e_core, crust = eos
    p_transition = p_core[0]
    core_energy = PchipInterpolator(
        p_core * MEV_FM3_TO_KM2,
        e_core * MEV_FM3_TO_KM2,
        extrapolate=False,
    )
    pc_mev = float(np.interp(n_c, densities, p_core))
    pc = pc_mev * MEV_FM3_TO_KM2
    ec = float(np.interp(n_c, densities, e_core)) * MEV_FM3_TO_KM2
    r_start = 1e-5
    m_start = 4 * math.pi * ec * r_start**3 / 3

    def epsilon(pressure_geo):
        r"""Interpolate $\epsilon(P)$, switching to the crust polytrope below $P_t$."""
        pressure_geo = max(pressure_geo, 0.0)
        if pressure_geo < p_transition * MEV_FM3_TO_KM2:
            n = crust["n_transition"] * (
                pressure_geo / (p_transition * MEV_FM3_TO_KM2)
            ) ** (1 / crust["gamma"])
            pressure_mev = pressure_geo / MEV_FM3_TO_KM2
            return (
                crust["linear_energy"] * n
                + pressure_mev / (crust["gamma"] - 1)
            ) * MEV_FM3_TO_KM2
        return float(core_energy(min(
            pressure_geo, p_core[-1] * MEV_FM3_TO_KM2
        )))

    def derivatives(radius, values):
        r"""Return TOV derivatives $(dm/dr,dP/dr)$ in geometric units.

        $dm/dr=4\pi r^2\epsilon$ and
        $dP/dr=-(\epsilon+P)(m+4\pi r^3P)/(r(r-2m))$.
        """
        mass, pressure_geo = values
        if pressure_geo <= 0 or radius <= 2 * mass:
            return 0.0, 0.0
        energy_geo = epsilon(pressure_geo)
        # TOV mass and hydrostatic-pressure gradients in G=c=1 units.
        dm_dr = 4 * math.pi * radius**2 * energy_geo
        dp_dr = -(energy_geo + pressure_geo) * (
            mass + 4 * math.pi * radius**3 * pressure_geo
        ) / (radius * (radius - 2 * mass))
        return dm_dr, dp_dr

    def surface(_radius, values):
        """Surface event $P(r)=0$, terminating integration on a downward crossing."""
        return values[1]

    surface.terminal = True
    surface.direction = -1

    result = solve_ivp(
        derivatives, (r_start, 40.0), (m_start, pc), events=surface,
        rtol=2e-6, atol=1e-10, max_step=0.12,
    )
    if not result.t_events[0].size:
        return math.nan, math.nan
    radius = float(result.t_events[0][0])
    mass = float(result.y_events[0][0, 0]) / MSUN_LENGTH_KM
    return radius, mass

