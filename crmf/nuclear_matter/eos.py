r"""Relativistic mean-field nuclear matter, translated from RMF_EOS_Nucleus.for.

Densities are fm**-3, masses and chemical potentials GeV, energy densities
and pressures GeV/fm**3. Meson fields retain the Fortran normalization:
multiply g*field by HBAR**1.5 to obtain a self-energy in GeV. The EOS table
and beta-equilibrium result explicitly document their MeV output columns.

Default REAL constants and constant expressions in the old source were
rounded to binary32 before promotion to REAL*8. That convention is retained
for numerical reproducibility, including the Fermi-momentum exponent.
No native executable or reference output is used by this module.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from math import log, sqrt
from typing import NamedTuple

import numpy as np
from scipy.optimize import brentq, root


def single_precision_real(value: float) -> float:
    r"""Round ``value`` to the source program's binary32 representation.

    The conversion is

    $$x_{32}=\operatorname{float32}(x),\qquad
      \operatorname{return}=\operatorname{float}(x_{32}).$$

    Keeping this operation explicit preserves the mixed REAL/REAL*8
    arithmetic used by the original Fortran EOS implementation.
    """
    return float(np.float32(value))


reduced_planck_constant = single_precision_real(0.197328284)
pi = single_precision_real(3.141592654)
pi_squared = pi**2
reduced_planck_constant_cubed = reduced_planck_constant**3
reduced_planck_constant_three_halves = reduced_planck_constant**1.5
nucleon_mass = single_precision_real(0.9390)
electron_mass = single_precision_real(0.000511)
muon_mass = single_precision_real(0.105658)
one_third = single_precision_real(1.0/3.0)

# Source-compatible constants retained for existing callers and data files.
HBAR = reduced_planck_constant
PI = pi
PI2 = pi_squared
HBAR3 = reduced_planck_constant_cubed
HBAR15 = reduced_planck_constant_three_halves
XM = nucleon_mass
ELECTRON_MASS = electron_mass
MUON_MASS = muon_mass
ONE_THIRD = one_third


@dataclass(frozen=True)
class RMFParameters:
    """Couplings and GeV meson masses; defaults describe main's input targets.

    Call :func:`default_parameters` to calibrate the couplings. Constructing
    this dataclass alone gives the original starting FSU couplings.
    """
    g_s: float = sqrt(112.1996)
    g_w: float = sqrt(204.5469)
    g_d: float = single_precision_real(6.2)
    g_r: float = sqrt(138.4701)/2
    b_s: float = single_precision_real(0.0014203)/2/XM
    c_s: float = single_precision_real(0.02376)/6
    c_w: float = single_precision_real(0.0052834)
    c_r: float = 0.0
    xl_s: float = 0.0
    xl_v: float = 8*single_precision_real(0.03)
    c_ds: float = single_precision_real(0.054)
    xms: float = single_precision_real(0.4915)
    xmw: float = single_precision_real(0.7825)
    xmd: float = single_precision_real(0.980)
    xmr: float = single_precision_real(0.763)
    saturation_density: float = single_precision_real(0.148)
    binding_energy: float = single_precision_real(-16.31)
    incompressibility: float = single_precision_real(229.2)
    effective_mass_ratio: float = single_precision_real(0.610)
    symmetry_energy: float = single_precision_real(32.53)
    symmetry_slope: float = single_precision_real(48.21)

    @property
    def scalar_coupling(self) -> float:
        return self.g_s

    @property
    def vector_coupling(self) -> float:
        return self.g_w

    @property
    def delta_coupling(self) -> float:
        return self.g_d

    @property
    def rho_coupling(self) -> float:
        return self.g_r

    @property
    def sigma_mass(self) -> float:
        return self.xms

    @property
    def omega_mass(self) -> float:
        return self.xmw

    @property
    def delta_mass(self) -> float:
        return self.xmd

    @property
    def rho_mass(self) -> float:
        return self.xmr

    @property
    def scalar_cubic_coupling(self) -> float:
        return self.b_s

    @property
    def scalar_quartic_coupling(self) -> float:
        return self.c_s

    @property
    def vector_quartic_coupling(self) -> float:
        return self.c_w

    @property
    def rho_quartic_coupling(self) -> float:
        return self.c_r

    @property
    def scalar_rho_mixing(self) -> float:
        return self.xl_s

    @property
    def vector_rho_mixing(self) -> float:
        return self.xl_v

    @property
    def scalar_delta_mixing(self) -> float:
        return self.c_ds

    def __post_init__(self):
        if not all(np.isfinite(value) for value in self.__dict__.values()):
            raise ValueError("RMF parameters must be finite")
        if min(self.xms, self.xmw, self.xmd, self.xmr) <= 0:
            raise ValueError("meson masses must be positive")
        if self.saturation_density <= 0 or not 0 < self.effective_mass_ratio < 1:
            raise ValueError("saturation density must be positive and 0 < m*/m < 1")


class MesonFields(NamedTuple):
    sigma: float
    omega: float
    rho: float = 0.0
    delta: float = 0.0


@dataclass(frozen=True)
class BetaEquilibriumResult:
    density: float
    asymmetry: float
    energy_density: float  # MeV/fm**3
    pressure: float  # MeV/fm**3
    electron_fraction: float
    muon_fraction: float
    chemical_potential: float  # GeV, mu_n - mu_p
    fields: MesonFields


def resolve_parameters(parameters: RMFParameters | None) -> RMFParameters:
    r"""Return the caller's parameter set, or the calibrated default set.

    In symbols, ``resolve_parameters(p) = default_parameters()`` when
    $p=\varnothing$, and ``resolve_parameters(p)=p`` otherwise.
    """
    return default_parameters() if parameters is None else parameters


def _validate(density: float, asymmetry: float = 0.0) -> None:
    r"""Validate the baryon density and isospin asymmetry domain.

    The accepted domain is
    $$n\in\mathbb{R},\quad n>0,\qquad -1\leq\alpha\leq1,$$
    with both values finite.  Species densities $n(1\pm\alpha)/2$ are then
    nonnegative.
    """
    if not np.isfinite(density) or density <= 0:
        raise ValueError("density must be finite and positive (fm^-3)")
    if not np.isfinite(asymmetry) or abs(asymmetry) > 1:
        raise ValueError("asymmetry must lie between -1 and 1")


def fermi_momentum(density: float) -> float:
    # Formula: k_F = (hbar*c) * (3*pi^2*n)^(1/3).
    r"""按自旋简并物种的数密度计算费米动量。

    $$
        k_F = \hbar c\,(3\pi^2 n)^{1/3}.
    $$

    Args:
        density (float): Species number density in $\mathrm{fm}^{-3}$.

    Returns:
        float: $k_F$ in GeV.
    """
    if density < 0 or not np.isfinite(density):
        raise ValueError("species density must be finite and nonnegative")
    return HBAR*(3*PI2*density)**ONE_THIRD


def scalar_density_integral(fermi_momentum_value: float, mass: float) -> float:
    # Formula: n_s = M*/(2*pi^2*hbar^3) * [k_F E_F - M*^2 ln((k_F+E_F)/M*)].
    r"""闭式计算零温相对论费米海的标量密度。

    $$
        n_s=\frac{M^*}{2\pi^2(\hbar c)^3}
        \left[k_FE_F^*-M^{*2}\ln\frac{k_F+E_F^*}{M^*}\right].
    $$

    Args:
        fermi_momentum_value (float): Fermi momentum in GeV.
        mass (float): Effective mass in GeV.

    Returns:
        float: Scalar density in $\mathrm{fm}^{-3}$.
    """
    if fermi_momentum_value == 0 or mass == 0:
        return 0.0
    fermi_energy = sqrt(fermi_momentum_value*fermi_momentum_value+mass*mass)
    return mass/(2*PI2)*(fermi_momentum_value*fermi_energy-mass*mass*log(abs((fermi_momentum_value+fermi_energy)/mass)))/HBAR3


def kinetic_energy_integral(fermi_momentum_value: float, mass: float) -> float:
    # Formula: epsilon_kin = (1/(pi^2*hbar^3)) integral_0^kF k^2 sqrt(k^2+M*^2) dk.
    r"""计算单个核子物种的相对论动能密度。

    This is $\pi^{-2}\int_0^{k_F} k^2\sqrt{k^2+M^{*2}}\,dk$.

    Args:
        fermi_momentum_value (float): Fermi momentum in GeV.
        mass (float): Effective mass in GeV.

    Returns:
        float: Kinetic energy density in $\mathrm{GeV}\,\mathrm{fm}^{-3}$.
    """
    if fermi_momentum_value == 0:
        return 0.0
    if mass == 0:
        return fermi_momentum_value**4/(4*PI2*HBAR3)
    fermi_energy = sqrt(fermi_momentum_value*fermi_momentum_value+mass*mass)
    return (fermi_energy*(2*fermi_momentum_value**3+fermi_momentum_value*mass**2)-mass**4*log(abs((fermi_momentum_value+fermi_energy)/mass)))/(8*PI2*HBAR3)


def kinetic_pressure_integral(fermi_momentum_value: float, mass: float) -> float:
    # Formula: P_kin = (1/(3*pi^2*hbar^3)) integral_0^kF k^4/sqrt(k^2+M*^2) dk.
    r"""计算单个核子物种的相对论动能压强。

    $$
        P_{\rm kin}=\frac{1}{3\pi^2(\hbar c)^3}
        \int_0^{k_F}\frac{k^4}{\sqrt{k^2+M^{*2}}}\,dk.
    $$

    Args:
        fermi_momentum_value (float): Fermi momentum in GeV.
        mass (float): Effective mass in GeV.

    Returns:
        float: Pressure in $\mathrm{GeV}\,\mathrm{fm}^{-3}$.
    """
    if fermi_momentum_value == 0:
        return 0.0
    if mass == 0:
        return fermi_momentum_value**4/(12*PI2*HBAR3)
    fermi_energy = sqrt(fermi_momentum_value*fermi_momentum_value+mass*mass)
    return (fermi_momentum_value*fermi_energy*(2*fermi_momentum_value*fermi_momentum_value-3*mass*mass)+3*mass**4*log(abs((fermi_momentum_value+fermi_energy)/mass)))/(24*PI2*HBAR3)


# Private source-style names remain available for existing callers.
_parameters = resolve_parameters
_scalar = scalar_density_integral
_kinetic_energy = kinetic_energy_integral
_kinetic_pressure = kinetic_pressure_integral


def effective_masses(fields: MesonFields, parameters: RMFParameters | None = None) -> tuple[float, float]:
    # Formula: M*_n = M - g_s*sigma*hbar^(3/2) + g_d*delta*hbar^(3/2),
    #          M*_p = M - g_s*sigma*hbar^(3/2) - g_d*delta*hbar^(3/2).
    r"""根据标量介子场计算中子和质子的 Dirac 有效质量。

    $$
        M_{n,p}^*=M-g_\sigma\sigma(\hbar c)^{3/2}
        \pm g_\delta\delta(\hbar c)^{3/2}.
    $$

    Args:
        fields (MesonFields): Uniform $\sigma,\omega,\rho,\delta$ fields.
        parameters (RMFParameters, optional): Couplings and meson masses.

    Returns:
        tuple[float, float]: Neutron and proton effective masses in GeV.
    """
    if len(fields) != 4 or not np.all(np.isfinite(fields)):
        raise ValueError("four finite meson fields are required")
    parameters = resolve_parameters(parameters)
    effective_scalar_mass = XM-parameters.g_s*fields.sigma*HBAR15
    delta_mass_shift = parameters.g_d*fields.delta*HBAR15
    return effective_scalar_mass+delta_mass_shift, effective_scalar_mass-delta_mass_shift


def scalar_densities(density: float, fields: MesonFields, asymmetry: float = 0.0,
                     parameters: RMFParameters | None = None) -> tuple[float, float]:
    # Formula: n_n^s = n_s(k_F[n*(1+alpha)/2], M*_n),
    #          n_p^s = n_s(k_F[n*(1-alpha)/2], M*_p).
    r"""分别计算中子和质子标量密度。

    The species densities are $n_{n,p}=n(1\pm\alpha)/2$; each scalar
    density is evaluated with its own Fermi momentum and effective mass.

    Args:
        density (float): Total baryon density in $\mathrm{fm}^{-3}$.
        fields (MesonFields): Uniform meson fields.
        asymmetry (float): $\alpha=(n_n-n_p)/n$.
        parameters (RMFParameters, optional): RMF couplings.

    Returns:
        tuple[float, float]: Neutron and proton scalar densities in
        $\mathrm{fm}^{-3}$.
    """
    _validate(density, asymmetry)
    neutron_mass, proton_mass = effective_masses(fields, parameters)
    return (scalar_density_integral(fermi_momentum(density*(1+asymmetry)/2), neutron_mass),
            scalar_density_integral(fermi_momentum(density*(1-asymmetry)/2), proton_mass))


def chemical_potentials(density: float, fields: MesonFields, asymmetry: float = 0.0,
                        parameters: RMFParameters | None = None) -> tuple[float, float]:
    # Formula: mu_n = E_F,n^* + (g_w*omega - g_r*rho)*hbar^(3/2),
    #          mu_p = E_F,p^* + (g_w*omega + g_r*rho)*hbar^(3/2).
    r"""计算中子和质子的化学势。

    $$
        \mu_{n,p}=\sqrt{k_{F,n,p}^2+M_{n,p}^{*2}}
        +g_\omega\omega(\hbar c)^{3/2}
        \mp g_\rho\rho(\hbar c)^{3/2}.
    $$

    Args:
        density (float): Total baryon density in $\mathrm{fm}^{-3}$.
        fields (MesonFields): Uniform meson fields.
        asymmetry (float): $(n_n-n_p)/n$.
        parameters (RMFParameters, optional): RMF couplings.

    Returns:
        tuple[float, float]: Neutron and proton chemical potentials in GeV.
    """
    _validate(density, asymmetry)
    parameters = resolve_parameters(parameters)
    neutron_mass, proton_mass = effective_masses(fields, parameters)
    neutron_fermi_momentum = fermi_momentum(density*(1+asymmetry)/2)
    proton_fermi_momentum = fermi_momentum(density*(1-asymmetry)/2)
    return (sqrt(neutron_mass*neutron_mass+neutron_fermi_momentum*neutron_fermi_momentum)+(parameters.g_w*fields.omega-parameters.g_r*fields.rho)*HBAR15,
            sqrt(proton_mass*proton_mass+proton_fermi_momentum*proton_fermi_momentum)+(parameters.g_w*fields.omega+parameters.g_r*fields.rho)*HBAR15)


def field_residuals(fields: MesonFields, density: float, asymmetry: float = 0.0,
                    parameters: RMFParameters | None = None) -> np.ndarray:
    # Formula: F_a = m_a^2*phi_a/hbar^(3/2) - g_a*(source_a + self/cross terms).
    r"""计算均匀核物质四个耦合介子场方程的残差。

    The residual is zero at a solution of

    $$
        \begin{aligned}
        0 &= \frac{m_\sigma^2\sigma}{H}-g_\sigma\left[n_s-b_\sigma Mx_\sigma^2
          -c_\sigma x_\sigma^3H+c_{\delta\sigma}x_\sigma x_\delta^2H\right],\\
        0 &= \frac{m_\omega^2\omega}{H}-g_\omega\left[n-c_\omega x_\omega^3H
          -\lambda_vx_\omega x_\rho^2H\right],\\
        0 &= \frac{m_\rho^2\rho}{H}-g_\rho\left[-\alpha n
          -\lambda_vx_\rho x_\omega^2H\right],\\
        0 &= \frac{m_\delta^2\delta}{H}-g_\delta\left[n_p^s-n_n^s
          +c_{\delta\sigma}x_\sigma^2x_\delta H\right].
        \end{aligned}
    $$

    Here $x_a=g_a\phi_a$, $H=(\hbar c)^{3/2}$,
    $n_s=n_n^s+n_p^s$, and $\alpha=(n_n-n_p)/n$. This normalized
    form keeps the field and coupling units used by this module.

    The original case 10 contains a debugging STOP after evaluating these
    residuals; removing that STOP makes its beta-equilibrium solver usable.
    As in that source solver, c_r and xl_s do not enter the field equations.
    """
    parameters = resolve_parameters(parameters)
    sigma_field, omega_field, rho_field, delta_field = fields
    scalar_neutron_density, scalar_proton_density = scalar_densities(
        density, fields, asymmetry, parameters)
    scalar_field = parameters.g_s*sigma_field
    vector_field = parameters.g_w*omega_field
    isovector_field = parameters.g_r*rho_field
    delta_self_energy = parameters.g_d*delta_field
    return np.array([
        sigma_field*parameters.xms**2/HBAR15-parameters.g_s*(scalar_neutron_density+scalar_proton_density-parameters.b_s*XM*scalar_field*scalar_field-parameters.c_s*scalar_field**3*HBAR15+parameters.c_ds*scalar_field*delta_self_energy*delta_self_energy*HBAR15),
        omega_field*parameters.xmw**2/HBAR15-parameters.g_w*(density-parameters.c_w*vector_field**3*HBAR15-parameters.xl_v*vector_field*isovector_field*isovector_field*HBAR15),
        rho_field*parameters.xmr**2/HBAR15-parameters.g_r*(-asymmetry*density-parameters.xl_v*isovector_field*vector_field*vector_field*HBAR15),
        delta_field*parameters.xmd**2/HBAR15-parameters.g_d*(scalar_proton_density-scalar_neutron_density+parameters.c_ds*scalar_field*scalar_field*delta_self_energy*HBAR15),
    ])


def solve_fields(density: float, asymmetry: float = 0.0,
                 parameters: RMFParameters | None = None) -> MesonFields:
    # Formula: solve F_sigma = F_omega = F_rho = F_delta = 0 for the
    #          stationary uniform-matter meson fields.
    r"""求解给定密度和不对称度下的均匀物质介子场。

    The nonlinear system is $F_i(\sigma,\omega,\rho,\delta)=0$,
    with residuals defined by :func:`field_residuals`. Symmetric matter has
    $\rho=\delta=0$; asymmetric matter also solves the isovector
    response.

    Args:
        density (float): Baryon density in $\mathrm{fm}^{-3}$.
        asymmetry (float): $(n_n-n_p)/n$, from -1 to 1.
        parameters (RMFParameters, optional): RMF parameter set.

    Returns:
        MesonFields: The $\sigma,\omega,\rho,\delta$ fields.

    Raises:
        RuntimeError: If no physical local solution with positive effective
            masses and positive $\omega$ field is found.

    This local solve does not search for a globally stable phase.
    """
    _validate(density, asymmetry)
    parameter_set = _parameters(parameters)
    if parameter_set.g_s <= 0 or parameter_set.g_w <= 0:
        raise ValueError("the meson solver requires positive sigma and omega couplings")

    def sigma_equation(sigma_field):
        # Formula: F_sigma = m_s^2*sigma/hbar^(3/2) - g_s*(n_s
        #          - b_s*M*(g_s*sigma)^2 - c_s*(g_s*sigma)^3*hbar^(3/2)).
        r"""Evaluate the scalar-field residual :math:`F_\sigma(\sigma)`."""
        neutron_scalar_density, proton_scalar_density = scalar_densities(
            density, MesonFields(sigma_field, 0), asymmetry, parameter_set)
        return (sigma_field*parameter_set.xms**2/HBAR15
                -parameter_set.g_s*(neutron_scalar_density+proton_scalar_density
                                    -parameter_set.b_s*XM*(parameter_set.g_s*sigma_field)**2
                                    -parameter_set.c_s*(parameter_set.g_s*sigma_field)**3*HBAR15))

    sigma_field = brentq(sigma_equation, 0,
                         XM/(parameter_set.g_s*HBAR15)*(1-1e-12), xtol=2e-14)
    delta_field = 0.0
    if asymmetry != 0 and parameter_set.g_d != 0:
        def scalar_equations(scalar_delta_fields):
            # Formula: return (F_sigma, F_delta) for trial (sigma, delta).
            r"""Evaluate the coupled scalar and delta residual vector."""
            trial_fields = MesonFields(scalar_delta_fields[0], 0, 0, scalar_delta_fields[1])
            return field_residuals(trial_fields, density, asymmetry, parameter_set)[[0, 3]]
        solution = root(scalar_equations, [sigma_field, 0.0], tol=1e-11)
        if np.max(np.abs(scalar_equations(solution.x))) > 1e-9:
            raise RuntimeError(f"scalar meson fields did not converge at density {density}")
        sigma_field, delta_field = solution.x

    def rho_field(omega_field):
        # Formula: rho = -g_r*n*alpha*hbar^(3/2) /
        #          [m_r^2 + xl_v*g_r^2*(g_w*omega)^2*hbar^3].
        r"""Return the rho field eliminated at a trial omega field."""
        return (-parameter_set.g_r*density*asymmetry*HBAR15
                /(parameter_set.xmr**2
                  +parameter_set.xl_v*parameter_set.g_r**2
                  *(parameter_set.g_w*omega_field)**2*HBAR3))

    def vector_equation(omega_field):
        # Formula: F_omega = m_w^2*omega/hbar^(3/2) - g_w*[n
        #          - c_w*(g_w*omega)^3*hbar^(3/2)
        #          - xl_v*(g_w*omega)*(g_r*rho)^2*hbar^(3/2)].
        r"""Evaluate the vector-field residual :math:`F_\omega(\omega)`."""
        rho_field_value = rho_field(omega_field)
        return (omega_field*parameter_set.xmw**2/HBAR15
                -parameter_set.g_w*(density
                                    -parameter_set.c_w*(parameter_set.g_w*omega_field)**3*HBAR15
                                    -parameter_set.xl_v*parameter_set.g_w*omega_field
                                    *(parameter_set.g_r*rho_field_value)**2*HBAR15))

    upper_bound = max(1.0, parameter_set.g_w*density*HBAR15/parameter_set.xmw**2)
    for _ in range(30):
        if vector_equation(upper_bound) > 0:
            break
        upper_bound *= 2
    else:
        raise RuntimeError("unable to bracket the positive omega field")
    omega_field = brentq(vector_equation, 0, upper_bound, xtol=2e-14)
    fields = MesonFields(float(sigma_field), omega_field,
                         rho_field(omega_field), float(delta_field))
    if min(effective_masses(fields, parameter_set)) <= 0:
        raise RuntimeError("field solution has a nonpositive nucleon effective mass")
    return fields


def energy_density(density: float, fields: MesonFields, asymmetry: float = 0.0,
                   parameters: RMFParameters | None = None) -> float:
    # Formula: epsilon = sum_J epsilon_kin,J + epsilon_free_fields
    #          + epsilon_nonlinear_fields, with J in {n,p}.
    #          epsilon_free = [(m_s*sigma)^2+(m_w*omega)^2
    #          +(m_r*rho)^2+(m_d*delta)^2]/2; the nonlinear part is
    #          b_s*M*S^3*hbar^(3/2)/3 + c_s*S^4*hbar^3/4
    #          + 3*(c_w*W^4+c_r*R^4)*hbar^3/4
    #          + R^2*(xl_s*S^2+3*xl_v*W^2)*hbar^3/2
    #          - c_ds*D^2*S^2*hbar^3/2, where S=g_s*sigma,
    #          W=g_w*omega, R=g_r*rho, and D=g_d*delta.
    r"""计算包含核子静质量的总重子能量密度。

    $$
        \epsilon=\sum_{J=n,p}\epsilon_{J,\mathrm{kin}}
        +\epsilon_{\mathrm{meson}}(\sigma,\omega,\rho,\delta).
    $$

    Args:
        density (float): Baryon density in $\mathrm{fm}^{-3}$.
        fields (MesonFields): Uniform meson fields.
        asymmetry (float): $(n_n-n_p)/n$.
        parameters (RMFParameters, optional): RMF couplings.

    Returns:
        float: Energy density in $\mathrm{GeV}\,\mathrm{fm}^{-3}$.
    """
    _validate(density, asymmetry)
    parameters = resolve_parameters(parameters)
    sigma_field, omega_field, rho_field, delta_field = fields
    neutron_mass, proton_mass = effective_masses(fields, parameters)
    kinetic_energy = (
        kinetic_energy_integral(fermi_momentum(density*(1+asymmetry)/2), neutron_mass)
        +kinetic_energy_integral(fermi_momentum(density*(1-asymmetry)/2), proton_mass))
    scalar_field = parameters.g_s*sigma_field
    vector_field = parameters.g_w*omega_field
    isovector_field = parameters.g_r*rho_field
    delta_self_energy = parameters.g_d*delta_field
    free_field_energy = ((parameters.xms*sigma_field)**2
                         +(parameters.xmw*omega_field)**2
                         +(parameters.xmr*rho_field)**2
                         +(parameters.xmd*delta_field)**2)/2
    nonlinear_field_energy = (parameters.b_s*XM*scalar_field**3*HBAR15/3+parameters.c_s*scalar_field**4*HBAR3/4
             +0.75*(parameters.c_w*vector_field**4+parameters.c_r*isovector_field**4)*HBAR3
             +isovector_field*isovector_field*(parameters.xl_s*scalar_field*scalar_field+3*parameters.xl_v*vector_field*vector_field)*HBAR3/2-parameters.c_ds*delta_self_energy*delta_self_energy*scalar_field*scalar_field*HBAR3/2)
    return kinetic_energy+free_field_energy+nonlinear_field_energy


def energy_per_particle(density: float, fields: MesonFields, asymmetry: float = 0.0,
                        parameters: RMFParameters | None = None) -> float:
    # Formula: e(n,alpha) = epsilon(n,alpha)/n - M.
    r"""按 $e=\epsilon/n-M$ 计算每核子结合能。

    Args:
        density (float): Baryon density in $\mathrm{fm}^{-3}$.
        fields (MesonFields): Uniform meson fields.
        asymmetry (float): $(n_n-n_p)/n$.
        parameters (RMFParameters, optional): RMF couplings.

    Returns:
        float: Binding energy per nucleon in GeV.
    """
    return energy_density(density, fields, asymmetry, parameters)/density-XM


def pressure(density: float, fields: MesonFields, asymmetry: float = 0.0,
             parameters: RMFParameters | None = None) -> float:
    # Formula: P = sum_J P_kin,J - P_free_fields - P_nonlinear_fields.
    #          P_free = [(m_s*sigma)^2-(m_w*omega)^2
    #          +(m_d*delta)^2-(m_r*rho)^2]/2; P_nonlinear uses the
    #          same S,W,R,D definitions and the coefficients in PXXX,
    #          including +c_ds*D^2*S^2*hbar^3/4.
    r"""按源程序约定计算平均场压强。

    The pressure has the kinetic-minus-scalar/vector field structure
    $P=\sum_J P_{J,\mathrm{kin}}-P_{\mathrm{field}}$.

    Args:
        density (float): Baryon density in $\mathrm{fm}^{-3}$.
        fields (MesonFields): Uniform meson fields.
        asymmetry (float): $(n_n-n_p)/n$.
        parameters (RMFParameters, optional): RMF couplings.

    Returns:
        float: Pressure in $\mathrm{GeV}\,\mathrm{fm}^{-3}$.

    PXXX uses +c_ds*d^2*s^2*HBAR3/4 in Pself. This differs from the
    coefficient implied by EdenXXX; it is retained for source fidelity.
    """
    _validate(density, asymmetry)
    parameters = resolve_parameters(parameters)
    sigma_field, omega_field, rho_field, delta_field = fields
    neutron_mass, proton_mass = effective_masses(fields, parameters)
    kinetic_pressure = (
        kinetic_pressure_integral(fermi_momentum(density*(1+asymmetry)/2), neutron_mass)
        +kinetic_pressure_integral(fermi_momentum(density*(1-asymmetry)/2), proton_mass))
    scalar_field = parameters.g_s*sigma_field
    vector_field = parameters.g_w*omega_field
    isovector_field = parameters.g_r*rho_field
    delta_self_energy = parameters.g_d*delta_field
    free_field_pressure = ((parameters.xms*sigma_field)**2
                           -(parameters.xmw*omega_field)**2
                           +(parameters.xmd*delta_field)**2
                           -(parameters.xmr*rho_field)**2)/2
    nonlinear_field_pressure = (parameters.b_s*XM*scalar_field**3*HBAR15/3+parameters.c_s*scalar_field**4*HBAR3/4
                 -(parameters.c_w*vector_field**4+parameters.c_r*isovector_field**4)*HBAR3/4
                 -isovector_field*isovector_field*(parameters.xl_s*scalar_field*scalar_field+parameters.xl_v*vector_field*vector_field)*HBAR3/2+parameters.c_ds*delta_self_energy*delta_self_energy*scalar_field*scalar_field*HBAR3/4)
    return kinetic_pressure-free_field_pressure-nonlinear_field_pressure


def symmetric_matter_response(density: float, fields: MesonFields, parameters: RMFParameters):
    # Formula: k_F = hbar*(3*pi^2*n/2)^(1/3), M* = M-g_s*sigma*hbar^(3/2),
    # E_F* = sqrt(k_F^2+M*^2), n_s = 2*n_s(k_F,M*).
    # The returned inverse responses are Q_s = m_s^2+g_s^2*C,
    # Q_w = m_w^2+3*c_w*g_w^4*omega^2*hbar^3,
    # Q_r = m_r^2+g_r^2*(xl_s*S^2+xl_v*W^2)*hbar^3,
    # Q_d = m_d^2+g_d^2*C-c_ds*g_d^2*S^2*hbar^3, with
    # C = (3*n_s/M* - 3*n/E_F*)*hbar^3.
    r"""计算对称核物质的费米量和介子响应分母。

    Args:
        density (float): Symmetric baryon density in $\mathrm{fm}^{-3}$.
        fields (MesonFields): Symmetric-matter $\sigma$ and
            $\omega$ fields.
        parameters (RMFParameters): Couplings and meson masses.

    Returns:
        tuple: Fermi quantities and inverse-response denominators used by
        incompressibility, skewness, symmetry energy, and its slope.
    """
    _validate(density)
    if len(fields) != 4 or not np.all(np.isfinite(fields)):
        raise ValueError("four finite meson fields are required")
    scalar_field, vector_field = fields.sigma, fields.omega
    fermi_momentum_value = fermi_momentum(density/2)
    effective_mass = XM-parameters.g_s*scalar_field*HBAR15
    if effective_mass <= 0:
        raise ValueError("symmetric response requires a positive effective mass")
    fermi_energy = sqrt(fermi_momentum_value*fermi_momentum_value+effective_mass*effective_mass)
    scalar_density = 2*scalar_density_integral(fermi_momentum_value, effective_mass)
    common_response = (3*scalar_density/effective_mass-3*density/fermi_energy)*HBAR3
    scalar_response = parameters.xms**2+parameters.g_s**2*common_response+2*parameters.b_s*XM*parameters.g_s**3*scalar_field*HBAR15+3*parameters.c_s*parameters.g_s**4*scalar_field*scalar_field*HBAR3
    vector_response = parameters.xmw**2+3*parameters.c_w*parameters.g_w**4*vector_field*vector_field*HBAR3
    rho_response = parameters.xmr**2+parameters.g_r**2*(parameters.xl_s*(parameters.g_s*scalar_field)**2+parameters.xl_v*(parameters.g_w*vector_field)**2)*HBAR3
    delta_response = parameters.xmd**2+parameters.g_d**2*common_response-parameters.c_ds*parameters.g_d**2*(parameters.g_s*scalar_field)**2*HBAR3
    return fermi_momentum_value, effective_mass, fermi_energy, scalar_density, scalar_response, vector_response, rho_response, delta_response


_symmetric_response = symmetric_matter_response


def incompressibility(density: float, fields: MesonFields,
                      parameters: RMFParameters | None = None) -> float:
    # Formula: K(n) = 9*dP/dn = 3*k_F^2/E_F*
    #          - 9*n*g_s^2*M*^2*hbar^3/(Q_s*E_F*^2)
    #          + 9*n*g_w^2*hbar^3/Q_w.
    r"""按 $K=9\,dP/dn$ 计算等标量压缩模量。

    $$
        K(n)=9\,\frac{dP}{dn}.
    $$

    Args:
        density (float): Symmetric baryon density in $\mathrm{fm}^{-3}$.
        fields (MesonFields): Symmetric-matter fields at this density.
        parameters (RMFParameters, optional): RMF couplings.

    Returns:
        float: Incompressibility in GeV.
    """
    parameters = resolve_parameters(parameters)
    fermi_momentum_value, effective_mass, fermi_energy, scalar_density, scalar_response, vector_response, rho_response, delta_response = symmetric_matter_response(density, fields, parameters)
    return 3*fermi_momentum_value*fermi_momentum_value/fermi_energy-9*density*effective_mass*effective_mass*parameters.g_s**2*HBAR3/(scalar_response*fermi_energy*fermi_energy)+9*density*parameters.g_w**2*HBAR3/vector_response


def skewness(density: float, fields: MesonFields,
             parameters: RMFParameters | None = None) -> float:
    # Formula: J_0(n) = 27*n^3*d^3(E/A)/dn^3, evaluated through the
    #          scalar, vector, and kinetic response terms below.
    r"""计算饱和偏度参数 $J_0$ 所需的三阶密度响应。

    At saturation the convention is
    $J_0=27n_0^3\,d^3(E/A)/dn^3|_{n_0}$.

    Args:
        density (float): Symmetric baryon density in $\mathrm{fm}^{-3}$.
        fields (MesonFields): Symmetric-matter fields at this density.
        parameters (RMFParameters, optional): RMF couplings.

    Returns:
        float: Skewness coefficient in GeV.
    """
    parameters = resolve_parameters(parameters)
    fermi_momentum_value, effective_mass, fermi_energy, scalar_density, scalar_response, vector_response, rho_response, delta_response = symmetric_matter_response(density, fields, parameters)
    scalar_derivative = (3*parameters.g_s**3*(2*scalar_density/effective_mass**2-3*density/(fermi_energy*effective_mass)+effective_mass*density/fermi_energy**3)*HBAR3
           -2*parameters.b_s*XM*parameters.g_s**3-6*parameters.c_s*parameters.g_s**4*fields.sigma*HBAR15)
    kinetic_derivative = 2*parameters.g_s**2*fermi_momentum_value*fermi_momentum_value/fermi_energy**3
    scalar_contribution = 27*parameters.g_s**2*effective_mass*effective_mass*density**2*HBAR**6/(scalar_response*fermi_energy**3)*(3*PI2/(2*fermi_momentum_value*fermi_energy)
          +2*parameters.g_s**2/scalar_response-parameters.g_s*effective_mass*scalar_derivative/scalar_response**2-2*parameters.g_s**2*effective_mass*effective_mass/(scalar_response*fermi_energy*fermi_energy)+fermi_energy*kinetic_derivative/(2*scalar_response))
    vector_contribution = -162*parameters.c_w*parameters.g_w**7*fields.omega*HBAR15*density**2*HBAR**6/vector_response**3
    return -3*(fermi_momentum_value*fermi_momentum_value/fermi_energy+fermi_momentum_value**4/fermi_energy**3)+scalar_contribution+vector_contribution-9*incompressibility(density, fields, parameters)


def symmetry_energy(density: float, fields: MesonFields,
                    parameters: RMFParameters | None = None) -> float:
    # Formula: S(n) = k_F^2/(6*E_F*) + n*hbar^3/2 *
    #          [g_r^2/Q_r - g_d^2*M*^2/(Q_d*E_F*^2)].
    r"""计算能量每核子对同位旋不对称度的二次响应。

    $$
        e(n,\alpha)=e(n,0)+S(n)\alpha^2+O(\alpha^4),\qquad
        S(n)=\frac12\left.\frac{\partial^2e}{\partial\alpha^2}\right|_{\alpha=0}.
    $$

    The analytic expression includes rho and delta meson response and the
    delta-sigma cross coupling.

    Args:
        density (float): Symmetric baryon density in $\mathrm{fm}^{-3}$.
        fields (MesonFields): Symmetric-matter fields.
        parameters (RMFParameters, optional): RMF couplings.

    Returns:
        float: Symmetry energy in GeV.
    """
    parameters = resolve_parameters(parameters)
    fermi_momentum_value, effective_mass, fermi_energy, scalar_density, scalar_response, vector_response, rho_response, delta_response = symmetric_matter_response(density, fields, parameters)
    return fermi_momentum_value*fermi_momentum_value/(6*fermi_energy)+0.5*density*HBAR3*(parameters.g_r**2/rho_response-parameters.g_d**2/delta_response*effective_mass*effective_mass/(fermi_energy*fermi_energy))


def symmetry_slope_terms(density: float, fields: MesonFields,
                         parameters: RMFParameters):
    # Formula: L = 3*n*dS/dn = L_kin + L_sigma + L_rho + L_delta.
    #          Each term differentiates the density dependence of k_F, M*,
    #          and the response denominators Q_s, Q_w, Q_r, Q_d.
    #          In particular, L_kin = k_F^2*(E_F*^2+M*^2)/(6*E_F*^3)
    #          and L_sigma = hbar^3*g_s^2*M*^2*k_F^2*n/(2*Q_s*E_F*^4);
    #          L_rho and L_delta are the corresponding Q_r and Q_d
    #          response derivatives evaluated explicitly below.
    r"""分解对称能斜率的动能、标量、矢量和 delta 场贡献。

    They sum to $L=3n\,dS/dn$; all mean fields and response
    denominators are differentiated with density.

    Args:
        density (float): Symmetric baryon density in $\mathrm{fm}^{-3}$.
        fields (MesonFields): Symmetric-matter fields.
        parameters (RMFParameters): RMF couplings.

    Returns:
        tuple[float, float, float, float]: Contributions to $L$ in GeV.
    """
    fermi_momentum_value, effective_mass, fermi_energy, scalar_density, scalar_response, vector_response, rho_response, delta_response = _symmetric_response(
        density, fields, parameters)
    scalar_delta_derivative = (
        3*parameters.g_s*parameters.g_d**2
        *(2*scalar_density/effective_mass**2
          -3*density/(fermi_energy*effective_mass)
          +effective_mass*density/fermi_energy**3)*HBAR3
        +2*parameters.c_ds*parameters.g_d**2*parameters.g_s**2
        *fields.sigma*HBAR15)
    delta_kinetic_derivative = 2*parameters.g_d**2*fermi_momentum_value**2/fermi_energy**3
    kinetic_slope = (fermi_momentum_value**2
                     *(fermi_energy**2+effective_mass**2)/(6*fermi_energy**3))
    scalar_slope = (HBAR3*parameters.g_s**2*effective_mass**2
                    *fermi_momentum_value**2*density
                    /(2*scalar_response*fermi_energy**4))
    vector_slope = (
        HBAR3*3*parameters.g_r**2*density/(2*rho_response)
        -HBAR**7.5*3*parameters.g_s**3*parameters.g_r**4
        *parameters.xl_s*fields.sigma*effective_mass*density**2
        /(fermi_energy*scalar_response*rho_response**2)
        -HBAR**7.5*3*parameters.g_w**3*parameters.g_r**4
        *parameters.xl_v*fields.omega*density**2
        /(vector_response*rho_response**2))
    delta_slope = (
        -3*parameters.g_s*parameters.g_d**2*effective_mass**2*density**2*HBAR**6
        /(2*fermi_energy**3*scalar_response*delta_response)
        *(2*parameters.g_s*effective_mass**2/fermi_energy**2
          +effective_mass*scalar_delta_derivative/delta_response
          -2*parameters.g_s)
        -3*parameters.g_d**2*effective_mass**2*density*HBAR3
        /(2*fermi_energy**2*delta_response)
        *(1-2*fermi_momentum_value**2/(3*fermi_energy**2)
          -density*HBAR3*delta_kinetic_derivative/(2*delta_response)))
    return kinetic_slope, scalar_slope, vector_slope, delta_slope


def _slope_terms(density: float, fields: MesonFields, p: RMFParameters):
    # Formula: _slope_terms(n,fields,p) = symmetry_slope_terms(n,fields,p).
    """Compatibility wrapper retaining the historical ``p`` keyword."""
    return symmetry_slope_terms(density, fields, p)


def symmetry_slope(density: float, fields: MesonFields,
                   parameters: RMFParameters | None = None) -> float:
    # Formula: L(n) = 3*n*dS/dn = sum(symmetry_slope_terms).
    r"""按 $L=3n\,dS/dn$ 计算对称能斜率。

    Args:
        density (float): Symmetric baryon density in $\mathrm{fm}^{-3}$.
        fields (MesonFields): Symmetric-matter fields.
        parameters (RMFParameters, optional): RMF couplings.

    Returns:
        float: Symmetry-energy slope in GeV.
    """
    return sum(symmetry_slope_terms(density, fields, _parameters(parameters)))


def calibrate_isoscalar(parameters: RMFParameters) -> tuple[RMFParameters, MesonFields]:
    # Formula: impose e(n0) = binding_energy, P(n0) = 0,
    #          K(n0) = incompressibility, and M*(n0)/M = effective_mass_ratio.
    #          The resulting (g_s,g_w,b_s,c_s) solve these four constraints
    #          while c_w and all isovector parameters remain fixed.
    """rmf_parameters: derive g_s, g_w, b_s, c_s at fixed c_w."""
    parameter_set = parameters
    saturation_density = parameter_set.saturation_density
    _validate(saturation_density)
    effective_mass = parameter_set.effective_mass_ratio*XM
    scalar_self_energy = XM-effective_mass
    fermi_momentum_value = fermi_momentum(saturation_density/2)
    fermi_energy = sqrt(effective_mass*effective_mass+fermi_momentum_value*fermi_momentum_value)
    vector_self_energy = parameter_set.binding_energy/1000+XM-fermi_energy
    scalar_density = 2*scalar_density_integral(fermi_momentum_value, effective_mass)
    vector_coupling = sqrt(parameter_set.xmw**2*vector_self_energy/(saturation_density*HBAR3-parameter_set.c_w*vector_self_energy**3))
    vector_response = HBAR3/(parameter_set.xmw**2/vector_coupling**2+3*parameter_set.c_w*vector_self_energy*vector_self_energy)
    scalar_response = (-scalar_self_energy*scalar_self_energy*effective_mass*effective_mass/fermi_energy**2/(parameter_set.incompressibility/1000/9/saturation_density-fermi_momentum_value*fermi_momentum_value/(3*fermi_energy*saturation_density)-vector_response)
          -3*scalar_self_energy*scalar_self_energy*(scalar_density/effective_mass-saturation_density/fermi_energy))*HBAR3
    energy_response = (2*saturation_density*(parameter_set.binding_energy/1000+XM)*HBAR3-0.5*(3*fermi_energy*saturation_density+effective_mass*scalar_density)*HBAR3
          -parameter_set.xmw**2*vector_self_energy*vector_self_energy/vector_coupling**2-1.5*parameter_set.c_w*vector_self_energy**4)
    scalar_quartic = (scalar_response+3*energy_response-4*scalar_self_energy*scalar_density*HBAR3)*2/scalar_self_energy**4
    scalar_cubic = (15*scalar_self_energy*scalar_density*HBAR3-3*scalar_response-12*energy_response)/(XM*scalar_self_energy**3)
    scalar_coupling = sqrt(scalar_self_energy*scalar_self_energy/(6*energy_response+scalar_response-6*scalar_self_energy*scalar_density*HBAR3))*parameter_set.xms
    calibrated = replace(parameter_set, g_s=scalar_coupling, g_w=vector_coupling, b_s=scalar_cubic, c_s=scalar_quartic)
    return calibrated, MesonFields(scalar_self_energy/(scalar_coupling*HBAR15), vector_self_energy/(vector_coupling*HBAR15))


def calibrate_isovector(parameters: RMFParameters, fields: MesonFields,
                         density: float | None = None) -> RMFParameters:
    # Formula: S_rho = S_target - S_kin - S_delta and
    #          L_rho = L_target - L_kin - L_sigma - L_delta.
    #          Solve S_rho = n*hbar^3*g_r^2/(2*Q_r) for (g_r, xl_v).
    """z_l_xy: determine g_r and xl_v from symmetry energy and its slope."""
    parameter_set = parameters
    selected_density = (parameter_set.saturation_density
                        if density is None else density)
    (fermi_momentum_value, effective_mass, fermi_energy, scalar_density,
     scalar_response, vector_response, rho_response,
     delta_response) = _symmetric_response(selected_density, fields, parameter_set)
    vector_symmetry_energy = (
        parameter_set.symmetry_energy/1000
        -fermi_momentum_value**2/(6*fermi_energy)
        +0.5*selected_density*HBAR3*parameter_set.g_d**2
        /delta_response*effective_mass**2/fermi_energy**2)
    kinetic_slope, scalar_slope, _, delta_slope = symmetry_slope_terms(
        selected_density, fields, parameter_set)
    vector_symmetry_slope = (parameter_set.symmetry_slope/1000
                             -kinetic_slope-scalar_slope-delta_slope)
    vector_rho_mixing = (
        (3*vector_symmetry_energy-vector_symmetry_slope)
        /12/HBAR15/fields.omega*vector_response/parameter_set.g_w**3
        /vector_symmetry_energy**2)
    rho_coupling = parameter_set.xmr/sqrt(
        selected_density*HBAR3*0.5/vector_symmetry_energy
        -vector_rho_mixing*parameter_set.g_w**2*fields.omega**2*HBAR3)
    return replace(parameter_set, xl_v=vector_rho_mixing, g_r=rho_coupling)


def default_parameters() -> RMFParameters:
    # Formula: p0 = calibrate_isovector(*calibrate_isoscalar(RMFParameters())).
    """Calibrated parameters for the default input values in Fortran main."""
    parameters, fields = calibrate_isoscalar(RMFParameters())
    return calibrate_isovector(parameters, fields)


def saturation_fields(parameters: RMFParameters) -> MesonFields:
    # Formula: sigma_0 = (M-M*_0)/(g_s*hbar^(3/2)),
    #          omega_0 = (B_0+M-E_F,0^*)/(g_w*hbar^(3/2)).
    """Return the scalar and vector fields at the calibrated saturation point."""
    effective_mass = parameters.effective_mass_ratio*XM
    fermi_energy = sqrt(
        effective_mass*effective_mass
        +fermi_momentum(parameters.saturation_density/2)**2)
    return MesonFields(
        (XM-effective_mass)/(parameters.g_s*HBAR15),
        (parameters.binding_energy/1000+XM-fermi_energy)
        /(parameters.g_w*HBAR15))


def _saturation_fields(p: RMFParameters) -> MesonFields:
    # Formula: _saturation_fields(p) = saturation_fields(p).
    """Compatibility wrapper retaining the historical ``p`` keyword."""
    return saturation_fields(p)


def parameters_from_scalar_quartic(parameters: RMFParameters,
                                    scalar_quartic: float):
    # Formula: eliminate (g_s,g_w,b_s,c_w) analytically at fixed c_s,
    #          using the same saturation equations and K(n0) as calibration.
    """Analytic coupling elimination from funcv case 9."""
    saturation_density = parameters.saturation_density
    effective_mass = parameters.effective_mass_ratio*XM
    scalar_self_energy = XM-effective_mass
    fermi_momentum_value = fermi_momentum(saturation_density/2)
    fermi_energy = sqrt(effective_mass*effective_mass+fermi_momentum_value**2)
    vector_self_energy = parameters.binding_energy/1000+XM-fermi_energy
    scalar_density = 2*scalar_density_integral(fermi_momentum_value, effective_mass)
    incompressibility_term = (parameters.incompressibility/9000
                              -fermi_momentum_value**2/(3*fermi_energy)) / saturation_density
    energy_term = (
        (12*(XM+parameters.binding_energy/1000)*saturation_density
         -9*fermi_energy*saturation_density-3*effective_mass*scalar_density
         -(8*scalar_density*scalar_self_energy
           +4.5*saturation_density*vector_self_energy)
         -(6*scalar_density/effective_mass
           -6*saturation_density/fermi_energy)*scalar_self_energy**2
         -scalar_quartic*scalar_self_energy**4/HBAR3))
    scalar_coefficient, vector_coefficient, mass_coefficient = (
        -2*scalar_self_energy**2,
        1.5*vector_self_energy**2,
        effective_mass**2/fermi_energy**2)
    discriminant_term = (incompressibility_term*energy_term
                         -scalar_coefficient*mass_coefficient
                         -vector_coefficient)
    scalar_response_root = (
        (-discriminant_term
         -sqrt(discriminant_term**2
               +4*energy_term*mass_coefficient*scalar_coefficient
               *incompressibility_term))*0.5
        /(energy_term*mass_coefficient))
    vector_response_root = incompressibility_term + mass_coefficient*scalar_response_root
    vector_quartic = ((0.5*vector_self_energy/vector_response_root
                       -0.5*saturation_density)
                      *(HBAR/vector_self_energy)**3)
    vector_coupling = sqrt(
        parameters.xmw**2*vector_self_energy
        /(1.5*saturation_density
          -0.5*vector_self_energy/vector_response_root)/HBAR3)
    scalar_cubic = (
        (scalar_self_energy/scalar_response_root-scalar_density
         -(3*scalar_density/effective_mass
           -3*saturation_density/fermi_energy)*scalar_self_energy
         -2*scalar_quartic*scalar_self_energy**3/HBAR3)
        *HBAR3/(XM*scalar_self_energy**2))
    scalar_coupling = sqrt(
        parameters.xms**2*scalar_self_energy
        /(scalar_density
          -scalar_cubic*XM*scalar_self_energy**2/HBAR3
          -scalar_quartic*scalar_self_energy**3/HBAR3)/HBAR3)
    calibrated_parameters = replace(
        parameters, g_s=scalar_coupling, g_w=vector_coupling,
        b_s=scalar_cubic, c_s=scalar_quartic, c_w=vector_quartic)
    response_values = (fermi_momentum_value, effective_mass, fermi_energy,
                       scalar_density, scalar_self_energy, vector_self_energy,
                       scalar_response_root, vector_response_root)
    return calibrated_parameters, saturation_fields(calibrated_parameters), response_values


def _parameters_from_scalar_quartic(p: RMFParameters, cs: float):
    # Formula: _parameters_from_scalar_quartic(p,cs)
    #          = parameters_from_scalar_quartic(p,cs).
    """Compatibility wrapper retaining historical ``p`` and ``cs`` keywords."""
    return parameters_from_scalar_quartic(p, cs)


def calibrate_with_skewness(parameters: RMFParameters, target_skewness: float,
                            *, method: str = "nonlinear") -> RMFParameters:
    # Formula: solve J_0(n0) = target_skewness, together with the four
    #          isoscalar saturation constraints, for the five isoscalar
    #          couplings (g_s,b_s,c_s,g_w,c_w).
    """Fit five isoscalar couplings with J0 in MeV (funcv cases 1 and 9).

    ``nonlinear`` solves case 1's five equations; ``analytic`` uses the
    single-variable elimination in case 9. Existing isovector couplings
    are retained; call calibrate_isovector afterward to refit Esym and L.
    """
    parameter_set = parameters
    if not np.isfinite(target_skewness):
        raise ValueError("target skewness must be finite (MeV)")
    if method == "nonlinear":
        initial_couplings = [parameter_set.g_s, parameter_set.b_s,
                             parameter_set.c_s, parameter_set.g_w,
                             parameter_set.c_w]
        residual_function = lambda coupling_values: funcv(
            1, coupling_values, parameter_set.saturation_density,
            parameters=parameter_set,
            target_skewness=target_skewness)
        solution = root(residual_function, initial_couplings, tol=1e-10)
        if np.max(np.abs(residual_function(solution.x))) > 1e-6:
            raise RuntimeError(f"coupling calibration failed: {solution.message}")
        scalar_coupling, scalar_cubic, scalar_quartic, vector_coupling, vector_quartic = solution.x
        return replace(parameter_set, g_s=scalar_coupling, b_s=scalar_cubic,
                       c_s=scalar_quartic, g_w=vector_coupling,
                       c_w=vector_quartic)
    if method == "analytic":
        residual_function = lambda coupling_values: funcv(
            9, coupling_values, parameter_set.saturation_density,
            parameters=parameter_set,
            target_skewness=target_skewness)
        solution = root(residual_function, [parameter_set.c_s], tol=1e-10)
        if np.max(np.abs(residual_function(solution.x))) > 1e-8:
            raise RuntimeError(f"analytic coupling calibration failed: {solution.message}")
        return parameters_from_scalar_quartic(parameter_set, solution.x[0])[0]
    raise ValueError("method must be 'nonlinear' or 'analytic'")


def _legacy_scalar_newton(function, initial: float) -> float:
    # Formula: x_(i+1) = x_i - F(x_i)/F'(x_i),
    #          F'(x_i) ~= [F(x_i+h)-F(x_i)]/h, with backtracking.
    """Match newt's finite-difference Jacobian and stopping tolerances.

    EOS.OUT retains the small residual left by this early stopping rule.
    Other public solvers use tighter SciPy tolerances. A backtracking line
    search supplies the same accepted full steps on the original EOS grid.
    """
    current_value = initial
    residual = function(current_value)
    relative_step = single_precision_real(1e-4)
    if abs(residual) < single_precision_real(0.01)*single_precision_real(1e-5):
        return current_value
    for _ in range(1000):
        finite_difference_step = (relative_step*abs(current_value)
                                   if current_value != 0 else relative_step)
        finite_difference_step = ((current_value+finite_difference_step)
                                  -current_value)
        derivative = (function(current_value+finite_difference_step)-residual)
        derivative /= finite_difference_step
        step = -residual/derivative
        old_value = current_value
        factor = 1.0
        while factor > 1e-10:
            current_value = old_value+factor*step
            candidate = function(current_value)
            if candidate*candidate <= residual*residual*(1-2*relative_step*factor):
                break
            factor *= 0.5
        residual = candidate
        if (abs(residual) < single_precision_real(1e-5)
                or abs(current_value-old_value)/max(abs(current_value), 1)
                < relative_step):
            return current_value
    raise RuntimeError("legacy EOS Newton iteration did not converge")


def _eos_fields(density: float, parameters: RMFParameters) -> MesonFields:
    # Formula: solve F_sigma(sigma)=0 and F_omega(g_w*omega)=0 on the
    #          legacy symmetric branch, then set rho=delta=0.
    """The specific case-3/case-6 solve used by eos_rmf."""
    _validate(density)
    def scalar_equation(sigma_field):
        # Formula: F_sigma = m_s^2*sigma/hbar^(3/2) - g_s*(n_s
        #          - b_s*M*(g_s*sigma)^2 - c_s*(g_s*sigma)^3*hbar^(3/2)).
        r"""Evaluate the legacy symmetric scalar-field residual."""
        scalar_density = rhoS0(density, sigma_field, parameters)
        scalar_self_energy = parameters.g_s*sigma_field
        return (sigma_field*parameters.xms**2/HBAR15
                -parameters.g_s*(scalar_density
                        -parameters.b_s*XM*scalar_self_energy**2
                        -parameters.c_s*scalar_self_energy**3*HBAR15))
    sigma_field = _legacy_scalar_newton(
        scalar_equation, parameters.g_s*density*HBAR15/parameters.xms**2)
    omega_times_coupling = _legacy_scalar_newton(
        lambda omega_value: field_equation(omega_value, density, 0, parameters),
        parameters.g_s*sigma_field*single_precision_real(0.9))
    return MesonFields(sigma_field,
                       omega_times_coupling/parameters.g_w, -0.0, 0.0)


def equation_of_state(parameters: RMFParameters | None = None,
                      densities=None) -> np.ndarray:
    # Formula: each row is (n, 1000*(epsilon/n)-939, sigma, omega, rho,
    #          epsilon/n-M, S(n), L(n)); the default grid is
    #          n_j=float32(0.1)+j*float32(0.0001), j=0,...,2000.
    """Return the original 8-column symmetric-matter EOS table.

    Columns: density (fm^-3), E/A-939 (MeV), sigma, omega, rho,
    E/A-XM (GeV), Esym (GeV), L (GeV). Field units are sqrt(fm^-3/GeV).
    The default density grid reproduces Fortran's REAL*8 DO loop with
    binary32 start and step literals: 2001 rows from about 0.1 to 0.3.
    """
    parameter_set = _parameters(parameters)
    if densities is None:
        densities = (single_precision_real(0.1)
                     +np.arange(2001)*single_precision_real(0.0001))
    grid = np.asarray(densities, dtype=float)
    if grid.ndim != 1:
        raise ValueError("densities must be a one-dimensional sequence")
    result = np.empty((len(grid), 8))
    for row_index, density_value in enumerate(grid):
        fields = _eos_fields(float(density_value), parameter_set)
        energy_per_nucleon = energy_density(
            density_value, fields, parameters=parameter_set)/density_value
        result[row_index] = (
            density_value, energy_per_nucleon*1000-939,
            fields.sigma, fields.omega, fields.rho,
            energy_per_nucleon-XM,
            symmetry_energy(density_value, fields, parameter_set),
            symmetry_slope(density_value, fields, parameter_set))
    return result


def beta_equilibrium(density: float, parameters: RMFParameters | None = None) -> BetaEquilibriumResult:
    # Formula: solve F(alpha) = n*(1-alpha)/2 - n_e(mu) - n_mu(mu) = 0,
    #          where mu = mu_n - mu_p and
    #          n_l(mu) = [max(mu^2-m_l^2,0)]^(3/2)/(3*pi^2*hbar^3).
    r"""求解电中性、满足 beta 平衡的中子星物质组分。

    The equilibrium conditions are

    $$
        \mu_n-\mu_p=\mu_e=\mu_\mu,\qquad n_p=n_e+n_\mu.
    $$

    Args:
        density (float): Baryon density in $\mathrm{fm}^{-3}$.
        parameters (RMFParameters, optional): RMF couplings.

    Returns:
        BetaEquilibriumResult: Composition, fields, energy density, and
        pressure. Energy density and pressure are in
        $\mathrm{MeV}\,\mathrm{fm}^{-3}$.

    This implements the equations beyond the original debugging STOP.
    It inherits solve_fields' physical-branch and convergence requirements,
    and pressure deliberately retains the documented PXXX coefficient.
    """
    _validate(density)
    parameter_set = _parameters(parameters)

    def species_density(chemical_potential, particle_mass):
        # Formula: n_l(mu) = (mu^2-m_l^2)^(3/2)/(3*pi^2*hbar^3) for mu>m_l,
        #          and n_l=0 otherwise.
        r"""Return the zero-temperature lepton density above threshold."""
        return (max(chemical_potential**2-particle_mass**2, 0)**1.5
                /(3*PI2*HBAR3)
                if chemical_potential > particle_mass else 0.0)

    def evaluate(asymmetry_value):
        # Formula: evaluate F(alpha), fields(alpha), mu=mu_n-mu_p,
        #          and the electron/muon densities at one trial alpha.
        r"""Evaluate charge residual and equilibrium quantities at alpha."""
        fields = solve_fields(density, asymmetry_value, parameter_set)
        neutron_chemical_potential, proton_chemical_potential = chemical_potentials(
            density, fields, asymmetry_value, parameter_set)
        lepton_chemical_potential = neutron_chemical_potential-proton_chemical_potential
        # Charge neutrality fixes the proton density to the occupied lepton seas.
        electron_density = species_density(lepton_chemical_potential, ELECTRON_MASS)
        muon_density = species_density(lepton_chemical_potential, MUON_MASS)
        return (density*(1-asymmetry_value)/2-electron_density-muon_density,
                fields, lepton_chemical_potential, electron_density, muon_density)

    charge_residual = lambda asymmetry_value: evaluate(asymmetry_value)[0]
    try:
        asymmetry = brentq(charge_residual, 0, 1, xtol=2e-12)
    except (RuntimeError, ValueError):
        # At high density the alpha=1 endpoint can leave the physical
        # effective-mass branch even though charge neutrality has a valid root.
        valid_points = []
        for trial_asymmetry in np.linspace(0, 1, 65):
            try:
                valid_points.append((trial_asymmetry,
                                     charge_residual(trial_asymmetry)))
            except (RuntimeError, ValueError):
                continue
        brackets = [(left, right) for left, right in zip(valid_points, valid_points[1:])
                    if left[1] == 0 or left[1] * right[1] < 0]
        if not brackets:
            raise RuntimeError(
                f"charge-neutral beta-equilibrium branch not found at density {density}"
            )
        lower, upper = brackets[0]
        asymmetry = brentq(charge_residual, lower[0], upper[0], xtol=2e-12)
    _, fields, lepton_chemical_potential, electron_density, muon_density = evaluate(asymmetry)
    electron_energy = _kinetic_energy(fermi_momentum(electron_density), ELECTRON_MASS)
    muon_energy = _kinetic_energy(fermi_momentum(muon_density), MUON_MASS)
    energy_density_value = (energy_density(density, fields, asymmetry, parameter_set)
                            +electron_energy+muon_energy)
    pressure_value = (pressure(density, fields, asymmetry, parameter_set)
                      +lepton_chemical_potential*(electron_density+muon_density)
                      -electron_energy-muon_energy)
    return BetaEquilibriumResult(
        density, asymmetry, energy_density_value*1000, pressure_value*1000,
        electron_density/density, muon_density/density,
        lepton_chemical_potential, fields)


# Original public physics names, with explicit state replacing Fortran globals.
eos_rmf = equation_of_state
betastab_rmf = beta_equilibrium
rmf_parameters = calibrate_isoscalar
z_l_xy = calibrate_isovector
mesonfield = solve_fields
EdenXXX = energy_density
cb_E = energy_per_particle
PXXX = pressure
XXK0 = incompressibility
XXJ0 = skewness
Esym = symmetry_energy
XXL = symmetry_slope


def EqState(density: float, fields: MesonFields,
            parameters: RMFParameters | None = None) -> float:
    # Formula: EqState = epsilon(n,0)/n - M, using only sigma and omega fields.
    """Symmetric binding energy in GeV; ignore isovector fields as Fortran did."""
    return energy_per_particle(density, MesonFields(fields.sigma, fields.omega), parameters=parameters)


def Press0(density: float, fields: MesonFields,
           parameters: RMFParameters | None = None) -> float:
    # Formula: Press0 = P(n,0), using only sigma and omega fields.
    r"""Return the symmetric-matter pressure :math:`P(n,\alpha=0)`."""
    return pressure(density, MesonFields(fields.sigma, fields.omega), parameters=parameters)


def XMstar0(sigma: float, parameters: RMFParameters | None = None) -> float:
    # Formula: M*(sigma) = M - g_s*sigma*hbar^(3/2).
    r"""Return :math:`M^*=M-g_\sigma\sigma(\hbar c)^{3/2}` in GeV."""
    return XM-_parameters(parameters).g_s*sigma*HBAR15


def EFstar0(density: float, sigma: float, parameters: RMFParameters | None = None) -> float:
    # Formula: E_F*(n,sigma) = sqrt(M*(sigma)^2 + k_F(n/2)^2).
    r"""Return the symmetric-matter Fermi energy :math:`E_F^*`."""
    return sqrt(XMstar0(sigma, parameters)**2+fermi_momentum(density/2)**2)


def rhoS0(density: float, sigma: float, parameters: RMFParameters | None = None) -> float:
    # Formula: rhoS0(n,sigma) = 2*n_s(k_F(n/2), M*(sigma)).
    r"""Return :math:`\rho_s=2\rho_s(k_F(n/2),M^*)` for symmetric matter."""
    return 2*_scalar(fermi_momentum(density/2), XMstar0(sigma, parameters))


def rhoS_n(density: float, fields: MesonFields, asymmetry: float = 0.0,
           parameters: RMFParameters | None = None) -> float:
    # Formula: rhoS_n = n_s(k_F[n*(1+alpha)/2], M*_n).
    r"""Return the neutron scalar density :math:`\rho_{s,n}`."""
    return scalar_densities(density, fields, asymmetry, parameters)[0]


def rhoS_p(density: float, fields: MesonFields, asymmetry: float = 0.0,
           parameters: RMFParameters | None = None) -> float:
    # Formula: rhoS_p = n_s(k_F[n*(1-alpha)/2], M*_p).
    r"""Return the proton scalar density :math:`\rho_{s,p}`."""
    return scalar_densities(density, fields, asymmetry, parameters)[1]


def XXmun(density: float, fields: MesonFields, asymmetry: float = 0.0,
           parameters: RMFParameters | None = None) -> float:
    # Formula: XXmun = mu_n = E_F,n^* + (g_w*omega-g_r*rho)*hbar^(3/2).
    r"""Return :math:`\mu_n=E_{F,n}^*+(g_\omega\omega-g_\rho\rho)(\hbar c)^{3/2}`."""
    return chemical_potentials(density, fields, asymmetry, parameters)[0]


def XXmup(density: float, fields: MesonFields, asymmetry: float = 0.0,
           parameters: RMFParameters | None = None) -> float:
    # Formula: XXmup = mu_p = E_F,p^* + (g_w*omega+g_r*rho)*hbar^(3/2).
    r"""Return :math:`\mu_p=E_{F,p}^*+(g_\omega\omega+g_\rho\rho)(\hbar c)^{3/2}`."""
    return chemical_potentials(density, fields, asymmetry, parameters)[1]


def vector_symmetry_energy_contribution(
        target: float, density: float, fields: MesonFields,
        parameters: RMFParameters | None = None) -> float:
    # Formula: S_rho = S_target - k_F^2/(6*E_F*)
    #          + n*hbar^3*g_d^2*M*^2/(2*Q_d*E_F*^2).
    """Vector contribution required for a specified symmetry energy in GeV."""
    parameter_set = _parameters(parameters)
    (fermi_momentum_value, effective_mass, fermi_energy, scalar_density,
     scalar_response, vector_response, rho_response,
     delta_response) = _symmetric_response(density, fields, parameter_set)
    return (target-fermi_momentum_value**2/(6*fermi_energy)
            +0.5*density*HBAR3*parameter_set.g_d**2/delta_response
            *effective_mass**2/fermi_energy**2)


# Original public physics name retained for existing callers.
z_esym_xy = vector_symmetry_energy_contribution


def field_equation(omega_times_coupling: float, density: float, asymmetry: float,
                   parameters: RMFParameters | None = None) -> float:
    # Formula: F_omega(w) = w*(m_w/g_w)^2/hbar^(3/2) - n
    #          + c_w*w^3*hbar^(3/2) + xl_v*w*rho(w)^2*hbar^(3/2),
    #          where rho(w) = -n*alpha /
    #          [(m_r/g_r)^2/hbar^(3/2) + xl_v*w^2*hbar^(3/2)].
    """Reduced omega equation from Fortran field_equation / funcv case 6."""
    parameter_set = _parameters(parameters)
    omega_value = omega_times_coupling
    if parameter_set.g_r == 0:
        rho_value = 0.0
    else:
        rho_value = (-density*asymmetry
                     /((parameter_set.xmr/parameter_set.g_r)**2/HBAR15
                       +parameter_set.xl_v*omega_value**2*HBAR15))
    return (omega_value*(parameter_set.xmw/parameter_set.g_w)**2/HBAR15
            -density+parameter_set.c_w*omega_value**3*HBAR15
            +parameter_set.xl_v*omega_value*rho_value**2*HBAR15)


def meson_iteration(x, density: float, asymmetry: float,
                    parameters: RMFParameters | None = None) -> np.ndarray:
    # Formula: x_new = (g_w^2*hbar^(3/2)/m_w^2)*[n-c_w*w^3*hbar^(3/2)
    #          - xl_v*w*r^2*hbar^(3/2)],
    #          y_new = (g_r^2*hbar^(3/2)/m_r^2)*[-n*alpha
    #          - xl_v*r*w^2*hbar^(3/2)].
    """One legacy fixed-point step for (g_w*omega, g_r*rho)."""
    parameter_set = _parameters(parameters)
    omega_times_coupling, rho_times_coupling = x
    return np.array([
        (density-parameter_set.c_w*omega_times_coupling**3*HBAR15
         -parameter_set.xl_v*omega_times_coupling
         *rho_times_coupling**2*HBAR15)
        *parameter_set.g_w**2/parameter_set.xmw**2*HBAR15,
        (-density*asymmetry-parameter_set.xl_v*rho_times_coupling
         *omega_times_coupling**2*HBAR15)
        *parameter_set.g_r**2/parameter_set.xmr**2*HBAR15])


def meson_iter(x, density: float, asymmetry: float,
               parameters: RMFParameters | None = None) -> np.ndarray:
    # Formula: meson_iter(x,n,alpha,p) = meson_iteration(x,n,alpha,p).
    """Compatibility wrapper retaining the historical ``x`` argument."""
    return meson_iteration(x, density, asymmetry, parameters)


def real_cubic_roots(first_coefficient: float, second_coefficient: float,
                     third_coefficient: float) -> np.ndarray:
    # Formula: return the real roots of x^3 + a*x^2 + b*x + c = 0.
    """Real roots of x^3+a*x^2+b*x+c=0, replacing the legacy cubic solver."""
    roots = np.roots([1, first_coefficient, second_coefficient,
                      third_coefficient])
    return np.sort(roots.real[np.abs(roots.imag) < 1e-7])


def cbceq(a: float, b: float, c: float) -> np.ndarray:
    # Formula: cbceq(a,b,c) = real_cubic_roots(a,b,c).
    """Compatibility wrapper retaining historical cubic coefficient names."""
    return real_cubic_roots(a, b, c)


def funcv(mode: int, x, density: float, asymmetry: float = 0.0,
          parameters: RMFParameters | None = None, *, target_skewness: float | None = None) -> np.ndarray:
    # Formula: return the selected legacy residual vector F_mode(x)=0.
    #          Modes 0,3,4,6,7,8,10 are field equations; modes 1,2,9
    #          append saturation, beta-equilibrium, or skewness constraints.
    #          Mode 1 is (F_sigma,F_omega,e-B,K-K0,J0-J_target), while
    #          mode 2 contains (F_sigma,F_omega,F_rho,mu_e+mu_p-mu_n,
    #          mu_mu+mu_p-mu_n). Modes 3-7 are reduced sigma/omega/rho
    #          residuals, and mode 9 is the scalar-quartic J0 residual.
    """Legacy nonlinear residual modes, with explicit immutable state.

    ``x`` has the same layout as the corresponding Fortran mode. Modes 1
    and 9 require target_skewness in MeV. Mode 8 is incomplete in the source
    (its third residual is never assigned), so it raises a descriptive
    error. The debugging STOP in mode 10 is removed. Mode 0 deliberately
    retains the source's g_s rather than g_d in its delta equation; use
    field_residuals/solve_fields for the case-10 delta-sigma equations.
    """
    parameter_set = _parameters(parameters)
    x = np.asarray(x, dtype=float)
    mode_vector_sizes = {
        0: 4, 1: 5, 2: 5, 3: 1, 4: 2, 5: 2,
        6: 1, 7: 2, 8: 3, 9: 1, 10: 2}
    if (mode not in mode_vector_sizes
            or x.shape != (mode_vector_sizes[mode],)
            or not np.all(np.isfinite(x))):
        raise ValueError("invalid residual mode, vector size, or nonfinite input")
    _validate(density, asymmetry)
    if mode == 8:
        raise ValueError("Fortran funcv case 8 is incomplete: its third equation is undefined")
    if mode in (1, 9) and (target_skewness is None or not np.isfinite(target_skewness)):
        raise ValueError("calibration modes require a finite target_skewness in MeV")
    if mode == 0:
        sigma_field, omega_field, rho_field, delta_field = x
        fields = MesonFields(sigma_field, omega_field, rho_field, delta_field)
        neutron_scalar_density, proton_scalar_density = scalar_densities(
            density, fields, asymmetry, parameter_set)
        residuals = field_residuals(
            fields, density, asymmetry, replace(parameter_set, c_ds=0))
        residuals[3] = (delta_field*parameter_set.xmd**2/HBAR15
                        -parameter_set.g_s
                        *(proton_scalar_density-neutron_scalar_density))
        return residuals
    if mode == 1:
        scalar_coupling, scalar_cubic, scalar_quartic, vector_coupling, vector_quartic = x
        trial_parameters = replace(
            parameter_set, g_s=scalar_coupling, b_s=scalar_cubic,
            c_s=scalar_quartic, g_w=vector_coupling, c_w=vector_quartic)
        fields = saturation_fields(trial_parameters)
        saturation_density = parameter_set.saturation_density
        residuals = field_residuals(fields, saturation_density,
                                    parameters=trial_parameters)
        return np.array([
            residuals[0], residuals[1],
            EqState(saturation_density, fields, trial_parameters)*1000
            -parameter_set.binding_energy,
            incompressibility(saturation_density, fields, trial_parameters)*1000
            -parameter_set.incompressibility,
            skewness(saturation_density, fields, trial_parameters)*1000
            -target_skewness])
    if mode == 2:
        sigma_field, omega_field, rho_field, electron_fraction, muon_fraction = x
        if electron_fraction < 0 or muon_fraction < 0:
            raise ValueError("lepton fractions must be nonnegative")
        asymmetry_value = 1-2*electron_fraction-2*muon_fraction
        fields = MesonFields(sigma_field, omega_field, rho_field)
        residuals = field_residuals(fields, density, asymmetry_value, parameter_set)
        neutron_chemical_potential, proton_chemical_potential = chemical_potentials(
            density, fields, asymmetry_value, parameter_set)
        # The source uses (2./3.) in these two equations, rounded as REAL.
        electron_chemical_potential = sqrt(
            ELECTRON_MASS**2
            +(3*PI2*density*HBAR3*electron_fraction)
            **single_precision_real(2/3))
        muon_chemical_potential = sqrt(
            MUON_MASS**2
            +(3*PI2*density*HBAR3*muon_fraction)
            **single_precision_real(2/3))
        return np.array([
            *residuals[:3],
            electron_chemical_potential+proton_chemical_potential
            -neutron_chemical_potential,
            muon_chemical_potential+proton_chemical_potential
            -neutron_chemical_potential
            if electron_chemical_potential > MUON_MASS else muon_fraction])
    if mode == 3:
        return field_residuals(
            MesonFields(x[0], 0), density, asymmetry, parameter_set)[:1]
    if mode == 4:
        omega_field, rho_field = x
        omega_times_coupling = parameter_set.g_w*omega_field
        rho_times_coupling = parameter_set.g_r*rho_field
        return np.array([
            omega_field*parameter_set.xmw**2/HBAR15
            -parameter_set.g_w*(density
                                -parameter_set.c_w*omega_times_coupling**3*HBAR15
                                -parameter_set.xl_v*omega_times_coupling
                                *rho_times_coupling**2*HBAR15),
            rho_field*parameter_set.xmr**2/HBAR15
            -parameter_set.g_r*(-asymmetry*density
                                -parameter_set.xl_v*rho_times_coupling
                                *omega_times_coupling**2*HBAR15)])
    if mode == 5:
        omega_times_coupling, rho_times_coupling = x
        return np.array([
            (omega_times_coupling*parameter_set.xmw/parameter_set.g_w)**2
            /HBAR15-omega_times_coupling*density
            +parameter_set.c_w*omega_times_coupling**4*HBAR15
            +parameter_set.xl_v*omega_times_coupling**2
            *rho_times_coupling**2*HBAR15,
            (rho_times_coupling*parameter_set.xmr/parameter_set.g_r)**2
            /HBAR15+rho_times_coupling*density*asymmetry
            +parameter_set.xl_v*rho_times_coupling**2
            *omega_times_coupling**2*HBAR15])
    if mode == 6:
        return np.array([field_equation(
            x[0], density, asymmetry, parameter_set)])
    if mode == 7:
        omega_times_coupling, rho_times_coupling = x
        return np.array([
            omega_times_coupling*(parameter_set.xmw/parameter_set.g_w)**2
            /HBAR15-density
            +parameter_set.c_w*omega_times_coupling**3*HBAR15,
            rho_times_coupling*(parameter_set.xmr/parameter_set.g_r)**2
            /HBAR15+density*asymmetry])
    if mode == 9:
        calibrated_parameters, fields, response = parameters_from_scalar_quartic(
            parameter_set, x[0])
        (fermi_momentum_value, effective_mass, fermi_energy, scalar_density,
         scalar_self_energy, vector_self_energy,
         scalar_response_root, vector_response_root) = response
        saturation_density = parameter_set.saturation_density
        residual = (
            (target_skewness+9*parameter_set.incompressibility)/1000
            +3*fermi_momentum_value**2/fermi_energy
            +3*fermi_momentum_value**4/fermi_energy**3)
        residual /= 27*saturation_density**2
        residual += (
            6*calibrated_parameters.c_w*vector_self_energy
            *vector_response_root**3/HBAR3
            -scalar_response_root*effective_mass**2/fermi_energy**3
            *(1.5*PI2/fermi_momentum_value/fermi_energy*HBAR3
              +3*scalar_response_root*fermi_momentum_value**2/fermi_energy**2
              -scalar_response_root**2*effective_mass
              *(6*scalar_density*XM/effective_mass**2/scalar_self_energy
                -6*saturation_density*XM/effective_mass/fermi_energy/scalar_self_energy
                -3*saturation_density*fermi_momentum_value**2
                /effective_mass/fermi_energy**3
                +2*scalar_density/scalar_self_energy**2
                -2/scalar_response_root/scalar_self_energy
                -2*calibrated_parameters.c_s*scalar_self_energy/HBAR3)))
        return np.array([residual])
    return field_residuals(
        MesonFields(x[0], 0, 0, x[1]), density, asymmetry, parameter_set)[[0, 3]]
