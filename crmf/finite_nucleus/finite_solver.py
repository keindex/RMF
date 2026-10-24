r"""Independent NumPy/SciPy implementation of the spherical Dirac-HFB solver.

Public routines follow the names in ``dirhbs_CRMF.f``. Shared arrays keep
Fortran's one-based indexing, with explicitly zero-based radial meshes.
Fields are in inverse fm; Hamiltonians, pairing gaps and energies are MeV.
"""
from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np

from ..math_utils.arrays import block_matrix
from ..math_utils.eigensolver import symmetric_diagonalization
from .numerics import large_component_orbital, small_component_orbital

# Keep the descriptive solver names available to callers of older releases.
large_component_orbital_index = large_component_orbital
small_component_orbital_index = small_component_orbital


# The packed COMMON-array view is shared with the generic array utilities.
def _matrix(array, total_dimension, column_index):
    """View a packed COMMON column, preserving the solver's old signature."""
    return block_matrix(array, total_dimension, column_index)


def _blocks(state, species_index):
    r"""Yield packed matrix metadata for species :math:`q=1,2`.

    Species ``q`` starts at packed column offset ``(q-1)n_{bx}``, so the
    returned column is :math:`c=b+(q-1)n_{bx}` for block ``b``.
    """
    for block_index in range(1, state.nb + 1):
        large_dimension, small_dimension = map(int, state.id[block_index, 1:3])
        total_dimension = large_dimension + small_dimension
        matrix_column = block_index + (species_index - 1) * state.nbx
        yield (
            block_index,
            large_dimension,
            small_dimension,
            total_dimension,
            matrix_column,
        )


def _log(state, message):
    """Append a diagnostic message when the mutable state exposes a log."""
    if hasattr(state, "log"):
        state.log.append(message)


def solve_dirac_hfb(state, species_index, lprh=False, lprl=False):
    r"""求解一种核子的 Dirac-HFB 本征问题并调节粒子数约束。

    $$
        \begin{pmatrix}h-\lambda&\Delta\\
        \Delta&-h+\lambda\end{pmatrix}
        \begin{pmatrix}U_\mu\\V_\mu\end{pmatrix}
        =E_\mu\begin{pmatrix}U_\mu\\V_\mu\end{pmatrix},\qquad
        N=\sum_\mu V_\mu^\dagger V_\mu.
    $$

    Args:
        state: Finite-nucleus working state; matrix elements and energies use
            MeV after the stored inverse-fm fields are converted.
        species_index (int): State channel 1 (neutrons) or 2 (protons).
        lprh (bool): Reserved source-compatible verbosity flag.
        lprl (bool): Record chemical-potential iteration diagnostics.

    Returns:
        None: Eigenvectors, quasiparticle energies, and chemical potential are
        written into ``state``.
    """
    chemical_potential = float(state.ala[species_index])
    lower_bound, upper_bound = chemical_potential - 100.0, chemical_potential + 100.0
    previous_number = 0.0
    previous_chemical_potential = chemical_potential
    for chemical_potential_iteration in range(1, 201):
        particle_number = 0.0
        large_state_offset = small_state_offset = 0
        for block_index, large_dimension, small_dimension, total_dimension, matrix_column in _blocks(state, species_index):
            hamiltonian = _matrix(state.hh, total_dimension, matrix_column).copy()
            pairing_matrix = _matrix(state.de, total_dimension, matrix_column)
            hamiltonian.flat[::total_dimension + 1] -= chemical_potential
            hfb_matrix = np.block([[hamiltonian, pairing_matrix], [pairing_matrix, -hamiltonian]])
            quasiparticle_energies, quasiparticle_vectors = symmetric_diagonalization(hfb_matrix)
            state.ka[block_index, species_index] = large_state_offset
            state.ka[block_index, species_index + 2] = small_state_offset
            for channel, indices, offset in (
                (species_index, range(total_dimension, total_dimension + large_dimension), large_state_offset),
                (species_index + 2, range(small_dimension - 1, -1, -1), small_state_offset),
            ):
                for state_index, column in enumerate(indices, offset + 1):
                    state.ibk[state_index, channel] = block_index
                    state.equ[state_index, channel] = quasiparticle_energies[column]
                    state.fguv[1:2 * total_dimension + 1, state_index, channel] = quasiparticle_vectors[:, column]
                    particle_number += float(quasiparticle_vectors[total_dimension:, column] @ quasiparticle_vectors[total_dimension:, column]) * state.mb[block_index]
                    if hasattr(state, "tk"):
                        orbital = large_component_orbital_index(state.kb[block_index]) if channel == species_index else small_component_orbital_index(state.kb[block_index])
                        label = state.tl[orbital] if hasattr(state, "tl") else str(orbital)
                        state.tk[state_index, channel] = f"{state_index-offset:2d}{label}{2*abs(state.kb[block_index])-1:2d}/2"
            large_state_offset += large_dimension
            small_state_offset += small_dimension
            state.kd[block_index, species_index] = large_dimension
            state.kd[block_index, species_index + 2] = small_dimension
        particle_number_error = particle_number - state.tz[species_index]
        if particle_number_error < 0:
            lower_bound = chemical_potential
        else:
            upper_bound = chemical_potential
        if chemical_potential_iteration == 1:
            candidate_chemical_potential = chemical_potential - np.clip(particle_number_error, -0.1, 0.1)
        else:
            denominator = chemical_potential - previous_chemical_potential
            slope = (particle_number - previous_number) / denominator if denominator else 0.0
            if slope == 0:
                slope = 1e-20
            candidate_chemical_potential = chemical_potential - particle_number_error / slope
            if not lower_bound <= candidate_chemical_potential <= upper_bound:
                candidate_chemical_potential = 0.5 * (lower_bound + upper_bound)
        if lprl:
            _log(state, f"Lambda {species_index} iteration {chemical_potential_iteration}: {chemical_potential:.12g}, particle error {particle_number_error:.6g}")
        if abs(candidate_chemical_potential - chemical_potential) < 1e-8:
            state.ala[species_index] = candidate_chemical_potential
            state.nk[species_index], state.nk[species_index + 2] = large_state_offset, small_state_offset
            return
        previous_chemical_potential, previous_number, chemical_potential = chemical_potential, particle_number, candidate_chemical_potential
    raise RuntimeError(f"Chemical-potential iteration failed for species {species_index}: N error {particle_number_error:g}")


def compute_oscillator_densities(state, species_index, lpr=False):
    r"""由准粒子波函数构造谐振子基底中的正常密度与反常密度。

    $$
        \rho=VV^T,\qquad \kappa=VU^T.
    $$

    Args:
        state: Finite-nucleus working state containing quasiparticle vectors.
        species_index (int): Neutron or proton channel, 1 or 2.
        lpr (bool): Reserved source-compatible verbosity flag.

    Returns:
        None: Density matrices and pair traces are stored on ``state``.
    """
    pair_trace = 0.0
    pairing_matrix_index = 0
    for block_index, large_dimension, small_dimension, total_dimension, matrix_column in _blocks(state, species_index):
        density_matrix = np.zeros((total_dimension, total_dimension))
        anomalous_density = np.zeros((large_dimension, large_dimension))
        for channel in (species_index, species_index + 2):
            begin = int(state.ka[block_index, channel]) + 1
            end = begin + int(state.kd[block_index, channel])
            quasiparticle_vectors = state.fguv[1:2 * total_dimension + 1, begin:end, channel]
            density_matrix += quasiparticle_vectors[total_dimension:] @ quasiparticle_vectors[total_dimension:].T
            anomalous_density += quasiparticle_vectors[total_dimension:total_dimension + large_dimension] @ quasiparticle_vectors[:large_dimension].T
        degeneracy = 2 * abs(state.kb[block_index])
        _matrix(state.rosh, total_dimension, matrix_column)[:] = density_matrix * degeneracy
        for second_index in range(large_dimension):
            for first_index in range(second_index, large_dimension):
                pairing_matrix_index += 1
                state.aka[pairing_matrix_index, species_index] = degeneracy * anomalous_density[first_index, second_index] * (1 if first_index == second_index else 2)
                if first_index == second_index:
                    pair_trace += state.aka[pairing_matrix_index, species_index]
    state.spk[species_index] = 0.5 * pair_trace


def update_pairing_field(state, species_index, lpr=False):
    r"""在谐振子基底中更新可分离配对场。

    For form-factor matrix $F$ and pair-density vector $\kappa$,
    the code applies $\Delta=-\tfrac12G F(F^T\kappa)$.

    Args:
        state: Working state with pairing strength and pair density.
        species_index (int): Neutron or proton channel, 1 or 2.
        lpr (bool): Reserved source-compatible verbosity flag.

    Returns:
        None: The symmetric pairing matrices are written into ``state.de``.
    """
    pairing_form_factor = state.vn[1:state.mv + 1, :state.nnmax + 1]
    pairing_projection = pairing_form_factor.T @ state.aka[1:state.mv + 1, species_index]
    pairing_values = -0.5 * state.gl[species_index] * (pairing_form_factor @ pairing_projection)
    pairing_matrix_index = 0
    for block_index, large_dimension, small_dimension, total_dimension, matrix_column in _blocks(state, species_index):
        pairing_matrix = _matrix(state.de, total_dimension, matrix_column)
        for second_index in range(large_dimension):
            for first_index in range(second_index, large_dimension):
                pairing_matrix[first_index, second_index] = pairing_matrix[second_index, first_index] = pairing_values[pairing_matrix_index]
                pairing_matrix_index += 1


def transform_densities(state, lpr=False):
    r"""将基底密度矩阵变换到径向网格，并计算密度 Laplacian。

    Radial densities are sums of oscillator products,
    $\rho(r)=\sum_{ab}\rho_{ab}R_a(r)R_b(r)$, with scalar and vector
    channels distinguished by the relative sign of the small Dirac component.

    Args:
        state: Working state with oscillator-basis density matrices.
        lpr (bool): Reserved source-compatible verbosity flag.

    Returns:
        None: Scalar/vector densities and Laplacians are stored on ``state``.
    """
    state.rs[:] = 0.0
    state.rv[:] = 0.0
    scalar_laplacian = np.zeros_like(state.rs)
    vector_laplacian = np.zeros_like(state.rv)
    radial_square = state.xh ** 2
    for block_index in range(1, state.nb + 1):
        large_dimension, small_dimension = map(int, state.id[block_index, 1:3])
        total_dimension = large_dimension + small_dimension
        for component, component_dimension, offset, orbital, sign in (
            (1, large_dimension, 0, large_component_orbital_index(state.kb[block_index]), 1),
            (2, small_dimension, large_dimension, small_component_orbital_index(state.kb[block_index]), -1),
        ):
            angular_factor = orbital * (orbital + 1)
            for second_index in range(component_dimension):
                for first_index in range(second_index, component_dimension):
                    radial_product = state.rnl[first_index + 1, orbital] * state.rnl[second_index + 1, orbital]
                    radial_derivative = state.rnl1[first_index + 1, orbital] * state.rnl1[second_index + 1, orbital]
                    radial_order = 2 * (first_index + second_index + 2 + orbital) - 1
                    laplacian = 2 * (radial_product * (radial_square + angular_factor / radial_square - radial_order) + radial_derivative)
                    for species_index in range(1, state.itx + 1):
                        matrix_column = block_index + (species_index - 1) * state.nbx
                        coefficient = state.rosh[offset + first_index + 1 + (offset + second_index) * total_dimension, matrix_column]
                        coefficient *= 1 if first_index == second_index else 2
                        state.rs[:, species_index] += sign * coefficient * radial_product
                        state.rv[:, species_index] += coefficient * radial_product
                        scalar_laplacian[:, species_index] += sign * coefficient * laplacian
                        vector_laplacian[:, species_index] += coefficient * laplacian
    for species_index in range(1, state.itx + 1):
        state.rs[:, species_index] /= state.wdcor
        state.rv[:, species_index] /= state.wdcor
        scalar_laplacian[:, species_index] /= state.wdcor * state.b0 ** 2
        vector_laplacian[:, species_index] /= state.wdcor * state.b0 ** 2
    if state.itx == 1:
        state.ro[:, 1], state.ro[:, 2] = 2 * state.rs[:, 1], 2 * state.rv[:, 1]
        state.dro[:, 1], state.dro[:, 2] = 2 * scalar_laplacian[:, 1], 2 * vector_laplacian[:, 1]
        state.ro[:, 3:5] = state.dro[:, 3:5] = 0.0
        state.drvp[:] = 0.0
    elif state.itx == 2:
        for values, scalar, vector in ((state.ro, state.rs, state.rv), (state.dro, scalar_laplacian, vector_laplacian)):
            values[:, 1] = scalar[:, 1] + scalar[:, 2]
            values[:, 2] = vector[:, 1] + vector[:, 2]
            values[:, 3] = -scalar[:, 1] + scalar[:, 2]
            values[:, 4] = -vector[:, 1] + vector[:, 2]
        state.drvp[:] = vector_laplacian[:, 2]
    else:
        raise ValueError("itx must be 1 or 2")


def update_density_dependent_couplings(state, lpr=False):
    r"""计算源程序采用的密度依赖耦合及其导数。

    The point-coupling branch uses
    $g_i(x)=a_i+(b_i+c_ix)e^{-d_ix}$, where
    $x=\rho_p/\rho_{\rm sat}$; the meson branch sets constant couplings.

    Args:
        state: Working state with density and coupling parameters.
        lpr (bool): Reserved source-compatible verbosity flag.

    Returns:
        None: Updates coupling arrays and derivatives on ``state``.
    """
    if state.ipc == 0:
        state.ff[:, 1:5, 1] = 1.0
        state.ff[:, 1:5, 2] = 0.0
    elif state.ipc == 1:
        proton_density_ratio = state.ro[:, 2] / state.dsat
        for meson_index in range(1, 5):
            scalar_coefficient, linear_coefficient, quadratic_coefficient, density_decay = state.a_m[meson_index], state.b_m[meson_index], state.c_m[meson_index], state.d_m[meson_index]
            exponential = np.exp(-density_decay * proton_density_ratio)
            state.ff[:, meson_index, 1] = scalar_coefficient + (linear_coefficient + quadratic_coefficient * proton_density_ratio) * exponential
            state.ff[:, meson_index, 2] = (quadratic_coefficient - density_decay * (linear_coefficient + quadratic_coefficient * proton_density_ratio)) * exponential / state.dsat
    else:
        raise ValueError("ipc must be 0 (meson) or 1 (point coupling)")


def apply_klein_gordon_green_function(state, meson_index, source):
    r"""Apply the discretized Klein-Gordon inverse on physical mesh points.

    For source samples :math:`s_j`, the returned field is
    :math:`\phi_i=\sum_jG_{ij}^{(a)}s_j` for meson channel ``a``.
    """
    return state.gg[1:state.ngh + 1, 1:state.ngh + 1, meson_index] @ np.asarray(source)


def solve_meson_fields(state, lpr=False):
    r"""通过 Green 函数和定点混合求解有限核介子场。

    Each channel satisfies the radial Klein-Gordon form
    $(-\nabla^2+m_a^2)\phi_a=s_a$; the stored Green matrix applies its
    discretized inverse to the density and nonlinear sources.

    Args:
        state: Working state with radial densities and meson Green matrices.
        lpr (bool): Reserved source-compatible verbosity flag.

    Returns:
        None: Updates ``state.phi``; point-coupling calculations return early.
    """
    if state.ipc == 1:
        return
    previous_fields = state.phi.copy()
    for iteration in range(1, 101):
        fraction = 0.9 if iteration <= 10 else 0.8 if iteration <= 20 else 0.7 if iteration <= 30 else 0.6
        sigma_field = -previous_fields[1:, 1] * state.ggmes[1]
        omega_field = previous_fields[1:, 2] * state.ggmes[2]
        delta_field = -previous_fields[1:, 3] * state.ggmes[3]
        rho_field = previous_fields[1:, 4] * state.ggmes[4]
        nonlinear = (
            -state.z_bs * sigma_field ** 2 - state.z_cs * sigma_field ** 3 + state.z_ds * sigma_field * delta_field ** 2,
            -state.z_cw * omega_field ** 3 - state.z_lv * omega_field * rho_field ** 2,
            state.z_ds * delta_field * sigma_field ** 2,
            -state.z_lv * rho_field * omega_field ** 2,
        )
        for meson_index in range(1, 5):
            source = state.ff[1:, meson_index, 1] * state.ro[1:, meson_index] + nonlinear[meson_index - 1]
            state.phi[1:, meson_index] = apply_klein_gordon_green_function(state, meson_index, source)
        error = np.sum(np.abs(state.phi - previous_fields))
        if not np.isfinite(error):
            raise FloatingPointError("Nonlinear meson field became non-finite")
        if error < 1e-10:
            break
        previous_fields = fraction * state.phi + (1 - fraction) * previous_fields
    state.field_iterations = iteration
    state.field_residual = float(error)


def compute_coulomb_potential(state, lpr=False):
    r"""将径向 Coulomb Green 函数作用于质子密度。

    The discrete operation is $V_C(r_i)=\sum_jG_C(r_i,r_j)\rho_p(r_j)$.

    Args:
        state: Working state with Coulomb kernel and proton density.
        lpr (bool): Reserved source-compatible verbosity flag.

    Returns:
        None: Updates the proton Coulomb potential when enabled.
    """
    if state.icou:
        state.cou[:] = (state.ggc @ state.drvp) * state.b0


def build_dirac_potentials(state, lpr=False):
    r"""根据当前介子场或点耦合密度构造 Dirac 标量势和矢量势。

    The spherical Dirac operator has the structure
    $h_D=\boldsymbol{\alpha}\cdot\mathbf p+\beta(m+S(r))+V(r)$;
    this routine forms the radial $S$ and $V$ channels and the
    density-dependent rearrangement term.

    Args:
        state: Working state containing meson or point-coupling fields.
        lpr (bool): Reserved source-compatible verbosity flag.

    Returns:
        None: Stores scalar and vector potentials on ``state``.
    """
    if state.ipc == 0:
        meson_potentials = state.ggmes[1:5] * state.ff[:, 1:5, 1] * state.phi[:, 1:5]
        rearrangement = np.sum(state.ggmes[1:5] * state.ff[:, 1:5, 2] * state.phi[:, 1:5] * state.ro[:, 1:5], axis=1)
    elif state.ipc == 1:
        meson_potentials = state.ggmes[1:5] * state.ff[:, 1:5, 1] * state.ro[:, 1:5] + state.ddmes[1:5] * state.dro[:, 1:5]
        rearrangement = 0.5 * np.sum(state.ggmes[1:5] * state.ff[:, 1:5, 2] * state.ro[:, 1:5] ** 2, axis=1)
    else:
        raise ValueError("ipc must be 0 or 1")
    meson_potentials[:, 1] += rearrangement
    constraint = state.xlambda * state.rb ** 2
    for species_index, isospin_sign in ((1, -1), (2, 1)):
        scalar_potential = meson_potentials[:, 0] + isospin_sign * meson_potentials[:, 2]
        vector_potential = meson_potentials[:, 1] + isospin_sign * meson_potentials[:, 3]
        if species_index == 2:
            vector_potential = vector_potential + state.cou
        state.vps[:, species_index] = (vector_potential + scalar_potential) * state.hbc - constraint
        state.vms[:, species_index] = (vector_potential - scalar_potential) * state.hbc - constraint


def project_potential_matrix(state, basis_dimension, orbital, potential):
    r"""将径向势投影到给定谐振子子空间。

    $$
        V_{ab}=\int_0^\infty R_a(r)V(r)R_b(r)r^2dr.
    $$

    Args:
        state: Working state with radial functions and quadrature weights.
        basis_dimension (int): Number of oscillator states in the block.
        orbital (int): Orbital angular momentum $l$.
        potential (numpy.ndarray): Radial potential in MeV.

    Returns:
        numpy.ndarray: Projected matrix in MeV.
    """
    basis = state.rnl[1:basis_dimension + 1, orbital, 1:state.ngh + 1]
    return (basis * np.asarray(potential)[1:state.ngh + 1]) @ basis.T


def assemble_dirac_hamiltonian(state):
    r"""将径向势和动能耦合投影到每个 Dirac 块哈密顿量。

    Args:
        state: Working state with basis functions, potentials, and spin-orbit
            derivative matrices.

    Returns:
        None: Fills the block Hamiltonians $h_D$ in ``state.hh``.
    """
    for species_index in range(1, state.itx + 1):
        for block_index, large_dimension, small_dimension, total_dimension, matrix_column in _blocks(state, species_index):
            hamiltonian = _matrix(state.hh, total_dimension, matrix_column)
            hamiltonian[:large_dimension, :large_dimension] = project_potential_matrix(state, large_dimension, large_component_orbital_index(state.kb[block_index]), state.vps[:, species_index])
            hamiltonian[large_dimension:, large_dimension:] = project_potential_matrix(state, small_dimension, small_component_orbital_index(state.kb[block_index]), state.vms[:, species_index])
            hamiltonian[large_dimension:, large_dimension:] -= np.eye(small_dimension) * (2 * state.amu * state.hbc)
            hamiltonian[large_dimension:, :large_dimension] = state.hbc / state.b0 * state.sp[1:large_dimension * small_dimension + 1, block_index].reshape((small_dimension, large_dimension), order="F")
            hamiltonian[:large_dimension, large_dimension:] = hamiltonian[large_dimension:, :large_dimension].T


def _mix_indices(state):
    """Return active Broyden indices and their disabled source slots."""
    indices = []
    disabled = []
    for species_index in range(1, state.itx + 1):
        for block_index, large_dimension, small_dimension, total_dimension, matrix_column in _blocks(state, species_index):
            for name, matrix_offset, dimension in (("hh", 0, large_dimension), ("hh", large_dimension, small_dimension), ("de", 0, large_dimension)):
                for second_index in range(dimension):
                    for first_index in range(second_index, dimension):
                        indices.append((name, matrix_offset + first_index + 1 + (matrix_offset + second_index) * total_dimension,
                                        matrix_offset + second_index + 1 + (matrix_offset + first_index) * total_dimension, matrix_column))
                        disabled.append(name == "de" and state.del_[species_index] < 1e-5)
        indices.append(("ala", species_index, species_index, None))
        disabled.append(False)
    return indices, np.array(disabled)


def apply_broyden_mixing(state, lpr=False):
    r"""用修正 Broyden 方法混合势、配对场和化学势以达到自洽。

    The residual is $R^{(k)}=X_{\rm out}^{(k)}-X_{\rm in}^{(k)}$;
    a seven-vector modified Broyden history estimates the next fixed-point
    correction. ``state.si`` stores $\max_i|R_i|$.

    Args:
        state: Working state holding current and previous iterates.
        lpr (bool): Reserved source-compatible verbosity flag.

    Returns:
        None: Updates the mixed input variables and convergence residual.
    """
    if state.ii == 0 or not hasattr(state, "_broyden"):
        indices, disabled = _mix_indices(state)
        initial_values = np.array([
            getattr(state, name)[row_index] if matrix_column is None else getattr(state, name)[row_index, matrix_column]
            for name, row_index, mirror_row_index, matrix_column in indices
        ])
        state._broyden = SimpleNamespace(indices=indices, disabled=disabled, vin=initial_values,
                                         df=np.zeros((initial_values.size, 7)), dv=np.zeros((initial_values.size, 7)))
        state.si = 0.0
        return
    broyden_state = state._broyden
    current = np.array([
        getattr(state, name)[row_index] if matrix_column is None else getattr(state, name)[row_index, matrix_column]
        for name, row_index, mirror_row_index, matrix_column in broyden_state.indices
    ])
    residual = current - broyden_state.vin
    residual[broyden_state.disabled] = 0.0
    broyden_state.vin[broyden_state.disabled] = 0.0
    state.si = float(np.max(np.abs(residual)))
    if state.ii == 1 or state.ibroyd == 0:
        broyden_state.vin += state.xmix * residual
    else:
        history_length = min(state.ii - 2, 7)
        previous_slot, next_slot = (state.ii - 3) % 7, (state.ii - 2) % 7
        if state.ii == 2:
            broyden_state.df[:] = broyden_state.dv[:] = 0.0
        else:
            residual_difference = residual - broyden_state.df[:, previous_slot]
            value_difference = broyden_state.vin - broyden_state.dv[:, previous_slot]
            norm = np.linalg.norm(residual_difference)
            if norm > np.finfo(float).tiny:
                broyden_state.df[:, previous_slot], broyden_state.dv[:, previous_slot] = residual_difference / norm, value_difference / norm
            else:
                broyden_state.df[:, previous_slot] = broyden_state.dv[:, previous_slot] = 0.0
        correction = 0.7 * residual
        if history_length:
            residual_history = broyden_state.df[:, :history_length]
            value_history = broyden_state.dv[:, :history_length]
            beta = residual_history.T @ residual_history
            np.fill_diagonal(beta, 1.0 + 0.01 ** 2)
            weights = np.linalg.solve(beta, residual_history.T @ residual)
            correction -= (value_history + 0.7 * residual_history) @ weights
        broyden_state.df[:, next_slot] = residual
        broyden_state.dv[:, next_slot] = broyden_state.vin
        broyden_state.vin += correction if residual @ correction > 0 else state.xmix * residual
    for value, (name, row_index, mirror_row_index, matrix_column) in zip(broyden_state.vin, broyden_state.indices):
        array = getattr(state, name)
        if matrix_column is None:
            array[row_index] = value
        else:
            array[row_index, matrix_column] = array[mirror_row_index, matrix_column] = value


def kinetic_energy(state, species_index):
    r"""由 Dirac 密度矩阵计算核子动能项。

    Args:
        state: Working state with kinetic and density matrices.
        species_index (int): Neutron or proton channel, 1 or 2.

    Returns:
        float: Kinetic energy in MeV.
    """
    total_energy = 0.0
    for block_index, large_dimension, small_dimension, total_dimension, matrix_column in _blocks(state, species_index):
        density_matrix = _matrix(state.rosh, total_dimension, matrix_column)
        spin_orbit_matrix = state.hbc / state.b0 * state.sp[1:large_dimension * small_dimension + 1, block_index].reshape((small_dimension, large_dimension), order="F")
        total_energy += 2 * np.sum(spin_orbit_matrix * density_matrix[large_dimension:, :large_dimension]) - 2 * state.amu * state.hbc * np.trace(density_matrix[large_dimension:, large_dimension:])
    return float(total_energy)


def particle_energy(state, species_index):
    r"""将单粒子哈密顿量与密度矩阵收缩，得到粒子能量和。

    $$
        E_{\rm part}=\operatorname{Tr}(h_D\rho).
    $$

    Args:
        state: Working state with Hamiltonian and density matrices.
        species_index (int): Neutron or proton channel, 1 or 2.

    Returns:
        float: Particle-energy sum in MeV.
    """
    return float(sum(np.sum(_matrix(state.hh, total_dimension, matrix_column) * _matrix(state.rosh, total_dimension, matrix_column))
                     for block_index, large_dimension, small_dimension, total_dimension, matrix_column in _blocks(state, species_index)))


def pairing_energy(state, species_index):
    r"""按 $E_{\rm pair}=-\tfrac12\operatorname{Tr}(\Delta\kappa)$ 计算配对能。

    Args:
        state: Working state with pairing field and anomalous density.
        species_index (int): Neutron or proton channel, 1 or 2.

    Returns:
        float: Pairing energy in MeV.
    """
    total_energy = 0.0
    pairing_matrix_index = 0
    for block_index, large_dimension, small_dimension, total_dimension, matrix_column in _blocks(state, species_index):
        pairing_matrix = _matrix(state.de, total_dimension, matrix_column)
        for second_index in range(large_dimension):
            for first_index in range(second_index, large_dimension):
                pairing_matrix_index += 1
                total_energy += pairing_matrix[first_index, second_index] * state.aka[pairing_matrix_index, species_index]
    return -0.5 * float(total_energy)


def field_energy(state):
    r"""积分介子/点耦合场能、重排能和 Coulomb 能。

    The radial integration uses $4\pi r^2dr$ quadrature weights stored
    in ``state.wdcor``.

    Args:
        state: Working state containing densities and fields.

    Returns:
        tuple: Meson-channel energies, rearrangement energy, and Coulomb
        energy, all in MeV.
    """
    meson_energies = np.zeros(5)
    integration_weights = state.wdcor[1:]
    density = state.ro[1:, 1:5]
    if state.ipc == 0:
        meson_fields = state.phi[1:, 1:5]
        energy_terms = 0.5 * state.ggmes[1:5] * state.ff[1:, 1:5, 1] * meson_fields * density
        rearrangement = np.sum(state.ggmes[1:5] * state.ff[1:, 1:5, 2] * meson_fields * density * density[:, 1:2] * integration_weights[:, None]) * state.hbc
        sigma_field = -meson_fields[:, 0] * state.ggmes[1]
        omega_field = meson_fields[:, 1] * state.ggmes[2]
        delta_field = -meson_fields[:, 2] * state.ggmes[3]
        rho_field = meson_fields[:, 3] * state.ggmes[4]
        energy_terms[:, 0] -= (state.z_bs * sigma_field ** 3 / 6
                               + 0.25 * state.z_cs * sigma_field ** 4
                               - 0.5 * state.z_ds * sigma_field ** 2 * delta_field ** 2)
        energy_terms[:, 1] += (0.25 * state.z_cw * omega_field ** 4
                               + 0.5 * state.z_lv * omega_field ** 2 * rho_field ** 2)
        meson_energies[1:] = state.hbc * np.sum(energy_terms * integration_weights[:, None], axis=0)
    elif state.ipc == 1:
        energy_terms = state.ggmes[1:5] * state.ff[1:, 1:5, 1] * density ** 2 + state.ddmes[1:5] * density * state.dro[1:, 1:5]
        meson_energies[1:] = 0.5 * state.hbc * np.sum(energy_terms * integration_weights[:, None], axis=0)
        rearrangement = 0.5 * state.hbc * np.sum(state.ggmes[1:5] * state.ff[1:, 1:5, 2] * density ** 2 * density[:, 1:2] * integration_weights[:, None])
    else:
        raise ValueError("ipc must be 0 or 1")
    coulomb = 0.0
    if state.icou:
        proton_density = 0.5 * (state.ro[1:, 2] + state.ro[1:, 4])
        coulomb = 0.5 * state.hbc * np.sum(state.cou[1:] * proton_density * integration_weights)
    return meson_energies, float(rearrangement), float(coulomb)


def compute_expectations(state, lpr=False):
    r"""计算粒子数、均方根半径以及总能量分项。

    $$
        N_q=\int\rho_q(r)\,d^3r,\qquad
        r_q=\sqrt{\frac{\int r^2\rho_q(r)\,d^3r}{N_q}},\qquad
        r_m^2=\frac{Nr_n^2+Zr_p^2}{A}.
    $$

    Args:
        state: Converged or intermediate finite-nucleus working state.
        lpr (bool): Append a compact result line to the state log.

    Returns:
        state: The same state with integrated observables updated.
    """
    particle_numbers, scalar_densities, radii, kinetic_energies, particle_energies, pairing_energies = (np.zeros(4) for _ in range(6))
    for species_index in range(1, state.itx + 1):
        measure = state.rv[1:, species_index] * state.wdcor[1:]
        particle_numbers[species_index] = np.sum(measure)
        scalar_densities[species_index] = np.sum(state.rs[1:, species_index] * state.wdcor[1:])
        radii[species_index] = math.sqrt(np.sum(measure * state.rb[1:] ** 2) / particle_numbers[species_index]) if particle_numbers[species_index] > 0 else 0.0
        kinetic_energies[species_index], particle_energies[species_index], pairing_energies[species_index] = kinetic_energy(state, species_index), particle_energy(state, species_index), pairing_energy(state, species_index)
    if state.itx == 1:
        for array in (particle_numbers, scalar_densities, radii, kinetic_energies, particle_energies, pairing_energies):
            array[2] = array[1]
    for array in (particle_numbers, scalar_densities, kinetic_energies, particle_energies, pairing_energies):
        array[3] = array[1] + array[2]
    radii[3] = math.sqrt((state.npr[1] * radii[1] ** 2 + state.npr[2] * radii[2] ** 2) / state.amas)
    meson_energies, rearrangement_energy, coulomb_energy = field_energy(state)
    energy_without_center_of_mass = kinetic_energies[3] + sum(meson_energies[1:]) + coulomb_energy + pairing_energies[3]
    total_energy = energy_without_center_of_mass + state.ecm[3]
    # The original consistency diagnostic omits the delta term here.
    consistency_energy = particle_energies[3] - meson_energies[1] - meson_energies[2] - meson_energies[4] - coulomb_energy - rearrangement_energy + pairing_energies[3]
    state.ea = state.binding_energy = total_energy / state.amas
    state.rms = state.radius_mass = radii[3]
    state.radius_neutron, state.radius_proton = radii[1:3]
    state.radius_charge = math.sqrt(radii[2] ** 2 + 0.64)
    state.etot, state.etot0, state.etest = total_energy, energy_without_center_of_mass, consistency_energy - energy_without_center_of_mass
    state.xn, state.xs, state.radii = particle_numbers, scalar_densities, radii
    state.ekt, state.epart, state.ept = kinetic_energies, particle_energies, pairing_energies
    state.emes, state.er, state.ecou = meson_energies, rearrangement_energy, coulomb_energy
    if lpr:
        _log(state, f"Total energy {total_energy:.10f} MeV; E/A {state.ea:.10f} MeV; charge radius {state.radius_charge:.10f} fm")
    return state


def resolve_degenerate_states(eigenvalues, vectors, operator, eps=1e-6):
    r"""在简并子空间内对角化指定算符，以确定唯一的本征矢组合。

    For an eigenspace basis $Q$, diagonalize $Q^TOQ$ and rotate
    $Q$ by its eigenvectors.
    """
    vectors = np.array(vectors, copy=True)
    start = 0
    while start < eigenvalues.size:
        end = start + 1
        while end < eigenvalues.size and eigenvalues[end] - eigenvalues[start] <= eps:
            end += 1
        if end - start > 1:
            subspace = vectors[:, start:end]
            _, rotation = symmetric_diagonalization(subspace.T @ operator @ subspace)
            vectors[:, start:end] = subspace @ rotation
        start = end
    return vectors


def canonicalize_states(state, lpr=False):
    r"""构造 canonical 单粒子态、占据数、径向波函数和能级差。

    The canonical orbitals diagonalize the density matrix; occupations are its
    eigenvalues and energies are $\langle i|h_D|i\rangle$.

    Args:
        state: Working state with converged HFB densities and Hamiltonians.
        lpr (bool): Reserved source-compatible verbosity flag.

    Returns:
        state: The same state with canonical levels and radial wavefunctions.
    """
    state.els_3p_n = state.els_2f_n = state.els_2d_p = 0.0
    coordinate_factor = state.xh[1:] * np.sqrt(state.wh[1:]) * state.b0 ** 1.5 * math.sqrt(4 * math.pi)
    for species_index in (1, 2):
        large_state_offset = small_state_offset = 0
        for block_index, large_dimension, small_dimension, total_dimension, matrix_column in _blocks(state, species_index):
            density_matrix = np.zeros((total_dimension, total_dimension))
            for channel, scale in ((species_index, 1.0), (species_index + 2, 100.0)):
                begin = int(state.ka[block_index, channel]) + 1
                end = begin + int(state.kd[block_index, channel])
                quasiparticle_vectors = state.fguv[total_dimension + 1:2 * total_dimension + 1, begin:end, channel]
                density_matrix += scale * (quasiparticle_vectors @ quasiparticle_vectors.T)
            occupation, vectors = symmetric_diagonalization(density_matrix)
            hamiltonian, pairing_matrix = _matrix(state.hh, total_dimension, matrix_column), _matrix(state.de, total_dimension, matrix_column)
            vectors = resolve_degenerate_states(occupation, vectors, hamiltonian)
            maxima = vectors[np.abs(vectors).argmax(axis=0), np.arange(total_dimension)]
            vectors *= np.where(maxima < 0, -1.0, 1.0)
            energies = np.sum(vectors * (hamiltonian @ vectors), axis=0)
            gaps = np.sum(vectors * (pairing_matrix @ vectors), axis=0)
            order = np.argsort(energies, kind="stable")
            energies, gaps, occupation, vectors = energies[order], gaps[order], occupation[order], vectors[:, order]
            occupation[(occupation < 0) | (occupation > 2)] = 0.0
            occupation[occupation > 1] = 1.0
            state.kacan[block_index, species_index], state.kacan[block_index, species_index + 2] = large_state_offset, small_state_offset
            for channel, source, offset in ((species_index, slice(small_dimension, total_dimension), large_state_offset), (species_index + 2, slice(0, small_dimension), small_state_offset)):
                count = large_dimension if channel == species_index else small_dimension
                destination = slice(offset + 1, offset + count + 1)
                state.eecan[destination, channel] = energies[source]
                state.decan[destination, channel] = gaps[source]
                state.vvcan[destination, channel] = occupation[source]
                state.mulcan[destination, channel] = 2 * abs(state.kb[block_index])
                state.fgcan[1:total_dimension + 1, destination, channel] = vectors[:, source]
                state.kdcan[block_index, channel] = count
            large_state_offset += large_dimension
            small_state_offset += small_dimension
            begin = int(state.kacan[block_index, species_index]) + 1
            end = begin + large_dimension
            coefficients = state.fgcan[1:total_dimension + 1, begin:end, species_index]
            for offset, count, orbital, rows in ((0, large_dimension, large_component_orbital_index(state.kb[block_index]), slice(1, state.ngh + 1)),
                                                  (large_dimension, small_dimension, small_component_orbital_index(state.kb[block_index]), slice(state.ngh + 1, 2 * state.ngh + 1))):
                state.wave[rows, begin:end, species_index] = (state.rnl[1:count + 1, orbital, 1:].T @ coefficients[offset:offset + count]) / coordinate_factor[:, None]
                state.dwave[rows, begin:end, species_index] = (state.rnl1[1:count + 1, orbital, 1:].T @ coefficients[offset:offset + count]) / (coordinate_factor * state.b0)[:, None]
            integration_weight = np.tile(state.wdcor[1:], 2)
            norm = np.sqrt(np.sum(state.wave[1:, begin:end, species_index] ** 2 * integration_weight[:, None], axis=0))
            state.wave[1:, begin:end, species_index] /= norm
            state.dwave[1:, begin:end, species_index] /= norm
            # n labels are radial ordinal positions in the canonical block.
            kappa = int(state.kb[block_index])
            if species_index == 1 and large_dimension >= 2 and kappa in (3, -4):
                state.els_2f_n += (1 if kappa == 3 else -1) * state.eecan[begin + 1, species_index]
            if species_index == 1 and large_dimension >= 3 and kappa in (1, -2):
                state.els_3p_n += (1 if kappa == 1 else -1) * state.eecan[begin + 2, species_index]
            if species_index == 2 and large_dimension >= 2 and kappa in (2, -3):
                state.els_2d_p += (1 if kappa == 2 else -1) * state.eecan[begin + 1, species_index]
        state.nkcan[species_index], state.nkcan[species_index + 2] = large_state_offset, small_state_offset
    return state


def center_of_mass_diagonal_correction(state, lpr=False):
    r"""计算质心动能修正中的占据轨道对角项。

    The radial expectation contains $|dR/dr|^2+l(l+1)|R/r|^2$ and is
    scaled by $-\hbar c/(2Am)$.
    """
    if state.icm != 2:
        return
    center_of_mass_factor = state.hbc / (2 * state.amu * state.amas)
    integration_weights = state.wdcor[1:]
    for species_index in range(1, state.itx + 1):
        diagonal_correction = 0.0
        for block_index, large_dimension, small_dimension, total_dimension, matrix_column in _blocks(state, species_index):
            begin = int(state.kacan[block_index, species_index]) + 1
            end = begin + int(state.kdcan[block_index, species_index])
            expectation = np.zeros(end - begin)
            for component, orbital in ((0, large_component_orbital_index(state.kb[block_index])), (1, small_component_orbital_index(state.kb[block_index]))):
                rows = slice(component * state.ngh + 1, (component + 1) * state.ngh + 1)
                wave, derivative = state.wave[rows, begin:end, species_index], state.dwave[rows, begin:end, species_index]
                expectation += np.sum(derivative ** 2 * integration_weights[:, None], axis=0)
                expectation += orbital * (orbital + 1) * np.sum(wave ** 2 * (integration_weights / state.rb[1:] ** 2)[:, None], axis=0)
            diagonal_correction += np.sum(state.vvcan[begin:end, species_index] * (2 * abs(state.kb[block_index])) * expectation)
        state.ecmd[species_index] = -center_of_mass_factor * diagonal_correction
    state.ecmd[3] = state.ecmd[1] + state.ecmd[2]


def center_of_mass_nondiagonal_correction(state, lpr=False):
    r"""计算微观质心修正中的非对角配对跃迁项。

    Contributions couple orbital angular momenta with $|l-l'|=1$ and
    are weighted by canonical occupations and vacancy amplitudes.
    """
    if state.icm < 2:
        state.ecm[1:4] = -0.75 * state.hom
        return
    center_of_mass_factor = state.hbc / (2 * state.amu * state.amas)
    for species_index in range(1, state.itx + 1):
        nondiagonal_correction = 0.0
        for first_block_index in range(1, state.nb + 1):
            first_kappa_magnitude = abs(state.kb[first_block_index])
            first_begin = int(state.kacan[first_block_index, species_index]) + 1
            first_end = first_begin + int(state.kdcan[first_block_index, species_index])
            first_orbitals = (large_component_orbital_index(state.kb[first_block_index]), small_component_orbital_index(state.kb[first_block_index]))
            first_occupations = np.sqrt(state.vvcan[first_begin:first_end, species_index])
            first_complements = np.sqrt(np.maximum(0, 1 - first_occupations ** 2))
            for second_block_index in range(1, state.nb + 1):
                second_kappa_magnitude = abs(state.kb[second_block_index])
                second_orbitals = (large_component_orbital_index(state.kb[second_block_index]), small_component_orbital_index(state.kb[second_block_index]))
                if all(abs(first_orbital - second_orbital) != 1 for first_orbital, second_orbital in zip(first_orbitals, second_orbitals)):
                    continue
                second_begin = int(state.kacan[second_block_index, species_index]) + 1
                second_end = second_begin + int(state.kdcan[second_block_index, species_index])
                second_occupations = np.sqrt(state.vvcan[second_begin:second_end, species_index])
                second_complements = np.sqrt(np.maximum(0, 1 - second_occupations ** 2))
                pair_factor = np.outer(first_occupations, second_occupations) * (np.outer(first_occupations, second_occupations) + np.outer(first_complements, second_complements))
                matrix_element = np.zeros((first_end - first_begin, second_end - second_begin))
                for component, (first_orbital, second_orbital) in enumerate(zip(first_orbitals, second_orbitals)):
                    if abs(first_orbital - second_orbital) != 1:
                        continue
                    if first_kappa_magnitude - first_orbital == second_kappa_magnitude - second_orbital:
                        numerator = (first_orbital + second_kappa_magnitude + 2) * (first_orbital + second_kappa_magnitude - 1)
                        angular_factor = math.sqrt(max(0.0, numerator / ((2 * first_orbital + 1) * (2 * second_orbital + 1))))
                    else:
                        numerator = (second_kappa_magnitude - first_orbital + 1) * (first_orbital - second_kappa_magnitude + 2)
                        angular_factor = (2 * second_kappa_magnitude - 2 * second_orbital - 1) * math.sqrt(max(0.0, numerator / ((2 * first_orbital + 1) * (2 * second_orbital + 1))))
                    if first_orbital > second_orbital:
                        derivative_factor, centrifugal_factor = math.sqrt(first_orbital), -first_orbital
                    else:
                        derivative_factor, centrifugal_factor = -math.sqrt(second_orbital), second_orbital
                    rows = slice(component * state.ngh + 1, (component + 1) * state.ngh + 1)
                    first_wave, second_wave = state.wave[rows, first_begin:first_end, species_index], state.wave[rows, second_begin:second_end, species_index]
                    derivative = state.dwave[rows, first_begin:first_end, species_index] - (centrifugal_factor - 1) * first_wave / state.rb[1:, None]
                    integral = (derivative * state.wdcor[1:, None]).T @ second_wave
                    matrix_element += angular_factor * derivative_factor * integral
                nondiagonal_correction += np.sum(pair_factor * matrix_element ** 2)
        state.ecmn[species_index] = center_of_mass_factor * nondiagonal_correction
        state.ecm[species_index] = state.ecmd[species_index] + state.ecmn[species_index]
    state.ecmn[3] = state.ecmn[1] + state.ecmn[2]
    state.ecm[3] = state.ecm[1] + state.ecm[2]


def center_of_mass_correction(state, lpr=False):
    r"""按输入选项累加对角和非对角质心修正。

    The leading scale is $1/(2Am)$ times the intrinsic total-momentum
    expectation; implementation details follow the source's selected
    diagonal approximation.

    Args:
        state: Working state with canonical orbitals and occupations.
        lpr (bool): Reserved source-compatible verbosity flag.

    Returns:
        state: The same state with ``state.ecm`` updated.
    """
    center_of_mass_diagonal_correction(state, lpr)
    center_of_mass_nondiagonal_correction(state, lpr)
    return state


def check_convergence(state):
    r"""按源程序残差条件 $s_i<\epsilon$ 检查自洽收敛。

    Args:
        state: Working state with residual, tolerance, and mixing controls.

    Returns:
        int: 2 when converged, otherwise 0.
    """
    if state.ii >= 2 and state.si < state.epsi:
        return 2
    if state.ibroyd == 0:
        if state.si < state.siold and state.iaut == 1:
            state.xmix = min(state.xmax, 1.04 * state.xmix)
        else:
            state.xmix = state.xmix0
    state.siold = state.si
    return 0


def iterate_self_consistency(state, lpr=False):
    r"""迭代准粒子、密度和场方程，直至混合变量达到自洽。

    Each iteration maps $(h,\Delta)\to(U,V)\to(\rho,\kappa)$ and then
    rebuilds the fields. Stopping uses the maximum mixed-variable residual,
    not the source energy-consistency diagnostic.

    Args:
        state: Initialized finite-nucleus working state.
        lpr (bool): Append iteration diagnostics to the state log.

    Returns:
        state: State with final iteration, history, and convergence flag.
    """
    state.ii = 0
    state.converged = False
    state.history = []
    assemble_dirac_hamiltonian(state)
    apply_broyden_mixing(state)
    for iteration in range(1, state.maxi + 1):
        state.ii = iteration
        for species_index in range(1, state.itx + 1):
            solve_dirac_hfb(state, species_index)
            compute_oscillator_densities(state, species_index)
        transform_densities(state)
        update_density_dependent_couplings(state)
        solve_meson_fields(state)
        compute_coulomb_potential(state)
        compute_expectations(state)
        build_dirac_potentials(state)
        assemble_dirac_hamiltonian(state)
        for species_index in range(1, state.itx + 1):
            update_pairing_field(state, species_index)
            state.spk0[species_index] = state.spk[species_index]
        apply_broyden_mixing(state)
        if not np.isfinite(state.si):
            raise FloatingPointError("Self-consistency residual became non-finite")
        record = {"iteration": iteration, "residual": state.si, "energy_per_particle": state.ea, "radius_mass": state.rms}
        state.history.append(record)
        callback = getattr(state, "progress_callback", None)
        if callback is not None:
            callback(record)
        if lpr:
            _log(state, f"Iteration {iteration}: residual {state.si:.6g}, E/A {state.ea:.9f}")
        if iteration > 2 and check_convergence(state) == 2:
            state.converged = True
            break
    return state


# Compatibility aliases for the original Fortran routine names.
dirhb = solve_dirac_hfb
denssh = compute_oscillator_densities
delta = update_pairing_field
densit = transform_densities
gdd = update_density_dependent_couplings
gordon = apply_klein_gordon_green_function
field = solve_meson_fields
coulom = compute_coulomb_potential
poten = build_dirac_potentials
pot = project_potential_matrix
gamma = assemble_dirac_hamiltonian
broyden = apply_broyden_mixing
ekin = kinetic_energy
epar = particle_energy
epair = pairing_energy
efield = field_energy
expect = compute_expectations
degen = resolve_degenerate_states
canon = canonicalize_states
cmcd = center_of_mass_diagonal_correction
cmcn = center_of_mass_nondiagonal_correction
centmas = center_of_mass_correction
itestc = check_convergence
iterate = iterate_self_consistency
iter = iterate_self_consistency
