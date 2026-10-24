"""High-level finite nucleus calculation, corresponding to rmf()/crmf()."""
from dataclasses import replace
import numpy as np

from .state import NucleusConfig,new_state,apply_config
from . import finite_setup as setup
from . import finite_solver as solver
from .pairing import initialize_separable_pairing
from .io import initialize_pairing,read_potentials,summary_values


def solve_nucleus(config=None,model='new1',*,restart_directory=None,
                  eos_parameters=None,progress_callback=None):
    r"""执行一次球形有限核 RHB 计算。

    The solution is a stationary point of the constrained energy,
    $\delta(E-\lambda_nN-\lambda_pZ)=0$, found by alternating HFB
    diagonalization, density construction, meson-field updates, and Broyden
    mixing.

    Args:
        config (NucleusConfig, optional): Nucleus, basis, and iteration setup.
        model (str): Interaction key from :data:`FORCE_NAMES`.
        restart_directory (path-like, optional): Directory containing restart
            fields when restart flags are enabled.
        eos_parameters (object, optional): Nuclear-matter parameters for the
            ``EOS`` finite-nucleus interaction.
        progress_callback (callable, optional): Receives one mapping per
            self-consistency iteration.

    Returns:
        state: Converged calculation state; energies are MeV and radii are fm.
    """
    config = config or NucleusConfig()
    state = new_state(config)
    setup.initialize_default_state(state)
    apply_config(state, config)
    setup.configure_interaction(state, model, eos_parameters)
    # A selected force carries the source's default Coulomb/CM prescriptions.
    if config.center_of_mass is not None:
        state.icm = config.center_of_mass
    if config.coulomb is not None:
        state.icou = config.coulomb
    state.progress_callback = progress_callback
    setup.build_gaussian_quadrature(state)
    setup.prepare_oscillator_mesh(state)
    if state.inin == 0:
        if restart_directory is None:
            raise ValueError('Potential restart requires restart_directory.')
        read_potentials(state, restart_directory)
    setup.initialize_starting_fields(state)
    setup.build_oscillator_basis(state)
    initialize_pairing(state, restart_directory)
    setup.build_oscillator_wavefunctions(state)
    setup.initialize_spin_orbit_fields(state)
    initialize_separable_pairing(state)
    setup.build_coulomb_green_function(state)
    setup.build_meson_green_functions(state)
    solver.iterate_self_consistency(state)
    if not state.converged:
        raise RuntimeError(
            f'{model} did not converge in {state.ii} iterations; residual={state.si:.6g}'
        )
    solver.canonicalize_states(state)
    solver.center_of_mass_correction(state)
    solver.compute_expectations(state)
    if not np.isfinite(summary_values(state)).all():
        raise FloatingPointError(f'{model} produced non-finite observables.')
    return state


def constrained_monopole(config=None, model='new1', step=.01, **keyword_arguments):
    r"""使用三个约束态计算单极共振的求和规则估计。

    The finite-difference inverse moment uses the curvature response
    $m_{-1}\propto d\langle r^2\rangle/d\lambda$, and the estimate is
    $E_{\rm GMR}=\sqrt{m_1/m_{-1}}$.

    Args:
        config (NucleusConfig, optional): Base nucleus configuration.
        model (str): Interaction key.
        step (float): Positive constraint displacement.
        **keyword_arguments: Additional arguments forwarded to
            :func:`solve_nucleus`.

    Returns:
        dict: Sum-rule energy in MeV, moments, and the three converged states.
    """
    config = config or NucleusConfig()
    if not np.isfinite(step) or step <= 0:
        raise ValueError('Constraint step must be finite and positive.')
    states = [
        solve_nucleus(
            replace(config, constraint=constraint), model, **keyword_arguments
        )
        for constraint in (-step, 0., step)
    ]
    mass_radii = np.array([state.radius_mass for state in states])
    inverse_moment = (mass_radii[2]**2 - mass_radii[0]**2) / (2 * step)
    moment = 4 * mass_radii[1]**2 * states[1].hbc**2 / 939.
    if inverse_moment <= 0:
        raise ValueError('Nonpositive constrained inverse-energy-weighted moment.')
    return dict(
        energy_MeV=float(np.sqrt(moment / inverse_moment)),
        m1=float(moment),
        m_minus1=float(inverse_moment),
        states=states,
    )


# Original main entry routine names, expressed with explicit Python parameters.
rmf=solve_nucleus
crmf=constrained_monopole
