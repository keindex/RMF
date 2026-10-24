"""Result files and Fortran-compatible sequential restart records."""
from dataclasses import asdict
from pathlib import Path
import json
import numpy as np
from scipy.io import FortranFile
from scipy.interpolate import CubicSpline

from ..math_utils.arrays import block_matrix


def _iter_pairing_matrices(state):
    """Yield iteration number, matrix, and active dimensions for restart.

    The source stores each species and iteration in a packed column.  For a
    block with dimensions ``n_l`` and ``n_s``, the active matrix is the view
    ``de[1:(n_l+n_s)^2+1, column]`` reshaped in Fortran order.
    """
    for iteration_index in range(1, state.itx + 1):
        for block_index in range(1, state.nb + 1):
            large_dimension, small_dimension = map(
                int, state.id[block_index, 1:3]
            )
            total_dimension = large_dimension + small_dimension
            matrix = block_matrix(
                state.de,
                total_dimension,
                block_index + (iteration_index - 1) * state.nbx,
            )
            yield iteration_index, large_dimension, matrix


def initialize_pairing(state, restart_directory=None):
    r"""Initialize or restore the symmetric pairing field.

    New calculations set the large-large block to the initial gap
    :math:`\Delta_q`; restart calculations restore the packed lower triangle
    and mirror it with :math:`\Delta_{ij}=\Delta_{ji}`.
    """
    state.de.fill(0)
    if state.inink == 0:
        if restart_directory is None:
            raise ValueError('Pairing restart requested without a restart directory.')
        with FortranFile(Path(restart_directory)/'dirhb.del', 'r') as fortran_file:
            if int(fortran_file.read_ints(np.int32)[0]) != state.mv:
                raise ValueError('Pairing restart uses a different oscillator basis.')
            for _iteration_index, large_dimension, matrix in _iter_pairing_matrices(state):
                for row_index in range(large_dimension):
                    row_values = fortran_file.read_reals(np.float64)
                    if len(row_values) != large_dimension - row_index:
                        raise ValueError('Invalid pairing restart record length.')
                    matrix[row_index:large_dimension, row_index] = row_values
                    matrix[row_index, row_index:large_dimension] = row_values
    else:
        for iteration_index, large_dimension, matrix in _iter_pairing_matrices(state):
            np.fill_diagonal(
                matrix[:large_dimension, :large_dimension],
                state.del_[iteration_index],
            )


def read_potentials(state, directory):
    """Read scalar and vector potentials from a Fortran restart file."""
    with FortranFile(Path(directory)/'dirhb.wel', 'r') as fortran_file:
        if int(fortran_file.read_ints(np.int32)[0]) != state.ngh:
            raise ValueError('Potential restart has a different radial grid.')
        state.ala[1:3] = fortran_file.read_reals(np.float64)
        for potential_name in ('vms', 'vps'):
            potential_values = fortran_file.read_reals(np.float64)
            getattr(state, potential_name)[:, 1:3] = potential_values.reshape(
                (state.ngh + 1, 2), order='F'
            )


def write_restart(state, directory):
    """Write potentials and packed symmetric pairing matrices for restart."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with FortranFile(directory / 'dirhb.wel', 'w') as fortran_file:
        fortran_file.write_record(np.array([state.ngh], dtype=np.int32))
        fortran_file.write_record(state.ala[1:3])
        fortran_file.write_record(state.vms[:, 1:3].ravel(order='F'))
        fortran_file.write_record(state.vps[:, 1:3].ravel(order='F'))
    with FortranFile(directory / 'dirhb.del', 'w') as fortran_file:
        fortran_file.write_record(np.array([state.mv], dtype=np.int32))
        for _iteration_index, large_dimension, matrix in _iter_pairing_matrices(state):
            for row_index in range(large_dimension):
                fortran_file.write_record(matrix[row_index:large_dimension, row_index])


def summary_values(state):
    # Formula: r_np = r_n - r_p; remaining values are reported observables.
    r"""Return the compact observable vector used in text and JSON output.

    The neutron skin is defined by :math:`\Delta r_{np}=r_n-r_p`; the other
    entries are copied from the converged state in their documented units.
    """
    return [
        float(state.ea),
        float(state.radius_charge),
        float(state.radius_neutron - state.radius_proton),
        float(state.els_2d_p),
        float(state.els_3p_n),
        float(state.els_2f_n),
    ]


def density_profile(state):
    r"""Interpolate the total radial density on the report grid.

    The reported profile is :math:`\rho(r)=S(r;\,r_i,\rho_i)`, where
    :math:`S` is a natural cubic spline through the solver's radial mesh.
    """
    radii = 1e-8 + np.arange(129) * float(np.float32(.1))
    total_density = CubicSpline(
        state.rb, state.ro[:, 2], bc_type='natural'
    )(radii)
    return np.column_stack((radii, total_density))


def detailed_report(state):
    """Format one converged state as a Fortran-style human-readable report."""
    lines = [
           f'Python spherical CRMF: {state.nucnam} {state.nama}, interaction {state.parname}',
           f'Oscillator shells: fermions {state.n0f}, bosons {state.n0b}',
           f'Iteration converged after {state.ii} steps   si = {state.si:.10g}',
           '                         neutron         proton          total',
           ' particle number     ' + ''.join(f'{value:15.6f}' for value in state.xn[1:]),
           ' trace scalar density' + ''.join(f'{value:15.6f}' for value in state.xs[1:]),
           ' lambda              ' + ''.join(f'{value:15.6f}' for value in state.ala[1:]),
           ' spk                 ' + ''.join(f'{value:15.6f}' for value in state.spk[1:]),
           ' rms-Radius          ' + ''.join(f'{value:15.6f}' for value in state.radii[1:]),
           f' charge-Radius       {state.radius_charge:45.6f}',
           ' Particle Energy     ' + ''.join(f'{value:15.6f}' for value in state.epart[1:]),
           f' Selfconsistency Test. {state.etest:44.6f}',
           ' Kinetic Energy      ' + ''.join(f'{value:15.6f}' for value in state.ekt[1:])
    ]
    for energy_label, energy_value in zip(
        ('E-scsc', 'E-scve', 'E-vesc', 'E-veve'), state.emes[1:]
    ):
        lines.append(f' {energy_label:<20} {energy_value:45.6f}')
    lines.extend([
        f' E-rearrang.          {state.er:45.6f}',
        f' Coulomb direct       {state.ecou:45.6f}',
        ' Pairing Energy      ' + ''.join(f'{value:15.6f}' for value in state.ept[1:]),
        f' Sum without E-cm     {state.etot0:45.6f}',
        f' E-cm                 {state.ecm[3]:45.6f}',
        f' Total Energy         {state.etot:45.6f}',
        f' E/A                  {state.ea:45.6f}',
    ])
    for species_index, species_name in ((1, 'neutron'), (2, 'proton')):
        lines.extend(['', f'Canonical single-particle states ({species_name})',
                      '   k  kappa  orbital       energy(MeV)       gap(MeV)     occupation'])
        for block_index in range(1, state.nb + 1):
            first_state_index = int(state.kacan[block_index, species_index]) + 1
            state_count = int(state.kdcan[block_index, species_index])
            for state_index in range(first_state_index, first_state_index + state_count):
                lines.append(
                    f'{state_index:4d} {state.kb[block_index]:6d} '
                    f'{state.tb[block_index]:>8s} '
                    f'{state.eecan[state_index, species_index]:17.8f} '
                    f'{state.decan[state_index, species_index]:14.8f} '
                    f'{state.vvcan[state_index, species_index]:14.10f}'
                )
        lines.extend(['', f'Quasiparticle states ({species_name})',
                      '   k   orbital         Eqp(MeV)             uu             vv'])
        for block_index in range(1, state.nb + 1):
            total_component_dimension = int(
                state.id[block_index, 1] + state.id[block_index, 2]
            )
            first_state_index = int(state.ka[block_index, species_index]) + 1
            state_count = int(state.kd[block_index, species_index])
            for state_index in range(first_state_index, first_state_index + state_count):
                if state.equ[state_index, species_index] >= 100:
                    continue
                quasiparticle_components = state.fguv[
                    1:2 * total_component_dimension + 1,
                    state_index,
                    species_index,
                ]
                lines.append(
                    f'{state_index:4d} {state.tk[state_index, species_index]:>8s} '
                    f'{state.equ[state_index, species_index]:17.8f} '
                    f'{quasiparticle_components[:total_component_dimension] @ quasiparticle_components[:total_component_dimension]:14.10f} '
                    f'{quasiparticle_components[total_component_dimension:] @ quasiparticle_components[total_component_dimension:]:14.10f}'
                )
    return '\n'.join(lines)+'\n'


def write_results(states, directory, eos_table=None):
    """Write reports, restart records, density data, and an optional EOS table."""
    directory=Path(directory); directory.mkdir(parents=True,exist_ok=True)
    if states:
        header='Eb/A, r_c, r_np, els_2d_p, els_3p_n, els_2f_n\n'
        for state in states:
            header += f'{state.parname:10s} ' + ' '.join(
                f'{value:.10g}' for value in summary_values(state)
            ) + '\n'
        (directory/f'{states[0].nucnam}_results.txt').write_text(header,encoding='utf-8')
        (directory/'dirhb.out').write_text(
            '\n'.join(detailed_report(state) for state in states), encoding='utf-8'
        )
        np.savetxt(
            directory / 'dirhb.plo',
            density_profile(states[-1]),
            fmt=['%10.3f', '%15.6f'],
        )
        write_restart(states[-1],directory)
        result_records = []
        for state in states:
            result_records.append(dict(
                model=state.parname,
                config=asdict(state.config),
                converged=bool(state.converged),
                iterations=state.ii,
                residual=state.si,
                values=summary_values(state),
                particle_numbers=state.xn[1:].tolist(),
                rms_radii=state.radii[1:].tolist(),
                energy_consistency_diagnostic=state.etest,
                history=state.history,
            ))
        (directory/'results.json').write_text(
            json.dumps(result_records, indent=2) + '\n', encoding='utf-8'
        )
        (directory/'run.log').write_text(
            '\n'.join(
                f'{state.nucnam} {state.nama} {state.parname}: '
                f'Iteration converged after {state.ii} steps   si = {state.si:.10g}'
                for state in states
            ) + '\n',
            encoding='utf-8',
        )
    if eos_table is not None:
        # Explicit delimiters also handle adjacent negative G13.6 source fields.
        np.savetxt(directory/'EOS.OUT',eos_table,fmt='%.10e')
