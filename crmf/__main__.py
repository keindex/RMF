"""Run with python -m crmf --input dirhb.dat --output python_output."""
import argparse
from dataclasses import replace
from pathlib import Path
import sys

from .finite_nucleus import NucleusConfig, solve_nucleus
from .finite_nucleus.finite_setup import FORCE_NAMES
from .finite_nucleus.io import write_results
from .nuclear_matter.eos import default_parameters, equation_of_state


def main(argv=None):
    """Parse CLI options, run selected calculations, and write their results."""
    parser = argparse.ArgumentParser(
        description='Independent Python spherical CRMF and nuclear matter EOS'
    )
    parser.add_argument('--input', type=Path, default=Path('dirhb.dat'))
    parser.add_argument('--output', type=Path, default=Path('python_output'))
    parser.add_argument('--models', nargs='+', choices=FORCE_NAMES,
                        default=['new1', 'new2', 'new3', 'new4', 'new5'])
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--eos-only', action='store_true')
    mode.add_argument('--nucleus-only', action='store_true')
    parser.add_argument('--n0f', type=int)
    parser.add_argument('--n0b', type=int)
    parser.add_argument('--max-iterations', type=int)
    parser.add_argument('--tolerance', type=float)
    parser.add_argument('--constraint', type=float)
    parser.add_argument('--verbose', action='store_true')
    parsed_arguments = parser.parse_args(argv)
    try:
        states = []
        parameters = default_parameters()
        if not parsed_arguments.eos_only:
            config = NucleusConfig.from_file(parsed_arguments.input)
            changes = {
                key: getattr(parsed_arguments, key)
                for key in ('n0f', 'n0b', 'max_iterations', 'tolerance', 'constraint')
                if getattr(parsed_arguments, key) is not None
            }
            config = replace(config, **changes)
            for model in parsed_arguments.models:
                print(f'Computing {config.symbol}-{config.mass_number}, {model}...',flush=True)
                progress_callback = (
                    lambda progress_row: print(
                        f"  {progress_row['iteration']:4d}  "
                        f"residual={progress_row['residual']:.4e}  "
                        f"E/A={progress_row['energy_per_particle']:.8f}",
                        flush=True,
                    )
                ) if parsed_arguments.verbose else None
                state = solve_nucleus(
                    config,
                    model,
                    restart_directory=parsed_arguments.input.parent,
                    eos_parameters=parameters,
                    progress_callback=progress_callback,
                )
                states.append(state)
                print(f'  converged in {state.ii} iterations: E/A={state.ea:.8f} MeV, r_c={state.radius_charge:.8f} fm',flush=True)
        equation_of_state_table = (
            None
            if parsed_arguments.nucleus_only
            else equation_of_state(parameters)
        )
        write_results(states, parsed_arguments.output, equation_of_state_table)
        if equation_of_state_table is not None:
            print(f'EOS: {len(equation_of_state_table)} density points.', flush=True)
        print(f'Results saved to {parsed_arguments.output.resolve()}', flush=True)
        return 0
    except (ValueError,RuntimeError,FloatingPointError,OSError) as exc:
        print(f'Calculation failed: {exc}',file=sys.stderr)
        return 1


if __name__=='__main__':
    raise SystemExit(main())
