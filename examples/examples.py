"""Reproduce selected figures and nuclear-structure results from docs/full.md.

Run from the repository root with ``python -m tools.reproduce_paper``.  The script
uses the local RMF EOS and spherical finite-nucleus solvers.  Published NICER
joint 68% contours and Table 2 values are supplied from published posterior
data and the paper, respectively.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from crmf.nuclear_matter.eos import (
    RMFParameters,
    pressure,
    solve_fields,
    symmetry_energy,
)
from crmf.finite_nucleus import NucleusConfig, solve_nucleus
from crmf.neutron_star import (
    build_beta_equilibrium_eos,
    integrate_star,
    make_crust,
)


# This script lives at the repository root, so its parent is the project root.
PROJECT_ROOT = Path(__file__).resolve().parent
OUT = PROJECT_ROOT / "output" / "paper_reproduction"
N0 = 0.148


MODEL_ROWS = {
    "FSUGold": dict(gs=10.5924, gw=14.3020, gr=5.88367, gd=0.0,
                    bs=0.000756283, cs=0.00396033, cw=0.0100000,
                    lv=0.240000, cds=0.0, nt=0.082, ncen=1.15),
    "FSU-J0": dict(gs=10.2143, gw=13.4353, gr=5.31345, gd=0.0,
                   bs=0.00159524, cs=0.000540269, cw=0.00528340,
                   lv=0.171042, cds=0.0, nt=0.081, ncen=1.00),
    "FSU-dlt67": dict(gs=10.2143, gw=13.4353, gr=7.26866, gd=6.70000,
                      bs=0.00159524, cs=0.000540269, cw=0.00528340,
                      lv=0.0214000, cds=0.0385000, nt=0.088, ncen=0.91),
    "FSU-dlt62": dict(gs=10.2143, gw=13.4353, gr=6.98518, gd=6.20000,
                      bs=0.00159524, cs=0.000540269, cw=0.00528340,
                      lv=0.0200112, cds=0.0540000, nt=0.089, ncen=0.92),
}
MODEL_LABELS = {
    "FSUGold": "FSUGold",
    "FSU-J0": "FSU-J0",
    "FSU-dlt67": r"FSU-$\delta$6.7",
    "FSU-dlt62": r"FSU-$\delta$6.2",
}
MODEL_COLORS = {
    "FSUGold": "#2468a2",
    "FSU-J0": "#d1495b",
    "FSU-dlt67": "#2a8c6b",
    "FSU-dlt62": "#8056a8",
}
MODEL_LINESTYLES = {
    "FSUGold": "-",
    "FSU-J0": "--",
    "FSU-dlt67": "-.",
    "FSU-dlt62": ":",
}
TOV_MAX_DENSITY = {
    "FSUGold": 1.35,
    "FSU-J0": 1.25,
    "FSU-dlt67": 1.05,
    "FSU-dlt62": 1.05,
}


def model_parameters(name: str) -> RMFParameters:
    row = MODEL_ROWS[name]
    return replace(
        RMFParameters(),
        g_s=row["gs"], g_w=row["gw"], g_r=row["gr"], g_d=row["gd"],
        b_s=row["bs"] / 0.9390, c_s=row["cs"], c_w=row["cw"],
        xl_v=row["lv"], c_ds=row["cds"], xms=0.4915, xmw=0.7825,
        xmd=0.9800, xmr=0.7630,
    )


# Approximate visual reconstruction of the Danielewicz et al. flow envelope.
# The source presents a band rather than tabulated coordinates.
FLOW_N_OVER_N0 = np.array([1.0, 1.4, 1.8, 2.2, 2.6, 3.0, 3.5, 4.0, 4.5, 5.0])
FLOW_PRESSURE_LOW = np.array([0, 3, 9, 17, 28, 41, 59, 79, 101, 125], dtype=float)
FLOW_PRESSURE_HIGH = np.array([0, 11, 23, 40, 61, 85, 116, 149, 184, 220], dtype=float)


PUBLISHED_TABLE = [
    (r"$n_0$ (fm$^{-3}$)", "0.148", "0.148", "0.148", "0.148", ""),
    (r"$e_0$ (MeV)", "-16.31", "-16.31", "-16.31", "-16.31", ""),
    (r"$K_0$ (MeV)", "229.2", "229.2", "229.2", "229.2", ""),
    (r"$J_0$ (MeV)", "-521.6", "-322.0", "-322.0", "-322.0", ""),
    (r"$m^*_{Dirac}$", "0.610", "0.610", "0.610", "0.610", ""),
    (r"$m^*_{Dirac,n,0.5}$", "0.610", "0.610", "0.576", "0.578", ""),
    (r"$m^*_{Dirac,p,0.5}$", "0.610", "0.610", "0.643", "0.640", ""),
    (r"$E_{sym,c}$ (MeV)", "25.88", "25.88", "25.88", "25.88", ""),
    (r"$L_c$ (MeV)", "48.02", "55.00", "55.00", "55.00", ""),
    (r"$E_{sym}$ (MeV)", "32.59", "33.45", "32.75", "32.53", ""),
    (r"$L$ (MeV)", "60.50", "68.14", "53.50", "48.21", ""),
    (r"$n_t$ (fm$^{-3}$)", "0.082", "0.081", "0.088", "0.089", ""),
    (r"$n_{cen}$ (fm$^{-3}$)", "1.15", "1.00", "0.91", "0.92", ""),
    (r"$R_{1.4}$ (km)", "12.66", "13.65", "13.67", "12.76", ""),
    (r"$\Lambda_{1.4}$", "401.0", "576.0", "578.5", "420.0", ""),
    (r"$M_{max}$ ($M_\odot$)", "1.72", "1.93", "2.05", "2.10", ""),
    (r"$E_B^{208}$ (MeV)", "-7.89", "-7.94", "-7.69", "-7.71", "-7.87"),
    (r"$R_{ch}^{208}$ (fm)", "5.54", "5.55", "5.55", "5.54", "5.50"),
    (r"$\Delta r_{np}^{208}$ (fm)", "0.207", "0.227", "0.235", "0.242", ""),
    (r"$\epsilon_{ls,2d\pi}^{208}$ (MeV)", "1.60", "1.57", "1.55", "1.55", "1.32"),
    (r"$\epsilon_{ls,3p\nu}^{208}$ (MeV)", "0.77", "0.75", "0.71", "0.71", "0.89"),
    (r"$\epsilon_{ls,2f\nu}^{208}$ (MeV)", "2.06", "1.98", "1.96", "1.95", "1.77"),
    (r"$E_B^{48}$ (MeV)", "-8.56", "-8.61", "-8.46", "-8.47", "-8.67"),
    (r"$R_{ch}^{48}$ (fm)", "3.46", "3.49", "3.49", "3.49", "3.48"),
    (r"$\Delta r_{np}^{48}$ (fm)", "0.197", "0.208", "0.213", "0.217", ""),
]


NICER_CONTOURS = [
    ("riley_2019", "Riley et al. 2019", "#377eb8"),
    ("miller_2019", "Miller et al. 2019", "#4daf4a"),
    ("riley_2021", "Riley et al. 2021", "#984ea3"),
    ("miller_2021", "Miller et al. 2021", "#ff7f00"),
]
NICER_CONTOUR_FILE = PROJECT_ROOT / "data" / "nicer_68_contours.csv"


def figure1(output: Path) -> dict[str, dict[str, np.ndarray]]:
    densities = np.linspace(0.035, 1.2, 250)
    curves = {}
    for name in MODEL_ROWS:
        parameters = model_parameters(name)
        pressures, symmetry = [], []
        for density in densities:
            fields = solve_fields(float(density), parameters=parameters)
            pressures.append(1000 * pressure(density, fields, parameters=parameters))
            symmetry.append(1000 * symmetry_energy(density, fields, parameters))
        curves[name] = {"density": densities, "pressure": np.array(pressures),
                        "symmetry": np.array(symmetry)}

    fig, (ax_p, ax_s) = plt.subplots(1, 2, figsize=(12.4, 5.1))
    flow_density = FLOW_N_OVER_N0 * N0
    ax_p.fill_between(flow_density, FLOW_PRESSURE_LOW, FLOW_PRESSURE_HIGH,
                      color="#c7d9e8", alpha=0.7, label="HIC flow band")
    for name, curve in curves.items():
        color = MODEL_COLORS[name]
        linestyle = MODEL_LINESTYLES[name]
        ax_p.plot(curve["density"], curve["pressure"], color=color, lw=2.1,
                  linestyle=linestyle,
                  label=MODEL_LABELS[name])
        ax_s.plot(curve["density"], curve["symmetry"], color=color, lw=2.1,
                  linestyle=linestyle,
                  label=MODEL_LABELS[name])
        nc = MODEL_ROWS[name]["ncen"]
        star = np.interp(nc, curve["density"], curve["symmetry"])
        ax_s.scatter([nc], [star], marker="*", s=105, color=color,
                     edgecolor="white", linewidth=0.5, zorder=5)
    ax_p.set(xlim=(0, 0.8), ylim=(0, 230), xlabel=r"Baryon density $n_B$ (fm$^{-3}$)",
             ylabel=r"Pressure $P$ (MeV fm$^{-3}$)", title="(a) Symmetric-matter EOS")
    ax_p.set_ylim(-10, 230)
    ax_s.set(xlim=(0, 1.2), ylim=(0, 100), xlabel=r"Baryon density $n_B$ (fm$^{-3}$)",
             ylabel=r"Symmetry energy $E_{sym}$ (MeV)", title="(b) Symmetry energy")
    ax_s.set_ylim(0, 180)
    for axis in (ax_p, ax_s):
        axis.grid(color="#d9d9d9", linewidth=0.6)
        axis.set_axisbelow(True)
    ax_p.legend(frameon=False, fontsize=8.6, loc="upper left")
    ax_s.legend(handles=[
        *[Line2D([0], [0], color=MODEL_COLORS[name], lw=2.1,
                 linestyle=MODEL_LINESTYLES[name],
                 label=MODEL_LABELS[name]) for name in MODEL_ROWS],
        Line2D([0], [0], marker="*", color="black", linestyle="None",
               label=r"$n_{cen}$ at $M_{max}$"),
    ], frameon=False, fontsize=8.6, loc="upper left")
    fig.suptitle("Pressure and symmetry energy", fontsize=14)
    fig.tight_layout(rect=(0, 0.055, 1, 0.94))
    fig.text(0.5, 0.012,
             "HIC flow envelope is an approximate visual reconstruction of Danielewicz et al. (2002).",
             ha="center", fontsize=8, color="#555555")
    save_figure(fig, output / "fig1_eos")
    with (output / "fig1_eos.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["density_fm-3"] + [f"{name}_{col}" for name in curves
                                            for col in ("pressure_MeV_fm-3", "Esym_MeV")])
        for row in range(len(densities)):
            writer.writerow([f"{densities[row]:.8g}"] + [
                f"{curves[name][column][row]:.10g}"
                for name in curves for column in ("pressure", "symmetry")])
    return curves


def stable_eos(name: str, max_density: float = 1.55):
    r"""Build the named paper model's beta-equilibrium core EOS.

    Args:
        name (str): Model key in ``MODEL_ROWS``.
        max_density (float): Maximum baryon density in $\mathrm{fm}^{-3}$.

    Returns:
        tuple: Density, pressure and energy-density arrays for the TOV solver.
    """
    return build_beta_equilibrium_eos(
        model_parameters(name), MODEL_ROWS[name]["nt"], max_density
    )


def figure2(output: Path):
    fig, ax = plt.subplots(figsize=(8.3, 6.3))
    curves = {}
    for name in MODEL_ROWS:
        densities, p_core, e_core = stable_eos(name, TOV_MAX_DENSITY[name])
        crust = make_crust(densities[0], p_core[0], e_core[0])
        eos = (densities, p_core, e_core, crust)
        central_densities = np.geomspace(densities[0] * 1.18, densities[-1], 90)
        stars = np.array([integrate_star(float(nc), eos) for nc in central_densities])
        valid = np.isfinite(stars).all(axis=1)
        stars = stars[valid]
        central_densities = central_densities[valid]
        if len(stars) < 8:
            raise RuntimeError(f"Too few TOV solutions for {name}")
        curves[name] = (central_densities, stars)
        color = MODEL_COLORS[name]
        ax.plot(stars[:, 0], stars[:, 1], color=color, lw=2.2,
                label=MODEL_LABELS[name])
        maximum = int(np.argmax(stars[:, 1]))
        ax.scatter(stars[maximum, 0], stars[maximum, 1], marker="*", s=75,
                   color=color, edgecolor="white", linewidth=0.5, zorder=5)

    regions = {}
    with NICER_CONTOUR_FILE.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            key = (row["source"], row["component"])
            regions.setdefault(key, ([], []))
            regions[key][0].append(float(row["radius_km"]))
            regions[key][1].append(float(row["mass_msun"]))
    for source, label, color in NICER_CONTOURS:
        components = [region for (key, _), region in regions.items() if key == source]
        if not components:
            raise RuntimeError(f"Missing NICER contour data for {label}: {NICER_CONTOUR_FILE}")
        for index, (radius, mass) in enumerate(components):
            ax.fill(radius, mass, color=color, alpha=0.12, linewidth=0, zorder=1.4)
            ax.plot(radius, mass, color=color, lw=1.25, alpha=0.9, zorder=2.8,
                    label=label + " (68%)" if index == 0 else None)
    ax.set(xlim=(8, 17), ylim=(0.5, 2.5), xlabel="Radius (km)",
           ylabel=r"Gravitational mass ($M_\odot$)",
           title="Static neutron-star mass-radius relation")
    ax.grid(color="#d9d9d9", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=8.2, ncol=2, loc="upper left")
    fig.tight_layout(rect=(0, 0.045, 1, 1))
    fig.text(0.5, 0.008,
             "68% joint NICER contours: archived Riley 2019 boundary; smoothed posterior-sample contours for the other analyses.",
             ha="center", fontsize=8, color="#555555")
    save_figure(fig, output / "fig2_mr")
    with (output / "fig2_mr.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["model", "central_density_fm-3", "radius_km", "mass_Msun"])
        for name, (density_values, stars) in curves.items():
            writer.writerows([[name, f"{density:.8g}", f"{radius:.8g}", f"{mass:.8g}"]
                              for density, (radius, mass) in zip(density_values, stars)])
    maxima = {}
    summary_rows = []
    for name, (density_values, stars) in curves.items():
        maximum_index = int(np.argmax(stars[:, 1]))
        rising = stars[:maximum_index + 1]
        radius_14 = float(np.interp(1.4, rising[:, 1], rising[:, 0]))
        maxima[name] = (float(density_values[maximum_index]),
                        float(stars[maximum_index, 0]),
                        float(stars[maximum_index, 1]), radius_14)
        summary_rows.append([name, *[f"{value:.8g}" for value in maxima[name]]])
    with (output / "tov_summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["model", "central_density_at_Mmax_fm-3", "R_at_Mmax_km",
                         "Mmax_Msun", "R1.4_km"])
        writer.writerows(summary_rows)
    return maxima


def save_table(output: Path):
    columns = ["Quantity", "FSUGold", "FSU-J0", r"FSU-$\delta$6.7",
               r"FSU-$\delta$6.2",
               "Experimental"]
    with (output / "table2.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        writer.writerow(columns)
        writer.writerows(PUBLISHED_TABLE)
    lines = ["# Table 2. Nuclear-matter, neutron-star, and finite-nucleus properties", "",
             "| " + " | ".join(columns) + " |",
             "| " + " | ".join(["---"] * len(columns)) + " |"]
    lines.extend("| " + " | ".join(row) + " |" for row in PUBLISHED_TABLE)
    lines.extend(["", "Source: Table 2 in `docs/full.md`. Experimental values are listed in the final column."])
    (output / "table2.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def collect_levels(state, species: int):
    levels = []
    for block in range(1, state.nb + 1):
        count = int(state.kdcan[block, species])
        first = int(state.kacan[block, species]) + 1
        kappa = int(state.kb[block])
        l = int(state.ilb[block, 1])
        j2 = int(state.mb[block]) - 1
        label_j = f"{j2}/2"
        letter = state.tl[l]
        for radial, index in enumerate(range(first, first + count), start=1):
            levels.append({
                "energy": float(state.eecan[index, species]),
                "occupation": float(state.vvcan[index, species]),
                "x": l + (0.18 if kappa > 0 else -0.18),
                "label": f"{radial}{letter}{label_j}",
            })
    return levels


def level_diagram(output: Path, nucleus: str, states: dict[str, object]):
    letters = list("spdfghik")
    fig, axes = plt.subplots(2, 4, figsize=(15, 7.5), sharey=True)
    for column, (name, state) in enumerate(states.items()):
        for row, (species, species_name) in enumerate(((1, "Neutrons"), (2, "Protons"))):
            axis = axes[row, column]
            fermi = float(state.ala[species])
            levels = collect_levels(state, species)
            visible = [level for level in levels
                       if -20 <= level["energy"] - fermi <= 8]
            for level in visible:
                occupied = level["occupation"] >= 0.5
                color = "#397c9d" if occupied else "#d16c54"
                y = level["energy"] - fermi
                x = level["x"]
                axis.hlines(y, x - 0.17, x + 0.17, color=color, linewidth=1.5)
                axis.text(x + 0.2, y, level["label"], fontsize=6.3,
                          va="center", color="#363636")
            axis.axhline(0, color="#333333", linewidth=0.7, linestyle="--")
            axis.set_xlim(-0.6, 7.7)
            axis.set_ylim(-20, 8)
            axis.set_xticks(range(len(letters)), letters)
            axis.grid(axis="y", color="#e2e2e2", linewidth=0.5)
            axis.set_axisbelow(True)
            if row == 0:
                axis.set_title(MODEL_LABELS[name], fontsize=10)
            if column == 0:
                axis.set_ylabel(f"{species_name}\n$E-\\lambda$ (MeV)")
    fig.suptitle(f"Spherical single-particle levels in {nucleus}", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save_figure(fig, output / f"levels_{nucleus}")


def finite_nuclei(output: Path, basis: int, max_iterations: int):
    states_by_nucleus = {}
    rows = []
    for nucleus, mass, proton_number, symbol in (("208Pb", 208, 82, "Pb"),
                                                   ("48Ca", 48, 20, "Ca")):
        states_by_nucleus[nucleus] = {}
        for model in MODEL_ROWS:
            print(f"Finite nucleus: {nucleus}, {model}", flush=True)
            finite_model = "FSU" if model == "FSUGold" else model
            config = NucleusConfig(
                symbol=symbol, mass_number=mass, n0f=basis, n0b=min(30, basis + 10),
                gap_neutron=0.0, gap_proton=0.0, force=finite_model,
                max_iterations=max_iterations, tolerance=1e-5,
            )
            state = solve_nucleus(config, finite_model)
            states_by_nucleus[nucleus][model] = state
            rows.append([nucleus, model, f"{state.ea:.8f}",
                         f"{state.radius_charge:.8f}",
                         f"{state.radius_neutron - state.radius_proton:.8f}",
                         f"{state.els_2d_p:.8f}", f"{state.els_3p_n:.8f}",
                         f"{state.els_2f_n:.8f}", str(state.ii), f"{state.si:.4g}"])
        level_diagram(output, nucleus, states_by_nucleus[nucleus])
    with (output / "finite_nucleus_code.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        writer.writerow(["nucleus", "model", "E/A_MeV", "Rch_fm", "skin_fm",
                         "d2p_MeV", "p3n_MeV", "f2n_MeV", "iterations", "residual"])
        writer.writerows(rows)


def save_figure(fig, prefix: Path):
    prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(prefix.with_suffix(".png"), dpi=240, bbox_inches="tight")
    fig.savefig(prefix.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument("--basis", type=int, default=20,
                        help="fermion oscillator shells for finite nuclei (1-20)")
    parser.add_argument("--max-iterations", type=int, default=500)
    parser.add_argument("--skip-finite", action="store_true",
                        help="only generate EOS/TOV figures and Table 2")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    save_table(args.output)
    print("Generating pressure and symmetry-energy figure...", flush=True)
    figure1(args.output)
    print("Solving beta-equilibrium EOS and TOV sequences...", flush=True)
    maxima = figure2(args.output)
    for name, values in maxima.items():
        central_density, radius, mass, radius_14 = values
        print(f"TOV {name}: Mmax={mass:.3f} Msun at n={central_density:.3f} fm^-3; "
              f"Rmax={radius:.3f} km, R1.4={radius_14:.3f} km", flush=True)
    if not args.skip_finite:
        finite_nuclei(args.output, args.basis, args.max_iterations)
    print(f"Outputs: {args.output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
