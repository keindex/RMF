"""Plot a precomputed nuclear-matter EOS table.

Run: python examples/plot_eos.py
The symmetry energy and slope are converted from GeV to MeV for display.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_table(input_path: Path):
    table = np.loadtxt(input_path, ndmin=2)
    if table.ndim != 2 or table.shape[1] != 8 or len(table) < 2:
        raise ValueError("EOS must contain at least two rows and eight columns")
    if not np.isfinite(table).all():
        raise ValueError("EOS contains nonfinite values")
    if np.any(np.diff(table[:, 0]) <= 0):
        raise ValueError("EOS densities must be strictly increasing")
    return table


def plot_table(table: np.ndarray, output: Path):
    blue = "#1864ab"
    field_unit = r"$\sqrt{\mathrm{fm}^{-3}/\mathrm{GeV}}$"
    panels = [
        (1, 1, "Energy per nucleon", r"$E/A - 939\,\mathrm{MeV}$ (MeV)"),
        (2, 1, r"Scalar field $\sigma$", field_unit),
        (3, 1, r"Vector field $\omega$", field_unit),
        (4, 1, r"Isovector field $\rho$", field_unit),
        (5, 1, "Energy per nucleon (source units)", r"$E/A - XM$ (GeV)"),
        (6, 1000, "Symmetry energy", r"$E_{\mathrm{sym}}$ (MeV)"),
        (7, 1000, "Symmetry-energy slope", r"$L$ (MeV)"),
    ]
    with plt.rc_context({"font.size": 11, "axes.titlesize": 12,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "savefig.facecolor": "white"}):
        fig, axes = plt.subplots(2, 4, figsize=(17, 8.8))
        for ax, (column, scale, title, ylabel) in zip(axes.flat, panels):
            ax.plot(table[:, 0], scale * table[:, column], color=blue,
                    linewidth=2)
            ax.set_title(title, pad=10)
            ax.set_ylabel(ylabel)
            ax.set_xlabel(r"Baryon density $n_B$ (fm$^{-3}$)")
            ax.set_xlim(table[0, 0], table[-1, 0])
            ax.grid(True, color="#dedede", linewidth=0.6, alpha=0.75)
            ax.set_axisbelow(True)
        axes.flat[-1].set_visible(False)

        fig.suptitle("Symmetric nuclear matter EOS", fontsize=19, y=0.98)
        fig.tight_layout(rect=(0, 0, 1, 0.94), h_pad=2.0, w_pad=2.0)
        output.parent.mkdir(parents=True, exist_ok=True)
        paths = [output.with_suffix(".png"), output.with_suffix(".pdf")]
        for path in paths:
            fig.savefig(path, dpi=220)
        plt.close(fig)
    return paths


def main():
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=project_root / "output" / "EOS.OUT",
                        help="Python EOS table (eight columns)")
    parser.add_argument("--output", type=Path, default=project_root / "output" / "eos",
                        help="Output filename prefix for PNG and PDF")
    args = parser.parse_args()
    table = load_table(args.input)
    paths = plot_table(table, args.output)
    print(f"EOS table: {len(table)} density points, {table.shape[1]} columns")
    for path in paths:
        print(f"Saved: {path.resolve()}")


if __name__ == "__main__":
    main()
