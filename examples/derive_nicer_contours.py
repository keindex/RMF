"""Convert public NICER M-R posterior data to joint 68% contour coordinates."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "nicer_68_contours.csv"


def read_contour_components(path: Path) -> list[np.ndarray]:
    components = []
    points = []
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = line.partition("#")[0].split()
        if len(fields) < 2:
            if points:
                components.append(np.asarray(points))
                points = []
            continue
        try:
            points.append((float(fields[0]), float(fields[1])))
        except ValueError:
            continue
    if points:
        components.append(np.asarray(points))
    components = [component for component in components if len(component) >= 4]
    if not components:
        raise ValueError(f"No closed 2D contour found in {path}")
    return components


def read_samples(path: Path, usecols: tuple[int, ...]) -> np.ndarray:
    samples = np.loadtxt(path, comments="#", usecols=usecols, ndmin=2)
    if samples.shape[0] < 100:
        raise ValueError(f"Too few posterior samples in {path}")
    return samples


def hpd_contours(radius: np.ndarray, mass: np.ndarray,
                 weights: np.ndarray | None = None, bins: int = 360,
                 probability: float = 0.68) -> list[np.ndarray]:
    valid = np.isfinite(radius) & np.isfinite(mass)
    if weights is not None:
        valid &= np.isfinite(weights) & (weights > 0)
        weights = weights[valid]
    radius, mass = radius[valid], mass[valid]
    if radius.size < 100:
        raise ValueError("Too few finite posterior samples")
    if weights is None:
        weights = np.ones(radius.size)

    limits = []
    for values in (radius, mass):
        low, high = np.quantile(values, (0.001, 0.999))
        padding = 0.04 * (high - low)
        limits.append((low - padding, high + padding))
    histogram, xedges, yedges = np.histogram2d(
        radius, mass, bins=bins, range=limits, weights=weights,
    )
    xcenters = (xedges[:-1] + xedges[1:]) / 2
    ycenters = (yedges[:-1] + yedges[1:]) / 2

    effective_n = weights.sum() ** 2 / np.square(weights).sum()
    covariance = np.cov(np.vstack((radius, mass)), aweights=weights)
    bandwidth = np.sqrt(np.diag(covariance)) * effective_n ** (-1 / 6)
    spacing = (xedges[1] - xedges[0], yedges[1] - yedges[0])
    smoothing = tuple(max(1.0, width / step) for width, step in zip(bandwidth, spacing))
    density = gaussian_filter(histogram, smoothing, mode="constant")
    density /= density.sum()
    ordered = np.sort(density.ravel())[::-1]
    threshold_index = min(np.searchsorted(np.cumsum(ordered), probability), ordered.size - 1)
    threshold = ordered[threshold_index]

    fig, ax = plt.subplots()
    contour_set = ax.contour(xcenters, ycenters, density.T, levels=[threshold])
    polygons = [poly for path in contour_set.get_paths()
                for poly in path.to_polygons(closed_only=True) if len(poly) >= 4]
    plt.close(fig)
    if not polygons:
        raise RuntimeError("The posterior KDE did not produce a closed HPD contour")
    return polygons


def write_region(writer, source: str, components: list[np.ndarray]) -> None:
    for component_index, points in enumerate(components):
        for radius, mass in points:
            writer.writerow((source, component_index, f"{radius:.8f}", f"{mass:.8f}"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--riley-2019-contour", type=Path, required=True,
                        help="M_R_contour_68.txt from Zenodo record 7096789")
    parser.add_argument("--miller-2019-samples", type=Path, required=True,
                        help="J0030_3spot_RM.txt from Zenodo record 3473466")
    parser.add_argument("--riley-2021-samples", type=Path, required=True,
                        help="Riley et al. J0740 equal-weight posterior samples")
    parser.add_argument("--miller-2021-samples", type=Path, required=True,
                        help="NICER+XMM_J0740_RM.txt from Zenodo record 4670689")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--bins", type=int, default=360)
    args = parser.parse_args()

    riley_2019 = read_contour_components(args.riley_2019_contour)

    miller_2019 = read_samples(args.miller_2019_samples, (0, 1))
    miller_2019_contours = hpd_contours(miller_2019[:, 0], miller_2019[:, 1],
                                        bins=args.bins)

    riley_2021 = read_samples(args.riley_2021_samples, (0, 1))
    riley_2021_contours = hpd_contours(riley_2021[:, 1], riley_2021[:, 0],
                                       bins=args.bins)

    miller_2021 = read_samples(args.miller_2021_samples, (0, 1, 2))
    miller_2021_contours = hpd_contours(miller_2021[:, 0], miller_2021[:, 1],
                                        weights=miller_2021[:, 2], bins=args.bins)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("source", "component", "radius_km", "mass_msun"))
        write_region(writer, "riley_2019", riley_2019)
        write_region(writer, "miller_2019", miller_2019_contours)
        write_region(writer, "riley_2021", riley_2021_contours)
        write_region(writer, "miller_2021", miller_2021_contours)
    print(f"Wrote {args.output} from the four published posterior datasets")


if __name__ == "__main__":
    main()
