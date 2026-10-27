"""Radial multipoles of static finite-range Fock propagators.

This module supplies the geometry-only Yukawa/Coulomb kernels used by a
spherical finite-nucleus RHF implementation. It does not construct Fock
operators or energy functionals.
"""
from __future__ import annotations

import numpy as np
from scipy.special import eval_legendre, ive, kve


def yukawa_multipole_coefficients(
    meson_mass_fm_inv: float,
    radii_fm: np.ndarray,
    max_multipole: int,
) -> np.ndarray:
    r"""Return radial multipoles for exp(-m R)/(4 pi R).

    The convention is D(r,r') = sum_LM d_L(r,r') Y_LM(rhat) Y*_LM(rhat'),
    where d_L = (2 m / pi) i_L(m r_<) k_L(m r_>). The zero-mass limit is
    the Coulomb multipole  r_<**L / ((2 L + 1) r_>**(L + 1)).
    """
    radii = np.asarray(radii_fm, dtype=float)
    if radii.ndim != 1 or radii.size == 0:
        raise ValueError("radii_fm must be a nonempty one-dimensional array")
    if not np.all(np.isfinite(radii)) or np.any(radii < 0):
        raise ValueError("radii_fm must contain finite nonnegative values")
    if not np.isfinite(meson_mass_fm_inv) or meson_mass_fm_inv < 0:
        raise ValueError("meson_mass_fm_inv must be finite and nonnegative")
    if not isinstance(max_multipole, (int, np.integer)) or max_multipole < 0:
        raise ValueError("max_multipole must be a nonnegative integer")

    r1, r2 = radii[:, None], radii[None, :]
    r_small, r_large = np.minimum(r1, r2), np.maximum(r1, r2)
    if np.any(r_large == 0):
        raise ValueError("the propagator is singular when both radii are zero")
    result = np.empty((max_multipole + 1, radii.size, radii.size))

    mass = float(meson_mass_fm_inv)
    if mass == 0.0:
        for ell in range(max_multipole + 1):
            result[ell] = r_small**ell / ((2 * ell + 1) * r_large ** (ell + 1))
        return result

    z_small, z_large = mass * r_small, mass * r_large
    for ell in range(max_multipole + 1):
        order = ell + 0.5
        # i_l(zs) k_l(zl) = pi/(2 sqrt(zs zl)) I_(l+1/2)(zs) K_(l+1/2)(zl).
        # Use exponentially scaled Bessel functions and restore the ratio.
        value = (
            mass
            * ive(order, z_small)
            * kve(order, z_large)
            * np.exp(z_small - z_large)
            / np.sqrt(z_small * z_large)
        )
        zero_small = z_small == 0.0
        if np.any(zero_small):
            if ell == 0:
                value[zero_small] = np.exp(-z_large[zero_small]) / r_large[zero_small]
            else:
                value[zero_small] = 0.0
        result[ell] = value
    return result


def reconstruct_yukawa_kernel(
    coefficients: np.ndarray,
    cosine_angle: float | np.ndarray,
) -> np.ndarray:
    """Reconstruct D(r,r',cos(gamma)) from radial multipoles.

    This diagnostic helper is not intended for production Fock contractions,
    which should operate directly on spherical-harmonic matrix elements.
    """
    radial = np.asarray(coefficients, dtype=float)
    if radial.ndim != 3 or radial.shape[0] == 0 or radial.shape[1] != radial.shape[2]:
        raise ValueError("coefficients must have shape (L, n_r, n_r)")
    cosine = np.asarray(cosine_angle, dtype=float)
    if not np.all(np.isfinite(cosine)) or np.any(np.abs(cosine) > 1.0):
        raise ValueError("cosine_angle values must be finite and in [-1, 1]")
    result = np.zeros((radial.shape[1], radial.shape[2]) + cosine.shape)
    expand = (None,) * cosine.ndim
    for ell, coefficient in enumerate(radial):
        result += (
            coefficient[(...,) + expand]
            * (2 * ell + 1)
            / (4.0 * np.pi)
            * eval_legendre(ell, cosine)
        )
    return result


__all__ = ["yukawa_multipole_coefficients", "reconstruct_yukawa_kernel"]
