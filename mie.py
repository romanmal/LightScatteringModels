"""
mie.py
Modular Mie coefficients and cross-section utilities.
Author: Roman Malyshev
"""

from typing import Tuple, Sequence
import numpy as np
from scipy.special import jv, yv
from math import pi, sqrt


# === Constants ===
C = 3e8             # m/s
H = 6.63e-34        # Js
E_CHARGE = 1.6e-19  # C


# === Utility conversions ===
def lda_to_ev(lda_m: float) -> float:
    """Convert wavelength in meters to electronvolts, rounded to 2 decimals."""
    if lda_m <= 0:
        raise ValueError("Wavelength must be positive")
    ev = H * C / lda_m / E_CHARGE
    return round(ev, 2)


def ev_to_lda_nm(ev: float) -> int:
    """Convert electronvolts to wavelength in nanometers (int)."""
    if ev <= 0:
        raise ValueError("Energy must be positive")
    lda_m = H * C / (E_CHARGE * ev)
    return int(lda_m * 1e9)


# === Mie cross sections ===
def mie_cross_sections(lambda0: float, n_particle: complex, r: float, n_medium: float
                      ) -> Tuple[float, float, float]:
    """
    Compute scattering (Qsca), absorption (Qabs) and extinction (Qext) cross-sections
    for a homogeneous sphere using Mie theory (dimensionless efficiencies scaled
    by geometric cross-section should be done externally if desired).

    Parameters
    ----------
    lambda0 : float
        Vacuum wavelength (m)
    n_particle : complex
        Complex refractive index of particle (n + i*k)
    r : float
        Particle radius (m)
    n_medium : float
        Refractive index of surrounding medium (real)

    Returns
    -------
    Qsca, Qabs, Qext : tuple of floats
    """
    # Validate
    if lambda0 <= 0:
        raise ValueError("lambda0 must be positive")
    if r <= 0:
        raise ValueError("radius must be positive")
    if n_medium <= 0:
        raise ValueError("n_medium must be positive")

    m = n_particle / n_medium
    # wavelength inside medium
    lda_med = lambda0 / n_medium
    k = 2 * pi / lda_med
    x = k * r
    mx = m * x

    # order cutoff
    N = int(np.round(2 + x + 4 * x ** (1 / 3)))
    if N < 1:
        N = 1

    n = np.arange(1, N + 1)
    nu = n + 0.5

    # spherical Bessel functions from cylinder Bessel jv,yv with argument x
    # careful for very small x: use series expansions would be ideal (not included)
    j_n = np.sqrt(0.5 * pi / x) * jv(nu, x)
    y_n = np.sqrt(0.5 * pi / x) * yv(nu, x)
    h_n = j_n + 1j * y_n
    j_n_mx = np.sqrt(0.5 * pi / mx) * jv(nu, mx)

    # first order simple spherical Bessel
    j0 = np.sin(x) / x
    y0 = -np.cos(x) / x
    j0_mx = np.sin(mx) / mx

    # Riccati-Bessel functions
    psi_n = x * j_n
    xi_n = x * h_n
    chi_n = x * y_n
    psi_mx = mx * j_n_mx

    psi_0 = x * j0
    chi_0 = x * y0
    psi_0_mx = mx * j0_mx

    psi_n_1 = np.concatenate(([psi_0], psi_n[:-1]))
    psi_mx_1 = np.concatenate(([psi_0_mx], psi_mx[:-1]))
    chi_n_1 = np.concatenate(([chi_0], chi_n[:-1]))

    d_psi = (psi_n_1 - n / x * psi_n)
    d_psim = (psi_mx_1 - n / mx * psi_mx)
    d_xi = (psi_n_1 + 1j * chi_n_1) - n / x * (psi_n + 1j * chi_n)

    a_n = (m * psi_mx * d_psi - psi_n * d_psim) / (m * psi_mx * d_xi - xi_n * d_psim)
    b_n = (psi_mx * d_psi - m * psi_n * d_psim) / (psi_mx * d_xi - m * xi_n * d_psim)

    Qsca = 2 * pi / (k ** 2) * np.sum((2 * n + 1) * (np.abs(a_n) ** 2 + np.abs(b_n) ** 2)).real
    Qext = 2 * pi / (k ** 2) * np.sum((2 * n + 1) * (a_n + b_n).real).real
    Qabs = Qext - Qsca

    return float(Qsca), float(Qabs), float(Qext)


# === Helper: normalize by geometric cross-section ===
def normalize_by_area(Q: float, r: float) -> float:
    """Return efficiency per geometric area: Q / (pi * r^2)."""
    if r <= 0:
        raise ValueError("radius must be positive")
    area = pi * r ** 2
    return Q / area
