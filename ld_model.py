#!/usr/bin/env python3
"""
ld_model.py

Lorentz-Drude module for gold (Au) and utilities to write DDSCAT-compatible tables.
- Clear physical constants and units
- Vectorized numerical operations
- Parameterized I/O and logging
- Optional plotting (matplotlib)
"""

from __future__ import annotations
import argparse
import logging
from pathlib import Path
from typing import Sequence, Tuple

import numpy as np
import pandas as pd

# Optional plotting backend
try:
    import matplotlib.pyplot as plt
    HAS_MPL = True
except Exception:
    HAS_MPL = False

# ---------- Physical constants (SI) ----------
C_M_S = 299_792_458.0                 # m / s
E_CHARGE = 1.602176634e-19            # C (elementary charge)
H_PLANCK = 6.62607015e-34             # J s
PI = np.pi

# ---------- Model parameters (Rakic 1998) in eV ----------
W_P = 9.03            # plasma frequency (eV)
F0 = 0.760
G0 = 0.053
OMEGA_P = np.sqrt(F0) * W_P

F_J = np.array([0.024, 0.010, 0.071, 0.601, 4.384])
G_J = np.array([0.241, 0.345, 0.870, 2.494, 2.214])
W_J = np.array([0.415, 0.830, 2.969, 4.304, 13.32])

# ---------- Helper functions ----------


def wavelength_nm_to_ev(lambda_nm: Sequence[float]) -> np.ndarray:
    """Convert wavelength in nm to photon energy in eV (vectorized).
    Input: array-like in nanometers.
    Returns: numpy array in eV.
    """
    lam_m = np.asarray(lambda_nm, dtype=float) * 1e-9
    freq = C_M_S / lam_m
    ev = (H_PLANCK * freq) / E_CHARGE
    return ev


def drude_intraband_epsilon(w_eV: np.ndarray, omega_p: float = OMEGA_P, g0: float = G0) -> np.ndarray:
    """Drude intraband contribution to the dielectric function.
    Formula: eps = 1 - Omega_p**2 / (w * (w + 1j * gamma))
    w and gamma are in eV; returns complex epsilon.
    """
    w = np.asarray(w_eV, dtype=np.complex128)
    return 1.0 - (omega_p ** 2) / (w * (w + 1j * g0))


def lorentz_interband_epsilon(
    w_eV: np.ndarray,
    f_j: np.ndarray = F_J,
    w_j: np.ndarray = W_J,
    g_j: np.ndarray = G_J,
    w_p: float = W_P,
) -> np.ndarray:
    """Vectorized Lorentz contributions for all oscillators.
    Returns complex epsilon = sum_j f_j * w_p^2 / (w_j^2 - w^2 + i w gamma_j).
    """
    w = np.asarray(w_eV, dtype=np.complex128)  # (N,)
    f = np.asarray(f_j, dtype=float)           # (M,)
    wjs = np.asarray(w_j, dtype=float)         # (M,)
    gjs = np.asarray(g_j, dtype=float)         # (M,)

    numerators = (f * (w_p ** 2))[:, None]                # (M,1)
    denominators = (wjs[:, None] ** 2) - (w[None, :] ** 2) + 1j * w[None, :] * gjs[:, None]  # (M,N)
    contributions = numerators / denominators
    return np.sum(contributions, axis=0)


def lorentz_drude_epsilon(w_eV: np.ndarray, **kwargs) -> np.ndarray:
    """Full dielectric function: Drude intraband + Lorentz interband."""
    return drude_intraband_epsilon(w_eV, **kwargs) + lorentz_interband_epsilon(w_eV, **kwargs)


def refractive_index_from_eps(eps: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Given complex eps, return n (real part) and k (imaginary part magnitude)."""
    n_complex = np.sqrt(eps.astype(np.complex128))
    n_real = np.real(n_complex)
    k_imag = np.imag(n_complex)
    return n_real, np.abs(k_imag)


# ---------- IO and formatting helpers ----------


def write_ddscat_table(
    outfile: Path,
    wavelengths_um: np.ndarray,
    n_vals: np.ndarray,
    k_vals: np.ndarray,
    eps_vals: np.ndarray,
    header_title: str,
) -> None:
    """Write DDSCAT-like table (columns: wave(um), Re(n), Im(n), eps1, eps2) with text header."""
    df = pd.DataFrame(
        {
            "wave (um)": np.round(wavelengths_um, 6),
            "Re(n)": np.round(n_vals, 3),
            "Im(n)": np.round(k_vals, 3),
            "eps1": np.round(np.real(eps_vals), 3),
            "eps2": np.round(np.imag(eps_vals), 3),
        }
    )
    temp = outfile.with_suffix(".tmp")
    with temp.open("w", encoding="utf-8") as f:
        f.write(f"{header_title}\n")
        f.write("1 2 3 0 0 = columns for wave, Re(n), Im(n), eps1, eps2\n")
        df.to_csv(f, sep="\t", index=False)
    temp.replace(outfile)


# ---------- Generation routines ----------


def generate_au_ld_table(
    output_path: Path,
    wavelengths_nm: Sequence[float],
    plot: bool = False,
) -> None:
    """Generate Au table based on the Lorentz-Drude model.
    wavelengths_nm: sequence in nanometers
    """
    wl_nm = np.asarray(wavelengths_nm, dtype=float)
    ev = wavelength_nm_to_ev(wl_nm)  # eV
    eps = lorentz_drude_epsilon(ev, f_j=F_J, w_j=W_J, g_j=G_J, w_p=W_P, omega_p=OMEGA_P, g0=G0)
    n, k = refractive_index_from_eps(eps)

    out_csv = output_path / "Au_ld_400_800.csv"
    pd.DataFrame({"wavelength_nm": wl_nm, "n": n, "k": k}).to_csv(out_csv, index=False)
    logging.info("Wrote %s", out_csv)

    wavelengths_um = wl_nm * 1e-3
    out_ddscat = output_path / "Au_ld_ddscat.txt"
    header = f"Gold, Lorentz-Drude model, {int(wl_nm.min())} - {int(wl_nm.max())} nm"
    write_ddscat_table(out_ddscat, wavelengths_um, n, k, eps, header)
    logging.info("Wrote %s", out_ddscat)

    if plot and HAS_MPL:
        plt.figure()
        plt.plot(wl_nm, n, label="n (real)")
        plt.plot(wl_nm, k, label="k (imag abs)")
        plt.xlabel("Wavelength (nm)")
        plt.legend()
        plt.title("Au Lorentz-Drude n,k")
        plt.tight_layout()
        plt.show()
    elif plot and not HAS_MPL:
        logging.warning("Matplotlib not available - skipping plot.")


def generate_polystyrene_ddscat(
    ps_input_file: Path, output_path: Path, plot: bool = False
) -> None:
    """Read a simple file with {wave_um Re(n)} and write a DDSCAT table for polystyrene.
    Expected input: whitespace-separated two columns without header.
    """
    if not ps_input_file.exists():
        raise FileNotFoundError(f"Input file not found: {ps_input_file}")

    df = pd.read_csv(ps_input_file, sep=r"\s+", header=None, names=["wave_um", "Re_n"])
    wave_um = df["wave_um"].to_numpy(dtype=float)
    n_vals = df["Re_n"].to_numpy(dtype=float)
    eps_vals = n_vals ** 2
    k_vals = np.zeros_like(n_vals)

    out_ddscat = output_path / "PS_sultanova_ddscat.txt"
    header = f"Polystyrene, Sultanova (2009) {wave_um.min()*1e3:.0f} - {wave_um.max()*1e3:.0f} nm"
    write_ddscat_table(out_ddscat, wave_um, n_vals, k_vals, eps_vals, header)
    logging.info("Wrote %s", out_ddscat)

    if plot and HAS_MPL:
        plt.figure()
        plt.plot(wave_um * 1e3, n_vals, label="n")
        plt.xlabel("Wavelength (nm)")
        plt.legend()
        plt.title("Polystyrene n (Sultanova 2009)")
        plt.tight_layout()
        plt.show()


# ---------- CLI ----------


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Generate material tables from Lorentz-Drude model for Au and PS")
    p.add_argument(
        "--outdir",
        type=Path,
        default=Path.cwd() / "out",
        help="Output directory for generated files (default: ./out)",
    )
    p.add_argument(
        "--wl-min",
        type=int,
        default=400,
        help="Min wavelength nm (default 400)",
    )
    p.add_argument(
        "--wl-max",
        type=int,
        default=800,
        help="Max wavelength nm (default 800)",
    )
    p.add_argument("--plot", action="store_true", help="Show plots (requires matplotlib)")
    p.add_argument("--ps-input", type=Path, default=None, help="Input file for polystyrene (two columns: um Re(n))")
    p.add_argument("--verbose", action="store_true", help="Verbose logging")
    return p


def main(argv: Sequence[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s: %(message)s")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    wl = np.arange(args.wl_min, args.wl_max + 1, 1)
    try:
        generate_au_ld_table(outdir, wl, plot=args.plot)
    except Exception as e:
        logging.exception("Error generating Au table: %s", e)
        raise

    if args.ps_input:
        try:
            generate_polystyrene_ddscat(args.ps_input, outdir, plot=args.plot)
        except Exception as e:
            logging.exception("Error generating PS table: %s", e)
            raise

    logging.info("Done. Results in %s", outdir)


if __name__ == "__main__":
    main()