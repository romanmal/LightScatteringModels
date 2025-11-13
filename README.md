# LightScatteringModels
Python implementation of Lorentz-Drude (LD) model, Mie theory for light scattering and absorption

The LD model computes the dielectric function of bulk metallic gold. Those optical constants are then used to compute light scattering and absorption of spherical nanoparticles with Mie theory. Furthermore, a table of values at a range of wavelengths can be printed (see example 2 below) for use in either Mie scattering, absorption and extinction ratios or in the the 3rd party program DDSCAT for computing field enhancement. Similarly, for FEM analysis of field enhancement of plasmonic structures. A file of polystyrene optical constants is also supplied, based on Sultanova (2009) to generate an input file for DDSCAT.

This code was included in my MSc thesis in Nanotechnology, specialization in Nanoelectronics and Photonics, 2020. See bibliography in the thesis for proper references to Sultanova (2009) and Rakic (1998).

Full thesis text here: https://hdl.handle.net/11250/2778159

## Requirements:

Python 3.8+ (tested with 3.9)

NumPy, pandas

Optional: matplotlib for plotting

Dev/test: pytest

## Usage example 1:
from ld_model import wavelength_nm_to_ev, lorentz_drude_epsilon, refractive_index_from_eps

import numpy as np


ev = wavelength_nm_to_ev(500.0)  # energy in eV for 500 nm

eps = lorentz_drude_epsilon(np.array([ev]))

n, k = refractive_index_from_eps(eps)

print("500 nm -> n, k:", n[0], k[0])

## Usage example 2:
### generate Au tables for 400–800 nm and write outputs to ./out
python -m ld_model --outdir out --wl-min 400 --wl-max 800
### show plots (requires matplotlib)
python -m ld_model --plot
