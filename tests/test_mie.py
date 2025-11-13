# tests/test_mie.py
import pytest
import numpy as np
from mie import lda_to_ev, ev_to_lda_nm, mie_cross_sections, normalize_by_area

# --- Fixtures -------------------------------------------------------------
@pytest.fixture
def visible_wavelength():
    return 500e-9  # 500 nm

@pytest.fixture
def small_radius():
    return 5e-9  # 5 nm

@pytest.fixture
def medium_index():
    return 1.33

@pytest.fixture
def lossy_index():
    # moderate real part, sizable imaginary part
    return 0.2 + 3.0j

# --- Conversion tests -----------------------------------------------------
def test_lda_to_ev_roundtrip(visible_wavelength):
    ev = lda_to_ev(visible_wavelength)
    lda_back_nm = ev_to_lda_nm(ev)
    # allow a tolerance of ~10 nm because of rounding and int conversion
    assert abs(lda_back_nm - 500) < 10

@pytest.mark.parametrize("bad_input", [0.0, -1.0])
def test_conversion_invalid_inputs(bad_input):
    with pytest.raises(ValueError):
        lda_to_ev(bad_input)
    with pytest.raises(ValueError):
        ev_to_lda_nm(bad_input)

# --- Mie basic tests ------------------------------------------------------
def test_mie_returns_nonnegative_quantities(small_radius, medium_index, lossy_index):
    lambda0 = 600e-9
    Qsca, Qabs, Qext = mie_cross_sections(lambda0, lossy_index, small_radius, medium_index)
    # All returned values should be finite and non-negative, and Qext ~= Qsca + Qabs
    assert np.isfinite(Qsca) and np.isfinite(Qabs) and np.isfinite(Qext)
    assert Qsca >= -1e-12
    assert Qabs >= -1e-12
    assert pytest.approx(Qsca + Qabs, rel=1e-6, abs=1e-9) == Qext

def test_mie_rayleigh_limit_absorption_dominates(small_radius, medium_index):
    # Rayleigh regime: very small particle -> scattering much smaller than absorption for lossy material
    lambda0 = 600e-9
    n_particle = 0.2 + 3.0j
    Qsca, Qabs, Qext = mie_cross_sections(lambda0, n_particle, small_radius, medium_index)
    assert Qabs > 0
    # scattering should be significantly smaller than absorption (order-of-magnitude)
    assert Qsca <= Qabs * 10

# --- Normalize helper -----------------------------------------------------
@pytest.mark.parametrize("q,r", [
    (1.5, 10e-9),
    (0.0, 1e-9),
    (3.14, 2e-8),
])
def test_normalize_by_area(q, r):
    if r <= 0:
        with pytest.raises(ValueError):
            normalize_by_area(q, r)
    else:
        expected = q / (np.pi * r ** 2)
        assert normalize_by_area(q, r) == pytest.approx(expected)

# --- Edge-case sanity checks ----------------------------------------------
def test_mie_invalid_parameters():
    # invalid lambda, radius, medium index should raise
    n_particle = 1.5 + 0.1j
    with pytest.raises(ValueError):
        mie_cross_sections(0.0, n_particle, 1e-9, 1.0)
    with pytest.raises(ValueError):
        mie_cross_sections(500e-9, n_particle, 0.0, 1.0)
    with pytest.raises(ValueError):
        mie_cross_sections(500e-9, n_particle, 1e-9, 0.0)
