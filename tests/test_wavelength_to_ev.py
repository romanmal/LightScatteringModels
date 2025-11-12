import numpy as np
from ld_model import wavelength_nm_to_ev

def test_wavelength_500nm_to_ev():
    # photon energy for 500 nm: E = hc / lambda
    # expected value (approx): 2.479684 eV
    ev = wavelength_nm_to_ev(500.0)
    assert np.isclose(ev, 2.479684, rtol=1e-6, atol=1e-8)

def test_wavelength_array_roundtrip():
    arr_nm = np.array([400.0, 500.0, 800.0])
    evs = wavelength_nm_to_ev(arr_nm)
    # monotonic decreasing energy for increasing wavelength
    assert evs[0] > evs[1] > evs[2]
    # basic sanity: values positive
    assert np.all(evs > 0)
