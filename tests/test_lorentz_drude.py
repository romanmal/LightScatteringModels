import numpy as np
from ld_model import lorentz_drude_epsilon, refractive_index_from_eps

def test_epsilon_at_2ev_is_finite():
    # test at a representative energy (2.0 eV)
    w = np.array([2.0])
    eps = lorentz_drude_epsilon(w)
    assert np.isfinite(eps).all()
    # check shape preserved
    assert eps.shape == (1,)

def test_n_k_relationship_and_signs():
    # test for a few energies that n,k are real and non-negative for k
    w = np.array([1.0, 2.0, 3.0, 5.0])
    eps = lorentz_drude_epsilon(w)
    n, k = refractive_index_from_eps(eps)
    assert n.shape == k.shape == (4,)
    # n can be positive or small, k must be non-negative magnitude
    assert np.all(np.isfinite(n))
    assert np.all(np.isfinite(k))
    assert np.all(k >= 0)
