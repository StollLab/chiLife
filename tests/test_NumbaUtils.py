import numpy as np
from scipy.stats import norm
from scipy.special import kl_div
from scipy.spatial.distance import cdist
import pytest
import ProEPR.numba_utils as nu


def test_compute_bin():
    bin_edges = np.linspace(0, 10, 5)
    var = 5
    assert nu.compute_bin(var, bin_edges) == 2

    var = 50
    assert nu.compute_bin(var, bin_edges) == 20


def test_dirichlet():
    alphas = np.array([0.1, 0.2, 0.3, 0.4], dtype=float)

    np.random.seed(100000)
    Ans = np.random.dirichlet(alphas, 1000000).mean(axis=0)
    Test = np.array([nu.dirichlet(alphas) for i in range(1000000)]).mean(axis=0)
    np.testing.assert_almost_equal(Test, Ans, decimal=3)


def test_get_delta_r():
    x = np.arange(0, 100, 10, dtype=float)
    assert nu.get_delta_r(x) == 9.0


def test_histogram():
    r = np.linspace(0, 100, 2 ** 8)
    data = np.random.normal(45, 5, 100)
    w = np.random.uniform(0, 1, 100)
    w /= w.sum()

    np_hist, _ = np.histogram(data, r, weights=w)
    nu_hist = nu.histogram(data, w, r)

    np.testing.assert_almost_equal(np_hist, nu_hist[:-1])


def test_jaccard():
    y1 = np.zeros(10)
    y1[3:7] = 1

    y2 = np.zeros(10)
    y2[4:8] = 1

    assert nu.jaccard(y1, y2) == 3/5


def test_kl_divergence():
    r = np.linspace(0, 100, 2 ** 8)
    p1 = norm(45, 5).pdf(r)
    p2 = norm(55, 6).pdf(r)

    assert nu.kl_divergence(p1, p2) == pytest.approx(kl_div(p1, p2).sum())


def test_norm():
    mu, sigma = 50, 10
    delta_r = 1e-2
    x = np.arange(mu - 3 * sigma, mu + 3 * sigma + delta_r, delta_r)

    y1 = norm(mu, sigma).pdf(x)
    y2 = nu.norm(delta_r, 50, 10)[1]

    np.testing.assert_almost_equal(y1, y2)


def test_pairwise_dist():
    from time import time
    A = np.random.rand(1000, 3)
    B = np.random.rand(1000, 3)
    print(A.shape)
    nu.pairwise_dist(A, B)

    t1 = time()
    D_cdist = cdist(A, B)
    t2 = time()
    D_numba = nu.pairwise_dist(A, B)
    t3 = time()
    print('cdist time: ', t2 - t1)
    print('numba time: ', t3 - t2)

    np.testing.assert_almost_equal(D_cdist, D_numba)