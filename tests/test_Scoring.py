import numpy as np
import pytest
import MDAnalysis as mda
import chilife

import matplotlib.pyplot as plt

r = np.linspace(1e-3, 3, 256)
eps = np.ones(len(r))
rmin = np.ones(len(r))
protein = mda.Universe("test_data/1ubq.pdb", in_memory=True)
lj_funcs = [chilife.get_lj_energy, chilife.get_lj_scwrl, chilife.get_lj_rep]
lj_ans = [
    [-1.2940752, -1.34917751, -0.92307548, 0.09701376, -0.81320862, -0.82519124, -0.8168207, -0.47411124, -0.64444488, -0.44549325],
    [-0.05528246, -0.14662195, 0., -0.20340994, 0., 0., 0., 0., 0., 0.],
    [0.66691997, 0.05328764, 0.01843083, 0.01921863, 0.00483313, 0.68429553, 0.00290177, 0.00411807, 1.25053418, 0.03108684],
]


@pytest.mark.parametrize(('func', 'ans'), zip(lj_funcs, lj_ans))
def test_lj(func, ans):
    RL = chilife.RotamerEnsemble('TRP', 28, protein, energy_func=func, eval_clash=True)
    np.testing.assert_almost_equal(RL.atom_energies[5], ans)


@pytest.mark.parametrize('func',  lj_funcs)
def test_efunc(func):
    RL = chilife.RotamerEnsemble('TRP', 28, protein)
    test = func(RL)
    ans = np.load(f'test_data/{func.__name__}.npy')
    np.testing.assert_almost_equal(test, ans)


@pytest.mark.parametrize('func',  lj_funcs)
def test_efunc_dlabel(func):
    dSL = chilife.dSpinLabel('DHC', (28, 32), protein, eval_clash=False)
    test = func(dSL)
    ans = np.load(f'test_data/d{func.__name__}.npy')
    np.testing.assert_almost_equal(test, ans, decimal=4)


def test_prep_internal_clash():
    SL = chilife.RotamerEnsemble('TRP', 28, protein)
    r, rmin, eps, shape = chilife.scoring.prep_internal_clash(SL)
    with np.load('test_data/internal_clash_prep.npz') as f:
        r_ans, rmin_ans, eps_ans = f['r'], f['rmin'], f['eps']

    assert shape == (36, 56)
    np.testing.assert_almost_equal(r, r_ans)
    np.testing.assert_almost_equal(rmin, rmin_ans)
    np.testing.assert_almost_equal(eps, eps_ans)


def test_molar_gas_constant():
    np.testing.assert_almost_equal(chilife.scoring.GAS_CONST, 1.987204258640832e-3, decimal=10)


def test_get_lj_case_sensitivity():
    x = chilife.get_lj_rmin(['CA', 'Ca', 'ca'])
    assert np.all(x == 1.367)