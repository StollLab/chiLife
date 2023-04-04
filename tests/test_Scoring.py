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
    [10.0, 6.80522669, -0.1013122, -0.99520751, -0.85544423, -0.61060211, -0.41786489],
    [10.0, 5.79644126, -0.22677809, -1.0, -0.7788375, -0.10599873, 0.0],
    [10.0, 4.0649501, 1.07172709, 0.32288702, 0.10849416, 0.03992503, 0.01586676],
]


@pytest.mark.parametrize(('func', 'ans'), zip(lj_funcs, lj_ans))
def test_lj(func, ans):
    RL = chilife.RotamerEnsemble('TRP', 28, protein, energy_func=func, eval_clash=True)
    print(RL.atom_energies)
    # E = func(r, rmin, eps)
    # np.testing.assert_almost_equal(E[60:116:8], ans)


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
    np.testing.assert_almost_equal(test, ans)


def test_prep_internal_clash():
    SL = chilife.RotamerEnsemble('TRP', 28, protein)
    stuff = chilife.scoring.prep_internal_clash(SL)

def test_molar_gas_constant():
    np.testing.assert_almost_equal(chilife.scoring.GAS_CONST, 1.987204258640832e-3, decimal=10)
