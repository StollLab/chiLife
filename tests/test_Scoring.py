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
lj_ans = [np.array([-1.29407456, -1.34917747, -0.92307553,  0.09701479, -0.81320874,
                    -0.82519134, -0.81681929, -0.47411124, -0.64444505, -0.44549313]),
          np.array([-0.05528211, -0.14662196,  0.        , -0.20341   ,  0.        ,
                     0.        ,  0.        ,  0.        ,  0.        ,  0.        ]),
          np.array([0.66692039, 0.05328766, 0.01843082, 0.01921865, 0.00483313,
                    0.68429473, 0.00290177, 0.00411807, 1.25053399, 0.03108683])]


@pytest.mark.parametrize(('func', 'ans'), zip(lj_funcs, lj_ans))
def test_lj(func, ans):
    f = chilife.ljEnergyFunc(func)
    RL = chilife.RotamerEnsemble('TRP', 28, protein, energy_func=f, eval_clash=True)
    np.testing.assert_almost_equal(RL.atom_energies[5], ans, decimal=6)


@pytest.mark.parametrize('func',  lj_funcs)
def test_efunc(func):
    RL = chilife.RotamerEnsemble('TRP', 28, protein, eval_clash=False)
    f = chilife.ljEnergyFunc(func)
    test = f(RL)
    ans = np.load(f'test_data/{func.__name__}.npy')
    np.testing.assert_almost_equal(test, ans, decimal=5)


@pytest.mark.parametrize('func',  lj_funcs)
def test_efunc_dlabel(func):
    dSL = chilife.dSpinLabel('DHC', (28, 32), protein, eval_clash=False, rotlib='test_data/DHC')
    f = chilife.ljEnergyFunc(func)
    test = f(dSL)
    ans = np.load(f'test_data/d{func.__name__}.npy')
    np.testing.assert_almost_equal(test, ans, decimal=3)


def test_molar_gas_constant():
    np.testing.assert_almost_equal(chilife.scoring.GAS_CONST, 1.987204258640832e-3, decimal=10)


def test_get_lj_case_sensitivity():
    ff = chilife.ljEnergyFunc()
    x = ff.get_lj_rmin(['CA', 'Ca', 'ca'])
    assert np.all(x == 1.367)