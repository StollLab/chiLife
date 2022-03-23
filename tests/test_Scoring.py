import numpy as np
import pytest
import ProEPR

import matplotlib.pyplot as plt
r = np.linspace(1e-3, 3, 256)
eps = np.ones(len(r))
rmin = np.ones(len(r))

lj_funcs = [ProEPR.get_lj_energy, ProEPR.get_lj_scwrl, ProEPR.get_lj_rep]
lj_ans = [[10., 6.80522669, -0.1013122, -0.99520751, -0.85544423, -0.61060211, -0.41786489],
          [10., 5.79644126, -0.22677809, -1., -0.7788375, -0.10599873, 0.],
          [10., 4.0649501, 1.07172709, 0.32288702, 0.10849416, 0.03992503, 0.01586676]]


@pytest.mark.parametrize(('func', 'ans'), zip(lj_funcs, lj_ans))
def test_lj(func, ans):
    E = func(r, rmin, eps)
    np.testing.assert_almost_equal(E[60:116:8], ans)
