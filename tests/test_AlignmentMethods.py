import numpy as np
from scipy.spatial.transform import Rotation

from chilife.alignment_methods import linear_alignment


def test_linear_alignment():

    X = np.random.randn(100, 3) - 0.5
    abc = np.random.randn(3) * 2 * np.pi - np.pi
    M = Rotation.from_euler('zyx', abc)

    Y = X @ M.as_matrix()
    mx, ori = linear_alignment(Y, X)
    Z = Y @ mx

    np.testing.assert_almost_equal(Z, X)


def test_multiple_linear_alignment():

    X = np.random.randn(100, 3) - 0.5

    multi_Y = []
    for i in range(10):
        abc = np.random.randn(3) * 2 * np.pi - np.pi
        M = Rotation.from_euler('zyx', abc)

        Y = X @ M.as_matrix()
        multi_Y.append(Y)

    multi_Y = np.array(multi_Y)
    multi_X = np.array([X for i in range(10)])

    mx, ori = linear_alignment(multi_Y, multi_X)

    Z = multi_Y @ mx

    np.testing.assert_almost_equal(Z, multi_X)

