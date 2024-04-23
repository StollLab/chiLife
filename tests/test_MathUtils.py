import numpy as np
from chilife.math_utils import *


def test_normalize_angle():
    angles = [-np.pi, 0, np.pi, 3 * np.pi/2, 2*np.pi]
    ans = [-np.pi, 0, np.pi, -np.pi/2, 0]
    norm_angles = normalize_angles(angles)
    np.testing.assert_almost_equal(norm_angles, ans)


def test_angle_dist():
    angles = np.linspace(0, 2*np.pi, 100)
    ref = np.zeros(100)
    ans = normalize_angles(angles)
    dist = angle_dist(angles, ref)
    np.testing.assert_almost_equal(dist, ans)

    # Rotate 90 degrees
    angles += np.pi/2
    ref += np.pi/2
    dist = angle_dist(angles, ref)
    np.testing.assert_almost_equal(dist, ans)
