import numpy as np

def normalize_angles(angles):
    """

    Parameters
    ----------
    angles

    Returns
    -------

    """
    return np.arctan2(np.sin(angles), np.cos(angles))


def angle_dist(angle1, angle2):
    """

    Parameters
    ----------
    angle1
    angle2

    Returns
    -------

    """
    diff = angle1 - angle2
    return np.arctan2(np.sin(diff), np.cos(diff))

