import numpy as np
from numba import njit

# TODO: Write tests for all numba utils


@njit(cache=True)
def compute_bin(x, bin_edges):
    """
    Compute determine bin for a given observation, x

    :param x: float
        Observation

    :param bin_edges: numpy ndarray
        Available bins

    :return bin: int
        index of bin to add observation to
    """

    # Assuming uniform bins for now
    n = bin_edges.shape[0] - 1
    a_min = bin_edges[0]
    a_max = bin_edges[-1]

    _bin = int(n * (x - a_min) / (a_max - a_min))
    return _bin


@njit(cache=True)
def dirichlet(x):
    """
    Dirichlet distribution sampler

    :param x: numpy ndarray
        Parameters of dirichlet distribution to sample from

    :return dist: numpy ndarray
         Random sample from dirichlet distribution with parameters x
    """
    dist = np.zeros_like(x)
    for i, xi in enumerate(x):
        dist[i] = np.random.gamma(xi)

    dist /= dist.sum()
    return dist


@njit(cache=True)
def get_delta_r(r):
    """Calculates increment of a sorted array with evenly distributed floats"""
    delta_r = (r[-1] - r[0]) / len(r)
    return delta_r


@njit(cache=True)
def histogram(a, weights, r):
    """
    Calculate histogram for observations, a, with weights over the domain, r, as precursor for KDE of distance
    distribution.

    :param a: numpy ndarray
        Array of relevant pairwise distances between NO midpoint libraries

    :param weights: numpy ndarray
        Weights array corresponding to distances in a

    :param r: numpy ndarray
        Domain of distance distribution.

    :return hist: numpy ndarray
        Weighted histogram of pairwise distances
    """
    # Calculate bin size
    delta_r = get_delta_r(r)

    # Preallocate histogram array
    hist = np.zeros((len(r),), dtype=np.float64)

    # Determine edges
    bin_edges = r

    # Compute histogram
    for x, weight in zip(a, weights):
        _bin = compute_bin(x, bin_edges)

        # If density is over the upper distance limit reallocate it to 0 (impossible distance).
        if _bin >= len(hist):
            hist[0] += weight
            continue

        hist[_bin] += weight

    return hist


@njit(cache=True)
def jaccard(x, y):
    """
    Calculate the jaccard index between the two provided vectors (distance distributions)

    :param x: numpy ndarray
        1d vector of length n

    :param y: numpy ndarray
        1d vector of length n

    :return jaccard(x,y): float
        jaccard index of the two vectors x and y

    """
    overlap = np.minimum(x, y).sum()
    union = np.maximum(x, y).sum()
    return overlap / union


@njit(cache=True)
def kl_divergence(p, q):
    """
    Compute the Kullback–Leibler divergence of P and Q

    :param p: numpy ndarray
        First pdf

    :param q:
        Second pdf

    :return:
        Kullback–Leibler divergence between P and Q
    """

    return np.sum(np.where(p != 0, p * np.log(p / q), 0))


@njit(cache=True)
def norm(delta_r, mu=0., sigma=1.):
    """
    Calculate normal distribution for convolution with histogram of distances between two spin label ensembles

    :param delta_r: float
        "Space between points in distance domain"

    :param mu: float
        Mean for normal distribution

    :param sigma: float
        Standard deviation of normal distribution

    :return x, y: ndarray
        Distance domain points and corresponding normal distribution values
    """

    # Calculate normal distribution
    x = np.arange(mu - 3. * sigma, mu + 3. * sigma + delta_r, delta_r)

    kernel_domain = (x - mu) ** 2
    coef = (1 / (np.sqrt(2 * np.pi * sigma ** 2)))
    denom = 2 * sigma ** 2
    y = coef * np.exp(- (kernel_domain / denom))

    return x, y

@njit(cache=True)
def pairwise_dist(X, Y):
    M = X.shape[0]
    N = Y.shape[0]
    L = X.shape[1]
    D = np.empty((M, N), dtype=np.float64)
    for i in range(M):
        for j in range(N):
            d = 0
            for l in range(L):
                tmp = X[i, l] - Y[j, l]
                tmp = tmp * tmp
                d += tmp

            D[i, j] = np.sqrt(d)

    return D


@njit(cache=True)
def _ic_to_cart(IC_idx_Array, ICArray):
    """
    Convert internal coordinates into cartesian coordinates

    :param ICs: dict of ICAtoms
        dictionary of ICAtoms containing internal coordinates

    :return coord: numpy.ndarray
        array of cartesian coordinates corresponding to ICAtom list atoms
    """

    coords = np.zeros((len(ICArray), 3))
    M = np.zeros((3, 3))
    N = np.zeros((3, 3))

    for i in np.arange(len(ICArray)):

        if IC_idx_Array[i, 0] < 0:
            continue

        elif IC_idx_Array[i, 1] < 0:
            coords[i, 0] = ICArray[i, 0]

        elif IC_idx_Array[i, 2] < 0:
            x = ICArray[i, 0] * np.cos(np.pi - ICArray[i, 1])
            y = ICArray[i, 0] * np.sin(np.pi - ICArray[i, 1])

            coords[i, 0] = coords[IC_idx_Array[i, 0], 0] + x
            coords[i, 1] = y

        else:
            sinAngle, sinDihedral = np.sin(ICArray[i, 1:])
            cosAngle, cosDihedral = np.cos(ICArray[i, 1:])

            x = ICArray[i, 0] * cosAngle
            y = ICArray[i, 0] * cosDihedral * sinAngle
            z = ICArray[i, 0] * sinDihedral * sinAngle

            a = coords[IC_idx_Array[i, 2]]
            b = coords[IC_idx_Array[i, 1]]
            c = coords[IC_idx_Array[i, 0]]

            ab = b - a
            bc = c - b
            bc /= np.linalg.norm(bc)
            n = np.cross(ab, bc)
            n /= np.linalg.norm(n)
            ncbc = np.cross(n, bc)
            for j in np.arange(3):
                M[0, j] = bc[j]
                M[1, j] = ncbc[j]
                M[2, j] = n[j]
            N = np.array([-x, y, z])

            coords[i] = M.T @ N + coords[IC_idx_Array[i, 0]]

    return coords

# @njit(cache=True)
def get_ICAtom_indices(i, j, k, index, bonds, angles, dihedrals, offset):
    found = False
    ordered = True
    i0, j0, k0 = i, j, k
    while not found:

        # Check that the bond, angle and dihedral are all defined by the same atoms in the same order
        condition = index == angles[j][-1]
        condition = condition and np.all(bonds[i] == angles[j][-2:])
        condition = condition and np.all(angles[j] == dihedrals[k][-3:])
        condition = condition and np.all(dihedrals[k] >= offset)
        if ordered:
            condition = condition and np.all([dihedrals[k][i] < dihedrals[k][i+1] for i in range(3)])
        else:
            condition = condition and np.all(dihedrals[k][:3] < dihedrals[k][3])

        # If all checks have been pased use bond i, angle j, and dihedral k
        if condition:
            found = True

        elif k == len(dihedrals) - 1 and j == len(angles) - 1 and i == len(bonds) - 1 and ordered:
            i, j, k = i0, j0, k0
            ordered = False

        # If all angles and dihedrals containing bond i have been searched increment bond
        elif k == len(dihedrals) - 1 and j == len(angles) - 1:
            k = 0
            j = 0
            i += 1
        # if all dihedrals containing bond i and angle j have been searched increment angle
        elif k == len(dihedrals) - 1:
            k = 0
            j += 1
        # If this dihedral does not contain atoms of the angle then try the next dihedral
        else:
            k += 1

    return i, j, k
