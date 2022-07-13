import numpy as np
from numba import njit
import math as m


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
def norm(delta_r, mu=0.0, sigma=1.0):
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
    x = np.arange(mu - 3.0 * sigma, mu + 3.0 * sigma + delta_r, delta_r)

    kernel_domain = (x - mu) ** 2
    coef = 1 / (np.sqrt(2 * np.pi * sigma**2))
    denom = 2 * sigma**2
    y = coef * np.exp(-(kernel_domain / denom))

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

            D[i, j] = m.sqrt(d)

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


@njit(cache=True)
def np_all_axis1(x):
    """Numba compatible version of np.all(x, axis=1)."""
    out = np.ones(x.shape[0], dtype=np.bool8)
    for i in range(x.shape[1]):
        out = np.logical_and(out, x[:, i])
    return out

#
# @njit(cache=True)
def get_ICAtom_indices(k, index, bonds, angles, dihedrals, offset):
    found = False
    ordered = True
    k0 = k

    while not found:
        dk = dihedrals[k]
        condition = np.all(dk >= offset)
        condition = condition and dk[-1] == index

        if ordered:
            srted = np.array([dk[i] < dk[i + 1] for i in range(3)])
            condition = condition and np.all(srted)

        else:
            condition = condition and np.all(dk[:3] < dk[3])

        if condition:
            j = np.argwhere(np_all_axis1(angles == dk[-3:]))[0, 0]
            i = np.argwhere(np_all_axis1(bonds == dk[-2:]))[0, 0]

            found = True
        elif k + 1 == len(dihedrals):
            k = k0
            ordered = False
        else:
            k += 1

    return i, j, k


@njit(cache=True)
def _get_sasa(atom_coords, atom_radii, environment_coords, environment_radii, probe_radius=1.4, npoints=1024):
    grid_points = fibonacci_points(npoints)
    atom_radii = atom_radii.copy()
    environment_radii = environment_radii.copy()

    atom_radii += probe_radius
    environment_radii += probe_radius
    all_radii = np.hstack((atom_radii, environment_radii))
    all_coords = np.vstack((atom_coords, environment_coords))

    atom_sasa = np.zeros(len(atom_coords), dtype=np.int64)
    for i, (posi, radi) in enumerate(zip(atom_coords, atom_radii)):

        neighbor_idxs = []
        for j, (posj, radj) in enumerate(zip(all_coords, all_radii)):
            if j == i:
                continue

            diff = posi - posj
            dist_squared = diff @ diff
            if dist_squared < (radi + radj) ** 2:
                neighbor_idxs.append(j)

        sphere = grid_points * radi + posi
        for j in neighbor_idxs:
            posj, radj = all_coords[j], all_radii[j]
            left_overs = []
            for k, posk in enumerate(sphere):
                diff = posj - posk
                dist_squared = diff @ diff
                if dist_squared > radj ** 2:
                    left_overs.append(posk)

            sphere = np.empty((len(left_overs), 3), dtype=np.float64)
            for k in range(len(left_overs)):
                sphere[k] = left_overs[k]

        atom_sasa[i] = len(sphere)

    return np.sum(atom_sasa) * (4.0 * np.pi / npoints) * radi ** 2


@njit(cache=True)
def fibonacci_points(n):
    phi = (3 - m.sqrt(5)) * np.pi * np.arange(n)
    z = np.linspace(1 - 1.0/n, 1.0/n - 1, n)
    radius = np.sqrt(1 - z*z)
    coords = np.empty((n, 3))
    coords[:, 0] = radius * np.cos(phi)
    coords[:, 1] = radius * np.sin(phi)
    coords[:, 2] = z
    return coords