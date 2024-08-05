from typing import Tuple
import math as m
import numpy as np
from memoization import cached
from numba import njit, prange


@njit(cache=True)
def compute_bin(x: float, bin_edges: np.ndarray) -> int:
    """Compute determine bin for a given observation, x

    Parameters
    ----------
    x : float
        Observation.
    bin_edges : np.ndarray
        Available bins.

    Returns
    -------
    bin : int
        Index of bin to add observation to.
    """

    # Assuming uniform bins for now
    n = bin_edges.shape[0] - 1
    a_min = bin_edges[0]
    a_max = bin_edges[-1]

    bin = int(n * (x - a_min) / (a_max - a_min))
    return bin


@njit(cache=True)
def dirichlet(x: np.ndarray) -> np.ndarray:
    """Dirichlet distribution sampler

    Parameters
    ----------
    x : np.ndarray
        Parameters of dirichlet distribution to sample from.

    Returns
    -------
    dist : np.ndarray
        Random sample from dirichlet distribution with parameters x.
    """
    dist = np.zeros_like(x)
    for i, xi in enumerate(x):
        dist[i] = np.random.gamma(xi)

    dist /= dist.sum()
    return dist


@njit(cache=True)
def get_delta_r(r: np.ndarray) -> float:
    """Calculates increment of a sorted array with evenly distributed floats

    Parameters
    ----------
    r : np.ndarray
        Array of equally spaced consecutive numbers.

    Returns
    -------
    delta_r : float
        Distance between numbers in r.

    """
    delta_r = (r[-1] - r[0]) / len(r)
    return delta_r


@njit(cache=True)
def histogram(a: np.ndarray, weights: np.ndarray, r: np.ndarray) -> np.ndarray:
    """Calculate histogram for observations, ``a`` , with weights over the domain, r, as precursor for KDE of distance
    distribution.

    Parameters
    ----------
    a : np.ndarray
        Array of relevant pairwise distances between NO midpoint libraries.
    weights : np.ndarray
        Weights array corresponding to distances in ``a`` .
    r : np.ndarray
        Domain of distance distribution.

    Returns
    -------
    hist : np.ndarray
        Weighted histogram of pairwise distances.

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
def jaccard(p: np.ndarray, q: np.ndarray) -> float:
    """Calculate the jaccard index between the two provided vectors (distance distributions)

    Parameters
    ----------
    p, q : np.ndarray
        Sets to calculate the jaccard index between.

    Returns
    -------
    jac : float
        Jaccard index between ``q`` and ``p`` .
    """
    overlap = np.minimum(p, q).sum()
    union = np.maximum(p, q).sum()
    jac = overlap / union
    return jac


@njit(cache=True)
def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """Compute the Kullback–Leibler divergence (KLD) of ``p`` and ``q`` .

    Parameters
    ----------
    p, q : np.ndarray
        The distributions to calculate the KLD between.

    Returns
    -------
    kld: float
        The Kullback–Leibler divergence between ``p`` and ``q`` .

    """
    kld = np.sum(np.where(p != 0, p * np.log(p / q), 0))
    return kld


@njit(cache=True)
def normdist(delta_r: float, mu: float = 0.0, sigma: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate normal distribution for convolution with histogram of distances between two spin label ensembles

    Parameters
    ----------
    delta_r : float
        Space between points in distance domain.
    mu : float
        Mean for normal distribution.
    sigma : float
        Standard deviation of normal distribution.

    Returns
    -------
        x : np.ndarray
            Domain of the distribution.
        y : np.ndarray
            PDF values of the distribution.
    """

    # Calculate normal distribution
    x = np.arange(mu - 3.5 * sigma, mu + 3.5 * sigma + delta_r, delta_r)

    kernel_domain = (x - mu) ** 2
    coef = 1 / (np.sqrt(2 * np.pi * sigma**2))
    denom = 2 * sigma**2
    y = coef * np.exp(-(kernel_domain / denom))

    return x, y


@njit(cache=True)
def pairwise_dist(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Calculate the pairwise (euclidean) distance between the coordinate sets.

    Parameters
    ----------
    x, y : np.ndarray
        Coordinate sets to calcualte the distance between

    Returns
    -------
    distances : np.ndarray
        Pairwise distance matrix between coordinates of x and y.
    """
    M = x.shape[0]
    N = y.shape[0]
    L = x.shape[1]
    distances = np.empty((M, N), dtype=np.float64)
    for i in range(M):
        for j in range(N):
            d = 0
            for l in range(L):
                tmp = x[i, l] - y[j, l]
                tmp = tmp * tmp
                d += tmp

            distances[i, j] = m.sqrt(d)

    return distances


@njit(cache=True)
def _ic_to_cart(IC_idx_Array: np.ndarray, ICArray: np.ndarray) -> np.ndarray:
    """Convert internal coordinates into cartesian coordinates.

    Parameters
    ----------
    IC_idx_Array : np.ndarray
        Array of indexes corresponding to the atoms defining the internal coordinate.

        IC_idx_Array[i, 0] # index of the atom that atom ``i`` is bonded to.
        IC_idx_Array[i, 1] # index of the atom that atom ``i`` creates an angle with.
        IC_idx_Array[i, 2] # index of the atom that atom ``i`` creates a dihedral angle with.

        A value of -1 indicates that there is no precursor atom to define the bond, angle or dihedral and that atom
        ``i`` is one of the coordinate system defining atoms.

    ICArray :
        Internal coordinate values

        IC_idx_Array[i, 0] # bond distance of the atom that atom ``i`` and one of its precursor atoms.
        IC_idx_Array[i, 1] # bond angle between atom ``i`` two of its precursor atoms.
        IC_idx_Array[i, 2] # index of the atom that atom ``i`` creates a dihedral angle with three of it's precursor
        atoms.

    Returns
    -------
    coords : np.ndarray
        Array of cartesian coordinates corresponding to ICAtom list atoms
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

@njit(parallel=True, cache=True)
def batch_ic2cart(IC_idx_Array: np.ndarray, ICArray: np.ndarray):
    coords = np.zeros_like(ICArray)
    for i in prange(len(ICArray)):
        coords[i] = _ic_to_cart(IC_idx_Array, ICArray[i])

    return coords

@njit(cache=True)
def np_all_axis1(x: np.ndarray) -> np.ndarray:
    """Numba compatible version of np.all(x, axis=1).

    Parameters
    ----------
    x : (M, N) np.ndarray
        Boolean input array.

    Returns
    -------
    out : (M,) np.ndarray
        Boolean array indicating if all values in the second axis of ``x`` are true.

    """
    out = np.ones(x.shape[0], dtype=np.bool8)
    for i in range(x.shape[1]):
        out = np.logical_and(out, x[:, i])
    return out


# @njit(cache=True, parallel=True)
def _get_sasas(
    atom_coords: np.ndarray,
    atom_radii: np.ndarray,
    env_coords: np.ndarray,
    all_radii: np.ndarray,
    grid_points: np.ndarray
) -> np.ndarray:
    """
    Get the solvent accessible surface areas (SASA) over all frames and aotms of ``atom_coords`` in the provided
    environment.

    Parameters
    ----------
    atom_coords : (M, N, 3) np.ndarray
        M frames of the 3D cartesian coordinates of the N atoms to calculate the SASA of.
    atom_radii : (N,) np.ndarray
        vdW radii + solvent radii of the atoms in ``atom_coords``
    env_coords : (L, 3) np.ndarray
        3D cartesian coordinates of the atoms nearby the atoms for which the SASA is desired.
    all_radii : (L,) np.ndarray
        vdW radii + solvent radii of the atoms in ``env_coords``
    grid_points : (K,) np.ndarray
        Spherical points to use for SASA calculation.

    Returns
    -------
    all_sasa: (M, N) np.ndarray
        Calculated SASA of each atom in each frame.
    """

    all_sasa = np.zeros(atom_coords.shape[:2], dtype=np.float64)
    for staten in range(len(atom_coords)):
        atc = atom_coords[staten]
        alc = np.vstack((atc, env_coords))
        all_sasa[staten] = _get_sasa(atc, atom_radii, alc, all_radii, grid_points)

    return all_sasa

@cached
@njit(cache=True)
def _get_sasa(
    atom_coords: np.ndarray,
    atom_radii: np.ndarray,
    all_coords: np.ndarray,
    all_radii: np.ndarray,
    grid_points: np.ndarray
) -> np.ndarray:
    """
    Get the solvent accessible surface area of ``atom_coords`` in the  provided environment.

    Parameters
    ----------
    atom_coords : (N, 3) np.ndarray
        3D cartesian coordinates of the N atoms to calculate the SASA of.
    atom_radii : (N,) np.ndarray
        vdW radii + solvent radii of the atoms in ``atom_coords``
    all_coords : (M, 3), np.ndarray
        3D cartesian coordinates of the atoms of ``atom_coords`` and the nearby atoms.
    all_radii : (M,) np.ndarray
        vdW radii + solvent radii of the atoms in ``all_coords``
    grid_points : (K,) np.ndarray
        Spherical points to use for SASA calculation.

    Returns
    -------
    atom_sasa: (N,) np.ndarray
        Array of SASAs for each atom in  atom_coords
    """

    atom_sasa = np.zeros(len(atom_coords), dtype=np.int64)
    for i in range(len(atom_radii)):
        posi, radi = atom_coords[i], atom_radii[i]

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

            sphere = np.zeros((len(left_overs), 3), dtype=np.float64)
            for k in range(len(left_overs)):
                sphere[k] = left_overs[k]

        atom_sasa[i] = sphere.shape[0]

    atom_sasa = atom_sasa * (4.0 * np.pi / len(grid_points)) * atom_radii ** 2

    return atom_sasa

def get_sasa(
    atom_coords: np.ndarray,
    atom_radii: np.ndarray,
    environment_coords: np.ndarray = None,
    environment_radii: np.ndarray = None,
    probe_radius: float = 1.4,
    npoints: int = 1024,
    by_atom: bool = False
) -> np.ndarray:
    """

    Parameters
    ----------
    atom_coords : (M, N, 3) np.ndarray
        M frames of the 3D cartesian coordinates of the N atoms to calculate the SASA of.
    atom_radii : (N,) np.ndarray
        vdW radii + solvent radii of the atoms in ``atom_coords``
    environment_coords : (L, 3) np.ndarray
        3D cartesian coordinates of the atoms nearby the atoms for which the SASA is desired.
    environment_radii : (L,) np.ndarray
        vdW radii + solvent radii of the atoms in ``env_coords``
    probe_radius : flaot
         Radius of the solvent probe.
    npoints : int
         Number of points to use in sphere used for SASA calculation
    by_atom : bool
        Return SASA of each atom rather than the set of atoms

    Returns
    -------
    sasa : np.ndarray:
        Solvent accessible surface areas of the provided coords in the provided environment.
    """

    grid_points = fibonacci_points(npoints)

    if environment_coords is None:
        environment_coords = np.empty(shape=(0, 3), dtype=np.float64)
        environment_radii = np.empty(shape=0, dtype=np.float64)

    atom_radii = atom_radii.copy()
    environment_radii = environment_radii.copy()

    atom_radii += probe_radius
    environment_radii += probe_radius

    atom_coords = atom_coords[None, ...] if len(atom_coords.shape) == 2 else atom_coords.copy()
    all_radii = np.hstack((atom_radii, environment_radii))

    atom_sasa = _get_sasas(atom_coords, atom_radii, environment_coords, all_radii, grid_points)

    if by_atom:
        return atom_sasa
    else:
        return np.sum(atom_sasa, axis=1)


def fibonacci_points(n: int) -> np.ndarray:
    """
    Get ``n`` evenly spaced points on the unit sphere using the fibonacci method.

    Parameters
    ----------
    n : int
        Number of points to return.
        

    Returns
    -------
        coords: np.ndarray
            3D coordinates of the points on the unit sphere.

    """
    phi = (3 - m.sqrt(5)) * np.pi * np.arange(n)
    z = np.linspace(1 - 1.0/n, 1.0/n - 1, n)
    radius = np.sqrt(1 - z*z)
    coords = np.empty((n, 3))
    coords[:, 0] = radius * np.cos(phi)
    coords[:, 1] = radius * np.sin(phi)
    coords[:, 2] = z
    return coords