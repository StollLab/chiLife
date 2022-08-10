from functools import wraps
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
from numba import njit
import chiLife


def clash_only(func):
    @wraps(func)
    def energy_func(protein, rotlib=None, **kwargs):

        rmax = kwargs.get("rmax", 10)
        forgive = kwargs.get("forgive", 1)

        if rotlib is None:
            bonds = {(a, b) for a, b in protein.atoms.bonds.indices}
            tree = cKDTree(protein.atoms.positions)
            pairs = tree.query_pairs(rmax)
            pairs = pairs - bonds
            pairs = np.array(list(pairs))

            r = np.linalg.norm(
                protein.atoms.positions[pairs[:, 0]]
                - protein.atoms.positions[pairs[:, 1]],
                axis=1,
            )

            lj_radii_1 = chiLife.get_lj_rmin(protein.atoms.types[pairs[:, 0]])
            lj_radii_2 = chiLife.get_lj_rmin(protein.atoms.types[pairs[:, 1]])

            lj_eps_1 = chiLife.get_lj_eps(protein.atoms.types[pairs[:, 0]])
            lj_eps_2 = chiLife.get_lj_eps(protein.atoms.types[pairs[:, 1]])

            join_rmin = chiLife.get_lj_rmin("join_protocol")[()]
            join_eps = chiLife.get_lj_eps("join_protocol")[()]

            rmin_ij = join_rmin(lj_radii_1 * forgive, lj_radii_2 * forgive, flat=True)
            eps_ij = join_eps(lj_eps_1, lj_eps_2, flat=True)
            E = func(r, rmin_ij, eps_ij, **kwargs)
            E = E.sum()

        else:
            if hasattr(rotlib, "protein"):
                if rotlib.protein is not protein:
                    raise NotImplementedError(
                        "The protein passed must be the same as the protein associated with the "
                        "rotamer library passed"
                    )
            else:
                # Attach the protein to the rotlib in case it is passed again but don't evaluate clashes
                rotlib.protein = protein
                rotlib.eval_clash = False
                rotlib.protein_setup()

            r, rmin, eps, shape = prep_external_clash(rotlib)
            E = func(r, rmin, eps, **kwargs).reshape(shape)

            if kwargs.get("internal", False):
                r, rmin, eps, shape = prep_internal_clash(rotlib)
                E += func(r, rmin, eps, **kwargs).reshape(rotlib._coords.shape[:2])

            E = E.sum(axis=1)

        return E

    return energy_func


@clash_only
@njit(cache=True, parallel=True)
def get_lj_energy(r, rmin, eps, forgive=1, cap=10, rmax=10):
    """
    Return a vector with the energy values for the flat bottom lenard-jones potential from a set of atom pairs with
    distance r, rmin values of rmin and epsilon values of eps. In the absence of solvent this energy function will
    overestimate attractive forces.
    :param r: numpy ndarray
        vector of distances

    :param rmin: numpy ndarray
        vector of rmin values corresponding to atoms pairs of r

    :param eps: numpy ndarray
        vector of epsilon values corresponding to atom pairs of r

    :param forgive: numpy ndarray
        fraction of

    :return lj_energy: numpy ndarray
        vector of energies calculated using the modified lj potential function.
    """
    lj_energy = np.zeros_like(r)

    eps = eps.copy()
    rmin_lower = forgive * rmin

    # Piecewise function for flat lj potential near rmin
    for i in range(len(r)):
        if r[i] < rmin_lower[i]:
            lj = rmin_lower[i] / r[i]
            lj = lj * lj * lj
            lj = lj * lj
            lj_energy[i] = np.minimum(eps[i] * (lj**2 - 2 * lj), cap * eps[i])

        elif rmin_lower[i] <= r[i] < rmin[i]:
            lj_energy[i] = -eps[i]
        elif r[i] < rmax:
            lj = rmin[i] / r[i]
            lj = lj * lj * lj
            lj = lj * lj
            lj_energy[i] = eps[i] * (lj**2 - 2 * lj)

    return lj_energy


@clash_only
@njit(cache=True, parallel=True)
def get_lj_scwrl(r, rmin, eps, forgive=1):
    """
    Return a vector with the energy values for the flat bottom lenard-jones potential from a set of atom pairs with
    distance r, rmin values of rmin and epsilon values of eps. In the absence of solvent this energy function will
    overestimate attractive forces.

    :param r: numpy ndarray
        vector of distances

    :param rmin: numpy ndarray
        vector of rmin values corresponding to atoms pairs of r

    :param eps: numpy ndarray
        vector of epsilon values corresponding to atom pairs of r

    :param forgive: numpy ndarray
        fraction of

    :return lj_energy: numpy ndarray
        vector of energies calculated using the modified lj potential function.
    """
    lj_energy = np.empty_like(r)

    eps = eps.copy()
    rmin_lower = rmin * forgive

    # Piecewise function for flat lj potential near rmin
    for i in range(len(r)):
        rat = r[i] / (rmin_lower[i] / 1.12246204831)
        if rat < 0.8254:
            lj_energy[i] = 10 * eps[i]
        elif rat <= 1:
            lj_energy[i] = 57.273 * (1 - rat) * eps[i]
        elif rat < 10 / 9:
            lj_energy[i] = eps[i] * (10 - 9 * rat) ** (57.273 / (9 * eps[i])) - eps[i]
        elif rat < 4 / 3:
            lj_energy[i] = (eps[i] / 4) * (9 * rat - 10) ** 2 - eps[i]
        else:
            lj_energy[i] = 0

    return lj_energy


@clash_only
@njit(cache=True, parallel=True)
def get_lj_rep(r, rmin, eps, forgive=0.9, cap=10):
    """
    Calculate only repulsive terms of lennard jones potential.

    :param r: numpy ndarray
        vector of distances

    :param rmin: numpy ndarray
        vector of rmin values corresponding to atoms pairs of r

    :param eps: numpy ndarray
        vector of epsilon values corresponding to atom pairs of r

    :param forgive: numpy ndarray
        fraction of

    :return lj_energy: numpy ndarray
        vector of energies calculated using the modified lj potential function.
    """
    lj_energy = np.empty_like(r)

    # Unit convert
    eps = eps.copy()

    rmin_lower = forgive * rmin

    # Piecewise function for flat lj potential near rmin
    for i in range(len(r)):
        lj = rmin_lower[i] / r[i]
        lj = lj * lj * lj
        lj = lj * lj
        lj_energy[i] = np.minimum(eps[i] * lj**2, cap * eps[i])

    return lj_energy


# @njit(cache=True, parallel=True)
# def get_lj_LR90(r, rmin, eps, forgive=0.9, cap=10):


def prep_external_clash(rotlib):

    # Calculate rmin and epsilon for all atoms in protein that may clash
    rmin_ij, eps_ij = get_lj_params(rotlib)

    # Calculate distances
    dist = cdist(
        rotlib.coords[:, rotlib.side_chain_idx].reshape(-1, 3),
        rotlib.protein_tree.data[rotlib.protein_clash_idx],
    ).ravel()
    shape = (
        len(rotlib.coords),
        len(rotlib.side_chain_idx) * len(rotlib.protein_clash_idx),
    )

    return dist, rmin_ij, eps_ij, shape


def prep_internal_clash(rotlib):

    a, b = [list(x) for x in zip(*rotlib.non_bonded)]
    a_eps = chiLife.get_lj_eps(rotlib.atom_types[a])
    a_radii = chiLife.get_lj_rmin(rotlib.atom_types[a])
    b_eps = chiLife.get_lj_eps(rotlib.atom_types[b])
    b_radii = chiLife.get_lj_rmin(rotlib.atom_types[b])

    join_rmin = chiLife.get_lj_rmin("join_protocol")[()]
    join_eps = chiLife.get_lj_eps("join_protocol")[()]

    rmin_ij = join_rmin(a_radii * rotlib.forgive, b_radii * rotlib.forgive, flat=True)
    eps_ij = join_eps(a_eps, b_eps, flat=True)

    dist = np.linalg.norm(rotlib._coords[:, a] - rotlib._coords[:, b], axis=2)
    shape = (len(rotlib._coords), len(a_radii))

    return dist, rmin_ij, eps_ij, shape


def reweight_rotamers(probabilities, weights, return_partition=False):
    """
    Adjust rotamer population frequencies based on energy calculated from clashes.

    :param probabilities: numpy ndarray
        Array of Boltzmann weighted probabilities from lennard jones potential

    :param weights: numpy ndarray
        Current weights of rotamers

    :return new_weights: numpy ndarray
        New weights adjusted by rotamer interaction energies.
    """
    partition = sum(probabilities * weights) / sum(weights)
    new_weights = (probabilities / sum(probabilities)) * weights
    new_weights /= new_weights.sum()

    if return_partition:
        return new_weights, partition

    return new_weights


def get_lj_params(rotlib):
    """
    rotlib
    """
    environment_atypes = rotlib.protein.atoms.types[rotlib.protein_clash_idx]
    protein_lj_radii = chiLife.get_lj_rmin(environment_atypes)
    protein_lj_eps = chiLife.get_lj_eps(environment_atypes)
    join_rmin = chiLife.get_lj_rmin("join_protocol")[()]
    join_eps = chiLife.get_lj_eps("join_protocol")[()]

    rmin_ij = np.tile(
        join_rmin(
            rotlib.rmin2 * rotlib.forgive, protein_lj_radii * rotlib.forgive
        ).reshape(-1),
        len(rotlib.coords),
    )
    eps_ij = np.tile(
        join_eps(rotlib.eps, protein_lj_eps).reshape((-1)), len(rotlib.coords)
    )

    return rmin_ij, eps_ij


def join_geom(x, y, flat=False):
    if flat:
        return np.sqrt(x * y)
    else:
        return np.sqrt(np.outer(x, y))


def join_arith(x, y, flat=False):
    if flat:
        return x + y
    else:
        return np.add.outer(x, y)


rmin_charmm = {
    "C": 2.0000,
    "H": 0.7000,
    "N": 1.8500,
    "O": 1.7000,
    "S": 2.0000,
    "SE": 2.0000,  # Default Selenium to Sulfur
    "Br": 1.9800,
    "Cu": 1.8000,
    "join_protocol": join_arith,
}  # Bromobenzene from Gutiérrez et al. 2016

eps_charmm = {
    "C": -0.110,
    "H": -0.022,
    "N": -0.200,
    "O": -0.120,
    "S": -0.450,
    "SE": -0.450,  # Default Selenium to Sulfur
    "Br": -0.320,
    "Cu": -0.170,
    "join_protocol": join_geom,
}  # Bromobenzene from Gutiérrez et al. 2016

rmin_uff = {
    "C": 3.851,
    "H": 2.886,
    "N": 3.660,
    "O": 3.500,
    "S": 4.035,
    "SE": 4.035,  # Default Selenium to Sulfur
    "Br": 4.189,
    "join_protocol": join_geom,
}

eps_uff = {
    "C": -0.105,
    "H": -0.044,
    "N": -0.069,
    "O": -0.060,
    "S": -0.274,
    "SE": -0.274,  # Default Selenium to Sulfur
    "Br": -0.251,
    "join_protocol": join_geom,
}

lj_params = {"uff": [rmin_uff, eps_uff], "charmm": [rmin_charmm, eps_charmm]}


def set_lj_params(forcefield):
    chiLife.using_lj_param = forcefield
    rmin_func, eps_func = lj_params[forcefield]
    chiLife.get_lj_rmin = np.vectorize(rmin_func.__getitem__)
    chiLife.get_lj_eps = np.vectorize(eps_func.__getitem__)


set_lj_params("charmm")

GAS_CONST = 1.98720425864083e-3  # Kcal/K-1 mol-1  # 8.314472  # J/(K*mol)
BOLTZ_CONST = 1.3806503e-23  # J/K
KCAL2J = 4.184e3
