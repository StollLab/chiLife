import numpy as np
from scipy.spatial.distance import cdist
from numba import njit

import chiLife


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
            lj_energy[i] = np.minimum(eps[i] * (lj**2 - 2*lj), cap * eps[i])

        elif rmin_lower[i] <= r[i] < rmin[i]:
            lj_energy[i] = -eps[i]
        elif r[i] < rmax:
            lj = rmin[i] / r[i]
            lj = lj * lj * lj
            lj = lj * lj
            lj_energy[i] = eps[i] * (lj**2 - 2*lj)

    return lj_energy

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
        elif rat < 10/9:
            lj_energy[i] = eps[i] * (10 - 9 * rat) ** (57.273/(9 * eps[i])) - eps[i]
        elif rat < 4/3:
            lj_energy[i] = ((eps[i] / 4) * (9 * rat - 10) ** 2 - eps[i])
        else:
            lj_energy[i] = 0

    return lj_energy


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
        lj_energy[i] = np.minimum(eps[i] * lj ** 2, cap * eps[i])

    return lj_energy

# @njit(cache=True, parallel=True)
# def get_lj_LR90(r, rmin, eps, forgive=0.9, cap=10):

def evaluate_clashes(ori, label_library,  label_lj_rmin2, label_lj_eps,
                     environment, environment_tree, ignore_idx=None, temp=298., energy_func=get_lj_rep,
                     clash_radius=14, **kwargs):
    """
    Calculate clash energies for each rotamer and return Boltzmann weighted probabilities of rotamers based off of clash
    energies.

    :param ori: numpy ndarray
        3D coordinate of library centroid. Used to find neighboring atoms for clash evaluation.

    :param label_library: numpy ndarray
        Library or rotamer atoms superimposed onto site of interest

    :param label_lj_rmin2: numpy ndarray
        Lennard jones radius parameters for atoms of spin label

    :param label_lj_eps: numpy ndarray
        Lennard jones energy parameters for atoms of spin label

    :param environment: numpy ndarray
        Atoms of protein excluding original atoms of site of interest.

    :param environment_tree: cKDTree
        KDTree of environment

    :param ignore_idx: list
        List of indices corresponding to atoms to ignore. Usually atoms of native residues.

    :param temp: float
        Reference temperature for calculation of Boltzmann probabilities.

    :param energy_func: function
        Energy function used to calculate rotamer probabilities in the given environment

    :param forgive: float
        Re-weighting factor for the energy function to help account for artifacts introduced by static rotamer libraries

    :param clash_radius: float
        Max distance from library centroid to consider when evaluating clashes

    :return rotamer_energies: numpy ndarray
        Boltzmann weighted probability distribution of rotamers
    """
    # Get all potential clashes

    protein_clash_idx = environment_tree.query_ball_point(ori, clash_radius)
    if ignore_idx is not None:
        protein_clash_idx =[idx for idx in protein_clash_idx if idx not in ignore_idx]

    # Calculate rmin and epsilon for all atoms in protein that may clash
    rmin_ij, eps_ij = get_lj_params(label_lj_rmin2, label_lj_eps,
                                    environment.atoms.types[protein_clash_idx],
                                    len(label_library), forgive=kwargs.get('forgive', 1.))

    # Calculate distances
    dist = cdist(label_library.reshape(-1, 3), environment_tree.data[protein_clash_idx]).ravel()
    shape = (len(label_library), len(label_library[0]) * len(protein_clash_idx))
    atom_energies = energy_func(dist, rmin_ij, eps_ij).reshape(shape)
    rotamer_probabilities = np.exp(-atom_energies.sum(axis=1) / (GAS_CONST * temp))

    return rotamer_probabilities


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

def get_lj_params(label_lj_rmin2, label_lj_eps, environment_atypes, lib_len, forgive):
    """
    Create tiled matrices of lj parameters for calculation of lj potential
    :param label_lj_rmin2:
    :param label_lj_eps:
    :param environment_atypes:
    :param lib_len:
    :return:
    """
    protein_lj_radii = chiLife.get_lj_rmin(environment_atypes)
    protein_lj_eps = chiLife.get_lj_eps(environment_atypes)
    join_rmin = chiLife.get_lj_rmin('join_protocol')[()]
    join_eps = chiLife.get_lj_eps('join_protocol')[()]

    rmin_ij = np.tile(join_rmin(label_lj_rmin2 * forgive, protein_lj_radii * forgive).reshape(-1), lib_len)
    eps_ij = np.tile(join_eps(label_lj_eps, protein_lj_eps).reshape((-1)), lib_len)

    return rmin_ij, eps_ij

def join_geom(x, y):
    return np.sqrt(np.outer(x, y))


def join_arith(x, y):
    return np.add.outer(x, y)

rmin_charmm = {"C": 2.0000,
               "H": 0.2245,
               "N": 1.8500,
               "O": 1.7000,
               "S": 2.0000,
               "SE": 2.0000,    # Default Selenium to Sulfur
               "Br": 1.9800,
               "join_protocol": join_arith}    # Bromobenzene from Gutiérrez et al. 2016

eps_charmm = {"C": -0.110,
              "H": -0.046,
              "N": -0.200,
              "O": -0.120,
              "S": -0.450,
             "SE": -0.450,  # Default Selenium to Sulfur
             "Br": -0.320,
              "join_protocol": join_geom}  # Bromobenzene from Gutiérrez et al. 2016

rmin_uff = {"C": 3.851,
           "H":  2.886,
           "N":  3.660,
           "O":  3.500,
           "S":  4.035,
          "SE":  4.035,  # Default Selenium to Sulfur
          "Br":  4.189,
          "join_protocol": join_geom}

eps_uff = {"C": -0.105,
           "H": -0.044,
           "N": -0.069,
           "O": -0.060,
           "S": -0.274,
          "SE": -0.274,  # Default Selenium to Sulfur
          "Br": -0.251,
          "join_protocol": join_geom}

lj_params ={'uff': [rmin_uff, eps_uff],
             'charmm': [rmin_charmm, eps_charmm]}

def set_lj_params(forcefield):
    chiLife.using_lj_param = forcefield
    rmin_func, eps_func = lj_params[forcefield]
    chiLife.get_lj_rmin = np.vectorize(rmin_func.__getitem__)
    chiLife.get_lj_eps = np.vectorize(eps_func.__getitem__)


set_lj_params('charmm')

GAS_CONST = 1.98720425864083e-3  # Kcal/K-1 mol-1  # 8.314472  # J/(K*mol)
BOLTZ_CONST = 1.3806503e-23      # J/K
KCAL2J = 4.184e3