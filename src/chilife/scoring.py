from typing import Union
from abc import ABC, abstractmethod
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
from numba import njit
import MDAnalysis as mda

import chilife.RotamerEnsemble as re
import chilife.dRotamerEnsemble as dre
from .MolSys import MolSys


@njit(cache=True)
def get_lj_energy(r, rmin, eps, forgive=1, cap=10, rmax=10):
    """Return a vector with the energy values for the flat bottom lenard-jones potential from a set of atom pairs with
    distance `r`, rmin values of `rmin` and epsilon values of `eps`.

    Parameters
    ----------
    r : numpy.ndarray
        Vector of inter-atomic distances between non-bonded atoms of a system.
    rmin : numpy.ndarray
        Vector of rmin parameters, in angstoms,  corresponding to atoms pairs of r.
    eps : numpy.ndarray
        Vector of epsilon parameters corresponding to atom pairs of r
    forgive : numpy.ndarray
        The `forgive` factor is a softening term to mitigate rigid body artifacts. It is set to a value between 0 1nd 1
        and modifies the rmin parameter of all atom pairs in `r` to be the fraction, `forgive` of the thair original
        value. This allows atoms to be closer than otherwise allowed to prevent explosion of the jennard-jones repulsion
        in situations that would otherwise be resolved with minor atomic displacements.
    cap : flaot
        Maximum allowed energy factor. Sets a cap on the maximum energy contribution of one atom pair interaction as
        another mechanism for softening. `cap` is sets the actual max value as a multiple of the `eps` parameter. i.e.
        the maximum allowed energy from a single atom pair interaction is the `eps` parameter multiplied by `cap`.
    rmax : float
        Maximum distance to consider for potential calculation. Any atom pairs with r > rmax will be set to 0.

    Returns
    -------
    lj_energy: numpy.ndarray
        Vector of atom pair energies calculated using the modified lj potential function.

    """
    r = np.atleast_2d(r)
    lj_energy = np.zeros_like(r)
    rmin_lower = forgive * rmin

    # Piecewise function for flat lj potential near rmin
    for j, rotamer in enumerate(r):
        for i in range(len(rotamer)):
            if rotamer[i] < rmin_lower[i]:
                lj = rmin_lower[i] / rotamer[i]
                lj = lj * lj * lj
                lj = lj * lj
                lj_energy[j, i] = np.minimum(eps[i] * (lj**2 - 2 * lj), cap * eps[i])

            elif rmin_lower[i] <= rotamer[i] < rmin[i]:
                lj_energy[j, i] = -eps[i]
            elif rotamer[i] < rmax:
                lj = rmin[i] / rotamer[i]
                lj = lj * lj * lj
                lj = lj * lj
                lj_energy[j, i] = eps[i] * (lj**2 - 2 * lj)

    return lj_energy


@njit(cache=True)
def get_lj_scwrl(r, rmin, eps, forgive=1):
    """Calculate a scwrl-like lenard-jones potential from a set of atom pairs with distance `r`, rmin values of `rmin`
    and epsilon values of `eps`.

    Parameters
    ----------
    r : numpy.ndarray
        Vector of inter-atomic distances between non-bonded atoms of a system.
    rmin : numpy.ndarray
        Vector of rmin parameters, in angstoms,  corresponding to atoms pairs of r.
    eps : numpy.ndarray
        Vector of epsilon parameters corresponding to atom pairs of r
    forgive : numpy.ndarray
        The `forgive` factor is a softening term to mitigate rigid body artifacts. It is set to a value between 0 1nd 1
        and modifies the rmin parameter of all atom pairs in `r` to be the fraction, `forgive` of the thair original
        value. This allows atoms to be closer than otherwise allowed to prevent explosion of the jennard-jones repulsion
        in situations that would otherwise be resolved with minor atomic displacements.

    Returns
    -------
    lj_energy: numpy.ndarray
        Vector of atom pair energies calculated using the modified lj potential function.
    """
    r = np.atleast_2d(r)
    lj_energy = np.empty_like(r)
    rmin_lower = rmin * forgive
    for j, rotamer in enumerate(r):
        # Piecewise function for flat lj potential near rmin
        for i in range(len(rotamer)):
            rat = rotamer[i] / (rmin_lower[i] / 1.12246204831)
            if rat < 0.8254:
                lj_energy[j, i] = 10 * eps[i]
            elif rat <= 1:
                lj_energy[j, i] = 57.273 * (1 - rat) * eps[i]
            elif rat < 10 / 9:
                lj_energy[j, i] = eps[i] * (10 - 9 * rat) ** (57.273 / (9 * eps[i])) - eps[i]
            elif rat < 4 / 3:
                lj_energy[j, i] = (eps[i] / 4) * (9 * rat - 10) ** 2 - eps[i]
            else:
                lj_energy[j, i] = 0

    return lj_energy


@njit(cache=True)
def get_lj_rep(r, rmin, eps, forgive=0.9, cap=10):
    """Calculate only the repulsive terms of the lenard-jones potential from a set of atom pairs with distance `r`,
    rmin values of `rmin` and epsilon values of `eps`.

    Parameters
    ----------
    r : numpy.ndarray
        Vector of inter-atomic distances between non-bonded atoms of a system.
    rmin : numpy.ndarray
        Vector of rmin parameters, in angstoms,  corresponding to atoms pairs of r.
    eps : numpy.ndarray
        Vector of epsilon parameters corresponding to atom pairs of r
    forgive : numpy.ndarray
        The `forgive` factor is a softening term to mitigate rigid body artifacts. It is set to a value between 0 1nd 1
        and modifies the rmin parameter of all atom pairs in `r` to be the fraction, `forgive` of the thair original
        value. This allows atoms to be closer than otherwise allowed to prevent explosion of the jennard-jones repulsion
        in situations that would otherwise be resolved with minor atomic displacements.
    cap : flaot
        Maximum allowed energy factor. Sets a cap on the maximum energy contribution of one atom pair interaction as
        another mechanism for softening. `cap` is sets the actual max value as a multiple of the `eps` parameter. i.e.
        the maximum allowed energy from a single atom pair interaction is the `eps` parameter multiplied by `cap`.

    Returns
    -------
    lj_energy: numpy.ndarray
        Vector of atom pair energies calculated using the modified lj potential function.
    """
    r = np.atleast_2d(r)
    lj_energy = np.empty_like(r)
    rmin_lower = forgive * rmin

    # Piecewise function for flat lj potential near rmin
    for i, rotamer in enumerate(r):
        for j in range(len(rotamer)):
            lj = rmin_lower[j] / rotamer[j]
            lj = lj * lj * lj
            lj = lj * lj
            lj_energy[i, j] = np.minimum(eps[j] * lj**2, cap * eps[j])

    return lj_energy

@njit(cache=True)
def get_lj_attr(r, rmin, eps, forgive=0.9, floor=-2):
    """Calculate only the attractive terms of the lenard-jones potential from a set of atom pairs with distance `r`,
      rmin values of `rmin` and epsilon values of `eps`.

      Parameters
      ----------
      r : numpy.ndarray
          Vector of inter-atomic distances between non-bonded atoms of a system.
      rmin : numpy.ndarray
          Vector of rmin parameters, in angstoms,  corresponding to atoms pairs of r.
      eps : numpy.ndarray
          Vector of epsilon parameters corresponding to atom pairs of r
      forgive : numpy.ndarray
          The `forgive` factor is a softening term to mitigate rigid body artifacts. It is set to a value between 0 1nd 1
          and modifies the rmin parameter of all atom pairs in `r` to be the fraction, `forgive` of the thair original
          value. This allows atoms to be closer than otherwise allowed to prevent explosion of the jennard-jones repulsion
          in situations that would otherwise be resolved with minor atomic displacements.
      cap : flaot
          Maximum allowed energy factor. Sets a cap on the maximum energy contribution of one atom pair interaction as
          another mechanism for softening. `cap` is sets the actual max value as a multiple of the `eps` parameter. i.e.
          the maximum allowed energy from a single atom pair interaction is the `eps` parameter multiplied by `cap`.

      Returns
      -------
      lj_energy: numpy.ndarray
          Vector of atom pair energies calculated using the modified lj potential function.
      """
    r = np.atleast_2d([r])
    lj_energy = np.empty_like(r)
    rmin_lower = forgive * rmin
    for j, rotamer in enumerate(r):
        # Piecewise function for flat lj potential near rmin
        for i in range(len(rotamer)):
            lj = rmin_lower[i] / rotamer[i]
            lj = lj * lj * lj
            lj = lj * lj
            lj_energy[j, i] = np.maximum(-2 * eps[i] * lj, eps[i] * floor)

    return lj_energy


def reweight_rotamers(energies, temp, weights):
    """Adjust rotamer population weights based on external energies (from clash evaluations).

    Parameters
    ----------
    energies : numpy.ndarray
        Array of external energies, in kcal/mol
    temp : scalar
        Temperature, in kelvin
    weights : numpy.ndarray
        Current weights of rotamers
    return_partition : bool
        If True, return the value of the partition function an additional argument; see below

    Returns
    -------
    new_weights : numpy.ndarray
        Adjusted weights
    partition : float (optional)
        The partition function relative to the free label. A small partition function suggests the interactions with
        neighboring atoms are unfavorable while a large partition function suggests the opposite.
    """

    probabilities = np.exp(-energies / (GAS_CONST * temp))
    p = probabilities * weights
    p_sum = np.sum(p)
    new_weights = p/p_sum
    partition = p_sum / weights.sum()

    return new_weights, partition


def join_geom(a, b, flat=False):
    """ Function to join Lennard-Jones parameters (``rmin`` or ``eps``) using their geometric mean. Parameters can be
    joined in two different ways (see keyword argument ``flat``)

    Parameters
    ----------
    a : numpy.ndarray
        Single atom parameters of atoms of group a
    b : numpy.ndarray
        Single atom parameters of atoms of group b
    flat : bool
        Only join parameters i of group a and j of group b if i=j. If false all combinations of i,j are computed.

    Returns
    -------
    param : numpy.ndarray
        The joined parameters of group a and b
    """

    if flat:
        return np.sqrt(a * b)
    else:
        return np.sqrt(np.outer(a, b))


def join_arith(a, b, flat=False):
    """ Function to join Lennard-Jones parameters (``rmin`` or ``eps``) using their arithmatic mean. Parameters can be
    joined in two different ways (see keyword argument ``flat``)

    Parameters
    ----------
    a : numpy.ndarray
        Single atom parameters of atoms of group a
    b : numpy.ndarray
        Single atom parameters of atoms of group b
    flat : bool
        Only join parameters i of group a and j of group b if i=j. If false all combinations of i,j are computed.

    Returns
    -------
    param : numpy.ndarray
        The joined parameters of group a and b
    """
    if flat:
        return a + b
    else:
        return np.add.outer(a, b)


rmin_charmm = {
     "C": 2.0000,
     "H": 1.27000,
     "N": 1.8500,
     "O": 1.7000,
     "S": 2.0000,
    "Se": 2.0000,  # Default Selenium to Sulfur
    "Br": 1.9800,
    "Cu": 0.7251,  # ion 2+
     'B': 2.5500,
     'F': 1.6300,
    'Na': 1.3638,  # ion
    'Mg': 1.1850,
     'P': 2.1500,
    'Cl': 1.9100,
     'K': 1.7638,  # ion
    'Ca': 1.3670,  # ion
    'Fe': 0.6500,  # HEM
    'Ni': 1.2760,
    'Zn': 1.0900,  # ion
    'Gd': 1.5050,  # ion 3+
    'Au': 0.6510,
    "join_protocol": join_arith,
}  # Bromobenzene from Gutiérrez et al. 2016

eps_charmm = {
    "C": -0.110,
    "H": -0.022,
    "N": -0.200,
    "O": -0.120,
    "S": -0.450,
    "Se": -0.450,
    "Br": -0.320,
    "Cu": -0.1505,
     'B': -0.0380,
     'F': -0.105,
    'Na': -0.0469,
    'Mg': -0.0150,
     'P': -0.5850,
    'Cl': -0.3430,
     'K': -0.0870,
    'Ca': -0.120,
    'Fe': 0.000,
    'Ni': -5.65,
    'Zn': -0.250,
    'Gd': -0.1723,
    'Au': -0.1330,
    "join_protocol": join_geom,
}  # Bromobenzene from Gutiérrez et al. 2016

rmin_uff = {
    "C": 3.851,
    "H": 2.886,
    "N": 3.660,
    "O": 3.500,
    "S": 4.035,
    "Se": 4.035,  # Default Selenium to Sulfur
    "Br": 4.189,
    "Na": 2.983,
    "Mg": 3.021,
    "P" : 4.147,
    "Cl": 3.947,
    "K" : 3.812,
    "Ca": 3.399,
    "Mn": 2.961,
    "Fe": 2.912,
    "Co": 2.872,
    "Ni": 2.834,
    "Cu": 3.495,
    "Gd": 3.368,
    "join_protocol": join_geom,
}

eps_uff = {
    "C": -0.105,
    "H": -0.044,
    "N": -0.069,
    "O": -0.060,
    "S": -0.274,
    "Se": -0.274,  # Default Selenium to Sulfur
    "Br": -0.251,
    "Na": -0.030,
    "Mg": -0.111,
    "P" : -0.305,
    "Cl": -0.227,
    "K" : -0.035,
    "Ca": -0.238,
    "Mn": -0.013,
    "Fe": -0.013,
    "Co": -0.014,
    "Ni": -0.015,
    "Cu": -0.005,
    "Gd": -0.009,
    "join_protocol": join_geom,
}

for dictionary in rmin_charmm, eps_charmm, rmin_uff, eps_uff:
    tdict = {**{key.upper(): val for key, val in dictionary.items()},
             **{key.lower(): val for key, val in dictionary.items()}}
    dictionary.update(tdict)

lj_params = {"uff": [rmin_uff, eps_uff], "charmm": [rmin_charmm, eps_charmm]}

class EnergyFunc:

    @abstractmethod
    def __call__(self, system):
        pass

    @abstractmethod
    def forces(self, system):
        pass

    @abstractmethod
    def prepare_system(self, system):
        pass

class ljEnergyFunc(EnergyFunc):
    def __init__(self, functional=None, params=None, extra_params=None, **kwargs):
        functional = get_lj_rep if functional is None else functional
        params = 'charmm' if params is None else params

        self.functional = functional
        self.rmax = kwargs.get("rmax", 10)
        self.forgive = kwargs.pop("forgive", 1)
        self.kwargs = kwargs


        if isinstance(params, dict):
            self._rmin_func, self._eps_func = params['rmin'], params['eps']
        elif params not in lj_params:
            raise RuntimeError(f'`{params}` is not a recognized forcefield.')
        else:
            self.name = params
            self._rmin_func, self._eps_func = lj_params[params]

        if extra_params is not None:
            self._rmin_func.update(extra_params['rmin'])
            self._eps_func.update(extra_params['eps'])


        self.join_rmin = self._rmin_func["join_protocol"]
        self.join_eps = self._eps_func["join_protocol"]

        self._system_hash = {}

    def get_lj_rmin(self, atypes):
        return np.array([self._rmin_func[a] for a in atypes])

    def get_lj_eps(self, atypes):
        return np.array([self._eps_func[a] for a in atypes])

    def prepare_system(self, system):

        if isinstance(system, (re.RotamerEnsemble, dre.dRotamerEnsemble)):
            # Prepare internal
            a, b = system.aidx, system.bidx
            a_eps = self.get_lj_eps(system.atom_types[a])
            a_radii = self.get_lj_rmin(system.atom_types[a])
            b_eps = self.get_lj_eps(system.atom_types[b])
            b_radii = self.get_lj_rmin(system.atom_types[b])

            system.irmin_ij = self.join_rmin(a_radii * self.forgive, b_radii * self.forgive, flat=True)
            system.ieps_ij = self.join_eps(a_eps, b_eps, flat=True)


            # Prepare external
            if system.protein is not None:
                environment_atypes = system.protein.atoms.types[system.protein_clash_idx]

                system.rmin2 = self.get_lj_rmin(system.atom_types[system.side_chain_idx])
                system.eps = self.get_lj_eps(system.atom_types[system.side_chain_idx])

                protein_lj_radii = self.get_lj_rmin(environment_atypes)
                protein_lj_eps = self.get_lj_eps(environment_atypes)

                system.ermin_ij = self.join_rmin(system.rmin2 * self.forgive, protein_lj_radii * self.forgive).reshape(-1)
                system.eeps_ij = self.join_eps(system.eps, protein_lj_eps).reshape((-1))

        elif isinstance(system, (mda.Universe, mda.AtomGroup, MolSys)):

            bonds = {(a, b) for a, b in system.atoms.bonds.indices}
            tree = cKDTree(system.atoms.positions)
            pairs = tree.query_pairs(self.rmax)
            pairs = pairs - bonds
            pairs = np.array(list(pairs))

            lj_radii_1 = self.get_lj_rmin(system.atoms.types[pairs[:, 0]])
            lj_radii_2 = self.get_lj_rmin(system.atoms.types[pairs[:, 1]])

            lj_eps_1 = self.get_lj_eps(system.atoms.types[pairs[:, 0]])
            lj_eps_2 = self.get_lj_eps(system.atoms.types[pairs[:, 1]])

            rmin_ij = self.join_rmin(lj_radii_1 * self.forgive, lj_radii_2*self.forgive, flat=True)
            eps_ij = self.join_eps(lj_eps_1, lj_eps_2, flat=True)

            if isinstance(system, MolSys):
                system.rmin_ij = rmin_ij
                system.eps_ij = eps_ij
                system.pairs = pairs
            else:
                self._system_hash[system] = rmin_ij, eps_ij, pairs

    def _get_functional_input(self, system, internal):
        if isinstance(system, (re.RotamerEnsemble, dre.dRotamerEnsemble)):
            if internal:
                rmin, eps = system.irmin_ij, system.ieps_ij
                r = np.linalg.norm(system.coords[:, system.aidx] - system.coords[:, system.bidx], axis=-1)
                shape = len(system), len(system.aidx)
            else:
                rmin, eps = system.ermin_ij, system.eeps_ij
                c1 = system.coords[:, system.side_chain_idx].reshape(-1, 3)
                c2 = system.protein_tree.data[system.protein_clash_idx]
                r = cdist(c1, c2).reshape(len(system), -1)
                shape = len(system), len(system.side_chain_idx)

        else:
            if isinstance(system, (mda.Universe, mda.AtomGroup)):
                rmin, eps, pairs = self._system_hash[system]
            else:
                rmin, eps, pairs = system.rmin_ij, system.eps_ij, system.pairs

            r = np.linalg.norm(system.positions[pairs[:,0]] - system.positions[pairs[:,1]], axis=-1)
            shape = len(system), len(pairs)

        return r, rmin, eps, shape

    def __call__(self, system, internal = False, **kwargs):
        tkwargs = {k: v for k, v in self.kwargs.items()}
        tkwargs.update(kwargs)
        r, rmin, eps, shape = self._get_functional_input(system, internal)
        E = self.functional(r, rmin, eps, **tkwargs)

        E = E.reshape(*shape, -1)
        if isinstance(system, (re.RotamerEnsemble, dre.dRotamerEnsemble)):
            system.atom_energies = E.sum(axis=2)

        return E.sum(axis=(1, 2))

KCAL2J = 4.184e3  # conversion factor form kcal to J (exact)
BOLTZ_CONST = 1.380649e-23  # Boltzmann constant, J K^-1 (exact)
AVOGADRO_CONST = 6.02214076e23  # Avogadro constant, mol^-1 (exact)
GAS_CONST = BOLTZ_CONST*AVOGADRO_CONST/KCAL2J  # molar gas constant, kcal mol^1 K^-1
