import warnings
from functools import wraps
from typing import Union

import MDAnalysis
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
from numba import njit
import MDAnalysis as mda

import chilife.RotamerEnsemble as re
import chilife.dRotamerEnsemble as dre
from .MolSys import MolSys


def clash_only(func):
    """Decorator to convert a Lennard-Jones style clash evaluation function into a chiLife compatible energy func.

    Parameters
    ----------
    func : callable
        Python function object that computes the (modified) Lennard-Jones potential given arrays of atom pair
        distances, `r`, rmin values `rmin`, energies, `eps`, and additional keyword arguments.

    Returns
    -------
    energy_func : callable
        The original input function now wrapped to accept a protein object and, optionally, a RotamerEnsemble/SpinLabel
        object.
    """

    @wraps(func)
    def energy_func(system, **kwargs):
        """
        Parameters
        ----------
        system : MDAnalysis.Universe, MDAnalysis.AtomGroup, RotamerEnsemble, MolSys
            Molecular system for which you wish to calculate the energy of

        **kwargs : dict
            Additional arguments to pass to the wrapped function

        Returns
        -------
        E : numpy.ndarray
            Array of energy values (kcal/mol) for each rotamer in an ensemble or for the whole system.
        """

        #TODO: Needs serious refactoring to deal with default forcefield issues.

        internal = kwargs.pop("internal", False)
        rmax = kwargs.get("rmax", 10)
        forgive = kwargs.get("forgive", 1)
        _protein = kwargs.pop('protein', None)
        forcefield = kwargs.pop('forcefield', None)

        if not isinstance(forcefield, (str, ForceField, type(None))):
            raise RuntimeError('forcefield kwarg value must be a string or ForceField object')
        elif isinstance(forcefield, str):
            forcefield = ForceField(forcefield)

        if isinstance(system, (re.RotamerEnsemble, dre.dRotamerEnsemble)):

            if system.protein is None and _protein is not None:
                system.protein = _protein

            if forcefield is not None and system.forcefield != forcefield:
                warnings.warn(f'The user provided forcefield and {type(system)} forcefield are not the same. chiLife'
                              'will use the user provided forcefield.')
            else:
                forcefield = None

            r, rmin, eps = prep_external_clash(system, forcefield=forcefield)
            E = func(r, rmin, eps, **kwargs)

            if internal:
                r, rmin, eps, shape = prep_internal_clash(system)
                Einternal = func(r, rmin, eps, **kwargs)
                E = np.concatenate((E, Einternal), axis=1)

            system.atom_energies = E.reshape(len(E), len(system.side_chain_idx), -1).sum(axis=2)
            E = E.sum(axis=1)

        elif isinstance(system, (mda.Universe, mda.AtomGroup, MolSys)):
            bonds = {(a, b) for a, b in system.atoms.bonds.indices}
            tree = cKDTree(system.atoms.positions)
            pairs = tree.query_pairs(rmax)
            pairs = pairs - bonds
            pairs = np.array(list(pairs))

            r = np.linalg.norm(
                system.atoms.positions[pairs[:, 0]]
                - system.atoms.positions[pairs[:, 1]],
                axis=1,
                )

            if forcefield is None:
                forcefield = ForceField('charmm')

            lj_radii_1 = forcefield.get_lj_rmin(system.atoms.types[pairs[:, 0]])
            lj_radii_2 = forcefield.get_lj_rmin(system.atoms.types[pairs[:, 1]])

            lj_eps_1 = forcefield.get_lj_eps(system.atoms.types[pairs[:, 0]])
            lj_eps_2 = forcefield.get_lj_eps(system.atoms.types[pairs[:, 1]])

            join_rmin = forcefield.get_lj_rmin("join_protocol")[()]
            join_eps = forcefield.get_lj_eps("join_protocol")[()]

            rmin_ij = join_rmin(lj_radii_1 * forgive, lj_radii_2 * forgive, flat=True)
            eps_ij = join_eps(lj_eps_1, lj_eps_2, flat=True)
            E = func(r, rmin_ij, eps_ij, **kwargs)
            E = E.sum()

        else:
            raise TypeError(f'Energy evaluations of a {type(system)} object are not supported at this time. Please '
                            f'pass an chilife.RotamerEnsemble, mda.Universe, mda.AtomGroup or chilife.MolSys')

        return E

    return energy_func


@clash_only
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


@clash_only
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


@clash_only
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

@clash_only
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


def prep_external_clash(ensemble, forcefield=None):
    """ Helper function to prepare the lj parameters of a rotamer ensemble, presumably with an associated protein.

    Parameters
    ----------
    ensemble : RotamerEnsemble
        The RotamerEnsemble to prepare the clash parameters for. The RotamerEnsemble should have an associated protein.

    Returns
    -------
        dist : numpy.ndarray
            Array of pairwise distances between atoms in the RotamerEnsemble and atoms in the associated protein.
        rmin_ij :  numpy.ndarray
            ``rmin`` parameters of the lj potential associated with the atom ``i`` and atom ``j`` pair.
        eps_ij : numpy.ndarray
            ``eps`` parameters of the lj potential associated with the atom ``i`` and atom ``j`` pair.
        shape : Tuple[int]
            Shape the array should be so that the energy is evaluated for each rotamer of the ensemble separately.
    """

    # Calculate rmin and epsilon for all atoms in protein that may clash
    if forcefield is None:
        forcefield = ensemble.forcefield

    if hasattr(ensemble, 'ermin_ij') and forcefield == ensemble.forcefield:
        rmin_ij = ensemble.ermin_ij
        eps_ij = ensemble.eeps_ij
    else:
        rmin_ij, eps_ij = forcefield.get_lj_params(ensemble)
        ensemble.ermin_ij = rmin_ij
        ensemble.eeps_ij = eps_ij

    if len(ensemble) == 1:
        ec = ensemble.coords[0, ensemble.side_chain_idx]
    else:
        ec = ensemble.coords[:, ensemble.side_chain_idx].reshape(-1, 3)

    pc = ensemble.protein_tree.data[ensemble.protein_clash_idx]

    # Calculate distances
    dist = cdist(ec, pc).reshape(len(ensemble), -1)

    return dist, rmin_ij, eps_ij


def prep_internal_clash(ensemble):
    """ Helper function to prepare the lj parameters of a rotamer ensemble to evaluate internal clashes.

        Parameters
        ----------
        ensemble : RotamerEnsemble
            The RotamerEnsemble to prepare the clash parameters for.

        Returns
        -------
            dist : numpy.ndarray
                Array of pairwise distances between atoms in the RotamerEnsemble and atoms in the associated protein.
            rmin_ij :  numpy.ndarray
                ``rmin`` parameters of the lj potential associated with the atom ``i`` and atom ``j`` pair.
            eps_ij : numpy.ndarray
                ``eps`` parameters of the lj potential associated with the atom ``i`` and atom ``j`` pair.
            shape : Tuple[int]
                Shape the array should be so that the energy is evaluated for each rotamer of the ensemble separately.
        """

    if hasattr(ensemble, 'ermin_ij'):
        rmin_ij, eps_ij = ensemble.ermin_ij, ensemble.eeps_ij
    else:
        a, b = [list(x) for x in zip(*ensemble.non_bonded)]
        a_eps = ensemble.forcefield.get_lj_eps(ensemble.atom_types[a])
        a_radii = ensemble.forcefield.get_lj_rmin(ensemble.atom_types[a])
        b_eps = ensemble.forcefield.get_lj_eps(ensemble.atom_types[b])
        b_radii = ensemble.forcefield.get_lj_rmin(ensemble.atom_types[b])

        join_rmin = ensemble.forcefield.get_lj_rmin("join_protocol")[()]
        join_eps = ensemble.forcefield.get_lj_eps("join_protocol")[()]

        rmin_ij = join_rmin(a_radii * ensemble.forgive, b_radii * ensemble.forgive, flat=True)
        eps_ij = join_eps(a_eps, b_eps, flat=True)

        ensemble.irmin_ij = rmin_ij
        ensemble.ieps_ij = eps_ij
        ensemble.aidx = a
        ensemble.bidx = b

    dist = np.linalg.norm(ensemble.coords[:, ensemble.aidx] - ensemble.coords[:, ensemble.bidx], axis=2)
    shape = (len(ensemble.coords), len(a_radii))

    return dist, rmin_ij, eps_ij, shape


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



class ForceField:

    def __init__(self, forcefield: Union[str, dict] = 'charmm', extra_params: dict = None):

        if isinstance(forcefield, dict):
            self._rmin_func, self._eps_func = forcefield['rmin'], forcefield['eps']
        elif forcefield not in lj_params:
            raise RuntimeError(f'`{forcefield}` is not a recognized forcefield.')
        else:
            self.name = forcefield
            self._rmin_func, self._eps_func = lj_params[forcefield]

        if extra_params is not None:
            self._rmin_func.update(extra_params['rmin'])
            self._eps_func.update(extra_params['eps'])

        self.get_lj_rmin = np.vectorize(self._rmin_func.__getitem__)
        self.get_lj_eps = np.vectorize(self._eps_func.__getitem__)

    def get_lj_params(self, ensemble: Union['RotamerEnsemble', 'MolSys', MDAnalysis.Universe, MDAnalysis.AtomGroup]):
        """ calculate the lennard jones parameters between atoms of a rotamer ensemble and associated protein.

        Parameters
        ----------
        ensemble : RotamerEnsemble
            The RotamerEnsemble to get the lj params for. Should have an associated protein object.

        Returns
        -------
        rmin_ij : nummpy.ndarray
            Vector of ``rmin`` lennard-jones parameters corresponding to atoms i and j of the RotamerEnsemble and the
            associated protein respectively.
        eps_ij : numpy.ndarray
            Vector of ``eps`` lennard-jones parameters corresponding to atoms i and j of the RotamerEnsemble and the
            associated protein respectively.
        """

        environment_atypes = ensemble.protein.atoms.types[ensemble.protein_clash_idx]
        protein_lj_radii = self.get_lj_rmin(environment_atypes)
        protein_lj_eps = self.get_lj_eps(environment_atypes)
        join_rmin = self.get_lj_rmin("join_protocol")[()]
        join_eps = self.get_lj_eps("join_protocol")[()]

        rmin_ij = join_rmin(ensemble.rmin2 * ensemble.forgive, protein_lj_radii * ensemble.forgive).reshape(-1)
        eps_ij = join_eps(ensemble.eps, protein_lj_eps).reshape((-1))

        return rmin_ij, eps_ij

    def __eq__(self, other):
        return self.name == other.name


KCAL2J = 4.184e3  # conversion factor form kcal to J (exact)
BOLTZ_CONST = 1.380649e-23  # Boltzmann constant, J K^-1 (exact)
AVOGADRO_CONST = 6.02214076e23  # Avogadro constant, mol^-1 (exact)
GAS_CONST = BOLTZ_CONST*AVOGADRO_CONST/KCAL2J  # molar gas constant, kcal mol^1 K^-1
