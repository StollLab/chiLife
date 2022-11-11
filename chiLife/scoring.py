from functools import wraps
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
from numba import njit
import chiLife


def clash_only(func):
    @wraps(func)
    def energy_func(protein, rotlib=None, **kwargs):
        """decorator to convert a Lennard-Jones style clash evaluation function into a chiLife compatible energy func.

    Parameters
    ----------
    func : callable
        Python function object that computes the (modified) Lennard-Jones potential given arrays of atom pair
        distances, `r`, rmin values `rmin`, energies, `eps`, and additional keyword arguments.

    Returns
    -------
    energy_func: callable
        The original input function now wrapped to accept a protein object and, optionally, a RotamerEnsemble/SpinLabel
         object.
    """
        """

        Parameters
        ----------
        protein :
            param rotlib:  (Default value = None)
        rotlib :
             (Default value = None)
        **kwargs :
            

        Returns
        -------

        """

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
    lj_energy = np.empty_like(r)

    # Unit convert
    eps = eps.copy()

    rmin_lower = forgive * rmin

    # Piecewise function for flat lj potential near rmin
    for i in range(len(r)):
        lj = rmin_lower[i] / r[i]
        lj = lj * lj * lj
        lj = lj * lj
        lj_energy[i] = np.maximum(-2 * eps[i] * lj, eps[i] * floor)

    return lj_energy


def prep_external_clash(rotlib):
    """ Helper function to prepare the lj parameters of a rotamer library, presumably with an associated protein.

    Parameters
    ----------
    rotlib : RotamerEnsemble
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
            Shape the array should be so that the energy is evaluated for each rotamer of the library separately.
    """

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
    """ Helper function to prepare the lj parameters of a rotamer library to evaluate internal clashes.

        Parameters
        ----------
        rotlib : RotamerEnsemble
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
                Shape the array should be so that the energy is evaluated for each rotamer of the library separately.
        """

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
    """Adjust rotamer population frequencies based on energy calculated from clashes.

    Parameters
    ----------
    probabilities : numpy.ndarray
        Array of Boltzmann weighted probabilities from lennard jones potential
    weights : numpy.ndarray
        Current weights of rotamers
    return_partition : bool
        Optionally return an additional argument, ``partition`` described below

    Returns
    -------
    new_weights : numpy.ndarray
        New weights adjusted by rotamer interaction energies.
    partition : float (optional)
        The partition function relative to the free label. A small partition function suggests the interactions with
        neighboring atoms are unfavorable while a large partition function suggests the opposite.
    """
    partition = np.sum(probabilities * weights) / weights.sum()
    new_weights = (probabilities / probabilities.sum()) * weights
    new_weights /= new_weights.sum()

    if return_partition:
        return new_weights, partition

    return new_weights


def get_lj_params(rotlib):
    """ calculate the lennard jones parameters between atoms of a rotamer library and associated protein.

    Parameters
    ----------
    rotlib : RotamerEnsemble
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
    """
        Global setting for using different atom type parameterization for clash evaluations. Running this function
        once will set the parameters for the whole session.

    Parameters
    ----------
    forcefield : str
        Name of the forcefield to be used.

    """
    chiLife.using_lj_param = forcefield
    rmin_func, eps_func = lj_params[forcefield]
    chiLife.get_lj_rmin = np.vectorize(rmin_func.__getitem__)
    chiLife.get_lj_eps = np.vectorize(eps_func.__getitem__)


set_lj_params("charmm")

GAS_CONST = 1.98720425864083e-3  # Kcal/K-1 mol-1  # 8.314472  # J/(K*mol)
BOLTZ_CONST = 1.3806503e-23  # J/K
KCAL2J = 4.184e3
