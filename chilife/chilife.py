from __future__ import annotations
import tempfile, logging, os, rtoml
import zipfile, shutil
from pathlib import Path
from itertools import combinations, product
from typing import Callable, Tuple, Union, List, Dict
from unittest import mock
from tqdm import tqdm

import numpy as np
from numpy.typing import ArrayLike
from scipy.spatial.distance import cdist
import MDAnalysis as mda
import MDAnalysis.transformations

import chilife
from .protein_utils import dihedral_defs, local_mx, sort_pdb, mutate, save_pdb, ProteinIC, get_min_topol
from .scoring import get_lj_rep, GAS_CONST
from .numba_utils import get_delta_r, normdist
from .SpinLabel import SpinLabel
from .RotamerEnsemble import RotamerEnsemble
from .SpinLabelTraj import SpinLabelTraj


logging.captureWarnings(True)

# Define useful global variables
SUPPORTED_BB_LABELS = ("R1C",)
DATA_DIR = Path(__file__).parent.absolute() / "data/"
RL_DIR = Path(__file__).parent.absolute() / "data/rotamer_libraries/"


with open(RL_DIR / "defaults.toml", "r") as f:
    rotlib_defaults = rtoml.load(f)

USER_LABELS = {f.name[:-11] for f in (RL_DIR / "user_rotlibs").glob("*rotlib.npz")}
USER_dLABELS = {f.name[:-12] for f in (RL_DIR / "user_rotlibs").glob("*drotlib.zip")}
USER_dLABELS = USER_dLABELS | {f.name[:-15] for f in (RL_DIR / "user_rotlibs").glob("*drotlib.zip")}
SUPPORTED_RESIDUES = set(dihedral_defs.keys())
[SUPPORTED_RESIDUES.remove(lab) for lab in ("CYR1", "MTN")]

def distance_distribution(
    *args: SpinLabel,
    r: ArrayLike = None,
    sigma: float = 1.0,
    use_spin_centers: bool = True,
    uq: bool = False,
) -> np.ndarray:
    """Calculates total distribution of spin-spin distances among an arbitrary number of spin labels, using the
    distance range ``r`` (in angstrom).

    The distance distribution is obtained by summing over all pair distance distributions. These in turn are calculated
    by summing over rotamer pairs with the appropriate weights. For each rotamer pair, the distance distribution is
    either just the distance between the ``spin_center`` coordinates of two labels (if ``spin_populations=False``) or
    the weighted sum over all pairs of spn-bearing atoms (``spin_populations=True``). The resulting distance histogram
    is convolved with a normal distribution with a standard deviation ``sigma``.

    Parameters
    ----------
    *args : SpinLabel
        Any number of spin label objects.
    r : ArrayLike
        Evenly spaced array of distances (in angstrom) to calculate the distance distribution over.
    sigma : float
        The standard deviation of the normal distribution used in convolution with the distance histogram, in angstrom.
        Default is 1.
    use_spin_centers : bool
        If False, distances are computed between spin centers. If True, distances are computed by summing over
        the distributed spin density on spin-bearing atoms on the labels.
    uq : bool
        Perform uncertainty analysis (experimental)

    Returns
    -------
    P : np.ndarray
        Predicted distance distribution, in 1/angstrom
    """

    # Allow r to be passed as last non-keyword argument
    if r is None and np.ndim(args[-1]) != 0:
        r = args[-1]
        args = args[:-1]

    if len(args)<2:
        raise TypeError('At least two spin label objects are required.')

    if r is None:
        raise TypeError('Keyword argument r with distance domain vector is missing.')

    if any(not hasattr(arg, atr) for arg in args for atr in ["spin_coords", "spin_centers", "weights"]):
        raise TypeError(
            "Arguments other than spin labels must be passed as a keyword arguments."
        )

    r = np.asarray(r)

    if any(isinstance(SL, SpinLabelTraj) for SL in args):

        P = traj_dd(*args, r=r, sigma=sigma)
        return P

    elif uq:

        Ps = []
        n_boots = uq if uq > 1 else 100
        for i in range(n_boots):
            dummy_labels = []
            for SL in args:
                idxs = np.random.choice(len(SL), len(SL))

                dummy_SL = mock.Mock()
                dummy_SL.spin_coords = np.atleast_2d(SL.spin_coords[idxs])
                dummy_SL.spin_centers = np.atleast_2d(SL.spin_centers[idxs])
                dummy_SL.spin_weights = SL.spin_weights
                dummy_SL.weights = SL.weights[idxs]
                dummy_SL.weights /= dummy_SL.weights.sum()
                dummy_labels.append(dummy_SL)

            Ps.append(pair_dd(*dummy_labels, r=r, sigma=sigma, use_spin_centers=use_spin_centers))
        Ps = np.array(Ps)
        return Ps

    else:

        P = pair_dd(*args, r=r, sigma=sigma, use_spin_centers=use_spin_centers)
        return P


def pair_dd(*args, r: ArrayLike, sigma: float = 1.0, use_spin_centers: bool = True) -> np.ndarray:
    """Obtain the total pairwise spin-spin distance distribution over ``r`` for a list of spin labels.
    The distribution is calculated by convolving the weighted histogram of pairwise spin-spin
    distances with a normal distribution with standard deviation ``sigma``.

    Parameters
    ----------
    *args : SpinLabel
        SpinLabels to use when calculating the distance distribution
    r : ArrayLike
        Distance domain vector, in angstrom
    sigma : float
         Standard deviation of normal distribution used for convolution, in angstrom
    use_spin_centers : bool
        If False, distances are computed between spin centers. If True, distances are computed by summing over
        the distributed spin density on spin-bearing atoms on the labels.

    Returns
    -------
    P : np.ndarray
        Predicted normalized distance distribution, in units of 1/angstrom

    """
    # Calculate pairwise distances and weights
    SL_Pairs = combinations(args, 2)
    weights, distances = [], []
    for SL1, SL2 in SL_Pairs:
        if use_spin_centers:
            coords1 = SL1.spin_centers
            coords2 = SL2.spin_centers
            weights1 = SL1.weights
            weights2 = SL2.weights
        else:
            coords1 = SL1.spin_coords.reshape(-1 ,3)
            coords2 = SL2.spin_coords.reshape(-1, 3)
            weights1 = np.outer(SL1.weights, SL1.spin_weights).flatten()
            weights2 = np.outer(SL2.weights, SL2.spin_weights).flatten()

        distances.append(cdist(coords1, coords2).flatten())
        weights.append(np.outer(weights1, weights2).flatten())

    distances = np.concatenate(distances)
    weights = np.concatenate(weights)

    # Calculate distance histogram
    hist, _ = np.histogram(
        distances, weights=weights, range=(min(r), max(r)), bins=len(r)
    )

    # Convolve with normal distribution if non-zero standard deviation is given
    if sigma != 0:
        delta_r = get_delta_r(r)
        _, g = normdist(delta_r, 0, sigma)
        P = np.convolve(hist, g, mode="same")
    else:
        P = hist

    # Normalize distribution
    integral = np.trapz(P,r)
    if integral != 0:
        P /= integral

    return P


def traj_dd(
    SL1: SpinLabelTraj,
    SL2: SpinLabelTraj,
    r: ArrayLike,
    sigma: float,
    **kwargs,
) -> np.ndarray:
    """Calculate a distance distribution from a trajectory of spin labels by calling ``distance_distribution`` on each frame and
    averaging the resulting distributions.

    Parameters
    ----------
    SL1, SL2: SpinLabelTrajectory
        Spin label to use for distance distribution calculation.
    r : ArrayLike
        Distance domain to use when calculating distance distribution.
    sigma : float
        Standard deviation of the gaussian kernel used to smooth the distance distribution
    filter : bool, float
        Option to prune out negligible population rotamer pairs from distance distribution calculation. The fraction
        omitted can be specified by assigning a float to ``prune``
    **kwargs : dict, optional
        Additional keyword arguments to pass to ``distance_distribution`` .

    Returns
    -------
    P: ndarray
        Distance distribution calculated from the provided SpinLabelTrajectories
    """

    # Ensure that the SpinLabelTrajectories have the same number of frames.
    if len(SL1) != len(SL2):
        raise ValueError("SpinLabelTraj objects must have the same length")

    # Calculate the distance distribution for each frame and sum
    P = np.zeros_like(r)
    for _SL1, _SL2 in zip(SL1, SL2):
        P += distance_distribution(_SL1, _SL2, r=r, sigma=sigma, **kwargs)

    # Normalize distance distribution
    P /= np.trapz(P, r)

    return P


def repack(
    protein: Union[mda.Universe, mda.AtomGroup],
    *spin_labels: RotamerEnsemble,
    repetitions: int = 200,
    temp: float = 1,
    energy_func: Callable = get_lj_rep,
    off_rotamer=False,
    **kwargs,
) -> Tuple[mda.Universe, ArrayLike]:
    """Markov chain Monte Carlo repack a protein around any number of SpinLabel or RotamerEnsemble objects.

    Parameters
    ----------
    protein : MDAnalysis.Universe, MDAnalysis.AtomGroup
        Protein to be repacked
    spin_labels : RotamerEnsemble, SpinLabel
        RotamerEnsemble or SpinLabel object placed at site of interest
    repetitions : int
        Number of successful MC samples to perform before terminating the MC sampling loop
    temp : float, ArrayLike
        Temperature (Kelvin) for both clash evaluation and metropolis-hastings acceptance criteria. Accepts a list or
        array like object of temperatures if a temperature schedule is desired
    energy_func : Callabale
        Energy function to be used for clash evaluation. Must accept a protein and RotmerLibrary object and return an
        array of potentials in kcal/mol, with one energy per rotamer in the rotamer ensemble
    off_rotamer : bool
        Boolean argument that decides whether off rotamer sampling is used when repacking the provided residues.
    kwargs : dict
        Additional keyword arguments to be passed to ``mutate`` .

    Returns
    -------
    protein: MDAnalysis.Universe
        MCMC trajectory of local repack
    deltaEs: np.ndarray:
        Change in energy_func score at each accept of MCMC trajectory

    """
    temp = np.atleast_1d(temp)
    KT = {t: GAS_CONST * t for t in temp}

    repack_radius = kwargs.pop("repack_radius") if "repack_radius" in kwargs else None  # Angstroms
    if repack_radius is None:
            repack_radius = max([SL.clash_radius for SL in spin_labels])
    # Construct a new spin labeled protein and preallocate variables to retain monte carlo trajectory
    spin_label_str = " or ".join(
        f"( {spin_label.selstr} )" for spin_label in spin_labels
    )
    protein = mutate(protein, *spin_labels, **kwargs).atoms

    # Determine the residues near the spin label that will be repacked
    repack_residues = protein.select_atoms(
        f"(around {repack_radius} {spin_label_str} ) " f"or {spin_label_str}"
    ).residues

    repack_res_kwargs = spin_labels[0].input_kwargs
    repack_residue_libraries = [
        RotamerEnsemble.from_mda(res, **repack_res_kwargs)
        for res in repack_residues
        if res.resname not in ["GLY", "ALA"]
    ]

    # Create new labeled protein construct to fill in any missing atoms of repack residues
    protein = mutate(protein, *repack_residue_libraries, **kwargs).atoms

    repack_residues = protein.select_atoms(
        f"(around {repack_radius} {spin_label_str} ) " f"or {spin_label_str}"
    ).residues

    repack_residue_libraries = [
        RotamerEnsemble.from_mda(res, **repack_res_kwargs)
        for res in repack_residues
        if res.resname not in ["GLY", "ALA"]
    ]

    traj = np.empty((repetitions, *protein.positions.shape))
    deltaEs = []

    sample_freq = np.array(
        [len(res.weights) for res in repack_residue_libraries], dtype=np.float64
    )
    sample_freq /= sample_freq.sum()

    count = 0
    acount = 0
    bcount = 0
    bidx = 0
    schedule = repetitions / (len(temp) + 1)
    with tqdm(total=repetitions) as pbar:
        while count < repetitions:

            # Randomly select a residue from the repack residues
            SiteLibrary = repack_residue_libraries[
                np.random.choice(len(repack_residue_libraries), p=sample_freq)
            ]
            if not hasattr(SiteLibrary, "dummy_label"):
                SiteLibrary.dummy_label = SiteLibrary.copy()
                SiteLibrary.dummy_label.protein = protein
                SiteLibrary.dummy_label.mask = np.isin(
                    protein.ix, SiteLibrary.clash_ignore_idx
                )
                SiteLibrary.dummy_label._coords = np.atleast_3d(
                    [protein.atoms[SiteLibrary.dummy_label.mask].positions]
                )

                with np.errstate(divide="ignore"):
                    SiteLibrary.dummy_label.E0 = energy_func(
                        SiteLibrary.dummy_label.protein, SiteLibrary.dummy_label
                    ) - KT[temp[bidx]] * np.log(SiteLibrary.current_weight)

            DummyLabel = SiteLibrary.dummy_label

            coords, weight = SiteLibrary.sample(off_rotamer=off_rotamer)

            with np.errstate(divide="ignore"):
                DummyLabel._coords = np.atleast_3d([coords])
                E1 = energy_func(DummyLabel.protein, DummyLabel) - KT[
                    temp[bidx]
                ] * np.log(weight)

            deltaE = E1 - DummyLabel.E0
            deltaE = np.maximum(deltaE, -10.0)
            if deltaE == -np.inf:
                print(SiteLibrary.name)
                print(E1, DummyLabel.E0)

            acount += 1
            # Metropolis-Hastings criteria

            if (
                E1 < DummyLabel.E0
                or np.exp(-deltaE / KT[temp[bidx]]) > np.random.rand()
            ):

                deltaEs.append(deltaE)
                try:
                    protein.atoms[DummyLabel.mask].positions = coords
                    DummyLabel.E0 = E1

                except ValueError as err:
                    print(SiteLibrary.name)
                    print(SiteLibrary.atom_names)

                    raise ValueError(err)

                traj[count] = protein.atoms.positions
                SiteLibrary.update_weight(weight)
                count += 1
                bcount += 1
                pbar.update(1)
                if bcount > schedule:
                    bcount = 0
                    bidx = np.minimum(bidx + 1, len(temp) - 1)
            else:
                continue

    logging.info(f"Total counts: {acount}")

    # Load MCMC trajectory into universe.
    protein.universe.load_new(traj)

    return protein, deltaEs


def add_library(
    libname: str,
    pdb: str,
    dihedral_atoms: List[List[str]],
    site: int = 1,
    resname: str = None,
    dihedrals: ArrayLike = None,
    weights: ArrayLike = None,
    sigmas: ArrayLike = None,
    permanent: bool = False,
    default: bool = False,
    force: bool = False,
    spin_atoms: List[str] = None
) -> None:
    """Add a user defined SpinLabel from a pdb file.

    Parameters
    ----------
    libname : str
        Name for the new rotamer library.
    pdb : str
        Name of (and path to) pdb file containing the user defined spin label structure. This pdb file should contain
        only the desired spin label and no additional residues.
    site : int
        The residue number of the side chain in the pdb file you would like to add.
    resname : str
        3-letter residue code.
    dihedral_atoms : list
        List of rotatable dihedrals. List should contain sublists of 4 atom names. Atom names must be the same as
        defined in the pdb file eg:
        
        .. code-block:: python

            [['CA', 'CB', 'SG', 'SD'],
            ['CB', 'SG', 'SD', 'CE']...]

    dihedrals : ArrayLike, optional
        Array of dihedral angles. If provided the new label object will be stored as a rotamer library with the
        dihedrals provided. Array should be n x m where n is the number of rotamers and m is the number of dihedrals.
    weights : ArrayLike, optional
        Weights associated with the dihedral angles provided by the ``dihedrals`` keyword argument. There should be one
        weight per rotamer and the rotamer weights should sum to 1.
    sigmas : ArrayLike, optional
        Sigma parameter for distributions of dihedral angles. Should be n x m matrix where n is the number of rotamers
        and m is the number of dihedrals. This feature will be used when performing off rotamer samplings.
    skews : ArrayLike, optional
        Skew parameter for distributions of dihedral angles. Should be n x m matrix where n is the number of rotamers
        and m is the number of dihedrals. This feature will be used when performing off rotamer samplings.
    permanent: bool
        If set to True the library will be stored in the chilife user_rotlibs directory in addition to the current
        working directory.
    default : bool
        If set to true and permanent is also set to true then this rotamer library will become the default rotamer
        library for the given resname
    force: bool = False,
        If set to True and permanent is also set to true this library will overwrite any existing library with the same
        name.
    spin_atoms : list
        List of atom names on which the spin density is localized.
    Returns
    -------
    None
    """
    resname = libname[:3] if resname is None else resname
    struct, spin_atoms = pre_add_library(pdb, spin_atoms)
    resi_selection = struct.select_atoms(f"resnum {site}")
    bonds = resi_selection.intra_bonds.indices - resi_selection.atoms[0].ix

    # Convert loaded rotamer ensemble to internal coords
    internal_coords = [
        chilife.get_internal_coords(
            resi_selection,
            preferred_dihedrals=dihedral_atoms,
            bonds=bonds
        )
        for _ in struct.trajectory
    ]

    # set resnum to 1 and remove chain operators so all rotamers are in the ic coordinate frame
    for ic in internal_coords:
        ic.shift_resnum(-(site - 1))
        ic.chain_operators = None
        if len(ic.chains) > 1:
            raise ValueError('The PDB of the label supplied appears to have a chain break. Please check your PDB and '
                             'make sure there are no chain breaks in the desired label and that there are no other '
                             'chains in the pdb file. If the error persists, check to be sure all atoms are the correct '
                             'element as chilife uses the elements to determine if atoms are bonded.')

    # If multi-state pdb extract dihedrals from pdb
    if dihedrals is None:
        dihedrals = np.rad2deg(
            [ic.get_dihedral(1, dihedral_atoms) for ic in internal_coords]
        )

    if dihedrals.shape == (len(dihedrals),):
        dihedrals.shape = (len(dihedrals), 1)

    if weights is None:
        weights = np.ones(len(dihedrals))
        weights /= weights.sum()

    save_dict = prep_restype_savedict(libname, resname, internal_coords,
                                      weights, dihedrals, dihedral_atoms,
                                      sigmas=sigmas, spin_atoms=spin_atoms)

    # Save rotamer library
    np.savez(Path().cwd() / f'{libname}_rotlib.npz', **save_dict, allow_pickle=True)

    if permanent:
        store_loc = RL_DIR / f"user_rotlibs/{libname}_rotlib.npz"
        add_to_defaults(resname, libname, default)
        if force or not store_loc.exists():
            np.savez(store_loc, **save_dict, allow_pickle=True)
            add_dihedral_def(libname, dihedral_atoms, force=force)
            global USER_LABELS
            USER_LABELS.add(libname)
        else:
            raise NameError("A rotamer library with this name already exists! Please choose a different name or do"
                            "not store as a permanent rotamer library")


def add_dlibrary(
    libname: str,
    pdb: str,
    increment: int,
    dihedral_atoms: List[List[List[str]]],
    site: int = 1,
    resname: str = None,
    dihedrals: ArrayLike = None,
    weights: ArrayLike = None,
    permanent: bool = False,
    default: bool = False,
    force: bool = False,
    spin_atoms: List[str] = None,
) -> None:
    """Add a user defined dSpinLabel from a pdb file.

    Parameters
    ----------
    libname: str,
        Name for the user defined label.
    increment : int
        The number of residues the second site away from the first site.
    pdb : str
        Name of (and path to) pdb file containing the user defined spin label structure. This pdb file should contain
        only the desired spin label and no additional residues.
    site : int
        The residue number of the first side chain in the pdb file you would like to add.
    resname : str
        Residue type 3-letter code.
    dihedral_atoms : list
        list of rotatable dihedrals. List should contain lists of 4 atom names. Atom names must be the same as defined
        in the pdb file eg:
        
        .. code-block:: python
        
            [['CA', 'CB', 'SG', 'SD'],
            ['CB', 'SG', 'SD', 'CE']...]

    dihedrals : ArrayLike, optional
        Array of dihedral angles. If provided the new label object will be stored as a rotamer library with the
        dihedrals provided.
    weights : ArrayLike, optional
        Weights associated with the dihedral angles provided by the ``dihedrals`` keyword argument
        permanent: bool
        If set to True the library will be stored in the chilife user_rotlibs directory in addition to the current
        working directory.
    default : bool
        If set to true and permanent is also set to true then this rotamer library will become the default rotamer
        library for the given resname
    force: bool = False,
        If set to True and permanent is also set to true this library will overwrite any existing library with the same
        name.
    spin_atoms : list, dict
        List dictionary of atom names on which the spin density is localized.
    Returns
    -------
    None
    """
    resname = libname[:3] if resname is None else resname
    if len(dihedral_atoms) != 2:
        dihedral_error = True
    elif not isinstance(dihedral_atoms[0], List):
        dihedral_error = True
    elif len(dihedral_atoms[0][0]) != 4:
        dihedral_error = True
    else:
        dihedral_error = False

    if dihedral_error:
        raise ValueError(
            "dihedral_atoms must be a list of lists where each sublist contains the list of dihedral atoms"
            "for the i and i+{increment} side chains. Sublists can contain any amount of dihedrals but "
            "each dihedral should be defined by exactly four unique atom names that belong to the same "
            "residue number"
        )

    struct, spin_atoms = pre_add_library(pdb, spin_atoms, uniform_topology=False)

    IC1 = [
        chilife.get_internal_coords(
            struct.select_atoms(f"resnum {site}"),
            preferred_dihedrals=dihedral_atoms[0],
        )
        for ts in struct.trajectory
    ]

    IC2 = [
        chilife.get_internal_coords(
            struct.select_atoms(f"resnum {site + increment}"),
            preferred_dihedrals=dihedral_atoms[1],
        )
        for ts in struct.trajectory
    ]

    for ic1, ic2 in zip(IC1, IC2):
        ic1.shift_resnum(-(site - 1))
        ic2.shift_resnum(-(site + increment - 1))

    # Identify atoms that dont move with respect to each other but move with the dihedrals
    maxindex1 = (
        max([IC1[0].ICs[1][1][tuple(d[::-1])].index for d in dihedral_atoms[0]]) - 1
    )
    maxindex2 = (
        max([IC2[0].ICs[1][1][tuple(d[::-1])].index for d in dihedral_atoms[1]]) - 1
    )

    # Get constraint pairs between SLs as constriants
    cst_pool1 = [
        atom.index
        for atom in IC1[0].ICs[1][1].values()
        if atom.index >= maxindex1 and atom.atype != "H"
    ]
    cst_pool2 = [
        atom.index
        for atom in IC2[0].ICs[1][1].values()
        if atom.index >= maxindex2 and atom.atype != "H"
    ]

    cst_pairs = np.array(list(product(cst_pool1, cst_pool2)))

    constraint_distances = [
        np.linalg.norm(
            ic1.coords[cst_pairs[:, 0]] - ic2.coords[cst_pairs[:, 1]], axis=1
        )
        for ic1, ic2 in zip(IC1, IC2)
    ]

    csts = {'cst_pairs': cst_pairs, 'cst_distances': constraint_distances}

    # If multi-state pdb extract rotamers from pdb
    if dihedrals is None:
        dihedrals = []
        for IC, resnum, dihedral_set in zip(
            [IC1, IC2], [site, site + increment], dihedral_atoms
        ):
            dihedrals.append(
                [[ICi.get_dihedral(1, ddef) for ddef in dihedral_set] for ICi in IC]
            )

    if weights is None:
        weights = np.ones(len(IC1))

    weights /= weights.sum()
    libname = libname+f'ip{increment}'
    save_dict_1 = prep_restype_savedict(libname + 'A', resname, IC1,
                                        weights, dihedrals[0], dihedral_atoms[0],
                                        spin_atoms=spin_atoms)
    save_dict_2 = prep_restype_savedict(libname + 'B', resname, IC2,
                                        weights, dihedrals[1], dihedral_atoms[1],
                                        resi=1 + increment,
                                        spin_atoms=spin_atoms)

    # Save individual data sets and zip
    cwd = Path().cwd()
    np.savez(cwd / f'{libname}A_rotlib.npz', **save_dict_1, allow_pickle=True)
    np.savez(cwd / f'{libname}B_rotlib.npz', **save_dict_2, allow_pickle=True)
    np.savez(cwd / f'{libname}_csts.npz', **csts)

    with zipfile.ZipFile(f'{libname}_drotlib.zip', mode='w') as archive:
        archive.write(f'{libname}A_rotlib.npz')
        archive.write(f'{libname}B_rotlib.npz')
        archive.write(f'{libname}_csts.npz')

    # Cleanup intermediate files
    os.remove(f'{libname}A_rotlib.npz')
    os.remove(f'{libname}B_rotlib.npz')
    os.remove(f'{libname}_csts.npz')

    if permanent:
        store_loc = RL_DIR / f"user_rotlibs/{libname}_drotlib.zip"
        add_to_defaults(resname, libname, default)
        if force or not store_loc.exists():
            shutil.copy(f'{libname}_drotlib.zip', str(store_loc))
            global USER_dLABELS
            USER_dLABELS.add(libname)
            add_dihedral_def(libname, dihedral_atoms, force=force)
        else:
            raise NameError("A rotamer library with this name already exists! Please choose a different name or do"
                            "not store as a permanent rotamer library")

def pre_add_library(
        pdb: str,
        spin_atoms: List[str],
        uniform_topology: bool = True,
) -> Tuple[MDAnalysis.Universe, Dict]:
    """Helper function to sort pdbs, save spin atoms, update lists, etc when adding a SpinLabel or dSpinLabel.

    Parameters
    ----------
    pdb : str
        Name (and path) of the pdb containing the new label.
    spin_atoms : List[str]
        Atoms of the SpinLabel where the unpaired electron is located.
    uniform_topology : bool
        Assume all rotamers of the library have the same topology (i.e. no differences in atom bonding). If false
        chilife will attempt to find the minimal topology shared between all rotamers for defining internal coordinates.

    Returns
    -------
    struct : MDAnalysis.Universe
        MDAnalysis Universe object containing the rotamer ensemble with each rotamer as a frame. All atoms should be
        properly sorted for consistent construction of internal coordinates.
    spin_atoms : dict
        Dictionary of spin atoms and weights if specified.
    """
    # Sort the PDB for optimal dihedral definitions
    pdb_lines = sort_pdb(pdb, uniform_topology=uniform_topology)
    bonds = get_min_topol(pdb_lines)

    # Write a temporary file with the sorted atoms
    if isinstance(pdb_lines[0], list):
        with tempfile.NamedTemporaryFile(suffix=".pdb", mode="w+", delete=False) as tmpfile:
            for i, model in enumerate(pdb_lines):
                tmpfile.write(f"MODEL {i + 1}\n")
                for atom in model:
                    tmpfile.write(atom)
                tmpfile.write("ENDMDL\n")
    else:
        with tempfile.NamedTemporaryFile(suffix=".pdb", mode="w+", delete=False) as tmpfile:
            for line in pdb_lines:
                tmpfile.write(line)

    # Load sorted atom pdb using MDAnalysis and remove tempfile
    struct = mda.Universe(tmpfile.name, in_memory=True)
    struct.universe.add_bonds(bonds)
    os.remove(tmpfile.name)

    # Store spin atoms if provided
    if spin_atoms is not None:
        if isinstance(spin_atoms, str):
            spin_atoms = spin_atoms.split()

        if isinstance(spin_atoms, dict):
            spin_weights = list(spin_atoms.values())
            spin_atoms = list(spin_atoms.keys())
        else:
            w = 1/len(spin_atoms)
            spin_weights = [w for _ in spin_atoms]

        spin_atoms = {'spin_atoms': spin_atoms, 'spin_weights': spin_weights}

    return struct, spin_atoms


def prep_restype_savedict(
      libname: str,
      resname: str,
      internal_coords: List[ProteinIC],
      weights: ArrayLike,
      dihedrals: ArrayLike,
      dihedral_atoms: ArrayLike,
      sigmas: ArrayLike = None,
      resi: int = 1,
      spin_atoms: List[str] = None
) -> Dict:
    """Helper function to add new residue types to chilife

    Parameters
    ----------
    libname : str
        Name of residue to be stored.
    resname : str
        Residue name (3-letter code)
    internal_coords : List[ProteinIC]
        list of internal coordinates of the new residue type.
    weights : ArrayLike
        Array of weights corresponding to each rotamer of the library
    dihedrals : ArrayLike
        Array of mobile dihedral angles for each rotamer fo the library
    dihedral_atoms : ArrayLike
        Definition of mobile dihedral angles for a single structure. Should be a list of 4 string lists where the
        four strings are the names of the atoms that define the dihedral.
    sigmas : ArrayLike
        Array of sigma values for each dihedral of each rotamer.
    resi: int
        The residue number to be stored.
    Returns
    -------
    save_dict : dict
        Dictionary of all the data needed to build a RotamerEnsemble.
    """
    # Extract coordinates and transform to the local frame
    bb_atom_idx = [
        i for i, atom in enumerate(internal_coords[0].atoms) if atom.name in ["N", "CA", "C"]
    ]
    coords = internal_coords[0].coords.copy()
    ori, mx = local_mx(*coords[bb_atom_idx])
    coords = (coords - ori) @ mx

    if len(internal_coords) > 1:
        coords = np.array([(IC.coords - ori) @ mx for IC in internal_coords])
    elif len(dihedrals) > 1:
        coords = np.array([internal_coords.set_dihedral(dihe, resi, dihedral_atoms) for dihe in dihedrals])
    else:
        if coords.ndim == 2:
            coords = np.expand_dims(coords, axis=0)

    if np.any(np.isnan(coords)):
        idxs = np.argwhere(np.isnan(coords, np.sum(axis=(1, 2))))
        raise(ValueError(f'Coords of rotamer {" ".join((str(idx) for idx in idxs))} cannot be converted to internal coords'))

    atom_types = np.array([atom.atype for atom in internal_coords[0].atoms])
    atom_names = np.array([atom.name for atom in internal_coords[0].atoms])

    save_dict = {'rotlib': libname,
                 'resname': resname,
                 'coords': coords,
                 'internal_coords': internal_coords,
                 'weights': weights,
                 'atom_types': atom_types,
                 'atom_names': atom_names,
                 'dihedrals': dihedrals,
                 'dihedral_atoms': dihedral_atoms}

    if sigmas is None:
        pass
    elif sigmas.shape == dihedrals.shape:
        save_dict['sigmas'] = sigmas
    elif sigmas.shape == (*dihedrals.shape, 3):
        save_dict['sigmas'] = sigmas[..., 2]
        save_dict['locs'] = sigmas[..., 1]
        save_dict['skews'] = sigmas[..., 0]

    if spin_atoms:
        save_dict.update(spin_atoms)

    save_dict['type'] = 'chilife rotamer library'
    save_dict['format_version'] = 1.0

    return save_dict


def add_dihedral_def(name: str, dihedrals: ArrayLike, force: bool = False) -> None:
    """Helper function to add the dihedral definitions of user defined labels and libraries to the chilife knowledge
    base.

    Parameters
    ----------
    name : str
        Name of the residue.
    dihedrals : ArrayLike
        List of lists of atom names defining the dihedrals.
    force : bool
        Overwrite any dihedral definition with the same name if it exists.
    Returns
    -------
    None
    """

    # Reload in case there were other changes
    if not add_to_toml(DATA_DIR / "dihedral_defs.toml", key=name, value=dihedrals, force=force):
        raise ValueError(f'There is already a dihedral definition for {name}. Please choose a different name.' )

    # Add to active dihedral def dict
    chilife.dihedral_defs[name] = dihedrals


def remove_label(name, prompt=True):
    global USER_LABELS, USER_dLABELS
    if (name not in USER_LABELS) and (name not in USER_dLABELS) and prompt:
        raise ValueError(f'{name} is not in the set of user labels or user dLables. Check to make sure you have the '
                         f'right label. Note that only user labels can be removed.')

    if prompt:
        ans = input(f'WARNING: You have requested the permanent removal of the {name} label/rotamer library. \n'
                    f'Are you sure you want to do this? (y/n)')
        if ans.lower().startswith('y'):
            pass
        elif ans.lower().startswith('n'):
            print('Canceling label removal')
            return None
        else:
            print(f'"{ans}" is not an intelligible answer. Canceling label removal')
            return None

    # Remove files
    files = list((RL_DIR / 'user_rotlibs').glob(f'{name}*')) + \
            list((RL_DIR / 'residue_internal_coords').glob(f'{name}*')) + \
            list((RL_DIR / 'residue_pdbs').glob(f'{name}*'))

    for file in files:
        if file.exists():
            os.remove(str(file))

    if name in USER_dLABELS:
        USER_dLABELS.remove(name)

    if name in USER_LABELS:
        USER_LABELS.remove(name)

    if name in dihedral_defs:
        del dihedral_defs[name]

    remove_from_toml(DATA_DIR / 'dihedral_defs.toml', name)
    remove_from_defaults(name)


def add_to_toml(file, key, value, force=False):
    with open(file, 'r') as f:
        local = rtoml.load(f)

    if key in local and not force:
        return False

    if key is not None:
        local[key] = value

    with open(file, 'w') as f:
        rtoml.dump(local, f)

    return True


def remove_from_toml(file, entry):
    with open(file, 'r') as f:
        local = rtoml.load(f)

    if entry in local:
        del local[entry]

    with open(file, 'w') as f:
        rtoml.dump(local, f)

    return True


def add_to_defaults(resname, rotlibname, default=False):
    file = RL_DIR / 'defaults.toml'
    with open(file, 'r') as f:
        local = rtoml.load(f)

    if resname not in local:
        local[resname] = []
    if resname not in chilife.rotlib_defaults:
        chilife.rotlib_defaults[resname] = []

    pos = 0 if default else len(local[resname])
    chilife.rotlib_defaults[resname].insert(pos, rotlibname)
    local[resname].insert(pos, rotlibname)

    with open(file, 'w') as f:
        rtoml.dump(local, f)


def remove_from_defaults(rotlibname):
    file = RL_DIR / 'defaults.toml'
    with open(file, 'r') as f:
        local = rtoml.load(f)

    keys = [key for key, val in local.items() if rotlibname in val]

    for key in keys:
        chilife.rotlib_defaults[key].remove(rotlibname)
        local[key].remove(rotlibname)
        if local[key] == []:
            del local[key]

    with open(file, 'w') as f:
        rtoml.dump(local, f)