from __future__ import annotations
import tempfile, logging, pickle, os
from pathlib import Path
from itertools import combinations, product
from typing import Callable, Tuple, Union, List
from unittest import mock
from tqdm import tqdm

import numpy as np
from numpy.typing import ArrayLike
from numba import njit
from scipy.spatial.distance import cdist
import MDAnalysis as mda
import MDAnalysis.transformations

import chiLife
from .protein_utils import dihedral_defs, local_mx, sort_pdb, mutate, save_pdb, ProteinIC, get_min_topol
from .scoring import get_lj_rep, GAS_CONST
from .numba_utils import get_delta_r, normdist
from .SpinLabel import SpinLabel
from .RotamerEnsemble import RotamerEnsemble
from .SpinLabelTraj import SpinLabelTraj


# Define useful global variables
SUPPORTED_LABELS = ("R1M", "R7M", "V1M", "I1M", "M1M", "R1C")
SUPPORTED_BB_LABELS = ("R1C",)
DATA_DIR = Path(__file__).parent.absolute() / "data/"
RL_DIR = Path(__file__).parent.absolute() / "data/rotamer_libraries/"

logging.captureWarnings(True)

with open(RL_DIR / "spin_atoms.txt", "r") as f:
    lines = f.readlines()
    SPIN_ATOMS = {x.split(":")[0]: tuple(eval(a) for a in x.split(":")[1:]) for x in lines}

USER_LABELS = {key for key in SPIN_ATOMS if key not in SUPPORTED_LABELS}
USER_dLABELS = {f.name[:3] for f in (RL_DIR / "UserRotlibs").glob("*ip*.npz")}
SUPPORTED_RESIDUES = set(
    list(SUPPORTED_LABELS) + list(USER_LABELS) + list(dihedral_defs.keys())
)
[SUPPORTED_RESIDUES.remove(lab) for lab in ("CYR1", "MTN")]


def get_dd(
    *args: SpinLabel,
    r: ArrayLike = None,
    sigma: float = 1.0,
    spin_populations = False,
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
    spin_populations : bool
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

            Ps.append(pair_dd(*dummy_labels, r=r, sigma=sigma, spin_populations=spin_populations))
        Ps = np.array(Ps)
        return Ps

    else:

        P = pair_dd(*args, r=r, sigma=sigma, spin_populations=spin_populations)
        return P


def pair_dd(*args, r: ArrayLike, sigma: float = 1.0, spin_populations = False) -> np.ndarray:
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
    spin_populations : bool
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
        if spin_populations:
            coords1 = SL1.spin_coords.reshape(-1 ,3)
            coords2 = SL2.spin_coords.reshape(-1, 3)
            weights1 = np.outer(SL1.weights, SL1.spin_weights).flatten()
            weights2 = np.outer(SL2.weights, SL2.spin_weights).flatten()
        else:
            coords1 = SL1.spin_centers
            coords2 = SL2.spin_centers
            weights1 = SL1.weights
            weights2 = SL2.weights
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
    """Calculate a distance distribution from a trajectory of spin labels by calling ``get_dd`` on each frame and
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
        Additional keyword arguments to pass to ``get_dd`` .

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
        P += get_dd(_SL1, _SL2, r=r, sigma=sigma, **kwargs)

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


def add_label(
    name: str,
    pdb: str,
    dihedral_atoms: List[List[str]],
    resi: int = 1,
    spin_atoms: List[str] = None,
    dihedrals: ArrayLike = None,
    weights: ArrayLike = None,
    sigmas: ArrayLike = None
) -> None:
    """Add a user defined SpinLabel from a pdb file.

    Parameters
    ----------
    name : str
        Name for the user defined label. Should be a 3-letter residue code.
    pdb : str
        Name of (and path to) pdb file containing the user defined spin label structure. This pdb file should contain
        only the desired spin label and no additional residues.
    resi : int
        The residue number of the side chain in the pdb file you would like to add.
    dihedral_atoms : list
        List of rotatable dihedrals. List should contain sublists of 4 atom names. Atom names must be the same as
        defined in the pdb file eg:
        
        .. code-block:: python

            [['CA', 'CB', 'SG', 'SD'],
            ['CB', 'SG', 'SD', 'CE']...]

    spin_atoms : list
        List of atom names on which the spin density is localized.
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

    Returns
    -------
    None
    """
    struct = pre_add_label(name, pdb, spin_atoms)
    pdb_resname = struct.select_atoms(f"resnum {resi}").resnames[0]
    add_dihedral_def(name, dihedral_atoms)
    resi_selection = struct.select_atoms(f"resnum {resi}")
    bonds = resi_selection.intra_bonds.indices - resi_selection.atoms[0].ix

    # Convert loaded rotamer ensemble to internal coords
    internal_coords = [
        chiLife.get_internal_coords(
            resi_selection,
            preferred_dihedrals=dihedral_atoms,
            bonds=bonds
        )
        for ts in struct.trajectory
    ]

    # set resnum to 1 and remove chain operators so all rotamers are in the ic coordinate frame
    for ic in internal_coords:
        ic.shift_resnum(-(resi - 1))
        ic.chain_operators = None
        if len(ic.chains) > 1:
            raise ValueError('The PDB of the label supplied appears to have a chain break. Please check your PDB and '
                             'make sure there are no chain breaks in the desired label and that there are no other '
                             'chains in the pdb file. If the error persists, check to be sure all atoms are the correct '
                             'element as chiLife uses the elements to determine if atoms are bonded.')

    # Add internal_coords to data dir
    with open(RL_DIR / f"residue_internal_coords/{name}_ic.pkl", "wb") as f:
        pickle.dump(internal_coords, f)

    # If multi-state pdb extract rotamers from pdb
    if dihedrals is None:
        dihedrals = np.rad2deg(
            [ic.get_dihedral(1, dihedral_atoms) for ic in internal_coords]
        )

    if weights is None:
        weights = np.ones(len(dihedrals))
        weights /= weights.sum()

    store_new_restype(name, internal_coords, weights, dihedrals, dihedral_atoms, sigmas=sigmas)


def add_dlabel(
    name: str,
    pdb: str,
    increment: int,
    dihedral_atoms: List[List[List[str]]],
    resi: int = 1,
    spin_atoms: List[str] = None,
    dihedrals: ArrayLike = None,
    weights: ArrayLike = None,
) -> None:
    """Add a user defined dSpinLabel from a pdb file.

    Parameters
    ----------
    name : str
        Name for the user defined label. Should be a 3-letter code.
    increment : int
        The number of residues the second site away from the first site.
    pdb : str
        Name of (and path to) pdb file containing the user defined spin label structure. This pdb file should contain
        only the desired spin label and no additional residues.
    resi : int
        The residue number of the first side chain in the pdb file you would like to add.
    dihedral_atoms : list
        list of rotatable dihedrals. List should contain lists of 4 atom names. Atom names must be the same as defined
        in the pdb file eg:
        
        .. code-block:: python
        
            [['CA', 'CB', 'SG', 'SD'],
            ['CB', 'SG', 'SD', 'CE']...]

    spin_atoms : list
        List of atom names on which the spin density is localized.
    dihedrals : ArrayLike, optional
        Array of dihedral angles. If provided the new label object will be stored as a rotamer library with the
        dihedrals provided.
    weights : ArrayLike, optional
        Weights associated with the dihedral angles provided by the ``dihedrals`` keyword argument

    Returns
    -------
    None
    """
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
    global USER_dLABELS
    USER_dLABELS.add(name)
    add_dihedral_def(name, dihedral_atoms)
    struct = pre_add_label(name, pdb, spin_atoms, uniform_topology=False)
    pdb_resname = struct.select_atoms(f"resnum {resi}").resnames[0]

    IC1 = [
        chiLife.get_internal_coords(
            struct.select_atoms(f"resnum {resi}"),
            preferred_dihedrals=dihedral_atoms[0],
        )
        for ts in struct.trajectory
    ]

    IC2 = [
        chiLife.get_internal_coords(
            struct.select_atoms(f"resnum {resi + increment}"),
            preferred_dihedrals=dihedral_atoms[1],
        )
        for ts in struct.trajectory
    ]

    for ic1, ic2 in zip(IC1, IC2):
        ic1.shift_resnum(-(resi - 1))
        ic2.shift_resnum(-(resi + increment - 1))

    # Identify atoms that dont move with respect to each other but move with the dihedrals
    maxindex1 = (
        max([IC1[0].ICs[1][1][tuple(d[::-1])].index for d in dihedral_atoms[0]]) - 1
    )
    maxindex2 = (
        max([IC2[0].ICs[1][1][tuple(d[::-1])].index for d in dihedral_atoms[1]]) - 1
    )

    # Store 4 pairs between SLs as constriants
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

    csts = [cst_pairs, constraint_distances]

    # Add internal_coords to data dir
    for suffix, save_data in zip(["A", "B", "C"], [IC1, IC2, csts]):
        with open(
                RL_DIR / f"residue_internal_coords/{name}ip{increment}{suffix}_ic.pkl", "wb"
        ) as f:
            pickle.dump(save_data, f)

    # If multi-state pdb extract rotamers from pdb
    if dihedrals is None:
        dihedrals = []
        for IC, resnum, dihedral_set in zip(
            [IC1, IC2], [resi, resi + increment], dihedral_atoms
        ):
            dihedrals.append(
                [[ICi.get_dihedral(1, ddef) for ddef in dihedral_set] for ICi in IC]
            )

    if weights is None:
        weights = np.ones(len(IC1))

    weights /= weights.sum()
    store_new_restype(name + f'ip{increment}A', IC1, weights, dihedrals[0], dihedral_atoms[0])
    store_new_restype(name + f'ip{increment}B', IC2, weights, dihedrals[1], dihedral_atoms[1], resi=1 + increment)


def pre_add_label(name: str, pdb: str, spin_atoms: List[str], uniform_topology: bool = True) -> MDAnalysis.Universe:
    """Helper function to sort pdbs, save spin atoms, update lists, etc when adding a SpinLabel or dSpinLabel.

    Parameters
    ----------
    name : str
        Name of the label being added. Should be a 3-letter code.
    pdb : str
        Name (and path) of the pdb containing the new label.
    spin_atoms : List[str]
        Atoms of the SpinLabel where the unpaired electron is located.
    uniform_topology : bool
        Assume all rotamers of the library have the same topology (i.e. no differences in atom bonding). If false
        chiLife will attempt to find the minimal topology shared between all rotamers for defining internal coordinates.

    Returns
    -------
    struct : MDAnalysis.Universe
        MDAnalysis Universe object containing the rotamer ensemble with each rotamer as a frame. All atoms should be
        properly sorted for consistent construction of internal coordinates.
    """
    # Sort the PDB for optimal dihedral definitions
    pdb_lines = sort_pdb(pdb, uniform_topology=uniform_topology)
    bonds = get_min_topol(pdb_lines)

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


        with open(RL_DIR / "spin_atoms.txt", "r+") as f:
            lines = f.readlines()
            spin_dict = {x.split(":")[0]: tuple(eval(a) for a in x.split(":")[1:]) for x in lines}
            if name in spin_dict:
                if spin_dict[name] != (spin_atoms, spin_weights):
                    raise NameError(
                        "There is already a chiLife spin label with this name"
                    )
            else:
                joinstr = "', '"
                line = f"{name}: ['{joinstr.join(spin_atoms)}'] : [{', '.join(str(w) for w in spin_weights)}]\n"
                f.write(line)
                SPIN_ATOMS[name] = spin_atoms, spin_weights

    # Update USER_LABELS to include the new label
    global USER_LABELS
    USER_LABELS = {key for key in SPIN_ATOMS if key not in SUPPORTED_LABELS}

    # Write a temporary file with the sorted atoms
    if isinstance(pdb_lines[0], list):
        with tempfile.NamedTemporaryFile(
            suffix=".pdb", mode="w+", delete=False
        ) as tmpfile:
            for i, model in enumerate(pdb_lines):
                tmpfile.write(f"MODEL {i + 1}\n")
                for atom in model:
                    tmpfile.write(atom)
                tmpfile.write("ENDMDL\n")
    else:
        with tempfile.NamedTemporaryFile(
            suffix=".pdb", mode="w+", delete=False
        ) as tmpfile:
            for line in pdb_lines:
                tmpfile.write(line)

    # Load sorted atom pdb using MDAnalysis and remove tempfile
    struct = mda.Universe(tmpfile.name, in_memory=True)
    struct.universe.add_bonds(bonds)
    os.remove(tmpfile.name)
    return struct


def store_new_restype(
      name: str,
      internal_coords: List[ProteinIC],
      weights: ArrayLike,
      dihedrals: ArrayLike,
      dihedral_atoms: ArrayLike,
      sigmas: ArrayLike = None,
      resi: int = 1
) -> None:
    """Helper function to add new residue types to chiLife

    Parameters
    ----------
    name : str
        Name of residue to be stored.
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
    None
    """
    # Extract coordinates and transform to the local frame
    bb_atom_idx = [
        i for i, atom in enumerate(internal_coords[0].atoms) if atom.name in ["N", "CA", "C"]
    ]
    coords = internal_coords[0].coords.copy()
    ori, mx = local_mx(*coords[bb_atom_idx])
    coords = (coords - ori) @ mx

    # Save pdb structure
    save_pdb(RL_DIR / f"residue_pdbs/{name}.pdb", internal_coords[0].atoms, coords)

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

    save_dict = {'coords': coords,
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

    # Save rotamer library
    np.savez(
        RL_DIR / f"UserRotlibs/{name}_rotlib.npz",
        **save_dict,
        allow_pickle=True,
    )


def add_dihedral_def(name: str, dihedrals: ArrayLike) -> None:
    """Helper function to add the dihedral definitions of user defined labels and libraries to the chiLife knowledge
    base.

    Parameters
    ----------
    name : str
        Name of the residue.
    dihedrals : ArrayLike
        List of lists of atom names defining the dihedrals.

    Returns
    -------
    None
    """

    # Reload in case there were other changes
    with open(DATA_DIR / "DihedralDefs.pkl", "rb") as f:
        local_dihedral_def = pickle.load(f)

    # Add new label defs and write file
    local_dihedral_def[name] = dihedrals
    with open(DATA_DIR / "DihedralDefs.pkl", "wb") as f:
        pickle.dump(local_dihedral_def, f)

    # Add to active dihedral def dict
    chiLife.dihedral_defs[name] = dihedrals


def remove_label(name, prompt=True):
    global USER_LABELS, USER_dLABELS
    if (name not in USER_LABELS) and (name not in USER_dLABELS):
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

    files = list((RL_DIR / 'UserRotlibs').glob(f'{name}*')) + \
            list((RL_DIR / 'residue_internal_coords').glob(f'{name}*')) + \
            list((RL_DIR / 'residue_pdbs').glob(f'{name}*'))

    for file in files:
        os.remove(str(file))

    if name in USER_dLABELS:
        USER_dLABELS.remove(name)

    if name in USER_LABELS:
        USER_LABELS.remove(name)

    if name in SPIN_ATOMS:
        del SPIN_ATOMS[name]

    del dihedral_defs[name]
    with open(DATA_DIR / 'DihedralDefs.pkl', 'wb') as f:
        pickle.dump(dihedral_defs, f)

    with open(RL_DIR / 'spin_atoms.txt', 'w') as f:
        joinstr = "', '"
        for n, (spin_atoms, spin_weights) in SPIN_ATOMS.items():
            line = f"{n}: ['{joinstr.join(spin_atoms)}'] : [{', '.join(str(w) for w in spin_weights)}]\n"
            f.write(line)

