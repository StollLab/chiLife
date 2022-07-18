import numbers, shutil, tempfile,  math, logging, pickle, os
from pathlib import Path
from collections.abc import Sized
from io import StringIO
from itertools import combinations, product
from typing import Callable, Tuple, Union, List
from unittest import mock

from memoization import cached
from tqdm import tqdm

import numpy as np
from numpy.typing import ArrayLike
from numba import njit

from scipy.signal import fftconvolve
from scipy.stats import gaussian_kde
from scipy.spatial.distance import cdist

import MDAnalysis as mda
import MDAnalysis.core.topologyattrs
import MDAnalysis.transformations

import chiLife
from .protein_utils import dihedral_defs, rotlib_indexes, local_mx, sort_pdb, mutate, save_pdb
from .scoring import get_lj_rep, GAS_CONST
from .numba_utils import get_delta_r, histogram, norm
from .SpinLabel import SpinLabel, dSpinLabel
from .RotamerLibrary import RotamerLibrary
from .SpinLabelTraj import SpinLabelTraj


# Define useful global variables
SUPPORTED_LABELS = ("R1M", "R7M", "V1M", "I1M", "M1M", "R1C")
SUPPORTED_BB_LABELS = ("R1C",)
DATA_DIR = Path(__file__).parent.absolute() / "data/rotamer_libraries/"
logging.captureWarnings(True)

with open(DATA_DIR / "spin_atoms.txt", "r") as f:
    lines = f.readlines()
    SPIN_ATOMS = {x.split(":")[0]: eval(x.split(":")[1]) for x in lines}

USER_LABELS = {key for key in SPIN_ATOMS if key not in SUPPORTED_LABELS}
USER_dLABELS = {f.name[:3] for f in (DATA_DIR / "UserRotlibs").glob("*i_*.npz")}
SUPPORTED_RESIDUES = set(
    list(SUPPORTED_LABELS) + list(USER_LABELS) + list(dihedral_defs.keys())
)
[SUPPORTED_RESIDUES.remove(lab) for lab in ("CYR1", "MTN")]


def read_distance_distribution(file_name: str) -> Tuple[ArrayLike, ArrayLike]:
    """
    Reads a DEER distance distribution file in the DeerAnalysis format.

    :param file_name: str
        File name of distance distribution

    :return r, p: (numpy ndarray, numpy ndarray)
        r: domain of distance distribution squared
        p: normalized probability density over r.
    """

    # Load DA file
    data = np.loadtxt(file_name)

    # Convert nm to angstroms
    r = data[:, 0] * 10

    # Extract distance domain coordinates
    p = data[:, 1]
    return r, p


def get_dd(
    *args,
    r: ArrayLike = (0, 100),
    sigma: float = 1.0,
    prune: bool = False,
    uq: bool = False,
    **kwargs,
) -> ArrayLike:
    """
    Wrapper function to calculate distance distribution using arbitrary function and arbitrary inputs

    :param *args:
        Variable length list of SpinLabel objects

    :param r: ndarray, tuple float
        ndarray: evenly spaced array containing the distance domain coordinates to calculate the distance distribution over
        tuple:  2 numbers bounding the desired distance domain
        number: A float/int to mark the  upper bound of the distance domain

    :param sigma: float
        The standard deviation of the normal distribution used in convolution with the distance histogram

    :param prune: bool, float
        Filter insignificant rotamer pairs from the distance distribution calculation.
        If filter is a float the bottom percentile to be filtered out.

    :param uq: bool, int
        Perform uncertainty quantification by subsampling rotamer libraries used to calculate the distance distribution.
        if uq is an int than that will be the number of subsamples defaulting to 100.

    :Keyword Arguments:
        * *size* (``int``) --
          Number of points in the distance domain

    :return P: ndarray
        The probability density of the distance distribution corresponding to r
    """

    # Allow r to be passed as that last non-keyword argument
    if not isinstance(args[-1], (SpinLabel, SpinLabelTraj, mock.Mock)):
        r = args[-1]
        args = args[:-1]

    if any(not hasattr(arg, atr) for arg in args for atr in ["spin_coords", "weights"]):
        raise TypeError(
            "Arguments other than spin labels must be passed as a keyword argument"
        )

    size = kwargs.get("size", 1024)
    if isinstance(r, numbers.Number):
        r = np.linspace(0, r, size)

    elif isinstance(r, Sized):
        if len(r) == 2:
            r = np.linspace(*r, size)
        elif len(r) == 3:
            r = np.linspace(*r)
        else:
            r = np.asarray(r)

    if any(isinstance(SL, SpinLabelTraj) for SL in args):
        return traj_dd(*args, r=r, sigma=sigma, prune=prune, **kwargs)

    if uq:
        if prune:
            raise ValueError(
                "Pruning is not supported when performing uncertainty analysis (yet)"
            )
        Ps = []
        n_boots = uq if uq > 1 else 100
        for i in range(n_boots):
            dummy_labels = []
            for SL in args:
                idxs = np.random.choice(len(SL), len(SL))

                dummy_SL = mock.Mock()
                dummy_SL.spin_coords = np.atleast_2d(SL.spin_coords[idxs])
                dummy_SL.weights = SL.weights[idxs]
                dummy_SL.weights /= dummy_SL.weights.sum()
                dummy_labels.append(dummy_SL)

            Ps.append(get_dd(*dummy_labels, r=r, sigma=sigma, prune=prune, **kwargs))
        Ps = np.array(Ps)
        return Ps

    if prune:
        if len(args) != 2:
            raise IndexError(
                "Pruned distance distributions are only supported when using two spin labels (for now)."
            )

        SL1, SL2 = args

        # Convert prune percentile to a fraction
        if isinstance(prune, bool):
            wts, idx = filter_by_weight(SL1.weights, SL2.weights)

        else:
            prune /= 100
            wts, idx = filter_by_weight(SL1.weights, SL2.weights, prune)

        NO1 = SL1.spin_coords[idx[:, 0]]
        NO2 = SL2.spin_coords[idx[:, 1]]
        P = filtered_dd(NO1, NO2, wts, r, sigma=sigma)

    else:
        P = unfiltered_dd(*args, r=r, sigma=sigma)

    return P


def unfiltered_dd(*args, r: ArrayLike, sigma: float = 1.0) -> ArrayLike:
    """
    Obtain the pairwise distance distribution from two rotamer libraries, NO1, NO2 with corresponding weights w1, w2.
    The distribution is calculated by convolving the weighted histogram of pairwise distances between NO1 and NO2 with
    a normal distribution of sigma.

    :param NO1, NO2: ndarray
        Electron coordinates of each rotamer library

    :param w1, w2: ndarray
        rotamer weights for each rotamer library

    :param r: ndarray
        Domain to compute distance distribution over

    :param sigma: float
    standard deviation of normal distribution used for convolution

    :return P: ndarray
        x and y coordinates of normalized distance distribution

    """
    # Calculate pairwise distances and weights
    SL_Pairs = combinations(args, 2)
    weights, distances = [], []

    for SL1, SL2 in SL_Pairs:
        distances.append(cdist(SL1.spin_coords, SL2.spin_coords).flatten())
        weights.append(np.outer(SL1.weights, SL2.weights).flatten())

    distances = np.concatenate(distances)
    weights = np.concatenate(weights)

    # Calculate histogram over x
    hist, _ = np.histogram(
        distances, weights=weights, range=(min(r), max(r)), bins=len(r)
    )
    if sigma != 0:
        # Calculate normal distribution for convolution
        delta_r = get_delta_r(r)
        _, g = norm(delta_r, 0, sigma)

        # Convolve normal distribution and histogram
        P = fftconvolve(hist, g, mode="same")

    else:
        P = hist

    # Clip values less than 0 (numerical artifacts)
    P = P.clip(0)

    # Normalize weights
    P /= np.trapz(P, r)

    return P


def filter_by_weight(
    w1: ArrayLike, w2: ArrayLike, cutoff: float = 0.001
) -> Tuple[ArrayLike, ArrayLike]:
    """
    Pre-calculates weights for each rotamer pair and returns a weight vector for corresponding to the weights for each
    significant pair and their coordinate indices.

    :param w1, w2: ndarray
        Rotamer weights in the order the rotamers appear in the rotamer library.

    :param cutoff: float
        Cutoff for significant rotamer pair weights. If the weight of the rotamer pair is is less than
        max(weight)*cutoff it will be excluded

    :return weights, idx:
        weights: weights vector for significant pairs
        idx: array of coordinate pair indices for significant pairs
    """
    # Calculate pairwise weights
    weights = np.outer(w1, w2)

    # Make index vector of significant weights
    idx = np.argwhere(weights > weights.sum() * cutoff)

    # Make weights vector of significant weights
    weights = weights[weights > weights.sum() * cutoff]

    return weights, idx


def filtered_dd(
    NO1: ArrayLike, NO2: ArrayLike, weights: ArrayLike, r: ArrayLike, sigma: float = 1.0
) -> ArrayLike:
    """
    Calculates the distance distribution for two sets of NO midpoints.

    :param NO1, NO2: ndarray
        coordinates of spin label's NO midpoints

    :param weights: ndarray
        Weights corresponding to each individual rotamer in NO1 and NO2. Weights should be the same between NO1 and NO2
        since they should be the same rotamer library and no clashes considered.

    :param r: numpy ndarray
        Domain to compute distance distribution over

    :param sigma: float
        standard deviation of normal distribution used for convolution

    :return: P:
        Distance distribution
    """
    # Compute distances and weights
    difference = NO1 - NO2
    distances = difference * difference
    distances = np.sqrt(distances.sum(axis=1))

    delta_r = get_delta_r(r)

    # Compute histogram to convolve with normal distribution
    hist = histogram(distances, weights, r)

    # Check if histogram has significant density at 0
    if hist[0] > 0.05 * hist.sum():
        hist[:] = 0

    # Set 0 distance to 0 density
    hist[0] = 0.0

    # Create normal distribution to convolve with histogram
    _, g = norm(delta_r, 0, sigma)

    # Determine indices of convolution output corresponding to r
    begin = int(len(g) / 2)
    end = begin
    if len(g) % 2 == 0:
        end -= 1

    # Compute distance distribution by convolution
    P = np.convolve(hist, g)

    # Trim ends created by convolution
    P = P[begin:-end]

    # Clip values less than 0 (numerical artifacts)
    P = P.clip(0)

    # Normalize weights
    P /= np.trapz(P, r)

    return P


def traj_dd(
    SL1: SpinLabelTraj,
    SL2: SpinLabelTraj,
    r: ArrayLike,
    sigma: float,
    filter: Union[bool, float],
    **kwargs,
) -> ArrayLike:
    """
    Calculate a distance distribution from a trajectory of spin labels

    :param SL1, SL2: SpinLabelTrajectory
        Spin label to use for distance distribution calculation.

    :param r: ndarray
        Distance domain to use when calculating distance distribution.

    :param sigma: float
        Standard deviation of the gaussian kernel used to smooth the distance distribution

    :param filter: float, bool
        Option to prune out negligible population rotamer pairs from distance distribution calculation. The fraction
        ommited can be specified by assigning a float to `prune`

    :param kwargs: dict
        Additional keyword arguments.

    :return P: ndarray
        Disctance distribution calculated from the provided SpinLabelTrajectories
    """

    # Ensure that the SpinLabelTrajectories have the same numebr of frames.
    if len(SL1) != len(SL2):
        raise ValueError("SpinLabelTraj objects must have the same length")

    # Calculate the distance distribution for each frame and sum
    P = np.zeros_like(r)
    for _SL1, _SL2 in zip(SL1, SL2):
        P += get_dd(_SL1, _SL2, r=r, sigma=sigma, prune=filter, **kwargs)

    # Normalize distance distribution
    P /= np.trapz(P, r)

    return P


@cached
def read_sl_library(label: str, user: bool = False) -> Tuple[ArrayLike, ...]:
    """
    Reads RotamerLibrary for stored spin labels.

    :param label: str
        3 character abbreviation for desired spin label

    :param user: bool
        Specifies if the library was defined by a user or if it is a precalculated library

    :return coords, (internal_coords), weights, atom_types, atom_names: ndarray
        Arrays of spin label coordinates, weights, atom types and atom names in the local coordinate frame. If
        internal_coord information is available it will be returned in between coords and weights.
    """
    subdir = "UserRotlibs/" if user else "MMM_RotLibs/"
    data = Path(__file__).parent / "data/rotamer_libraries/"
    with np.load(data / subdir / (label + "_rotlib.npz"), allow_pickle=True) as files:
        lib = dict(files)

    del lib["allow_pickle"]

    with open(chiLife.DATA_DIR / f"residue_internal_coords/{label}_ic.pkl", "rb") as f:
        IC = pickle.load(f)
        if isinstance(IC, list):
            ICn = IC
        else:
            ICn = [
                IC.copy().set_dihedral(
                    np.deg2rad(r), 1, atom_list=lib["dihedral_atoms"]
                )
                for r in lib["dihedrals"]
            ]

    lib["internal_coords"] = ICn

    if "sigmas" not in lib:
        lib["sigmas"] = np.array([])

    lib["_rdihedrals"] = np.deg2rad(lib["dihedrals"])
    lib["_rsigmas"] = np.deg2rad(lib["sigmas"])

    return lib


@cached
def read_bbdep(res: str, Phi: float, Psi: float) -> Tuple[ArrayLike, ...]:
    """
    Read the dunbrack rotamer library for for the provided residue and backbone conformation.

    :param res: str
        3 letter residue code

    :param Phi: float
        Protein backbone Phi dihedral angle for the provided residue

    :param Psi: float
        Protein backbone Psi dihedral angle for the provided residue

    :return coords, internal_coords, weights, atom_types, atom_names: numpy.ndarray
        Numpy arrays containing all rotamer library information in cartesian and dihedral space
    """
    lib = {}
    Phi, Psi = str(Phi), str(Psi)

    # Read residue internal coordinate structure
    with open(DATA_DIR / f"residue_internal_coords/{res.lower()}_ic.pkl", "rb") as f:
        ICs = pickle.load(f)

    atom_types = ICs.atom_types.copy()
    atom_names = ICs.atom_names.copy()

    maxchi = 5 if res in SUPPORTED_BB_LABELS else 4
    nchi = np.minimum(len(dihedral_defs[res]), maxchi)

    if res not in ("ALA", "GLY"):
        library = "R1C.lib" if res in SUPPORTED_BB_LABELS else "ALL.bbdep.rotamers.lib"
        start, length = rotlib_indexes[f"{res}  {Phi:>4}{Psi:>5}"]

        with open(DATA_DIR / library, "rb") as f:
            f.seek(start)
            rotlib_string = f.read(length).decode()
            s = StringIO(rotlib_string)
            s.seek(0)
            data = np.genfromtxt(s, usecols=range(maxchi + 4, maxchi + 5 + 2 * maxchi))

        lib["weights"] = data[:, 0]
        lib["dihedrals"] = data[:, 1 : nchi + 1]
        lib["sigmas"] = data[:, maxchi + 1 : maxchi + nchi + 1]
        dihedral_atoms = dihedral_defs[res][:nchi]

        # Calculate cartesian coordinates for each rotamer
        coords = []
        internal_coords = []
        for r in lib["dihedrals"]:
            ICn = ICs.copy().set_dihedral(np.deg2rad(r), 1, atom_list=dihedral_atoms)

            coords.append(ICn.to_cartesian())
            internal_coords.append(ICn)

    else:
        lib["weights"] = np.array([1])
        lib["dihedrals"], lib["sigmas"], dihedral_atoms = [], [], []
        coords = [ICs.to_cartesian()]
        internal_coords = [ICs.copy()]

    # Get origin and rotation matrix of local frame
    mask = np.in1d(atom_names, ["N", "CA", "C"])
    ori, mx = local_mx(*coords[0][mask])

    # Set coords in local frame and prepare output
    lib["coords"] = np.array([(coord - ori) @ mx for coord in coords])
    lib["internal_coords"] = internal_coords
    lib["atom_types"] = np.asarray(atom_types)
    lib["atom_names"] = np.asarray(atom_names)
    lib["dihedral_atoms"] = np.asarray(dihedral_atoms)
    lib["_rdihedrals"] = np.deg2rad(lib["dihedrals"])
    lib["_rsigmas"] = np.deg2rad(lib["sigmas"])

    return lib


def read_library(
    res: str, Phi: float = None, Psi: float = None
) -> Tuple[ArrayLike, ...]:
    """
    Generalized wrapper function to aid selection of rotamer library reading function.

    :param res: str
        3 letter residue code

    :param Phi: float
        Protein backbone Phi dihedral angle for the provided residue

    :param Psi: float
        Protein backbone Psi dihedral angle for the provided residue

    :return coords, weights, atom_types, atom_names: numpy.ndarray
        Numpy arrays containing all rotamer library information in cartesian space
    """
    backbone_exists = Phi and Psi

    if backbone_exists:
        Phi = int((Phi // 10) * 10)
        Psi = int((Psi // 10) * 10)

    if res in SUPPORTED_LABELS and res not in SUPPORTED_BB_LABELS:
        return read_sl_library(res)
    elif res in USER_LABELS or res[:3] in USER_dLABELS:
        return read_sl_library(res, user=True)
    elif backbone_exists:
        return read_bbdep(res, Phi, Psi)
    else:
        return read_bbdep(res, -60, -50)


def get_site(site: Union[int, str], label: str) -> Tuple[int, str]:
    """
    Function to obtain site information from a (user provided) string

    :param site: int, str
        residue number and possibly chain and label of site

    :param label: str
         default label if none is provided by site

    :return resi, chain: int, str
        The residue number and chain identifier of the provided site
    """
    chain = None
    if isinstance(site, str):
        # Check for explicit label
        if site.endswith(SUPPORTED_LABELS):
            label = site[-3:]
            site = site[:-3]

        # Extract Chain if it exists
        chain = site.strip("0123456789")

    if chain:
        resi = int(site[len(chain):])
    else:
        resi = int(site)
        chain = "A"

    return resi, chain, label


@njit(cache=True)
def optimize_weights(
    ensemble: ArrayLike,
    idx: ArrayLike,
    start_weights: ArrayLike,
    start_score: float,
    data: ArrayLike,
) -> Tuple[ArrayLike, ArrayLike, ArrayLike, float]:
    """
    Fit weights to an ensemble of distance distributions optimizing the score with respect to user defined data.

    :param ensemble: numpy ndarray
        array of distance distributions for each structure and each site pair in the ensemble

    :param idx: numpy ndarray
        indices of structures in parent matrix of structures

    :param start_weights: numpy ndarray
        initial values of weights

    :param start_score: float
        score of ensemble to data with initial weight values

    :param data: numpy ndarray
        experimental data to fit simulation to

    :return ensemble, idx, best_weights, best_score: numpy ndarray, numpy ndarray, numpy ndarray, float
        returns array of distance distributions for the optimized ensemble, their indicates in the parent array, the
        corresponding weights and the score that accompanies the fitted ensemble.
    """
    best_score = start_score
    best_weights = start_weights.copy()

    count = 0
    while count < 100:

        # Assign new weights from dirichlet distribution of best weights
        new_weights = dirichlet(best_weights)
        new_score = jaccard(np.dot(ensemble, new_weights), data)

        # Keep score if improved
        if new_score > best_score:
            best_score = new_score

            # Drop all structures present at less than 0.1%
            ensemble = ensemble[:, new_weights > 1e-3]
            idx = idx[new_weights > 1e-3]
            best_weights = new_weights[new_weights > 1e-3]

            # Rest count
            count = 0

        else:
            count += 1

    return ensemble, idx, best_weights, best_score


def save(
    file_name: str,
    *labels: SpinLabel,
    protein: Union[mda.Universe, mda.AtomGroup, str] = None,
    **kwargs,
) -> None:
    """
    Save a pdb file of the provided labels

    :param file_name: str
        Desired file name for output file. Will be automatically made based off of protein name and labels if not
        provided.

    :param labels: SpinLabel(s)
        SpinLabel object(s) to save. Can add as many as desired.

    :param protein: str
        File name of protein structure file SpinLabels were attached to.

    :return: None
        Writes a PDB file in the current directory
    """
    # Check for filename at the beginning of args
    labels = list(labels)
    if isinstance(file_name, (SpinLabel, dSpinLabel)):
        labels.insert(0, file_name)
        file_name = None
    elif hasattr(file_name, "atoms"):
        labels.insert(-1, file_name)
        file_name = None

    # Check for protein structures at the end of args
    if protein is None:
        if (
            isinstance(labels[-1], str)
            or isinstance(labels[-1], mda.Universe)
            or isinstance(labels[-1], MDAnalysis.AtomGroup)
        ):
            protein = labels.pop(-1)

    # Create a file name from spin label and protein information
    if file_name is None:
        if isinstance(protein, str):
            f = Path(protein)
            file_name = f.name[:-4]
        elif getattr(protein, "filename", None) is not None:
            file_name = protein.filename[:-4]
        else:
            file_name = "No_Name_Protein"

        if 0 < len(labels) < 3:
            for label in labels:
                file_name += f"_{label.site}{label.res}"
        else:
            file_name += "_many_labels"
        file_name += ".pdb"

    if protein is not None:
        if isinstance(protein, str):
            print(protein, file_name)
            shutil.copy(protein, file_name)
        elif isinstance(protein, mda.Universe) or isinstance(
            protein, MDAnalysis.AtomGroup
        ):
            write_protein(file_name, protein)
        else:
            raise TypeError(
                "`protein` must be a string or an MDAnalysis Universe/AtomGroup object"
            )

    if len(labels) > 0:
        write_labels(file_name, *labels, **kwargs)


def write_protein(
    file: str, protein: Union[mda.Universe, mda.AtomGroup], **kwargs
) -> None:

    fmt_str = "ATOM  {:5d} {:^4s} {:3s} {:1s}{:4d}    {:8.3f}{:8.3f}{:8.3f}{:6.2f}{:6.2f}          {:>2s}  \n"
    with open(file, "w") as f:
        f.write(f'HEADER {file.rstrip(".pdb")}\n')
        for mdl, ts in enumerate(protein.universe.trajectory):
            f.write(f"MODEL {mdl}\n")
            [
                f.write(
                    fmt_str.format(
                        atom.index,
                        atom.name,
                        atom.resname[:3],
                        atom.segid,
                        atom.resnum,
                        *atom.position,
                        1.00,
                        1.0,
                        atom.type,
                    )
                )
                for atom in protein.atoms
            ]
            f.write("TER\n")
            f.write("ENDMDL\n")


def write_labels(file: str, *args: SpinLabel, KDE: bool = True, **kwargs) -> None:
    """
    Lower level helper function for save. Loops over SpinLabel objects and appends atoms and electron coordinates to the
    provided file.

    :param file: string
        File name to write to.

    :param *args: SpinLabel(s)
        SpinLabel objects to write to file

    :param KDE: bool
        Switch to perform Kernel Density Estimate (KDE) on spin label weights to produce a smooth surface to
        visualize pseudo-spin density with b-factor
    """

    # Check for dSpinLables
    rotlibs = []
    for arg in args:
        if isinstance(arg, chiLife.RotamerLibrary):
            rotlibs.append(arg)
        elif isinstance(arg, chiLife.dSpinLabel):
            rotlibs.append(arg.SL1)
            rotlibs.append(arg.SL2)
        else:
            raise TypeError(
                f"Cannot save {arg}. *args must be RotamerLibrary SpinLabel or dSpinLabal objects"
            )

    fmt_str = "ATOM  {:5d} {:^4s} {:3s} {:1s}{:4d}    {:8.3f}{:8.3f}{:8.3f}{:6.2f}{:6.2f}          {:>2s}  \n"
    with open(file, "a+", newline="\n") as f:
        # Write spin label models
        f.write("\n")
        for k, label in enumerate(rotlibs):
            f.write(f"HEADER {label.name}\n")

            # Save models in order of weight
            sorted_index = np.argsort(label.weights)[::-1]
            for mdl, (conformer, weight) in enumerate(
                zip(label.coords[sorted_index], label.weights[sorted_index])
            ):
                f.write("MODEL {}\n".format(mdl))

                [
                    f.write(
                        fmt_str.format(
                            i,
                            label.atom_names[i],
                            label.res[:3],
                            label.chain,
                            int(label.site),
                            *conformer[i],
                            1.00,
                            weight * 100,
                            label.atom_types[i],
                        )
                    )
                    for i in range(len(label.atom_names))
                ]
                f.write("TER\n")
                f.write("ENDMDL\n")

        # Write electron density at electron coordinates
        for k, label in enumerate(rotlibs):
            if not hasattr(label, "spin_coords"):
                continue
            if np.any(np.isnan(label.spin_coords)):
                continue

            f.write(f"HEADER {label.name}_density\n".format(label.label, k + 1))
            NO = np.atleast_2d(label.spin_coords)

            if KDE and np.all(np.linalg.eigh(np.cov(NO.T))[0] > 0) and len(NO) > 5:
                # Perform gaussian KDE to determine electron density
                gkde = gaussian_kde(NO.T, weights=label.weights)

                # Map KDE density to pseudoatoms
                vals = gkde.pdf(NO.T)
            else:
                vals = label.weights

            [
                f.write(
                    fmt_str.format(
                        i,
                        "NEN",
                        label.label[:3],
                        label.chain,
                        int(label.site),
                        *NO[i],
                        1.00,
                        vals[i] * 100,
                        "N",
                    )
                )
                for i in range(len(vals))
            ]

            f.write("TER\n")


def repack(
    protein: Union[mda.Universe, mda.AtomGroup],
    *spin_labels: SpinLabel,
    repetitions: int = 200,
    temp: float = 1,
    energy_func: Callable = get_lj_rep,
    **kwargs,
) -> Tuple[mda.Universe, ArrayLike, Tuple[SpinLabel, ...]]:
    """
    Given a protein and a SpinLabel object, repack the local environment using monte carlo sampling.

    :param protein: MDAnalysis.Universe or MDAnalysis.AtomGroup
        Protein to be repacked

    :param spin_labels: SpinLabel
        SpinLabel placed at site of interest.

    :param repetitions: int
        Number of successful MC samples to perform before terminating the MC sampling loop

    :param temp: float
        Temperature (Kelvin) for both clash evaluation and metropolis-hastings acceptance criteria.

    :param energy_func: function
        Energy function to be used for clash evaluation

    :param kwargs: dict
        Additional keyword arguments to be passed to downstream functions.

    :return opt_protein, deltaEs, SLs: MDAnalysis.Universe, ndarray, tuple
        opt_protein: MCMC trajectory of local repack
        deltaEs: Change in energy_func score at each accept of MCMC trajectory
        SLs: tuple of spin label objects attached to the lowest energy structure of the trajectory for each input label
    """
    temp = np.atleast_1d(temp)
    KT = {t: GAS_CONST * t for t in temp}

    repack_radius = kwargs.get("repack_radius", 10)  # Angstroms

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
        RotamerLibrary.from_mda(res, **repack_res_kwargs)
        for res in repack_residues
        if res.resname not in ["GLY", "ALA"]
    ]

    # Create new labeled protein construct to fill in any missing atoms of repack residues
    protein = mutate(protein, *repack_residue_libraries, **kwargs).atoms

    repack_residues = protein.select_atoms(
        f"(around {repack_radius} {spin_label_str} ) " f"or {spin_label_str}"
    ).residues

    repack_residue_libraries = [
        RotamerLibrary.from_mda(res, **repack_res_kwargs)
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

            coords, weight = SiteLibrary.sample(
                off_rotamer=kwargs.get("off_rotamer", False)
            )

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
    dihedral_atoms: List[str],
    resi: int = 1,
    spin_atoms: List[str] = None,
    dihedrals: ArrayLike = None,
    weights: ArrayLike = None,
    sigmas: ArrayLike = None
):
    """
    Add a user defined SpinLabel from a pdb file.

    :param name: str
        Name for the user defined label.

    :param pdb: str
        Name of (and path to) pdb file containing the user defined spin label structure. This pdb file should contain
        only the desired spin label and no additional residues.

    :param dihedral_atoms: list
        list of rotatable dihedrals. List should contain lists of 4 atom names. Atom names must be the same as defined
        in the pdb file eg:
        [ ['CA', 'CB', 'SG', 'SD'],
          ['CB', 'SG', 'SD', 'CE']...]

    :param spin_atoms: list
        List of atom names on which the spin density is localized.

    :param dihedrals: ndarray, optional
        Array of dihedral angles. If provided the new label object will be stored as a rotamer library with the
        dihedrals provided.

    :param weights: ndarray, optional
        Weights associated with the dihedral angles provided by the `dihedrals` keyword argument

    :param sigmas: ndarray, optional
        Sigma paramter for disributions of dihedral angles.

    :param skews: ndarray, optional
        Skew parameter for disributions of dihedral angles.
    """
    struct = pre_add_label(name, pdb, spin_atoms)
    pdb_resname = struct.select_atoms(f"resnum {resi}").resnames[0]
    add_dihedral_def(name, dihedral_atoms)

    # Convert loaded rotamer library to internal coords
    internal_coords = [
        chiLife.get_internal_coords(
            struct.select_atoms(f"resnum {resi}"),
            resname=pdb_resname,
            preferred_dihedrals=dihedral_atoms,
        )
        for ts in struct.trajectory
    ]

    # Remove chain operators so all rotamers are in the ic coordinate frame
    for ic in internal_coords:
        ic.chain_operators = None

    # Add internal_coords to data dir
    with open(DATA_DIR / f"residue_internal_coords/{name}_ic.pkl", "wb") as f:
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
):
    """
    Add a user defined SpinLabel from a pdb file.

    :param name: str
        Name for the user defined label.

    :param increment: int
        The number of residues the second site away from the first site.

    :param pdb: str
        Name of (and path to) pdb file containing the user defined spin label structure. This pdb file should contain
        only the desired spin label and no additional residues.

    :param dihedral_atoms: list
        list of rotatable dihedrals. List should contain lists of 4 atom names. Atom names must be the same as defined
        in the pdb file eg:
        [ ['CA', 'CB', 'SG', 'SD'],
          ['CB', 'SG', 'SD', 'CE']...]

    :param spin_atoms: list
        List of atom names on which the spin density is localized.

    :param dihedrals: ndarray, optional
        Array of dihedral angles. If provided the new label object will be stored as a rotamer library with the
        dihedrals provided.

    :param weights: ndarray, optional
        Weights associated with the dihedral angles provided by the `dihedrals` keyword argument
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

    struct = pre_add_label(name, pdb, spin_atoms)
    pdb_resname = struct.select_atoms(f"resnum {resi}").resnames[0]

    IC1 = [
        chiLife.get_internal_coords(
            struct.select_atoms(f"resnum {resi}"),
            resname=pdb_resname,
            preferred_dihedrals=dihedral_atoms[0],
        )
        for ts in struct.trajectory
    ]

    IC2 = [
        chiLife.get_internal_coords(
            struct.select_atoms(f"resnum {resi + increment}"),
            resname=pdb_resname,
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
    for suffix, save_data in zip(["", "i", f"ip{increment}"], [csts, IC1, IC2]):
        with open(
            DATA_DIR / f"residue_internal_coords/{name}{suffix}_ic.pkl", "wb"
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
    store_new_restype(name, IC1, weights, dihedrals[0], dihedral_atoms[0], increment=0)
    store_new_restype(
        name, IC2, weights, dihedrals[1], dihedral_atoms[1], increment=increment
    )


def pre_add_label(name, pdb, spin_atoms):
    # Sort the PDB for optimal dihedral definitions
    pdb_lines = sort_pdb(pdb)

    # Store spin atoms if provided
    if spin_atoms is not None:
        if isinstance(spin_atoms, str):
            spin_atoms = spin_atoms.split()

        with open(DATA_DIR / "spin_atoms.txt", "r+") as f:
            lines = f.readlines()
            spin_dict = {x.split(":")[0]: eval(x.split(":")[1]) for x in lines}
            if name in spin_dict:
                if spin_dict[name] != spin_atoms:
                    raise NameError(
                        "There is already a chiLife spin label with this name"
                    )
            else:
                joinstr = "', '"
                line = f"{name}: ['{joinstr.join(spin_atoms)}']\n"
                f.write(line)
                SPIN_ATOMS[name] = spin_atoms

    # Update USER_LABELS to include the new label
    global USER_LABELS
    USER_LABELS = tuple(key for key in SPIN_ATOMS if key not in SUPPORTED_LABELS)

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
    os.remove(tmpfile.name)
    return struct


def store_new_restype(name, internal_coords, weights, dihedrals, dihedral_atoms, sigmas=None, increment=None):
    # Extract coordinates and transform to the local frame
    bb_atom_idx = [
        i for i, atom in enumerate(internal_coords[0].atoms) if atom.name in ["N", "CA", "C"]
    ]
    coords = internal_coords[0].coords.copy()
    ori, mx = local_mx(*coords[bb_atom_idx])
    coords = (coords - ori) @ mx

    if increment == 0:
        name += "i"
    elif increment is None:
        increment = 0

    if increment > 0:
        name += f"ip{increment}"

    # Save pdb structure
    save_pdb(DATA_DIR / f"residue_pdbs/{name}.pdb", internal_coords[0].atoms, coords)

    if len(internal_coords) > 1:
        coords = np.array([(IC.coords - ori) @ mx for IC in internal_coords])
    elif len(dihedrals) > 1:
        coords = np.array(
            [
                internal_coords.set_dihedral(dihe, 1 + increment, dihedral_atoms)
                for dihe in dihedrals
            ]
        )
    else:
        if coords.ndim == 2:
            coords = np.expand_dims(coords, axis=0)

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
        DATA_DIR / f"UserRotlibs/{name}_rotlib.npz",
        **save_dict,
        allow_pickle=True,
    )


def add_dihedral_def(name, dihedrals):

    # Reload in case there were other changes
    with open(os.path.join(os.path.dirname(__file__), "data/DihedralDefs.pkl"), "rb") as f:
        local_dihedral_def = pickle.load(f)

    # Add new label defs and write file
    local_dihedral_def[name] = dihedrals
    with open(os.path.join(os.path.dirname(__file__), "data/DihedralDefs.pkl"), "wb") as f:
        pickle.dump(local_dihedral_def, f)

    # Add to active dihedral def dict
    chiLife.dihedral_defs[name] = dihedrals