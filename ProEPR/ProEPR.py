import logging, numbers, shutil, pickle, sqlite3, tempfile, os
from unittest import mock
from itertools import combinations
from typing import Tuple, Union, Callable
from io import StringIO
from pathlib import Path
from collections.abc import Sized
import MDAnalysis
import MDAnalysis.core.topologyattrs
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from numpy.typing import ArrayLike
from scipy.stats import gaussian_kde
from scipy.signal import fftconvolve
from memoization import cached
from .RotamerLibrary import RotamerLibrary
from .SpinLabel import SpinLabel
from .SpinLabelTraj import SpinLabelTraj
from .protein_utils import *
from .numba_utils import *
from .scoring import *
from .superimpositions import superimpositions

# TODO: Implement all atom parameters from CHARMM
#       -Requires implementation of all spin labels in CHARMM and atom renaming to match RTP --> Huge undertaking
# TODO: Consider electrostatic energy evaluation
# TODO: Consider solvation energy evaluation


# Define useful global variables
SUPPORTED_LABELS = ('R1M', 'R7M', 'V1M', 'I1M', 'M1M', 'R1C')
SUPPORTED_BB_LABELS = ('R1C',)
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data/rotamer_libraries/')
cursor = sqlite3.connect(DATA_DIR + 'bbdep.db').cursor()
logging.captureWarnings(True)

with open(DATA_DIR + 'spin_atoms.txt', 'r') as f:
    lines = f.readlines()
    SPIN_ATOMS = {x.split(':')[0]: eval(x.split(':')[1]) for x in lines}

USER_LABELS = {key for key in SPIN_ATOMS if key not in SUPPORTED_LABELS}
SUPPORTED_RESIDUES = set(list(SUPPORTED_LABELS) + list(USER_LABELS) + ['NHH'] + list(dihedral_defs.keys()))


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


def get_dihedral_rotation_matrix(theta: float, v: ArrayLike) -> ArrayLike:
    """
    Build a matrix that will rotate coordinates about a vector, v, by theta in radians.

    :param theta: float
        Rotation angle in radians.

    :param v: numpy ndarray (1x3)
        Three dimensional vector to rotate about.

    :return rotation_matrix: numpy ndarray
        Matrix that will rotate coordinates about the vector, V by angle theta.
    """

    # Normalize input vector
    v = v / np.linalg.norm(v)

    # Compute Vx matrix
    Vx = np.zeros((3, 3))
    Vx[[2, 0, 1], [1, 2, 0]] = v
    Vx -= Vx.T

    # Rotation matrix. See https://en.wikipedia.org/wiki/Rotation_matrix#Rotation_matrix_from_axis_and_angle
    rotation_matrix = np.identity(3) * np.cos(theta) + np.sin(theta) * Vx + (1 - np.cos(theta)) * np.outer(v, v)

    return rotation_matrix


def get_dihedral(p: ArrayLike) -> float:
    """
    Calculates dihedral of a given set of atoms, p = [0, 1, 2, 3]. Returns value in degrees.

                    3
         ------>  /
        1-------2
      /
    0

    :param p: numpy ndarray (4x3)
        matrix containing coordinates to be used to calculate dihedral.

    :retrun: float
        Dihedral angle in degrees
    """

    p0 = p[0]
    p1 = p[1]
    p2 = p[2]
    p3 = p[3]

    # Define vectors from coordinates
    b0 = -1.0 * (p1 - p0)
    b1 = p2 - p1
    b2 = p3 - p2

    # Normalize dihedral bond vector
    b1 /= np.linalg.norm(b1)

    # Calculate dihedral projections orthogonal to the bond vector
    v = b0 - np.dot(b0, b1) * b1
    w = b2 - np.dot(b2, b1) * b1

    # Calculate angle between projections
    x = np.dot(v, w)
    y = np.dot(np.cross(b1, v), w)

    # Return as degrees
    return np.degrees(np.arctan2(y, x))


def set_dihedral(p: ArrayLike, angle: float, mobile: ArrayLike) -> ArrayLike:
    """
    Sets the dihedral angle by rotating all 'mobile' atoms from their current position about the dihedral bond defined
    by the four atoms in p. Dihedral will be set to the value of 'angle' in degrees.

    :param p: array-like int
        Indices of atoms that define dihedral to rotate about.

    :param angle: float
        New angle to set the dihedral to (degrees).

    :param mobile: ndarray
        Atom coordinates to move by setting dihedral.

    :returns: ndarray
        New positions for the mobile atoms
    """

    current = get_dihedral(p)
    angle = angle - current
    angle = np.deg2rad(angle)

    ori = p[1]
    mobile -= ori
    v = p[2] - p[1]
    v /= np.linalg.norm(v)
    R = get_dihedral_rotation_matrix(angle, v)

    new_mobile = R.dot(mobile.T).T + ori

    return new_mobile


def local_mx(N: ArrayLike, CA: ArrayLike, C: ArrayLike, method: str='rosetta') -> Tuple[ArrayLike, ArrayLike]:
    """
    Calculates a translation vector and rotation matrix to transform the provided rotamer library from the global
    coordinate frame to the local coordinate frame using the specified method.
    :param rotamer_library: RotamerLibrary
        RotamerLibrary to be moved
    :param method: str
        Method to use for generation of rotation matrix
    :return origin, rotation_matrix: ndarray, ndarray
        origin and rotation matrix for rotamer library
    """

    if method in {'fit'}:
        rotation_matrix, _ = superimpositions[method](N, CA, C)
    else:
        # Transform coordinates such that the CA atom is at the origin
        Nn = N - CA
        Cn = C - CA
        CAn = CA - CA

        # Local Rotation matrix is the inverse of the global rotation matrix
        rotation_matrix, _ = superimpositions[method](Nn, CAn, Cn)

    rotation_matrix = rotation_matrix.T

    # Set origin at Calpha
    origin = CA

    return origin, rotation_matrix


def global_mx(N: ArrayLike, CA: ArrayLike, C: ArrayLike, method: str='rosetta') -> Tuple[ArrayLike, ArrayLike]:
    rotation_matrix, origin = superimpositions[method](N, CA, C)
    return rotation_matrix, origin


def ic_mx(atom1: ArrayLike, atom2: ArrayLike, atom3: ArrayLike) -> Tuple[ArrayLike, ArrayLike]:
    """
    Calculates a rotation matrix and translation to transform a set of atoms to global coordinate frame from a local
    coordinated frame defined by atom1, atom2 and atom 3. The X-vector is defined as the bond between atom1 and atom2
    the Y-vector is defined as the vector orthogonal to the X vector in the atom1-atom2-atom3 plane and the Z-vector
    is the cross product between the X and Y Vectors

    :param atom1: numpy ndarray (1x3)
        Backbone nitrogen coordinates

    :param atom2: numpy ndarray (1x3)
        Backbone carbonyl carbon coordinates

    :param atom3: numpy ndarray (1x3)
        Backbone Calpha carbon coordinates

    :return (rotation_matrix, origin) : (numpy ndarray (1x3), numpy ndarray (3x3))
        rotation_matrix: rotation  matrix to rotate spin label to
        origin: new origin position in 3 dimensional space
    """

    p1 = atom1
    p2 = atom2
    p3 = atom3

    # Define new X axis
    v12 = p2 - p1
    v12 /= np.linalg.norm(v12)

    # Define new Y axis
    v23 = p3 - p2
    p23_x_comp = v23.dot(v12)
    v23 -= p23_x_comp * v12
    v23 /= np.linalg.norm(v23)

    # Define new z axis
    zaxis = np.cross(v12, v23)

    # Create rotation matrix
    rotation_matrix = np.array([v12, v23, zaxis])
    origin = p1

    return rotation_matrix, origin


def get_dd(*args, r: Union[Tuple, ArrayLike]=(0, 100), sigma: float=1.0,
           prune: bool=False, uq: bool=False, **kwargs) -> ArrayLike:
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

    :return P: ndarray
        The probability density of the distance distribution corresponding to r
    """

    # Allow r to be passed as that last non-keyword argument
    if not isinstance(args[-1], (SpinLabel, SpinLabelTraj, mock.Mock)):
        r = args[-1]
        args = args[:-1]

    if any(not hasattr(arg, atr) for arg in args for atr in ['spin_coords', 'weights']):
        raise TypeError('Arguments other than spin labels must be passed as a keyword argument')

    size = kwargs.get('size', 1024)
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
            raise ValueError('Pruning is not supported when performing uncertainty analysis (yet)')
        Ps = []
        nboots = uq if uq > 1 else 100
        for i in tqdm(range(nboots)):
            dummy_labels = []
            for SL in args:
                idxs = np.random.choice(len(SL), len(SL)    )

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
            raise IndexError("Pruned distance distributions are only supported when using two spin labels (for now).")

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


def unfiltered_dd(*args, r: ArrayLike, sigma: float=1.) -> ArrayLike:
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
    hist, _ = np.histogram(distances, weights=weights, range=(min(r), max(r)), bins=len(r))
    if sigma != 0:
        # Calculate normal distribution for convolution
        delta_r = get_delta_r(r)
        _, g = norm(delta_r, 0, sigma)

        # Convolve normal distribution and histogram
        P = fftconvolve(hist, g, mode='same')

    else:
        P = hist

    # Normalize weights
    P /= np.trapz(P, r)

    return P


def filter_by_weight(w1:ArrayLike, w2:ArrayLike, cutoff: float=0.001) -> Tuple[ArrayLike, ArrayLike]:
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


@njit(cache=True)
def filtered_dd(NO1: ArrayLike, NO2: ArrayLike, weights: ArrayLike, r: ArrayLike, sigma: float=1.) -> ArrayLike:
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
    hist[0] = 0.

    # Create normal distribution to convolve with histogram
    _, g = norm(delta_r, 0, sigma)

    # Determine indices of convolution output corresponding to r
    begin = int(len(g) / 2)
    end = begin
    if len(g) % 2 == 0:
        end -= 1

    # Compute distance distribution by convolution
    distance_distribution = np.convolve(hist, g)
    distance_distribution = distance_distribution / distance_distribution.sum()

    return distance_distribution[begin:-end]


def traj_dd(SL1: SpinLabelTraj, SL2: SpinLabelTraj, r: ArrayLike, sigma: float,
            filter: Union[bool, float], **kwargs) -> ArrayLike:
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
        raise ValueError('SpinLabelTraj objects must have the same length')

    # Calculate the distance distribution for each frame and sum
    P = np.zeros_like(r)
    for _SL1, _SL2 in zip(SL1, SL2):
        P += get_dd(_SL1, _SL2, r=r, sigma=sigma, prune=filter, **kwargs)

    # Normalize distance distribution
    P /= np.trapz(P, r)

    return P


@cached
def read_sl_library(label: str, user: bool = False) -> Tuple[ArrayLike,...]:
    """
    Reads ProEPR rotamer library for spin label.

    :param label: str
        3 character abbreviation for desired spin label

    :param user: bool
        Specifies if the library was defined by a user or if it is a precalculated library from ProEPR

    :return coords, (internal_coords), weights, atom_types, atom_names: ndarray
        Arrays of spin label coordinates, weights, atom types and atom names in the local coordinate frame. If
        internal_coord information is available it will be returned in between coords and weights.
    """
    subdir = 'UserRotlibs/' if user else 'MMM_RotLibs/'
    data = os.path.join(os.path.dirname(__file__), 'data/rotamer_libraries/')
    with np.load(data + subdir + label + '_rotlib.npz', allow_pickle=True) as files:

        coords, internal_coords, weights = files['coords'], files['internal_coords'], files['weights']
        atom_types, atom_names = files['atom_types'], files['atom_names']
        dihedrals, dihedral_atoms = files['dihedrals'], files['dihedral_atoms']
    return coords, internal_coords, weights, atom_types, atom_names, dihedrals, dihedral_atoms


@cached
def read_bbdep(res: str, Phi: float, Psi: float) -> Tuple[ArrayLike,...]:
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
    Phi, Psi = str(Phi), str(Psi)

    # Read residue internal coordinate structure
    with open(DATA_DIR + 'residue_internal_coords/' + res.lower() + '_ic.pkl', 'rb') as f:
        ICs = pickle.load(f)

    atom_types = [atom.atype for atom in ICs]
    atom_names = [atom.name for atom in ICs]

    maxchi = 5 if res in SUPPORTED_BB_LABELS else 4
    nchi = np.minimum(len(dihedral_defs[res]), maxchi)

    if res not in ('ALA', 'GLY'):
        library = 'R1C.lib' if res in SUPPORTED_BB_LABELS else 'ALL.bbdep.rotamers.lib'
        start, length = rotlib_indexes[f'{res}  {Phi:>4}{Psi:>5}']

        with open(DATA_DIR + library, 'rb') as f:
            f.seek(start)
            rotlib_string = f.read(length).decode()
            s = StringIO(rotlib_string)
            s.seek(0)
            data = np.genfromtxt(s, usecols=range(maxchi + 4, maxchi + 5 + 2 * maxchi))

        weights, dihedrals, sigmas = data[:, 0], data[:, 1:nchi+1], data[:, maxchi+1:maxchi + nchi + 1]
        dihedral_atoms = dihedral_defs[res][:nchi]

        # Calculate cartesian coordinates for each rotamer
        coords = []
        internal_coords = []
        for r in dihedrals:
            ICn = ICs.copy().set_dihedral(np.deg2rad(r), 1, atom_list=dihedral_atoms)

            coords.append(ICn.to_cartesian())
            internal_coords.append(ICn)

    else:
        weights = [1]
        dihedrals, sigmas, dihedral_atoms = [], [], []
        coords = [ICs.to_cartesian()]
        internal_coords = [ICs.copy()]

    # Get origin and rotation matrix of local frame
    mask = np.in1d(atom_names, ['N', 'CA', 'C'])
    ori, mx = local_mx(*coords[0][mask])

    # Set coords in local frame and prepare output
    coords = np.array([(coord - ori) @ mx for coord in coords])
    atom_types = np.asarray(atom_types)
    atom_names = np.asarray(atom_names)
    dihedral_atoms = np.asarray(dihedral_atoms)

    return coords, internal_coords, weights, sigmas, atom_types, atom_names, dihedrals, dihedral_atoms


def read_library(res: str, Phi: float=None, Psi: float=None) -> Tuple[ArrayLike, ...]:
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
        res = res.upper()
        Phi = int((Phi // 10) * 10)
        Psi = int((Psi // 10) * 10)

    if res in SUPPORTED_LABELS and res not in SUPPORTED_BB_LABELS:
        return read_sl_library(res)
    elif res in USER_LABELS:
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
        chain = site.strip('0123456789')

    if chain:
        resi = int(site[len(chain):])
    else:
        resi = int(site)
        chain = 'A'

    return resi, chain, label


@njit(cache=True)
def optimize_weights(ensemble: ArrayLike, idx: ArrayLike, start_weights: ArrayLike,
                     start_score: float, data: ArrayLike) -> Tuple[ArrayLike, ArrayLike, ArrayLike, float]:
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


def save(file_name: str, *labels: SpinLabel, protein: Union[mda.Universe, mda.AtomGroup, str] = None, **kwargs) -> None:
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
    if isinstance(file_name, SpinLabel):
        labels.insert(0, file_name)
        file_name = None
    elif hasattr(file_name, 'atoms'):
        labels.insert(-1, file_name)
        file_name = None

    # Check for protein structures at the end of args
    if protein is None:
        if isinstance(labels[-1], str) or \
                isinstance(labels[-1], mda.Universe) or \
                isinstance(labels[-1], MDAnalysis.AtomGroup):
            protein = labels.pop(-1)

    # Create a file name from spin label and protein information
    if file_name is None:
        if isinstance(protein, str):
            f = Path(protein)
            file_name = f.name.rstrip('.pdb')
        elif hasattr(protein, 'atoms'):
            if protein.filename is not None:
                file_name = protein.filename.rstrip('.pdb')
            else:
                file_name = 'No_Name_Protein'

        if 0 < len(labels) < 3:
            for label in labels:
                file_name += f'_{label.site}{label.label}'
        else:
            file_name += '_many_labels'
        file_name += '.pdb'

    if protein is not None:
        if isinstance(protein, str):
            print(protein, file_name)
            shutil.copy(protein, file_name)
        elif isinstance(protein, mda.Universe) or isinstance(protein, MDAnalysis.AtomGroup):
            protein.atoms.write(file_name)
        else:
            raise TypeError('`protein` must be a string or an MDAnalysis Universe/AtomGroup object')

    if len(labels) > 0:
        write_labels(file_name, *labels, **kwargs)


def write_labels(file: str, *args: SpinLabel, KDE: bool = True) -> None:
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
    fmt_str = "ATOM  {:5d} {:<4s} {:3s} {:1s}{:4d}    {:8.3f}{:8.3f}{:8.3f}{:6.2f}{:6.2f}          {:>2s}\n"
    with open(file, 'a+', newline='\n') as f:
        # Write spin label models
        f.write('\n')
        for k, label in enumerate(args):
            f.write(f'HEADER {label.name}\n')

            # Save models in order of weight
            sorted_index = np.argsort(label.weights)[::-1]
            for mdl, (conformer, weight) in enumerate(zip(label.coords[sorted_index],
                                                          label.weights[sorted_index])):
                f.write('MODEL {}\n'.format(mdl))

                [f.write(fmt_str.format(i, label.atom_names[i], label.label, label.chain, int(label.site),
                                        *conformer[i], 1.00, weight * 100, label.atom_types[i]))
                 for i in range(len(label.atom_names))]
                f.write('TER\n')
                f.write('ENDMDL\n')

        # Write electron density at electron coordinates
        for k, label in enumerate(args):
            f.write(f'HEADER {label.name}_density\n'.format(label.label, k + 1))
            NO = np.atleast_2d(label.spin_coords)

            if KDE:
                # Perform gaussian KDE to determine electron density
                gkde = gaussian_kde(NO.T, weights=label.weights)

                # Map KDE density to pseudoatoms
                vals = gkde.pdf(NO.T)
            else:
                vals = label.weights

            [f.write(fmt_str.format(i, 'NEN', label.label, label.chain, int(label.site),
                                    *NO[i], 1.00, vals[i] * 100, 'N'))
             for i in range(len(vals))]

            f.write('TER\n')


def repack(protein: Union[mda.Universe, mda.AtomGroup], *spin_labels: SpinLabel, repetitions: int=200, temp: float=1,
           energy_func: Callable=get_lj_rep, **kwargs) -> Tuple[mda.Universe, ArrayLike, Tuple[SpinLabel, ...]]:
    """
    Given a protein and a SpinLabel object, repack the local environment using monte carlo sampling.

    :param protein: MDAnalysis.Universe or MDAnalysis.AtomGroup
        Protein to be repacked

    :param spin_labels: ProEPR.SpinLabel
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

    repack_radius = kwargs.get('repack_radius', 10)  # Angstroms

    # Construct a new spin labeled protein and preallocate variables to retain monte carlo trajectory
    spin_label_str = " or ".join(f'( {spin_label.selstr} )' for spin_label in spin_labels)
    protein = mutate(protein, *spin_labels).atoms

    # Determine the residues near the spin label that will be repacked
    repack_residues = protein.select_atoms(f'(around {repack_radius} {spin_label_str} ) '
                                           f'or {spin_label_str}').residues
    repack_residue_libraries = [RotamerLibrary.from_mda(res) for res in repack_residues if
                                res.resname not in ['GLY', 'ALA']]

    # Create new labeled protein construct to fill in any missing atoms of repack residues
    protein = mutate(protein, *repack_residue_libraries).atoms

    repack_residues = protein.select_atoms(f'(around {repack_radius} {spin_label_str} ) '
                                           f'or {spin_label_str}').residues

    repack_residue_libraries = [RotamerLibrary.from_mda(res) for res in repack_residues if
                                res.resname not in ['GLY', 'ALA']]

    traj = np.empty((repetitions, *protein.positions.shape))
    deltaEs = []

    sample_freq = np.array([len(res.weights) for res in repack_residue_libraries], dtype=np.float64)
    sample_freq /= sample_freq.sum()

    # Select atoms to consider when calculating lennard-jones potential
    lj_atoms = repack_residues.atoms.select_atoms(f'(around 10. protein) or protein')
    lj_radii = ProEPR.get_lj_rmin(lj_atoms.atoms.types)
    lj_eps = ProEPR.get_lj_eps(lj_atoms.atoms.types)

    count = 0
    acount = 0
    bcount = 0
    bidx = 0
    schedule = repetitions / (len(temp) + 1)
    with tqdm(total=repetitions) as pbar:
        while count < repetitions:

            # Randomly select a residue from the repack residues
            SiteLibrary = repack_residue_libraries[np.random.choice(len(repack_residue_libraries), p=sample_freq)]
            coords, weight = SiteLibrary.sample(off_rotamer=kwargs.get('off_rotamer', False))
            lj_mask = ~np.isin(lj_atoms.ix, SiteLibrary.clash_ignore_idx)
            mask = ~np.isin(protein.ix, SiteLibrary.clash_ignore_idx)

            # Calculate energy before the change
            dist = cdist(lj_atoms.positions[~lj_mask], lj_atoms.positions[lj_mask]).ravel()
            tlj_eps = np.sqrt(np.outer(lj_eps[~lj_mask], lj_eps[lj_mask]).reshape((-1)))
            tlj_radii = np.add.outer(lj_radii[~lj_mask], lj_radii[lj_mask]).reshape(-1)
            with np.errstate(divide='ignore'):
                E0 = energy_func(dist[dist < 10], tlj_radii[dist < 10], tlj_eps[dist < 10]).sum() \
                     - KT[temp[bidx]] * np.log(SiteLibrary.current_weight)

            # Calculate energy after the change
            dist = cdist(coords, lj_atoms.positions[lj_mask]).ravel()
            with np.errstate(divide='ignore'):
                E1 = energy_func(dist[dist < 10], tlj_radii[dist < 10], tlj_eps[dist < 10]).sum() \
                     - KT[temp[bidx]] * np.log(weight)

            deltaE = E1 - E0
            deltaE = np.maximum(deltaE, -10.)
            if deltaE == -np.inf:
                print(SiteLibrary.name)
                print(E1, E0)

            acount += 1
            # Metropolis-Hastings criteria
            if E1 < E0 or np.exp(-deltaE / KT[temp[bidx]]) > np.random.rand():

                deltaEs.append(deltaE)
                try:
                    protein.atoms[~mask].positions = coords
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

    logging.info(f'Total counts: {acount}')

    # Load MCMC trajectory into universe.
    protein.universe.load_new(traj)

    # Set frame to lowest energy
    protein.universe.trajectory[np.argmin(np.cumsum(deltaE))]

    return_labels = []
    for spin_label in spin_labels:
        return_labels.append(SpinLabel(spin_label.label, spin_label.site, spin_label.chain,
                                       protein=protein, energy_func=energy_func))

    return protein, deltaEs, return_labels


def add_label(name: str, pdb: str, dihedral_atoms: List[str], spin_atoms: List[str] = None,
              dihedrals: ArrayLike = None, weights: ArrayLike = None):
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
    """
    # TODO: Add dihedral definitions to DihedralDefs.pkl
    # Sort the PDB for optimal dihedral definitions
    pdb_lines = sort_pdb(pdb)
    pdb_resname = pdb_lines[0][17:20] if isinstance(pdb_lines[0], str) else pdb_lines[0][0][17:20]

    # Store spin atoms if provided
    if spin_atoms is not None:
        if isinstance(spin_atoms, str):
            spin_atoms = spin_atoms.split()

        with open(DATA_DIR + 'spin_atoms.txt', 'r+') as f:
            lines = f.readlines()
            spin_dict = {x.split(':')[0]: eval(x.split(':')[1]) for x in lines}
            if name in spin_dict:
                if spin_dict[name] != spin_atoms:
                    raise NameError('There is already a ProEPR spin label with this name')
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
        with tempfile.NamedTemporaryFile(suffix='.pdb', mode='w+', delete=False) as tmpfile:
            for i, model in enumerate(pdb_lines):
                tmpfile.write(f'MODEL {i + 1}\n')
                for atom in model:
                    tmpfile.write(atom)
                tmpfile.write('TER\nENDMDL\n')
    else:
        with tempfile.NamedTemporaryFile(suffix='.pdb', mode='w+', delete=False) as tmpfile:
            for line in pdb_lines:
                tmpfile.write(line)

    # Load sorted atom pdb using MDAnalysis and remove tempfile
    struct = mda.Universe(tmpfile.name, in_memory=True)
    os.remove(tmpfile.name)

    # Convert loaded rotamer library to internal coords
    internal_coords = [ProEPR.get_internal_coords(struct, resname=pdb_resname, preferred_dihedrals=dihedral_atoms) for ts
                       in struct.trajectory]

    # Add internal_coords to data dir
    with open(DATA_DIR + 'residue_internal_coords/' + name + '_ic.pkl', 'wb') as f:
        pickle.dump(internal_coords, f)

    # If multi-state pdb extract rotamers from pdb
    if dihedrals is None:
        dihedrals = []
        for ts in struct.trajectory:
            dihedrals.append(
                [get_dihedral(struct.select_atoms(f'name {" ".join(a)}').positions) for a in dihedral_atoms])

    if weights is None:
        weights = np.ones(len(dihedrals))
        weights /= weights.sum()

    # Extract coordinats and transform to the local frame
    coords = internal_coords[0].to_cartesian()
    ori, mx = local_mx(*struct.select_atoms('name N CA C').positions)
    coords = (coords - ori) @ mx

    # Save pdb structure
    save_pdb(DATA_DIR + 'residue_pdbs/' + name + '.pdb', internal_coords[0], coords)

    if len(struct.trajectory) > 1:
        coords = np.array([(struct.atoms.positions - ori) @ mx for ts in struct.trajectory])
    elif len(dihedrals) > 1:
        coords = np.array([internal_coords.set_dihedral(dihe, 1, dihedral_atoms) for dihe in dihedrals])
    else:
        if coords.ndim == 2:
            coords = np.expand_dims(coords, axis=0)

    atom_types = struct.atoms.types
    atom_names = struct.atoms.names

    # Save rotamer library
    np.savez(DATA_DIR + 'UserRotlibs/' + name + '_rotlib.npz',
             coords=coords, internal_coords=internal_coords, weights=weights,
             atom_types=atom_types, atom_names=atom_names,
             dihedrals=dihedrals, dihedral_atoms=dihedral_atoms,
             allow_pickle=True)
