from __future__ import annotations
import tempfile, logging, os, rtoml, re, textwrap
import zipfile, shutil
from copy import deepcopy
from pathlib import Path
from itertools import combinations, chain
from collections import Counter
from typing import Callable, Tuple, Union, List, Dict
from unittest import mock

import igraph as ig
from tqdm import tqdm

import numpy as np
from numpy.typing import ArrayLike
from scipy.spatial.distance import cdist
from scipy.stats import t
import MDAnalysis as mda

import chilife.io as io
from .globals import (RL_DIR, USER_RL_DIR, DATA_DIR, SUPPORTED_RESIDUES, USER_LIBRARIES, USER_dLIBRARIES,
                      rotlib_defaults, dihedral_defs)
from .alignment_methods import local_mx
from .Topology import get_min_topol, guess_bonds
from .pdb_utils import sort_pdb, get_backbone_atoms, get_bb_candidates
from .protein_utils import mutate, guess_mobile_dihedrals
from .MolSysIC import MolSysIC
from .scoring import GAS_CONST, reweight_rotamers, ljEnergyFunc
from .numba_utils import get_delta_r, normdist
from .SpinLabel import SpinLabel
from .RotamerEnsemble import RotamerEnsemble
from .SpinLabelTraj import SpinLabelTraj

logging.captureWarnings(True)


def distance_distribution(
        *args: SpinLabel,
        r: ArrayLike = None,
        sigma: float = 1.0,
        use_spin_centers: bool = True,
        dependent: bool = False,
        uq: bool = False,
) -> np.ndarray:
    """Calculates total distribution of spin-spin distances among an arbitrary number of spin labels, using the
    distance range ``r`` (in angstrom). The distance distribution is obtained by summing over all label pair distance
    distributions. These in turn are calculated by convolving a weighted histogram of pairwise distances between lables
    with a normal distribution. Distance between labels are calculated either between label ``spin_center`` if
    ``spin_populations=False``, or between all pairs of spn-bearing atoms if ``spin_populations=True``.

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
        If False, distances are computed between the weighted centroids of the spin atoms. If True, distances are
        computed by summing over the distributed spin density on spin-bearing atoms on the labels.
    dependent: bool
        Consider the (clash) effects of spin label rotamers on each other.
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

    if len(args) < 2:
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

        P = pair_dd(*args, r=r, sigma=sigma, use_spin_centers=use_spin_centers, dependent=dependent)
        return P


def pair_dd(*args, r: ArrayLike, sigma: float = 1.0, use_spin_centers: bool = True, dependent=False) -> np.ndarray:
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
    dependent: bool
        Consider the (clash) effects of spin label rotamers on each other.

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
            coords1 = SL1.spin_coords.reshape(-1, 3)
            coords2 = SL2.spin_coords.reshape(-1, 3)
            weights1 = np.outer(SL1.weights, SL1.spin_weights).flatten()
            weights2 = np.outer(SL2.weights, SL2.spin_weights).flatten()

        distances.append(cdist(coords1, coords2).flatten())
        weights.append(np.outer(weights1, weights2).flatten())

        if dependent:
            if not isinstance(SL1.energy_func, ljEnergyFunc):
                raise RuntimeError('Currently only ljEnergyFunc objects are supported when using dependent=True')

            if SL1.energy_func.join_rmin is not SL2.energy_func.join_rmin or \
               SL1.energy_func.join_eps is not SL2.energy_func.join_eps:
                raise RuntimeError('At least two labels passed use different energy functions. Make sure all '
                                   'labels use the same energy functions when setting `dependent=True`. This does not'
                                   'mean that the energy functions use the same parameters. They have to be the SAME '
                                   'object and satisfy `SL1.energy_func is SL2.energy_func`. This can be achieved be '
                                   'creating an energy function object')


            nrot1, nrot2 = len(SL1), len(SL2)
            nat1, nat2 = len(SL1.side_chain_idx), len(SL2.side_chain_idx)
            join_rmin = SL1.energy_func.join_rmin
            join_eps = SL1.energy_func.join_eps

            rmin_ij = join_rmin(SL1.rmin2, SL2.rmin2)
            eps_ij = join_eps(SL1.eps, SL2.eps)

            rot_coords1 = SL1.coords[:, SL1.side_chain_idx]
            rot_coords2 = SL2.coords[:, SL2.side_chain_idx].reshape(-1, 3)

            ljs = []
            for i, rots in enumerate(rot_coords1):
                lj = cdist(rots, rot_coords2)
                lj = lj.reshape(1, nat1, nrot2, nat2).transpose(0, 2, 1, 3)

                lj = rmin_ij[None, None, ...] / lj
                lj = lj * lj * lj
                lj = lj * lj
                lj = lj * lj

                # Cap
                lj[lj > 10] = 10
                # Rep only
                lj = eps_ij * (lj * lj)
                lj = lj.sum(axis=(2, 3))
                ljs.append(lj)

            ljs = np.concatenate(ljs)
            weights[-1], _ = reweight_rotamers(ljs.flatten(), SL1.temp, weights[-1])

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
    integral = np.trapz(P, r)
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
    """Calculate a distance distribution from a trajectory of spin labels by calling ``distance_distribution`` on each
    frame and averaging the resulting distributions.

    Parameters
    ----------
    SL1, SL2: SpinLabelTrajectory
        Spin label to use for distance distribution calculation.
    r : ArrayLike
        Distance domain to use when calculating distance distribution.
    sigma : float
        Standard deviation of the gaussian kernel used to smooth the distance distribution
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


def confidence_interval(data: ArrayLike, cutoff: float = 0.95, non_negative: bool = True) -> Tuple[np.array, np.array]:
    """

    Parameters
    ----------
    data: Array Like
        Array of distance distribution samples
    cutoff: float
        Confidence cutoff to calculate confidence interval over.
    non_negative: bool
        Clip negative elements of confidence intervals to 0.

    Returns
    -------
    interval: Tuple[Array, Array]
        Lower and upper confidence intervals at the provided cutoff

    """
    mu = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    ub, lb = t.interval(cutoff, df=len(data) - 1, loc=mu, scale=std)

    lb[np.isnan(lb)] = 0.
    ub[np.isnan(ub)] = 0.

    if non_negative:
        lb = lb.clip(0)
        ub = ub.clip(0)

    return lb, ub


def create_library(
        libname: str,
        pdb: str,
        dihedral_atoms: List[List[str]] = None,
        site: int = 1,
        resname: str = None,
        dihedrals: ArrayLike = None,
        weights: ArrayLike = None,
        sigmas: ArrayLike = None,
        permanent: bool = False,
        default: bool = False,
        force: bool = False,
        spin_atoms: List[str] = None,
        aln_atoms: List[str] = None,
        backbone_atoms: List[str] = None,
        description: str = None,
        comment: str = None,
        reference: str = None
) -> None:
    """Create a chiLife rotamer library from a pdb file and a user defined set of parameters.

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
        a list of rotatable dihedrals. List should contain sub-lists of 4 atom names. Atom names must be the same as
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
        A list of atom names on which the spin density is localized.
    aln_atoms: List[str]
        The three atoms defining the alignment coordinate frame of the target residue. Usually the N-CA-C atoms of
        the protein backbone or similar for nucleotide backbones.
    backbone_atoms: List[str]
        Names of the atoms that correspond to backbone atoms. This includes aln_atoms and any other atoms that may need
        to be adjusted to fit the target residue backbone, e.g. the hydrogen atoms of carboxyl oxygen atoms. Any atoms
        not part of the backbone will be considered side chain atoms and may be subject to sampling.
    description : str
        A short (< 60 characters) description of the side chain library being created
    comment : str
        Additional information about the rotamer library that may not fit in description.
    reference : str
        Any relevant citations associated with the rotamer library.
    """
    resname = libname[:3] if resname is None else resname
    struct, spin_atoms = pre_add_library(pdb, spin_atoms, aln_atoms=aln_atoms)
    resi_selection = struct.select_atoms(f"resnum {site}")
    bonds = resi_selection.intra_bonds.indices

    if not continuous_topol(resi_selection, bonds):
        raise RuntimeError("The  RotamerEnsemble does not seem to have a consistent and continuous bond topology over "
                           "all states. Please check the input PDB and remove any states that do not have the "
                           "expected bond topology. Alternatively, you can force certain bonds by adding CONECT "
                           "information to the PDB file. See "
                           "https://www.wwpdb.org/documentation/file-format-content/format33/sect10.html")

    # Convert loaded rotamer ensemble to internal coords
    internal_coords = MolSysIC.from_atoms(resi_selection,
                                                  preferred_dihedrals=dihedral_atoms,
                                                  bonds=bonds,
                                                  use_chain_operators=False)

    if not dihedral_atoms:
        dihedral_atoms = guess_mobile_dihedrals(internal_coords, aln_atoms=aln_atoms)

    internal_coords.shift_resnum(-(site - 1))

    if len(internal_coords.chains) > 1:
        raise ValueError('The PDB of the label supplied appears to have a chain break. Please check your PDB and '
                         'make sure there are no chain breaks in the desired label and that there are no other '
                         'chains in the pdb file. If the error persists, check to be sure all atoms are the correct '
                         'element as chilife uses the elements to determine if atoms are bonded.')

    backbone_atoms, aln_atoms = aln_sanity(internal_coords, resname, backbone_atoms, aln_atoms)

    # If multi-state pdb extract dihedrals from pdb
    if dihedrals is None:
        dihedrals = np.rad2deg([ic.get_dihedral(1, dihedral_atoms) for ic in internal_coords])

    if dihedrals.shape == (len(dihedrals),):
        dihedrals.shape = (len(dihedrals), 1)

    if weights is None:
        weights = np.ones(len(dihedrals))
        weights /= weights.sum()

    save_dict = prep_restype_savedict(libname, resname, internal_coords,
                                      weights, dihedrals, dihedral_atoms,
                                      sigmas=sigmas, spin_atoms=spin_atoms,
                                      aln_atoms=aln_atoms, backbone_atoms=backbone_atoms,
                                      description=description, comment=comment,
                                      reference=reference)

    # Check that all rotamer backbones are aligned
    if np.sum(np.abs(np.diff(save_dict['coords'][:,0]))) / len(save_dict['coords']) > 1e-3:
        aln_idxs = [np.argwhere(save_dict['atom_names'] == a).flat[0] for a in save_dict['aln_atoms']]
        backbone = np.squeeze(save_dict['coords'][:, aln_idxs])
        oris, mxs = [np.array(x) for x in zip(*[local_mx(*bb) for bb in backbone])]
        save_dict['coords'] = ( save_dict['coords'] - oris[:, None, :]) @ mxs
    else:
        backbone = save_dict['coords'][0, :3]
        ori, mx = local_mx(*backbone)
        save_dict['coords'] = np.einsum('ijk,kl->ijl', save_dict['coords'] - ori, mx)


    # Save rotamer library
    np.savez(Path().cwd() / f'{libname}_rotlib.npz', **save_dict, allow_pickle=True)

    if permanent:
        add_library(f'{libname}_rotlib.npz', libname=libname, default=default, force=force)


def create_dlibrary(
        libname: str,
        pdb: str,
        sites: Tuple,
        dihedral_atoms: List[List[List[str]]],
        resname: str = None,
        dihedrals: ArrayLike = None,
        weights: ArrayLike = None,
        sort_atoms: Union[List[str], Dict, str] = True,
        permanent: bool = False,
        default: bool = False,
        force: bool = False,
        spin_atoms: List[str] = None,
        description: str = None,
        comment: str = None,
        reference: str = None
) -> None:
    """Create a chiLife bifunctional rotamer library from a pdb file and a user defined set of parameters.

    Parameters
    ----------
    libname: str,
        Name for the user defined label.
    pdb : str
        Name of (and path to) pdb file containing the user defined spin label structure. This pdb file should contain
        only the desired spin label and no additional residues.
    sites : Tuple[int]
        The residue numbers that the bifunctional label is attached to.
    dihedral_atoms: List[List[List[str]]]
        A list defining the mobile dihedral atoms of both halves of the bifunctional label. There should be  two
        entries, one for each site that the bifunctional label attaches to. Each entry should contain a sublist of
        dihedral definitions. Dihedral definitions are a list of four strings that are the names of the four atoms
        defining the dihedral angle.

            .. code-block:: python

                [
                # Side chain 1
                [['CA', 'CB', 'SG', 'SD'],
                ['CB', 'SG', 'SD', 'CE']...],

                # Side chain 2
                [['CA', 'CB', 'SG', 'SD'],
                ['CB', 'SG', 'SD', 'CE']...]
                ]

    resname : str
        Residue type 3-letter code.
    dihedrals : ArrayLike, optional
        Array of dihedral angles. If provided the new label object will be stored as a rotamer library with the
        dihedrals provided.
    weights : ArrayLike, optional
        Weights associated with the dihedral angles provided by the ``dihedrals`` keyword argument
        permanent: bool
        If set to True the library will be stored in the chilife user_rotlibs directory in addition to the current
        working directory.
    sort_atoms: bool
        Rotamer library atoms are sorted to achieve consistent internal coordinates by default. Sometimes this sorting
        is not desired or the PDB file is already sorted. Setting this option to ``False`` turns off atom sorting.
    permanent : bool
        If set to True, a copy of the rotamer library will be stored in the chiLife rotamer libraries directory
    default : bool
        If set to true and permanent is also set to true then this rotamer library will become the default rotamer
        library for the given resname
    force: bool = False,
        If set to True and permanent is also set to true this library will overwrite any existing library with the same
        name.
    spin_atoms : list, dict
        List of atom names on which the spin density is localized, e.g ['N', 'O'], or dictionary with spin atom
        names (key 'spin_atoms') and spin atom weights (key 'spin_weights').
    description : str
        A short (< 60 characters) description of the side chain library being created
    comment : str
        Additional information about the rotamer library that may not fit in description.
    reference : str
        Any relevant citations associated with the rotamer library.
    """

    site1, site2 = sites
    increment = site2 - site1
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
        raise RuntimeError(
            "dihedral_atoms must be a list of lists where each sublist contains the list of dihedral atoms"
            "for the i and i+{increment} side chains. Sublists can contain any amount of dihedrals but "
            "each dihedral should be defined by exactly four unique atom names that belong to the same "
            "residue number"
        )

    struct, spin_atoms = pre_add_library(pdb, spin_atoms, uniform_topology=False, sort_atoms=sort_atoms)
    res1 = struct.select_atoms(f'resnum {site1}')
    res2 = struct.select_atoms(f'resnum {site2}')

    # Identify the cap based off of the user defined mobile dihedrals
    last_bonds = []
    for i, res in enumerate((res1, res2)):
        site = res.resnums[0]
        dh_bonds = [dihedral[1:3] for dihedral in dihedral_atoms[i]]

        dh_bond_idxs = []
        for anames in dh_bonds:
            asel = struct.select_atoms(f'resnum {site} and name {" ".join(anames)}')

            # Sometimes dihedrals are defined with the bonded atom(s) on the other residue
            if len(asel) < 2:
                missing_name = anames[0] if asel.names[0] == anames[1] else anames[1]
                other_site = res1.resnums[0] if i == 1 else res2.resnums[0]
                asel += struct.select_atoms(f'resnum {other_site} and name {missing_name}')

            dh_bond_idxs.append(asel.ix)
        dh_bond_idxs = np.array(dh_bond_idxs)

        last_bonds.append(dh_bond_idxs[np.argmax(dh_bond_idxs[:, 0])])

    bond_indices = struct.bonds.indices
    graph = ig.Graph(n=np.max(bond_indices), edges=bond_indices)

    # disconnect the cap by cutting the rotatable bond of the dihedrals in each site
    last_bonds = [tuple(bond) for bond in last_bonds]
    graph.delete_edges(last_bonds)

    # Find all atoms in between the cut bonds
    starts = [node[1] for node in last_bonds]
    cap1_idxs, _ = graph.dfs(starts[0])
    cap2_idxs, _ = graph.dfs(starts[1])

    ovlp1_selection = struct.atoms[cap1_idxs]
    csts = ovlp1_selection.names
    csts = csts.astype('U4')

    ovlp1_selection.residues.resnums = site1
    ovlp1_selection.residues.resids = site1
    res1 += ovlp1_selection[~np.isin(ovlp1_selection.ix, res1.ix)]
    res1_bonds = res1.intra_bonds.indices

    error_message = "The  dRotamerEnsemble does not seem to have a consistent and continuous bond topology over" \
                           " all states. Please check the input PDB and remove any states that do not have the " \
                           "expected bond topology. Alternatively, you can force certain bonds by adding CONECT " \
                           "information to the PDB file. See " \
                           "https://www.wwpdb.org/documentation/file-format-content/format33/sect10.html"

    if not continuous_topol(res1, res1_bonds):
        raise RuntimeError(error_message)

    IC1 = MolSysIC.from_atoms(res1, dihedral_atoms[0], res1_bonds, use_chain_operators=False)

    ovlp2_selection = struct.atoms[cap2_idxs]
    ovlp2_selection.residues.resnums = site2
    ovlp2_selection.residues.resids = site2
    res2 += ovlp2_selection[~np.isin(ovlp2_selection.ix, res2.ix)]
    res2_bonds = res2.intra_bonds.indices

    if not continuous_topol(res2, res2_bonds):
        raise RuntimeError(error_message)

    IC2 = MolSysIC.from_atoms(res2, dihedral_atoms[1], res2_bonds, use_chain_operators=False)

    IC1.shift_resnum(-(site1 - 1))
    IC2.shift_resnum(-(site2 - 1))
    if len(IC1.chains) > 1 or len(IC2.chains) > 1:
        raise RuntimeError('The PDB of the label supplied appears to have a chain break. Please check your PDB and '
                         'make sure there are no chain breaks in the desired label and that there are no other '
                         'chains in the pdb file. If the error persists, check to be sure all atoms are the correct '
                         'element as chilife uses the elements to determine if atoms are bonded.')

    backbone1, aln1 = aln_sanity(IC1, resname)
    backbone2, aln2 = aln_sanity(IC2, resname)

    # If multi-state pdb extract rotamers from pdb
    if dihedrals is None:
        dihedrals = []
        for IC, resnum, dihedral_set in zip([IC1, IC2], [site1, site2], dihedral_atoms):
            dihedrals.append([ICi.get_dihedral(1, dihedral_set) for ICi in IC])

    if weights is None:
        weights = np.ones(len(IC1))

    weights /= weights.sum()
    libname = libname + f'ip{increment}'

    save_dict_1 = prep_restype_savedict(libname + 'A', resname, IC1,
                                        weights, dihedrals[0], dihedral_atoms[0],
                                        spin_atoms=spin_atoms,
                                        backbone_atoms=backbone1, aln_atoms=aln1,
                                        description=description,
                                        comment=comment, reference=reference)

    save_dict_2 = prep_restype_savedict(libname + 'B', resname, IC2,
                                        weights, dihedrals[1], dihedral_atoms[1],
                                        backbone_atoms=backbone2, aln_atoms=aln2,
                                        resi=1 + increment,
                                        spin_atoms=spin_atoms)

    # Save individual data sets and zip
    cwd = Path().cwd()
    np.savez(cwd / f'{libname}A_rotlib.npz', **save_dict_1, allow_pickle=True)
    np.savez(cwd / f'{libname}B_rotlib.npz', **save_dict_2, allow_pickle=True)
    np.save(cwd / f'{libname}_csts.npy', csts)

    with zipfile.ZipFile(f'{libname}_drotlib.zip', mode='w') as archive:
        archive.write(f'{libname}A_rotlib.npz')
        archive.write(f'{libname}B_rotlib.npz')
        archive.write(f'{libname}_csts.npy')

    # Cleanup intermediate files
    os.remove(f'{libname}A_rotlib.npz')
    os.remove(f'{libname}B_rotlib.npz')
    os.remove(f'{libname}_csts.npy')

    if permanent:
        add_library(f'{libname}_drotlib.zip', libname=libname, default=default, force=force)


def pre_add_library(
        pdb: str,
        spin_atoms: List[str],
        uniform_topology: bool = True,
        sort_atoms: bool = True,
        aln_atoms: List[str] = None,
) -> Tuple[mda.Universe, Dict]:
    """Helper function to sort PDBs, save spin atoms, update lists, etc. when adding a SpinLabel or dSpinLabel.

    Parameters
    ----------
    pdb : str
        Name (and path) of the pdb containing the new label.
    spin_atoms : List[str]
        Atoms of the SpinLabel where the unpaired electron is located.
    uniform_topology : bool
        Assume all rotamers of the library have the same topology (i.e. no differences in atom bonding). If false
        chilife will attempt to find the minimal topology shared between all rotamers for defining internal coordinates.
    sort_atoms : bool
        Switch to turn off atom sorting. Only set to true if you know what you are doing.
    aln_atoms : List[str]
        Atoms intended for rotamer ensemble alignment. This will affect sorting if using non-standard alignment atoms.
        Should only be used internally.

    Returns
    -------
    struct : MDAnalysis.Universe
        MDAnalysis Universe object containing the rotamer ensemble with each rotamer as a frame. All atoms should be
        properly sorted for consistent construction of internal coordinates.
    spin_atoms : dict
        Dictionary of spin atoms and weights if specified.
    """
    if sort_atoms:
        # Sort the PDB for optimal dihedral definitions
        pdb_lines, bonds = sort_pdb(pdb, uniform_topology=uniform_topology, return_bonds=True, aln_atoms=aln_atoms)
        bonds = get_min_topol(pdb_lines, forced_bonds=bonds)

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
        struct.add_bonds(bonds)
        os.remove(tmpfile.name)
    else:
        struct = mda.Universe(pdb, in_memory=True)
        bonds = guess_bonds(struct.atoms.positions, struct.atoms.types)
        struct.add_bonds(bonds)

    # Store spin atoms if provided
    if spin_atoms is not None:
        if isinstance(spin_atoms, str):
            spin_atoms = spin_atoms.split()

        if isinstance(spin_atoms, dict):
            spin_weights = list(spin_atoms.values())
            spin_atoms = list(spin_atoms.keys())

        else:
            w = 1 / len(spin_atoms)
            spin_weights = [w for _ in spin_atoms]

        spin_atoms = {'spin_atoms': spin_atoms, 'spin_weights': spin_weights}

    return struct, spin_atoms


def aln_sanity(internal_coords, resname, backbone_atoms=None, aln_atoms=None):
    """
    Sanity check that the alignment coordinates of mono-functional rotamer ensembles and backbone indices are correct.

    Parameters
    ----------
    internal_coords : chiLife.MolSysIC
        Internal coordinates of a rotamer ensemble.
    resname : str
        Name of the residue making up the rotamer ensemble.
    backbone_atoms: List[str]
        Names of the atoms belonging to the backbone rotamer ensemble.
    aln_atoms: List[str]
        Names of the atoms to use for alignments.

    Returns
    -------
    backbone_atoms, aln_atoms

    """
    if backbone_atoms is None and aln_atoms:
        root_idx = np.argwhere(internal_coords.atom_names == aln_atoms[1]).flat[0]
        aname_lst = internal_coords.atom_names.tolist()
        neighbor_idx = [aname_lst.index(a) for a in aln_atoms[::2]]
        backbone_atoms = get_backbone_atoms(internal_coords.topology.graph, root_idx, neighbor_idx)
        backbone_atoms = [aname_lst[idx] for idx in backbone_atoms]
    elif backbone_atoms is None:
        bb_candidates = get_bb_candidates(internal_coords.atom_names, resname)
        backbone_atoms = [b for b in bb_candidates if b in internal_coords.atom_names]

        # determine alignment atoms based on where side chain atoms connect to backbone atoms
        bb_idxs = np.argwhere(np.isin(internal_coords.atom_names, backbone_atoms)).flatten()
        root_candidates = []
        aln_candidates = []
        for bb in bb_idxs:
            neighbors = [i for i in internal_coords.topology.graph.neighbors(bb) if internal_coords.atom_types[i] !='H']
            if not all(np.isin(neighbors, bb_idxs)) and len(neighbors) > 1:
                root_candidates.append(bb)
                aln_candidates.append([n for n in neighbors if n in bb_idxs] + [bb])

        aln_candidates = [cdt for cdt in aln_candidates if len(cdt) == 3]

        if len(root_candidates) == 1:
            aln_atoms = [internal_coords.atom_names[idx] for idx in sorted(aln_candidates[0])]
        elif len(root_candidates) > 1:
            raise RuntimeError('There are more than one possible sets of alignment atoms for this residue. These '
                               'include:\n' +
                               "\n".join([f'atoms names: {[backbone_atoms[idx] for idx in idxs]}' for idxs in aln_candidates]) +
                               "\nPlease select one using the `aln_atoms` keyword argument. The alignment atoms should "
                               "be the backbone atom preceding the first side chain atom and it's two adjacent "
                               "backbone atoms")

        else:
            raise RuntimeError('The optimal set of alignment atoms to use for this residue is ambiguous. Please specify'
                               'explicitly using the `aln_atoms` keyword argument.')

    if backbone_atoms == []:
        raise RuntimeError('chiLife was unable to find a suitable set of atoms representing the macromolecular '
                           'backbone. Please ensure your rotamer ensemble uses standard protein or nucleic acid '
                           'backbone atom names, or specify the alignment atoms manually using the `aln_atoms`'
                           'keyword argument.')

    return backbone_atoms, aln_atoms


def prep_restype_savedict(
        libname: str,
        resname: str,
        internal_coords: MolSysIC,
        weights: ArrayLike,
        dihedrals: ArrayLike,
        dihedral_atoms: ArrayLike,
        sigmas: ArrayLike = None,
        resi: int = 1,
        spin_atoms: List[str] = None,
        aln_atoms: List[str] = None,
        backbone_atoms: List[str] = None,
        description: str = None,
        comment: str = None,
        reference: str = None
) -> Dict:
    """
    Helper function to add new residue types to chilife.

    Parameters
    ----------
    libname : str
        Name of residue to be stored.
    resname : str
        Residue name (3-letter code)
    internal_coords : List[MolSysIC]
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
    spin_atoms: List[str]
        A list of atom names corresponding to the atoms where the spin density resides.
    aln_atoms: List[str]
        The three atoms defining the alignment coordinate frame of the target residue. Usually the N-CA-C atoms of
        the protein backbone or similar for nucleotide backbones.
    backbone_atoms: List[str]
        Names of the atoms that correspond to backbone atoms. This includes aln_atoms and any other atoms that may need
        to be adjusted to fit the target residue backbone, e.g. the hydrogen atoms of carboxyl oxygen atoms. Any atoms
        not part of the backbone will be considered side chain atoms and may be subject to sampling.
    description : str
        A short (< 60 characters) description of the side chain library being created
    comment : str
        Additional information about the rotamer library that may not fit in description.
    reference : str
        Any relevant citations associated with the rotamer library.
    Returns
    -------
    save_dict : dict
        Dictionary of all the data needed to build a RotamerEnsemble.
    """
    # Extract coordinates and transform to the local frame

    if len(dihedrals) > 1 and len(internal_coords) == 1:
        idxs = np.zeros(len(dihedrals), dtype=int)
        new_z_matrix = internal_coords.batch_set_dihedrals(idxs, dihedrals, resi, dihedral_atoms)
        internal_coords.load_new(new_z_matrix)

    atom_types = internal_coords.atom_types.copy()
    atom_names = internal_coords.atom_names.copy()


    coords = internal_coords.protein.trajectory.coordinate_array.copy()
    if np.any(np.isnan(coords)):
        idxs = np.argwhere(np.isnan(np.sum(coords, axis=(1, 2)))).T[0]
        adxs = np.argwhere(np.isnan(np.sum(coords, axis=(0, 2)))).T[0]
        adxs = atom_names[adxs]

        raise ValueError(
            f'Coords of rotamer {" ".join((str(idx) for idx in idxs))} at atoms {" ".join((str(idx) for idx in adxs))} '
            f'cannot be converted to internal coords because there is a chain break in this rotamer. Either enforce '
            f'bonds explicitly by adding CONECT information to the PDB or fix the structure(s)')


    save_dict = {'rotlib': libname,
                 'resname': resname,
                 'coords': coords,
                 'internal_coords': internal_coords,
                 'weights': weights,
                 'atom_types': atom_types,
                 'atom_names': atom_names,
                 'dihedrals': dihedrals,
                 'dihedral_atoms': dihedral_atoms,
                 'aln_atoms': aln_atoms,
                 'backbone_atoms': backbone_atoms,
                 'description': description,
                 'comment': comment,
                 'reference': reference}

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
    save_dict['format_version'] = max(io.rotlib_formats.keys())

    return save_dict


def add_rotlib_dir(directory: Union[Path, str]) -> None:
    """
    Add a directory to search for rotlibs when none are found in te working directory. This directory will be
    searched before the chiLife default rotlib directory.

    Parameters
    ----------
    directory: Path, str
        String or Path object of the desired directory.
    """


    directory = Path(directory)
    with open(RL_DIR / 'additional_rotlib_dirs.txt', 'a+') as f:
        f.write(str(directory))
        f.write('\n')

    USER_RL_DIR.append(Path(directory))


def remove_rotlib_dir(directory: Union[Path, str]) -> None:
    """
    Remove a user added directory to search for rotlibs.

    Parameters
    ----------
    directory: Path, str
        String or Path object of the desired directory.
    """

    directory = Path(directory)
    if directory in USER_RL_DIR:
        USER_RL_DIR.remove(directory)

    with open(RL_DIR / 'additional_rotlib_dirs.txt', 'r') as f:
        TMP = [Path(line.strip()) for line in f.readlines() if Path(line.strip()) != directory]

    with open(RL_DIR / 'additional_rotlib_dirs.txt', 'w') as f:
        for p in TMP:
            f.write(str(p))
            f.write('\n')



def add_library(filename: Union[str, Path], libname: str = None, default: bool = False, force: bool = False):
    """
    Add the provided rotamer library to the chilife rotamer library directory so that it does not need to be
    in the working directory when utilizing.

    .. note:: It is generally preferred to keep a custom directory of rotlibs and use
        :func:`~chilife.chilife.add_rotlib_dir` to force chiLife to search that directory for rotlibs since this
        function adds rotlibs to the chilife install directory that can be overwritten with updates and is not
        preserved when updating python.

    Parameters
    ----------
    filename : str, Path
        Name or Path object oof the rotamer library npz file.
    libname : str
        Unique name of the rotamer library
    default : bool
        If True, chilife will make this the default rotamer library for this residue type.
    force : bool
        If True, chilife will overwrite any existing rotamer library with the same name if it exists.
    """

    store_loc = (RL_DIR / f"user_rotlibs/") / filename
    filename = Path(filename)
    if libname is None:
        libname = re.sub("_d{0,1}rotlib.(npz|zip)", "", filename.name)

    library = io.read_library(Path(filename), None, None)
    drotlib = False
    if isinstance(library, tuple):
        drotlib = True
        library, _, _ = library

    resname = str(library['resname'])
    add_to_defaults(resname, libname, default)
    if force or not store_loc.exists():
        shutil.copy(filename, str(store_loc))
        if drotlib:
            global USER_dLIBRARIES
            USER_dLIBRARIES.add(libname)
        else:
            global USER_LIBRARIES
            USER_LIBRARIES.add(libname)
        add_dihedral_def(libname, library['dihedral_atoms'].tolist(), force=force)
    else:
        raise NameError("A rotamer library with this name already exists! Please choose a different name or do"
                        "not store as a permanent rotamer library")


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
    """

    # Reload in case there were other changes
    if not add_to_toml(DATA_DIR / "dihedral_defs.toml", key=name, value=dihedrals, force=force):
        raise ValueError(f'There is already a dihedral definition for {name}. Please choose a different name.')

    # Add to active dihedral def dict
    dihedral_defs[name] = dihedrals


def remove_library(name: str, prompt: bool = True):
    """
    Removes a library from the chilife rotamer library directory and from the chilife dihedral definitions

    Parameters
    ----------
    name : str
        Name of the rotamer library
    prompt : bool
        If set to False al warnings will be ignored.
    """
    global USER_LIBRARIES, USER_dLIBRARIES
    if (name not in USER_LIBRARIES) and (name not in USER_dLIBRARIES) and prompt:
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

    if name in USER_dLIBRARIES:
        USER_dLIBRARIES.remove(name)

    if name in USER_LIBRARIES:
        USER_LIBRARIES.remove(name)

    if name in dihedral_defs:
        del dihedral_defs[name]

    remove_from_toml(DATA_DIR / 'dihedral_defs.toml', name)
    remove_from_defaults(name)


def add_to_toml(file: Union[str, Path], key: str, value: Union[str, List, dict], force: bool = False):
    """Helper function to add new key:value pairs to toml files like dihedral_defs.toml

    Parameters
    ----------
    file : str, Path
        Name or path to the toml file to be edited.
    key : str
        toml entry to be added.
    value : str, List, dict
        Item to add to the toml file under the key.
    force : bool
        If True, any existing data under the `key` entry will be overwritten.
    """
    with open(file, 'r') as f:
        local = rtoml.load(f)

    if key in local and not force:
        return False

    if key is not None:
        local[key] = value

    with open(file, 'w') as f:
        rtoml.dump(local, f)

    return True


def remove_from_toml(file: Union[str, Path], entry: str):
    """
    Remove an entry for a chilife toml file

    Parameters
    ----------
    file : str, Path
        Name or Path of the toml file to be edited.
    entry : str
        Entry or key of the toml file to be removed.
    """
    with open(file, 'r') as f:
        local = rtoml.load(f)

    if entry in local:
        del local[entry]

    with open(file, 'w') as f:
        rtoml.dump(local, f)

    return True


def add_to_defaults(resname: str, rotlib_name: str, default: bool = False):
    """
    Helper function to add a rotamer library to the defaults stack.

    Parameters
    ----------
    resname : str
        3-letter code name of the residue.
    rotlib_name : str
        Name of the rotamer library.
    default : bool
        If True the rotamer library will be added to the top of the defaults stacking making it the new default
        rotamer library for the residue type. If False it will be added to the bottom of the stack and will only be the
        default if there are no other rotamer libraries for the residue type.
    """
    file = RL_DIR / 'defaults.toml'
    with open(file, 'r') as f:
        local = rtoml.load(f)
        backup = deepcopy(local)

    if resname not in local:
        local[resname] = []
    if resname not in rotlib_defaults:
        rotlib_defaults[resname] = []

    pos = 0 if default else len(local[resname])

    if rotlib_name not in rotlib_defaults[resname]:
        rotlib_defaults[resname].insert(pos, rotlib_name)
    if rotlib_name not in local[resname]:
        local[resname].insert(pos, rotlib_name)

    safe_save(file, local, backup)


def remove_from_defaults(rotlib_name: str):
    """
    Helper function to remove a rotamer library from the defaults stack.

    Parameters
    ----------
    rotlib_name : str
        Name of the rotamer library to be removed.
    """
    file = RL_DIR / 'defaults.toml'
    with open(file, 'r') as f:
        local = rtoml.load(f)
        backup = deepcopy(local)

    keys = [key for key, val in local.items() if rotlib_name in val]

    for key in keys:
        rotlib_defaults[key].remove(rotlib_name)
        local[key].remove(rotlib_name)
        if local[key] == []:
            del local[key]

    safe_save(file, local, backup)


def safe_save(file: Union[str, Path], data: dict, backup: dict):
    """
    Helper function to save toml files. Will preserve original toml file if there is an error.

    Parameters
    ----------
    file : str, Path
        Name of the toml file to be saved.
    data : dict
        Data to save in the toml file.
    backup : dict
        Original data from the toml file to save if there is an error.
    """
    try:
        with open(file, 'w') as f:
            rtoml.dump(data, f)
    except rtoml._rtoml.TomlSerializationError:
        with open(file, 'w') as f:
            rtoml.dump(backup, f)
        raise


def _print_monofunc(files: List[Path]):
    """
    Print information of a list of mono-functional rotamer libraries.

    Parameters
    ----------
    files : List[Path]
        List of Path objects corresponding to the rotamer library files.
    """
    maxw = max([len(file.stem) for file in files]) - 6
    wrapper = textwrap.TextWrapper(width=80, subsequent_indent=" "*(maxw + 3),
                                   replace_whitespace=False, drop_whitespace=False)

    print(f"{'monofunctional' + ':':^80}")
    print("-" * 80)

    for file in files:
        with np.load(file, allow_pickle=True) as f:
            for key in f.keys():
                resname = file.stem
                if resname.endswith('_rotlib'):
                    resname = resname[:-len('_rotlib')]
                if 'description' in f:
                    descr = f['description']
                else:
                    descr = "No description"

        print("\n".join(wrapper.wrap(f'{resname:<{maxw}} : {descr}')))
    print("-" * 80)


def _print_bifunc(files):
    """
    Print information of a list of bifunctional rotamer libraries.

    Parameters
    ----------
    files : List[Path]
        List of Path objects corresponding to the rotamer library files.
    """

    maxw = max([len(file.stem) for file in files]) - 7
    wrapper = textwrap.TextWrapper(width=80, subsequent_indent=" "*(maxw + 3),
                                   replace_whitespace=False, drop_whitespace=False)


    print(f"{'bifunctional' + ':':^80}")
    print("-" * 80)

    for file in files:
        with zipfile.ZipFile(file, 'r') as archive:
            name = archive.namelist()[0]
            resname = file.stem
            context = name[3:6]
            if resname.endswith('_drotlib'):
                resname = resname[:-len('_drotlib')]
            with archive.open(name) as of:
                with np.load(of, allow_pickle=True) as f:
                    if 'description' in f:
                        descr = f['description']
                    else:
                        descr = "No description"

        print("\n".join(wrapper.wrap(f'{resname:<{maxw}} : {descr} in the {context} context')))
    print("-" * 80)


def list_available_rotlibs():
    """
    Lists residue types and rotamer libraries that are currently available. More information on any individual rotamer
    library by using the rotlib_info function.
    """

    print()

    dirs = {'current working directory': Path.cwd()}
    dirs.update({f'user rotlib directory {n + 1}': u_dir for n, u_dir in enumerate(USER_RL_DIR)})
    dirs.update({'chilife rotlib directory': RL_DIR / 'user_rotlibs'})

    for dname, dpath in dirs.items():
        monofunctional = tuple(dpath.glob('*_rotlib.npz'))
        bifunctional = tuple(dpath.glob('*_drotlib.zip'))
        if len(monofunctional) + len(bifunctional) > 0:

            print("*" * 80)
            print(f"*{f'Rotlibs in {dname}':^78}*")
            if 'user' in dname:
                print(f'*{dname.name:^78}*')
            print("*" * 80)

            if len(monofunctional) > 0 : _print_monofunc(monofunctional)
            if len(bifunctional) > 0 : _print_bifunc(bifunctional)

    print()
    print("*" * 80)
    print(f"*{'DUNBRACK ROTLIBS':^78}*")
    print(f"*{'ARG, ASN, ASP, CSY, GLN, GLU, HIS, ILE, LEU, LYS, MET, PHE, PRO, SER,':^78}*")
    print(f"*{'THR, TRP, TYR, VAL':^78}*")
    print(f"*{'(no ALA, GLY)':^78}*")
    print("*" * 80)


def rotlib_info(rotlib: Union[str, Path]):
    """
    Display detailed information about the rotamer library.

    Parameters
    ----------
    rotlib : str, Path
        Name of the rotamer library to print the information of.
    """
    mono_rotlib = io.get_possible_rotlibs(rotlib, suffix='rotlib', extension='.npz', was_none=False, return_all=True)
    bifunc_rotlib = io.get_possible_rotlibs(rotlib, suffix='drotlib', extension='.zip', was_none=False, return_all=True)

    if (mono_rotlib is None) and (bifunc_rotlib is None):
        print("No rotlib has been found with this name. Please make sure it is spelled correctly and that " /
              "it includes the path to the file if it is not a directory specified by `chilife.add_rotlib_dir`")

    if mono_rotlib is not None:
        if len(mono_rotlib) > 1:
            print('chiLife has found several mono-functional rotlibs that match the provided name:')
        for lib_file in mono_rotlib:
            _print_rotlib_info(lib_file)

    if bifunc_rotlib is not None:
        if len(bifunc_rotlib) > 1:
            print('chiLife has found several bi-functional rotlibs that match the provided name:')
        for lib_file in bifunc_rotlib:
            _print_drotlib_info(lib_file)


def _print_rotlib_info(lib_file: Union[str, Path]):
    """
    Print information about a single rotamer library file.

    Parameters
    ----------
    lib_file : Union[str, Path]
        string designating the path or a Path object to the rotamer library .npz file.
    """
    lib = io.read_rotlib(lib_file)
    wrapper = textwrap.TextWrapper(width=80, subsequent_indent="    ", replace_whitespace=False, drop_whitespace=False)

    print()
    print("*"*80)
    print(f"*  Rotamer Library Name: {lib['rotlib']:>52}  *")
    print("*"*80)
    atom_counts = ", ".join(f"{key}: {val}" for key, val in Counter(lib['atom_types']).items())

    myl = [wrapper.wrap(i) for i in (f"Rotamer Library Name: {lib['rotlib']}",
                                     f"File: {lib_file}",
                                     f"Description: {lib['description']}",
                                     f"Comment: {lib['comment']}\n",
                                     f"Length of library: {len(lib['weights'])}",
                                     f"Dihedral definitions: ",
                                     *[f'    {d}' for d in lib['dihedral_atoms']],
                                     f"Spin atoms: {lib.get('spin_atoms')}",
                                     f"Number of atoms: {atom_counts}",
                                     f"Number of heavy atoms: {np.sum(lib['atom_types'] != 'H')}",
                                     f"Reference: {lib['reference']}",
                                     f"chiLife rotlib format: {lib['format_version']}",
                                     f"*"*80)]

    print("\n".join(list(chain.from_iterable(myl))))

def _print_drotlib_info(lib_file : Union[str, Path]):
    """
    Print information about a single bifunctional rotamer library file.

    Parameters
    ----------
    lib_file : Union[str, Path]
        String designating the path or a Path object to the bifunctional  rotamer library .zip file.
    """
    lib = io.read_drotlib(lib_file)
    wrapper = textwrap.TextWrapper(width=80, subsequent_indent="    ", replace_whitespace=False, drop_whitespace=False)

    libA, libB, csts = lib
    libBmask = ~np.isin(libB['atom_names'], csts)

    print()
    print("*"*80)
    print(f"*  Rotamer Library Name: {libA['rotlib']:>52}  *")
    print("*"*80)

    atypes = np.concatenate((libA['atom_types'], libB['atom_types'][libBmask]))
    atom_counts = ", ".join(f"{key}: {val}" for key, val in Counter(atypes).items())

    myl = [wrapper.wrap(i) for i in (f"Rotamer Library Name: {libA['rotlib']}",
                                     f"File: {lib_file}",
                                     f"Description: {libA['description']}",
                                     f"Comment: {libA['comment']}\n",
                                     f"Length of library: {len(libA['weights'])}",
                                     f"Dihedral definitions: ",
                                     f"  site 1:",
                                     *[f'    {d}' for d in libA['dihedral_atoms']],
                                     f"  site 2:",
                                     *[f'    {d}' for d in libB['dihedral_atoms']],
                                     f"Spin atoms: {libA['spin_atoms']}",
                                     f"Number of atoms: {atom_counts}\n",
                                     f"Reference: {libA['reference']}",
                                     f"chiLife rotlib format: {libA['format_version']}",
                                     f"*"*80)]

    print("\n".join(list(chain.from_iterable(myl))))


def repack(
        protein: Union[mda.Universe, mda.AtomGroup],
        *spin_labels: RotamerEnsemble,
        repetitions: int = 200,
        temp: float = 1,
        energy_func: Callable = None,
        off_rotamer=False,
        **kwargs,
) -> Tuple[mda.Universe, ArrayLike]:
    """Markov chain Monte Carlo repack a protein around any number of SpinLabel or RotamerEnsemble objects.

    Parameters
    ----------
    protein : MDAnalysis.Universe, MDAnalysis.AtomGroup
        MolSys to be repacked
    spin_labels : RotamerEnsemble, SpinLabel
        RotamerEnsemble or SpinLabel object placed at site of interest
    repetitions : int
        Number of successful MC samples to perform before terminating the MC sampling loop
    temp : float, ArrayLike
        Temperature (Kelvin) for both clash evaluation and metropolis-hastings acceptance criteria. Accepts a list or
        array like object of temperatures if a temperature schedule is desired
    energy_func : Callable
        Energy function to be used for clash evaluation. Must accept a protein and RotamerEnsemble object and return an
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

    energy_func = ljEnergyFunc() if energy_func is None else energy_func


    repack_radius = kwargs.pop("repack_radius") if "repack_radius" in kwargs else None  # Angstroms
    if repack_radius is None:
        repack_radius = max([SL.clash_radius for SL in spin_labels])
    # Construct a new spin labeled protein and preallocate variables to retain monte carlo trajectory
    spin_label_str = " or ".join(f"( {spin_label.selstr} )" for spin_label in spin_labels)
    protein = mutate(protein, *spin_labels, **kwargs).atoms

    # Determine the residues near the spin label that will be repacked
    repack_residues = protein.select_atoms(f"around {repack_radius} {spin_label_str}").residues

    repack_res_kwargs = spin_labels[0].input_kwargs
    repack_res_kwargs['eval_clash'] = False
    repack_residue_libraries = [
        RotamerEnsemble.from_mda(res, **repack_res_kwargs)
        for res in repack_residues
        if res.resname not in ["GLY", "ALA"]
           and res.resname in SUPPORTED_RESIDUES
    ]
    repack_residue_libraries += list(spin_labels)

    # Create new labeled protein construct to fill in any missing atoms of repack residues
    protein = mutate(protein, *repack_residue_libraries, **kwargs).atoms

    repack_residues = protein.select_atoms(f"around {repack_radius} {spin_label_str}").residues

    repack_residue_libraries = [
        RotamerEnsemble.from_mda(res, **repack_res_kwargs)
        for res in repack_residues
        if res.resname not in ["GLY", "ALA"]
           and res.resname in SUPPORTED_RESIDUES
    ]
    repack_residue_libraries += [RotamerEnsemble(RL.res, RL.site, protein, chain=RL.chain, rotlib=RL._rotlib,
                                                 **repack_res_kwargs)
                                    for RL in spin_labels]

    traj = np.empty((repetitions, *protein.positions.shape))
    deltaEs = []


    sample_freq = np.array([len(res.dihedral_atoms) for res in repack_residue_libraries], dtype=np.float64)
    mean = sample_freq[sample_freq > 1].mean()
    sample_freq[sample_freq == 1] = mean
    sample_freq /= sample_freq.sum()

    count, acount, bcount, bidx = 0, 0, 0, 0
    schedule = repetitions / (len(temp) + 1)
    with tqdm(total=repetitions) as pbar:
        while count < repetitions:

            # Randomly select a residue from the repacked residues
            SiteLibrary = repack_residue_libraries[np.random.choice(len(repack_residue_libraries), p=sample_freq)]
            if not hasattr(SiteLibrary, "dummy_label"):
                SiteLibrary.dummy_label = SiteLibrary.copy()
                SiteLibrary.dummy_label._protein = protein
                SiteLibrary.dummy_label.mask = np.isin(protein.ix, SiteLibrary.clash_ignore_idx)
                SiteLibrary.dummy_label._coords = np.atleast_3d([protein.atoms[SiteLibrary.dummy_label.mask].positions])

                with np.errstate(divide="ignore"):
                    SiteLibrary.dummy_label.E0 = energy_func(SiteLibrary.dummy_label) - \
                                                 KT[temp[bidx]] * np.log(SiteLibrary.current_weight)

            DummyLabel = SiteLibrary.dummy_label
            coords, weight = SiteLibrary.sample(off_rotamer=off_rotamer)
            with np.errstate(divide="ignore"):
                DummyLabel._coords = np.atleast_3d([coords])
                E1 = energy_func(DummyLabel) - KT[temp[bidx]] * np.log(weight)

            deltaE = E1 - DummyLabel.E0
            deltaE = np.maximum(deltaE, -10.0)

            acount += 1
            # Metropolis-Hastings criteria
            if (E1 < DummyLabel.E0 or np.exp(-deltaE / KT[temp[bidx]]) > np.random.rand()):
                deltaEs.append(deltaE)
                protein.atoms[DummyLabel.mask].positions = coords
                DummyLabel.E0 = E1

                traj[count] = protein.atoms.positions
                SiteLibrary.update_weight(weight)
                count += 1
                bcount += 1
                pbar.update(1)
                if bcount > schedule:
                    bcount = 0
                    bidx = np.minimum(bidx + 1, len(temp) - 1)

    logging.info(f"Total counts: {acount}")

    # Load MCMC trajectory into universe.
    protein.universe.load_new(traj)

    return protein, np.squeeze(deltaEs)


def continuous_topol(atoms, bonds):
    """
    Determines if there is a chain break in the set of atoms and bonds provided.

    Parameters
    ----------
    atoms: Universe, AtomGroup, MolSys
        Set of atoms to be checked for chain breaks.

    bonds: ArrayLike[Tuple[int]]
        Array of tuples of integers defining bonds by their atom indices.

    Returns
    -------
    G.is_connected: bool
        The boolean value of whether the set of atoms are connected.

    """
    ixmap = {ix: i for i, ix in enumerate(atoms.ix)}
    ibonds = np.vectorize(ixmap.get)(bonds)
    G = ig.Graph(n=atoms.n_atoms, edges=ibonds)
    return G.is_connected()
