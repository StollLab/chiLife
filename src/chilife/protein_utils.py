import os
import pickle, math
import warnings

from pathlib import Path
from typing import Set, List, Union, Tuple

import chilife
from numpy.typing import ArrayLike
from scipy.spatial.distance import cdist
from scipy.spatial import cKDTree
from dataclasses import dataclass

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    rdkit_found = True
except:
    rdkit_found = False

import MDAnalysis
import numpy as np
from scipy.spatial.transform import Rotation

from MDAnalysis.core.topologyattrs import Atomindices, Resindices, Segindices, Segids
import MDAnalysis as mda

from .globals import SUPPORTED_RESIDUES
from .MolSys import MolecularSystemBase, MolSys, concat_molsys
from .numba_utils import get_sasa, _ic_to_cart
from .pdb_utils import get_backbone_atoms, get_bb_candidates
import chilife as xl

import chilife.scoring as scoring
import chilife.io as io

import chilife.RotamerEnsemble as re
import chilife.dRotamerEnsemble as dre

import igraph as ig


def get_dihedral_rotation_matrix(theta: float, v: ArrayLike) -> ArrayLike:
    """Build a matrix that will rotate coordinates about a vector, v, by theta in radians.

    Parameters
    ----------
    theta : float
        Rotation angle in radians.
    v : (3,) ArrayLike
        Three dimensional vector to rotate about.

    Returns
    -------
    rotation_matrix : np.ndarray
            Matrix that will rotate coordinates about the vector, V by angle theta.
    """

    # Normalize input vector
    v = v / np.linalg.norm(v)

    # Compute Vx matrix
    Vx = np.zeros((3, 3))
    Vx[[2, 0, 1], [1, 2, 0]] = v
    Vx -= Vx.T

    # Rotation matrix. See https://en.wikipedia.org/wiki/Rotation_matrix#Rotation_matrix_from_axis_and_angle
    rotation_matrix = (
            np.identity(3) * np.cos(theta)
            + np.sin(theta) * Vx
            + (1 - np.cos(theta)) * np.outer(v, v)
    )

    return rotation_matrix


def get_dihedral(p: ArrayLike) -> float:
    """Calculates dihedral of a given set of atoms, ``p`` . Returns value in radians.

     .. code-block:: python

                         3
              ------>  /
             1-------2
           /
         0

    Parameters
    ----------
    p : (4, 3) ArrayLike
        Matrix containing coordinates to be used to calculate dihedral.

    Returns
    -------
    dihedral : float
        Dihedral angle in radians.
    """

    # Unpack p
    p0, p1, p2, p3 = p

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

    dihedral = math.atan2(y, x)

    return dihedral

def get_dihedrals(p1: ArrayLike, p2: ArrayLike, p3: ArrayLike, p4: ArrayLike) -> ArrayLike:
    """Vectorized version of get_dihedral

    Parameters
    ----------
    p1 : ArrayLike
        Array containing coordinates of the first point in the dihedral.
    p2 : ArrayLike
        Array containing coordinates of the second point in the dihedral
    p3 : ArrayLike
        Array containing coordinates of the third point in the dihedral
    p4 : ArrayLike
        Array containing coordinates of the fourth point in the dihedral

    Returns
    -------
    dihedrals : ArrayLike
        Dihedral angles in radians.
    """

    # Define vectors from coordinates
    b0 = p1 - p2
    b1 = p3 - p2
    b2 = p4 - p3

    # Normalize dihedral bond vector
    b1 /= np.linalg.norm(b1, axis=-1)[:, None]

    # Calculate dihedral projections orthogonal to the bond vector
    v = b0 - np.einsum('ij,ij->i', b0, b1)[:, None] * b1
    w = b2 - np.einsum('ij,ij->i',b2, b1)[:, None] * b1

    # Calculate angle between projections
    x = np.einsum('ij,ij->i', v, w)
    y = np.einsum('ij,ij->i', np.cross(b1, v, axis=-1), w)

    dihedral = np.arctan2(y, x)

    return dihedral

def get_angle(p: ArrayLike) -> float:
    r"""Calculate the angle created by 3 points.

     .. code-block:: python

               p2
             / θ \
          p1      p3

    Parameters
    ----------
    p: ArrayLike :
        Array of three points to calculate the angle between.

    Returns
    -------
    angle : float
        Angle created by the three points.

    """
    p1, p2, p3 = p
    v1 = p1 - p2
    v2 = p3 - p2
    X = v1 @ v2
    Y = np.cross(v1, v2)
    Y = math.sqrt(Y @ Y)

    angle = math.atan2(Y, X)

    return angle


def get_angles(p1: ArrayLike, p2: ArrayLike, p3: ArrayLike) -> ArrayLike:
    r"""Vectorized version of get_angle.

    Parameters
    ----------
    p1: ArrayLike :
        Array of first points in the angles.
    p2: ArrayLike :
        Array of second points in the angles.
    p3: ArrayLike :
        Array of third points in angle.


    Returns
    -------
    angles : float
        Array of anlges.

    """
    v1 = p1 - p2
    v2 = p3 - p2
    X =  np.einsum('ij,ij->i', v1, v2)
    Y = np.cross(v1, v2, axis=-1)
    Y = np.sqrt((Y * Y).sum(axis=-1))

    angles = np.arctan2(Y, X)

    return angles


def set_dihedral(p: ArrayLike, angle: float, mobile: ArrayLike) -> ArrayLike:
    """Sets the dihedral angle by rotating all ``mobile`` atoms from their current position about the dihedral bond
    defined by the four atoms in ``p`` . Dihedral will be set to the value of ``angle`` in degrees.

    Parameters
    ----------
    p : ArrayLike
        Coordinates of atoms that define dihedral to rotate about.
    angle : float
        New angle to set the dihedral to (degrees).
    mobile : np.ndarray
        Atom coordinates to move by setting dihedral.

    Returns
    -------
    new_mobile : np.ndarray
        New positions for the mobile atoms
    """

    current = get_dihedral(p)
    angle = np.deg2rad(angle) - current
    angle = angle

    ori = p[1]
    mobile -= ori
    v = p[2] - p[1]
    v /= np.linalg.norm(v)
    R = get_dihedral_rotation_matrix(angle, v)

    new_mobile = R.dot(mobile.T).T + ori

    return new_mobile

def guess_mobile_dihedrals(ICs, aln_atoms=None):
    """
    Guess the flexible, uniqe, side chain dihedrals of a MolSysIC object.

    Parameters
    ----------
    ICs : MolSysIC
        Internal coordinates object you wish to guess dihedrals for.

    aln_atoms : List[str]
        List of atom names corresponding to the "alignment atoms" of the molecule. These are usually the core backbone
        atoms, e.g. N CA C for a protein.

    Returns
    -------
    dihedral_defs : List[List[str]]
        List of unique mobile dihedral definitions
    """
    if aln_atoms is not None:
        root_idx = np.argwhere(ICs.atom_names == aln_atoms[1]).flat[0]
        aname_lst = ICs.atom_names.tolist()
        neighbor_idx = [aname_lst.index(a) for a in aln_atoms[::2]]
        aln_idx = [neighbor_idx[0], root_idx, neighbor_idx[1]]
        bb_atoms = aln_idx + get_backbone_atoms(ICs.topology.graph, root_idx, neighbor_idx, sorted_args=aln_idx)
        bb_atoms = [ICs.atom_names[i] for i in bb_atoms]
        bb_atoms += [ICs.atom_names[i] for i in ICs.topology.graph.neighbors(root_idx) if i not in neighbor_idx]

    else:
        bb_atoms = get_bb_candidates(ICs.atom_names, ICs.resnames[0])
        bb_atoms = [atom for atom in bb_atoms if atom in ICs.atom_names]
        bb_idxs = np.argwhere(np.isin(ICs.atom_names, bb_atoms)).flatten()
        neighbors = set(sum([ICs.topology.graph.neighbors(i) for i in bb_idxs], start=[]))
        neighbor_names = [ICs.atom_names[n] for n in neighbors]
        bb_atoms += [n for n in neighbor_names if n not in bb_atoms]

    sc_mask = ~np.isin(ICs.atom_names, bb_atoms)

    ha_mask = ~(ICs.atom_types=='H')
    mask = ha_mask * sc_mask
    idxs = np.argwhere(mask).flatten()

    cyverts = ICs.topology.ring_idxs
    rotatable_bonds = {}
    _idxs = []
    for idx in idxs:
        dihedral = ICs.z_matrix_idxs[idx]
        bond = tuple(dihedral[1:3])

        # Skip duplicate dihedral defs
        if bond in rotatable_bonds:
            continue

        # Skip ring dihedrals
        elif any(all(a in ring for a in bond) for ring in cyverts):
            continue

        else:
            rotatable_bonds[bond] = dihedral
            _idxs.append(idx)

    idxs = _idxs
    dihedral_defs = [ICs.z_matrix_names[idx][::-1] for idx in idxs]
    return dihedral_defs



@dataclass
class FreeAtom:
    """Atom class for atoms in cartesian space that do not belong to a MolSys.

    Attributes
    ----------
    name : str
        Atom name
    atype : str
        Atom type
    index : int
        Atom number
    resn : str
        Name of the residue that the atom belongs to
    resi : int
        The residue index/number that the atom belongs to
    coords : np.ndarray
        The cartesian coordinates of the Atom
    """

    name: str
    atype: str
    index: int
    resn: str
    resi: int
    coords: np.ndarray


def save_ensemble(name: str, atoms: ArrayLike, coords: ArrayLike = None) -> None:
    """Save a rotamer ensemble as multiple states of the same molecule.

    Parameters
    ----------
    name : str
        file name to save rotamer ensemble to
    atoms : ArrayLike
        list of Atom objects
    coords : ArrayLike
        Array of atom coordinates corresponding to Atom objects
    """

    if not name.endswith(".pdb"):
        name += ".pdb"

    if coords is None and isinstance(atoms[0], list):
        with open(name, "w", newline="\n") as f:
            for i, model in enumerate(atoms):
                f.write(f"MODEL {i + 1}\n")
                for atom in model:
                    f.write(
                        f"ATOM  {atom.index + 1:5d}  {atom.name:<4s}{atom.resn:3s} {'A':1s}{atom.resi:4d}   "
                        f"{atom._coords[0]:8.3f}{atom._coords[1]:8.3f}{atom._coords[2]:8.3f}{1.0:6.2f}{1.0:6.2f}        "
                        f"  {atom.atype:>2s}\n"
                    )
                f.write("ENDMDL\n")

    elif len(coords.shape) > 2:
        with open(name, "w", newline="\n") as f:
            for i, model in enumerate(coords):
                f.write(f"MODEL {i + 1}\n")
                for atom, coord in zip(atoms, model):
                    f.write(
                        f"ATOM  {atom.index + 1:5d}  {atom.name:<4s}{atom.resn:3s} {'A':1s}{atom.resi:4d}   "
                        f"{coord[0]:8.3f}{coord[1]:8.3f}{coord[2]:8.3f}{1.0:6.2f}{1.0:6.2f}          {atom.atype:>2s}\n"
                    )
                f.write("ENDMDL\n")

    else:
        save_pdb(name, atoms, coords)


def save_pdb(name: Union[str, Path], atoms: ArrayLike, coords: ArrayLike, mode: str = "w") -> None:
    """Save a single state pdb structure of the provided atoms and coords.

    Parameters
    ----------
    name : str, Path
        Name or Path object of file to save
    atoms : ArrayLike
        List of Atom objects to be saved
    coords : ArrayLike
        Array of atom coordinates corresponding to atoms
    mode : str
        File open mode. Usually used to specify append ("a") when you want to add structures to a PDB rather than
        overwrite that pdb.
    """
    name = Path(name) if isinstance(name, str) else name
    name = name.with_suffix(".pdb")

    with open(name, mode, newline="\n") as f:
        f.write('MODEL\n')
        for atom, coord in zip(atoms, coords):
            f.write(
                f"ATOM  {atom.index + 1:5d} {atom.name:^4s} {atom.resn:3s} {'A':1s}{atom.resi:4d}    "
                f"{coord[0]:8.3f}{coord[1]:8.3f}{coord[2]:8.3f}{1.0:6.2f}{1.0:6.2f}          {atom.atype:>2s}  \n"
            )
        f.write('ENDMDL\n')


def get_missing_residues(
        protein: Union[MDAnalysis.Universe, MDAnalysis.AtomGroup],
        ignore: Set[int] = None,
        use_H: bool = False,
        ignore_waters = True
) -> List:
    """Get a list of RotamerEnsemble objects corresponding to the residues of the provided protein that are missing
    heavy atoms.

    Parameters
    ----------
    protein : MDAnalysis.Universe, MDAnalysis.AtomGroup
        MolSys to search for residues with missing atoms.
    ignore : set
        List of residue numbers to ignore. Usually sites you plan to label or mutate.
    use_H : bool
        Whether the new side chain should have hydrogen atoms.

    Returns
    -------
    missing_residues : list
        A list of RotamerEnsemble objects corresponding to residues with missing heavy atoms.
    """
    ignore = set() if ignore is None else ignore
    missing_residues = []
    cache = {}

    for res in protein.residues:
        # Only consider supported residues because otherwise chiLife wouldn't know what's missing
        if (
                res.resname not in SUPPORTED_RESIDUES
                or res.resnum in ignore
                or res.resname in ["ALA", "GLY"]
        ):
            continue

        # Check if there are any missing heavy atoms
        heavy_atoms = res.atoms.types[res.atoms.types != "H"]
        resn = res.resname

        a = cache.setdefault(resn, len(re.RotamerEnsemble(resn).atom_names))
        if len(heavy_atoms) != a:
            missing_residues.append(
                re.RotamerEnsemble(
                    res.resname,
                    res.resnum,
                    protein=protein,
                    chain=res.segid,
                    use_H=use_H,
                    ignore_waters=ignore_waters
                )
            )

    return missing_residues


def mutate(
        protein: MDAnalysis.Universe,
        *ensembles: 'RotamerEnsemble',
        add_missing_atoms: bool = True,
        rotamer_index: Union[int, str, None] = None,
        use_H: bool = None,
        ignore_waters: bool = None
        ) -> MDAnalysis.Universe:
    """Create a new Universe where the native residue is replaced with the highest probability rotamer from a
    RotamerEnsemble or SpinLabel object.

    Parameters
    ----------
    protein : MDAnalysis.Universe
        An MDA Universe object containing protein to be spin labeled
    ensembles : RotamerEnsemble, SpinLabel
        Precomputed RotamerEnsemble or SpinLabel object to use for selecting and replacing the spin native amino acid
    rotamer_index : int, str, None
        Index of the rotamer to be used for the mutation. If None, the most probable rotamer will be used. If 'all',
        mutate will return an ensemble with each rotamer, if random, mutate will return an ensemble with a random
        rotamer.
    add_missing_atoms : bool
        Model side chains missing atoms if they are not present in the provided structure.
    ignore_waters : bool
        ignore waters when selecting conforers for mutation.
    Returns
    -------
    U : MDAnalysis.Universe
        New Universe with a copy of the spin labeled protein with the highest probability rotamer
    """

    # Check for dRotamerEnsembles in ensembles
    temp_ensemble = []
    for lib in ensembles:
        if isinstance(lib, (re.RotamerEnsemble, dre.dRotamerEnsemble)):
            temp_ensemble.append(lib)
        else:
            raise TypeError(f"mutate only accepts (d)RotamerEnsemble and (d)SpinLabel objects, not {lib}.")

    ensembles = temp_ensemble

    if add_missing_atoms:
        use_H = use_H if use_H is not None else param_from_rotlibs('use_H', ensembles)
        ignore_waters = ignore_waters if ignore_waters is not None else param_from_rotlibs('ignore_waters', ensembles)

        missing_residues = get_missing_residues(protein,
                                                ignore={res.site for res in ensembles},
                                                use_H=use_H,
                                                ignore_waters=ignore_waters)

        ensembles = list(ensembles) + missing_residues

    label_sites = {}
    for spin_label in ensembles:
        if isinstance(spin_label, dre.dRotamerEnsemble):
            label_sites[spin_label.site1, spin_label.icode1, spin_label.chain] = spin_label
            label_sites[spin_label.site2, spin_label.icode2, spin_label.chain] = spin_label
        else:
            label_sites[spin_label.site, spin_label.icode, spin_label.chain] = spin_label

    # Remove waters if they are being ignored. 
    if ignore_waters:
        protein = protein.select_atoms(f'(not altloc B) and (not (byres name OH2 or resname HOH))')
    else:
        protein = protein.select_atoms(f'(not altloc B)')
        
    label_selstr = " or ".join([f"({label.selstr})" for label in ensembles])
    other_atoms = protein.select_atoms(f"not ({label_selstr})")

    resids = [res.resid for res in protein.residues]
    icodes = [res.icode for res in protein.residues] if hasattr(protein, "icodes") else None
    # Allocate lists for universe information
    atom_info = []
    res_names = []
    segidx = []

    # Loop over residues in old universe
    for i, res in enumerate(protein.residues):
        resloc = (res.resid, res.icode, res.segid)

        # If the residue is the spin labeled residue replace it with the highest probability spin label
        if resloc in label_sites:
            rot_ens = label_sites[resloc]
            if isinstance(rot_ens, dre.dRotamerEnsemble):
                r1l = len(rot_ens.rl1mask)
                r2l = len(rot_ens.rl2mask)
                both = r1l + r2l

                if resloc[0] == rot_ens.site1:
                    # Add site 1
                    atom_info += [
                        (i, name, atype)
                        for name, atype in zip(rot_ens.atom_names[:r1l], rot_ens.atom_types[:r2l])
                    ]

                elif resloc[0] == rot_ens.site2:
                    atom_info += [
                        (i, name, atype)
                        for name, atype in zip(rot_ens.atom_names[r1l:r1l + r2l], rot_ens.atom_types[r1l:r1l + r2l])
                    ]
                    # Add cap
                    atom_info += [
                        (i, name, atype)
                        for name, atype in zip(rot_ens.atom_names[both:], rot_ens.atom_types[both:])
                    ]
                else:
                    raise RuntimeError("The residue specified is not part of the dRotamerEnsemble being constructed")
            else:
                atom_info += [
                    (i, name, atype)
                    for name, atype in zip(rot_ens.atom_names, rot_ens.atom_types)
                ]

            # Add missing Oxygen from rotamer ensemble
            res_names.append(rot_ens.res)
            segidx.append(rot_ens.segindex)

        # Else retain the atom information from the parent universe
        else:
            atom_info += [
                (i, atom.name, atom.type) for atom in res.atoms if atom.altLoc != "B"
            ]
            res_names.append(res.resname)
            segidx.append(res.segindex)

    # Reindex segments in case any were dropped from the parent universe
    idxmap = {idx: i for i, idx in enumerate(np.unique(segidx))}
    segidx = np.fromiter((idxmap[idx] for idx in segidx), dtype=int)

    # Unzip atom information into individual lists
    residx, atom_names, atom_types = zip(*atom_info)
    segids = protein.segments.segids
    # Allocate a new universe with the appropriate information

    if isinstance(protein, (mda.Universe, mda.AtomGroup)):
        U = make_mda_uni(atom_names, atom_types, res_names, residx, resids, segidx, segids, icodes=icodes)
    elif isinstance(protein, MolecularSystemBase):
        U = MolSys.from_arrays(atom_names, atom_types, res_names, residx, resids, segidx, segids=segids, icodes=icodes)

    # Apply old coordinates to non-spinlabel atoms
    new_other_atoms = U.select_atoms(f"not ({label_selstr})")
    new_other_atoms.atoms.positions = other_atoms.atoms.positions

    if rotamer_index == 'all':
        max_label_len = max([len(label) for label in label_sites.values()])
        coordinates = np.array([U.atoms.positions.copy() for _ in range(max_label_len)])
        U.load_new(coordinates)

        for i, ts in enumerate(U.trajectory):
            for spin_label in label_sites.values():
                sl_atoms = U.select_atoms(spin_label.selstr)
                if len(spin_label) <= i:
                    sl_atoms.positions = spin_label.coords[-1]
                else:
                    sl_atoms.positions = spin_label.coords[i]

    else:
        for spin_label in label_sites.values():
            sl_atoms = U.select_atoms(spin_label.selstr)
            if rotamer_index == 'random':
                rand_idx = np.random.choice(len(spin_label.coords), p=spin_label.weights)
                sl_atoms.atoms.positions = spin_label.coords[rand_idx]
            elif isinstance(rotamer_index, int):
                sl_atoms.atoms.positions = spin_label.coords[rotamer_index]
            else:
                sl_atoms.atoms.positions = spin_label.coords[np.argmax(spin_label.weights)]

    return U


def randomize_rotamers(
        protein: Union[mda.Universe, mda.AtomGroup],
        rotamer_libraries: List['RotamerEnsemble'],
        **kwargs,
) -> None:
    """Modify a protein object in place to randomize side chain conformations.

    Parameters
    ----------
    protein : MDAnalysis.Universe, MDAnalysis.AtomGroup
        MolSys object to modify.
    rotamer_libraries : list
        RotamerEnsemble objects attached to the protein corresponding to the residues to be repacked/randomized.
    **kwargs : dict
        Additional Arguments to pass to ``sample`` method. See :mod:`sample <chiLife.RotamerEnsemble.sample>` .
    """
    for rotamer in rotamer_libraries:
        coords, weight = rotamer.sample(off_rotamer=kwargs.get("off_rotamer", False))
        mask = ~np.isin(protein.ix, rotamer.clash_ignore_idx)
        protein.atoms[~mask].positions = coords


def param_from_rotlibs(param: str, ensembles: List['RotamerEnsemble']):
    """
    Get the value of a parameter used across a set of rotamer ensembles if that parameter is consistent. If the
    parameter is not the same value for all rotamer ensembles, an AttributeError will be thrown.
    Parameters
    ----------
    param : str
        The name of the parameter.
    ensembles : List[RotamerEnsembles]
        The rotamer ensembles to search for the parameter int

    Returns
    -------
    param_value: Any
        The value of the parameter across all the rotamer libarries.
    """
    # If there are no residues being mutated just return defaults
    if len(ensembles) == 0:
        defaults = {}
        defaults.update(re.assign_defaults(defaults))
        return defaults[param]

    # Otherise check to make sure they are all the same
    ensemble_params = [getattr(ensemble, param, None) for ensemble in ensembles]
    if all(e == ensemble_params[0] for e in ensemble_params[1:]):
        param_value = ensemble_params[0]
        return param_value

    # And return an error if they are not
    else:
        raise AttributeError(f"User provided ensembles with different {param} parameters. Make sure all "
                             f"ensembles use the same value for {param}")



def get_sas_res(
        protein: Union[mda.Universe, mda.AtomGroup],
        cutoff: float = 30,
        forcefield = 'charmm',
) -> Set[Tuple[int, str]]:
    """Run FreeSASA to get solvent accessible surface residues in the provided protein

    Parameters
    ----------
    protein : MDAnalysis.Universe, MDAnalysis.AtomGroup
        MolSys object to measure Solvent Accessible Surfaces (SAS) area of and report the SAS residues.
    cutoff : float
        Exclude residues from list with SASA below cutoff in angstroms squared.
    forcefield : Union[str, chilife.ljParams]
        Forcefiled to use defining atom radii for calculating solvent accessibility.

    Returns
    -------
    SAResi : set
        A set of solvent accessible surface residues.

    """
    if isinstance(forcefield, str):
        forcefield = scoring.ljEnergyFunc(forcefield)

    environment_coords = protein.atoms.positions
    environment_radii = forcefield.get_lj_rmin(protein.atoms.types)
    atom_sasa = get_sasa(environment_coords, environment_radii, by_atom=True)

    SASAs = {(residue.resnum, residue.segid) for residue in protein.residues if
             atom_sasa[0, residue.atoms.ix].sum() >= cutoff}

    return SASAs


def pose2mda(pose) -> MDAnalysis.Universe:
    """Create an MDAnalysis universe from a pyrosetta pose

    Parameters
    ----------
    pose : pyrosetta.rosetta.core.Pose
        pyrosetta pose object.

    Returns
    -------
    mda_protein : MDAnalysis.Universe
        Copy of the input pose as an MDAnalysis Universe object
    """
    coords = np.array(
        [
            res.xyz(atom)
            for res in pose.residues
            for atom in range(1, res.natoms() + 1)
            if res.atom_type(atom).element().strip() != "X"
        ]
    )
    atypes = np.array(
        [
            str(res.atom_type(atom).element()).strip()
            for res in pose.residues
            for atom in range(1, res.natoms() + 1)
            if res.atom_type(atom).element().strip() != "X"
        ]
    )
    anames = np.array(
        [
            str(res.atom_name(atom)).strip()
            for res in pose.residues
            for atom in range(1, res.natoms() + 1)
            if res.atom_type(atom).element().strip() != "X"
        ]
    )
    resindices = np.array(
        [
            res.seqpos() - 1
            for res in pose.residues
            for atom in range(1, res.natoms() + 1)
            if res.atom_type(atom).element().strip() != "X"
        ]
    )

    n_residues = len(pose)

    segindices = np.array([0] * n_residues)
    resnames = np.array([res.name() for res in pose])
    resnums = np.array([res.seqpos() for res in pose])

    mda_protein = make_mda_uni(anames, atypes, resnames, resindices, resnums, segindices)
    mda_protein.positions = coords

    return mda_protein


def xplor2mda(xplor_sim) -> MDAnalysis.Universe:
    """
    Converts an Xplor-NIH xplor.simulation object to an MDAnalysis Universe object

    Parameters
    ----------
    xplor_sim : xplor.simulation
        an Xplor-NIH simulation object. (See https://nmr.cit.nih.gov/xplor-nih/doc/current/python/ref/simulation.html)
    Returns
    -------

    """
    n_atoms = xplor_sim.numAtoms()
    a_names = np.array([xplor_sim.atomName(i) for i in range(n_atoms)])
    a_types = np.array([xplor_sim.chemType(i)[0] for i in range(n_atoms)])

    resnames = np.array([xplor_sim.residueName(i) for i in range(len(xplor_sim.residueNameArr()))])
    resnums = np.array(xplor_sim.residueNumArr())

    _, uidx = np.unique(resnums, return_index=True)
    _, resindices = np.unique(resnums, return_inverse=True)

    resnums = resnums[uidx]
    resnames = resnames[uidx]

    n_residues = len(resnames)
    segindices = np.array([0] * n_residues)

    mda_protein = make_mda_uni(a_names, a_types, resnames, resindices, resnums, segindices)
    mda_protein.atoms.positions = np.array(xplor_sim.atomPosArr())

    return mda_protein


def make_mda_uni(anames: ArrayLike,
                 atypes: ArrayLike,
                 resnames: ArrayLike,
                 resindices: ArrayLike,
                 resnums: ArrayLike,
                 segindices: ArrayLike,
                 segids: ArrayLike = None,
                 icodes: ArrayLike = None,
                 ) -> MDAnalysis.Universe:
    """
    Create an MDAnalysis universe from numpy arrays of atom information.

    Parameters
    ----------
    anames : ArrayLike
        Array of atom names. Length should be equal to the number of atoms.
    atypes : ArrayLike
        Array of atom elements or types. Length should be equal to the number of atoms.
    resnames : ArrayLike
        Array of residue names. Length should be equal to the number of residues.
    resindices : ArrayLike
        Array of residue indices. Length should be equal to the number of atoms. Elements of resindices should
        map to resnames and resnums of the atoms they represent.
    resnums : ArrayLike
        Array of residue numbers. Length should be equal to the number of residues.
    segindices : ArrayLike
        Array of segment/chain indices. Length should be equal to the number of residues.
    segids : ArrayLike, None
        Array of segment/chain IDs. Length should be equal to the number of segs/chains.

    Returns
    -------
    mda_uni : MDAnalysis.Universe
        The Universe created by the function.
    """

    n_atoms = len(anames)
    n_residues = len(np.unique(resindices))

    if segids is None:
        segids = np.array(["ABCDEFGHIJKLMNOPQRSTUVWXYZ"[i] for i in range(len(np.unique(segindices)))])
    elif len(segids) != len(np.unique(segindices)):
        pass

    if icodes is None:
        icodes = np.array(["" for _ in range(n_residues)])

    mda_uni = mda.Universe.empty(
        n_atoms,
        n_residues=n_residues,
        atom_resindex=resindices,
        residue_segindex=segindices,
        trajectory=True,
    )

    mda_uni.add_TopologyAttr("type", atypes)
    mda_uni.add_TopologyAttr("resnum", resnums)
    mda_uni.add_TopologyAttr("resids", resnums)
    mda_uni.add_TopologyAttr("icodes", icodes)
    mda_uni.add_TopologyAttr("resname", resnames)
    mda_uni.add_TopologyAttr("name", anames)
    mda_uni.add_TopologyAttr("altLocs", [""] * len(atypes))
    mda_uni.add_TopologyAttr("segid")

    for i, segid in enumerate(segids):
        if i == 0:
            i_segment = mda_uni.segments[0]
            i_segment.segid = segid
        else:
            i_segment = mda_uni.add_Segment(segid=str(segid))

        mask = np.argwhere(np.asarray(segindices) == i).flatten()
        mda_uni.residues[mask.tolist()].segments = i_segment

    mda_uni.add_TopologyAttr(Segids(np.array(segids)))
    mda_uni.add_TopologyAttr(Atomindices())
    mda_uni.add_TopologyAttr(Resindices())
    mda_uni.add_TopologyAttr(Segindices())

    return mda_uni

def template_ICs(template):
    """
    Get the internal coordinate parameters (bond angle and dihedral angle) of the protein backbone atoms of the
    template.


    Parameters
    ----------
    template : Union[str, Path, MolSys]
        The PDB file or MolSys object to serve as the template.

    Returns
    -------
    phi, psi, omega, bond_angles : ArrayLike
        Arrays containing backbone dihedral and bond angles as calculated from the template structure.
    """

    if isinstance(template, (str, Path)):
        template = chilife.MolSys.from_pdb(template)
    elif isinstance(template, (mda.Universe, mda.AtomGroup)):
        template = chilife.MolSys.from_atomsel(template)
    elif not isinstance(template, MolecularSystemBase):
        raise RuntimeError('Template must be a PDB file or a MDA.Universe, AtomGroup or chiLife.MolSys object.')


    Ns  = template.select_atoms('name N')
    CAs = template.select_atoms('name CA')
    Cs = template.select_atoms('name C')
    Os = template.select_atoms('name O')

    phis = get_dihedrals(Cs.positions[:-1], Ns.positions[1:], CAs.positions[1:], Cs.positions[1:])
    psis = get_dihedrals(Ns.positions[:-1], CAs.positions[:-1], Cs.positions[:-1], Ns.positions[1:])
    omegas = get_dihedrals(CAs.positions[:-1], Cs.positions[:-1], Ns.positions[1:], CAs.positions[1:])

    # Add to phis and psis for N_CA_C_O and H_N_CA_C

    BA1 = np.concatenate([[0.], get_angles(CAs.positions[:-1], Cs.positions[:-1], Ns.positions[1:])])
    BA2 = np.concatenate([[0.], get_angles(Cs.positions[:-1], Ns.positions[1:], CAs.positions[1:])])
    BA3 = get_angles(Ns.positions, CAs.positions, Cs.positions)
    BA4 = get_angles(CAs.positions, Cs.positions, Os.positions)
    bond_angles = np.array([BA1, BA2, BA3, BA4]).T

    return tuple(np.rad2deg(x) for x in (phis, psis, omegas, bond_angles))


def make_peptide(sequence: str, phi=None, psi=None, omega=None, bond_angles=None, template=None) -> MolSys:
    """
    Create a peptide from a string of amino acids. chilife NCAA rotamer libraries and smiles can be inserted by using
    square brackets and angle brackets respectively , e.g. ``[ACE]AA<C1=CC2=C(C(=C1)F)C(=CN2)CC(C(=O)O)N>AA[NME]``
    where ACE and NME are peptide caps and <C1=CC2=C(C(=C1)F)C(=CN2)CC(C(=O)O)N> is a smiles for a NCAA. Backbone
    torsion and bond angles can be set using the phi, psi, omega, and bond_angel keyword argument. All angles should
    be passed in degrees. Alternativly backbone angles can be set with a protein template that can either be a MolSys
    object, or a sting/Path object pointing to a PDB file.


    Parameters
    ----------
    sequence : str
        The Amino acid sequence.

    phi : ArrayLike
        N-1 length array specifying the peptide backbone phi angles (degrees).
    psi : ArrayLike
        N-1 length array specifying the peptide backbone psi angles (degrees).
    omega : ArrayLike
        N-1 length array specifying the peptide backbone omega angles (degrees).
    bond_angles : ArrayLike
        Nx4 shaped array specifying the 4 peptide backbone bond angles  (degrees). Note that, for the first residue any
        value for the first and second bond angles will be ignored because they are undefined.

        1 - CA(i-1)-C(i-1)-N
        2 - C(i-1)_N_CA
        3 - N-CA-C
        4 - CA-C-O

    template : str, Path, MolSys


    Returns
    -------
    mol : MolSys
        A chiLife MolSys object
    """
    C_N_LENGTH = 1.34
    N_CA_LENGTH = 1.46
    CA_C_LENGTH = 1.54

    CA_C_N_ANGLE = 1.9897
    C_N_CA_ANGLE = 2.1468
    N_CA_C_ANGLE = 1.9199
    base_IC = {}

    # Strip whitespace from multiline strings
    sequence = sequence.strip()

    # Parse sequences
    sequence = parse_sequence(sequence)
    zmat = []
    zmat_idxs = []
    anames = []
    atypes = []
    resnames = []
    resindices = []
    resnums = []

    seqiter = iter(sequence)
    atom_idx = 0
    residue_idx = 0
    prev_N = 0

    ncap = None
    ccap = None

    if template is None:
        pass
    elif all(x is None for x in (phi, psi, omega, bond_angles)):
        phi, psi, omega, bond_angles = template_ICs(template)
    else:
        raise RuntimeError('You can not pass both a template and explicit bond_angle or backbone dihedral values')

    # Alter definitions to be based off of other atoms, e.g. N-CA-C-O instead of N-CA-C-N
    phi = np.deg2rad(phi) if phi is not None else np.ones(len(sequence)) * -0.82030
    psi = np.deg2rad(psi) if psi is not None else np.ones(len(sequence)) * -0.99484
    omega = np.deg2rad(omega) if omega is not None else np.ones(len(sequence)) * -np.pi

    if len(phi) < len(sequence):
        diff = len(sequence) - len(phi)
        if diff > 3:
            raise RuntimeError('The number of phi backbone dihedrals does not match the number of residues')
        phi = (np.concatenate(([-0.82030], phi)) if diff == 1 else
               np.concatenate(([-0.82030], phi, [-0.82030])) if diff == 2 else
               np.concatenate(([-0.82030, -0.82030], phi, [-0.82030])))

    if len(psi) < len(sequence):
        diff = len(sequence) - len(psi)
        if diff > 3:
            raise RuntimeError('The number of psi backbone dihedrals does not match the number of residues')
        psi = (np.concatenate(([-0.99484], psi)) if diff == 1 else
               np.concatenate(([-0.99484, -0.99484], psi)) if diff == 2 else
               np.concatenate(([-0.99484], psi, [-0.99484, -0.99484])))

    if len(omega) < len(sequence):
        diff = len(sequence) - len(omega)
        if diff > 3:
            raise RuntimeError('The number of omega backbone dihedrals does not match the number of residues')
        omega = (np.concatenate(([-np.pi], omega)) if diff == 1 else
                 np.concatenate(([-np.pi, -np.pi], omega, [])) if diff == 2 else
                 np.concatenate(([-np.pi], omega, [-np.pi, -np.pi])))

    for i, res in enumerate(seqiter):
        if res in base_IC:
            msysIC = base_IC[res]
        elif res in xl.nataa_codes or res in xl.SUPPORTED_BB_LABELS:
            with open(chilife.RL_DIR / f"residue_internal_coords/{res.lower()}_ic.pkl", 'rb') as f:
                msysIC = pickle.load(f)
            base_IC[res] = msysIC
        elif res in xl.SUPPORTED_RESIDUES:
            RL = xl.RotamerEnsemble(res, use_H=True)
            idxmax = np.argmax(RL.weights)
            msysIC = RL.internal_coords
            msysIC.trajectory[idxmax]
            base_IC[res] = msysIC
        elif res in xl.ncaps:
            ncap = res
            continue
        elif res in xl.ccaps:
            ccap = res
            continue
        else:
            mol = smiles2residue(res)
            msysIC = xl.MolSysIC.from_atoms(mol)

        anames.append(msysIC.atom_names)
        atypes.append(msysIC.atom_types)
        resnames.append(msysIC.atom_resnames)
        resnums.append(msysIC.atom_resnums + residue_idx)
        resindices.append(msysIC.atom_resnums + residue_idx-1)

        # Set backbone dihedrals
        if i < len(sequence)-1:
            msysIC.set_dihedral(psi[i+1] + np.pi, 1, ['N', 'CA', 'C', 'O'])
        if 'H' in msysIC.atom_names:
            msysIC.set_dihedral(phi[i] + np.pi, 1, ['C', 'CA', 'N', 'H'])

        tzmat = msysIC.z_matrix.copy()
        tzmat_idxs = msysIC.z_matrix_idxs.copy()

        if atom_idx != 0:
            tzmat_idxs += atom_idx
            tzmat_idxs[0] = [atom_idx, prev_N + 2, prev_N + 1, prev_N]
            tzmat_idxs[1] = [atom_idx + 1, atom_idx, prev_N + 2, prev_N + 1]
            tzmat_idxs[2] = [atom_idx + 2, atom_idx + 1, atom_idx, prev_N + 2]

            tzmat[0] = [C_N_LENGTH, CA_C_N_ANGLE, psi[i]]
            tzmat[1] = [N_CA_LENGTH, C_N_CA_ANGLE, omega[i]]
            tzmat[2] = [CA_C_LENGTH, N_CA_C_ANGLE, phi[i]]

            prev_N = atom_idx

        zmat.append(tzmat)
        zmat_idxs.append(tzmat_idxs)

        atom_idx = atom_idx + len(tzmat)
        residue_idx += 1

    anames = np.concatenate(anames)
    atypes = np.concatenate(atypes)
    resnames = np.concatenate(resnames)
    resnums = np.concatenate(resnums)
    resindices = np.concatenate(resindices)
    zmat = np.concatenate(zmat)
    zmat_idxs = np.concatenate(zmat_idxs)
    segindices = np.array([0] * len(anames))
    segids = np.array(['A'] * len(anames))

    if bond_angles is not None:
        assert len(bond_angles) == len(np.unique(resindices))
        for i, atom_name in enumerate(('N', 'CA', 'C', 'O')):
            offset = 1 if atom_name in ('N', 'CA') else 0
            idxs = np.argwhere(anames == atom_name).flat[offset:]
            zmat[idxs, 1] = np.deg2rad(bond_angles[offset:, i])

    trajectory = _ic_to_cart(zmat_idxs[:, 1:], zmat)

    mol = MolSys.from_arrays(anames, atypes, resnames, resindices, resnums, segindices,
                             segids=segids, trajectory=trajectory)

    if ncap is not None:
        d = phi[1] if phi[1] != -0.82030 else None
        mol = append_cap(mol, ncap, dihedral=d)
    if ccap is not None:
        d = psi[-1] if psi[-1] != -0.99484 else None
        mol = append_cap(mol, ccap, dihedral=d)

    return mol


def parse_sequence(sequence: str) -> List[str]:
    """
    Input a string of amino acids with mized 1-letter codes, square brackted ``[]`` 3-letter codes or angle ``<>``
    bracketed smiles and return a list of 3-letter codes and smiles.

    Parameters
    ----------
    sequence : str
    The Amino acid sequence.

    Returns
    -------
    parsed_sequence : List[str]
    A list of the amino acid sequences.

    """
    parsed_sequence = []
    seqiter = iter(sequence)
    for aa in seqiter:
        if aa.upper() in chilife.nataa_codes:
            parsed_sequence.append(chilife.nataa_codes[aa.upper()])

        # Parse chiLife compatible NCAAs
        elif aa == '[':
            code = ""
            aa = next(seqiter)
            while aa != ']':
                code += aa
                try:
                    aa = next(seqiter)
                except StopIteration:
                    raise RuntimeError("Cannot parse sequence because there is an unbalaced square bracket ``[]``.")
            parsed_sequence.append(code)

        elif aa == "<":
            smiles = ""
            aa = next(seqiter)
            while aa != '>':
                smiles += aa
                try:
                    aa = next(seqiter)
                except:
                    raise RuntimeError("Cannot parse sequence because there is an unbalnced angle bracket ``<>``.")

            parsed_sequence.append(smiles)

        elif aa in ('>', ']'):
            raise RuntimeError(f"A terminating ``{aa}`` bracket has been detected, but no opening bracket was placed "
                               f"ahead of it")

        else:
            raise RuntimeError(f'``{aa}`` is not a known amino acid')

    return parsed_sequence


def smiles2residue(smiles : str, **kwargs) -> MolSys:
    """
    Create a protein residue from a smiles string. Smiles string must contain an N-C-C-O dihedral to identify the
    protein backbone. If no dihedral is found or there are too many N-C-C-O dihedrals that are indistinguishable
    this function will fail.


    Parameters
    ----------
    smiles : str
        SMILES string to convert to 3d and a residue
    kwargs : dict
        Keyword arguments to pass to rdkit.AllChem.EmbedMolecule (usually randomSeed)

    Returns
    -------
    msys : MolSys
        chiLife molecular system object containing the smiles string as a 3D molecule and single residue.
    """

    if not rdkit_found:
        raise RuntimeError("Using smiles2residue or make_peptide with a smile string requires rdkit to be installed.")

    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)

    success = AllChem.EmbedMolecule(mol, **kwargs)
    if success != 0:
        AllChem.EmbedMolecule(mol, enforceChirality=False, **kwargs)

    AllChem.MMFFOptimizeMolecule(mol, maxIters=200)

    res = MolSys.from_rdkit(mol)

    atoms = res.atoms
    res.filename = 'UNK.pdb'
    Natoms = atoms.select_atoms('type N')
    NCCOs = []
    for atom in Natoms:
        for dih in atom.dihedrals:
            if np.all(dih.atoms.types == ['N', 'C', 'C', 'O']):
                NCCOs.append(dih.atoms)
            elif np.all(dih.atoms.types == ['O', 'C', 'C', 'N']):
                NCCOs.append(dih.atoms[::-1])

    bb_candidates = []
    for dih in NCCOs:
        assert dih[-1].type == 'O'

        if len(dih[-1].bonds) == 1:
            bb_candidates.append(dih)

    if len(bb_candidates) == 1:
        bb = bb_candidates[0]

    else:
        bb = None
        for can in bb_candidates:
            cooh = "".join(sorted(can[2].bonded_atoms.types))
            if cooh != 'COO':
                continue
            else:
                bb = can
                break

    bb.names = ['N', 'CA', 'C', 'O']
    count_dict = {}
    for atom in atoms:

        if atom in bb:
            pass

        elif 'N' in atom.bonded_atoms.names:
            if atom.type == 'H':
                atom.name = count_dict.get('X', 'H')
                count_dict['X'] = 'X'

        elif 'CA' in atom.bonded_atoms.names and atom.type == 'H':
            atom.name = 'HA'

        elif 'O' in atom.bonded_atoms.names and atom.type == 'H':
            atom.name = 'X'

        elif 'C' in atom.bonded_atoms.names and atom.type == 'O':
            atom.name = 'X'
            for oatom in atom.bonded_atoms:
                if oatom.type == 'H':
                    oatom.name ='X'
        elif atom.name == 'X':
            pass

        else:
            n = count_dict.setdefault(atom.type, 1)
            atom.name = f'{atom.type}{n}'
            count_dict[atom.type] += 1

    save_atoms = atoms.select_atoms('not name X')
    xl.save('UNK.pdb', save_atoms)
    Msys = MolSys.from_pdb('UNK.pdb', sort_atoms=True)
    os.remove('UNK.pdb')

    return Msys


def append_cap(mol : MolSys, cap : str, resnum = None, dihedral: float = None) -> MolSys:
    """
    Append a peptide cap to a molecular system.

    Parameters
    ----------
    mol : MolSys
        The molecular system to be capped.
    cap : str
        The name (3-letter identifier) of the cap to be attached.

    Returns
    -------
    mol : MolSys
        The molecular system with the cap.
    """

    cap_name = cap.upper()
    term = "N" if cap_name in xl.ncaps else "C" if cap_name in xl.ccaps else None

    if resnum is not None:
        neighbor = mol.select_atoms(f'resnum {resnum}')
    elif term == "N":
        neighbor = mol.residues[0]
        resnum = neighbor.resnum
    elif term == "C":
        neighbor = mol.select_atoms('protein').residues[-1]
        resnum = neighbor.resnum
    else:
        raise RuntimeError('`term` not found in neighboring residue. Note that `term` must be an `N` or a `C`.')

    nmx, nori = xl.global_mx(*neighbor.select_atoms('name N CA C').positions)
    cap_struct = xl.MolSys.from_pdb(xl.RL_DIR / f'cap_pdbs/{cap_name}.pdb')
    cap_struct.positions = cap_struct.positions @ nmx + nori
    cap_struct.resnum = resnum + 1 if term == "C" else resnum - 1

    if dihedral is not None:
        # Identify nearst atom of cap as bonded atom
        neighbor_atom = neighbor.atoms[neighbor.names == term].position[0]
        bonded_atom = np.linalg.norm(cap_struct.positions - neighbor_atom, axis=1).argmin()
        bonded_atom = cap_struct.atoms[bonded_atom]
        
        # Set up dih
        if term == 'N':
            dihedral_points = bonded_atom.positions, *neighbor.select_atoms('name N CA C').positions
        else:
            dihedral_points = *neighbor.select_atoms('name N CA C').positions, bonded_atom.positions
            dihedral_points = dihedral_points[::-1]

        start_angle = get_dihedral(dihedral_points)
        theta = start_angle - dihedral
        v = dihedral_points[2] - dihedral_points[1]
        v /= np.linalg.norm(v)
        mx = get_dihedral_rotation_matrix(theta, v)
        cap_struct.positions = (cap_struct.positions - dihedral_points[2]) @ mx.T + dihedral_points[2]
    elif term=='C' and 'OXT' in neighbor.names:
        C_pos = neighbor.select_atoms('name C').positions
        vfrom =  cap_struct[0].position - C_pos
        vfrom /= np.linalg.norm(vfrom)
        vto = neighbor.select_atoms(f'name OXT').positions - C_pos
        vto /= np.linalg.norm(vto)

        R, _ = Rotation.align_vectors(vfrom, vto)
        R = R.as_matrix()
        cap_struct.positions = (cap_struct.positions - C_pos) @ R + C_pos
        mol = mol.select_atoms(f'not (resid {resnum} and name OXT)')

    systems = [cap_struct, mol] if term == "N" else [mol, cap_struct]

    mol = concat_molsys(systems)

    return mol


def store_cap(name, mol, term):
    """
    Save the cap attached to a molecule. A pdb file will be saved in the chilife/data/rotamer_libraries/cap_pdbs folder
    with the provided name and the cap will be registered in either the ncaps.txt or ccaps.txt file in the
    chilife/data/rotamer_libraries directory.

    Parameters
    ----------
    name : str
        Name (3-letter identifier) of the cap to be attached.
    mol : str, Path, Universe, AtomGroup, MolSys
        Path, MDA.AtomGroup or MolSys object containing the cap. The cap must be its own residue number, i.e. not merged
        with the reside before or after it.
    term: str
        The terminus the cap is attached to. Only two values are accepted ``N`` and ``C``
    """

    name = name.upper()
    term = term.upper()
    if isinstance(mol, (str, Path)):
        mol = xl.MolSys.from_pdb(mol)
    elif isinstance(mol, (mda.Universe, mda.AtomGroup)):
        mol = xl.MolSys.from_atomsel(mol)

    bonds = chilife.guess_bonds(mol.positions, mol.types)
    top = chilife.Topology(mol, bonds)
    mol.topology = top

    if term == "N":
        cap = mol.residues[0]
        neighbor = mol.residues[1]
        txt_file = xl.RL_DIR / f'ncaps.txt'
        resnum=0

    elif term == "C":
        cap = mol.residues[-1]
        neighbor = mol.residues[-2]
        txt_file = xl.RL_DIR / f'ccaps.txt'
        resnum = 1
    else:
        raise RuntimeError('`term` not found in neighboring residue. Note that `term` must be an `N` or a `C`.')

    ori, mx = xl.local_mx(*neighbor.select_atoms("name N CA C").positions)
    cap.positions = (cap.positions - ori) @ mx
    with open(txt_file, 'r+') as f:
        line = f.readline()
        keys = line.split()
        if name not in keys:
            f.write(f" {name}")
    cap.resnum = resnum
    xl.save(xl.RL_DIR / f'cap_pdbs/{name}.pdb', cap)


def get_grid_mx(N, CA, C):
    """
    Get the rotation matrix and translation vector to orient a grid of lennard-jones spheres into a new coordinate
    frame. It is expected that the grid is defined as it is in :func:`screen_site_volume`, i.e. the xy plane of the grid
    is centered at the origin and the z-axis is positive.

    Parameters
    ----------
    N : np.ndarray
        First coordinate of the site.
    CA : np.ndarray
        Second coordinate of the site (The origin).
    C : np.ndarray
        Third coordinate of the site.

    Returns
    -------
    rotation matrix: np.ndarray
        Rotation matrix for the grid. Should be applied before translation.
    origin : np.ndarray
        Translation vector for the grid. Should be applied after rotation.
    """

    CA_N = N - CA
    CA_N /= np.linalg.norm(CA_N)

    CA_C = C - CA
    CA_C /= np.linalg.norm(CA_C)

    dummy_axis = CA_N + CA_C
    dummy_axis = dummy_axis / np.linalg.norm(dummy_axis)

    # Define new y-axis
    xaxis_plane = N - C
    dummy_comp = xaxis_plane.dot(dummy_axis)
    xaxis = xaxis_plane - dummy_comp * dummy_axis
    xaxis = xaxis / np.linalg.norm(xaxis)

    theta = np.deg2rad(35.25)
    mx = get_dihedral_rotation_matrix(theta, xaxis)
    yaxis = mx @ dummy_axis

    zaxis = np.cross(xaxis, yaxis)
    rotation_matrix = np.array([xaxis, yaxis, zaxis])
    origin = CA
    return rotation_matrix, origin


def write_grid(grid, name='grid.pdb', atype='Se'):
    """
    Given an array of 3-dimensional points (a grid usually), write a PDB so the grid can be visualized.

    Parameters
    ----------
    grid : ArrayLike
        The 3D points to write
    name: str
        The name of the file to write the grid to.
    atype : str
        Atom type to use for the grid.
    """

    with open(name, 'w') as f:
        for i, p in enumerate(grid):
            f.write(io.fmt_str.format(i+1, 'SEN', 'GRD', 'A', 1, *p, 1.0, 1.0, atype))


def get_site_volume(site: int,
                    mol: Union[MDAnalysis.Universe, MDAnalysis.AtomGroup, MolecularSystemBase],
                    grid_size: Union[float, ArrayLike] = 10,
                    offset: Union[float, ArrayLike] = -0.5,
                    spacing: float = 1.0,
                    vdw_r = 2.8,
                    write: str = False,
                    return_grid: bool = False
                    ) -> Union[float, ArrayLike]:
    """
    Calculate the (approximate) accessible volume of a single site. The volume is calculated by superimposing
    a grid of lennard-jones spheres at the site, eliminating those with clashes and calculating the volume from the
    remaining spheres. Note that this is a different "accessible volume" than the "accessible volume" method made
    popular by MtsslWizard.

    Parameters
    ----------
    site : int
        Site of the `mol` to calculate the accessible volume of.
    mol : MDAnalysis.Universe, MDAnalysis.AtomGroup, chilife.MolecularSystemBase
       The molecule/set of atoms containing the site to calculate the accessible volume of.
    grid_size: float, ArrayLike
        Dimensions of grid to use to calculate the accessible volume. Can be a float, specifying a cube with equal
        dimensions, or an array specifying a unique value for each dimension. Uses units of Angstroms.
    offset: float, ArrayLike
        The offset of the initial grid with respect to the superimposition site. If ``offset`` is a float, the grid will
        be offset in the z-dimension (along the CA-CB bond). If ``offset`` is an array, the grid will be offset in
        each x, y, z dimension by the corresponding value of the array. Uses units of Angstroms
    spacing : float
        Spacing between grid points. Defaults to 1.0 Angstrom
    vdw_r : float
        van der Waals radius for clash detection. Defaults to 2.8 Angstroms.
    write : bool, str, default False
        Switch to make chiLife write the grid to a PDB file. If ``True`` the grid will be written to grid.pdb.
        If given a string chiLife will use the string as the file name.
    return_grid: bool, default False
        If ``return_grid=True``, the volume will be returned as a numpy array containing the grid points that do not
        clash with the neighboring atoms.

    Returns
    -------
    volume : float, np.ndarray
        The accessible volume at the given site. If ``return_grid=True`` the volume will be returned as the set of
        grid points that do not clash with neighboring atoms, otherwise volume will be the estimated accessible volume
        of the site in cubic angstroms (Å :sup:`3`).
    """

    if isinstance(grid_size, (int, float, complex)):
        x = y = z = grid_size
    elif hasattr(grid_size, '__len__'):
        x, y, z = grid_size
    else:
        raise RuntimeError("grid_size must be a float or an array like object with 3 elements.")

    half_x = x / 2
    half_y = y / 2

    if isinstance(offset, (int, float, complex)):
        xo, yo = 0, 0
        zo = offset
    elif hasattr(offset, '__len__'):
        xo, yo, zo = offset
    else:
        raise RuntimeError("offset must be a float or an array like object with 3 elements.")

    xp = x / spacing
    yp = y / spacing
    zp = z / spacing

    # Create grid
    grid = np.mgrid[-half_x + xo:half_x + xo:xp * 1j, -half_y + yo:half_y + yo:yp * 1j, zo :z+zo:zp * 1j].swapaxes(0, -1)
    xl, yl, zl, _ = grid.shape
    grid = grid.reshape(-1, 3)

    # Superimpose grid on site
    bb = mol.select_atoms(f"resid {site} and name N CA C")
    if len(bb.atoms) != 3:
        raise RuntimeError(f'The provided protein does not have a canonical backbone at residue {site}.')

    N, CA, C = bb.positions

    mx, ori = get_grid_mx(N, CA, C)
    grid_tsf = grid @ mx + ori

    # Perform clash evaluation
    sel = mol.select_atoms(f'protein and not resid {site}')
    d = cdist(grid_tsf, sel.positions)
    mask = d < vdw_r
    mask2 = ~np.all(d > 5, axis=-1)

    # Remove clashing grid points
    mask = (~mask).prod(axis=-1) * mask2
    idxs = np.argwhere(mask).flatten()
    if len(idxs) == 0:
        return 0

    grid_tsf = grid_tsf[idxs]

    # check for discontinuities
    tree = cKDTree(grid_tsf)
    neighbors = tree.query_pairs(np.sqrt(2 * spacing) * 1.2)
    root_idx = np.argmin(np.linalg.norm(grid_tsf - CA, axis=-1))
    graph = ig.Graph(neighbors)

    # Use only points continuous with root_idx
    for g in graph.connected_components():
        if root_idx in g:
            grid_tsf = grid_tsf[g]
            break

    # process output
    if write:
        filename = write if isinstance(write, str) else 'grid.pdb'
        write_grid(grid_tsf, name=filename)

    if return_grid:
        return grid_tsf
    else:
        return len(grid_tsf) * spacing**3
