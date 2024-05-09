import pickle, math, rtoml
from operator import itemgetter
from pathlib import Path
from typing import Set, List, Union, Tuple
from numpy.typing import ArrayLike
from dataclasses import dataclass
from collections import defaultdict
import MDAnalysis
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist

from MDAnalysis.core.topologyattrs import Atomindices, Resindices, Segindices, Segids
import MDAnalysis as mda

import chilife
from .numba_utils import get_sasa
from .RotamerEnsemble import RotamerEnsemble
from .dRotamerEnsemble import dRotamerEnsemble

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


def local_mx(*p, method: Union[str, callable] = "bisect") -> Tuple[ArrayLike, ArrayLike]:
    """Calculates a translation vector and rotation matrix to transform a set of coordinates from the global
    coordinate frame to a local coordinate frame defined by ``p`` , using the specified method.

    Parameters
    ----------
    p : ArrayLike
        3D coordinates of the three points defining the coordinate system (Usually N, CA, C).
    method : str, callable
        Method to use for generation of rotation matrix

    Returns
    -------
    origin : np.ndarray
        Cartesian coordinate of the origin to be subtracted from the coordinates before applying the rotation matrix.
    rotation_matrix : np.ndarray
        Rotation matrix to transform a set of coordinates to the local frame defined by p and the selected method.
    """

    if isinstance(method, str):
        method = chilife.alignment_methods[method]

    p1, p2, p3 = p

    if method.__name__ == 'fit_alignment':
        rotation_matrix, _ = method(p1, p2, p3)
        origin = np.mean([p1[0], p2[0], p3[0]], axis=0)
    else:
        # Transform coordinates such that the CA atom is at the origin
        p1n = p1 - p2
        p3n = p3 - p2
        p2n = p2 - p2

        origin = p2

        # Local Rotation matrix is the inverse of the global rotation matrix
        rotation_matrix, _ = method(p1n, p2n, p3n)

    rotation_matrix = rotation_matrix.T

    return origin, rotation_matrix


def global_mx(*p: ArrayLike, method: Union[str, callable] = "bisect") -> Tuple[ArrayLike, ArrayLike]:
    """Calculates a translation vector and rotation matrix to transform the a set of coordinates from the local
    coordinate frame to the global coordinate frame using the specified method.

    Parameters
    ----------
    p : ArrayLike
        3D coordinates of the three points used to define the new coordinate system (Usually N, CA, C)
    method : str
        Method to use for generation of rotation matrix

    Returns
    -------
    rotation_matrix : np.ndarray
        Rotation matrix to be applied to the set of coordinates before translating
    origin : np.ndarray
        Vector to be added to the coordinates after rotation to translate the coordinates to the global frame.
    """

    if isinstance(method, str):
        method = chilife.alignment_methods[method]

    if method.__name__ == 'fit_alignment':
        p = [pi[::-1] for pi in p]

    rotation_matrix, origin = method(*p)
    return rotation_matrix, origin


@dataclass
class FreeAtom:
    """Atom class for atoms in cartesian space.

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
                res.resname not in chilife.SUPPORTED_RESIDUES
                or res.resnum in ignore
                or res.resname in ["ALA", "GLY"]
        ):
            continue

        # Check if there are any missing heavy atoms
        heavy_atoms = res.atoms.types[res.atoms.types != "H"]
        resn = res.resname

        a = cache.setdefault(resn, len(RotamerEnsemble(resn).atom_names))
        if len(heavy_atoms) != a:
            missing_residues.append(
                RotamerEnsemble(
                    res.resname,
                    res.resnum,
                    protein=protein,
                    chain=res.segid,
                    use_H=use_H,
                )
            )

    return missing_residues


def mutate(
        protein: MDAnalysis.Universe,
        *ensembles: RotamerEnsemble,
        add_missing_atoms: bool = True,
        random_rotamers: bool = False,
) -> MDAnalysis.Universe:
    """Create a new Universe where the native residue is replaced with the highest probability rotamer from a
    RotamerEnsemble or SpinLabel object.

    Parameters
    ----------
    protein : MDAnalysis.Universe
        An MDA Universe object containing protein to be spin labeled
    ensembles : RotamerEnsemble, SpinLabel
        Precomputed RotamerEnsemble or SpinLabel object to use for selecting and replacing the spin native amino acid
    random_rotamers :bool
        Randomize rotamer conformations
    add_missing_atoms : bool
        Model side chains missing atoms if they are not present in the provided structure.

    Returns
    -------
    U : MDAnalysis.Universe
        New Universe with a copy of the spin labeled protein with the highest probability rotamer
    """

    # Check for dRotamerEnsembles in ensembles
    temp_ensemble = []
    for lib in ensembles:
        if isinstance(lib, (RotamerEnsemble, dRotamerEnsemble)):
            temp_ensemble.append(lib)
        else:
            raise TypeError(
                f"mutate only accepts RotamerEnsemble, SpinLabel and dSpinLabel objects, not {lib}."
            )

    ensembles = temp_ensemble

    if add_missing_atoms:
        if len(ensembles) > 0 and all(not hasattr(lib, "H_mask") for lib in ensembles):
            use_H = True
        elif any(not hasattr(lib, "H_mask") for lib in ensembles):
            raise AttributeError(
                "User provided some ensembles with hydrogen atoms and some without. Make sure all "
                "ensembles either do or do not use hydrogen"
            )
        else:
            use_H = False

        missing_residues = get_missing_residues(
            protein, ignore={res.site for res in ensembles}, use_H=use_H
        )
        ensembles = list(ensembles) + missing_residues

    label_sites = {}
    for spin_label in ensembles:
        if isinstance(spin_label, dRotamerEnsemble):
            label_sites[spin_label.site1, spin_label.chain] = spin_label
            label_sites[spin_label.site2, spin_label.chain] = spin_label
        else:
            label_sites[spin_label.site, spin_label.chain] = spin_label

    protein = protein.select_atoms(
        f'(not altloc B) and (not (byres name OH2 or resname HOH))'
    )
    label_selstr = " or ".join([f"({label.selstr})" for label in ensembles])
    other_atoms = protein.select_atoms(f"not ({label_selstr})")

    resids = [res.resid for res in protein.residues]

    # Allocate lists for universe information
    atom_info = []
    res_names = []
    segidx = []

    # Loop over residues in old universe
    for i, res in enumerate(protein.residues):
        resloc = (res.resnum, res.segid)

        # If the residue is the spin labeled residue replace it with the highest probability spin label
        if resloc in label_sites:
            rot_ens = label_sites[resloc]
            if isinstance(rot_ens, dRotamerEnsemble):
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
        U = make_mda_uni(atom_names, atom_types, res_names, residx, resids, segidx, segids)
    elif isinstance(protein, chilife.MolecularSystemBase):
        U = chilife.MolSys.from_arrays(atom_names, atom_types, res_names, residx, resids, segidx, segids)

    # Apply old coordinates to non-spinlabel atoms
    new_other_atoms = U.select_atoms(f"not ({label_selstr})")
    new_other_atoms.atoms.positions = other_atoms.atoms.positions

    # Apply most probable spin label coordinates to spin label atoms
    for spin_label in label_sites.values():
        sl_atoms = U.select_atoms(spin_label.selstr)
        if random_rotamers:
            sl_atoms.atoms.positions = spin_label.coords[
                np.random.choice(len(spin_label.coords), p=spin_label.weights)
            ]
        else:
            sl_atoms.atoms.positions = spin_label.coords[np.argmax(spin_label.weights)]

    return U


def randomize_rotamers(
        protein: Union[mda.Universe, mda.AtomGroup],
        rotamer_libraries: List[RotamerEnsemble],
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


def get_sas_res(
        protein: Union[mda.Universe, mda.AtomGroup], cutoff: float = 30
) -> Set[Tuple[int, str]]:
    """Run FreeSASA to get solvent accessible surface residues in the provided protein

    Parameters
    ----------
    protein : MDAnalysis.Universe, MDAnalysis.AtomGroup
        MolSys object to measure Solvent Accessible Surfaces (SAS) area of and report the SAS residues.
    cutoff : float
        Exclude residues from list with SASA below cutoff in angstroms squared.

    Returns
    -------
    SAResi : set
        A set of solvent accessible surface residues.

    """
    environment_coords = protein.atoms.positions
    environment_radii = chilife.get_lj_rmin(protein.atoms.types)
    atom_sasa = get_sasa(environment_coords, environment_radii, by_atom=True)

    SASAs = {(residue.resnum, residue.segid) for residue in protein.residues if
             atom_sasa[0, residue.atoms.ix].sum() >= cutoff}

    return SASAs


def atom_sort_key(pdb_line: str) -> Tuple[str, int, int]:
    """Assign a base rank to sort atoms of a pdb.

    Parameters
    ----------
    pdb_line : str
        ATOM line from a pdb file as a string.

    Returns
    -------
    tuple :
        chain_id, resid, name_order.
        ordered ranking of atom for sorting the pdb.
    """
    chain_id = pdb_line[21]
    res_name = pdb_line[17:20].strip()
    resid = int(pdb_line[22:26].strip())
    atom_name = pdb_line[12:17].strip()
    atom_type = pdb_line[76:79].strip()
    if res_name == "ACE":
        if atom_type != 'H' and atom_name not in ('CH3', 'C', 'O'):
            raise ValueError(f'"{atom_name}" is not canonical name of an ACE residue atom. \n'
                             f'Please rename to "CH3", "C", or "O"')
        name_order = (
            {"CH3": 0, "C": 1, "O": 2}.get(atom_name, 4) if atom_type != "H" else 5
        )

    else:
        name_order = atom_order.get(atom_name, 4) if atom_type != "H" else atom_order.get(atom_name, 7)

    return chain_id, resid, name_order


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


def guess_bonds(coords: ArrayLike, atom_types: ArrayLike) -> np.ndarray:
    """ Given a set of coordinates and their atom types (elements) guess the bonds based off an empirical metric.

    Parameters
    ----------
    coords : ArrayLike
        Array of three-dimensional coordinates of the atoms of a molecule or set of molecules for which you would like
        to guess the bonds of.
    atom_types : ArrayLike
        Array of the element symbols corresponding to the atoms of ``coords``

    Returns
    -------
    bonds : np.ndarray
        An array of the atom index pairs corresponding to the atom pairs that are thought ot form bonds.
    """
    atom_types = np.array([a.title() for a in atom_types])
    kdtree = cKDTree(coords)
    pairs = kdtree.query_pairs(4., output_type='ndarray')
    pair_names = [tuple(x) for x in atom_types[pairs].tolist()]
    bond_lengths = itemgetter(*pair_names)(chilife.bond_hmax_dict)
    a_atoms = pairs[:, 0]
    b_atoms = pairs[:, 1]

    dist = np.linalg.norm(coords[a_atoms] - coords[b_atoms], axis=1)
    bonds = pairs[dist < bond_lengths]
    sorted_args = np.lexsort((bonds[:, 0], bonds[:,1]))
    return bonds[sorted_args]


def get_min_topol(lines: List[List[str]],
                  forced_bonds: set = None) -> Set[Tuple[int, int]]:
    """ Git the minimum topology shared by all the states/models a PDB ensemble. This is to ensure a consistent
    internal coordinate system between all conformers of an ensemble even when there are minor differences in topology.
    e.g. when dHis-Cu-NTA has the capping ligand in different bond orientations.

    Parameters
    ----------
    lines : List[List[str]]
        List of lists corresponding to individual states/models of a pdb file. All models must have the same stoma in
        the same order and only the coordinates should differ.
    forced_bonds : set
        A set of bonds to that must be used regardless even if the bond lengths are not physically reasonable.
    Returns
    -------
    minimal_bond_set : Set[Tuple[int, int]]
        A set of tuples holding the indices of atom pairs which are thought to be bonded in all states/models.
    """
    bonds_list = []
    if isinstance(lines[0], str):
        lines = [lines]

    # Get bonds for all structures
    for struct in lines:
        coords = np.array([(line[30:38], line[38:46], line[46:54]) for line in struct], dtype=float)
        atypes = np.array([line[76:78].strip() for line in struct])
        pairs = guess_bonds(coords, atypes)
        bonds = set(tuple(pair) for pair in pairs)
        bonds_list.append(bonds)

    # Get the shared bonds between all structures.
    minimal_bond_set = set.intersection(*bonds_list)
    # Include any forced bonds
    if forced_bonds is not None:
        minimal_bond_set |= forced_bonds

    return minimal_bond_set


def parse_connect(connect: List[str]) -> Tuple[Set[Tuple[int]]]:
    """
    Parse PDB CONECT information to get a list covalent bonds, hydrogen bonds and ionic bonds.

    Parameters
    ----------
    connect : List[str]
        A list of strings that are the CONECT lines from a PDB file.

    Returns
    -------
    c_bonds : Set[Tuple[int]]
        Set of atom index pairs corresponding to atoms that are bound covalently.
    h_bonds : Set[Tuple[int]]
        Set of atom index pairs corresponding to atoms that are hydrogen bound.
    ci_bonds : Set[Tuple[int]]
        Set of atom index pairs corresponding to atoms that are bound ionically.
    """
    c_bonds, h_bonds, i_bonds = set(), set(), set()
    for line in connect:
        line = line.ljust(61)
        a0 = int(line[6:11])
        c_bonds |= {tuple(sorted((a0 - 1, int(b) - 1))) for b in line[11:31].split()}
        h_bonds |= {tuple(sorted((a0 - 1, int(b) - 1))) for b in (line[31:41].split() + line[46:56].split())}
        i_bonds |= {tuple(sorted((a0 - 1, int(b) - 1))) for b in (line[41:46], line[56:61]) if not b.isspace()}

    return c_bonds, h_bonds, i_bonds


def sort_pdb(pdbfile: Union[str, List[str], List[List[str]]],
             uniform_topology: bool = True,
             index: bool = False,
             bonds: Union[ArrayLike, Set] = set(),
             **kwargs) -> Union[List[str], List[List[str]], List[int]]:
    """Read ATOM lines of a pdb and sort the atoms according to chain, residue index, backbone atoms and side chain atoms.
    Side chain atoms are sorted by distance to each other/backbone atoms with atoms closest to the backbone coming
    first and atoms furthest from the backbone coming last. This sorting is essential to making internal-coordinates
    with consistent and preferred dihedral definitions.

    Parameters
    ----------
    pdbfile : str, List[str], List[List[str]]
        Name of the PDB file, a list of strings containing ATOM lines of a PDB file or a list of lists containing
        ATOM lines of a PDB file, where each sublist corresponds to a state/model of a multi-state pdb.
    uniform_topology: bool
        When given a multi-state pdb, assume that all states have the same topology (bonds) as the first state.
    index: bool :
         Return the sorted index rather than the sorted lines.
    bonds: ArrayLike :
         When sorting the PDB, use the provided bond list to as the topology rather than guessing the bonds.

    Returns
    -------
    lines : List[str], List[List[str]]
        Sorted list of strings corresponding to the ATOM entries of a PDB file.
    """
    if isinstance(pdbfile, (str, Path)):
        with open(pdbfile, "r") as f:
            lines = f.readlines()

        start_idxs = []
        end_idxs = []
        connect = []
        lines = [line for line in lines if line.startswith(('MODEL', 'ENDMDL', 'CONECT', 'ATOM', 'HETATM'))]
        for i, line in enumerate(lines):
            if line.startswith('MODEL'):
                start_idxs.append(i + 1)
            elif line.startswith("ENDMDL"):
                end_idxs.append(i)
            elif line.startswith('CONECT'):
                connect.append(line)

        # Use connect information for bonds if present
        if connect != [] and bonds == set():
            connect, _, _ = parse_connect(connect)
            kwargs['additional_bonds'] = kwargs.get('additional_bonds', set()) | connect

        # If it's a multi-state pdb...
        if start_idxs != []:

            if uniform_topology:
                # Assume that all states have the same topology as the first
                idxs = _sort_pdb_lines(lines[start_idxs[0]:end_idxs[0]], bonds=bonds, index=True, **kwargs)

            else:
                # Calculate the shared topology and force it
                atom_lines = [lines[s:e] for s, e in zip(start_idxs, end_idxs)]
                min_bonds_list = get_min_topol(atom_lines, forced_bonds=bonds)
                idxs = _sort_pdb_lines(lines[start_idxs[0]:end_idxs[0]], bonds=min_bonds_list, index=True, **kwargs)

            if isinstance(idxs, tuple):
                idxs, bonds = idxs

            lines[:] = [[lines[idx + start][:6] + f"{i + 1:5d}" + lines[idx + start][11:]
                         for i, idx in enumerate(idxs)]
                        for start in start_idxs]

            if kwargs.get('return_bonds', False):
                lines = lines, bonds
        else:
            lines = _sort_pdb_lines(lines, bonds=bonds, index=index, **kwargs)

    elif isinstance(pdbfile, list):
        lines = _sort_pdb_lines(pdbfile, bonds=bonds, index=index, **kwargs)

    return lines


def _sort_pdb_lines(lines, bonds=None, index=False, **kwargs) -> \
        Union[List[str], List[int], Tuple[list[str], List[Tuple[int]]]]:
    """
    Helper function to sort PDB ATOM and HETATM lines based off of the topology of the topology of the molecule.

    Parameters
    ----------
    lines : List[str]
        A list of the PDB ATOM and HETATM lines.
    bonds : Set[Tuple[int]]
        A Set of tuples of atom indices corresponding to atoms ( `lines` ) that are bound to each other.
    index : bool
        If True a list of atom indices will be returned
    **kwargs : dict
        Additional keyword arguments.
        return_bonds : bool
            Return bond indices as well, usually only used when letting the function guess the bonds.
        additional_bonds: set(tuple(int))

    Returns
    -------
    lines : List[str | int]
        The sorted lines or indices corresponding to the sorted lines.
    bonds: Set[Tuple[int]]
        A set of tuples containing pars of indices corresponding to the atoms bound to in lines.
    """

    waters = [line for line in lines if line[17:20] in ('SOL', 'HOH')]
    water_idx = [idx for idx, line in enumerate(lines) if line[17:20] in ('SOL', 'HOH')]
    lines = [line for line in lines if line.startswith(("ATOM", "HETATM")) and line[17:20] not in ('SOL', 'HOH')]
    n_atoms = len(lines)
    index_key = {line[6:11]: i for i, line in enumerate(lines)}

    # Presort
    lines.sort(key=atom_sort_key)
    presort_idx_key = {line[6:11]: i for i, line in enumerate(lines)}
    presort_bond_key = {index_key[line[6:11]]: i for i, line in enumerate(lines)}

    coords = np.array([[float(line[30:38]), float(line[38:46]), float(line[46:54])] for line in lines])
    atypes = np.array([line[76:78].strip() for line in lines])
    anames = np.array([line[12:17].strip() for line in lines])

    if bonds:
        input_bonds = {tuple(b) for b in bonds}
        presort_bonds = set(tuple(sorted((presort_bond_key[b1], presort_bond_key[b2]))) for b1, b2 in bonds)
    else:
        bonds = guess_bonds(coords, atypes)
        presort_bonds = set(tuple(sorted((b1, b2))) for b1, b2 in bonds)
        if kwargs.get('additional_bonds', set()) != set():
            presort_bonds.union(kwargs['additional_bonds'])
    # get residue groups
    chain, resi = lines[0][21], int(lines[0][22:26].strip())
    start = 0
    resdict = {}
    for curr, pdb_line in enumerate(lines):

        if chain != pdb_line[21] or resi != int(pdb_line[22:26].strip()):
            resdict[chain, resi] = start, curr
            start = curr
            chain, resi = pdb_line[21], int(pdb_line[22:26].strip())

    resdict[chain, resi] = start, curr + 1
    midsort_key = []
    for key in resdict:
        start, stop = resdict[key]
        n_heavy = np.sum(atypes[start:stop] != 'H')

        #  Force N, CA, C,
        if np.array_equal(anames[start: start + 4], ['N', 'CA', 'C', 'O']):
            sorted_args = [0, 1, 2, 3]
        # if not a canonical and the first amino acid use the first heavy atom
        elif start == 0:
            sorted_args = [0]
        else:
            # If not a connected via peptide backbone
            for a, b in presort_bonds:
                # Find the atom bonded to a previous residue
                if a < start and start <= b < stop and atypes[b] != 'H':
                    sorted_args = [b - start]
                    break
                # Otherwise get the closest to any previous atom
            else:
                dist_mat = cdist(coords[:start], coords[start:stop])
                sorted_args = [np.squeeze(np.argwhere(dist_mat == dist_mat.min()))[1]]

        if len(sorted_args) != n_heavy:

            root_idx = 1 if len(sorted_args) == 4 else sorted_args[0]
            bonds = np.array([bond for bond in presort_bonds
                              if (start <= bond[0] < stop) and (start <= bond[1] < stop)])
            bonds -= start
            bonds = np.asarray(bonds)

            # Get all nearest neighbors and sort by distance
            distances = np.linalg.norm(coords[start:stop][bonds[:, 0]] - coords[start:stop][bonds[:, 1]], axis=1)
            distances = np.around(distances, decimals=3)

            idx_sort = np.lexsort((bonds[:, 0], bonds[:, 1], distances))
            pairs = bonds[idx_sort]
            pairs = [pair for pair in pairs if np.any(~np.isin(pair, sorted_args))]

            graph = ig.Graph(edges=pairs)

            if root_idx not in graph.vs.indices:
                root_idx = min(graph.vs.indices)

            # Start stemming from CA atom
            CA_edges = [edge[1] for edge in bfs_edges(pairs, root_idx) if edge[1] not in sorted_args]

            # Check for disconnected parts of residue
            if not graph.is_connected():
                for g in graph.connected_components():
                    if np.any([arg in g for arg in sorted_args]):
                        continue
                    CA_nodes = [idx for idx in CA_edges if atypes[start + idx] != 'H']
                    g_nodes = [idx for idx in g if atypes[start + idx] != 'H']
                    near_root = cdist(coords[start:stop][CA_nodes], coords[start:stop][g_nodes]).argmin()

                    yidx = near_root % len(g_nodes)
                    subnodes, _, _ = graph.bfs(g_nodes[yidx])
                    CA_edges += list(subnodes)

        elif stop - start > n_heavy:
            # Assumes  non-heavy atoms come after the heavy atoms, which should be true because of the pre-sort
            CA_edges = list(range(n_heavy, n_heavy + (stop - start - len(sorted_args))))
        else:
            CA_edges = []

        sorted_args = sorted_args + CA_edges

        # get any leftover atoms (eg HN)
        if len(sorted_args) != stop - start:
            for idx in range(stop - start):
                if idx not in sorted_args:
                    sorted_args.append(idx)

        midsort_key += [x + start for x in sorted_args]

    lines[:] = [lines[i] for i in midsort_key]
    lines.sort(key=atom_sort_key)

    if 'input_bonds' not in locals():
        input_bonds = presort_bonds
        idxmap = {presort_idx_key[line[6:11]]: i for i, line in enumerate(lines)}
    else:
        idxmap = {index_key[line[6:11]]: i for i, line in enumerate(lines)}

    # Return line indices if requested
    if index:
        str_lines = lines
        lines = [index_key[line[6:11]] for line in lines] + water_idx

    # Otherwise make new indices
    else:
        lines = [line[:6] + f"{i + 1:5d}" + line[11:] for i, line in enumerate(lines)] + waters

    if kwargs.get('return_bonds', False):
        bonds = {tuple(sorted((idxmap[a], idxmap[b]))) for a, b in input_bonds}
        return lines, bonds

    return lines


def make_mda_uni(anames: ArrayLike,
                 atypes: ArrayLike,
                 resnames: ArrayLike,
                 resindices: ArrayLike,
                 resnums: ArrayLike,
                 segindices: ArrayLike,
                 segids: ArrayLike = None,
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


def neighbors(edges, node):
    """
    Given a graph defined by edges and a node, find all neighbors of that node.

    Parameters
    ----------
    edges : ArrayLike
        Array of tuples defining all edges between nodes
    node : int
        The node of the graph for which to find neighbors.

    Returns
    -------
    nbs : ArrayLike
        Neighbor nodes.
    """
    nbs = []
    for edge in edges:
        if node not in edge:
            continue
        elif node == edge[0]:
            nbs.append(edge[1])
        elif node == edge[1]:
            nbs.append(edge[0])
    return nbs


def bfs_edges(edges, root):
    """
    Breadth first search of nodes given a set of edges
    Parameters
    ----------
    edges : ArrayLike
        Array of tuples defining edges between nodes.
    root : int
        Starting (root) node to begin the breadth first search at.

    Yields
    ------
    parent : int
        The node from which the children node stem
    child: List[int]
        All children node of parent.
    """
    nodes = np.unique(edges)

    depth_limit = len(nodes)
    seen = {root}

    n = len(nodes)
    depth = 0
    next_parents_children = [(root, neighbors(edges, root))]

    while next_parents_children and depth < depth_limit:
        this_parents_children = next_parents_children
        next_parents_children = []
        for parent, children in this_parents_children:
            for child in children:
                if child not in seen:
                    seen.add(child)
                    next_parents_children.append((child, neighbors(edges, child)))
                    yield parent, child
            if len(seen) == n:
                return
        depth += 1


DATA_DIR = Path(__file__).parent.absolute() / "data/"
RL_DIR = Path(__file__).parent.absolute() / "data/rotamer_libraries/"

# Define rotamer dihedral angle atoms
with open(DATA_DIR / "dihedral_defs.toml", "r") as f:
    dihedral_defs = rtoml.load(f)

with open(RL_DIR / "RotlibIndexes.pkl", "rb") as f:
    rotlib_indexes = pickle.load(f)

with open(DATA_DIR / 'BondDefs.pkl', 'rb') as f:
    bond_hmax_dict = {key: (val + 0.4 if 'H' in key else val + 0.35) for key, val in pickle.load(f).items()}
    bond_hmax_dict = defaultdict(lambda: 0, bond_hmax_dict)


    def bond_hmax(a): return bond_hmax_dict.get(tuple(i for i in a), 0)


    bond_hmax = np.vectorize(bond_hmax, signature="(n)->()")

atom_order = {"N": 0, "CA": 1, "C": 2, "O": 3, 'H': 5}

nataa_codes = {'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
               'LYS': 'K', 'LEU': 'L', 'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 'ARG': 'R', 'SER': 'S',
               'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'}

inataa = {val: key for key, val in nataa_codes.items()}

nataa_codes.update(inataa)
del inataa
