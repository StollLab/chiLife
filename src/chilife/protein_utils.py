import pickle, math, rtoml

from pathlib import Path
from typing import Set, List, Union, Tuple
from numpy.typing import ArrayLike
from scipy.spatial.distance import cdist
from scipy.spatial import cKDTree
from dataclasses import dataclass

import MDAnalysis
import numpy as np

from MDAnalysis.core.topologyattrs import Atomindices, Resindices, Segindices, Segids
import MDAnalysis as mda

from .globals import SUPPORTED_RESIDUES
from .MolSys import MolecularSystemBase, MolSys
from .numba_utils import get_sasa
from .pdb_utils import get_backbone_atoms, get_bb_candidates
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
                )
            )

    return missing_residues


def mutate(
        protein: MDAnalysis.Universe,
        *ensembles: 'RotamerEnsemble',
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
        if isinstance(lib, (re.RotamerEnsemble, dre.dRotamerEnsemble)):
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
        if isinstance(spin_label, dre.dRotamerEnsemble):
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
        U = make_mda_uni(atom_names, atom_types, res_names, residx, resids, segidx, segids)
    elif isinstance(protein, MolecularSystemBase):
        U = MolSys.from_arrays(atom_names, atom_types, res_names, residx, resids, segidx, segids)

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


def get_sas_res(
        protein: Union[mda.Universe, mda.AtomGroup], cutoff: float = 30,
        forcefield = 'charmm',
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
    if isinstance(forcefield, str):
        forcefield = scoring.ForceField(forcefield)

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
    """

    with open(name, 'w') as f:
        for i, p in enumerate(grid):
            f.write(io.fmt_str.format(i+1, 'SEN', 'GRD', 'A', 1, *p, 1.0, 1.0, atype))


def get_site_volume(site: int,
                    mol: Union[MDAnalysis.Universe, MDAnalysis.AtomGroup, MolecularSystemBase],
                    grid_size: Union[float, ArrayLike] = 10.0,
                    offset: Union[float, ArrayLike] = -2,
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

    vdw_r = 2.5

    if isinstance(grid_size, (int, float, complex)):
        x = y = z = grid_size
    elif isinstance(grid_size, ArrayLike):
        x, y, z = grid_size
    else:
        raise RuntimeError("grid_size must be a float or an array like object with 3 elements.")

    half_x = x / 2
    half_y = y / 2

    if isinstance(offset, (int, float, complex)):
        xo, yo = 0, 0
        zo = offset
    elif isinstance(offset, ArrayLike):
        xo, yo, zo = offset
    else:
        raise RuntimeError("offset must be a float or an array like object with 3 elements.")

    # Create grid
    grid = np.mgrid[-half_x + xo:half_x + xo:x * 1j, -half_y + yo:half_y + yo:y * 1j, zo :z+zo:z * 1j].swapaxes(0, -1)
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

    # Remove clashing grid points
    mask = (~mask).prod(axis=-1)
    idxs = np.argwhere(mask).flatten()
    grid_tsf = grid_tsf[idxs]

    # check for discontinuities
    tree = cKDTree(grid_tsf)
    neighbors = tree.query_pairs(1.5)
    root_idx = np.argmin(np.linalg.norm(grid_tsf - CA, axis=-1))
    graph = ig.Graph(neighbors)
    for g in graph.connected_components():
        if root_idx in g:
            break

    # Use only points continuous with root_idx
    grid_tsf = grid_tsf[g]

    # process output
    if write:
        filename = write if isinstance(write, str) else 'grid.pdb'
        write_grid(grid_tsf, name=filename)

    if return_grid:
        return grid_tsf
    else:
        return len(grid_tsf)
