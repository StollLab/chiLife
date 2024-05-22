import pickle, math, rtoml

from pathlib import Path
from typing import Set, List, Union, Tuple
from numpy.typing import ArrayLike
from dataclasses import dataclass

import MDAnalysis
import numpy as np

from MDAnalysis.core.topologyattrs import Atomindices, Resindices, Segindices, Segids
import MDAnalysis as mda

from .globals import SUPPORTED_RESIDUES
from .MolSys import MolecularSystemBase, MolSys
from .numba_utils import get_sasa
import chilife.scoring as scoring
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

def guess_mobile_dihedrals(ICs):
    sc_mask = ~np.isin(ICs.atom_names, ['N', 'CA', 'C', 'O', 'CB'])
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


