import logging, os, urllib, pickle, itertools, math
from pathlib import Path
from typing import Set, List, Union, Tuple
from numpy.typing import ArrayLike
from dataclasses import dataclass, replace
from collections import Counter
import MDAnalysis
import numpy as np
from scipy.spatial import cKDTree

from MDAnalysis.core.topologyattrs import Atomindices, Resindices, Segindices, Segids
import MDAnalysis as mda
import freesasa

import chiLife
from .numba_utils import _ic_to_cart
from .superimpositions import superimpositions
from .RotamerLibrary import RotamerLibrary
from .SpinLabel import SpinLabel, dSpinLabel


import networkx as nx


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
    rotation_matrix = (
        np.identity(3) * np.cos(theta)
        + np.sin(theta) * Vx
        + (1 - np.cos(theta)) * np.outer(v, v)
    )

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

    :return: float
        Dihedral angle in radians
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

    return math.atan2(y, x)


def get_angle(p: ArrayLike) -> float:
    p1, p2, p3 = p
    v1 = p1 - p2
    v2 = p3 - p2
    X = v1 @ v2
    Y = np.cross(v1, v2)
    Y = math.sqrt(Y @ Y)
    return math.atan2(Y, X)


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
    angle = np.deg2rad(angle) - current
    angle = angle

    ori = p[1]
    mobile -= ori
    v = p[2] - p[1]
    v /= np.linalg.norm(v)
    R = get_dihedral_rotation_matrix(angle, v)

    new_mobile = R.dot(mobile.T).T + ori

    return new_mobile


def local_mx(
    N: ArrayLike, CA: ArrayLike, C: ArrayLike, method: str = "bisect"
) -> Tuple[ArrayLike, ArrayLike]:
    """
    Calculates a translation vector and rotation matrix to transform the provided rotamer library from the global
    coordinate frame to the local coordinate frame using the specified method.

    :param N: ArrayLike
        3D coordinates of the amino Nitrogen of the amino acid backbone

    :parma CA: ArrayLike
        3D coordinates of the alpha carbon of the amino acid backbone

    :param C: ArrayLike
        3D coordinates of the carboxyl carbon of the amino acid backbone

    :param method: str
        Method to use for generation of rotation matrix

    :return origin, rotation_matrix: ndarray, ndarray
        origin and rotation matrix for rotamer library
    """

    if method in {"fit"}:
        rotation_matrix, _ = superimpositions[method](N, CA, C)
    else:
        # Transform coordinates such that the CA atom is at the origin
        Nn = N - CA
        Cn = C - CA
        CAn = CA - CA

        # Local Rotation matrix is the inverse of the global rotation matrix
        rotation_matrix, _ = superimpositions[method](Nn, CAn, Cn)

    rotation_matrix = rotation_matrix.T

    # Set origin at C-alpha
    origin = CA

    return origin, rotation_matrix


def global_mx(
    N: ArrayLike, CA: ArrayLike, C: ArrayLike, method: str = "bisect"
) -> Tuple[ArrayLike, ArrayLike]:
    """
        Calculates a translation vector and rotation matrix to transform the provided rotamer library from the local
    coordinate frame to the global coordinate frame using the specified method.

    :param N: ArrayLike
        3D coordinates of the amino Nitrogen of the amino acid backbone

    :parma CA: ArrayLike
        3D coordinates of the alpha carbon of the amino acid backbone

    :param C: ArrayLike
        3D coordinates of the carboxyl carbon of the amino acid backbone

    :param method: str
        Method to use for generation of rotation matrix

    :return rotation_matrix, origin: ndarray, ndarray
        rotation matrix and origin for rotamer library
    """
    rotation_matrix, origin = superimpositions[method](N, CA, C)
    return rotation_matrix, origin


def ic_mx(
    atom1: ArrayLike, atom2: ArrayLike, atom3: ArrayLike
) -> Tuple[ArrayLike, ArrayLike]:
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
        Backbone C-alpha carbon coordinates

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
    z_axis = np.cross(v12, v23)

    # Create rotation matrix
    rotation_matrix = np.array([v12, v23, z_axis])
    origin = p1

    return rotation_matrix, origin


# TODO: Align Atom and ICAtom names with MDA
# TODO: Implement Internal Coord Residue object


@dataclass
class ICAtom:
    """
    Internal coordinate atom class. Used for building the ProteinIC (internal coordinate proteine).

    :param name: str
        Atom name
    :param atype: str
        Atom type
    :param index: int
        Atom number
    :param resn: str
        Name of the residue that the atom belongs to
    :param resi: int
        The residue index/number that the atom belongs to
    :param atom_names: tuple
        The names of the atoms that define the Bond-Angle-Torsion coordinates of the atom
    :param bond_idx: int
        Index of the coordinate atom bonded to this atom
    :param bond: float
        Distance of the bond in angstroms
    :param angle_idx: int
        Index of the atom that creates an angle with this atom and the bonded atom.
    :param angle: float
        The value of the angle between this atom, the bonded atom and the angle atom in radians.
    :param dihedral_idx: int
        The index of the atom that defines the dihedral with this atom, the bonded atom and the angled atom.
    :param dihedral: float
        The value of the dihedral angle defined above in radians.
    :param dihederal_resi: int
        The residues index of the residue that the dihedral angle belongs to. Note that this is not necessarily the same
        residue as the atom, e.g. the location of the nitrogen atom of the i+1 residue often defines the phi dihedral
        angle of the ith residue.
    """

    name: str
    atype: str
    index: int
    resn: str
    resi: int
    atom_names: tuple

    bond_idx: int = np.nan
    bond: float = np.nan
    angle_idx: int = np.nan
    angle: float = np.nan
    dihedral_idx: int = np.nan
    dihedral: float = np.nan
    dihedral_resi: int = np.nan

    def __post_init__(self):
        """If no dihedral_resi is defined default to the atom residue"""
        if np.isnan(self.dihedral_resi):
            self.dihedral_resi: int = self.resi


@dataclass
class Atom:
    """Atom class for atoms in cartesian space."""

    name: str
    atype: str
    index: int
    resn: str
    resi: int
    coords: np.ndarray

    @property
    def x(self):
        return self.coords[0]

    @x.setter
    def x(self, x):
        self.coords[0] = x

    @property
    def y(self):
        return self.coords[1]

    @y.setter
    def y(self, y):
        self.coords[1] = y

    @property
    def z(self):
        return self.coords[1]

    @z.setter
    def z(self, z):
        self.coords[1] = z


class ProteinIC:

    def __init__(self, zmats, zmat_idxs, atom_dict, ICs, **kwargs):
        self.zmats = zmats
        self.zmat_idxs = zmat_idxs
        self.atom_dict = atom_dict
        self.ICs = ICs
        self.atoms = [
            ic
            for chain in self.ICs.values()
            for resi in chain.values()
            for ic in resi.values()
        ]

        self.atom_types = [atom.atype for atom in self.atoms]
        self.atom_names = [atom.name for atom in self.atoms]

        self.resis = [key for i in self.ICs for key in self.ICs[i]]
        self.resnames = {
            j: next(iter(self.ICs[i][j].values())).resn
            for i in self.ICs
            for j in self.ICs[i]
        }
        self.chains = [name for name in ICs]

        self.chain_operators = kwargs.get("chain_operators", None)
        self.bonded_pairs = np.array(kwargs.get("bonded_pairs", None))
        self.nonbonded_pairs = kwargs.get('nonbonded_pairs', None)

        if self.nonbonded_pairs is None and not (self.bonded_pairs is None or self.bonded_pairs.any() is None):
            bonded_pairs = {(a, b) for a, b in self.bonded_pairs}
            possible_bonds = itertools.combinations(range(len(self.atoms)), 2)
            self.nonbonded_pairs = np.fromiter(
                itertools.chain.from_iterable(
                    nb for nb in possible_bonds if nb not in bonded_pairs
                ),
                dtype=int,
            )
            self.nonbonded_pairs.shape = (-1, 2)

        self.perturbed = False
        self._coords = self.to_cartesian()
        self.dihedral_defs = self.collect_dih_list()

    @classmethod
    def from_ICatoms(cls, ICs, **kwargs):
        """
        Object collecting internal coords of atoms making up a protein.
        :param ICs: dict
            dictionary of dicts, of dicts containing ICAtom objects. Top dict specifies chain, middle dict specifies
            residue and bottom dict holds ICAtom objects.
        """

        dihedral_dict, angle_dict, bond_dict = {}, {}, {}
        for chain, chain_dict in ICs.items():
            dihedral_dict[chain], angle_dict[chain], bond_dict[chain] = {}, {}, {}
            for resi, resi_dict in chain_dict.items():
                for atom_def, atom in  resi_dict.items():
                    if len(atom_def) == 4:
                        stem = atom_def[1:]
                        if (resi, stem) in dihedral_dict[chain]:
                            dihedral_dict[chain][resi, stem][atom.name] = atom.index
                        else:
                            dihedral_dict[chain][resi, stem] = {atom.name: atom.index}

                    if len(atom_def) > 2:
                        angle_dict[chain][resi, atom_def[:3]] = atom.index
                    if len(atom_def) > 1:
                        bond_dict[chain][resi, atom_def[:2]] = atom.index

        atom_dict = {'bonds': bond_dict, 'angles': angle_dict, 'dihedrals': dihedral_dict}
        zmats, zmat_idxs = {}, {}
        for chain, chain_dict in ICs.items():
            zmat_concat = np.fromiter(itertools.chain.from_iterable(itertools.chain(
                                    (
                                        ic.bond_idx,
                                        ic.angle_idx,
                                        ic.dihedral_idx,
                                        ic.bond,
                                        ic.angle,
                                        ic.dihedral,
                                    )
                                )
                    for res in chain_dict.values()
                    for ic in res.values()
                                ),
                dtype=float,
            )

            zmat_concat.shape = (-1, 6)
            zmat_idxs[chain], zmats[chain] = zmat_concat[:, :3].astype(int), zmat_concat[:, 3:].copy()

        return cls(zmats, zmat_idxs, atom_dict, ICs, **kwargs)

    def copy(self):
        """Create a deep copy of an ProteinIC instance"""
        zmats = {key: value.copy() for key, value in self.zmats.items()}
        zmat_idxs = {key: value.copy() for key, value in self.zmat_idxs.items()}
        kwargs = {"chain_operators": self.chain_operators,
                  "bonded_pairs": self.bonded_pairs,
                  "nonbonded_pairs": self.nonbonded_pairs}

        return ProteinIC(zmats, zmat_idxs, self.atom_dict, self.ICs, **kwargs)

    @property
    def chain_operators(self):
        """Chain_operators are a set of coordinate transformations that can orient multiple chains that are not
        covalently linked. e.g. structures with missing residues or protein complexes"""
        return self._chain_operators

    @chain_operators.setter
    def chain_operators(self, op):
        """
        Assign or calculate operators for all chains.

        :param op: dict
            Dictionary containing an entry for each chain in the ProteinIC molecule. Each entry must contain a
            rotation matrix, 'mx' and translation vector 'ori'.
        """
        if op is None:
            logging.info(
                "No protein chain origins have been provided. All chains will start at [0, 0, 0]"
            )
            op = {
                chain: {"ori": np.array([0, 0, 0]), "mx": np.identity(3)}
                for chain in self.ICs
            }

        self._chain_operators = op
        self.perturbed = True

    @property
    def coords(self):
        if (self._coords is None) or self.perturbed:
            self._coords = self.to_cartesian()
            self.perturbed = False
        return self._coords

    @coords.setter
    def coords(self, coords):
        coords = np.asarray(coords)
        if coords.shape != self._coords.shape:
            raise ValueError('the coordinate array supplied does not match the ProteinIC object coords array')
        if len(self.zmats) > 1:
            raise NotImplementedError('ProteinIC does not currently support cartesian coordinate assignemnt for '
                                      'multichain structures.')
        chain = self.chains[0]
        for idx, coord in enumerate(coords):
            if np.all(np.isnan(coord)):
                continue

            bidx, aidx, tidx = self.zmat_idxs[chain][idx]
            bond, angle, dihedral = np.nan, np.nan, np.nan
            if idx == 0:
                pass
                # Need to calculate new chain operators
            if idx > 0:
                bond = np.linalg.norm(self.coords[bidx] - coord)
            if idx > 1:
                angle = get_angle([self.coords[aidx],
                                           self.coords[bidx],
                                           coord])
            if idx > 2:
                dihedral = get_dihedral([self.coords[tidx],
                                                 self.coords[aidx],
                                                 self.coords[bidx],
                                                 coord])




            self._coords[idx] = coord
            self.zmats[chain][idx] = bond, angle, dihedral

        # Update the location of any atoms that weren't included (e.g. hydrogens or backbones)
        self._coords = self.to_cartesian()

    def set_dihedral(self, dihedrals, resi, atom_list, chain=None):
        """
        Set one or more dihedral angles of a single residue in internal coordinates for the atoms defined in atom list.

        :param dihedrals: float, ndarray
            Angle or array of angles to set the dihedral(s) to

        :param resi: int
            Residue number of the site being altered

        :param atom_list: ndarray, list, tuple
            Names or array of names of atoms involved in the dihedral(s)

        :return coords: ndarray
            ProteinIC object with new dihedral angle(s)
        """
        self.perturbed = True

        if chain is None and len(self.chains) == 1:
            chain = list(self.ICs.keys())[0]
        elif chain is None and len(self.chains) > 1:
            raise ValueError("You must specify the protein chain")

        dihedrals = np.atleast_1d(dihedrals)
        atom_list = np.atleast_2d(atom_list)

        for i, (dihedral, atoms) in enumerate(zip(dihedrals, atom_list)):
            stem, stemr = tuple(atoms[2::-1]), tuple(atoms[1:])
            if (resi, stem) in self.atom_dict['dihedrals'][chain]:
                aidx = self.atom_dict['dihedrals'][chain][resi, stem][atoms[-1]]
                delta = self.zmats[chain][aidx, 2] - dihedral
                self.zmats[chain][aidx, 2] = dihedral
                idxs = [idx for idx in self.atom_dict['dihedrals'][chain][resi, stem].values() if idx != aidx]

            elif (resi, stemr) in self.atom_dict['dihedrals'][chain]:
                aidx = self.atom_dict['dihedrals'][chain][resi, stemr][atoms[0]]
                delta = self.zmats[chain][aidx, 2] - dihedral
                self.zmats[chain][aidx, 2] = dihedral
                idxs = [idx for idx in self.atom_dict['dihedrals'][chain][resi, stemr].values() if idx != aidx]

            else:
                raise ValueError(
                    f"Dihedral with atoms {atoms} not found in chain {chain} on resi {resi} internal coordinates:\n"
                    + "\n".join([ic for ic in self.ICs[chain][resi]])
                )

            if idxs:
                self.zmats[chain][idxs, 2] -= delta

        return self

    def get_dihedral(self, resi, atom_list, chain=None):
        """
        Get the dihedral angle(s) of one or more atom sets at the specified residue. Dihedral angles are returned in
        radians

        :param resi: int
            Residue number of the site being altered

        :param atom_list: ndarray, list, tuple
            Names or array of names of atoms involved in the dihedral(s)

        :return angles: ndarray
            array of dihedral angles corresponding to the atom sets in atom list
        """

        if chain is None and len(self.ICs) == 1:
            chain = list(self.ICs.keys())[0]
        elif chain is None and len(self.ICs) > 1:
            raise ValueError("You must specify the protein chain")

        atom_list = np.atleast_2d(atom_list)
        dihedrals = []
        for atoms in atom_list:
            stem, stemr = tuple(atoms[2::-1]), tuple(atoms[1:])
            if (resi, stem) in self.atom_dict['dihedrals'][chain]:
                aidx = self.atom_dict['dihedrals'][chain][resi, stem][atoms[-1]]
            elif (resi, stemr) in self.atom_dict['dihedrals'][chain]:
                aidx = self.atom_dict['dihedrals'][chain][resi, stemr][atoms[0]]
            else:
                raise ValueError(
                    f"Dihedral with atoms {atoms} not found in chain {chain} on resi {resi} internal coordinates:\n"
                    + "\n".join([ic for ic in self.ICs[chain][resi]])
                )
            dihedrals.append(self.zmats[chain][aidx, 2])

        return dihedrals[0] if len(dihedrals) == 1 else np.array(dihedrals)

    def to_cartesian(self):
        """
        Convert internal coordinates into cartesian coordinates.

        :return coord: ndarray
            Array of cartesian coordinates corresponding to ICAtom list atoms
        """
        coord_arrays = []
        for segid in self.chain_operators:
            # Prepare variables for numba compiled function

            IC_idx_array, ICArray = self.zmat_idxs[segid], self.zmats[segid]
            cart_coords = _ic_to_cart(IC_idx_array, ICArray)

            # Apply chain operations if any exist
            if ~np.allclose(
                self.chain_operators[segid]["mx"], np.identity(3)
            ) and ~np.allclose(self.chain_operators[segid]["ori"], np.array([0, 0, 0])):
                cart_coords = (
                    cart_coords @ self.chain_operators[segid]["mx"]
                    + self.chain_operators[segid]["ori"]
                )

            coord_arrays.append(cart_coords)

        return np.concatenate(coord_arrays)

    def to_site(self, N, CA, C, method="bisect"):
        new_co = {}
        gmx, gori = global_mx(N, CA, C)
        lori, lmx = local_mx(*np.squeeze(self.coords[:3]), method=method)
        m2m3 = lmx @ gmx
        for segid in self.chain_operators:

            new_mx = self.chain_operators[segid]["mx"] @ m2m3
            new_ori = (self.chain_operators[segid]["ori"] - lori) @ m2m3 + gori
            new_co[segid] = {"mx": new_mx, "ori": new_ori}

        self.chain_operators = new_co
        self.perturbed = True

    def save_pdb(self, filename: str, mode="w"):
        """
        Save a pdb structure file from a ProteinIC object

        :param filename: str
            Name of file to save

        :param mode: str
            file open mode
        """
        if "a" in mode:
            with open(str(filename), mode, newline="\n") as f:
                f.write("MODEL\n")
            save_pdb(filename, self.atoms, self.coords, mode=mode)
            with open(str(filename), mode, newline="\n") as f:
                f.write("ENDMDL\n")
        else:
           save_pdb(filename, self.atoms, self.to_cartesian(), mode=mode)

    def has_clashes(self, distance=1.5):
        """
        Checks for an internal clash between nonbonded atoms
        :param distance: float
            Minimum distance allowed between non-bonded atoms

        :return:
        """
        diff = (
            self.coords[self.nonbonded_pairs[:, 0]]
            - self.coords[self.nonbonded_pairs[:, 1]]
        )
        dist = np.linalg.norm(diff, axis=1)
        has_clashes = np.any(dist < distance)
        return has_clashes

    def get_resi_dihs(self, resi: int):
        """
        Gets the list of heavy atom dihedral definitions for the provided residue.

        :param resi: int
            Index of the residue whose dihedrals will be returned
        """

        if resi == 0 or resi == list(self.ICs[1].keys())[-1]:
            dihedral_defs = []
        elif resi == 1:
            dihedral_defs = [
                ["CH3", "C", "N", "CA"],
                ["C", "N", "CA", "C"],
                ["N", "CA", "C", "N"],
            ]
        else:
            dihedral_defs = [["C", "N", "CA", "C"], ["N", "CA", "C", "N"]]
        return dihedral_defs

    def collect_dih_list(self) -> list:
        """
        Returns a list of all heavy atom dihedrals

        :return dihs: list
            list of protein heavy atom dihedrals
        """
        dihs = []

        # Get backbone and sidechain dihedrals for the provided universe
        for resi in self.ICs[1]:
            resname = self.resnames[resi]
            res_dihs = self.get_resi_dihs(resi)
            res_dihs += dihedral_defs.get(resname, [])
            dihs += [(resi, d) for d in res_dihs]

        return dihs

    def shift_resnum(self, delta):
        for key in self.atom_dict:
            for chain in self.atom_dict[key]:

                self.atom_dict[key][chain] = {(res + delta, other): self.atom_dict[key][chain][res, other]
                                              for res, other in self.atom_dict[key][chain]}


        for segid in self.ICs:
            for resi in list(self.ICs[segid].keys()):
                self.ICs[segid][resi + delta] = self.ICs[segid][resi]
                del self.ICs[segid][resi]
                for atom in self.ICs[segid][resi + delta].values():
                    atom.resi += delta
                    atom.dihedral_resi += delta

        self.resis = [key for i in self.ICs for key in self.ICs[i]]


def get_ICAtom(
    mol: mda.core.groups.Atom, dihedral: List[int], offset: int = 0, preferred_dihedral: List = None
) -> ICAtom:
    """
    Construct internal coordinates for an atom given that atom is linked to an MDAnalysis Universe.

    :param atom: MDAnalysis.Atom
        Atom object to obtain internal coordinates for.

    :param offset: int
        Index offset used to construct internal coordinates for separate chains.

    :param preferred_dihedral: list
        Atom names defining the preferred dihedral to be use in the bond-angle-torsion coordinate system.

    :return: ICAtom
        Atom with internal coordinates.
    """
    atom = mol.atoms[dihedral[-1]]
    if len(dihedral) == 1:
        return ICAtom(
            atom.name,
            atom.type,
            dihedral[-1] - offset,
            atom.resname,
            atom.resid,
            (atom.name,),
        )

    elif len(dihedral) == 2:
        atom2 = mol.atoms[dihedral[-2]]
        atom_names = (atom.name, atom2.name)
        return ICAtom(
            atom.name,
            atom.type,
            dihedral[-1] - offset,
            atom.resname,
            atom.resid,
            atom_names,
            dihedral[-2] - offset,
            np.linalg.norm(atom.position - atom2.position),
        )

    elif len(dihedral) == 3:

        atom2 = mol.atoms[dihedral[-2]]
        atom3 = mol.atoms[dihedral[-3]]

        atom_names = (
            atom.name,
            atom2.name,
            atom3.name,
        )
        return ICAtom(
            atom.name,
            atom.type,
            dihedral[-1] - offset,
            atom.resname,
            atom.resid,
            atom_names,
            dihedral[-2] - offset,
            np.linalg.norm(atom.position - atom2.position),
            dihedral[-3] - offset,
            get_angle(mol.atoms[dihedral[-3:]].positions),
        )

    else:

        atom4, atom3, atom2 = mol.atoms[dihedral[:-1]]

        atom_names = (atom.name, atom2.name, atom3.name, atom4.name)

        if atom_names != ("N", "C", "CA", "N"):
            dihedral_resi = atom.resnum
        else:
            dihedral_resi = atom.resnum - 1

        p1, p2, p3, p4 = mol.atoms[dihedral].positions

        bl = np.linalg.norm(p3 - p4)
        al = get_angle([p2, p3, p4])
        dl = get_dihedral([p1, p2, p3, p4])

        if any(np.isnan(np.array([bl, al, dl]))):
            print("bd")

        return ICAtom(
            atom.name,
            atom.type,
            dihedral[-1] - offset,
            atom.resname,
            atom.resnum,
            atom_names,
            dihedral[-2] - offset,
            bl,
            dihedral[-3] - offset,
            al,
            dihedral[-4] - offset,
            dl,
            dihedral_resi=dihedral_resi,
        )


def get_ICResidues(ICAtoms):
    residues = {}
    prev_res = None
    for atom in ICAtoms:
        if atom.dihedral_resi == prev_res:
            residues[prev_res][atom.atom_names] = atom
        else:
            prev_res = atom.dihedral_resi
            residues[prev_res] = {atom.atom_names: atom}

    return residues


def get_n_pred(G, node, n, inner=False):
    """get the n predecessors of node i"""

    # Check if node is new chain
    if G.in_degree(node) == 0:
        n = 0

    # Try the most direct line
    dihedral = [node]
    for i in range(n):
        active = dihedral[-1]
        try:
            dihedral.append(next(G.predecessors(active)))
        except:
            break

    dihedral.reverse()

    # Try all ordered lines
    if len(dihedral) != n + 1:
        dihedral = [node]
        tmp = []
        for p in G.predecessors(node):
            tmp = get_n_pred(G, p, n-1, inner=True)
            if len(tmp) == n:
                break

        dihedral = tmp + dihedral

    # get any path or start a new segment
    if len(dihedral) != n + 1 and not inner:
        uG = G.to_undirected()
        path = [node]
        for i in nx.single_source_shortest_path_length(uG, node, cutoff=n):
            if i < node:
                path = nx.shortest_path(uG, i, node)

        dihedral = path

    return dihedral


def get_internal_coords(
    mol: Union[MDAnalysis.Universe, MDAnalysis.AtomGroup],
    resname: str = None,
    preferred_dihedrals: List = None,
) -> ProteinIC:
    """
    Gather a list of Internal

    :param mol: MDAnalysis.Universe, MDAnalysis.AtomGroup
        Molecule to convert into internal coords.

    :param resname: str
        Residue name (3-letter code) of any non-canonical or otherwise unsupported amino acid that should be included in
        the ProteinIC object.

    :param preferred_dihedrals: list
        Atom names of  preffered dihedral definitions to be used in the bond-angle-torsion coordinate system. Often
        used to specify dihedrals of user defined or unsupported amino acids that the user wishes to directly interact
        with.

    :return ICs: ProteinIC
        An ProteinIC object of the supplied molecule
    """
    mol = mol.select_atoms("not (byres name OH2 or resname HOH)")
    U = mol.universe

    tree = cKDTree(mol.atoms.positions)
    pairs = tree.query_pairs(5., output_type='ndarray')

    a_atoms = pairs[:, 0]
    b_atoms = pairs[:, 1]

    a = chiLife.get_lj_rmin(mol.atoms[a_atoms].types)
    b = chiLife.get_lj_rmin(mol.atoms[b_atoms].types)
    join = chiLife.get_lj_rmin('join_protocol')[()]
    ab = join(a, b, flat=True) * 0.6

    dist = np.linalg.norm(mol.atoms.positions[a_atoms] - mol.atoms.positions[b_atoms], axis=1)
    bonds = pairs[dist < ab]

    G = nx.DiGraph()
    G.add_edges_from(bonds)
    dihedrals = [get_n_pred(G, node, np.minimum(node, 3)) for node in range(len(mol.atoms))]

    if preferred_dihedrals is not None:
        present = False
        for dihe in preferred_dihedrals:
            idx_of_interest = np.argwhere(mol.atoms.names == dihe[-1]).flatten()
            for idx in idx_of_interest:
                if np.all(mol.atoms[dihedrals[idx]].names == dihe):
                    present = True
                    continue

                dihedral = [idx]
                tmp = []
                for p in G.predecessors(idx):
                    tmp = get_n_pred(G, p, 2, inner=True)
                    if np.all(mol.atoms[tmp].names == dihe[:-1]):
                        dihedral = tmp + dihedral
                        break

                if len(dihedral) == 4:
                    present = True
                    dihedrals[dihedral[-1]] = dihedral
        if not present:
            raise ValueError(f'There is no dihedral `{dihe}` in the provided protien. Perhaps there is typo or the '
                             f'atoms are not sorted correctly')


    idxstop = [i for i, sub in enumerate(dihedrals) if len(sub) == 1]
    dihedral_segs = [dihedrals[s:e] for s, e in zip(idxstop, (idxstop + [None])[1:])]

    all_ICAtoms = {}
    chain_operators = {}
    segid = 0

    for seg in dihedral_segs:
        if len(seg[4]) == 3:
            seg[4] = list(reversed(seg[3][1:])) + seg[4][-1:]

        offset = seg[0][0]
        segid += 1
        mx, ori = ic_mx(*mol.atoms[seg[2]].positions)

        chain_operators[segid] = {"ori": ori, "mx": mx}
        #
        ICatoms = [
            get_ICAtom(mol, dihedral, offset=offset, preferred_dihedral=preferred_dihedrals)
            for dihedral in seg
        ]
        all_ICAtoms[segid] = get_ICResidues(ICatoms)

    # Get bonded pairs within selection
    bonded_pairs = bonds

    return ProteinIC.from_ICatoms(
        all_ICAtoms, chain_operators=chain_operators, bonded_pairs=bonded_pairs
    )


def save_rotlib(name: str, atoms: ArrayLike, coords: ArrayLike = None) -> None:
    """
    Save a rotamer library as multiple states of the same molecule.

    :param name: str
        file name to save rotamer library to

    :param atoms: list, tuple
        list of Atom objects

    :param coords: np.ndarray
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


def save_pdb(
    name: Union[str, Path], atoms: ArrayLike, coords: ArrayLike, mode: str = "w"
) -> None:
    """
    Save a single state pdb structure of the provided atoms and coords

    :param name: str
        Name of file to save

    :param atoms: list, tuple
        List of Atom objects to be saved

    :param coords: np.ndarray
         Array of atom coordinates corresponding to atoms

    :param mode: str
        File open mode. Usually used to specify append ("a") when you want to add structures to a PDB rather than
        overwrite that pdb.
    """
    name = Path(name) if isinstance(name, str) else name
    name = name.with_suffix(".pdb")

    with open(name, mode, newline="\n") as f:
        for atom, coord in zip(atoms, coords):
            f.write(
                f"ATOM  {atom.index + 1:5d} {atom.name:^4s} {atom.resn:3s} {'A':1s}{atom.resi:4d}    "
                f"{coord[0]:8.3f}{coord[1]:8.3f}{coord[2]:8.3f}{1.0:6.2f}{1.0:6.2f}          {atom.atype:>2s}  \n"
            )


def get_missing_residues(
    protein: Union[MDAnalysis.Universe, MDAnalysis.AtomGroup],
    ignore: Set[int] = None,
    use_H: bool = False,
) -> List:
    """
    Get a list of RotamerLibrary objects corresponding to the residues of the provided protein that are missing heavy
    atoms
    :param protein: MDAnalysis.Universe, MDAnalysis.AtomGroup
        Protein to search for residues with missing atoms.
    :param ignore: ArrayLike
        List of residue numbers to ignore. Usually sites you plan to label or mutate.
    :param use_H: bool
        Whether the new side chain should have hydrogen atoms
    :return: ArrayLike
        List of RotamerLibrary objects corresponding to residues with missing heavy atoms
    """
    ignore = set() if ignore is None else ignore
    missing_residues = []
    cache = {}

    for res in protein.residues:
        # Only consider supported residues because otherwise chiLife wouldn't know what's missing
        if (
            res.resname not in chiLife.SUPPORTED_RESIDUES
            or res.resnum in ignore
            or res.resname in ["ALA", "GLY"]
        ):
            continue

        # Check if there are any missing heavy atoms
        heavy_atoms = res.atoms.types[res.atoms.types != "H"]
        if len(heavy_atoms) != cache.get(
            res.resname, len(RotamerLibrary(res.resname).atom_names)
        ):
            missing_residues.append(
                RotamerLibrary(
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
    *rotlibs: RotamerLibrary,
    add_missing_atoms: bool = True,
    random_rotamers: bool = False,
    **kwargs
) -> MDAnalysis.Universe:
    """
    Create a new Universe where the native residue is replaced with the highest probability rotamer from a
    RotamerLibrary or SpinLabel object.

    :param protein: MDAnalysis.Universe
        Universe containing protein to be spin labeled

    :param rotlibs: RotamerLibrary, SpinLabel
        Precomputed RotamerLibrary or SpinLabel object to use for selecting and replacing the spin native amino acid

    :param random_rotamers: bool
        Randomize rotamer conformations

    :param add_missing_atoms:
        Remodel side chains missing atoms

    :return U: MDAnalysis.Universe
        New Universe with a copy of the spin labeled protein with the highest probability rotamer
    """

    # Check for dRotamerLibraries in rotlibs
    trotlibs = []
    for lib in rotlibs:
        if isinstance(lib, RotamerLibrary):
            trotlibs.append(lib)
        elif isinstance(lib, dSpinLabel):
            trotlibs.append(lib.SL1)
            trotlibs.append(lib.SL2)
        else:
            raise TypeError(
                f"mutate only accepts RotamerLibrary, SpinLabel and dSpinLabel objects, not {lib}."
            )

    rotlibs = trotlibs

    if add_missing_atoms:
        if len(rotlibs) > 0 and all(not hasattr(lib, "H_mask") for lib in rotlibs):
            use_H = True
        elif any(not hasattr(lib, "H_mask") for lib in rotlibs):
            raise AttributeError(
                "User provided some rotlibs with hydrogen atoms and some without. Make sure all "
                "rotlibs either do or do not use hydrogen"
            )
        else:
            use_H = False

        missing_residues = get_missing_residues(
            protein, ignore={res.site for res in rotlibs}, use_H=use_H
        )
        rotlibs = list(rotlibs) + missing_residues

    label_sites = {
        (int(spin_label.site), spin_label.chain): spin_label for spin_label in rotlibs
    }

    protein = protein.select_atoms(
        f'(protein or resname {" ".join(chiLife.SUPPORTED_RESIDUES)}) and not altloc B'
    )
    label_selstr = " or ".join([f"({label.selstr})" for label in rotlibs])
    other_atoms = protein.select_atoms(f"not ({label_selstr})")

    # Get new universe information
    n_residues = len(other_atoms.residues) + len(rotlibs)
    n_atoms = len(other_atoms) + sum(
        len(spin_label.atom_names) for spin_label in rotlibs
    )
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
            atom_info += [
                (i, name, atype)
                for name, atype in zip(
                    label_sites[resloc].atom_names, label_sites[resloc].atom_types
                )
            ]

            # Add missing Oxygen from rotamer libraries
            res_names.append(label_sites[resloc].res)
            segidx.append(label_sites[resloc].segindex)

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
    segids = list(Counter(protein.residues.segids))
    # Allocate a new universe with the appropriate information
    U = mda.Universe.empty(
        n_atoms,
        n_residues=n_residues,
        atom_resindex=residx,
        residue_segindex=segidx,
        trajectory=True,
    )

    # Add necessary topology attributes
    U.add_TopologyAttr("name", atom_names)
    U.add_TopologyAttr("type", atom_types)
    U.add_TopologyAttr("resname", res_names)
    U.add_TopologyAttr("resid", resids)
    U.add_TopologyAttr("altLoc", ["A" for atom in range(n_atoms)])
    U.add_TopologyAttr("resnum", resids)
    U.add_TopologyAttr("segid")

    for i, segid in enumerate(segids):
        if i == 0:
            i_segment = U.segments[0]
            i_segment.segid = segid
        else:
            i_segment = U.add_Segment(segid=str(segid))

        mask = np.argwhere(np.asarray(segidx) == i).squeeze()
        U.residues[mask.tolist()].segments = i_segment

    U.add_TopologyAttr(Segids(np.array(segids)))
    U.add_TopologyAttr(Atomindices())
    U.add_TopologyAttr(Resindices())
    U.add_TopologyAttr(Segindices())

    # Apply old coordinates to non-spinlabel atoms
    new_other_atoms = U.select_atoms(f"not ({label_selstr})")
    new_other_atoms.atoms.positions = other_atoms.atoms.positions

    # Apply most probable spin label coordinates to spin label atoms
    for spin_label in label_sites.values():
        sl_atoms = U.select_atoms(spin_label.selstr)
        if random_rotamers:
            sl_atoms.atoms.positions = spin_label._coords[
                np.random.choice(len(spin_label._coords), p=spin_label.weights)
            ]
        else:
            sl_atoms.atoms.positions = spin_label._coords[np.argmax(spin_label.weights)]

    return U


def randomize_rotamers(
    protein: Union[mda.Universe, mda.AtomGroup],
    rotamer_libraries: List[RotamerLibrary],
    **kwargs,
) -> None:
    """
    Modify a protein object in place to randomize side chain conformations.

    :param protein: MDAnalysis.Universe, MDAnalysis.AtomGroup
        Protein object to modify.

    :param rotamer_libraries: list
        RotamerLibrary objects attached to the protein corresponding to the residues to be repacked/randomized.
    """
    for rotamer in rotamer_libraries:
        coords, weight = rotamer.sample(off_rotamer=kwargs.get("off_rotamer", False))
        mask = ~np.isin(protein.ix, rotamer.clash_ignore_idx)
        protein.atoms[~mask].positions = coords


def get_sas_res(
    protein: Union[mda.Universe, mda.AtomGroup], cutoff: float = 30
) -> Set[Tuple[int, str]]:
    """
    Run FreeSASA to get solvent accessible surface residues in the provided protein

    :param protein: MDAnalysis.Universe, MDAnalysis.AtomGroup
        Protein object to measure Solvent Accessible Surfaces (SAS) area of and report the SAS residues.

    :param cutoff:
        Exclude residues from list with SASA below cutoff in angstroms squared.

    :return: SAResi
        Set of solvent accessible surface residues
    """
    freesasa.setVerbosity(1)
    # Create FreeSASA Structure from MDA structure
    fSASA_structure = freesasa.Structure()

    [
        fSASA_structure.addAtom(
            atom.name, atom.resname, str(atom.resnum), atom.segid, *atom.position
        )
        for atom in protein.atoms
    ]

    residues = [(resi.segid, resi.resnum) for resi in protein.residues]

    # Calculate SASA
    SASA = freesasa.calc(fSASA_structure)
    SASA = freesasa.selectArea(
        (f"{resi}_{chain}, resi {resi} and chain {chain}" for chain, resi in residues),
        fSASA_structure,
        SASA,
    )

    SAResi = [key.split("_") for key in SASA if SASA[key] >= cutoff]
    SAResi = [(int(R), C) for R, C in SAResi]

    return set(SAResi)


def fetch(pdbid: str, save: bool = False) -> MDAnalysis.Universe:
    """
    Fetch pdb file from the protein data bank and optionally save to disk.

    :param pdbid: str
        4 letter structure ID

    :param save: bool
        If true the fetched PDB will be saved to the disk.

    :return U: MDAnalysis.Universe
        MDAnalysis Universe object of the protein corresponding to the provided PDB ID
    """

    if not pdbid.endswith(".pdb"):
        pdbid += ".pdb"

    urllib.request.urlretrieve(f"http://files.rcsb.org/download/{pdbid}", pdbid)

    U = mda.Universe(pdbid, in_memory=True)

    if not save:
        os.remove(pdbid)

    return U


def atom_sort_key(pdb_line: str, include_name=False) -> Tuple[str, int, int]:
    """
    Assign a base rank to sort atoms of a pdb.

    :param pdb_line: str
        ATOM line from a pdb file as a string.

    :return: chainid, resid, name_order
        ordered ranking of atom for sorting the pdb.
    """
    chainid = pdb_line[21]
    res_name = pdb_line[17:20].strip()
    resid = int(pdb_line[22:26].strip())
    atom_name = pdb_line[12:17].strip()
    atom_type = pdb_line[76:79].strip()
    if res_name == "ACE":
        name_order = (
            {"CH3": 0, "C": 1, "O": 2}.get(atom_name, 4) if atom_type != "H" else 5
        )
    else:
        name_order = atom_order.get(atom_name, 4) if atom_type != "H" else 5

    if include_name:
        return chainid, resid, name_order, atom_name[-1]
    else:
        return chainid, resid, name_order


def pose2mda(pose):
    """
    Create an MDAnalysis universe from a pyrosetta pose
    :param pose: pyrosetta.rosetta.core.Pose
        pyrosetta pose object.

    :return mda_protein: MDAnalysis.Universe
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
    n_atoms = len(coords)

    segindices = np.array([0] * n_residues)
    resnames = np.array([res.name() for res in pose])
    resnums = np.array([res.seqpos() for res in pose])

    mda_protein = mda.Universe.empty(
        n_atoms,
        n_residues=n_residues,
        atom_resindex=resindices,
        residue_segindex=segindices,
        trajectory=True,
    )

    mda_protein.add_TopologyAttr("type", atypes)
    mda_protein.add_TopologyAttr("resnum", resnums)
    mda_protein.add_TopologyAttr("resids", resnums)
    mda_protein.add_TopologyAttr("resname", resnames)
    mda_protein.add_TopologyAttr("name", anames)
    mda_protein.add_TopologyAttr("segid", ["A"])
    mda_protein.add_TopologyAttr("altLocs", [""] * len(atypes))
    mda_protein.atoms.positions = coords

    return mda_protein


def sort_pdb(pdbfile: Union[str, List], index=False) -> Union[List[str], List[int]]:
    """
    Read ATOM lines of a pdb and sort the atoms according to chain, residue index, backbone atoms and side chain atoms.
    Side chain atoms are sorted by distance to each other/backbone atoms with atoms closest to the backbone coming
    first and atoms furthest from the backbone coming last. This sorting is essential to making internal-coordinates
    with consistent and preferred dihedral definitions.

    :param pdbfile: str, list
        Name of the PDB file or a list of strings containing ATOM lines of a PDB file

    :return lines: list
        Sorted list of strings corresponding to the ATOM entries of a PDB file.
    """

    if isinstance(pdbfile, str):
        with open(pdbfile, "r") as f:
            lines = f.readlines()

        start_idxs = []
        end_idxs = []

        for i, line in enumerate(lines):
            if "MODEL" in line:
                start_idxs.append(i)
            elif "ENDMDL" in line:
                end_idxs.append(i)

        if start_idxs != []:
            # Assume all models have the same atoms
            idxs = sort_pdb(lines[start_idxs[0]:end_idxs[0]], index=True)
            lines[:] = [
                [
                    lines[idx + start][:6] + f"{i + 1:5d}" + lines[idx + start][11:]
                    for i, idx in enumerate(idxs)
                ]
                for start in start_idxs
            ]
            return lines

    elif isinstance(pdbfile, list):
        lines = pdbfile

    index_key = {line: i for i, line in enumerate(lines)}
    lines = [line for line in lines if line.startswith(("ATOM", "HETATM"))]

    # Presort
    lines.sort(key=lambda x: atom_sort_key(x))

    coords = np.array(
        [[float(line[30:38]), float(line[38:46]), float(line[46:54])] for line in lines]
    )

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
        kdtree = cKDTree(coords[start:stop])

        # Get all nearest neighbors and sort by distance
        pairs = kdtree.query_pairs(2.2, output_type="ndarray")
        distances = np.linalg.norm(
            kdtree.data[pairs[:, 0]] - kdtree.data[pairs[:, 1]], axis=1
        )
        distances = np.around(distances, decimals=3)
        idx_sort = np.lexsort((pairs[:, 0], pairs[:, 1], distances))
        pairs = pairs[idx_sort]

        # Start steming from CA atom
        idx = 1
        sorted_args = list(range(np.minimum(4, stop - start)))
        i = 0

        while len(sorted_args) < stop - start:
            # If you have already started adding atoms
            if "search_len" in locals():
                appendid = []

                # Go back to the first atom you added
                for idx in sorted_args[-search_len:]:

                    # skip hydrogen as base atom
                    if lines[idx+start][76:79].strip() == 'H':
                        continue

                    # Find anything bound to that atom that is not in already sorted
                    pairs_of_interest = pairs[np.any(pairs == idx, axis=1)]
                    for pair in pairs_of_interest:
                        ap = pair[0] if pair[0] != idx else pair[1]
                        if ap not in sorted_args and ap not in appendid:
                            appendid.append(ap)

                # Add all the new atoms to the sorted list
                if appendid != []:
                    sorted_args += appendid
                    search_len = len(appendid)
                elif search_len > len(sorted_args):
                    print('pause')
                else:
                    search_len += 1

            # If you have not added any atoms yet
            else:
                appendid = []
                # Look at the closest atoms
                for pair in pairs:
                    # Get atoms bound to this atom
                    if idx in pair:
                        ap = pair[0] if pair[0] != idx else pair[1]
                        if ap not in sorted_args:
                            appendid.append(ap)

                # If there are atoms bound
                if appendid != []:
                    # Add them to the list of sorted atoms and keep track of where you left off
                    sorted_args += appendid
                    search_len = len(appendid)
                else:
                    pass
                    # idx += 1
        midsort_key += [x + start for x in sorted_args]

        # Delete search length to start new on the next residue
        if "search_len" in locals():
            del search_len

    lines[:] = [lines[i] for i in midsort_key]
    lines.sort(key=atom_sort_key)

    # Return line indices if requested
    if index:
        return [index_key[line] for line in lines]

    # Otherwise replace atom index for new sorted liens
    lines = [line[:6] + f"{i + 1:5d}" + line[11:] for i, line in enumerate(lines)]

    return lines


# Define rotamer dihedral angle atoms
with open(os.path.join(os.path.dirname(__file__), "data/DihedralDefs.pkl"), "rb") as f:
    dihedral_defs = pickle.load(f)

with open(
    os.path.join(os.path.dirname(__file__), "data/rotamer_libraries/RotlibIndexes.pkl"),
    "rb",
) as f:
    rotlib_indexes = pickle.load(f)

atom_order = {"N": 0, "CA": 1, "C": 2, "O": 3}
