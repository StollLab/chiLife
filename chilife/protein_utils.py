import logging, os, urllib, pickle, itertools, math, rtoml
from operator import itemgetter
from pathlib import Path
from typing import Set, List, Union, Tuple, Dict
from numpy.typing import ArrayLike
from dataclasses import dataclass, replace
from collections import Counter, defaultdict
import MDAnalysis
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist

from MDAnalysis.core.topologyattrs import Atomindices, Resindices, Segindices, Segids
import MDAnalysis as mda

import chilife
from .numba_utils import _ic_to_cart, get_sasa
from .superimpositions import superimpositions
from .RotamerEnsemble import RotamerEnsemble
from .SpinLabel import SpinLabel
from .dSpinLabel import dSpinLabel

import networkx as nx


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
        method = chilife.superimpositions[method]

    p1, p2, p3 = p

    if method.__name__ == 'fit_superimposition':
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
        method = chilife.superimpositions[method]

    if method.__name__ == 'fit_superimposition':
        p = [pi[::-1] for pi in p]

    rotation_matrix, origin = method(*p)
    return rotation_matrix, origin


def ic_mx(*p: ArrayLike) -> Tuple[ArrayLike, ArrayLike]:
    """Calculates a rotation matrix and translation to transform a set of atoms to global coordinate frame from a local
    coordinated frame defined by ``p``. This function is a wrapper to quickly and easily transform cartesian
    coordinates, made from internal coordinates, to their appropriate location in the global frame.

    Parameters
    ----------
    p : ArrayLike
        Coordinates of the three atoms defining the local coordinate frame of the IC object

    Returns
    -------
    rotation_matrix : np.ndarray
        Rotation  matrix to rotate from the internal coordinate frame to the global frame
    origin : np.ndarray
        New origin position in 3 dimensional space
    """

    p1, p2, p3 = p

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
    """Internal coordinate atom class. Used for building the ProteinIC (internal coordinate protein).

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
    atom_names : tuple
        The names of the atoms that define the Bond-Angle-Torsion coordinates of the atom
    bond_idx : int
        Index of the coordinate atom bonded to this atom
    bond : float
        Distance of the bond in angstroms
    angle_idx : int
        Index of the atom that creates an angle with this atom and the bonded atom.
    angle : float
        The value of the angle between this atom, the bonded atom and the angle atom in radians.
    dihedral_idx : int
        The index of the atom that defines the dihedral with this atom, the bonded atom and the angled atom.
    dihedral : float
        The value of the dihedral angle defined above in radians.
    dihederal_resi : int
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


class ProteinIC:
    """
    A class for protein internal coordinates.

    Attributes
    ----------
    zmats : dict[np.ndarray]
        Dictionary of Z-matrices of the molecule. Each entry corresponds to a contiguous segment of bonded atoms.
    zmat_idxs : dict[np.ndarray]
        Dictionary of Z-matrix indices.
    atom_dict : dict
        Nested dictionaries containing indices of the atoms that define the coordinates (bond, angle dihedral) objects.
        Nesting pattern: [coord_type][chain][resi, stem][name] where coord_type is one of 'bond', 'angle' or 'dihedral',
        chain is the chainid, resi is the residue number, stem is a tuple of the names of the 3 atoms preceding the atom
        coordinate being defined.
    ICS : dict
        Nested dictionaries containing AtomIC objects. Nesting pattern: [chain][resi][stem][name] where chain is the
        chainid, resi is the residue number, stem is a tuple of the names of the 3 atoms preceding the atom coordinate
        being defined.
    atoms : np.ndarray[AtomIC]
        List of AtomIC objects making up the protein in order of their indices.
    atom_types : np.ndarray
        Array of atom types (elements) corresponding to the atom indices.
    atom_names : np.ndarray
        Array of atom names corresponding to the atom indices.
    resis : np.ndarray
        Ordered array of residue numbers.
    resnames : dict[int]:str
        Dictionary mapping residue numbers to their residue names.
    chains : np.ndarray
        Array of chainids
    bonded_pairs : np.ndarray
        Array of index pairs corresponding to atoms that are bonded.
    perturbed : bool
        Indicates if any internal coordinates have changed since the last time cartesian coordinates were calculated.
    dihedral_defs : list
        List of major side chain and backbone dihedral definitions as defined by user defined and supported residues.
    """

    def __init__(self, zmats: Dict, zmat_idxs: Dict, atom_dict: Dict, ICs: Dict, **kwargs: object):
        """
        ProteinIC constructor method.

        Parameters
        ----------
        zmats : dict
            Dictionary of Z-matrix indices.
        zmat_idxs : dict
            Dictionary of Z-matrix indices.
        atom_dict : dict
            Nested dictionaries containing indices of the atoms that define the coordinates (bond, angle dihedral) objects.
            Nesting pattern: [coord_type][chain][resi, stem][name] where coord_type is one of 'bond', 'angle' or 'dihedral',
            chain is the chainid, resi is the residue number, stem is a tuple of the names of the 3 atoms preceding the atom
            coordinate being defined.
        ICS : dict
            Nested dictionaries containing AtomIC objects. Nesting pattern: [chain][resi][stem][name] where chain is the
            chainid, resi is the residue number, stem is a tuple of the names of the 3 atoms preceding the atom coordinate
            being defined.

        **kwargs : dict
            Aditional arguments.
            - chain_operators
            - bonded_pairs
            - nonbonded_pairs
        """
        self.zmats = zmats
        self.zmat_idxs = zmat_idxs
        self.atom_dict = atom_dict
        self.ICs = ICs
        self.atoms = np.array([
            ic
            for chain in self.ICs.values()
            for resi in chain.values()
            for ic in resi.values()
        ])

        self.atom_types = np.array([atom.atype for atom in self.atoms])
        self.atom_names = np.array([atom.name for atom in self.atoms])

        self.resis = np.array([key for i in self.ICs for key in self.ICs[i]])
        self.resnames = {
            j: next(iter(self.ICs[i][j].values())).resn
            for i in self.ICs
            for j in self.ICs[i]
        }
        self.chains = np.array([name for name in ICs])

        self.chain_operators = kwargs.get("chain_operators", None)
        self.bonded_pairs = np.array(kwargs.get("bonded_pairs", None))
        self._nonbonded_pairs = kwargs.get('nonbonded_pairs', None)

        self.perturbed = False
        self._coords = kwargs['coords'] if 'coords' in kwargs else self.to_cartesian()
        self.dihedral_defs = self.collect_dih_list()

    @classmethod
    def from_ICatoms(cls, ICs, **kwargs):
        """Constructs a ProteinIC object from a collection of internal coords of atoms.

        Parameters
        ----------
        ICs : dict
            dictionary of dicts, of dicts containing ICAtom objects. Top dict specifies chain, middle dict specifies
            residue and bottom dict holds ICAtom objects.

        **kwargs : dict
            Aditional arguments.

            - ``chain_operators`` : Dictionary of tranlation vectors and rotation matrices that can transform chains from the
              internal coordinate frame to a global coordinate frame.
            - ``bonded_pairs`` : Array of atom index pairs indicating which atoms are bonded
            - ``nonbonded_pairs`` : Array of atom index pairs indicating which atoms are not bonded

        Returns
        -------
        ProteinIC
            A protein internal coords object
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
                  "_nonbonded_pairs": self._nonbonded_pairs,
                  'coords': self.coords.copy()}

        return ProteinIC(zmats, zmat_idxs, self.atom_dict, self.ICs, **kwargs)

    @property
    def chain_operators(self):
        """dict: A set of coordinate transformations that can orient multiple chains that are not covalently linked. e.g.
        structures with missing residues or protein complexes. The ``chain_operator`` property is a dictionary mapping
        chainIDs to a sub-dictionary containing a translation vector, ``ori``  and rotation matrix ``mx`` that will
        transform the protein coordinates from tie internal coordinate frame to some global frame.
        """
        return self._chain_operators

    @chain_operators.setter
    def chain_operators(self, op: Dict):
        """Assign or calculate operators for all chains.

        Parameters
        ----------
        op : dict
            Dictionary containing an entry for each chain in the ProteinIC molecule. Each entry must contain a
            rotation matrix, 'mx' and translation vector 'ori'.

        Returns
        -------
        None
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
        """np.ndarray : The cartesian coordinates of the protein"""
        if (self._coords is None) or self.perturbed:
            self._coords = self.to_cartesian()
            self.perturbed = False
        return self._coords

    @coords.setter
    def coords(self, coords: ArrayLike):
        """Setter to update the internal coords when the cartesian coords are altered

        Parameters
        ----------
        coords : ArrayLike
            cartesian coords to use to update.

        Returns
        -------
        None
        """
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

    @property
    def nonbonded_pairs(self):
        """np.ndarray: Array of atom index pairs of atoms that are not bonded"""
        if self._nonbonded_pairs is None and not (self.bonded_pairs is None or self.bonded_pairs.any() is None):
            bonded_pairs = {(a, b) for a, b in self.bonded_pairs}
            possible_bonds = itertools.combinations(range(len(self.atoms)), 2)
            self._nonbonded_pairs = np.fromiter(
                itertools.chain.from_iterable(
                    nb for nb in possible_bonds if nb not in bonded_pairs), dtype=int)

            self._nonbonded_pairs.shape = (-1, 2)

        return self._nonbonded_pairs

    def set_dihedral(
        self,
        dihedrals: Union[int, ArrayLike],
        resi: int,
        atom_list: ArrayLike,
        chain: Union[int, str] = None
    ):
        """Set one or more dihedral angles of a single residue in internal coordinates for the atoms defined in atom
        list.

        Parameters
        ----------
        dihedrals : float, ArrayLike
            Angle or array of angles to set the dihedral(s) to.
        resi : int
            Residue number of the site being altered
        atom_list : ArrayLike
            Names or array of names of atoms involved in the dihedral(s)
        chain : int, str, optional
            Chain containing the atoms that are defined by the dihedrals that are being set. Only necessary when there
            is more than one chain in the proteinIC object.

        Returns
        -------
        self : ProteinIC
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
                    + "\n".join([str(ic) for ic in self.ICs[chain][resi]])
                )

            if idxs:
                self.zmats[chain][idxs, 2] -= delta

        return self

    def get_dihedral(self, resi: int, atom_list: ArrayLike, chain: Union[int, str] = None):
        """Get the dihedral angle(s) of one or more atom sets at the specified residue. Dihedral angles are returned in
        radians

        Parameters
        ----------
        resi : int
            Residue number of the site being altered
        atom_list : ArrayLike
            Names or array of names of atoms involved in the dihedral(s)
        chain : int, str
             Chain identifier. required if there is more than one chain in the protein. Default value = None

        Returns
        -------
        angles: numpy.ndarray
            Array of dihedral angles corresponding to the atom sets in atom list.
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
                    + "\n".join([str(ic) for ic in self.ICs[chain][resi]])
                )
            dihedrals.append(self.zmats[chain][aidx, 2])

        return dihedrals[0] if len(dihedrals) == 1 else np.array(dihedrals)

    def to_cartesian(self):
        """Convert internal coordinates into cartesian coordinates.

        Returns
        -------
        coords_array : np.ndarray
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
            has_nan = False
            if np.any(np.isnan(cart_coords)):
                has_nan = True

            if has_nan:
                breakpoint()
            coord_arrays.append(cart_coords)

        return np.concatenate(coord_arrays)

    def to_site(self, *p: ArrayLike, method="bisect"):
        """
        Move the ProteinIC object to a site defined by ``p`` . Usually ``p`` Will be the coordinates of the N, CA, C
        backbone atoms. Alignment to this sight will be done by aligning the first residues (the internal coordinate
        frame) to the coordinates of ``p`` using the provided method.

        Parameters
        ----------
ond        p : ArrayLike
            Three 3D cartesian coordinates defining the site to move the ProteinIC object to.
        method : str
            Alignment method to use. See :mod:`Alignment Methods <chiLife.superimpositions>` .

        """
        p1, p2, p3 = p

        new_co = {}
        gmx, gori = global_mx(p1, p2, p3)
        lori, lmx = local_mx(*np.squeeze(self.coords[:3]), method=method)
        m2m3 = lmx @ gmx
        for segid in self.chain_operators:

            new_mx = self.chain_operators[segid]["mx"] @ m2m3
            new_ori = (self.chain_operators[segid]["ori"] - lori) @ m2m3 + gori
            new_co[segid] = {"mx": new_mx, "ori": new_ori}

        self.chain_operators = new_co
        self.perturbed = True

    def save_pdb(self, filename: str, mode="w"):
        """Save a pdb structure file from a ProteinIC object

        Parameters
        ----------
        filename : str
            Name of file to save
        mode : str
            file open mode (Default value = "w")
        """
        if "a" in mode:
            with open(str(filename), mode, newline="\n") as f:
                f.write("MODEL\n")
            save_pdb(filename, self.atoms, self.coords, mode=mode)
            with open(str(filename), mode, newline="\n") as f:
                f.write("ENDMDL\n")
        else:
           save_pdb(filename, self.atoms, self.to_cartesian(), mode=mode)

    def has_clashes(self, distance: float = 1.5) -> bool:
        """Checks for an internal clash between non-bonded atoms.

        Parameters
        ----------
        distance : float
            Minimum distance allowed between non-bonded atoms in angstroms. (Default value = 1.5).

        Returns
        -------

        """
        diff = (
            self.coords[self.nonbonded_pairs[:, 0]]
            - self.coords[self.nonbonded_pairs[:, 1]]
        )
        dist = np.linalg.norm(diff, axis=1)
        has_clashes = np.any(dist < distance)
        return has_clashes

    def get_resi_dihs(self, resi: int) -> List[List[str]]:
        """Gets the list of heavy atom dihedral definitions for the provided residue as defined by preexisting and user
        defined rotamer ensemble.

        Parameters
        ----------
        resi : int
            Index of the residue whose dihedrals will be returned

        Returns
        -------
        dihedral_defs : List[List[str]]
            Lists of atom names defining the dihedrals of residue ``resi``.
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

    def collect_dih_list(self) -> List:
        """Returns a list of all heavy atom dihedrals as defined by preexisting and user defined rotamer libraries.

        Returns
        -------
        List[Tuple[int, List[str]]]
            List of all heavy atom dihedrals. Each element of the list contains a tuple with an int corresponding to the
            residue number and a list of dihedral definitions corresponding to the dihedrals of the residue.
        """
        dihs = []

        # Get backbone and sidechain dihedrals for the provided universe
        for resi in self.ICs[1]:
            resname = self.resnames[resi]
            res_dihs = self.get_resi_dihs(resi)
            res_dihs += dihedral_defs.get(resname, [])
            dihs += [(resi, d) for d in res_dihs]

        return dihs

    def shift_resnum(self, delta: int):
        """Shift all residue numbers of the proteinIC object by some integer, ``delta`` .

        Parameters
        ----------
        delta : int
            Integer by which you wish to shift the residue numbers of the ProteinIC object. Can be positive or
            negative.
        """
        for key in self.atom_dict:
            for chain in self.atom_dict[key]:

                self.atom_dict[key][chain] = {(res + delta, other): self.atom_dict[key][chain][res, other]
                                              for res, other in self.atom_dict[key][chain]}


        for segid in self.ICs:
            for resi in list(self.ICs[segid].keys()):
                if abs(delta) > 0:
                    self.ICs[segid][resi + delta] = self.ICs[segid][resi]
                    del self.ICs[segid][resi]

                for atom in self.ICs[segid][resi + delta].values():
                    atom.resi += delta
                    atom.dihedral_resi += delta

        self.resis = [key for i in self.ICs for key in self.ICs[i]]


def get_ICAtom(
    mol: mda.core.groups.Atom,
    dihedral: List[int],
    offset: int = 0
) -> ICAtom:
    """Construct internal coordinates for an atom given that atom is linked to an MDAnalysis Universe.

    Parameters
    ----------
     mol : mda.core.groups.Atom
        Atom group object containing the atom of interest and the preceding atoms that define the dihedral.
    dihedral : List[int]
        Dihedral defining the coordination of the atom being constructred
    offset : int
        Index offset used to construct internal coordinates for separate chains.

    Returns
    -------
    ICAtom
        Atom with internal coordinates.

    """
    atom = mol.atoms[dihedral[-1]]
    if len(dihedral) == 1:
        return ICAtom(atom.name, atom.type, dihedral[-1] - offset, atom.resname, atom.resid, (atom.name,))

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
        atom_names = (atom.name, atom2.name, atom3.name)

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

        if atom_names == ("N", "C", "CA", "N"):
            dihedral_resi = atom.resnum - 1
        else:
            dihedral_resi = atom.resnum

        p1, p2, p3, p4 = mol.atoms[dihedral].positions

        bl = np.linalg.norm(p3 - p4)
        al = get_angle([p2, p3, p4])
        dl = get_dihedral([p1, p2, p3, p4])

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


def get_ICResidues(ICAtoms: List[ICAtom]) -> Dict:
    """Create a collection of ICAtoms grouped by residue. This function will create a dictionary where the keys are
    residues and the values are the lists of ICAtoms of that residues.

    Parameters
    ----------
    ICAtoms : List[ICAtom]
        List of internal coordinate atoms  to seperate into groups

    Returns
    -------
    residues : Dict
        Dictionary where the keys are residues and the values are the lists of ICAtoms of that residues.
    """
    residues = {}
    prev_res = None
    for atom in ICAtoms:
        if atom.dihedral_resi == prev_res:
            residues[prev_res][atom.atom_names] = atom
        else:
            prev_res = atom.dihedral_resi
            residues[prev_res] = {atom.atom_names: atom}

    return residues


def get_n_pred(G : nx.DiGraph, node: int, n: int, inner: bool = False):
    """Given a directed graph, ``G`` and a ``node`` , find the ``n`` predecessors of ``node`` and
    return them as a list. The returned list will include ``node`` and be of length ``n + 1`` . For pathways with less
    than ``n`` predecessors the function will return all predecessors.

    Parameters
    ----------
    G : networkx.DiGraph
        The directed graph containing ``node`` .
    node : int
        The node of ``G`` for which you would like the predecessors of.
    n : int
        The number of predecessors to find.
    inner :
         (Default value = False)

    Returns
    -------
    predecessors : List[int]
        List of nodes that come before ``node``.
    """

    # Check if node is new chain
    if G.in_degree(node) == 0:
        n = 0

    # Try the most direct line
    predecessors = [node]
    for i in range(n):
        active = predecessors[-1]
        try:
            predecessors.append(next(G.predecessors(active)))
        except:
            break

    predecessors.reverse()

    # Try all ordered lines
    if len(predecessors) != n + 1:
        predecessors = [node]
        tmp = []
        for p in G.predecessors(node):
            tmp = get_n_pred(G, p, n-1, inner=True)
            if len(tmp) == n:
                break

        predecessors = tmp + predecessors

    # get any path or start a new segment
    if len(predecessors) != n + 1 and not inner:
        uG = G.to_undirected()
        path = [node]
        for i in nx.single_source_shortest_path_length(uG, node, cutoff=n):
            if i < node:
                path = nx.shortest_path(uG, i, node)

        predecessors = path

    return predecessors


def get_all_n_pred(G :nx.DiGraph, node: int, n: int) -> List[List[int]]:
    """ Get all possible permutations of ``n`` (or less) predecessors of ``node`` of the directed graph ``G`` .

    Parameters
    ----------
    G : networkx.DiGraph
        Directed graph containing ``node``.
    node : int
        Node you wish to find all predecessors of.
    n : int
        Max number of predecessors to find.

    Returns
    -------
    preds : List[List[int]]
        List of lists containing all permutations of ``n`` predecessors of ``node`` of ``G`` .
    """
    if n == 0:
        return [[node]]

    if G.in_degree(node) == 0 and n > 0:
        return [[node]]

    preds = []
    for parent in G.predecessors(node):
        for sub in get_all_n_pred(G, parent, n-1):
            preds.append([node] + sub)

    return preds


def get_internal_coords(
    mol: Union[MDAnalysis.Universe, MDAnalysis.AtomGroup],
    preferred_dihedrals: List = None,
    bonds: ArrayLike = None,
) -> ProteinIC:
    """Gather a list of internal coordinate atoms ( ``ICAtom`` ) from and MDAnalysis ``Universe`` of ``AtomGroup`` and
    create a ``ProteinIC`` object from them. If ``preferred_dihedrals`` is passed then any atom that can be defineid by
    a dihedral present in ``preferred_dihedrals`` will be.

    Parameters
    ----------
    mol : MDAnalysis.Universe, MDAnalysis.AtomGroup
        Molecule to convert into internal coords.
    resname : str
        Residue name (3-letter code) of any non-canonical or otherwise unsupported amino acid that should be included in
        the ProteinIC object.
    preferred_dihedrals : list
        Atom names of  preffered dihedral definitions to be used in the bond-angle-torsion coordinate system. Often
        used to specify dihedrals of user defined or unsupported amino acids that the user wishes to directly interact
        with.

    Returns
    -------
    ProteinIC
        An ``ProteinIC`` object of the supplied molecule.

    """
    mol = mol.select_atoms("not (byres name OH2 or resname HOH)")
    U = mol.universe
    bonds = bonds if bonds is not None else guess_bonds(mol.atoms.positions, mol.atoms.types)

    G = nx.DiGraph()
    G.add_edges_from(bonds)

    dihedrals = [get_n_pred(G, node, np.minimum(node, 3)) for node in range(len(mol.atoms))]

    if preferred_dihedrals is not None:
        present = False
        for dihe in preferred_dihedrals:

            # Get the index of the atom being defined by the prefered dihedral
            idx_of_interest = np.argwhere(mol.atoms.names == dihe[-1]).flatten()

            for idx in idx_of_interest:
                if np.all(mol.atoms[dihedrals[idx]].names == dihe):
                    present = True
                    break

                dihedral = [idx]
                for p in get_all_n_pred(G, idx, 3):
                    if np.all(mol.atoms[p[::-1]].names == dihe):
                        dihedral = p[::-1]
                        break

                if len(dihedral) == 4:
                    present = True
                    dihedrals[dihedral[-1]] = dihedral
        if not present and preferred_dihedrals != []:
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
        ICatoms = [get_ICAtom(mol, dihedral, offset=offset) for dihedral in seg]
        all_ICAtoms[segid] = get_ICResidues(ICatoms)

    # Get bonded pairs within selection
    bonded_pairs = bonds

    return ProteinIC.from_ICatoms(
        all_ICAtoms, chain_operators=chain_operators, bonded_pairs=bonded_pairs
    )


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
    """Save a single state pdb structure of the provided atoms and coords

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
    """Get a list of RotamerEnsemble objects corresponding to the residues of the provided protein that are missing heavy
    atoms

    Parameters
    ----------
    protein : MDAnalysis.Universe, MDAnalysis.AtomGroup
        Protein to search for residues with missing atoms.
    ignore : set
        List of residue numbers to ignore. Usually sites you plan to label or mutate.
    use_H : bool
        Whether the new side chain should have hydrogen atoms.

    Returns
    -------
    missing_residues : list
        List of RotamerEnsemble objects corresponding to residues with missing heavy atoms.
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
        if len(heavy_atoms) != cache.get(res.resname, len(RotamerEnsemble(res.resname).atom_names)):
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
        Universe containing protein to be spin labeled
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
    tensembles = []
    for lib in ensembles:
        if isinstance(lib, RotamerEnsemble):
            tensembles.append(lib)
        elif isinstance(lib, dSpinLabel):
            tensembles.append(lib.SL1)
            tensembles.append(lib.SL2)
        else:
            raise TypeError(
                f"mutate only accepts RotamerEnsemble, SpinLabel and dSpinLabel objects, not {lib}."
            )

    ensembles = tensembles

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

    label_sites = {
        (int(spin_label.site), spin_label.chain): spin_label for spin_label in ensembles
    }

    protein = protein.select_atoms(
        f'(not altloc B) and (not (byres name OH2 or resname HOH))'
    )
    label_selstr = " or ".join([f"({label.selstr})" for label in ensembles])
    other_atoms = protein.select_atoms(f"not ({label_selstr})")

    # Get new universe information
    n_residues = len(other_atoms.residues) + len(ensembles)
    n_atoms = len(other_atoms) + sum(
        len(spin_label.atom_names) for spin_label in ensembles
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

            # Add missing Oxygen from rotamer ensemble
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

    if isinstance(protein, (mda.Universe, mda.AtomGroup)):
        U = make_mda_uni(atom_names, atom_types, res_names, residx, resids, segidx, segids)
    elif isinstance(protein, chilife.BaseSystem):
        U = chilife.Protein.from_arrays(atom_names, atom_types, res_names, residx, resids, segidx, segids)

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
    rotamer_libraries: List[RotamerEnsemble],
    **kwargs,
) -> None:
    """Modify a protein object in place to randomize side chain conformations.

    Parameters
    ----------
    protein : MDAnalysis.Universe, MDAnalysis.AtomGroup
        Protein object to modify.
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
        Protein object to measure Solvent Accessible Surfaces (SAS) area of and report the SAS residues.
    cutoff : float
        Exclude residues from list with SASA below cutoff in angstroms squared.

    Returns
    -------
    SAResi : set
        Set of solvent accessible surface residues.

    """
    environment_coords = protein.atoms.positions
    environment_radii = chilife.get_lj_rmin(protein.atoms.types)
    atom_sasa = get_sasa(environment_coords, environment_radii, by_atom=True)

    SASAs = {(residue.resnum, residue.segid) for residue in protein.residues if
             atom_sasa[0, residue.atoms.ix].sum() >= cutoff}

    return SASAs


def fetch(accession_number: str, save: bool = False) -> MDAnalysis.Universe:
    """Fetch pdb file from the protein data bank or the AlphaFold Database and optionally save to disk.

    Parameters
    ----------
    accession_number : str
        4 letter structure PDBID or alpha fold accession number. Note that AlphaFold accession numbers must begin with
        'AF-'.
    save : bool
        If true the fetched PDB will be saved to the disk.

    Returns
    -------
    U : MDAnalysis.Universe
        MDAnalysis Universe object of the protein corresponding to the provided PDB ID or AlphaFold accession number

    """
    accession_number = accession_number.split('.pdb')[0]
    pdb_name = accession_number + '.pdb'

    if accession_number.startswith('AF-'):
        print(f"https://alphafold.ebi.ac.uk/files/{accession_number}-F1-model_v3.pdb")
        urllib.request.urlretrieve(f"https://alphafold.ebi.ac.uk/files/{accession_number}-F1-model_v3.pdb", pdb_name)
    else:
        urllib.request.urlretrieve(f"http://files.rcsb.org/download/{pdb_name}", pdb_name)

    U = mda.Universe(pdb_name, in_memory=True)

    if not save:
        os.remove(pdb_name)

    return U


def atom_sort_key(pdb_line: str) -> Tuple[str, int, int]:
    """Assign a base rank to sort atoms of a pdb.

    Parameters
    ----------
    pdb_line : str
        ATOM line from a pdb file as a string.

    Returns
    -------
    tuple :
        chainid, resid, name_order.
        ordered ranking of atom for sorting the pdb.
    """
    chainid = pdb_line[21]
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
        name_order = atom_order.get(atom_name, 4) if atom_type != "H" else 5

    return chainid, resid, name_order


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
    kdtree = cKDTree(coords)
    pairs = kdtree.query_pairs(4., output_type='ndarray')
    pair_names = [tuple(x) for x in atom_types[pairs].tolist()]
    bond_lengths = itemgetter(*pair_names)(chilife.bond_hmax_dict)
    a_atoms = pairs[:, 0]
    b_atoms = pairs[:, 1]

    dist = np.linalg.norm(coords[a_atoms] - coords[b_atoms], axis=1)
    bonds = pairs[dist < bond_lengths]

    return bonds


def get_min_topol(lines: List[List[str]]) -> Set[Tuple[int, int]]:
    """ Git the minimum topology shared by all the states/models a PDB ensemble. This is to ensure a consistent
    internal coordinate system between all conformers of an ensemble even when there are minor differences in topology.
    e.g. when dHis-Cu-NTA has the capping ligand in different bond orientations.

    Parameters
    ----------
    lines : List[List[str]]
        List of lists corresponding to individual states/models of a pdb file. All models must have the same stoma in
        the same order and only the coordinates should differ.

    Returns
    -------
    minimal_bond_set : Set[Tuple[int, int]]
        A set of tuples holding the indices of atom pairs which are thought to be bonded in all states/models.
    """
    bonds_list = []
    if isinstance(lines[0], str):
        lines = [lines]

    for struct in lines:

        coords = np.array([(line[30:38], line[38:46], line[46:54]) for line in struct], dtype=float)
        atypes = np.array([line[76:78].strip() for line in struct])
        pairs = guess_bonds(coords, atypes)
        bonds = set(tuple(pair) for pair in pairs)
        bonds_list.append(bonds)

    minimal_bond_set = set.intersection(*bonds_list)

    return minimal_bond_set


def sort_pdb(pdbfile: Union[str, List[str], List[List[str]]],
             uniform_topology: bool = True,
             index: bool = False,
             bonds: ArrayLike = None) -> Union[List[str], List[List[str]], List[int]]:
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

    if isinstance(pdbfile, str):
        with open(pdbfile, "r") as f:
            lines = f.readlines()

        start_idxs = []
        end_idxs = []

        for i, line in enumerate(lines):
            if line.startswith('MODEL'):
                start_idxs.append(i)
            elif line.startswith("ENDMDL"):
                end_idxs.append(i)

        if start_idxs != []:

            # Assume all models have the same topology
            idxs = sort_pdb(lines[start_idxs[0]:end_idxs[0]], index=True)
            lines[:] = [[lines[idx + start][:6] + f"{i + 1:5d}" + lines[idx + start][11:]
                        for i, idx in enumerate(idxs)]
                       for start in start_idxs]

            if not uniform_topology:
                min_bonds_list = get_min_topol(lines)
                idxs = sort_pdb(lines[0], index=True, bonds=min_bonds_list)

                lines[:] = [[struct[idx][:6] + f"{i + 1:5d}" + struct[idx][11:]
                            for i, idx in enumerate(idxs)]
                           for struct in lines]

            return lines

    elif isinstance(pdbfile, list):
        lines = pdbfile

    index_key = {line: i for i, line in enumerate(lines)}
    lines = [line for line in lines if line.startswith(("ATOM", "HETATM"))]

    # Presort
    lines.sort(key=atom_sort_key)
    parent_bonds = set(tuple(bond) for bond in bonds) if bonds is not None else set()
    coords = np.array(
        [[float(line[30:38]), float(line[38:46]), float(line[46:54])] for line in lines]
    )

    atypes = np.array([line[76:78].strip() for line in lines])

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
        sorted_args = list(range(np.minimum(4, n_heavy)))
        if len(sorted_args) != n_heavy:
            root_idx = 1 if len(sorted_args) == 4 else 0

            bonds = guess_bonds(coords[start:stop], atypes[start:stop])

            if len(parent_bonds) > 0:
                bonds = [bond for bond in bonds if tuple(bond + start) in parent_bonds]

            bonds = np.asarray(bonds)
            # Get all nearest neighbors and sort by distance
            distances = np.linalg.norm(coords[start:stop][bonds[:, 0]] - coords[start:stop][bonds[:, 1]], axis=1)
            distances = np.around(distances, decimals=3)
            idx_sort = np.lexsort((bonds[:, 0], bonds[:, 1], distances))
            pairs = bonds[idx_sort]

            pairs = [pair for pair in pairs if np.any(~np.isin(pair, sorted_args))]

            graph = nx.Graph()
            graph.add_edges_from(pairs)

            # Start stemming from CA atom
            CA_edges = [edge[1] for edge in nx.bfs_edges(graph, root_idx) if edge[1] not in sorted_args]

            # check for disconnected parts of residue
            if not nx.is_connected(graph):
                for g in nx.connected_components(graph):
                    if np.any([arg in g for arg in sorted_args]):
                        continue
                    CA_nodes = [idx for idx in CA_edges if atypes[start + idx] != 'H']
                    g_nodes = [idx for idx in g if atypes[start + idx] != 'H']
                    near_root = cdist(coords[start:stop][CA_nodes], coords[start:stop][g_nodes]).argmin()
                    # xidx = near_root // len(g_nodes)
                    yidx = near_root % len(g_nodes)
                    CA_edges += [g_nodes[yidx]] + [edge[1] for edge in nx.bfs_edges(graph, g_nodes[yidx]) if edge[1]]


        elif stop - start > n_heavy:
            # Assumes  non-heavy atoms come after the heavy atoms, which should be true because of the pre-sort
            CA_edges = list(range(n_heavy, n_heavy + (stop - start - len(sorted_args))))

        else:
            CA_edges = []

        n_base = len(sorted_args)
        sorted_args = sorted_args + CA_edges

        # get any leftover hydrogen atoms (eg HN)
        if len(sorted_args) != stop-start:
            for i, idx in enumerate(range(n_heavy, stop-start)):
                if idx not in sorted_args:
                    sorted_args.insert(i+n_base, idx)

        midsort_key += [x + start for x in sorted_args]

    lines[:] = [lines[i] for i in midsort_key]
    lines.sort(key=atom_sort_key)

    # Return line indices if requested
    if index:
        return [index_key[line] for line in lines]

    # Otherwise replace atom index for new sorted liens
    lines = [line[:6] + f"{i + 1:5d}" + line[11:] for i, line in enumerate(lines)]

    return lines


def make_mda_uni(anames: ArrayLike,
                 atypes: ArrayLike,
                 resnames: ArrayLike,
                 resindices: ArrayLike,
                 resnums: ArrayLike,
                 segindices: ArrayLike,
                 segids: ArrayLike = None,
) -> MDAnalysis.Universe:

    n_atoms = len(anames)
    n_residues = len(np.unique(resindices))

    if segids is None:
        segids = np.array(["ABCDEFGHIJKLMNOPQRSTUVWXYZ"[i] for i in range(len(np.unique(segindices)))])

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

        mask = np.argwhere(np.asarray(segindices) == i).squeeze()
        mda_uni.residues[mask.tolist()].segments = i_segment

    mda_uni.add_TopologyAttr(Segids(np.array(segids)))
    mda_uni.add_TopologyAttr(Atomindices())
    mda_uni.add_TopologyAttr(Resindices())
    mda_uni.add_TopologyAttr(Segindices())

    return mda_uni

DATA_DIR = Path(__file__).parent.absolute() / "data/"
RL_DIR = Path(__file__).parent.absolute() / "data/rotamer_libraries/"

# Define rotamer dihedral angle atoms
with open(DATA_DIR / "dihedral_defs.toml", "r") as f:
    dihedral_defs = rtoml.load(f)

with open(RL_DIR / "RotlibIndexes.pkl", "rb") as f:
    rotlib_indexes = pickle.load(f)

with open(DATA_DIR / 'BondDefs.pkl', 'rb') as f:
    bond_hmax_dict = {key: (val + 0.4 if 'H' in key else val + 0.35) for key, val in pickle.load(f).items()}
    bond_hmax_dict = defaultdict(lambda : 0, bond_hmax_dict)
    def bond_hmax(a): return bond_hmax_dict.get(tuple(i for i in a), 0)
    bond_hmax = np.vectorize(bond_hmax, signature="(n)->()")

atom_order = {"N": 0, "CA": 1, "C": 2, "O": 3}


nataa_codes = {'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
               'LYS': 'K', 'LEU': 'L', 'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 'ARG': 'R', 'SER': 'S',
               'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'}
inataa = {val: key for key, val in nataa_codes.items()}

nataa_codes.update(inataa)
del inataa
