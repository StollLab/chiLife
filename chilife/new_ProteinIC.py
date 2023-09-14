import itertools
import logging
from dataclasses import dataclass
from typing import List, Union, Dict, Tuple

from memoization import cached, suppress_warnings
import MDAnalysis
import igraph as ig
import numpy as np
from numpy.typing import ArrayLike
from .Protein import MolecularSystem, Trajectory
from .Topology import Topology
from .protein_utils import dihedral_defs, save_pdb, local_mx, global_mx, get_angles, get_dihedrals, guess_bonds
from .numba_utils import _ic_to_cart


class newProteinIC:
    """
    A class for protein internal coordinates.

    Attributes
    ----------
    zmats : dict[np.ndarray]
        Dictionary of Z-matrices of the molecule. Each entry corresponds to a contiguous segment of bonded atoms.
    z_matrix_idxs : dict[np.ndarray]
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

    def __init__(self,
                 z_matrix: ArrayLike,
                 z_matrix_idxs: ArrayLike,
                 protein: Union[MolecularSystem, MDAnalysis.Universe, MDAnalysis.AtomGroup],
                 **kwargs):
        """
        ProteinIC constructor method.

        Parameters
        ----------
        """
        # Internal coords
        self.protein = protein.atoms if isinstance(protein, MDAnalysis.Universe) else protein
        self.trajectory = Trajectory(z_matrix, self)
        self.z_matrix_idxs = z_matrix_idxs
        self._chain_operator_idxs = kwargs.get('chain_operator_idxs', None)
        self._chain_operators = kwargs.get('chain_operators', None)

        # Topology
        self.bonds = kwargs['bonds'] if 'bonds' in kwargs else \
            guess_bonds(protein.positions, protein.types)
        self._nonbonded = kwargs.get('nonbonded', None)
        self.topology = kwargs['topology'] if 'topology' in kwargs else \
            Topology(np.arange(len(self.protein.atoms)), self.bonds)

        # Misc
        self.perturbed = False
        self._coords = kwargs['coords'] if 'coords' in kwargs \
            else np.array([self.protein.positions for ts in self.protein.universe.trajectory])
        self._dihedral_defs = None

    @classmethod
    def from_protein(cls,
                     protein: Union[MDAnalysis.Universe, MDAnalysis.AtomGroup, MolecularSystem],
                     preferred_dihedrals: List = None,
                     bonds: ArrayLike = None,
                     **kwargs: Dict):

        if kwargs.get('ignore_water', True):
            protein = protein.select_atoms("not (byres name OH2 or resname HOH)")
        elif isinstance(protein, MDAnalysis.Universe):
            protein = protein.atoms

        # Explicit passing of bonds means no other  bonds will be considered.
        if bonds is not None:
            bonds = bonds.copy()
        # Otherwise use bonds defined by the protein object
        elif hasattr(protein, 'bonds'):
            bonds = protein.bonds
        # Guess the bonds as a last resort
        else:
            # Add selection's lowest index in case the user is selecting a subset of atoms from a larger system.
            bonds = guess_bonds(protein.positions, protein.types) + protein[0].ix

        # Keep track of atom indexes in parent protein object in case atoms come from a larger system
        atom_idxs = protein.ix.tolist()

        # Remove cap atoms for atom_idxs
        cap = kwargs.get('cap', [])
        if cap:
            atom_idxs = reconfigure_cap(cap, atom_idxs, bonds)

        topology = Topology(atom_idxs, bonds)
        z_matrix_dihedrals = topology.get_zmatrix_dihedrals()
        z_mat_map = {k[-1]: i for i, k in enumerate(z_matrix_dihedrals)}

        if preferred_dihedrals:
            present = False
            for dihe in preferred_dihedrals:

                # Get the indices of the atom being defined by the preferred dihedral
                preferred_idx = np.argwhere(protein.names == dihe[-1]).flatten()
                u_idx_of_interest = protein[preferred_idx].ix
                idx_of_interest = np.argwhere(np.isin(atom_idxs, u_idx_of_interest)).flatten()

                for idx, uidx in zip(idx_of_interest, u_idx_of_interest):
                    # Check if it is already in use
                    if np.all(protein.universe.atoms[z_matrix_dihedrals[idx]].names == dihe):
                        present = True
                        continue

                    # Check for alternative dihedral definitions that satisfy the preferred dihedral
                    for p in topology.dihedrals_by_atoms[uidx]:
                        if np.all(protein.universe.atoms[p].names == dihe):
                            dihedral = [a for a in p]
                            break
                    else:
                        dihedral = [uidx]

                    # If an alternative is found, replace it in the dihedral list.
                    if len(dihedral) == 4:
                        present = True
                        z_matrix_dihedrals[idx] = dihedral

                        # also change any other dependent dihedrals
                        for dihe_idxs in topology.dihedrals_by_bonds[tuple(dihedral[1:3])]:
                            zmidx = z_mat_map[dihe_idxs[-1]]
                            tmp = z_matrix_dihedrals[zmidx]
                            if all(a == b for a, b in zip(tmp[1:3], dihedral[1:3])):
                                z_matrix_dihedrals[zmidx, 0] = dihedral[0]
            if not present and preferred_dihedrals != []:
                raise ValueError(f'There is no dihedral `{dihe}` in the provided protein. Perhaps there is typo or the '
                                 f'atoms are not sorted correctly')

        z_matrix_coordinates = np.zeros((len(protein.universe.trajectory), len(z_matrix_dihedrals), 3))
        z_matrix_dihedrals = zmatrix_idxs_to_local(z_matrix_dihedrals)
        nan_int = -2147483648

        chain_operator_idxs = get_chainbreak_idxs(z_matrix_dihedrals)
        chain_operators = []

        for i, ts in enumerate(protein.universe.trajectory):
            z_matrix, chain_operator = get_z_matrix(protein.atoms.positions, z_matrix_dihedrals, chain_operator_idxs)
            z_matrix_coordinates[i] = z_matrix
            chain_operators.append(chain_operator)

        return cls(z_matrix_coordinates, z_matrix_dihedrals, protein,
                   chain_operators=chain_operators, chain_operator_idxs=chain_operator_idxs)

    def copy(self):
        """Create a deep copy of an ProteinIC instance"""
        z_matrix = self.trajectory.coordinates_array.copy()
        z_matrix_idxs = self.z_matrix_idxs.copy()
        kwargs = {'chain_operators': self._chain_operators,
                  'chain_operator_idxs': self._chain_operator_idxs,
                  'bonds': self.bonds,
                  'nonbonded': self._nonbonded,
                  'coords': self._coords.copy(),
                  'topology': self.topology,
                  'protein': self.protein}

        return newProteinIC(z_matrix, z_matrix_idxs, **kwargs)

    @property
    def chain_operator_idxs(self):
        return self._chain_operator_idxs

    @chain_operator_idxs.setter
    def chain_operator_idxs(self, val):
        if val is None:
            self._chain_operator_idxs  = get_chainbreak_idxs(self.z_matrix_idxs)
        else:
            self._chain_operator_idxs = val

    @property
    def chain_operators(self):
        """
        dict: A set of coordinate transformations that can orient multiple chains that are not covalently linked. e.g.
        structures with missing residues or protein complexes. The ``chain_operator`` property is a dictionary mapping
        chainIDs to a sub-dictionary containing a translation vector, ``ori``  and rotation matrix ``mx`` that will
        transform the protein coordinates from tie internal coordinate frame to some global frame.
        """
        if isinstance(self._chain_operators, dict):
            return self._chain_operators
        else:
            return self._chain_operators[self.trajectory.frame]

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
            logging.info("No protein chain origins have been provided. All chains will start at [0, 0, 0]")

            op = {idx: {"ori": np.array([0, 0, 0]), "mx": np.identity(3)}for idx in self._chain_operator_idxs}
            self.has_chain_operators = False
        else:
            self.has_chain_operators = True
        self._chain_operators[self.trajectory.frame] = op
        self.perturbed = True

    @property
    def coords(self):
        """np.ndarray : The cartesian coordinates of the protein"""
        if (self._coords is None) or self.perturbed:
            self._coords[self.trajectory.frame] = self.to_cartesian()
            self.perturbed = False
        return self._coords[self.trajectory.frame]

    @coords.setter
    def coords(self, val):
        self.coords[self.trajectory.frame] = val
        z_matrix, chain_operator = get_z_matrix(val, self.z_matrix_idxs, self.chain_operator_idxs)

        self.trajectory.coordinates_array[self.trajectory.frame] = z_matrix
        self.chain_operators[self.trajectory.frame] = chain_operator

    @property
    def atoms(self):
        return self.protein.atoms

    @property
    def nonbonded(self):
        """np.ndarray: Array of atom index pairs of atoms that are not bonded"""
        if self._nonbonded is None and not (self.bonds is None or self.bonds.any() is None):
            bonded_pairs = {(a, b) for a, b in self.bonds}
            possible_bonds = itertools.combinations(range(len(self.atoms)), 2)
            self._nonbonded = np.fromiter(
                itertools.chain.from_iterable(
                    nb for nb in possible_bonds if nb not in bonded_pairs), dtype=int)

            self._nonbonded.shape = (-1, 2)

        return self._nonbonded




def reconfigure_cap(cap, atom_idxs, bonds):
    for idx in cap:
        if idx in atom_idxs:
            atom_idxs.remove(idx)

    # Get bonds to all cap atoms
    sub_bonds = [tuple(bond) for bond in bonds if np.any(np.isin(bond, cap))]

    # Identify bound atoms outside the cap and use the first atom as root
    root = min([bnd[0] for bnd in sub_bonds if bnd[0] not in cap] +
               [bnd[1] for bnd in sub_bonds if bnd[1] not in cap])

    G = ig.Graph(edges=bonds)
    nodes, _, parents = G.bfs(root)
    edges = [(parents[node], node) for node in nodes if node != root]

    for edge in edges:
        mask = np.all(bonds == edge[::-1], axis=1)
        if np.any(mask):
            bndidx = np.argwhere(mask).flat[0]
            bonds[bndidx] = edge

    cap_idxs = [edge[1] for edge in edges]
    atom_idxs += cap_idxs

    return atom_idxs


def zmatrix_idxs_to_local(zmatrix_idxs):
    idxmap = {d[-1]: i for i, d in enumerate(zmatrix_idxs)}
    new_zmatrix_idxs = []
    for d in zmatrix_idxs:
        d = [idxmap[di] for di in d]
        if (dl := len(d)) < 4:
            d += [np.nan for i in range(4 - dl)]
        new_zmatrix_idxs.append(d)

    return np.array(new_zmatrix_idxs).astype(int)

def get_chainbreak_idxs(z_matrix_idxs, nan_int=-2147483648):
    """
    Get indices of atoms that  chain breaks
    Parameters
    ----------
    z_matrix_idxs: ArrayLike
        index map of z_matrix
    nan_int : int
        The value of ``np.nan`` when viewed as an integer. `` np.array(np.nan).astype(int)``

    Returns
    -------
    chainbreak_idxs:
        indices of atoms that start new chains
    """
    chainbreak_idxs = []
    for idxs in z_matrix_idxs:
        if np.sum(idxs == nan_int) == 3:
            chainbreak_idxs.append(idxs[0])

    return chainbreak_idxs


def ic_mx(*p: ArrayLike) -> Tuple[ArrayLike, ArrayLike]:
    """
    Calculates a rotation matrix and translation to transform a set of atoms to global coordinate frame from a local
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
        New origin position in 3-dimensional space
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


def get_z_matrix(coords, z_matrix_idxs, chain_operator_idxs=None, nan_int=-2147483648):
    z_matrix = np.zeros((len(z_matrix_idxs), 3))
    bond_mask = ~(z_matrix_idxs[:, 1] == nan_int)
    bond_values = coords[z_matrix_idxs[bond_mask, 1]] - coords[z_matrix_idxs[bond_mask, 0]]
    bond_values = np.linalg.norm(bond_values, axis=1)

    angle_mask = ~(z_matrix_idxs[:, 2] == nan_int)
    angle_values = [coords[z_matrix_idxs[angle_mask, i]] for i in range(3)]
    angle_values = get_angles(*angle_values)

    dihedral_mask = ~(z_matrix_idxs[:, 3] == nan_int)
    dihedral_values = [coords[z_matrix_idxs[dihedral_mask, i]] for i in range(4)]
    dihedral_values = get_dihedrals(*dihedral_values)

    z_matrix[bond_mask, 0] = bond_values
    z_matrix[angle_mask, 1] = angle_values
    z_matrix[dihedral_mask, 2] = dihedral_values
    if chain_operator_idxs is None:
        return z_matrix

    chain_operator = {}
    for cidx in chain_operator_idxs:
        chain_operator_def = z_matrix_idxs[cidx + 2]
        if chain_operator_def[-1] == nan_int and chain_operator_def[-2] != nan_int:
            pos = coords[chain_operator_def[:3]]
            mx, ori = ic_mx(*pos)
        else:
            mx, ori = np.eye(3), coords[cidx].copy()
        chain_operator[cidx] = {'mx': mx, 'ori': ori}

    return z_matrix, chain_operator
