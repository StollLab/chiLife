import itertools
import logging
import warnings
from collections import defaultdict
from typing import List, Union, Dict, Tuple

import MDAnalysis
import igraph as ig
import numpy as np
from numpy.typing import ArrayLike

from .MolSys import MolecularSystemBase, Trajectory, MolSys
from .Topology import Topology, guess_bonds
from .protein_utils import get_angles, get_dihedrals
from .numba_utils import _ic_to_cart, batch_ic2cart
from .globals import dihedral_defs


class MolSysIC:
    """
    A class for protein internal coordinates.
    
    Parameters
    ----------
    z_matrix : np.ndarray
        Array of z-matricies for all frames of the ensemble/trajectory
    z_matrix_idxs : np.ndarray
        Indices of the attoms that define the bond lengths, angles, and dihedrals of the z-matrix
    protein :
    kwargs

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
                 protein: Union[MolecularSystemBase, MDAnalysis.Universe, MDAnalysis.AtomGroup],
                 **kwargs):
        # Internal coords and atom info
        if isinstance(protein, MolSys):
            self.protein = protein.copy()
        else:
            self.protein = MolSys.from_atomsel(protein)

        self.atoms = self.protein.atoms
        self.atom_names = self.protein.names
        self.atom_chains = self.protein.segids
        self.atom_resnums = self.protein.resnums
        self.atom_types = self.protein.types
        self.atom_resnames = self.protein.resnames
        uresid, uresidx = np.unique(self.protein.resixs, return_index=True)
        self.resnums = self.atom_resnums[uresidx]
        self.resnames = self.atom_resnames[uresidx]
        self.trajectory = Trajectory(z_matrix, self)
        self.z_matrix_idxs = z_matrix_idxs
        self.chain_operator_idxs = kwargs.get('chain_operator_idxs', None)
        self.has_chain_operators = bool(kwargs.get('chain_operators', None))

        self.chains = np.unique(self.atom_chains)
        self._chain_segs = [[a, b] for a, b in zip(self.chain_operator_idxs,
                                                   self.chain_operator_idxs[1:] + [len(self.z_matrix_idxs)])]

        if not self.has_chain_operators:
            self.chain_operators = None
        else:
            self._chain_operators = kwargs['chain_operators']


        # Topology
        self.bonds = kwargs['bonds'] if 'bonds' in kwargs else \
            guess_bonds(self.protein.positions, self.protein.types)
        self._nonbonded = kwargs.get('nonbonded', None)
        self.topology = kwargs['topology'] if 'topology' in kwargs else \
            Topology(self.protein, self.bonds)

        self.non_nan_idxs = kwargs.get('non_nan_idxs', None)
        if self.non_nan_idxs is None:
            self.non_nan_idxs = np.argwhere(~np.any(self.z_matrix_idxs < 0, axis=1)).flatten()

        self.nan_idxs = np.argwhere(np.any(self.z_matrix_idxs < 0, axis=1)).flatten()

        self.chain_res_name_map = kwargs.get('chain_res_name_map', defaultdict(list))
        if self.chain_res_name_map == {}:
            idxs, b2s, b1s,  _ = self.z_matrix_idxs[self.non_nan_idxs].T
            chains = self.atoms[b2s].segids
            resnums = self.atoms[b2s].resnums
            [self.chain_res_name_map[(chain, res, b1, b2)].append(idx)
             for chain, res, b1, b2, idx in
             zip(chains, resnums, self.atom_names[b1s], self.atom_names[b2s], idxs)]

            self.chain_res_name_map = {k: v for k, v in self.chain_res_name_map.items()}

        # Misc
        self.perturbed = False
        self._dihedral_defs = None

    @classmethod
    def from_atoms(cls,
                   atoms: Union[MDAnalysis.Universe, MDAnalysis.AtomGroup, MolecularSystemBase],
                   preferred_dihedrals: List = None,
                   bonds: ArrayLike = None,
                   **kwargs: Dict):
        """
        Generate a MolSysIC object from a :class:`~chilife.AtomSelection` or `MDAnalysis.AtomGroup` object.

        Parameters
        ----------
        atoms : MDAnalysis.Universe, MDAnalysis.AtomGroup, MolecularSystemBase
            MDA Universe, AtomGroup or chiLife Molecular System to create the internal coordinates from.
        preferred_dihedrals : List[List[str]]
            List of preferred dihedral definitions to use when defining the internal coordinate system.
        bonds : ArrayLike
            Array of tuples defining the atom pairs that are bonded.
        kwargs : dict
            Additional keyword arguments.

            ignore_water : bool
                Ignore atoms that belong to water molecules.
            use_chain_operators : bool
                Allow for the use of a translation and rotation vectors to orient cains that are not covalently linked.

        Returns
        -------
        cls : MDAnalysis.Universe
            A chiLife internal coordinates molecular system.
        """

        if isinstance(atoms, MDAnalysis.Universe):
            atoms = atoms.atoms

        if kwargs.get('ignore_water', True):
            water_atoms = atoms.select_atoms("byres (name OH2 or resname HOH)")
            atoms = atoms[~np.isin(atoms.ix, water_atoms.ix)]

        # Keep track of atom indexes in parent protein object in case atoms come from a larger system
        atom_idxs = atoms.ix

        # Explicitly passing bonds means no other bonds will be used and that the bonds are using universe indices.
        if bonds is not None:
            bonds = bonds.copy()
            ibonds = bonds[np.all(np.isin(bonds, atom_idxs), axis=1)]
            ixmap = {ix: i for i, ix in enumerate(atom_idxs)}

            # renumber bonds to current selection
            bonds = np.vectorize(ixmap.get)(ibonds)
            bonds = np.sort(bonds, axis=1)

        # # Otherwise use bonds defined by the protein object
        # elif hasattr(protein, 'bonds'):
        #     bonds = protein.bonds
        # Guess the bonds as a last resort
        else:
            # Add selection's lowest index in case the user is selecting a subset of atoms from a larger system.
            bonds = guess_bonds(atoms.positions, atoms.types)

        # Remove bonds outside of selection
        topology = Topology(atoms, bonds)
        z_matrix_dihedrals = topology.get_zmatrix_dihedrals()
        # z_mat_map = {k[-1]: i for i, k in enumerate(z_matrix_dihedrals)}

        if preferred_dihedrals is not None:
            present = False
            for dihe in preferred_dihedrals:

                # Get the indices of the atom being defined by the preferred dihedral
                idx_of_interest = np.argwhere(atoms.names == dihe[-1]).flatten()
                for idx in idx_of_interest:

                    # Skip dihedral defs of chain terminal atoms
                    if len(z_matrix_dihedrals[idx]) != len(dihe):
                        continue

                    # Check if it is already in use
                    if np.all(atoms[z_matrix_dihedrals[idx]].names == dihe):
                        present = True
                        continue

                    if idx not in topology.dihedrals_by_atoms:
                        dihedral = [idx]
                        continue

                    # Check for alternative dihedral definitions that satisfy the preferred dihedral
                    for p in topology.dihedrals_by_atoms[idx]:
                        if np.all(atoms[list(p)].names == dihe):
                            dihedral = [a for a in p]
                            break
                    else:
                        dihedral = [idx]

                    # If an alternative is found, replace it in the dihedral list.
                    if len(dihedral) == 4:
                        present = True
                        z_matrix_dihedrals[idx] = dihedral

                        # also change any other dependent dihedrals
                        for dihe_idxs in topology.dihedrals_by_bonds[tuple(dihedral[1:3])]:
                            zmidx = dihe_idxs[-1]
                            tmp = z_matrix_dihedrals[zmidx]
                            if all(a == b for a, b in zip(tmp[1:3], dihedral[1:3])):
                                z_matrix_dihedrals[zmidx][0] = dihedral[0]

            if not present and preferred_dihedrals != []:
                raise ValueError(f'There is no dihedral `{dihe}` in the provided protein. Perhaps there is typo or the '
                                 f'atoms are not sorted correctly')

        z_matrix_coordinates = np.zeros((len(atoms.universe.trajectory), len(z_matrix_dihedrals), 3))
        z_matrix_dihedrals = zmatrix_idxs_to_local(z_matrix_dihedrals)

        chain_operator_idxs = get_chainbreak_idxs(z_matrix_dihedrals)

        chain_operators = []
        for i, ts in enumerate(atoms.universe.trajectory):
            z_matrix, chain_operator = get_z_matrix(atoms.positions, z_matrix_dihedrals, chain_operator_idxs)
            z_matrix_coordinates[i] = z_matrix
            chain_operators.append(chain_operator)

        if not kwargs.get('use_chain_operators', True):
            chain_operators = None

        return cls(z_matrix_coordinates, z_matrix_dihedrals, atoms,
                   chain_operators=chain_operators, chain_operator_idxs=chain_operator_idxs)

    def copy(self):
        """Create a deep copy of an MolSysIC instance"""
        z_matrix = self.trajectory.coordinate_array.copy()
        z_matrix_idxs = self.z_matrix_idxs.copy()
        if isinstance(self._chain_operators, list):
            chain_operators = [{k: {k2: v2.copy() for k2, v2 in v.items()} for k, v in co.items()} for co in self._chain_operators]
        else:
            chain_operators = {k: {k2: v2.copy() for k2, v2 in v.items()} for k, v in self._chain_operators.items()}

        kwargs = {'chain_operators': chain_operators,
                  'chain_operator_idxs': self._chain_operator_idxs,
                  'bonds': self.bonds,
                  'nonbonded': self._nonbonded,
                  'topology': self.topology,
                  'protein': self.protein,
                  'non_nan_idxs': self.non_nan_idxs,
                  'chain_res_name_map': self.chain_res_name_map}

        return MolSysIC(z_matrix, z_matrix_idxs, **kwargs)

    @property
    def chain_operator_idxs(self):
        """An array of indices defining the starting and ending indices of the different chains."""
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
            Dictionary containing an entry for each chain in the MolSysIC molecule. Each entry must contain a
            rotation matrix, 'mx' and translation vector 'ori'.
        """

        error_message = 'chain operators must be set to \n' \
                        '    1) a dict to set the chain operators for a single frame \n' \
                        '    2) a list of dicts the same length of the trajectory to set chain operators' \
                        'for each frame\n' \
                        '    3) a list with one dict to set all all frames to the same chain operator.'

        if op is None:
            if hasattr(self, '_chain_operators'):
                from_list = True if isinstance(self._chain_operators, list) else False
            else:
                from_list=False

            logging.info("No protein chain origins have been provided. All chains will start at [0, 0, 0]")

            op = {idx: {"ori": np.array([0, 0, 0]), "mx": np.identity(3)}for idx in self._chain_operator_idxs}
            self.has_chain_operators = False
            self._chain_operators = op
            self.apply_chain_operators(from_list=from_list)
        else:
            self.has_chain_operators = True
            if isinstance(op, dict):
                self._chain_operators[self.trajectory.frame] = op
                self.apply_chain_operators(self.trajectory.frame)
            elif not isinstance(op, list):
                raise RuntimeError(error_message)
            elif len(op) == 1:
                self._chain_operators = op[0]
                self.apply_chain_operators()
            elif len(op) == len(self.trajectory):
                self._chain_operators = op
                self.apply_chain_operators()
            else:
                raise RuntimeError(error_message)

    @property
    def z_matrix(self):
        """The z-matrix (internal) coordinates of the molecular system for the current frame."""
        return self.trajectory.coordinate_array[self.trajectory.frame]

    @property
    def coords(self):
        """The cartesian coordinates of the molecular system for the current frame"""
        self.protein.trajectory[self.trajectory.frame]
        if (self.protein.positions is None) or self.perturbed:
            self.protein.positions = self.to_cartesian()
            self.perturbed = False

        return self.protein.positions

    @coords.setter
    def coords(self, val):
        self.protein.trajectory[self.trajectory.frame]
        self.protein.positions = val
        z_matrix, chain_operator = get_z_matrix(val, self.z_matrix_idxs, self.chain_operator_idxs)

        self.trajectory.coordinate_array[self.trajectory.frame] = z_matrix
        self.chain_operators[self.trajectory.frame] = chain_operator

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

    def to_cartesian(self):
        """Method to convert z-matrix (internal) coordinate to cartesian coordinate"""
        cart_coords = _ic_to_cart(self.z_matrix_idxs[:, 1:], self.z_matrix)

        # Apply chain operations if any exist
        if self.has_chain_operators:
            for start, end in self._chain_segs:
                op = self.chain_operators[start]
                cart_coords[start:end] = cart_coords[start:end] @ op["mx"] + op["ori"]

        return cart_coords

    def set_dihedral(
            self,
            dihedrals: Union[float, ArrayLike],
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
        self : MolSysIC
            ProteinIC object with new dihedral angle(s)
        """

        self.perturbed = True

        chain = self._check_chain(chain)

        dihedrals = np.atleast_1d(dihedrals)
        atom_list = np.atleast_2d(atom_list)
        for i, (dihedral, atoms) in enumerate(zip(dihedrals, atom_list)):
            if (tag := (chain, resi, atoms[1], atoms[2])) not in self.chain_res_name_map:
                raise RuntimeError(f'{atoms} is not a recognized dihedrals of residue {resi}. Please make sure you '
                                   f'have the correct residue number and atom names.')

            pert_idxs = self.chain_res_name_map[tag]  + self.chain_res_name_map.get((chain, resi, atoms[2], atoms[1]), [])
            for idx in pert_idxs:
                protein_atom_names = self.z_matrix_names[idx][::-1]
                if tuple(protein_atom_names) == tuple(atoms):
                    delta = self.z_matrix[idx, 2] - dihedral
                    break
            else:
                raise RuntimeError('')

            self.trajectory.coords[self.trajectory.frame, pert_idxs, 2] -= delta

        return self

    def batch_set_dihedrals(
            self,
            idxs: ArrayLike,
            dihedrals: ArrayLike,
            resi: int,
            atom_list: ArrayLike,
            chain: Union[int, str] = None
    ):
        """Sets dihedral angles of a single residue in internal coordinates for the atoms defined in atom
        list and in trajectory indices in idxs.

        Parameters
        ----------
        idxs: ArrayLike
            an array in indices corresponding to the progenitor structure in the MolSysIC trajectory.
        dihedrals : ArrayLike
            an array of angles for each idx in ``idxs`` and for each dihedral in ``atom_list``. Should have the shape
            ``(len(idxs), len(atom_list))``
        resi : int
            Residue number of the site being altered
        atom_list : ArrayLike
            Names or array of names of atoms involved in the dihedrals
        chain : int, str, optional
            Chain containing the atoms that are defined by the dihedrals that are being set. Only necessary when there
            is more than one chain in the proteinIC object.

        Returns
        -------
        z_matrix : ArrayLike
            the z_matrix for every frame in ``idxs`` with the new dihedral angel values defined in ``dihedrals``
        """
        chain = self._check_chain(chain)

        dihedrals = np.atleast_2d(dihedrals).T
        atom_list = np.atleast_2d(atom_list)
        z_matrix = self.trajectory.coordinate_array[idxs]

        for i, (values, atoms) in enumerate(zip(dihedrals, atom_list)):
            if (tag := (chain, resi, atoms[1], atoms[2])) not in self.chain_res_name_map:
                raise RuntimeError(f'{atoms} is not a recognized dihedrals of residue {resi}. Please make sure you '
                                   f'have the correct residue number and atom names.')

            pert_idxs = self.chain_res_name_map[tag] + self.chain_res_name_map.get((chain, resi, atoms[2], atoms[1]), [])
            for idx in pert_idxs:
                protein_atom_names = self.z_matrix_names[idx][::-1]
                if tuple(protein_atom_names) == tuple(atoms):
                    deltas = z_matrix[:, idx, 2] - values
                    break
            else:
                raise RuntimeError('')

            z_matrix[:, pert_idxs, 2] -= deltas[:, None]

        return z_matrix

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
        if len(atom_list) == 0:
            return np.array([])

        chain = self._check_chain(chain)

        atom_list = np.atleast_2d(atom_list)
        dihedral_idxs = []
        for atoms in atom_list:
            if (tag := (chain, resi, *atoms)) not in self.topology.dihedrals_by_resnum:
                raise RuntimeError(f'{atoms} is not a recognized dihedral of chain {chain} and residue {resi}. Please '
                                   f'make sure you have the correct residue number and atom names. Note that chilife '
                                   f'considers dihedrals as directional so you may want to try the reverse dihedral.')

            dihedral_idxs.append(list(self.topology.dihedrals_by_resnum[tag]))

        dihedral_idxs = np.array(dihedral_idxs).T
        dihedral_values = self.coords[dihedral_idxs]
        dihedrals = get_dihedrals(*dihedral_values)
        return dihedrals[0] if len(dihedrals) == 1 else dihedrals

    def get_z_matrix_idxs(self, resi: int, atom_list: ArrayLike, chain: Union[int, str] = None):
        """Get the z-matrix indices of the dihedral angle(s) defined by ``atom_list`` and the specified residue and
        chain. Dihedral angles are returned in radians.

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
        dihedral_idxs: numpy.ndarray
            Array of dihedral angles corresponding to the atom sets in atom list.
        """

        if len(atom_list) == 0:
            return np.array([])

        chain = self._check_chain(chain)

        atom_list = np.atleast_2d(atom_list)
        idxs = []
        for atoms in atom_list:
            if (tag := (chain, resi, atoms[1], atoms[2])) not in self.chain_res_name_map:
                raise RuntimeError(f'{atoms} is not a recognized dihedral of chain {chain} and residue {resi}. Please '
                                   f'make sure you have the correct residue number and atom names. Note that chilife '
                                   f'considers dihedrals as directional so you may want to try the reverse dihedral.')

            pert_idxs = self.chain_res_name_map[tag]
            for idx in pert_idxs:
                protein_atom_names = self.z_matrix_names[idx][::-1]
                if tuple(protein_atom_names) == tuple(atoms):
                    idxs.append(idx)
                    break
            else:
                raise RuntimeError('')

        return idxs[0] if len(idxs) == 1 else np.array(idxs)

    def has_clashes(self, distance: float = 1.5) -> bool:
        """Checks for an internal clash between non-bonded atoms.

        Parameters
        ----------
        distance : float
            Minimum distance allowed between non-bonded atoms in angstroms. (Default value = 1.5).

        Returns
        -------
        has_clashes : bool
            True if there are any atoms within ``distance`` angstroms from each other that would constitute a clash
            otherwise False
        """
        diff = self.coords[self.nonbonded[:, 0]] - self.coords[self.nonbonded[:, 1]]
        dist = np.linalg.norm(diff, axis=1)
        has_clashes = np.any(dist < distance)
        return has_clashes

    def _check_chain(self, chain):
        if chain is None and len(self.chains) == 1:
            chain = self.chains[0]

        elif chain is None and len(self.chains) > 1:
            raise ValueError("You must specify the protein chain")

        return chain

    def phi_idxs(self, resnums: ArrayLike = None, chain: Union[int, str] = None):
        """
        Method to return the Z-matrix indices of Phi backbone dihedrals

        Parameters
        ----------
        resnums: int, ArrayLike[int]
            Residues for which to return the Phi value index of the z-matrix
        chain: int, str
            Chain corresponding to the resnums of interest

        Returns
        -------
        idxs: ndarray[int]
            Array of indices corresponding to the phi dihedral angles of the selected residues on the chain
        """
        chain = self._check_chain(chain)
        mask = (self.atom_names == 'C') * (self.atom_chains == chain)
        if resnums is not None:
            resnums = np.atleast_1d(resnums)
            mask *= np.isin(self.atom_resnums, resnums)

        idxs = np.argwhere(mask).flatten()
        idxs = np.array([idx for idx in idxs if idx not in self.nan_idxs])
        return idxs

    def psi_idxs(self, resnums: ArrayLike = None, chain: Union[int, str] = None):
        """
        Method to return the Z-matrix indices of Psi backbone dihedrals

        Parameters
        ----------
        resnums: int, ArrayLike[int]
            Residues for which to return the Psi value index of the z-matrix
        chain: int, str
            Chain corresponding to the resnums of interest

        Returns
        -------
        idxs: ndarray[int]
            Array of indices corresponding to the Psi dihedral angles of the selected residues on the chain
        """
        chain = self._check_chain(chain)
        mask = (self.atom_names == 'N') * (self.atom_chains == chain)
        if resnums is not None:
            resnums = np.atleast_1d(resnums) + 1
            mask *= np.isin(self.atom_resnums, resnums)

        idxs = np.argwhere(mask).flatten()
        idxs = np.array([idx for idx in idxs if idx not in self.nan_idxs])

        return idxs

    def omega_idxs(self, resnums: ArrayLike = None, chain: Union[int, str] = None):
        """
        Method to return the Z-matrix indices of Psi backbone dihedrals

        Parameters
        ----------
        resnums: int, ArrayLike[int]
            Residues for which to return the Psi value index of the z-matrix
        chain: int, str
            Chain corresponding to the resnums of interest

        Returns
        -------
        idxs: ndarray[int]
            Array of indices corresponding to the Psi dihedral angles of the selected residues on the chain
        """

        chain = self._check_chain(chain)
        mask = (self.atom_names == 'CA') * (self.atom_chains == chain)
        if resnums is not None:
            resnums = np.atleast_1d(resnums) + 1
            mask *= np.isin(self.atom_resnums, resnums)

        idxs = np.argwhere(mask).flatten()
        idxs = np.array([idx for idx in idxs if idx not in self.nan_idxs])

        return idxs

    def chi_idxs(self, resnums: ArrayLike = None, chain: Union[int, str] = None):
        """
         Create a list of index arrays corresponding the Z-matrix indices of flexible side chain (chi) dihedrals. Note
         that only chi dihedrals defined in ``chilife.dihedral_defs`` will be  considered.

        Parameters
        ----------
        resnums: int, ArrayLike[int]
            Residues for which to return the Psi value index of the z-matrix
        chain: int, str
            Chain corresponding to the resnums of interest

        Returns
        -------
        chi_idxs: List[np.ndarray]
            list of arrays containing the indices of the z-matrix dihedrals corresponding to the chi dihedrals. Each
            array in the list refers to a specific dihedral: ``[χ1, χ2, ... χn]``. because different residues will
            have different numbers of chi dihedrals the arrays will not be the same size, but they can be concatenated
            using numpy: ``chi_idxs = np.concatenate(IC.chi_idxs(...)))``.

        """
        chain = self._check_chain(chain)
        mask = self.atom_chains == chain
        mask *= ~np.isin(self.atom_names,['N', 'CA', 'C', 'O']) * (self.atom_types != 'H')

        if resnums is not None:
            resnums = np.atleast_1d(resnums)
            mask *= np.isin(self.atom_resnums, resnums)
        else:
            resnums = self.resnums

        chi_idxs = []
        for resnum in resnums:
            tmask = (self.atom_resnums == resnum)
            resname = self.atom_names[tmask][0]
            if resname in dihedral_defs:
                atom_names = [x[-1] for x in dihedral_defs[resname]]
                res_chi_idxs = np.argwhere(tmask * np.isin(self.atom_names, atom_names)).flatten()
            else:
                tmask *= mask
                res_chi_idxs = []
                taken = []
                for idx in np.argwhere(tmask).flat:
                    ddef_names = self.z_matrix_names[idx]
                    ddef_idxs = self.z_matrix_idxs[idx]
                    not_cycle = len(self.topology.graph.get_all_simple_paths(ddef_idxs[1], ddef_idxs[2], 8)) < 2
                    if (ddef_names[1] != 'CA') and (ddef_idxs[1] not in taken) and not_cycle:
                        res_chi_idxs.append(idx)
                        taken.append(ddef_idxs[1])

            chi_idxs.append(res_chi_idxs)

        return chi_idxs

    def __len__(self):
        return len(self.trajectory)

    def load_new(self, z_matrix, **kwargs):
        """
        Replace the current z-matrix ensemble/trajectory with new one.

        Parameters
        ----------
        z_matrix : np.ndarray
            New z-matrix to load into the MolSysIC
        kwargs : dict
            Additional keyword arguments.

            op : dict
                New chain operators for the new z-matrix.
        """
        self.trajectory.load_new(coordinates=z_matrix)
        cart_coords = batch_ic2cart(self.z_matrix_idxs[:, 1:], z_matrix)
        self.protein.trajectory.load_new(cart_coords)

        if 'op' in kwargs:
            op = kwargs['op']
            self.has_chain_operators = True

            if isinstance(op, list):
                # check if all chain operators are virtually the same
                OP0 = op[0]
                for iop in op:
                    scores = [np.abs(OP0[chain][key] - iop[chain][key]).max() for chain in iop for key in iop[chain]]
                    if any( x > 1e-3 for x in scores):
                        break
                else:
                    # if they are we only need the first copy of op
                    op = [op[0]]

                self.chain_operators = op

        elif isinstance(self._chain_operators, list):
            warnings.warn('You are loading in a new internal coordinate trajectory for an internal coordinates object '
                          'that has unique translations and rotations for each frame. These translations and rotations '
                          'will be lost, which can be particularly detrimental for systems with multiple chains. New '
                          'translations & rotations can be applied using the `co` keyword argument.')
            self.chain_operators = None
            self.has_chain_operators = False
        else:
            self.apply_chain_operators()

    def apply_chain_operators(self, idx=None, from_list=False):
        """
        Apply chain operators to the specified frames (``idx``) of the MolSysIC trajectory. If no ``idx`` is provided
        then all chain operators will be applied to all frames.
        Parameters
        ----------
        from_list : bool
            A boolean argument to specify whether each frame previously had its own chain operators.

        idx : int, Array
            Frames or array of frames that should have the chain operators applied.
        """

        idx = np.arange(len(self._chain_operators), dtype=int) if idx is None else idx
        idx = np.atleast_1d(idx)

        cart_coords = self.protein.trajectory.coordinate_array
        if isinstance(self._chain_operators, list):
            for i, op in zip(idx, self._chain_operators[idx]):
                for start, stop in self._chain_segs:
                    current_mx, current_ori = ic_mx(*cart_coords[i, start:start+3])
                    mx = op[start]['mx']
                    ori = op[start]['ori']
                    m2m3 = current_mx @ mx
                    cart_coords[i, start:stop] = (cart_coords[i, start:stop] - current_ori) @ m2m3 + ori

        elif from_list:
            for i in idx:
                for start, stop in self._chain_segs:
                    current_mx, current_ori = ic_mx(*cart_coords[i, start:start+3])
                    mx = self.chain_operators[start]['mx']
                    ori = self.chain_operators[start]['ori']
                    m2m3 = current_mx @ mx
                    cart_coords[i, start:stop] = (cart_coords[i, start:stop] - current_ori) @ m2m3 + ori

        elif isinstance(self._chain_operators, dict):
            for start, end in self._chain_segs:
                current_mx, current_ori = ic_mx(*cart_coords[0, start:start + 3])
                mx = self.chain_operators[start]['mx']
                ori = self.chain_operators[start]['ori']
                m2m3 = current_mx.T @ mx
                cart_coords[:, start:end] = np.einsum('ijk,kl->ijl', cart_coords[:, start:end] - current_ori, m2m3) + ori


    def use_frames(self, idxs):
        """
        Remove all frames except those specified by `idxs`
        Parameters
        ----------
        idxs : int, ArrayLike
            Index or array if indices to keep
        """
        self.trajectory.load_new(coordinates=self.trajectory.coordinate_array[idxs])
        self.protein.load_new(self.protein.trajectory.coordinate_array[idxs])

    def __iter__(self):
        for ts in self.trajectory:
            yield self

    def set_cartesian_coords(self, coords, mask):
        """
        Alter the coordinates of the current frame by assigning new cartwsian coordinates to the atoms set to True and
        ``mask``.

        Parameters
        ----------
        coords : np.ndarray
            Cartesian coordinates to apply to the atoms defined by ``mask`` of the current frame.
        mask : np.ndarray
            Array of bool values defining which atoms of the MolSysIC are being set.
        """
        nrots = len(coords)
        cpy = np.tile(self.protein.trajectory.coordinate_array[0].copy(), (nrots, 1, 1))

        cpy[:, mask] = coords

        z_mats = []
        ops = []
        for submat in cpy:
            z, op = get_z_matrix(submat, self.z_matrix_idxs, self.chain_operator_idxs)
            z_mats.append(z)
            ops.append(op)

        z_mats = np.array(z_mats)
        zcpy = np.tile(self.trajectory.coordinate_array[0].copy(), (nrots, 1, 1))
        zcpy[:, mask] = z_mats[:, mask]

        self.load_new(zcpy, op=ops)

    def shift_resnum(self, delta: int):
        """
        Alter the residue number of all residues in the MolSysIC by adding ``delta``.

        Parameters
        ----------
        delta : int
            Amount to shift residue numbers by. Can be positive or negative.
        """
        self.resnums += delta
        self.atom_resnums += delta

        self.chain_res_name_map = defaultdict(list)
        idxs, b2s, b1s, _ = self.z_matrix_idxs[self.non_nan_idxs].T
        chains = self.atoms[b2s].segids
        resnums = self.atoms[b2s].resnums
        [self.chain_res_name_map[(chain, res, b1, b2)].append(idx)
         for chain, res, b1, b2, idx in
         zip(chains, resnums, self.atom_names[b1s], self.atom_names[b2s], idxs)]

        self.chain_res_name_map = {k: v for k, v in self.chain_res_name_map.items()}

        self.topology.update_resnums()

    @property
    def z_matrix_names(self):
        if not hasattr(self, '_z_matrix_names'):
            self._z_matrix_names = np.array([self.atom_names[[x for x in y if x >= 0]].tolist()
                                             for y in self.z_matrix_idxs], dtype=object)

        return self._z_matrix_names

def reconfigure_cap(cap, atom_idxs, bonds):
    """
    Helper function to reconfigure the "cap" region of bifunctional labels so that the internal coordinates are defined
    correctly.

    Parameters
    ----------
    cap : ArrayLike
        Indices of the cap atoms in the molecule with atoms defined in atom_idxs.
    atom_idxs : ArrayLike
        Indices of all atoms to be considered for reconfiguration containing the cap.
    bonds : ArrayLike
        Array of tuples defining the bonds between atoms in atom_idxs.

    Returns
    -------
    atom_idxs : ArrayLike
        Array of atom indices rearranged so that the cap is terminal to the rest of the atoms.
    """
    atom_idxs = atom_idxs[~np.isin(atom_idxs, cap)]

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

    cap_idxs = [edge[1] for edge in edges if edge[1] not in atom_idxs]
    atom_idxs = np.concatenate((atom_idxs, cap_idxs))

    return atom_idxs


def zmatrix_idxs_to_local(zmatrix_idxs):
    """
    Convert Z-matrix indices so that they reference the indices of the z-matrix rather than the parental atom selection
    that was used to construct the z-matrix and may have an index offset or missing atoms.

    Parameters
    ----------
    zmatrix_idxs : ArrayLike
        z-matrix indices

    Returns
    -------
    zmatrix_idxs : ArrayLike
        Array of indices modified for internal reference.
    """
    idxmap = {d[-1]: i for i, d in enumerate(zmatrix_idxs)}
    new_zmatrix_idxs = []
    for d in zmatrix_idxs:
        d = [idxmap[di] for di in d]
        if (dl := len(d)) < 4:
            d = [np.nan for i in range(4 - dl)] + d
        new_zmatrix_idxs.append(d[::-1])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        new_z_matrix_idxs = np.array(new_zmatrix_idxs).astype(int)

    return new_z_matrix_idxs


def get_chainbreak_idxs(z_matrix_idxs):
    """
    Get indices of atoms that  chain breaks
    Parameters
    ----------
    z_matrix_idxs: ArrayLike
        index map of z_matrix
    Returns
    -------
    chainbreak_idxs:
        indices of atoms that start new chains
    """
    chainbreak_idxs = []
    for idxs in z_matrix_idxs:
        if np.sum(idxs < 0) == 3:
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


def get_z_matrix(coords, z_matrix_idxs, chain_operator_idxs=None):
    """
    Given a set of cartesian coordinates and indices defining a z-matrix, calcualte the z-matrix.

    Parameters
    ----------
    coords : np.ndarray
        Array of cartesian coordinates.
    z_matrix_idxs : np.ndarray
        Array of indices defining bond-angle-torsions of the z-matrix.
    chain_operator_idxs : np.ndarray
        Array of indices defining different chains.

    Returns
    -------
    z_matrix : np.ndarray
        Z-matrix (internal) coordinates.
    """
    z_matrix = np.zeros((len(z_matrix_idxs), 3))
    bond_mask = ~(z_matrix_idxs[:, 1] < 0)
    bond_values = coords[z_matrix_idxs[bond_mask, 1]] - coords[z_matrix_idxs[bond_mask, 0]]
    bond_values = np.linalg.norm(bond_values, axis=1)

    angle_mask = ~(z_matrix_idxs[:, 2] < 0)
    angle_values = [coords[z_matrix_idxs[angle_mask, i]] for i in range(3)]
    angle_values = get_angles(*angle_values)

    dihedral_mask = ~(z_matrix_idxs[:, 3] < 0)
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
        if chain_operator_def[-1] < 0 <= chain_operator_def[-2]:
            pos = coords[chain_operator_def[:3]][::-1]
            mx, ori = ic_mx(*pos)
        else:
            mx, ori = np.eye(3), coords[cidx].copy()
        chain_operator[cidx] = {'mx': mx, 'ori': ori}

    return z_matrix, chain_operator
