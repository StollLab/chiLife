import itertools
import logging
from dataclasses import dataclass
from typing import List, Union, Dict, Tuple

from memoization import cached
import MDAnalysis
import networkx as nx
import numpy as np
from numpy.typing import ArrayLike

from .protein_utils import dihedral_defs, save_pdb, local_mx, global_mx, get_dihedral, get_angle, guess_bonds
from .numba_utils import _ic_to_cart

# TODO: Align Atom and ICAtom names with MDA
# TODO: Implement Internal Coord Residue object

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

    def __init__(self, zmats: Dict, zmat_idxs: Dict, atom_dict: Dict, ICs: Dict, **kwargs: Dict):
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
        self.atoms = np.asarray(kwargs['atoms']) if 'atoms' in kwargs else np.array([ic for chain in self.ICs.values()
                                                                                     for resi in chain.values()
                                                                                  for ic in resi.values()])

        self.atom_types = np.asarray(kwargs['atom_types']) if 'atom_types' in kwargs else \
            np.array([atom.atype for atom in self.atoms])
        self.atom_names = np.asarray(kwargs['atom_names']) if 'atom_names' in kwargs else \
            np.array([atom.name for atom in self.atoms])

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
        self._dihedral_defs = None

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
                  'atoms': self.atoms.copy(),
                  'atom_types': self.atom_types,
                  'atom_names': self.atom_names,
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
            self.has_chain_operators = False
        else:
            self.has_chain_operators = True
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
        zmc = self.zmats[chain]
        for i, (dihedral, atoms) in enumerate(zip(dihedrals, atom_list)):
            idxs, stem, idx = self.get_zmat_idxs(resi, atoms, chain)
            aidx = self.atom_dict['dihedrals'][chain][resi, stem][atoms[-1]]
            delta = self.zmats[chain][aidx, 2] - dihedral
            zmc[aidx, 2] = dihedral

            if idxs:
                zmc[idxs, 2] -= delta

        return self

    @cached
    def get_zmat_idxs(self, resi, atoms, chain):
        stem, stemr = tuple(atoms[2::-1]), tuple(atoms[1:])
        atom_idx = -1
        if (resi, stem) in self.atom_dict['dihedrals'][chain]:
            aidx = self.atom_dict['dihedrals'][chain][resi, stem][atoms[-1]]
            idxs = []
            for key, idx in self.atom_dict['dihedrals'][chain][resi, stem].items():
                if idx != aidx:
                    idxs.append(idx)

                restem = (stem[1], stem[0], key)
                if (resi, restem) in self.atom_dict['dihedrals'][chain]:
                    idxs += [idx for idx in self.atom_dict['dihedrals'][chain][resi, restem].values()]

        elif (resi, stemr) in self.atom_dict['dihedrals'][chain]:
            stem = stemr
            atom_idx = 0
            aidx = self.atom_dict['dihedrals'][chain][resi, stemr][atoms[0]]

            idxs = []
            for key, idx in self.atom_dict['dihedrals'][chain][resi, stemr].items():
                if idx != aidx:
                    idxs.append(idx)

                restem = (stemr[1], stemr[0], key)
                if (resi, restem) in self.atom_dict['dihedrals'][chain]:
                    idxs += [idx for idx in self.atom_dict['dihedrals'][chain][resi, restem].values()]
        else:
            raise ValueError(
                f"Dihedral with atoms {atoms} not found in chain {chain} on resi {resi} internal coordinates:\n"
                + "\n".join([str(ic) for ic in self.ICs[chain][resi]])
            )

        return idxs, stem, atom_idx

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
            if self.has_chain_operators:
                cart_coords = cart_coords @ self.chain_operators[segid]["mx"] + self.chain_operators[segid]["ori"]

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

    @property
    def dihedral_defs(self):
        if self._dihedral_defs is None:
            self._dihedral_defs = self.collect_dih_list()
        return self._dihedral_defs


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


def get_ICAtom(
    mol: MDAnalysis.core.groups.Atom,
    dihedral: List[int],
    ixmap: Dict = None
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
    atom = mol.universe.atoms[dihedral[-1]]
    if len(dihedral) == 1:
        return ICAtom(atom.name, atom.type,  ixmap[dihedral[-1]], atom.resname, atom.resid, (atom.name,))

    elif len(dihedral) == 2:
        atom2 = mol.universe.atoms[dihedral[-2]]
        atom_names = (atom.name, atom2.name)
        return ICAtom(
            atom.name,
            atom.type,
            ixmap[dihedral[-1]],
            atom.resname,
            atom.resid,
            atom_names,
            ixmap[dihedral[-2]],
            np.linalg.norm(atom.position - atom2.position),
        )

    elif len(dihedral) == 3:
        atom2 = mol.universe.atoms[dihedral[-2]]
        atom3 = mol.universe.atoms[dihedral[-3]]
        atom_names = (atom.name, atom2.name, atom3.name)

        return ICAtom(
            atom.name,
            atom.type,
            ixmap[dihedral[-1]],
            atom.resname,
            atom.resid,
            atom_names,
            ixmap[dihedral[-2]],
            np.linalg.norm(atom.position - atom2.position),
            ixmap[dihedral[-3]],
            get_angle(mol.universe.atoms[dihedral[-3:]].positions),
        )

    else:
        atom4 = mol.universe.atoms[dihedral[0]]
        atom3 = mol.universe.atoms[dihedral[1]]
        atom2 = mol.universe.atoms[dihedral[2]]
        atom_names = (atom.name, atom2.name, atom3.name, atom4.name)

        if atom_names == ("N", "C", "CA", "N"):
            dihedral_resi = atom.resnum - 1
        else:
            dihedral_resi = atom.resnum

        p1, p2, p3, p4 = mol.universe.atoms[dihedral].positions

        bl = np.linalg.norm(p3 - p4)
        al = get_angle([p2, p3, p4])
        dl = get_dihedral([p1, p2, p3, p4])

        return ICAtom(
            atom.name,
            atom.type,
            ixmap[dihedral[-1]],
            atom.resname,
            atom.resnum,
            atom_names,
            ixmap[dihedral[-2]],
            bl,
            ixmap[dihedral[-3]] ,
            al,
            ixmap[dihedral[-4]],
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
    **kwargs: Dict
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
    bonds = bonds.copy() if bonds is not None else guess_bonds(mol.atoms.positions, mol.atoms.types) + mol.atoms[0].ix
    atom_idxs = mol.atoms.ix.tolist()

    # Remove cap atoms for atom_idxs
    cap = kwargs.get('cap', [])
    for idx in cap:
        if idx in atom_idxs:
            atom_idxs.remove(idx)

    # Get directional topoly of cap and .
    if cap:
        # Get bonds to all cap atoms
        sub_bonds = [tuple(bond) for bond in bonds if np.any(np.isin(bond, cap))]

        # Identify bound atoms outside the cap and use the first atom as root
        root = min([bnd[0] for bnd in sub_bonds if bnd[0] not in cap] +
                   [bnd[1] for bnd in sub_bonds if bnd[1] not in cap])


        G = nx.Graph()
        G.add_edges_from(sub_bonds)
        edges = [edge for edge in nx.bfs_edges(G, root)]

        for edge in edges:
            mask = np.all(bonds == edge[::-1], axis=1)
            if np.any(mask):
                bndidx = np.argwhere(mask).flat[0]
                bonds[bndidx] = edge

        cap_idxs =  [edge[1] for edge in edges]
        atom_idxs += cap_idxs

    G = nx.DiGraph()
    G.add_nodes_from(atom_idxs)
    G.add_edges_from(bonds)
    dihedrals = [get_n_pred(G, node, np.minimum(node, 3)) for node in atom_idxs]

    if preferred_dihedrals is not None:
        present = False
        for dihe in preferred_dihedrals:

            # Get the index of the atom being defined by the preferred dihedral
            idx_of_interest = np.argwhere(mol.atoms.names == dihe[-1]).flatten()
            u_idx_of_interest = mol[idx_of_interest].ix
            idx_of_interest = np.argwhere(np.isin(atom_idxs, u_idx_of_interest)).flatten()
            for idx, uidx in zip(idx_of_interest, u_idx_of_interest):
                # Check if it is already in use
                if np.all(mol.universe.atoms[dihedrals[idx]].names == dihe):
                    present = True
                    continue

                # Check for alternative dihedral definitions that satisfy the preferred dihedral
                for p in get_all_n_pred(G, uidx, 3):
                    if np.all(mol.universe.atoms[p[::-1]].names == dihe):
                        dihedral = p[::-1]
                        break
                else:
                    dihedral = [uidx]

                # If an alternative is found, replace it in the dihedral list.
                if len(dihedral) == 4:
                    present = True
                    didx = np.argwhere(atom_idxs == dihedral[-1]).flatten()[0]
                    dihedrals[didx] = dihedral

        if not present and preferred_dihedrals != []:
            raise ValueError(f'There is no dihedral `{dihe}` in the provided protien. Perhaps there is typo or the '
                             f'atoms are not sorted correctly')

    idxstop = [i for i, sub in enumerate(dihedrals) if len(sub) == 1]
    dihedral_segs = [dihedrals[s:e] for s, e in zip(idxstop, (idxstop + [None])[1:])]

    all_ICAtoms = {}
    chain_operators = {}
    segid = 0
    ixmapped_bonds = []
    for seg in dihedral_segs:

        ixmap = {ix[-1]: i for i, ix in enumerate(seg)}
        segid += 1
        mx, ori = ic_mx(*mol.universe.atoms[seg[2]].positions)

        chain_operators[segid] = {"ori": ori, "mx": mx}
        #
        ICatoms = [get_ICAtom(mol, dihedral, ixmap=ixmap) for dihedral in seg]
        all_ICAtoms[segid] = get_ICResidues(ICatoms)
        ixmapped_bonds += [(ixmap[a], ixmap[b]) for a, b in bonds if a in ixmap and b in ixmap]
    # Get bonded pairs within selection
    bonded_pairs = ixmapped_bonds

    return ProteinIC.from_ICatoms(
        all_ICAtoms, chain_operators=chain_operators, bonded_pairs=bonded_pairs
    )


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
