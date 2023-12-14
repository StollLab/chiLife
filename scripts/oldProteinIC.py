import numpy as np
from dataclasses import dataclass
from chilife.numba_utils import _ic_to_cart

class ProteinIC:

    def __init__(self, zmats, zmat_idxs, atom_dict, ICs, **kwargs):
        self.zmats = zmats
        self.zmat_idxs = zmat_idxs
        self.atom_chains = np.concatenate([[key] * len(zmats[key]) for key in zmats])
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
        self.has_chain_operators = True
        self.bonded_pairs = np.array(kwargs.get("bonded_pairs", None))
        self._nonbonded_pairs = kwargs.get('nonbonded_pairs', None)

        self.perturbed = False
        self._coords = kwargs['coords'] if 'coords' in kwargs else self.to_cartesian()
        self._dihedral_defs = None

    def copy(self):
        """Create a deep copy of an MolSysIC instance"""
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
    def coords(self):
        if (self._coords is None) or self.perturbed:
            self._coords = self.to_cartesian()
            self.perturbed = False
        return self._coords

    def get_dihedral(self, resi, atom_list, chain=None):

        if len(atom_list) == 0:
            return np.array([])

        chain = self._check_chain(chain)

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

    def get_dihedral_idx(self, resi, atom_list, chain=None):
        if len(atom_list) == 0:
            return np.array([])

        chain = self._check_chain(chain)

        atom_list = np.atleast_2d(atom_list)
        idxs = []
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
            idxs.append(aidx)

        return idxs[0] if len(idxs) == 1 else np.array(idxs)

    def has_clashes(self, distance=1.5) -> bool:

        diff = (
            self.coords[self.nonbonded_pairs[:, 0]]
            - self.coords[self.nonbonded_pairs[:, 1]]
        )
        dist = np.linalg.norm(diff, axis=1)
        has_clashes = np.any(dist < distance)
        return has_clashes

    def get_resi_dihs(self, resi):

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

    @property
    def dihedral_defs(self):
        if self._dihedral_defs is None:
            self._dihedral_defs = self.collect_dih_list()
        return self._dihedral_defs


    def shift_resnum(self, delta: int):
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

    def phi_idxs(self, resnums = None, chain = None):
        chain = self._check_chain(chain)
        idxs =  np.argwhere((self.atom_names == 'C') * (self.atom_chains == chain)).flatten()
        if resnums is not None:
            resnums = np.atleast_1d(resnums)
            idxs = [self.atoms[idx].index for idx in idxs if (self.atoms[idx].resi in resnums)]
        idxs = np.asarray(idxs)
        if len(idxs) > 0:
            idxs = idxs[~np.isnan(self.zmats[chain][idxs, 2])]
        return idxs

    def psi_idxs(self, resnums = None, chain = None):
        chain = self._check_chain(chain)
        idxs = np.argwhere((self.atom_names == 'N')).flatten()
        if resnums is not None:
            resnums = np.atleast_1d(resnums) + 1
            idxs = [self.atoms[idx].index for idx in idxs if (self.atoms[idx].resi in resnums)]
        idxs = np.asarray(idxs)

        if len(idxs) > 0:
            idxs = idxs[~np.isnan(self.zmats[chain][idxs, 2])]
        return idxs

    def bb_idxs(self,  resnums= None, chain = None):
        return np.concatenate((self.phi_idxs(resnums, chain), self.psi_idxs(resnums, chain)))

    def _check_chain(self, chain):
        if chain is None and len(self.ICs) == 1:
            chain = list(self.ICs.keys())[0]

        elif chain is None and len(self.ICs) > 1:
            raise ValueError("You must specify the protein chain")

        return chain


    #===============================================================================================================
    #                              KNOWN NECESSARY
    #===============================================================================================================

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
            if hasattr(self, 'has_chain_operators'):
                if self.has_chain_operators:
                    cart_coords = cart_coords @ self.chain_operators[segid]["mx"] + self.chain_operators[segid]["ori"]

            coord_arrays.append(cart_coords)

        return np.concatenate(coord_arrays)

    @property
    def chain_operators(self):
        return self._chain_operators

    @chain_operators.setter
    def chain_operators(self, op):
        if op is None:
            op = {
                chain: {"ori": np.array([0, 0, 0]), "mx": np.identity(3)}
                for chain in self.ICs
            }
            self.has_chain_operators = False
        else:
            self.has_chain_operators = True
        self._chain_operators = op
        self.perturbed = True

@dataclass
class ICAtom:
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