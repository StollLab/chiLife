from __future__ import annotations
import numbers
import operator
from functools import partial, update_wrapper

import MDAnalysis
from MDAnalysis.core.selection import AtomSelection

from .pdb_utils import sort_pdb
from .Topology import Topology
import numpy as np
from numpy.typing import ArrayLike
from scipy.spatial import cKDTree
import chilife


# TODO:
#   Behavior: AtomSelections should have orders to be enforced when indexing.

masked_properties = ('record_types', 'atomids', 'names', 'altlocs', 'altLocs', 'resnames', 'resnums', 'icodes', 'resids',
                     'resis', 'chains', 'occupancies', 'bs', 'segs', 'segids', 'atypes', 'types', 'charges', 'ix',
                     'resixs', 'segixs', '_Atoms', 'atoms')

singles = ('record_type', 'name', 'altloc', 'altLoc', 'atype', 'type', 'resn', 'resname', 'resnum', 'resid', 'icode',
           'resi', 'chain', 'segid', 'charge')


class MolecularSystemBase:
    """Base class for molecular systems containing attributes universal to """

    def __getattr__(self, key):
        if 'molsys' not in self.__dict__:
            self.__getattribute__(key)
        elif key in singles:
            return np.squeeze(self.molsys.__getattribute__(key + 's')[self.mask]).flat[0]
        elif key not in self.__dict__['molsys'].__dict__:
            self.molsys.__getattribute__(key)
        elif key == 'trajectory':
            return self.molsys.trajectory
        elif key == 'atoms':
            return self.molsys.__getattribute__(key)[self.mask]
        elif key in masked_properties or key in singles:
            return np.squeeze(self.molsys.__getattribute__(key)[self.mask])
        else:
            return self.molsys.__getattribute__(key)

    def __setattr__(self, key, value):
        if key in ('molsys', 'mask'):
            super(MolecularSystemBase, self).__setattr__(key, value)
        elif key in singles:
            self.molsys.__getattribute__(key + 's')[self.mask] = value
        elif key not in self.__dict__['molsys'].__dict__:
            super(MolecularSystemBase, self).__setattr__(key, value)
        elif key == 'trajectory':
            self.molsys.__dict__['trajectory'] = value
        elif key in masked_properties or key in singles:
            self.molsys.__getattribute__(key)[self.mask] = value
        else:
            super(MolecularSystemBase, self).__setattr__(key, value)

    def select_atoms(self, selstr):
        """
        Select atoms from :class:`~MolSys` or :class:`~AtomSelection` based on selection a selection string, similar
        to the `select_atoms` method from `MDAnalysis <https://docs.mdanalysis.org/stable/documentation_pages/selections.html>`_

        Parameters
        ----------
        selstr : str
            Selection string

        Returns
        -------
        atoms : :class:`~AtomSelection`
            Selected atoms
        """

        mask = process_statement(selstr, self.logic_keywords, self.molsys_keywords)
        if hasattr(self, 'mask'):
            t_mask = np.zeros(self.molsys.n_atoms, dtype=bool)
            t_mask[self.mask] = True
            mask *= t_mask

        mask = np.argwhere(mask).T[0]
        return AtomSelection(self.molsys, mask)

    def __getitem__(self, item):

        if np.issubdtype(type(item), np.integer):
            relidx = self.molsys.ix[self.mask][item]
            return Atom(self.molsys, relidx)

        if isinstance(item, slice):
            return AtomSelection(self.molsys, self.mask[item])

        elif isinstance(item, (np.ndarray, list, tuple)):
            item = np.asarray(item)
            if len(item) == 1 or item.sum() == 1:
                return Atom(self.molsys, self.mask[item])
            elif item.dtype == bool:
                item = np.argwhere(item).T[0]
                return AtomSelection(self.molsys, item)
            else:
                return AtomSelection(self.molsys, self.mask[item])

        elif hasattr(item, '__iter__'):
            if all([np.issubdtype(type(x), int) for x in item]):
                return AtomSelection(self.molsys, self.mask[item])

        raise TypeError('Only integer, slice type, and boolean mask arguments are supported at this time')


    @property
    def fname(self):
        """File name or functional name of the molecular system"""
        return self.molsys._fname

    @fname.setter
    def fname(self, val):
        self.molsys._fname = val

    @property
    def positions(self):
        """3-Dimensional cartesian coordinates of the atoms in the molecular system"""
        return self.coords

    @positions.setter
    def positions(self, val):
        self.coords = np.asarray(val)

    @property
    def coords(self):
        """3-dimensional cartesian coordinates of the atoms in the molecular system"""
        return self.trajectory.coords[self.trajectory.frame, self.mask]

    @coords.setter
    def coords(self, val):
        self.trajectory.coords[self.trajectory.frame, self.mask] = np.asarray(val)

    @property
    def n_atoms(self):
        """"Number of atoms in the molecular system"""
        return len(self.mask)

    @property
    def n_residues(self):
        """Number of residues in the molecular system"""
        return len(np.unique(self.resixs[self.mask]))

    @property
    def n_chains(self):
        """Number of chains in the molecular system"""
        return len(np.unique(self.chains[self.mask]))

    @property
    def residues(self):
        """:class:`~ResidueSelection` for all residues of the molecular system"""
        return ResidueSelection(self.molsys, self.mask)

    @property
    def segments(self):
        """:class:`~SegmentSelection` for all segments (chains) of the molecular system"""
        return SegmentSelection(self.molsys, self.mask)

    @property
    def logic_keywords(self):
        """Dictionary of logic keywords (strings) pertaining to :meth:`atom_selection` method. Keywords are mapped
        to the associated logic function """
        return self.molsys._logic_keywords

    @property
    def molsys_keywords(self):
        """Dictionary of molecular system keywords (strings) pertaining to :meth:`atom_selection` method. Keywords are
        mapped to the associated molecular system attributes"""
        return self.molsys._molsys_keywords


    @property
    def universe(self):
        """Wrapper attribute to allow molecular systems to behave like MDAnalysis AtomGroup objects"""
        return self.molsys

    def copy(self):
        """Creates a deep copy of the underlying MolSys and returns an :class:`AtomSelection` from the new copy
        matching the current selection if it exists."""
        p2 = self.molsys.copy()
        return AtomSelection(p2, self.mask.copy())

    def __iter__(self):
        """Iterate over all atoms of the MolSys"""
        for idx in self.mask:
            yield self.molsys.atoms[idx]

    def __eq__(self, value):
        """Checks if two molecular systems are equal by making sure they have the same ``self.molsys`` and the same
        mask"""
        return self.molsys is value.molsys and self.mask == value.mask



class MolSys(MolecularSystemBase):
    """
    An object containing all attributes of a molecular system.

    Parameters
    ----------
    atomids : np.ndarray
        Array of atom indices where each index is unique to each atom in the molecular system.
    names : np.ndarray
        Array of atom names, e.g. 'N', 'CA' 'C'.
    altlocs : np.ndarray
        Array of alternative location identifiers for each atom.
    resnames : np.ndarray
        Array of residue names for each atom.
    resnums : np.ndarray
        Array of residue numbers for each atom.
    chains : np.ndarray
        Array of chain identifiers for each atom.
    trajectory : np.ndarray
        Cartesian coordinates of each atom for each state of an ensemble or for each timestep of a trajectory.
    occupancies : np.ndarray
        Occupancy or q-factory for each atom.
    bs : np.ndarray
        thermal, or b-factors for each atom.
    segs : np.ndarray
        Segment/chain IDs for each atom.
    atypes : np.ndarray
        Atom/element types.
    charges : np.ndarray
        Formal or partial charges assigned to each atom.
    bonds : ArrayLike
        Array of atom ID pairs corresponding to all atom bonds in the system.
    name : str
        Name of molecular system
    """
    def __init__(
            self,
            record_types: np.ndarray,
            atomids: np.ndarray,
            names: np.ndarray,
            altlocs: np.ndarray,
            resnames: np.ndarray,
            resnums: np.ndarray,
            icodes: np.ndarray,
            chains: np.ndarray,
            trajectory: np.ndarray,
            occupancies: np.ndarray,
            bs: np.ndarray,
            segs: np.ndarray,
            atypes: np.ndarray,
            charges: np.ndarray,
            bonds: ArrayLike = None,
            name: str = 'Noname_MolSys'

    ):

        self.molsys = self
        self.record_types = record_types.copy()
        self.atomids = atomids.copy()
        self.names = names.copy().astype('U4')
        self.altlocs = altlocs.copy()
        self.resnames = resnames.copy()
        self.resnums = resnums.copy()
        self.icodes = icodes.copy()
        self.chains = chains.copy()
        self.trajectory = Trajectory(trajectory.copy(), self)
        self.occupancies = occupancies.copy()
        self.bs = bs.copy()
        self.segs = segs.copy()
        self.atypes = atypes.copy()
        self.charges = charges.copy()
        self._fname = name

        self.ix = np.arange(len(self.atomids))
        self.mask = np.arange(len(self.atomids))

        # resix_borders = np.nonzero(np.r_[1, np.diff(self.resnums)[:-1]])
        # resix_borders = np.append(resix_borders, [self.n_atoms])
        # resixs = []
        # for i in range(len(resix_borders) - 1):
        #     dif = resix_borders[i + 1] - resix_borders[i]
        #     resixs.append(np.ones(dif, dtype=int) * i)

        residues, resixs = [], []
        i = 0
        for aidx in range(self.n_atoms):
            residue_id = self.resnums[aidx], self.icodes[aidx], self.chains[aidx]
            if residue_id not in residues:
                residues.append(residue_id)
                i += 1

            resixs.append(i)

        self.resixs = np.array(resixs)
        self.resindices = self.resixs
        if np.all(self.chains == '') or np.all(self.chains == 'SYSTEM'):
            self.segixs = np.array([ord('A') - 65 for x in self.chains])
        else:
            self.segixs = np.array([ord(x) - 65 for x in self.chains])
        self.segindices = self.segixs

        self.is_protein = np.array([res in chilife.SUPPORTED_RESIDUES for res in resnames])

        self._molsys_keywords = {'id': self.atomids,
                                 'record_type': self.record_types,
                                 'name': self.names,
                                 'altloc': self.altlocs,
                                 'resname': self.resnames,
                                 'resnum': self.resnums,
                                 'icode': self.icodes,
                                 'chain': self.chains,
                                 'occupancies': self.occupancies,
                                 'b': self.bs,
                                 'segid': self.chains,
                                 'type': self.atypes,
                                 'charges': self.charges,

                                 'resi': self.resnums,
                                 'resid': self.resnums,
                                 'i.': self.resnums,
                                 's.': self.chains,
                                 'c.': self.chains,
                                 'r.': self.resnames,
                                 'n.': self.names,
                                 'resn': self.resnames,
                                 'q': self.occupancies,
                                 'elem': self.atypes,
                                 'element': self.atypes,

                                 'protein': self.is_protein,
                                 '_len': self.n_atoms}

        self._logic_keywords = {'and': operator.mul,
                                'or': operator.add,
                                'not': unot,
                                '!': unot,
                                '<': operator.lt,
                                '>': operator.gt,
                                '==': operator.eq,
                                '<=': operator.le,
                                '>=': operator.ge,
                                '!=': operator.ne,
                                'byres': partial(byres, molsys=self.molsys),
                                'within': update_wrapper(partial(within, molsys=self.molsys), within),
                                'around': update_wrapper(partial(within, molsys=self.molsys), within)}


        # Aliases
        self.resns = self.resnames
        self.resis = self.resnums
        self.altLocs = self.altlocs
        self.resids = self.resnums
        self.segids = self.chains
        self.types = self.atypes

        # Atoms creation is required to be last
        self.atoms = AtomSelection(self, self.mask)
        self.topology = Topology(self, bonds) if bonds is not None else None


    @classmethod
    def from_pdb(cls, file_name, sort_atoms=False):
        """
        Reads a pdb file and returns a MolSys object

        Parameters
        ----------
        file_name : str, Path
            Path to the PDB file to read.
        sort_atoms : bool
            Presort atoms for consisted internal coordinate definitions

        Returns
        -------
        cls : MolSys
            Molecular system object of the PDB structure.
        """

        keys = ["record_types", "atomids", "names", "altlocs", "resnames", "chains", "resnums",
                "icodes", "coords", "occupancies", "bs", "segs", "atypes", "charges"]
        if sort_atoms:
            lines = sort_pdb(file_name)
        else:
            with open(file_name, 'r') as f:
                lines = f.readlines()

            lines = [line for line in lines if line.startswith(('MODEL', 'ENDMDL', 'ATOM', 'HETATM'))]
            start_idxs, end_idxs = [],  []
            for i, line in enumerate(lines):
                if line.startswith('MODEL'):
                    start_idxs.append(i + 1)
                elif line.startswith("ENDMDL"):
                    end_idxs.append(i)

            if len(start_idxs) > 0:
                lines = [lines[start:end] for start, end in zip(start_idxs, end_idxs)]

        if isinstance(lines[0], str):
            lines = [lines]

        PDB_data = [(line[:6].strip(), i, line[12:16].strip(), line[16:17].strip(),
                     line[17:20].strip(), line[21:22].strip(), int(line[22:26]), line[26:27].strip(),
                     (float(line[30:38]), float(line[38:46]), float(line[46:54])), float(line[54:60]),
                     float(line[60:66]), line[72:73].strip(), line[76:78].strip(), line[78:80].strip())
                    for i, line in enumerate(lines[0])]

        pdb_dict = {key: np.array(data) for key, data in zip(keys, zip(*PDB_data))}
        trajectory = [pdb_dict.pop('coords')]

        if len(lines) > 1:
            for struct in lines[1:]:
                frame_coords = [(line[30:38], line[38:46], line[46:54]) for line in struct]

                if len(frame_coords) != len(PDB_data):
                    raise ValueError('All models in a multistate PDB must have the same atoms')

                trajectory.append(frame_coords)

        pdb_dict['trajectory'] = np.array(trajectory, dtype=float)
        pdb_dict['name'] = file_name
        return cls(**pdb_dict)

    @classmethod
    def from_arrays(cls,
                    anames: ArrayLike,
                    atypes: ArrayLike,
                    resnames: ArrayLike,
                    resindices: ArrayLike,
                    resnums: ArrayLike,
                    segindices: ArrayLike,
                    record_types: ArrayLike = None,
                    altlocs: ArrayLike = None,
                    icodes: ArrayLike = None,
                    segids: ArrayLike = None,
                    trajectory: ArrayLike = None,
                    **kwargs
                    ) -> MolSys:
        """
        Create a MolSys object from a minimal set of arrays.

        Parameters
        ----------
        anames : ArrayLike
            Array of atom names.
        atypes : ArrayLike
            Array of atom types/elements.
        resnames : ArrayLike
            Array of residue names for each atom.
        resindices : ArrayLike
            Array of residue indices for each atom.
        resnums : ArrayLike
            Array of residue numbers for each atom.
        segindices : ArrayLike
            Array of segment indices for each atom.
        segids : ArrayLike
            Array of segment IDs for each atom.
        trajectory : ArrayLike
            Array of cartesian coordinates for each atom and each state of an ensemble or frame of a trajectory.

        Returns
        -------
        cls : MolSys
            Molecular system object of the PDB structure.
        """
        anames = np.asarray(anames)
        atypes = np.asarray(atypes)
        resindices = np.asarray(resindices)
        segindices = np.asarray(segindices)

        n_atoms = len(anames)
        atomids = np.arange(n_atoms)
        altlocs = np.array([''] * n_atoms) if altlocs is None else np.array(altlocs)
        icodes = np.array([''] * n_atoms) if icodes is None else np.array(icodes)


        if len(resnums) != n_atoms:
            resnums = np.array([resnums[residx] for residx in resindices])
        else:
            resnums = np.asarray(resnums)

        if len(resnames) != n_atoms:
            resnames = np.array([resnames[residx] for residx in resindices])
        else:
            resnames = np.asarray(resnames)

        if len(icodes) != n_atoms:
            icodes = np.array([icodes[residx] for residx in resindices])
        else:
            icodes = np.asarray(icodes)

        if len(segindices) != n_atoms:
            segindices = np.array([segindices[x] for x in resindices])

        if segids is None:
            segids = np.array(["ABCDEFGHIJKLMNOPQRSTUVWXYZ"[i] for i in segindices])
        elif len(segids) != n_atoms:
            segids = np.array([segids[i] for i in segindices])
        else:
            segids = np.asarray(segids)

        chains = segids

        if trajectory is None:
            trajectory = np.empty((1, n_atoms, 3))
        elif len(trajectory.shape) == 2:
            trajectory = trajectory[None, ...]

        if record_types is None:
            record_types = np.array(['ATOM' if rnme in chilife.SUPPORTED_RESIDUES else 'HETATM' for rnme in resnames])

        occupancies = np.ones(n_atoms)
        bs = np.ones(n_atoms)
        charges = kwargs.pop('charges', np.zeros(n_atoms))

        return cls(record_types, atomids, anames, altlocs, resnames, resnums, icodes, chains,
                   trajectory, occupancies, bs, segids, atypes, charges, **kwargs)

    @classmethod
    def from_atomsel(cls, atomsel, frames=None):
        """
        Create a new MolSys object from an existing :class:`~AtomSelection` or
        `MDAnalysis.AtomGroup <https://userguide.mdanalysis.org/stable/atomgroup.html>`_ object.

        Parameters
        ----------
        atomsel : :class:`~AtomSelection`, MDAnalysis.AtomGroup
            The :class:`~AtomSelection` or MDAnalysis.AtomGroup from which to create the new MolSys object.
        frames : int, Slice, ArrayLike
            Frame index, slice or array of frame indices corresponding to the frames of the atom selection you wish to
            extract into a new :class:`~MolSys` object.

        Returns
        -------
        cls : MolSys
            Molecular system object of the PDB structure.
        """

        if isinstance(atomsel, (MDAnalysis.AtomGroup, AtomSelection)):
            U = atomsel.universe
        elif isinstance(atomsel, MDAnalysis.Universe):
            U = atomsel
            atomsel = U.atoms

        if frames is None:
            frames = slice(0, len(U.trajectory))

        record_types = atomsel.record_types if hasattr(atomsel, 'record_types') else None
        icodes = atomsel.icodes if hasattr(atomsel, 'icodes') else None
        anames = atomsel.names
        atypes = atomsel.types
        resnames = atomsel.resnames
        resnums = atomsel.resnums
        ridx_map = {num: i for i, num in enumerate(np.unique(atomsel.resindices))}
        resindices = np.array([ridx_map[num] for num in atomsel.resindices ])
        segids = atomsel.segids
        sidx_map = {num: i for i, num in enumerate(np.unique(atomsel.segindices))}
        segindices = np.array([sidx_map[num] for num in atomsel.segindices])

        if hasattr(U.trajectory, 'coordinate_array'):
            trajectory = U.trajectory.coordinate_array[frames][:, atomsel.ix, :]
        else:
            trajectory = []
            for ts in U.trajectory[frames]:
                trajectory.append(atomsel.positions)
            trajectory = np.array(trajectory)

        return cls.from_arrays(anames, atypes, resnames, resindices, resnums, segindices,
                               segids=segids, trajectory=trajectory, record_types=record_types,
                               icodes=icodes)


    @classmethod
    def from_rdkit(cls, mol):
        """
        Create a MolSys from an rdkit Mol object with embedded conformers.

        Parameters
        ----------
        mol

        Returns
        -------

        """
        atypes = np.array([a.GetSymbol() for a in mol.GetAtoms()])
        anames = np.array([a + str(i) for i, a in enumerate(atypes)])
        resnames = np.array(["UNK" for _ in anames])
        resindices = np.array([0] * len(anames))
        resnums = np.array([1] * len(anames))
        segindices = np.array([0] * len(anames))
        segids = np.array(["A"] * len(anames))
        trajectory = np.array([conf.GetPositions() for conf in mol.GetConformers()])
        bonds = np.array([[b.GetBeginAtomIdx(), b.GetEndAtomIdx()] for b in mol.GetBonds()])
        return cls.from_arrays(anames, atypes, resnames, resindices, resnums, segindices,
                               segids=segids, trajectory=trajectory, bonds=bonds)

    @classmethod
    def from_openMM(cls, simulation):
        """
        Create a MolSys from an openMM Simulation object.

        Parameters
        ----------
        simulation : openmm.simulation.Simulation
            The simulation object to create the MolSys object from.

        Returns
        -------
        sys : chilife.MolSys
            chiLife molecular system as a clone from
        """
        try:
            from openmm.unit import angstrom
        except:
            raise RuntimeError("You must have the optional openMM dependency installed to use from_openMM")

        top = simulation.topology
        atypes = np.array([a.element.symbol for a in top.atoms()])
        anames = np.array([a.name for a in top.atoms()])
        resnames = np.array([a.residue.name for a in top.atoms()])
        resindices = np.array([a.residue.index for a in top.atoms()])
        resnums = np.array([int(a.residue.id) for a in top.atoms()])
        segindices = np.array([a.residue.chain.index for a in top.atoms()])
        segids = np.array([a.residue.chain.id for a in top.atoms()])
        trajectory = simulation.context.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit(angstrom)
        bonds = np.array([[bond.atom1.index, bond.atom2.index] for bond in top.bonds()])
        sys = cls.from_arrays(anames, atypes, resnames, resindices, resnums,
                              segindices, segids=segids, trajectory=trajectory, bonds=bonds)
        return sys

    def copy(self):
        """Create a deep copy of the MolSys."""

        return MolSys(
            record_types=self.record_types,
            atomids=self.atomids,
            names=self.names,
            altlocs=self.altlocs,
            resnames=self.resnames,
            resnums=self.resnums,
            icodes=self.icodes,
            chains=self.chains,
            trajectory=self.trajectory.coords,
            occupancies=self.occupancies,
            bs=self.bs,
            segs=self.segs,
            atypes=self.atypes,
            charges=self.charges,
            name=self.names)

    def load_new(self, coordinates):
        """
        Load a new set of 3-dimensional coordinates into the MolSys.

        Parameters
        ----------
        coordinates : ArrayLike
            Set of new coordinates to load into the MolSys
        """
        self.trajectory.load_new(coordinates=coordinates)

    @property
    def _Atoms(self):
        if not hasattr(self, '__Atoms'):
            self.__Atoms = np.array([Atom(self, i) for i in range(self.n_atoms)])
        return self.__Atoms


class Trajectory:
    """
    Object containing and managing 3-dimensional coordinates of :class:`~MolSys` objects, particularly when there are
    multiple states or time steps.

    Parameters
    ----------
    coordinates : np.ndarray
        3-dimensional coordinates for all atoms and all states/frames of the associated :class:`~MolSys` object.
    molsys : MolSys
        The associated :class:`~MolSys` object.
    timestep : float
        The time step or change in time between frames/states of the trajectory/ensemble.
    """

    def __init__(self, coordinates : np.ndarray, molsys : MolSys, timestep : float=1):

        self.molsys = molsys
        self.timestep = timestep
        self.coords = coordinates
        self.coordinate_array = coordinates
        self._frame = 0
        self.time = np.arange(0, len(self.coords)) * self.timestep

    def load_new(self, coordinates):
        """
        Load a new set (trajectory or ensemble) of 3-dimensional coordinates into the Trajectory.

        Parameters
        ----------
        coordinates : ArrayLike
            Set of new coordinates to load into the MolSys
        """
        self.coords = coordinates
        self.coordinate_array = coordinates
        self.time = np.arange(0, len(self.coords)) * self.timestep

    def __getitem__(self, item):
        if isinstance(item, (slice, list, np.ndarray)):
            return TrajectoryIterator(self, item)

        elif isinstance(item, numbers.Integral):
            self._frame = item
            return self.time[self._frame]

    def __iter__(self):
        return iter(TrajectoryIterator(self))

    def __len__(self):
        return len(self.time)

    @property
    def frame(self):
        """The frame number the trajectory is currently on"""
        return self._frame

    @frame.setter
    def frame(self, value):
        self._frame = value


class TrajectoryIterator:
    """An iterator object for looping over :class:`~Trajectory` objects."""

    def __init__(self, trajectory, arg=None):
        self.trajectory = trajectory
        if arg is None:
            self.indices = range(len(self.trajectory.time))
        elif isinstance(arg, slice):
            self.indices = range(*arg.indices(len(self.trajectory.time)))
        elif isinstance(arg, (np.ndarray, list, tuple)):
            if np.issubdtype(arg.dtype, np.integer):
                self.indices = arg
            elif np.dtype == bool:
                self.indices = np.argwhere(arg)
            else:
                raise TypeError(f'{arg.dtype} is not a valid indexor for a trajectory')
        else:
            raise TypeError(f'{arg} is not a valid indexor for a trajectory')

    def __iter__(self):
        for frame in self.indices:
            yield self.trajectory[frame]

    def __len__(self):
        return len(self.indices)


def process_statement(statement, logickws, subjectkws):
    """
    Parse a selection string to identify subsets of atoms that satisfy the given conditions.

    Parameters
    ----------
    statement : str
        The selection string to parse.
    logickws : dict
        Dictionary of logic keywords (strings) mapped to the associated logic function.
    subjectkws : dict
        Dictionary of subject keywords (strings) mapped to the associated subject attributes.

    Returns
    -------
    mask : np.ndarray
        Array of subject indices that satisfy the given conditions.
    """
    sub_statements = parse_paren(statement)

    mask = np.ones(subjectkws['_len'], dtype=bool)
    operation = None

    for stat in sub_statements:
        if stat.startswith('('):
            tmp = process_statement(stat, logickws, subjectkws)
            mask = operation(mask, tmp) if operation else tmp
            continue

        stat_split = stat.split()
        subject = None
        values = []
        while len(stat_split) > 0:

            if stat_split[0] in subjectkws:
                subject = subjectkws[stat_split.pop(0)]
                values = []

            elif stat_split[0] in logickws:
                if subject is not None:
                    values = np.array(values, dtype=subject.dtype)
                    tmp = process_sub_statement(subject, values)
                    subject, values = None, []
                    mask = operation(mask, tmp) if operation else tmp

                # Get next operation
                operation, stat_split = build_operator(stat_split, logickws)

            else:
                values += parse_value(stat_split.pop(0))

        if subject is not None:
            values = np.array(values, dtype=subject.dtype)
            tmp = process_sub_statement(subject, values)
            mask = operation(mask, tmp) if operation else tmp
            operation = None

    return mask


def parse_paren(string):
    """
    Break down a given string to parenthetically defined subcomponents

    Parameters
    ----------
    string : str
        The string to break down.

    Returns
    -------
    results : List[str]
        List of parenthetically defined subcomponents.
    """
    stack = 0
    start_idx = 0
    end_idx = 0
    results = []

    for i, c in enumerate(string):
        if c == '(':
            if stack == 0:
                start_idx = i + 1
                if i != 0:
                    results.append(string[end_idx:i].strip())
            stack += 1
        elif c == ')':
            # pop stack
            stack -= 1

            if stack == 0:
                results.append(f'({string[start_idx:i].strip()})')
                start_idx = end_idx
                end_idx = i + 1

    if end_idx != len(string) - 1:
        results.append(string[end_idx:].strip())

    results = [res for res in results if res != '']
    if len(results) == 1 and results[0].startswith('('):
        results = parse_paren(results[0][1:-1])

    if stack != 0 :
        raise RuntimeError('The provided statement is missing a parenthesis or has an '
                           'extra one.')
    return results


def check_operation(operation, stat_split, logickws):
    """
    Given an advanced operation, identify the additional arguments being passed to the advanced operation and create
    a simple operation that defined specifically for the provided arguments.
    Parameters
    ----------
    operation : callable
        A function or callable object that operates on molecular system attributes.
    stat_split : List[str]
        the arguments associated with the operation
    logickws : dict
        Dictionary of logic keywords (strings) mapped to the associated logic function.

    Returns
    -------
    operation : callable
        A simplified version of the provided operation now accounting for the user provided parameters.
    """
    advanced_operators = (logickws['within'], logickws['around'], logickws['byres'])
    if operation in advanced_operators:
        outer_operation = logickws['and']
        args = [stat_split.pop(i) for i in range(1, 1 + operation.nargs)]
        operation = partial(operation, *args)


        def toperation(a, b, outer_operation, _io):
            return outer_operation(a, _io(b))

        operation = partial(toperation, outer_operation=outer_operation, _io=operation)

    return operation


def build_operator(stat_split, logickws):
    """
    A function to combine multiple sequential operators into one.

    Parameters
    ----------
    stat_split : List[str]
        The list of string arguments passed by the user to define the sequence of operations
    logickws : dict
        Dictionary of logic keywords (strings) mapped to the associated logic function.

    Returns
    -------
    operation : callable
        A compression of sequential operations combined into one.
    split_stat : List[str]
        Any remaining arguments passed by the user that were not used to define ``operation``
    """
    operation = logickws['and']
    unary_operators = (logickws['not'], logickws['byres'])
    binary_operators = (logickws['and'], logickws['or'])
    advanced_operators = (logickws['within'], logickws['around'])

    while _io := logickws.get(stat_split[0], False):

        if _io in unary_operators:
            def toperation(a, b, operation, _io):
                return operation(a, _io(b))

            operation = partial(toperation, operation=operation, _io=_io)
            stat_split = stat_split[1:]

        elif _io in advanced_operators:
            stat_split = stat_split[1:]
            args = [stat_split.pop(0) for i in range(_io.nargs)]
            _io = partial(_io, *args)

            def toperation(a, b, operation, _io):
                return operation(a, _io(b))

            operation = partial(toperation, operation=operation, _io=_io)

        elif _io in binary_operators:
            if operation != logickws['and']:
                raise RuntimeError('Cannot have two binary logical operators in succession')
            operation = _io
            stat_split = stat_split[1:]

        if len(stat_split) == 0:
            break

    return operation, stat_split


def parse_value(value):
    """
    Helper function to parse values that may have slices.

    Parameters
    ----------
    value : str, int, ArrayLike[int]
        Values to parse.

    Returns
    -------
    return_value : List
        A list of values that satisfy the input.

    """
    return_value = []
    if '-' in value:
        start, stop = [int(x) for x in value.split('-')]
        return_value += list(range(start, stop + 1))
    elif ':' in value:
        start, stop = [int(x) for x in value.split(':')]
        return_value += list(range(start, stop + 1))
    else:
        return_value.append(value)

    return return_value


def process_sub_statement(subject, values):

    if subject.dtype == bool:
        return subject

    return np.isin(subject, values)


class AtomSelection(MolecularSystemBase):
    """
    An object containing a group of atoms from a :class:`~MolSys` object

    Parameters
    ----------
    molsys : :class:`~MolSys`
        The :class:`~MolSys` object from which the AtomSelection belongs to.
    mask : np.ndarray
        An array of atom indices that defines the atoms of ``molsys`` that make up the atom selection.
    """
    def __new__(cls, *args):
        if len(args) == 2:
            molsys, mask = args
        else:
            return object.__new__(cls)

        if isinstance(mask, int):
            return Atom(molsys, mask)
        else:
            return object.__new__(cls)

    def __init__(self, molsys, mask):

        self.molsys = molsys
        self.mask = mask\

    def __getitem__(self, item):

        if np.issubdtype(type(item), np.integer):
            relidx = self.molsys.ix[self.mask][item]
            return Atom(self.molsys, relidx)

        if isinstance(item, slice):
            return AtomSelection(self.molsys, self.mask[item])

        elif isinstance(item, (np.ndarray, list, tuple)):
            item = np.asarray(item)
            if len(item) == 1 or item.sum() == 1:
                return Atom(self.molsys, self.mask[item])
            elif item.dtype == bool:
                item = np.argwhere(item).T[0]
                return AtomSelection(self.molsys, item)
            else:
                return AtomSelection(self.molsys, self.mask[item])

        elif hasattr(item, '__iter__'):
            if all([np.issubdtype(type(x), int) for x in item]):
                return AtomSelection(self.molsys, self.mask[item])

        raise TypeError('Only integer, slice type, and boolean mask arguments are supported at this time')

    def __len__(self):
        return len(self.mask)


class ResidueSelection(MolecularSystemBase):
    """
    An object containing a group of residues and all their atoms from a :class:`~MolSys` object. Note that if one atom of
    a residue is present in the AtomSelection used to create the group, all atoms of that residue will be present in
    the ResidueSelection object.

    .. note::
        Unlike a regular :class:`~AtomSelection` object, the ``resnames``, ``resnums``, ``segids`` and ``chains`` 
        attributes of this object are tracked with respect to residue not atom and changes made to these attributes
        on this object will not be present in the parent object and vice versa.

    Parameters
    ----------
    molsys : :class:`~MolSys`
        The :class:`~MolSys` object from which the AtomSelection belongs to.
    mask : np.ndarray
        An array of atom indices that defines the atoms of ``molsys`` that make up the atom selection.
    """

    def __init__(self, molsys, mask):

        resixs = np.unique(molsys.resixs[mask])
        self.mask = np.argwhere(np.isin(molsys.resixs, resixs)).flatten()
        self.molsys = molsys

        _, self.first_ix = np.unique(molsys.resixs[self.mask], return_index=True)

        self.resixs = self.resixs[self.first_ix].flatten()
        self.resnames = self.resnames[self.first_ix].flatten()
        self.resnums = self.resnums[self.first_ix].flatten()
        self.segids = self.segids[self.first_ix].flatten()
        self.chains = self.chains[self.first_ix].flatten()

    def __setattr__(self, key, value):
        self.__dict__[key] = value

    def __getitem__(self, item):
        resixs = np.unique(self.molsys.resixs[self.mask])
        new_resixs = resixs[item]
        new_mask = np.argwhere(np.isin(self.molsys.resixs, new_resixs)).flatten()

        if np.issubdtype(type(item), int):
            return Residue(self.molsys, new_mask)
        elif isinstance(item, slice):
            return ResidueSelection(self.molsys, new_mask)
        elif hasattr(item, '__iter__'):
            if all([np.issubdtype(type(x), int) for x in item]):
                return ResidueSelection(self.molsys, new_mask)

        raise TypeError('Only integer and slice type arguments are supported at this time')

    def __len__(self):
        return len(self.first_ix)

    def __iter__(self):
        for resix, segid in zip(self.resixs, self.segids):
            mask = (self.molsys.resixs == resix) * (self.molsys.segids == segid)
            yield Residue(self.molsys, mask)


class SegmentSelection(MolecularSystemBase):
    """
    An object containing a group of segments and all their atoms from a :class:`~MolSys` object. Note that if one atom
    of a segment is present in the AtomSelection used to create the group, all atoms of that segment will be present in
    the SegmentSelection object.

    .. note::
        Unlike a regular :class:`~AtomSelection` object, the ``segids`` and ``chains`` 
        attributes of this object are tracked with respect to segements not atoms and changes made to these attributes
        on this object will not be present in the parent object and vice versa.

    Parameters
    ----------
    molsys : :class:`~MolSys`
        The :class:`~MolSys` object from which the AtomSelection belongs to.
    mask : np.ndarray
        An array of atom indices that defines the atoms of ``molsys`` that make up the atom selection.
    """

    def __init__(self, molsys, mask):
        self.seg_ixs = np.unique(molsys.segixs[mask])
        self.mask = np.argwhere(np.isin(molsys.segixs, self.seg_ixs)).flatten()
        self.molsys = molsys

        _, self.first_ix = np.unique(molsys.segixs[self.mask], return_index=True)

        self.segids = self.segids[self.first_ix].flatten()
        self.chains = self.chains[self.first_ix].flatten()

    def __setattr__(self, key, value):
        self.__dict__[key] = value

    def __getitem__(self, item):
        segixs = np.unique(self.molsys.segixs[self.mask])
        new_segixs = segixs[item]
        new_mask = np.argwhere(np.isin(self.molsys.segixs, new_segixs))

        if np.issubdtype(type(item), int):
            return Segment(self.molsys, new_mask)
        elif isinstance(item, slice):
            return SegmentSelection(self.molsys, new_mask)
        elif hasattr(item, '__iter__'):
            if all([np.issubdtype(type(x), int) for x in item]):
                return SegmentSelection(self.molsys, new_mask)

        raise TypeError('Only integer and slice type arguments are supported at this time')

    def __len__(self):
        return len(self.first_ix)

    def __iter__(self):
        for new_segix in self.seg_ixs:
            new_mask = np.argwhere(np.isin(self.molsys.segixs, new_segix))
            yield Segment(self.molsys, new_mask)



class Atom(MolecularSystemBase):
    """
    An object containing information for a single atom.
    
    Parameters
    ----------
    molsys : :class:`~MolSys`
        The :class:`~MolSys` object from which the Atom belongs to.
    mask : np.ndarray
        The index that defines the atoms of ``molsys`` that make up the atom selection.
    """
    def __init__(self, molsys, mask):
        self.__dict__['molsys'] = molsys
        self.index = mask
        self.mask = mask

        self.name = molsys.names[self.index]
        self.altloc = molsys.altlocs[self.index]
        self.atype = molsys.types[self.index]
        self.resname = molsys.resnames[self.index]
        self.resnum = molsys.resnums[self.index]
        self.chain = molsys.segids[self.index]
        self.segid = molsys.chains[self.index]
        self.charge = molsys.charges[self.index]

        self.resn = self.resname
        self.resi = self.resnum
        self.altLoc = self.altloc
        self.type = self.atype
        self.resid = self.resnum


    @property
    def position(self):
        return self.molsys.coords[self.index]

    @property
    def bonds(self):
        idxs = self.topology.bonds_any_atom[self.mask]
        return [AtomSelection(self.molsys, idx) for idx in idxs]

    @property
    def angles(self):
        idxs = self.topology.angles_any_atom[self.mask]
        return [AtomSelection(self.molsys, idx) for idx in idxs]

    @property
    def dihedrals(self):
        idxs = self.topology.dihedrals_any_atom[self.mask]
        return [AtomSelection(self.molsys, idx) for idx in idxs]

    @property
    def bonded_atoms(self):
        all_idxs = np.unique(np.concatenate(self.topology.bonds_any_atom[self.mask]))
        all_idxs  = np.array([idx for idx in all_idxs if idx != self.mask])
        return AtomSelection(self.molsys, all_idxs)


class Residue(MolecularSystemBase):
    """
    An object representing a single residue.

    Parameters
    ----------
    molsys : :class:`~MolSys`
        The :class:`~MolSys` object from which the Residue belongs to.
    mask : int
        An array of atom indices that defines the atoms of the residue
    """

    def __init__(self, molsys, mask):
        resix = np.unique(molsys.resixs[mask])[0]
        self.mask = np.argwhere(np.isin(molsys.resixs, resix)).T[0]
        self.molsys = molsys

        self.resname = molsys.resnames[self.mask][0]
        self.resname = molsys.resnames[self.mask][0]

        self.resnum = molsys.resnums[self.mask][0]
        self.resid = self.resnum
        self.icode = molsys.icodes[self.mask][0]
        self.resindex = molsys.resixs[self.mask][0]
        self.segid = molsys.segids[self.mask][0]
        self.segindex = molsys.segixs[self.mask][0]
        self.chain = molsys.chains[self.mask][0]

    def __len__(self):
        return len(self.mask)

    def phi_selection(self):
        """
        Get an :class:`~AtomSelection` of the atoms defining the Phi backbone dihedral angle of the residue.

        Returns
        -------
        sel : AtomSelection
            An :class:`~AtomSelection` object containing the atoms that make up the Phi backbone dihedral angle of
            the selected residue.
        """

        prev = self.atoms.resixs[0] - 1

        maskN_CA_C = self.mask[np.isin(self.names, ['N', 'CA', 'C'])]
        maskC = np.argwhere(np.isin(self.molsys.resixs, prev)).flatten()
        maskC = maskC[self.molsys.names[maskC] == 'C']
        mask_phi = np.concatenate((maskC, maskN_CA_C))
        sel = self.molsys.atoms[mask_phi]
        return sel if len(sel) == 4 else None

    def psi_selection(self):
        """
        Get an :class:`~AtomSelection` of the atoms defining the Psi backbone dihedral angle of the residue.

        Returns
        -------
        sel : AtomSelection
            An :class:`~AtomSelection` object containing the atoms that make up the Psi backbone dihedral angle of
            the selected residue.
        """

        nex = self.atoms.resnums + 1
        nex = np.unique(nex[nex <= self.molsys.resnums.max()])

        maskN_CA_C = self.mask[np.isin(self.names, ['N', 'CA', 'C'])]
        maskN = np.argwhere(np.isin(self.molsys.resnums, nex)).flatten()
        maskN = maskN[self.molsys.names[maskN] == 'N']
        mask_psi = np.concatenate((maskN_CA_C, maskN))
        sel = self.molsys.atoms[mask_psi]
        return sel if len(sel) == 4 else None


class Segment(MolecularSystemBase):
    """
    An object representing a single segment of a molecular system.

    Parameters
    ----------
    molsys : MolSys
        The :class:`~MolSys` object from which the segment belongs to.
    mask : int
        An array of atom indices that defines the atoms of the segment.
    """

    def __init__(self, molsys, mask):
        segix = np.unique(molsys.segixs[mask])[0]
        self.mask = np.argwhere(np.isin(molsys.segixs, segix)).T[0]
        self.molsys = molsys

        self.segid = molsys.segids[segix]
        self.chain = molsys.chains[segix]


def byres(mask, molsys):
    """Advanced operator to select atoms by residue"""
    residues = np.unique(molsys.resixs[mask])
    mask = np.isin(molsys.resixs, residues)
    return mask


def unot(mask):
    """Unitary `not` operator"""
    return ~mask


def within(distance, mask, molsys):
    """
    Advanced logic operator to identify atoms within a user defined distance.

    Parameters
    ----------
    distance : float
        The distance window defining which atoms will be included in the selection.
    mask : np.ndarray
        A boolean array defining a subset of  atoms of a :class:`~MolSys` from which the distance cutoff will be
        measured by.
    molsys : :class:`~MolSys`
        A chiLife :class:`~MolSys` from which to select atoms from .

    Returns
    -------
    out_mask : np.ndarray
        A boolean array defining a subset of  atoms of a :class:`~MolSys` that are within the user defined distance
        of the user defined selection.
    """
    distance = float(distance)
    tree1 = cKDTree(molsys.coords)
    tree2 = cKDTree(molsys.molsys.coords[mask])
    results = np.concatenate(tree2.query_ball_tree(tree1, distance))
    results = np.unique(results)
    out_mask = np.zeros_like(mask, dtype=bool)
    out_mask[results] = True
    out_mask = out_mask * ~mask
    return out_mask


within.nargs = 1


def concat_molsys(systems):
    """
    Function to concatenate two or more :class:`~MolSys` objects into a single :class:`~MolSys` object. Atoms will be
    concatenated in the order they are placed in the list.

    Parameters
    ----------
    systems : List[MolSys]
        List of :class:`~MolSys` objects to concatenate.

    Returns
    -------
    mol : MolSys
        The concatenated `~MolSys` object.
    """
    record_types = []
    anames = []
    atypes = []
    resnames = []
    resnums = []
    resindices = []
    icodes = []
    trajectory = []
    segindices = []
    segids = []

    for sys in systems:

        resind = sys.resindices + resindices[-1][-1] if len(resindices) > 0 else sys.resindices
        record_types.append(sys.record_types)
        anames.append(sys.names)
        atypes.append(sys.atypes)
        resnames.append(sys.resnames)
        resnums.append(sys.resnums)
        icodes.append(sys.icodes)
        resindices.append(resind)
        trajectory.append(sys.positions)
        segindices.append(sys.segindices)
        segids.append([segid if len(segid) == 1 else 'A' for segid in sys.segids])

    record_types = np.concatenate(record_types)
    anames = np.concatenate(anames)
    atypes = np.concatenate(atypes)
    resnames = np.concatenate(resnames)
    resnums = np.concatenate(resnums)
    resindices = np.concatenate(resindices)
    icodes = np.concatenate(icodes)
    segindices = np.concatenate(segindices)
    segids = np.concatenate(segids)
    trajectory = np.concatenate(trajectory)

    mol = MolSys.from_arrays(anames, atypes, resnames, resindices, resnums, segindices,
                             segids=segids, trajectory=trajectory, record_types=record_types, icodes=icodes)

    return mol