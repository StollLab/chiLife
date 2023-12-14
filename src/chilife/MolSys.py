from __future__ import annotations
import numbers
import operator
from functools import partial, update_wrapper

import MDAnalysis

from .protein_utils import sort_pdb
from .Topology import Topology
import numpy as np
from numpy.typing import ArrayLike
from scipy.spatial import cKDTree
import chilife


# TODO:
#   Performance enhancement: Preconstruct Atom objects
#   Behavior: AtomSelections should have orders to be enforced when indexing.
#   Performance enhancement: Find a faster way to retrieve coordinate data from trajectory @property seems to have
#   Feature: Add from_mda class method.
masked_properties = ('atomids', 'names', 'altlocs', 'resnames', 'resnums', 'chains', 'occupancies',
                     'bs', 'segs', 'segids', 'atypes', 'charges', 'ix', 'resixs', 'segixs', '_Atoms', 'atoms')

class MolecularSystemBase:
    """Base class for molecular systems"""

    def __getattr__(self, key):
        if 'molsys' not in self.__dict__:
            self.__getattribute__(key)
        elif key not in self.__dict__['molsys'].__dict__:
            self.molsys.__getattribute__(key)
        elif key == 'trajectory':
            return self.molsys.trajectory
        elif key == 'atoms':
            return self.molsys.__getattribute__(key)[self.mask]
        elif key in masked_properties:
            return np.squeeze(self.molsys.__getattribute__(key)[self.mask])
        else:
            return self.molsys.__getattribute__(key)

    def __setattr__(self, key, value):
        if key in ('molsys', 'mask'):
            super(MolecularSystemBase, self).__setattr__(key, value)
        elif key not in self.__dict__['molsys'].__dict__:
            super(MolecularSystemBase, self).__setattr__(key, value)
        elif key == 'trajectory':
            self.molsys.__dict__['trajectory'] = value
        elif key in masked_properties:
            self.molsys.__getattribute__(key)[self.mask] = value
        else:
            super(MolecularSystemBase, self).__setattr__(key, value)

    def select_atoms(self, selstr):
        mask = process_statement(selstr, self.logic_keywords, self.molsys_keywords)
        if hasattr(self, 'mask'):
            t_mask = np.zeros(self.molsys.n_atoms, dtype=bool)
            t_mask[self.mask] = True
            mask *= t_mask

        mask = np.argwhere(mask).T[0]
        return AtomSelection(self.molsys, mask)

    @property
    def fname(self):
        return self.molsys._fname

    @fname.setter
    def fname(self, val):
        self.molsys._fname = val

    @property
    def positions(self):
        return self.coords

    @positions.setter
    def positions(self, val):
        self.coords = np.asarray(val)

    @property
    def coords(self):
        return self.trajectory.coords[self.trajectory.frame, self.mask]

    @coords.setter
    def coords(self, val):
        self.trajectory.coords[self.trajectory.frame, self.mask] = np.asarray(val)

    @property
    def resindices(self):
        return np.unique(self.molsys.resixs[self.mask])

    @property
    def segindices(self):
        return np.unique(self.molsys.segixs[self.mask])

    @property
    def n_atoms(self):
        return len(self.mask)

    @property
    def n_residues(self):
        return len(np.unique(self.resnums[self.mask]))

    @property
    def n_chains(self):
        return len(np.unique(self.chains[self.mask]))

    @property
    def residues(self):
        return ResidueSelection(self.molsys, self.mask)

    @property
    def segments(self):
        return SegmentSelection(self.molsys, self.mask)

    @property
    def logic_keywords(self):
        return self.molsys._logic_keywords

    @property
    def molsys_keywords(self):
        return self.molsys._molsys_keywords

    @property
    def types(self):
        return self.molsys.atypes[self.mask]

    @types.setter
    def types(self, value):
        self.molsys.atypes[self.mask] = value

    @property
    def universe(self):
        return self.molsys

    def copy(self):
        p2 = self.molsys.copy()
        return AtomSelection(p2, self.mask.copy())

    def __iter__(self):
        for idx in self.mask:
            yield self.molsys.atoms[idx]


class MolSys(MolecularSystemBase):
    """MolSys object"""
    def __init__(
            self,
            atomids: np.ndarray,
            names: np.ndarray,
            altlocs: np.ndarray,
            resnames: np.ndarray,
            resnums: np.ndarray,
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
        self.atomids = atomids.copy()
        self.names = names.copy().astype('U4')
        self.altlocs = altlocs.copy()
        self.resnames = resnames.copy()
        self.resnums = resnums.copy()
        self.chains = chains.copy()
        self.trajectory = Trajectory(trajectory.copy(), self)
        self.occupancies = occupancies.copy()
        self.bs = bs.copy()
        self.segs = segs.copy()
        self.segids = self.chains
        self.atypes = atypes.copy()
        self.charges = charges.copy()
        self._fname = name

        self.topology = Topology(bonds) if bonds is not None else None

        self.ix = np.arange(len(self.atomids))
        self.mask = np.arange(len(self.atomids))

        resix_borders = np.nonzero(np.r_[1, np.diff(self.resnums)[:-1]])
        resix_borders = np.append(resix_borders, [self.n_atoms])
        resixs = []
        for i in range(len(resix_borders) - 1):
            dif = resix_borders[i + 1] - resix_borders[i]
            resixs.append(np.ones(dif, dtype=int) * i)

        self.resixs = np.concatenate(resixs)
        if np.all(self.chains == '') or np.all(self.chains == 'SYSTEM'):
            self.segixs = np.array([ord('A') - 65 for x in self.chains])
        else:
            self.segixs = np.array([ord(x) - 65 for x in self.chains])

        uidx, uidxidx, nuidxs = np.unique(self.resixs, return_index=True, return_inverse=True)
        resnames = self.resnames[uidx]
        truth = np.array([res in chilife.SUPPORTED_RESIDUES for res in resnames])
        self.is_protein = truth[nuidxs]

        self._molsys_keywords = {'id': self.atomids,
                                  'name': self.names,
                                  'altloc': self.altlocs,
                                  'resname': self.resnames,
                                  'resnum': self.resnums,
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
                                  '_len': self.n_atoms,
                                  }

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

        self.atoms = AtomSelection(self, self.mask)

    @classmethod
    def from_pdb(cls, file_name, sort_atoms=False):
        """reads a pdb file and returns a MolSys object"""

        keys = ["skip", "atomids", "names", "altlocs", "resnames", "chains", "resnums",
                "skip", "coords", "occupancies", "bs", "segs", "atypes", "charges"]
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

        pdb_dict = {key: np.array(data) for key, data in zip(keys, zip(*PDB_data)) if key != "skip"}
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
                    segids: ArrayLike = None,
                    trajectory: ArrayLike = None,
                    ) -> MolSys:
        anames = np.asarray(anames)
        atypes = np.asarray(atypes)
        resindices = np.asarray(resindices)
        segindices = np.asarray(segindices)

        n_atoms = len(anames)
        atomids = np.arange(n_atoms)
        altlocs = np.array([''] * n_atoms)

        if len(resnums) != n_atoms:
            resnums = np.array([resnums[residx] for residx in resindices])
        else:
            resnums = np.asarray(resnums)

        if len(resnames) != n_atoms:
            resnames = np.array([resnames[residx] for residx in resindices])
        else:
            resnames = np.asarray(resnames)

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

        occupancies = np.ones(n_atoms)
        bs = np.ones(n_atoms)
        charges = np.zeros(n_atoms)

        return cls(atomids, anames, altlocs, resnames, resnums, chains,
                   trajectory, occupancies, bs, segids, atypes, charges)

    @classmethod
    def from_atomsel(cls, atomsel, frames=None):

        if isinstance(atomsel, (MDAnalysis.AtomGroup, AtomSelection)):
            U = atomsel.universe
        elif isinstance(atomsel, MDAnalysis.Universe):
            U = atomsel
            atomsel = U.atoms

        if frames is None:
            frames = slice(0, len(U.trajectory))

        anames = atomsel.names
        atypes = atomsel.types
        resnames = atomsel.resnames
        resnums = atomsel.resnums
        ridx_map = {num: i for i, num in enumerate(np.unique(resnums))}
        resindices = np.array([ridx_map[num] for num in resnums ])
        segids = atomsel.segids
        segindices = np.array([ridx_map[num] for num in resnums])

        if hasattr(U.trajectory, 'coordinate_array'):
            trajectory = U.trajectory.coordinate_array[frames][:, atomsel.ix, :]
        else:
            trajectory = []
            for ts in U.trajectory[frames]:
                trajectory.append(atomsel.positions)
            trajectory = np.array(trajectory)

        return cls.from_arrays(anames, atypes, resnames, resindices, resnums, segindices, segids, trajectory)

    def copy(self):
        return MolSys(
            atomids=self.atomids,
            names=self.names,
            altlocs=self.altlocs,
            resnames=self.resnames,
            resnums=self.resnums,
            chains=self.chains,
            trajectory=self.trajectory.coords,
            occupancies=self.occupancies,
            bs=self.bs,
            segs=self.segs,
            atypes=self.atypes,
            charges=self.charges,
            name=self.names)

    def load_new(self, coordinates):
        self.trajectory.load_new(coordinates=coordinates)

    @property
    def _Atoms(self):
        if not hasattr(self, '__Atoms'):
            self.__Atoms = np.array([Atom(self, i) for i in range(self.n_atoms)])
        return self.__Atoms


class Trajectory:

    def __init__(self, coordinates, molsys, timestep=1):
        self.molsys = molsys
        self.timestep = timestep
        self.coords = coordinates
        self.coordinate_array = coordinates
        self._frame = 0
        self.time = np.arange(0, len(self.coords)) * self.timestep

    def load_new(self, coordinates):
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
        return self._frame

    @frame.setter
    def frame(self, value):
        self._frame = value


class TrajectoryIterator:

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
        self.mask = mask

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

    def __init__(self, molsys, mask):

        resixs = np.unique(molsys.resixs[mask])
        self.mask = np.argwhere(np.isin(molsys.resixs, resixs)).flatten()
        self.molsys = molsys

        _, self.first_ix = np.unique(molsys.resixs[self.mask], return_index=True)

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
        for resnum in self.resnums:
            mask = self.molsys.resnums == resnum
            yield Residue(self.molsys, mask)


class SegmentSelection(MolecularSystemBase):

    def __init__(self, molsys, mask):
        seg_ixs = np.unique(molsys.segixs[mask])
        self.mask = np.argwhere(np.isin(molsys.segixs, seg_ixs)).flatten()
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


class Atom(MolecularSystemBase):
    def __init__(self, molsys, mask):
        self.__dict__['molsys'] = molsys
        self.index = mask
        self.mask = mask


        self.name = molsys.names[self.index]
        self.altLoc = molsys.altlocs[self.index]
        self.atype = molsys.types[self.index]
        self.type = self.atype

        self.resn = molsys.resnames[self.index]
        self.resname = self.resn
        self.resi = molsys.resnums[self.index]
        self.resid = self.resi
        self.resnum = self.resi
        self.chain = molsys.segids[self.index]
        self.segid = molsys.chains[self.index]

    @property
    def position(self):
        return self.molsys.coords[self.index]


class Residue(MolecularSystemBase):
    def __init__(self, molsys, mask):
        resix = np.unique(molsys.resixs[mask])[0]
        self.mask = np.argwhere(np.isin(molsys.resixs, resix)).T[0]
        self.molsys = molsys

        self.resname = molsys.resnames[self.mask][0]
        self.resnum = molsys.resnums[self.mask][0]
        self.resid = self.resnum
        self.resindex = molsys.resixs[self.mask][0]
        self.segid = molsys.segids[self.mask][0]
        self.segindex = molsys.segixs[self.mask][0]
        self.chain = molsys.chains[self.mask][0]

    def __len__(self):
        return len(self.mask)

    def phi_selection(self):
        prev = self.atoms.resnums - 1
        prev = np.unique(prev[prev > 0])

        maskN_CA_C = self.mask[np.isin(self.names, ['N', 'CA', 'C'])]
        maskC = np.argwhere(np.isin(self.molsys.resnums, prev)).flatten()
        maskC = maskC[self.molsys.names[maskC] == 'C']
        mask_phi = np.concatenate((maskC, maskN_CA_C))
        sel = self.molsys.atoms[mask_phi]
        return sel if len(sel) == 4 else None

    def psi_selection(self):
        nex = self.atoms.resnums + 1
        nex = np.unique(nex[nex <= self.molsys.resnums.max()])

        maskN_CA_C = self.mask[np.isin(self.names, ['N', 'CA', 'C'])]
        maskN = np.argwhere(np.isin(self.molsys.resnums, nex)).flatten()
        maskN = maskN[self.molsys.names[maskN] == 'N']
        mask_psi = np.concatenate((maskN_CA_C, maskN))
        sel = self.molsys.atoms[mask_psi]
        return sel if len(sel) == 4 else None


class Segment(MolecularSystemBase):

    def __init__(self, molsys, mask):
        segix = np.unique(molsys.segixs[mask])[0]
        self.mask = np.argwhere(np.isin(molsys.segixs, segix)).T[0]
        self.molsys = molsys

        self.segid = molsys.segids[segix]
        self.chain = molsys.chains[segix]


def byres(mask, molsys):
    residues = np.unique(molsys.resnums[mask])
    mask = np.isin(molsys.resnums, residues)
    return mask


def unot(mask):
    return ~mask


def within(distance, mask, molsys):
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
