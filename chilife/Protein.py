from __future__ import annotations
import numbers
import operator
from functools import partial, update_wrapper
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
                     'bs', 'segs', 'segids', 'atypes', 'charges', 'ix', 'resixs', 'segixs', '_Atoms')

class MolecularSystem:
    """Base class for molecular systems"""

    def __getattr__(self, key):
        if 'protein' not in self.__dict__:
            self.__getattribute__(key)
        elif key not in self.__dict__['protein'].__dict__:
            self.protein.__getattribute__(key)
        elif key == 'trajectory':
            return self.protein.trajectory
        elif key in masked_properties:
            return np.squeeze(self.protein.__getattribute__(key)[self.mask])
        else:
            return self.protein.__getattribute__(key)

    def __setattr__(self, key, value):
        if key in ('protein', 'mask'):
            super(MolecularSystem, self).__setattr__(key, value)
        elif key not in self.__dict__['protein'].__dict__:
            super(MolecularSystem, self).__setattr__(key, value)
        elif key == 'trajectory':
            self.protein.__dict__['trajectory'] = value
        elif key in masked_properties:
            self.protein.__getattribute__(key)[self.mask] = value
        else:
            super(MolecularSystem, self).__setattr__(key, value)

    def select_atoms(self, selstr):
        mask = process_statement(selstr, self.logic_keywords, self.protein_keywords)
        if hasattr(self, 'mask'):
            t_mask = np.zeros(self.protein.n_atoms, dtype=bool)
            t_mask[self.mask] = True
            mask *= t_mask

        mask = np.argwhere(mask).T[0]
        return AtomSelection(self.protein, mask)

    @property
    def fname(self):
        return self.protein._fname

    @fname.setter
    def fname(self, val):
        self.protein.fname = val

    @property
    def atoms(self):
        return self.select_atoms("")

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
        return np.unique(self.protein.resixs[self.mask])

    @property
    def segindices(self):
        return np.unique(self.protein.segixs[self.mask])

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
        return ResidueSelection(self.protein, self.mask)

    @property
    def segments(self):
        return SegmentSelection(self.protein, self.mask)

    @property
    def logic_keywords(self):
        return self.protein._logic_keywords

    @property
    def protein_keywords(self):
        return self.protein._protein_keywords

    @property
    def types(self):
        return self.protein.atypes[self.mask]

    @types.setter
    def types(self, value):
        self.protein.atypes[self.mask] = value

    @property
    def universe(self):
        return self.protein

    def copy(self):
        p2 = self.protein.copy()
        return AtomSelection(p2, self.mask.copy())

    def __iter__(self):
        for idx in self.mask:
            yield self.protein.atoms[idx]


class Protein(MolecularSystem):
    """Protein object"""
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
            name: str = 'Noname_Protein'

    ):
        self.protein = self
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

        self._Atoms = np.array([Atom(self, i) for i in range(self.n_atoms)])

        self._protein_keywords = {'id': self.atomids,
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

                                  'protein': np.isin(self.resnames, list(chilife.SUPPORTED_RESIDUES)),
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
                                'byres': partial(byres, protein=self.protein),
                                'within': update_wrapper(partial(within, protein=self.protein), within),
                                'around': update_wrapper(partial(within, protein=self.protein), within)}

    @classmethod
    def from_pdb(cls, file_name):
        """reads a pdb file and returns a Protein object"""

        keys = ["skip", "atomids", "names", "altlocs", "resnames", "chains", "resnums",
                "skip", "coords", "occupancies", "bs", "segs", "atypes", "charges"]

        lines = sort_pdb(file_name)

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
                    ) -> Protein:
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
        anames = atomsel.names
        atypes = atomsel.types
        resnames = atomsel.resnames
        resnums = atomsel.resnums
        ridx_map = {num: i for i, num in enumerate(np.unique(resnums))}
        resindices = np.array([ridx_map[num] for num in resnums ])
        segids = atomsel.segids
        sidx_map = {num: i for i, num in enumerate(np.unique(segids))}
        segindices = np.array([ridx_map[num] for num in resnums])

        if frames is None:
            trajectory = np.array([atomsel.positions for ts in atomsel.universe.trajectory])
        else:
            trajectory = []
            for frame in frames:
                atomsel.universe.trajectory[frame]
                trajectory.append(atomsel.positions)

            trajectory = np.array(trajectory)

        return cls.from_arrays(anames, atypes, resnames, resindices, resnums, segindices, segids, trajectory)

    def copy(self):
        return Protein(
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


class Trajectory:

    def __init__(self, coordinates, protein, timestep=1):
        self.protein = protein
        self.timestep = timestep
        self.coords = coordinates
        self.coordinates_array = coordinates
        self._frame = 0
        self.time = np.arange(0, len(self.coords)) * self.timestep

    def load_new(self, coordinates):
        self.coords = coordinates
        self.coordinates_array = coordinates

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


class AtomSelection(MolecularSystem):

    def __new__(cls, *args):
        if len(args) == 2:
            protein, mask = args
        else:
            return object.__new__(cls)

        if isinstance(mask, int):
            return Atom(protein, mask)
        else:
            return object.__new__(cls)

    def __init__(self, protein, mask):
        self.protein = protein
        self.mask = mask

    def __getitem__(self, item):

        if np.issubdtype(type(item), np.integer):
            relidx = self.protein.ix[self.mask][item]
            return Atom(self.protein, relidx)

        if isinstance(item, slice):
            return AtomSelection(self.protein, self.mask[item])

        elif isinstance(item, (np.ndarray, list, tuple)):
            item = np.asarray(item)
            if len(item) == 1 or item.sum() == 1:
                return Atom(self.protein, self.mask[item])
            elif item.dtype == bool:
                item = np.argwhere(item).T[0]
                return AtomSelection(self.protein, item)
            else:
                return AtomSelection(self.protein, self.mask[item])

        elif hasattr(item, '__iter__'):
            if all([np.issubdtype(type(x), int) for x in item]):
                return AtomSelection(self.protein, self.mask[item])

        raise TypeError('Only integer, slice type, and boolean mask arguments are supported at this time')

    def __len__(self):
        return len(self.mask)


class ResidueSelection(MolecularSystem):

    def __init__(self, protein, mask):

        resixs = np.unique(protein.resixs[mask])
        self.mask = np.isin(protein.resixs, resixs)
        self.protein = protein

        first_ix = np.nonzero(np.r_[1, np.diff(protein.resixs)[:-1]])[0]
        self.first_ix = np.array([ix for ix in first_ix if
                                  np.isin(protein.resixs[ix], protein.resixs[self.mask])],
                                 dtype=int)

        self.resnames = protein.resnames[self.first_ix].flatten()
        self.resnums = protein.resnums[self.first_ix].flatten()
        self.segids = protein.segids[self.first_ix].flatten()
        self.chains = protein.chains[self.first_ix].flatten()

    def __setattr__(self, key, value):
        self.__dict__[key] = value

    def __getitem__(self, item):
        resixs = np.unique(self.protein.resixs[self.mask])
        new_resixs = resixs[item]
        new_mask = np.argwhere(np.isin(self.protein.resixs, new_resixs))

        if np.issubdtype(type(item), int):
            return Residue(self.protein, new_mask)
        elif isinstance(item, slice):
            return ResidueSelection(self.protein, new_mask)
        elif hasattr(item, '__iter__'):
            if all([np.issubdtype(type(x), int) for x in item]):
                return ResidueSelection(self.protein, new_mask)

        raise TypeError('Only integer and slice type arguments are supported at this time')

    def __len__(self):
        return len(self.first_ix)

    def __iter__(self):
        for resnum in self.resnums:
            mask = self.protein.resnums == resnum
            yield Residue(self.protein, mask)


class SegmentSelection(MolecularSystem):

    def __init__(self, protein, mask):
        seg_ixs = np.unique(protein.segixs[mask])
        self.mask = np.argwhere(np.isin(protein.segixs, seg_ixs)).T[0]
        self.protein = protein

        first_ix = np.nonzero(np.r_[1, np.diff(protein.segixs)[:-1]])
        self.first_ix = np.array([ix for ix in first_ix if protein.segixs[ix] in protein.segixs[self.mask]], dtype=int)

        self.segids = protein.segids[self.first_ix].flatten()
        self.chains = protein.chains[self.first_ix].flatten()

    def __setattr__(self, key, value):
        self.__dict__[key] = value

    def __getitem__(self, item):
        segixs = np.unique(self.protein.segixs[self.mask])
        new_segixs = segixs[item]
        new_mask = np.argwhere(np.isin(self.protein.segixs, new_segixs))

        if np.issubdtype(type(item), int):
            return Segment(self.protein, new_mask)
        elif isinstance(item, slice):
            return SegmentSelection(self.protein, new_mask)
        elif hasattr(item, '__iter__'):
            if all([np.issubdtype(type(x), int) for x in item]):
                return SegmentSelection(self.protein, new_mask)

        raise TypeError('Only integer and slice type arguments are supported at this time')

    def __len__(self):
        return len(self.first_ix)


class Atom(MolecularSystem):
    def __init__(self, protein, mask):
        self.__dict__['protein'] = protein
        self.index = mask
        self.mask = mask


        self.name = protein.names[self.index]
        self.altLoc = protein.altlocs[self.index]
        self.atype = protein.types[self.index]
        self.type = self.atype

        self.resn = protein.resnames[self.index]
        self.resname = self.resn
        self.resi = protein.resnums[self.index]
        self.resid = self.resi
        self.resnum = self.resi
        self.chain = protein.segids[self.index]
        self.segid = protein.chains[self.index]

    @property
    def position(self):
        return self.protein.coords[self.index]


class Residue(MolecularSystem):
    def __init__(self, protein, mask):
        resix = np.unique(protein.resixs[mask])[0]
        self.mask = np.argwhere(np.isin(protein.resixs, resix)).T[0]
        self.protein = protein

        self.resname = protein.resnames[self.mask][0]
        self.resnum = protein.resnums[self.mask][0]
        self.resid = self.resnum
        self.segid = protein.segids[self.mask][0]
        self.segindex = protein.segixs[self.mask][0]
        self.chain = protein.chains[self.mask][0]

    def __len__(self):
        return len(self.mask)

    def phi_selection(self):
        prev = self.atoms.resnums - 1
        prev = np.unique(prev[prev > 0])
        resnums = np.unique(self.atoms.resnums)
        sel = self.protein.select_atoms(f"(resnum {' '.join(str(r) for r in resnums)} and name N CA C) or "
                                        f"(resnum {' '.join(str(r) for r in prev)} and name C)")
        return sel if len(sel) == 4 else None

    def psi_selection(self):
        nex = self.atoms.resnums + 1
        nex = np.unique(nex[nex <= self.protein.resnums.max()])
        resnums = np.unique(self.resnums)
        sel = self.protein.select_atoms(f"(resnum {' '.join(str(r) for r in resnums)} and name N CA C) or "
                                        f"(resnum {' '.join(str(r) for r in nex)} and name N)")
        return sel if len(sel) == 4 else None


class Segment(MolecularSystem):

    def __init__(self, protein, mask):
        segix = np.unique(protein.segixs[mask])[0]
        self.mask = np.argwhere(np.isin(protein.segixs, segix)).T[0]
        self.protein = protein

        self.segid = protein.segids[segix]
        self.chain = protein.chains[segix]


def byres(mask, protein):
    residues = np.unique(protein.resnums[mask])
    mask = np.isin(protein.resnums, residues)
    return mask


def unot(mask):
    return ~mask


def within(distance, mask, protein):
    distance = float(distance)
    tree1 = cKDTree(protein.coords)
    tree2 = cKDTree(protein.protein.coords[mask])
    results = np.concatenate(tree2.query_ball_tree(tree1, distance))
    results = np.unique(results)
    out_mask = np.zeros_like(mask, dtype=bool)
    out_mask[results] = True
    out_mask = out_mask * ~mask
    return out_mask


within.nargs = 1
