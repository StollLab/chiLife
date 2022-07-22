import functools
import operator
from functools import partial
from .protein_utils import sort_pdb
import numpy as np


class BaseSystem:

    def select_atoms(self, selstr):
        mask = process_statement(selstr, self.logic_keywords, self.protein_keywords)
        if hasattr(self, 'mask'):
            mask *= self.mask

        return AtomSelection(self.protein, mask)

    @property
    def atoms(self):
        return self.select_atoms("")

    @property
    def coords(self):
        return self.trajectory[self.protein.frame]

    @property
    def positions(self):
        return self.trajectory[self.protein.frame]

    @property
    def resindices(self):
        return np.unique(self.resixs)

    @property
    def segindices(self):
        return np.unique(self.segixs)

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
        return self.atypes


class Protein(BaseSystem):
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
    ):
        self.protein = self
        self.atomids = atomids.copy()
        self.names = names.copy()
        self.altlocs = altlocs.copy()
        self.resnames = resnames.copy()
        self.resnums = resnums.copy()
        self.chains = chains.copy()
        self.trajectory = trajectory.copy()
        self.occupancies = occupancies.copy()
        self.bs = bs.copy()
        self.segs = segs.copy()
        self.segids = self.chains
        self.atypes = atypes.copy()
        self.charges = charges.copy()


        self.n_atoms = len(self.atomids)
        self.n_residues = len(np.unique(self.resnums))
        self.n_chains = len(np.unique(self.chains))

        self.frame = 0

        self.ix = np.arange(self.n_atoms)
        self.mask = np.ones(self.n_atoms, dtype=bool)

        resix_borders = np.nonzero(np.r_[1, np.diff(self.resnums)[:-1]])
        resix_borders = np.append(resix_borders, [self.n_atoms])
        resixs = []
        for i in range(len(resix_borders) - 1):
            dif = resix_borders[i + 1] - resix_borders[i]
            resixs.append(np.ones(dif, dtype=int) * i)

        self.resixs = np.concatenate(resixs)
        self.segixs = np.array([ord(x)-65 for x in self.chains])

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
                              }



    @classmethod
    def from_pdb(cls, file_name):
        """reads a pdb file and returns a Protein object"""

        keys = ["skip", "atomids", "names", "altlocs", "resnames", "chains", "resnums",
                "skip", "coords", "occupancies", "bs", "segs", "atypes", "charges"]

        lines = sort_pdb(file_name)

        if isinstance(lines[0],  str):
            lines = [lines]

        PDB_data = [(line[:6].strip(), int(line[6:11]), line[12:16].strip(), line[16:17].strip(),
                     line[17:20].strip(), line[21:22].strip(), int(line[22:26]), line[26:27].strip(),
                     (float(line[30:38]), float(line[38:46]), float(line[46:54])), float(line[54:60]),
                     float(line[60:66]), line[72:73].strip(), line[76:78].strip(), line[78:80].strip())
                    for line in lines[0]]

        pdb_dict = {key: np.array(data) for key, data in zip(keys, zip(*PDB_data)) if key != "skip"}
        trajectory = [pdb_dict.pop('coords')]

        if len(lines) > 1:
            for struct in lines[1:]:
                frame_coords = [(line[30:38], line[38:46], line[46:54]) for line in struct]

                if len(frame_coords) != len(PDB_data):
                    raise ValueError('All models in a multistate PDB must have the same atoms')

                trajectory.append(frame_coords)

        pdb_dict['trajectory'] = np.array(trajectory, dtype=float)

        return cls(**pdb_dict)


def process_statement(statement, logickws, subjectkws):
    sub_statements = parse_paren(statement)
    unary_operators = [logickws['byres'], logickws['not']]

    mask = np.ones(subjectkws['_len'], dtype=bool)
    operation = logickws['and']
    next_operation = None

    for stat in sub_statements:
        cont = False
        stat_split = stat.split()

        if stat_split[0] in logickws:
            if len(stat_split) == 1:
                if logickws[stat_split[0]] in unary_operators:
                    _io = logickws[stat_split[0]]
                    def toperation(a, b, operation, _io): return operation(a, _io(b))
                    operation = functools.partial(toperation, operation=operation, _io=_io)
                else:
                    operation = logickws[stat_split[0]]

                continue

            elif operation in unary_operators:
                _io = logickws[stat_split[0]]
                operation = lambda a, b: operation(_io(a, b))

            elif logickws[stat_split[0]] in unary_operators:
                _io = logickws[stat_split[0]]
                def toperation(a, b, operation, _io): return operation(a, _io(b))
                operation = functools.partial(toperation, operation=operation, _io=_io)

            elif operation != None:
                raise ValueError('Cannot have two logical operators in a row')

            else:
                operation = logickws[stat_split[0]]

            stat_split = stat_split[1:]

        elif stat.startswith('('):
            cont = True
            sub_out = process_statement(stat, logickws, subjectkws)

        if stat_split[0] not in subjectkws and not cont:
            raise ValueError(f"{stat_split[0]} is not a valid keyword. Please start expressions with valid keywords")


        if stat_split[0] in subjectkws:
            finished = False
            sub_out = np.ones(subjectkws['_len'], dtype=bool)
            internal_operation = logickws['and']
            next_internal_operation = None
            while not finished:
                if stat_split[0] in subjectkws:
                    subject = subjectkws[stat_split.pop(0)]
                    values = []

                elif stat_split[0] in logickws:
                    if logickws[stat_split[0]] in unary_operators:
                        _io = logickws[stat_split[0]]
                        def toperation(a, b, operation, _io): return operation(a, _io(b))
                        internal_operation = functools.partial(toperation, operation=internal_operation, _io=_io)
                        stat_split = stat_split[1:]
                        continue
                    else:
                        raise ValueError('Cannot have two logical operators in succession unless the second one is '
                                        '`not`')
                else:
                    raise ValueError(f'{stat_split[0]} is not a valid keyword')

                for i, val in enumerate(stat_split):
                    if val in logickws:
                        next_internal_operation = logickws[val]
                        stat_split = stat_split[i+1:]
                        if stat_split == []:
                            next_operation = next_internal_operation
                            next_internal_operation = None
                        break
                    elif '-' in val:
                        start, stop = [int(x) for x in val.split('-')]
                        values += list(range(start, stop+1))
                    elif ':' in val:
                        start, stop = [int(x) for x in val.split(':')]
                        values += list(range(start, stop+1))
                    else:
                        values.append(val)
                else:
                    finished = True

                values = np.array(values, dtype=subject.dtype)
                tmp = np.isin(subject, values)
                sub_out = internal_operation(sub_out, tmp)


                internal_operation = next_internal_operation
                next_internal_operation = None
                if internal_operation is None:
                    finished = True

        if operation is None:
            print(stat)
            raise ValueError('Need an operation between two selections')

        mask = operation(mask, sub_out)
        operation = next_operation
        next_operation = None

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

    return results


class AtomSelection(BaseSystem):

    def __init__(self, protein, mask):
        self.protein = protein
        self.mask = mask
        prot_dict = {kw: self.protein.__dict__[kw][self.mask] for kw in
                     ("atomids", "names", "altlocs", "resnames", "chains", "segids", 'segixs',
                      "resnums", "occupancies", "bs", "atypes", "charges", 'ix', 'resixs')}
        prot_dict['trajectory'] = self.protein.trajectory[:, self.mask]
        self.__dict__.update(prot_dict)

    def __getitem__(self, item):
        self_args = np.argwhere(self.mask)
        new_args = self_args[item]
        new_mask = np.zeros_like(self.mask, dtype=bool)
        new_mask[new_args] = True
        return AtomSelection(self.protein, new_mask)


class ResidueSelection(BaseSystem):

    def __init__(self, protein, mask):

        resixs = np.unique(protein.resixs[mask])
        self.mask = np.isin(protein.resixs, resixs)
        self.protein = protein

        first_ix = np.nonzero(np.r_[1, np.diff(protein.resixs)[:-1]])[0]
        self.first_ix = np.array([ix for ix in first_ix if np.isin(protein.resixs[ix], protein.resixs[self.mask])], dtype=int)

        self.resnames = protein.resnames[self.first_ix].flatten()
        self.resnums = protein.resnums[self.first_ix].flatten()
        self.segids = protein.segids[self.first_ix].flatten()
        self.chains = protein.chains[self.first_ix].flatten()


    def __getitem__(self, item):
        resixs = np.unique(self.protein.resixs[self.mask])
        new_resixs = resixs[item]
        new_mask = np.isin(self.protein.resixs, new_resixs)
        return ResidueSelection(self.protein, new_mask)


    def phi_selection(self):
        prev = self.atoms.resnums-1
        prev = np.unique(prev[prev > 0])
        resnums = np.unique(self.atoms.resnums)

        return self.protein.select_atoms(f"(resnum {' '.join(str(r) for r in resnums)} and name N CA C) or "
                                         f"(resnum {' '.join(str(r) for r in prev)} and name C))")

    def psi_selection(self):
        nex = self.atoms.resnums + 1
        nex = np.unique(nex[nex <= self.protein.resnums.max()])
        resnums = np.unique(self.resnums)

        return self.protein.select_atoms(f"(resnum {' '.join(str(r) for r in resnums)} and name N CA C) or "
                                         f"(resnum {' '.join(str(r) for r in nex)} and name C))")


class SegmentSelection(BaseSystem):

    def __init__(self, protein, mask):
        segixs = np.unique(protein.segixs[mask])
        self.mask = np.isin(protein.segixs, segixs)
        self.protein = protein

        first_ix = np.nonzero(np.r_[1, np.diff(protein.segixs)[:-1]])
        self.first_ix = np.array([ix for ix in first_ix if protein.segixs[ix] in protein.segixs[self.mask]])

        self.segids = protein.segids[self.first_ix].flatten()
        self.chains = protein.chains[self.first_ix].flatten()

    def __getitem__(self, item):
        segixs = np.unique(self.protein.segixs[self.mask])
        new_segixs = segixs[item]
        new_mask = np.isin(self.protein.segixs, new_segixs)
        return ResidueSelection(self.protein, new_mask)


def byres(mask, protein):
    residues = np.unique(protein.resnums[mask])
    mask = np.isin(protein.resnums, residues)
    return mask


def unot(mask):
    return ~mask
