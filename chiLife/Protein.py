import operator
from itertools import groupby
import parse
import numpy as np

class Protein:
    def __init__(
        self,
        atomids: np.ndarray,
        names: np.ndarray,
        altlocs: np.ndarray,
        resnames: np.ndarray,
        resnums: np.ndarray,
        chains: np.ndarray,
        coords: np.ndarray,
        occupancies: np.ndarray,
        bs: np.ndarray,
        atypes: np.ndarray,
        charges: np.ndarray,
    ):

        self.atomids = atomids.copy()
        self.names = names.copy()
        self.altlocs = altlocs.copy()
        self.resnames = resnames.copy()
        self.resnums = resnums.copy()
        self.chains = chains.copy()
        self.coords = coords.copy()
        self.occupancies = occupancies.copy()
        self.bs = bs.copy()
        self.atypes = atypes.copy()
        self.charges = charges.copy()

        self.n_atoms = len(self.atomids)
        self.n_residues = len(np.unique(self.resnums))
        self.n_chains = len(np.unique(self.chains))

        self.protein_keywords = {'id': self.atomids,
                                 'name': self.names,
                                 'altloc': self.altlocs,
                                 'resname': self.resnames,
                                 'resnum': self.resnums,
                                 'chain': self.chains,
                                 'occupancies': self.occupancies,
                                 'b': self.bs,
                                 'type': self.atypes,
                                 'charges': self.charges,
                                 '_len': self.n_atoms}



    def select_atoms(self, selstr):
        mask = process_statement(selstr, logic_keywords, self.protein_keywords)
        return AtomSelection(self, mask)

    @classmethod
    def from_pdb(cls, file_name):
        """reads a pdb file and returns a Protein object"""

        fmt_str = "{:6.6S}{:5.5d} {:4.4S}{:1.1}{:3.3} {:1.1S}{:4.4d}{:1.1}   " \
                  "{:8.8f}{:8.8f}{:8.8f}{:6.6f}{:6.6f}          " \
                  "{:2.2S}{:2.2}\n"

        keys = ["skip", "atomids", "names", "altlocs", "resnames", "chains", "resnums",
                "skip", "x", "y", "z", "occupancies", "bs", "atypes", "charges"]

        with open(file_name, 'r') as f:
            lines = f.readlines()

        PDB_data = [parse.parse(fmt_str, line) for line in lines if line.startswith(('ATOM', 'HETATM'))]
        pdb_dict = {key: np.array(data) for key, data in zip(keys, zip(*PDB_data)) if key != "skip"}

        pdb_dict['coords'] = np.vstack([pdb_dict.pop(xyz) for xyz in 'xyz']).T

        return cls(**pdb_dict)



logic_keywords = {'and': operator.mul,
                  'or': operator.add,
                  'not': operator.sub,
                  '!': operator.sub,
                  '<': operator.lt,
                  '>': operator.gt,
                  '==': operator.eq,
                  '<=': operator.le,
                  '>=': operator.ge,
                  "!=": operator.ne}

def process_statement(statement, logickws, subjectkws):
    sub_statements = parse_paren(statement)

    mask = np.ones(subjectkws['_len'], dtype=bool)
    operation = logickws['and']
    next_operation = None
    for stat in sub_statements:
        cont = False
        stat_split = stat.split()

        if stat_split[0] in logickws:

            if operation != None:
                raise ValueError('Cannot have two logical operators in a row')

            operation = logickws[stat_split[0]]
            if len(stat_split) == 1:
                continue

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
                    if logickws[stat_split[0]] is operator.sub:
                        dummy = internal_operation
                        internal_operation = lambda a, b: dummy(a, ~b)
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


class AtomSelection:

    def __init__(self, protein, mask):
        self.protein = protein
        self.mask = mask
        prot_dict = {kw: self.protein.__dict__[kw][self.mask] for kw in
                     ("atomids", "names", "altlocs", "resnames", "chains", "resnums",
                      "coords", "occupancies", "bs", "atypes", "charges")}
        self.__dict__.update(prot_dict)

    def select_atoms(self, selstr):
        output = process_statement(selstr, logic_keywords, self.protein_keywords)
        return output