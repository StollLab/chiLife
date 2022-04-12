from itertools import groupby
import numpy as np

class Protein:

    def __init__(self, coords, atom_names, atom_types, atom_residues, atom_chain, res_names):
        self.coords = coords
        self.atom_names = atom_names
        self.atom_types = atom_types
        self.atom_residues = atom_residues
        self.atom_chain = atom_chain
        self.res_names = res_names

    @classmethod
    def from_pdb(cls, file_name):
        atom_names = None
        # Read file
        with open(file_name, 'r') as f:
            lines = f.readlines()

        # group by model
        models = groupby(lines, lambda x: x.startswith('MODEL'))

        coords = []
        # loop over models and extract information
        for key, model in models:

            model = [line.strip().split() for line in model if line.startswith('ATOM')]

            for line in model:
                if len(line) != 13:
                    line.insert(3, line[2][4:])
                    line[2] = line[2][:3]

            model = np.array(model)

            if model.size == 0:
                continue

            coords.append(model[:, 6:9].astype(float))

            if atom_names is None:
                atom_names = model[:, 2]
                atom_types = model[:, 12]
                atom_resnums = model[:, 5].astype(int)
                res_names = model[:, 3]
                atom_chain = model[:, 4]
            else:
                assert np.all(atom_names == model[:, 2])
                assert np.all(atom_types == model[:, 12])
                assert np.all(atom_resnums == model[:, 5].astype(int))
                assert np.all(res_names == model[:, 3])
                assert np.all(atom_chain == model[:, 4])
        coords = np.array(coords)

        return cls(coords, atom_names, atom_types, atom_resnums, atom_chain, res_names)

    # def get_coords(self, selstring):
    #     idxs = _parse_selstring()
    #
    # def _parse_selstring(self, selstring):
    #     sel = selstring.split()
    #
    #     # Group logical operators
    #     groupby(sel, lambda x: x in ['resnum, resname, name'])

