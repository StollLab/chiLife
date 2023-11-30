import logging
from copy import deepcopy
import numpy as np
import chilife
from .dRotamerEnsemble import dRotamerEnsemble


class dSpinLabel(dRotamerEnsemble):
    def __init__(self, label, sites, protein=None, chain=None, rotlib=None, **kwargs):
        super().__init__(label, sites, protein=protein, chain=chain, rotlib=rotlib, **kwargs)
        self.label = self.res

    def create_ensembles(self):
        self.SL1 = chilife.SpinLabel(self.res,
                                     self.site1,
                                     self.protein,
                                     self.chain,
                                     self.libA,
                                     **self.kwargs)

        self.SL2 = chilife.SpinLabel(self.res,
                                     self.site2,
                                     self.protein,
                                     self.chain,
                                     self.libB,
                                     **self.kwargs)

        self.RE1, self.RE2 = self.SL1, self.SL2

    def copy(self):
        new_copy = chilife.dSpinLabel(self.res, (self.site1, self.site2), chain=self.chain,
                                      protein=self.protein,
                                      rotlib={'csts': self.csts, 'libA': self.libA, 'libB': self.libB},
                                      minimize=False,
                                      eval_clash=False)
        for item in self.__dict__:
            if isinstance(self.__dict__[item], np.ndarray):
                new_copy.__dict__[item] = self.__dict__[item].copy()

            elif item == 'SL1':
                new_copy.__dict__[item] == self.__dict__[item].copy(rotlib=self.libA)
                new_copy.__dict__['RE1'] == new_copy.__dict__[item]

            elif item == 'SL2':
                new_copy.__dict__[item] == self.__dict__[item].copy(rotlib=self.libB)
                new_copy.__dict__['RE2'] = new_copy.__dict__[item]

            elif item in ('protein', 'RE1', 'RE2'):
                pass

            else:
                new_copy.__dict__[item] = deepcopy(self.__dict__[item])

        return new_copy

    @property
    def spin_atoms(self):
        return np.unique(np.concatenate((self.SL1.spin_atoms, self.SL2.spin_atoms)))

    @property
    def spin_idx(self):
        return np.argwhere(np.isin(self.atom_names, self.spin_atoms)).flatten()

    @property
    def spin_coords(self):
        return self.coords[:, self.spin_idx]

    @property
    def spin_centers(self):
        return np.average(self.spin_coords, axis=1, weights=self.spin_weights)

    @property
    def spin_weights(self):
        return self.SL1.spin_weights

    @property
    def spin_centroid(self):
        return np.average(self.spin_coords, weights=self.weights, axis=0)
