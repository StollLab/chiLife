import logging
import numpy as np
import chilife
from .dRotamerEnsemble import dRotamerEnsemble

class dSpinLabel(dRotamerEnsemble):
    def __init__(self, label, sites, protein=None, chain=None, rotlib=None, **kwargs):
        super().__init__(label, sites, protein=protein, chain=chain, rotlib=rotlib, **kwargs)
        self.label = self.res

    def create_ensembles(self):
        self.SL1 = chilife.SpinLabel(self.res,
                                     self.site,
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

        self.RL1, self.RL2 = self.SL1, self.SL2

        self.RL1.cst_idx = np.argwhere(np.isin(self.RL1.atom_names, self.csts)).flatten()
        self.RL2.cst_idx = np.argwhere(np.isin(self.RL2.atom_names, self.csts)).flatten()
        self.rl1mask = ~np.isin(self.RL2.atom_names, self.csts)
        self.rl2mask = ~np.isin(self.RL2.atom_names, self.csts)


    @property
    def spin_coords(self):
        s_coords = [SL.spin_coords.reshape(len(SL.weights), -1, 3)
                    for SL in self.sub_labels
                    if np.any(SL.spin_coords)]

        return np.concatenate(s_coords, axis=1)

    @property
    def spin_centers(self):
        return np.average(self.spin_coords, axis=1, weights=self.spin_weights)

    @property
    def spin_weights(self):
        return np.concatenate([SL.spin_weights for SL in self.sub_labels])

    @property
    def spin_centroid(self):
        return np.average(self.spin_coords, weights=self.weights, axis=0)