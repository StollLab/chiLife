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

    @property
    def centroid(self):
        return self.coords.mean(axis=(0, 1))

    @property
    def clash_ori(self):

        if isinstance(self._clash_ori_inp, (np.ndarray, list)):
            if len(self._clash_ori_inp) == 3:
                return self._clash_ori_inp

        elif isinstance(self._clash_ori_inp, str):
            if self._clash_ori_inp in ["cen", "centroid"]:
                return self.centroid

            elif (ori_name := self._clash_ori_inp.upper()) in self.atom_names:
                return np.squeeze(self.coords[0][ori_name == self.atom_names])

        else:
            raise ValueError(
                f"Unrecognized clash_ori option {self._clash_ori_inp}. Please specify a 3D vector, an "
                f"atom name or `centroid`"
            )

        return self._clash_ori

    @clash_ori.setter
    def clash_ori(self, inp):
        self._clash_ori_inp = inp

    @property
    def side_chain_idx(self):
        return np.concatenate(
            [
                self.SL1.side_chain_idx,
                self.SL2.side_chain_idx + len(self.SL1.atom_names),
            ]
        )

    @property
    def rmin2(self):
        return np.concatenate([self.SL1.rmin2, self.SL2.rmin2])

    @property
    def eps(self):
        return np.concatenate([self.SL1.eps, self.SL2.eps])

    def trim_rotamers(self):
        self.SL1.trim_rotamers()
        self.SL2.trim_rotamers()
