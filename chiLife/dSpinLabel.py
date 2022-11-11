import pickle
import logging

import numpy as np
from scipy.spatial import cKDTree
import scipy.optimize as opt

import chiLife


class dSpinLabel:
    def __init__(self, label, sites, protein=None, chain=None, **kwargs):
        """ """
        self.label = label
        self.res = label
        self.site, self.site2 = sorted(sites)
        self.increment = self.site2 - self.site
        self.kwargs = kwargs

        self.protein = protein
        self.chain = chain if chain is not None else self.guess_chain()
        self.protein_tree = self.kwargs.setdefault("protein_tree", None)

        self.name = self.res
        if self.site is not None:
            self.name = f"{self.site}_{self.site2}_{self.res}"
        if self.chain is not None:
            self.name += f"_{self.chain}"

        self.selstr = (
            f"resid {self.site} {self.site2} and segid {self.chain} and not altloc B"
        )

        self.forgive = kwargs.setdefault("forgive", 1.0)
        self.clash_radius = kwargs.setdefault("clash_radius", 14.0)
        self._clash_ori_inp = kwargs.setdefault("clash_ori", "cen")
        self.restraint_weight = kwargs.pop("restraint_weight") if "restraint_weight" in kwargs else 200     # kcal/mol/A^2
        self.superimposition_method = kwargs.setdefault(
            "superimposition_method", "bisect"
        ).lower()
        self.dihedral_sigmas = kwargs.setdefault("dihedral_sigmas", 25)
        self.minimize = kwargs.pop("minimize", True)
        self.eval_clash = kwargs.pop("eval_clash", True)
        self.energy_func = kwargs.setdefault("energy_func", chiLife.get_lj_rep)
        self.temp = kwargs.setdefault("temp", 298)
        self.get_lib()
        self.protein_setup()
        self.sub_labels = (self.SL1, self.SL2)

    def protein_setup(self):
        self.protein = self.protein.select_atoms("not (byres name OH2 or resname HOH)")
        self.clash_ignore_idx = self.protein.select_atoms(
            f"resid {self.site} {self.site2} and segid {self.chain}"
        ).ix

        self.resindex = self.protein.select_atoms(self.selstr).residues[0].resindex
        self.segindex = self.protein.select_atoms(self.selstr).residues[0].segindex

        if self.protein_tree is None:
            self.protein_tree = cKDTree(self.protein.atoms.positions)

        protein_clash_idx = self.protein_tree.query_ball_point(
            self.clash_ori, self.clash_radius
        )
        self.protein_clash_idx = [
            idx for idx in protein_clash_idx if idx not in self.clash_ignore_idx
        ]

        if self.minimize:
            self._minimize()

        if self.eval_clash:
            self.evaluate()

    def guess_chain(self):
        if self.protein is None:
            chain = "A"
        elif len(nr_segids := set(self.protein.segments.segids)) == 1:
            chain = self.protein.segments.segids[0]
        elif np.isin(self.protein.residues.resnums, self.site).sum() == 0:
            raise ValueError(
                f"Residue {self.site} is not present on the provided protein"
            )
        elif np.isin(self.protein.residues.resnums, self.site).sum() == 1:
            chain = self.protein.select_atoms(f"resid {self.site}").segids[0]
        else:
            raise ValueError(
                f"Residue {self.site} is present on more than one chain. Please specify the desired chain"
            )
        return chain

    def get_lib(self):
        PhiSel, PsiSel = None, None
        if self.protein is not None:
            # get site backbone information from protein structure
            sel_txt = f"resnum {self.site} and segid {self.chain}"
            PhiSel = self.protein.select_atoms(sel_txt).residues[0].phi_selection()
            PsiSel = self.protein.select_atoms(sel_txt).residues[0].psi_selection()

        # Default to helix backbone if none provided
        Phi = (
            None
            if PhiSel is None
            else np.rad2deg(chiLife.get_dihedral(PhiSel.positions))
        )
        Psi = (
            None
            if PsiSel is None
            else np.rad2deg(chiLife.get_dihedral(PsiSel.positions))
        )

        with open(
                chiLife.RL_DIR / f"residue_internal_coords/{self.label}ip{self.increment}C_ic.pkl", "rb"
        ) as f:
            self.cst_idxs, self.csts = pickle.load(f)

        self.kwargs["eval_clash"] = False

        self.SL1 = chiLife.SpinLabel(self.label + f"ip{self.increment}A",
                                     self.site,
                                     self.protein,
                                     self.chain,
                                     **self.kwargs)

        self.SL2 = chiLife.SpinLabel(self.label + f"ip{self.increment}B",
                                     self.site2,
                                     self.protein,
                                     self.chain,
                                     **self.kwargs)

    def save_pdb(self, name=None):
        if name is None:
            name = self.name + ".pdb"
        if not name.endswith(".pdb"):
            name += ".pdb"

        chiLife.save(name, self.SL1, self.SL2)

    def _minimize(self):
        def objective(dihedrals, ic1, ic2, opt):
            coords1 = ic1.set_dihedral(
                dihedrals[: len(self.SL1.dihedral_atoms)], 1, self.SL1.dihedral_atoms
            ).to_cartesian()
            coords2 = ic2.set_dihedral(
                dihedrals[-len(self.SL2.dihedral_atoms) :], 1, self.SL2.dihedral_atoms
            ).to_cartesian()

            distances = np.linalg.norm(
                coords1[self.cst_idxs[:, 0]] - coords2[self.cst_idxs[:, 1]], axis=1
            )
            diff = distances - opt
            return diff @ diff

        scores = np.empty_like(self.weights)
        for i, (ic1, ic2) in enumerate(
            zip(self.SL1.internal_coords, self.SL2.internal_coords)
        ):
            d0 = np.concatenate(
                [
                    ic1.get_dihedral(1, self.SL1.dihedral_atoms),
                    ic2.get_dihedral(1, self.SL2.dihedral_atoms),
                ]
            )
            lb = [-np.pi] * len(d0)  # d0 - np.deg2rad(40)  #
            ub = [np.pi] * len(d0)  # d0 + np.deg2rad(40) #
            bounds = np.c_[lb, ub]
            xopt = opt.minimize(
                objective, x0=d0, args=(ic1, ic2, self.csts[i]), bounds=bounds
            )
            self.SL1._coords[i] = ic1.coords[self.SL1.H_mask]
            self.SL2._coords[i] = ic2.coords[self.SL2.H_mask]
            scores[i] = xopt.fun

        self.SL1.backbone_to_site()
        self.SL2.backbone_to_site()

        scores /= len(self.cst_idxs)
        scores -= scores.min()

        self.weights *= np.exp(-scores * self.restraint_weight /(chiLife.GAS_CONST * self.temp) / np.exp(-scores).sum())
        self.weights /= self.weights.sum()

    @property
    def weights(self):
        return self.SL1.weights

    @weights.setter
    def weights(self, value):
        self.SL1.weights = value
        self.SL2.weights = value

    @property
    def coords(self):
        return np.concatenate([self.SL1._coords, self.SL2._coords], axis=1)

    @coords.setter
    def coords(self, value):
        if value.shape[1] != self.SL1._coords.shape[1] + self.SL2._coords.shape[1]:
            raise ValueError(
                f"The provided coordinates do not match the number of atoms of this label ({self.label})"
            )

        self.SL1._coords = value[:, : self.SL1._coords.shape[1]]
        self.SL2._coords = value[:, -self.SL2._coords.shape[1]:]

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

    def trim(self):
        self.SL1.trim()
        self.SL2.trim()

    def evaluate(self):

        if self.protein_tree is None:
            self.protein_tree = cKDTree(self.protein.atoms.positions)

        rotamer_energies = self.energy_func(self.protein, self)
        rotamer_probabilities = np.exp(
            -rotamer_energies / (chiLife.GAS_CONST * self.temp)
        )

        self.weights, self.partition = chiLife.reweight_rotamers(
            rotamer_probabilities, self.weights, return_partition=True
        )
        logging.info(f"Relative partition function: {self.partition:.3}")
        self.trim()
