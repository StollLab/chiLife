import pickle
import logging
from copy import deepcopy
from functools import partial
import numpy as np
from scipy.spatial.distance import cdist
from scipy.spatial import cKDTree
import scipy.optimize as opt
from .RotamerLibrary import RotamerLibrary
import chiLife


class SpinLabel(RotamerLibrary):
    """
    A RotamerLibrary made from a side chain with one or more unpaired electrons.

    :param label: str
        Name of desired spin label, e.g. R1A.

    :param site: int
        Protein residue number to attach spin label to.

    :param chain: str
        Protein chain identifier to attach spin label to.

    :param protein: MDAnalysis.Universe, MDAnalysis.AtomGroup
        Object containing all protein information (coords, atom types, etc.)

    :param protein_tree: cKDtree
        k-dimensional tree object associated with the protein coordinates.
    """

    backbone_atoms = ["H", "N", "CA", "HA", "C", "O"]

    def __init__(self, label, site=1, protein=None, chain=None, **kwargs):

        # Overide RotamerLibrary default of not evaluating clashes
        kwargs.setdefault("eval_clash", True)
        super().__init__(label, site, protein=protein, chain=chain, **kwargs)

        self.label = label

        # Parse important indices
        self.spin_idx = np.argwhere(np.isin(self.atom_names, chiLife.SPIN_ATOMS[label[:3]]))

    @property
    def spin_coords(self):
        """get the spin coordinates of the rotamer library"""
        return self._coords[:, self.spin_idx, :].mean(axis=1)[:, 0]

    @property
    def spin_centroid(self):
        """Average location of all the label's `spin_coords` weighted based off of the rotamer weights"""
        return np.average(self.spin_coords, weights=self.weights, axis=0)

    def protein_setup(self):
        self.protein = self.protein.select_atoms("not (byres name OH2 or resname HOH)")
        self.to_site()
        self.backbone_to_site()
        self.clash_ignore_idx = self.protein.select_atoms(
            f"resid {self.site} and segid {self.chain}"
        ).ix

        self.resindex = self.protein.select_atoms(self.selstr).resindices[0]
        self.segindex = self.protein.select_atoms(self.selstr).segindices[0]

        if self.protein_tree is None:
            self.protein_tree = cKDTree(self.protein.atoms.positions)

        protein_clash_idx = self.protein_tree.query_ball_point(
            self.clash_ori, self.clash_radius
        )
        self.protein_clash_idx = [
            idx for idx in protein_clash_idx if idx not in self.clash_ignore_idx
        ]

        if self.eval_clash:
            self.evaluate()

    @classmethod
    def from_mmm(cls, label, site, protein=None, chain=None, **kwargs):
        """Create a SpinLabel object using the default MMM protocol with any modifications passed via kwargs"""

        MMM_maxdist = {
            "R1M": 9.550856367392733,
            "R7M": 9.757254987175209,
            "V1M": 8.237071322458029,
            "M1M": 8.985723827323680,
            "I1M": 12.952083029729994,
        }

        # Store the force field parameter set being used before creating the spin label
        curr_lj = chiLife.using_lj_param
        user_lj = kwargs.pop("lj_params", "uff")
        # Set MMM defaults or user defined overrides
        chiLife.set_lj_params(user_lj)

        clash_radius = kwargs.pop("clash_radius", MMM_maxdist[label] + 4)
        superimposition_method = kwargs.pop("superimposition_method", "mmm")
        clash_ori = kwargs.pop("clash_ori", "CA")
        energy_func = kwargs.pop(
            "energy_func", partial(chiLife.get_lj_energy, cap=np.inf)
        )
        use_H = kwargs.pop("use_H", True)
        forgive = kwargs.pop("forgive", 0.5)

        # Calculate the SpinLabel
        SL = chiLife.SpinLabel(
            label,
            site,
            protein,
            chain,
            superimposition_method=superimposition_method,
            clash_radius=clash_radius,
            clash_ori=clash_ori,
            energy_func=energy_func,
            use_H=use_H,
            forgive=forgive,
            **kwargs,
        )

        # restore the force field parameter set being used before creating the spin label
        chiLife.set_lj_params(curr_lj)
        return SL

    @classmethod
    def from_wizard(
        cls,
        label,
        site=1,
        protein=None,
        chain=None,
        to_find=200,
        to_try=10000,
        vdw=2.5,
        clashes=5,
        **kwargs,
    ):
        """Create a SpinLabel object using the default MTSSLWizard protocol with any modifications passed via kwargs"""

        prelib = cls(label, site, protein, chain, eval_clash=False, **kwargs)
        prelib.sigmas = np.array([])

        coords = np.zeros((to_find, len(prelib.atom_names), 3))
        internal_coords = []
        i = 0
        if protein is not None:
            protein_clash_idx = prelib.protein_tree.query_ball_point(
                prelib.centroid(), 19.0
            )
            protein_clash_idx = [
                idx for idx in protein_clash_idx if idx not in prelib.clash_ignore_idx
            ]

        a, b = [list(x) for x in zip(*prelib.non_bonded)]
        for _ in range(np.rint(to_try / to_find).astype(int)):
            sample, _, internal_sample = prelib.sample(
                n=to_find, off_rotamer=True, return_dihedrals=True
            )

            # Evaluate internal clashes
            dist = np.linalg.norm(sample[:, a] - sample[:, b], axis=2)
            sidx = np.atleast_1d(np.squeeze(np.argwhere(np.all(dist > 2, axis=1))))
            if len(sidx) == 0:
                continue

            sample = sample[sidx, ...]
            internal_sample = internal_sample[sidx]
            if protein is not None:
                # Evaluate external clashes
                dist = cdist(
                    sample[:, prelib.side_chain_idx].reshape(-1, 3),
                    prelib.protein_tree.data[protein_clash_idx],
                )
                dist = dist.reshape(len(sidx), -1)
                nclashes = np.sum(dist < vdw, axis=1)
                sidx = np.atleast_1d(np.squeeze(np.argwhere(nclashes < clashes)))
            else:
                sidx = np.arange(len(sample))

            if len(sidx) == 0:
                continue

            if i + len(sidx) >= to_find:
                sidx = sidx[: to_find - i]

            coords[i: i + len(sidx)] = sample[sidx]
            internal_coords.append(internal_sample[sidx])
            i += len(sidx)

            if i == to_find:
                break

        coords = coords[coords.sum(axis=(1, 2)) != 0]
        prelib.internal_coords = (
            np.concatenate(internal_coords) if len(internal_coords) > 0 else []
        )
        prelib._dihedrals = np.rad2deg(
            [IC.get_dihedral(1, prelib.dihedral_atoms) for IC in prelib.internal_coords]
        )
        prelib._coords = coords
        prelib.weights = np.ones(len(coords))
        prelib.weights /= prelib.weights.sum()
        return prelib

    def __deepcopy__(self, memodict={}):
        new_copy = chiLife.SpinLabel(self.res, self.site)
        for item in self.__dict__:
            if item != "protein":
                new_copy.__dict__[item] = deepcopy(self.__dict__[item])
            elif self.__dict__[item] is None:
                new_copy.protein = None
            else:
                new_copy.protein = self.protein.copy()
        return new_copy


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

        self.weights *= np.exp(-scores) / np.exp(-scores).sum()
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
        sc_matrix = [
            SL.spin_coords
            for SL in self.sub_labels
            if not np.any(np.isnan(SL.spin_coords))
        ]
        return np.sum(sc_matrix, axis=0) / len(sc_matrix)

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
