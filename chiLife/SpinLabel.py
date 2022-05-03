import pickle
from functools import partial
import numpy as np
from scipy.spatial.distance import cdist
from scipy.spatial import cKDTree
from .RotamerLibrary import RotamerLibrary
import chiLife


class SpinLabel(RotamerLibrary):
    """
    Base object for spin labeling experiments.

    Parameters
    ----------
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

    backbone_atoms = ['H', 'N', 'CA', 'HA', 'C', 'O']

    def __init__(self, label, site=1, protein=None, chain=None, **kwargs):
        """
        Create new SpinLabel object.

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
        # Overide RotamerLibrary default of not evaluating clashes
        kwargs.setdefault('eval_clash', True)
        super().__init__(label, site, protein=protein, chain=chain, **kwargs)

        self.label = label

        # Parse important indices
        self.spin_idx = np.argwhere(np.isin(self.atom_names, chiLife.SPIN_ATOMS[label[:3]]))

    @property
    def spin_coords(self):
        """get the spin coordinates of the rotamer library"""
        return self.coords[:, self.spin_idx, :].mean(axis=1)[:, 0]

    @property
    def spin_centroid(self):
        return np.average(self.spin_coords, weights=self.weights, axis=0)

    def protein_setup(self):
        self.protein = self.protein.select_atoms('not (byres name OH2 or resname HOH)')
        self._to_site()
        self.clash_ignore_idx = \
            self.protein.select_atoms(f'resid {self.site} and segid {self.chain}').ix

        self.resindex = self.protein.select_atoms(self.selstr).residues[0].resindex
        self.segindex = self.protein.select_atoms(self.selstr).residues[0].segindex

        if self.protein_tree is None:
            self.protein_tree = cKDTree(self.protein.atoms.positions)

        if self.eval_clash:
            self.evaluate()

    @classmethod
    def from_mmm(cls, label, site, protein=None, chain=None, **kwargs):
        """
        Create a SpinLabel object using the default MMM protocol with any modifications passed via kwargs
        """
        MMM_maxdist = {'R1M': 9.550856367392733,
                       'R7M': 9.757254987175209,
                       'V1M': 8.237071322458029,
                       'M1M': 8.985723827323680,
                       'I1M': 12.952083029729994}

        # Store the force field parameter set being used before creating the spin label
        curr_lj = chiLife.using_lj_param
        user_lj = kwargs.pop('lj_params', 'uff')
        # Set MMM defaults or user defined overrides
        chiLife.set_lj_params(user_lj)

        clash_radius = kwargs.pop('clash_radius', MMM_maxdist[label] + 4)
        superimposition_method = kwargs.pop('superimposition_method', 'mmm')
        clash_ori = kwargs.pop('clash_ori', 'CA')
        energy_func = kwargs.pop('energy_func', partial(chiLife.get_lj_energy, cap=np.inf))
        use_H = kwargs.pop('use_H', True)
        forgive = kwargs.pop('forgive', 0.5)


        # Calculate the SpinLabel
        SL = chiLife.SpinLabel(label, site, protein, chain, superimposition_method=superimposition_method,
                               clash_radius=clash_radius, clash_ori=clash_ori, energy_func=energy_func,
                               use_H=use_H, forgive=forgive, **kwargs)

        # restore the force field parameter set being used before creating the spin label
        chiLife.set_lj_params(curr_lj)
        return SL

    @classmethod
    def from_wizard(cls, label, site=1, protein=None, chain=None, to_find=200, to_try=10000, vdw=2.5, clashes=5, **kwargs):
        prelib = cls(label, site, protein, chain, eval_clash=False, **kwargs)
        if not kwargs.setdefault('use_prior', False):
            prelib.sigmas = np.array([])

        coords = np.zeros((to_find, len(prelib.atom_names), 3))
        internal_coords = []
        i = 0
        if protein is not None:
            protein_clash_idx = prelib.protein_tree.query_ball_point(prelib.centroid(), 19.0)
            protein_clash_idx = [idx for idx in protein_clash_idx if idx not in prelib.clash_ignore_idx]

        a, b = [list(x) for x in zip(*prelib.non_bonded)]
        for _ in range(np.rint(to_try/to_find).astype(int)):
            sample, _, internal_sample = prelib.sample(n=to_find, off_rotamer=True, return_dihedrals=True)

            # Evaluate internal clashes
            dist = np.linalg.norm(sample[:, a] - sample[:, b], axis=2)
            sidx = np.atleast_1d(np.squeeze(np.argwhere(np.all(dist > 2, axis=1))))
            if len(sidx) == 0:
                continue

            sample = sample[sidx, ...]
            internal_sample = internal_sample[sidx]
            if protein is not None:
                # Evaluate external clashes
                dist = cdist(sample[:, prelib.side_chain_idx].reshape(-1, 3), prelib.protein_tree.data[protein_clash_idx])
                dist = dist.reshape(len(sidx), -1)
                nclashes = np.sum(dist < vdw,  axis=1)
                sidx = np.atleast_1d(np.squeeze(np.argwhere(nclashes < clashes)))
            else:
                sidx = np.arange(len(sample))

            if len(sidx) == 0:
                continue

            if i + len(sidx) >= to_find:
                sidx = sidx[:to_find - i]

            coords[i:i+len(sidx)] = sample[sidx]
            internal_coords.append(internal_sample[sidx])
            i += len(sidx)

            if i == to_find:
                break

        coords = coords[coords.sum(axis=(1, 2)) != 0]
        prelib.internal_coords = np.concatenate(internal_coords) if len(internal_coords) > 0 else []
        prelib.dihedrals = np.array([IC.get_dihedral(1, prelib.dihedral_atoms) for IC in prelib.internal_coords])
        prelib.coords = coords
        prelib.weights = np.ones(len(coords))
        prelib.weights /= prelib.weights.sum()
        return prelib


class dSpinLabel:

    def __init__(self, label, site, increment, protein=None, chain=None, **kwargs):
        """

        """
        self.label = label
        self.res = label
        self.site = site
        self.site2 = site + increment
        self.increment = increment
        self.kwargs = kwargs

        self.protein = protein
        self.chain = chain if chain is not None else self.guess_chain()
        self.protein_tree = self.kwargs.setdefault('protein_tree', None)

        self.name = self.res
        if self.site is not None:
            self.name = f'{self.site}_{self.site2}_{self.res}'
        if self.chain is not None:
            self.name += f'_{self.chain}'

        self.selstr = f'resid {self.site} {self.site2} and segid {self.chain} and not altloc B'

        self.forgive = kwargs.setdefault('forgive', 1.)
        self.clash_radius = kwargs.setdefault('clash_radius', 14.)
        self._clash_ori_inp = kwargs.setdefault('clash_ori', 'cen')
        self.superimposition_method = kwargs.setdefault('superimposition_method', 'bisect').lower()
        self.dihedral_sigma = kwargs.setdefault('dihedral_sigma', 25)

        self.eval_clash = kwargs['eval_clash'] = False

        self.get_lib()

    def protein_setup(self):
        self.protein = self.protein.select_atoms('not (byres name OH2 or resname HOH)')
        self.clash_ignore_idx = \
            self.protein.select_atoms(f'resid {self.site} and segid {self.chain}').ix

        self.resindex = self.protein.select_atoms(self.selstr).residues[0].resindex
        self.segindex = self.protein.select_atoms(self.selstr).residues[0].segindex

        if self.protein_tree is None:
            self.protein_tree = cKDTree(self.protein.atoms.positions)

        if self.minimize:
            self._minimize()


    def guess_chain(self):
        if self.protein is None:
            chain = 'A'
        elif len(nr_segids := set(self.protein.segments.segids)) == 1:
            chain = self.protein.segments.segids[0]
        elif np.isin(self.protein.residues.resnums, self.site).sum() == 0:
            raise ValueError(f'Residue {self.site} is not present on the provided protein')
        elif np.isin(self.protein.residues.resnums, self.site).sum() == 1:
            chain = self.protein.select_atoms(f'resid {self.site}').segids[0]
        else:
            raise ValueError(f'Residue {self.site} is present on more than one chain. Please specify the desired chain')
        return chain

    def get_lib(self):
        PhiSel, PsiSel = None, None
        if self.protein is not None:
            # get site backbone information from protein structure
            sel_txt = f'resnum {self.site} and segid {self.chain}'
            PhiSel = self.protein.select_atoms(sel_txt).residues[0].phi_selection()
            PsiSel = self.protein.select_atoms(sel_txt).residues[0].psi_selection()

        # Default to helix backbone if none provided
        Phi = None if PhiSel is None else chiLife.get_dihedral(PhiSel.positions)
        Psi = None if PsiSel is None else chiLife.get_dihedral(PsiSel.positions)

        with open(chiLife.DATA_DIR/f'residue_internal_coords/{self.label}_ic.pkl', 'rb') as f:
            self.cst_idxs, self.csts = pickle.load(f)

        self.SL1 = chiLife.SpinLabel(self.label + 'i', self.site, self.protein, self.chain, **self.kwargs)
        self.SL2 = chiLife.SpinLabel(self.label + f'ip{self.increment}', self.site2, self.protein, self.chain, **self.kwargs)

    def save_pdb(self, name=None):
        if name is None:
            name = self.name + '.pdb'
        if not name.endswith('.pdb'):
            name += '.pdb'

        chiLife.save(name, self.SL1, self.SL2)

    # def _minimize(self):
    #
    #     def objective(dihedrals, ic1, ic2, opt):
    #         coords1 = ic1.set_dihedral(dihedrals[:len(ic1.dihedral_atoms)], 1, ic1.dihedral_atoms).to_cartesian()
    #         coords2 = ic2.set_dihedral(dihedrals[-len(ic2.dihedral_atoms):], 1, ic2.dihedral_atoms).to_cartesian()
    #
    #         distances = np.linal.norm(coords1[self.cst_idxs[0]] - coords2[self.cst_idxs[1]], axis=1)
    #         diff = distances - opt
    #         return diff@diff
    #
    #         coords = (coords - self.ic_ori) @ self.ic_mx
    #         coords = coords @ self.mx + self.ori
    #
    #         dist = cdist(coords[self.side_chain_idx], self.protein.atoms.positions[protein_clash_idx]).ravel()
    #         dist2 = np.linalg.norm(coords[a] - coords[b], axis=1)
    #         with np.errstate(divide='ignore'):
    #             E1 = self.energy_func(dist[dist < 10], rmin_ij[dist < 10], eps_ij[dist < 10]).sum()
    #             E2 = self.energy_func(dist2, ab_lj_radii, ab_lj_eps).sum()
    #
    #         E = E1 + E2
    #         return E
    #
    #     for i, IC in enumerate(self.internal_coords):
    #         d0 = IC.get_dihedral(1, self.dihedral_atoms)
    #         lb = d0 - np.deg2rad(self.dihedral_sigma) / 2
    #         ub = d0 + np.deg2rad(self.dihedral_sigma) / 2
    #         bounds = np.c_[lb, ub]
    #         xopt = opt.minimize(objective, x0=self._rdihedrals[i], args=IC, bounds=bounds)
    #
    #         self.coords[i] = (IC.to_cartesian()[self.ic_mask] - self.ic_ori) @ self.ic_mx @ self.mx + self.ori
    #         self.weights[i] = self.weights[i] * np.exp(-xopt.fun / (chiLife.GAS_CONST * 298))
    #
    #     self.weights /= self.weights.sum()
    #     self.trim()
    # # def protein_setup(self):