import logging
import multiprocessing as mp
import numpy as np
from scipy.spatial.distance import cdist
from scipy.spatial import cKDTree
import MDAnalysis as mda
from .RotamerLibrary import RotamerLibrary
import ProEPR


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

    def __init__(self, label, site=1, chain='A', protein=None, protein_tree=None, **kwargs):
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

        super().__init__(label, site, chain=chain, protein=protein, protein_tree=protein_tree, **kwargs)

        self.label = label

        # Parse important indices
        self.spin_idx = np.argwhere(np.isin(self.atom_names, ProEPR.SPIN_ATOMS[label]))

    @property
    def spin_coords(self):
        """get the spin coordinates of the rotamer library"""
        return self.coords[:, self.spin_idx, :].mean(axis=1)[:, 0]

    @property
    def spin_centroid(self):
        return np.average(self.spin_coords, weights=self.weights, axis=0)

    def protein_setup(self):
        if hasattr(self.protein, 'atoms') and isinstance(self.protein.atoms, mda.AtomGroup):
            self.protein = self.protein.select_atoms('not (byres name OH2 or resname HOH)')
            self._to_site()
            self.clash_ignore_idx = \
                self.protein.select_atoms(f'resid {self.site} and segid {self.chain}').ix

            self.resindex = self.protein.select_atoms(self.selstr).residues[0].resindex
            self.segindex = self.protein.select_atoms(self.selstr).residues[0].segindex

            if self.protein_tree is None:
                self.protein_tree = cKDTree(self.protein.atoms.positions)

            if self.kwargs.get('eval_clash', True):
                self.evaluate()

    @classmethod
    def from_mmm(cls, label, site, chain='A', protein=None, **kwargs):
        """
        Create a SpinLabel object using the default MMM protocol with any modifications passed via kwargs
        """
        MMM_maxdist = {'R1M': 9.550856367392733,
                       'R7M': 9.757254987175209,
                       'V1M': 8.237071322458029,
                       'M1M': 8.985723827323680,
                       'I1M': 12.952083029729994}

        # Store the force field parameter set being used before creating the spin label
        curr_lj = ProEPR.using_lj_param

        # Set MMM defaults or user defined overrides
        ProEPR.set_lj_params(kwargs.get('lj_params', 'uff'))

        clash_radius = kwargs.pop('clash_radius', MMM_maxdist[label] + 4)
        superimposition_method = kwargs.pop('superimposition_method', 'mmm')
        clash_ori = kwargs.pop('clash_ori', 'CA')
        energy_func = kwargs.pop('energy_func', ProEPR.get_lj_MMM)
        use_H = kwargs.pop('use_H', True)
        forgive = kwargs.pop('forgive', 0.5)

        # Calculate the SpinLabel
        SL = ProEPR.SpinLabel(label, site, chain=chain, protein=protein, superimposition_method=superimposition_method,
                              clash_radius=clash_radius, clash_ori=clash_ori, energy_func=energy_func,
                              use_H=use_H, forgive=forgive, **kwargs)

        # restore the force field parameter set being used before creating the spin label
        ProEPR.set_lj_params(curr_lj)
        return SL

    @classmethod
    def from_wizard(cls, label, site=1, chain='A', protein=None, to_find=200, to_try=10000, vdw=2.5, clashes=5, **kwargs):
        prelib = cls(label, site, chain, protein, eval_clash=False, **kwargs)
        if not kwargs.get('use_prior', False):
            del prelib.sigmas

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