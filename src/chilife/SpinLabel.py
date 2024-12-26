from functools import partial
import numpy as np
from scipy.spatial.distance import cdist

from .scoring import ljEnergyFunc, get_lj_energy
from .RotamerEnsemble import RotamerEnsemble


class SpinLabel(RotamerEnsemble):
    """
    A RotamerEnsemble made from a side chain with one or more unpaired electrons.

    Parameters
    ----------
    label: str
        Name of desired spin label, e.g. R1A.
    site: int
        MolSys residue number to attach spin label to.
    chain: str
        MolSys chain identifier to attach spin label to.
    protein: MDAnalysis.Universe, MDAnalysis.AtomGroup
        Object containing all protein information (coords, atom types, etc.)
    protein_tree: cKDtree
        k-dimensional tree object associated with the protein coordinates.
    """

    def __init__(self, label, site=None, protein=None, chain=None, rotlib=None, **kwargs):

        # Override RotamerEnsemble default of not evaluating clashes
        if not kwargs.get('minimize', False):
            kwargs.setdefault("eval_clash", True)

        super().__init__(label, site, protein=protein, chain=chain, rotlib=rotlib, **kwargs)

        self.label = label

        # Parse spin delocalization information
        sa_mask = np.isin(self.spin_atoms, self.atom_names)
        self.spin_atoms = self.spin_atoms[sa_mask]
        self.spin_weights = self.spin_weights[sa_mask]
        self.spin_idx = np.argwhere(np.isin(self.atom_names, self.spin_atoms))
        self.spin_idx.shape = -1

    @property
    def spin_coords(self):
        """get the spin coordinates of the rotamer ensemble"""
        return np.squeeze(self._coords[:, self.spin_idx, :])

    @property
    def spin_centers(self):
        """get the spin center of the rotamers in the ensemble"""
        if np.any(self.spin_idx):
            spin_centers = np.average(self._coords[:, self.spin_idx, :], weights=self.spin_weights, axis=1)
        else:
            spin_centers = np.array([])
        return np.atleast_2d(np.squeeze(spin_centers))

    @property
    def spin_centroid(self):
        """Average location of all the label's `spin_coords` weighted based off of the rotamer weights"""
        return np.average(self.spin_centers, weights=self.weights, axis=0)


    @classmethod
    def from_mmm(cls, label, site=None, protein=None, chain=None, **kwargs):
        """Create a SpinLabel object using the default MMM protocol with any modifications passed via kwargs"""

        MMM_maxdist = {
            "R1M": 9.550856367392733 + 4,
            "R7M": 9.757254987175209 + 4,
            "V1M": 8.237071322458029 + 4,
            "M1M": 8.985723827323680 + 4,
            "I1M": 12.952083029729994 + 4,
        }


        clash_radius = kwargs.pop("clash_radius", MMM_maxdist.get(label, None))
        alignment_method = kwargs.pop("alignment_method", "mmm")
        clash_ori = kwargs.pop("clash_ori", "CA")
        energy_func = kwargs.pop('energy_func', ljEnergyFunc(get_lj_energy, 'uff', forgive=0.5, cap=np.inf))
        use_H = kwargs.pop("use_H", True)

        # Calculate the SpinLabel
        SL = SpinLabel(
            label,
            site,
            protein,
            chain,
            alignment_method=alignment_method,
            clash_radius=clash_radius,
            clash_ori=clash_ori,
            energy_func=energy_func,
            use_H=use_H,
            **kwargs,
        )

        return SL

    @classmethod
    def from_wizard(
        cls,
        label,
        site=None,
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
            protein_clash_idx = prelib.protein_tree.query_ball_point(prelib.centroid, 19.0)
            protein_clash_idx = [idx for idx in protein_clash_idx if idx not in prelib.clash_ignore_idx]

        a, b = [list(x) for x in zip(*prelib.non_bonded)]
        for _ in range(np.rint(to_try / to_find).astype(int)):
            sample, _, internal_sample = prelib.sample(n=to_find, off_rotamer=True, return_dihedrals=True)

            # Evaluate internal clashes
            dist = np.linalg.norm(sample[:, a] - sample[:, b], axis=2)
            sidx = np.atleast_1d(np.squeeze(np.argwhere(np.all(dist > 2, axis=1))))
            if len(sidx) == 0:
                continue

            sample = sample[sidx, ...]
            internal_sample.use_frames(sidx)
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
            internal_sample.use_frames(sidx)

            internal_coords.append(internal_sample.trajectory.coordinate_array.copy())
            i += len(sidx)

            if i == to_find:
                break

        coords = coords[coords.sum(axis=(1, 2)) != 0]
        z_matrix = np.concatenate(internal_coords) if len(internal_coords) > 0 else None
        if z_matrix is not None:
            internal_sample.trajectory.load_new(z_matrix)
            internal_sample.set_cartesian_coords(coords, prelib.ic_mask)

            prelib.internal_coords = internal_sample
            prelib._dihedrals = np.rad2deg([ic.get_dihedral(1, prelib.dihedral_atoms) for ic in prelib.internal_coords])
        else:
            prelib.internal_coords = None
            prelib._dihedrals = np.array([[]])

        prelib._coords = coords
        prelib.weights = np.ones(len(coords))
        prelib.weights /= prelib.weights.sum()
        return prelib

    def __str__(self):
        return (super().__str__() +
                f"  spin atoms:\n    {self.spin_atoms}")



    def _base_copy(self, rotlib=None):
        return SpinLabel(self.res, self.site, rotlib=rotlib, chain=self.chain)
