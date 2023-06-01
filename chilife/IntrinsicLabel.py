import numpy as np


class IntrinsicLabel:

    def __init__(self, atom_selection, spin_atoms=None):
        """
        A helper object to assign spin density to part of a protein that is already present, e.g. metal ligands.
        IntrinsicLabels can be used with the :func:`~chilife.chilife.distance_distribution` function

        Parameters
        ----------
        atom_selection : MDAnalysis.AtomGroup, chiLife.System
            Group of atoms constituting the intrinsic label. These selections can consist of multiple residues, which
            can be useful in the case of ions with multiple coordinating residues

        spin_atoms : str, list, tuple, array, dict, MDAnalysis.AtomGroup
            Atoms of the intrinsic label that host the unpaired electron density. Can be a single atom name, a
            list/tuple/array of atom names or a dictionary mapping atom names to their relative populations.
            ``spin_atoms `` can also be an ``MDAnalysis.AtomGroup`` object derived from the same MDAnalysis.Universe
            as the ``atom_selection`` keyword argument. Use of an ``AtomGroup`` is particularly useful when there is
            spin density is distributed on several atoms with the same name on different residues within the
            ``IntrinsicLabel``.
        """
        self._selection = atom_selection
        self._coords = atom_selection.positions.copy()[None, :, :]
        self.coords = self._coords
        self.atom_names = atom_selection.names.copy()

        self.weights = np.array([1])
        if isinstance(spin_atoms, (list, tuple, np.ndarray)):
            self.spin_atoms = np.asarray(spin_atoms)
            self.spin_weights = np.ones(len(spin_atoms)) / len(spin_atoms)
        elif isinstance(spin_atoms, dict):
            self.spin_atoms = np.asarray([key for key in spin_atoms])
            self.spin_weights = np.asarray([val for val in spin_atoms.values()])
        elif isinstance(spin_atoms, str):
            self.spin_atoms = np.asarray([spin_atoms])
            self.spin_weights = np.asarray([1])
        else:
            raise RuntimeError("spin_atoms must contain a string, a list/tuple/array of strings, a dict")

        sa_mask = np.isin(self.spin_atoms, self.atom_names)

        self.spin_idx = np.argwhere(sa_mask)
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