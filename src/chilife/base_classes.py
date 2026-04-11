from abc import ABC
import numpy as np


class Ensemble(ABC):
    def __str__(self):
        return (
            f"Ligand ensemble with {np.size(self.weights)} members\n"
            + f"  Name: {self.name}\n"
            + f"  Ligand: {self.res}\n"
            + f"  Site: {self.site}\n"
            + "  Dihedral definitions:\n"
            + "\n".join([f"    {d}" for d in self.dihedral_atoms])
            + "\n"
        )

    def __repr__(self):
        return str(self)

    def update_weight(self, weight: float) -> None:
        """
         Function to assign `self.current_weight`, which is the estimated weight of the rotamer currently occupying the
         attachment site on `self.protein`. This is only relevant if the residue type on `self.protein` is the same as
         the  RotamerLibrary.

        Parameters
        ----------
        weight : float
            New weight for the current residue.
        """
        self.current_weight = weight


class FreeRadical(ABC):
    @property
    def spin_coords(self):
        """get the spin coordinates of the rotamer ensemble"""
        return self.coords[:, self.spin_idx]

    @property
    def spin_centers(self):
        """get the spin center of the rotamers in the ensemble"""
        if len(self.spin_idx) > 0:
            spin_centers = np.average(
                self.spin_coords, weights=self.spin_weights, axis=1
            )
        else:
            spin_centers = np.array([])
        return np.atleast_2d(np.squeeze(spin_centers))

    @property
    def spin_centroid(self):
        """Average location of all the label's `spin_coords` weighted based off of the rotamer weights"""
        return np.average(self.spin_centers, weights=self.weights, axis=0)

    def __str__(self):
        return super().__str__() + f"  spin atoms:\n    {self.spin_atoms}"