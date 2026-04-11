import numpy as np


from .base_classes import FreeRadical
from .LigandEnsemble import LigandEnsemble


class SpinLigand(LigandEnsemble, FreeRadical):
    def __init__(self, res, site=None, protein=None, chain=None, rotlib=None, **kwargs):
        super().__init__(res, site, protein, chain, rotlib, **kwargs)

        if "spin_atoms" in kwargs:
            spin_atoms = kwargs.pop("spin_atoms")
            if isinstance(spin_atoms, (list, tuple, np.ndarray)):
                spin_atoms = {sa: 1 / len(spin_atoms) for sa in spin_atoms}
            elif isinstance(spin_atoms, dict):
                pass
            else:
                raise TypeError("the `spin_atoms` kwarg must be a dict, list or tuple")

            spin_atoms = np.array(list(spin_atoms.keys()))
            spin_weights = np.array(list(spin_atoms.values()))

        elif hasattr(self, "_sdf_data") and "RAD" in self._sdf_data[0]["properties"]:
            rad_data = self._sdf_data[0]["properties"]["RAD"]

            # get spin atoms from sdf file but account for 1 indexing
            spin_atoms = rad_data["atom_ids"]
            spin_weights = np.ones_like(spin_atoms, dtype=float) / len(spin_atoms)

        else:
            raise RuntimeError(
                "Provided rotamer library or SDF data does not contain spin information"
            )

        self.spin_atoms = self.atom_names[spin_atoms].copy()
        self.spin_weights = spin_weights
        self.spin_idx = spin_atoms
        self.spin_idx.shape = -1