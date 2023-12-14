from .SpinLabel import SpinLabel
from functools import partial
import numpy as np


def _get_spin_label(frame, label, site, chain, protein, **kwargs):
    protein.universe.trajectory[frame]
    SL = SpinLabel(label, site, chain, protein, **kwargs)
    print(SL)
    return SL


class SpinLabelTraj:
    def __init__(self, label, site=None, chain="A", protein=None, **kwargs):
        get_sl_frame = partial(
            _get_spin_label,
            label=label,
            site=site,
            chain=chain,
            protein=protein,
            **kwargs
        )

        self.LabelTraj = [
            get_sl_frame(i) for i in np.arange(protein.universe.trajectory.n_frames)
        ]

    def __iter__(self):
        return iter(self.LabelTraj)

    def __len__(self):
        return len(self.LabelTraj)
