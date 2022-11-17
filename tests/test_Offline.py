import os
import MDAnalysis as mda
import numpy as np
import chilife as xl

def test_add_dlabel():
    Energies = np.loadtxt("test_data/DHC.energies")[:, 1]
    P = np.exp(-Energies / (xl.GAS_CONST * 298))
    P /= P.sum()
    xl.add_dlibrary(
        "___",
        "test_data/DHC.pdb",
        4,
        site=2,
        weights=P,
        dihedral_atoms=[
            [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "ND1"]],
            [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "ND1"]],
        ],
        spin_atoms=["Cu1"],
    )
    xl.remove_label('___', prompt=False)
    os.remove('___ip4_drotlib.npz')

def test_polyproII():
    PPII = mda.Universe('test_data/PolyProII.pdb', in_memory=True)
    PPII_IC = xl.get_internal_coords(PPII)

    for i in range(2, 13):
        assert ("C", "CA", "N", "C") in PPII_IC.ICs[1][i]
        angles = np.deg2rad((-75, 150))
        PPII_IC.set_dihedral(angles, i, [["C", "N", "CA", "C"], ["N", "CA", "C", "N"]])

    newc = PPII_IC.coords
    PPII.atoms.positions = newc
