import numpy as np
import chiLife as xl

def test_add_dlabel2():
    Energies = np.loadtxt('test_data/HCS.energies')

    P = np.exp(-Energies/ (xl.GAS_CONST * 278))
    P /= P.sum()

    dihedral_atoms = [[["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "ND1"]],
                      [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "ND1"]]]

    xl.add_dlabel('HCS', 'test_data/HCS.pdb', resi=2, increment=2, dihedral_atoms=dihedral_atoms, spin_atoms='Cu1')


def test_add_dlabel():
    Energies = np.loadtxt("test_data/DHC.energies")[:, 1]
    P = np.exp(-Energies / (xl.GAS_CONST * 298))
    P /= P.sum()
    xl.add_dlabel(
        "DHC",
        "test_data/DHC.pdb",
        4,
        resi=2,
        weights=P,
        dihedral_atoms=[
            [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "ND1"]],
            [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "ND1"]],
        ],
        spin_atoms=["Cu1"],
    )
#
# def test_add_HIN():
#     xl.add_dlabel(name='HIN',
#                   pdb='comp_sorted.pdb',
#                   increment=2,
#                   dihedral_atoms=[[['N', 'CA', 'C13', 'C5'],
#                                    ['CA', 'C13', 'C5', 'C6']],
#                                   [['N', 'CA', 'C12', 'C2'],
#                                    ['CA', 'C12', 'C2', 'C3']]],
#                   resi=1,
#                   spin_atoms='Cu1')