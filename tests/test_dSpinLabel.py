import numpy as np
import chiLife as xl
import MDAnalysis as mda


def test_add_dlabel():
    xl.add_dlabel('DHC', 'test_data/DHC.pdb', 4, resi=2,
                  dihedral_atoms=[[['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'ND1']],
                                  [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'ND1']]],
                  spin_atoms=['Cu1'])