import numpy as np
import chiLife as xl



def test_add_dlabel():
    xl.add_dlabel('DHC', 4, 'test_data/DHC.pdb',
                  dihedral_atoms=[[['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'ND1']],
                                  [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'ND1']]],
                  spin_atoms=['Cu1'])