import numpy as np

import chiLife
import chiLife as xl
import MDAnalysis as mda

protein = xl.fetch('1ubq')

def test_add_dlabel():
    xl.add_dlabel('DHC', 'test_data/DHC.pdb', 4, resi=2,
                  dihedral_atoms=[[['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'ND1']],
                                  [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'ND1']]],
                  spin_atoms=['Cu1'])

def test_dSpinLabel():
    SL = xl.dSpinLabel('DHC', 28, 4, protein)
    SL.save_pdb()
    print(SL)