import os
import hashlib
import numpy as np
import pytest
import MDAnalysis as mda
import chiLife as xl

import matplotlib.pyplot as plt


protein = xl.fetch('1ubq')
gb1 = xl.fetch('4wh4').select_atoms('protein and segid A')
SL2 = xl.dSpinLabel('DHC', 28, 4, gb1)

def test_add_dlabel():
    Energies = np.loadtxt('test_data/DHC.energies')[:, 1]
    P = np.exp(-Energies / (xl.GAS_CONST * 298))
    P /= P.sum()
    xl.add_dlabel('DHC', 'test_data/DHC.pdb', 4, resi=2, weights=P,
                  dihedral_atoms=[[['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'ND1']],
                                  [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'ND1']]],
                  spin_atoms=['Cu1'])


def test_distance_distribution():
    r = np.linspace(15, 50, 256)
    SL1 = xl.SpinLabel('R1A', 6, gb1)
    dd = xl.get_dd(SL1, SL2, r)
    np.testing.assert_almost_equal(dd[50:60], [0.00183767, 0.00297856, 0.00437123, 0.00802083, 0.01147641,
                                               0.01642888, 0.02282988, 0.03100296, 0.04135329, 0.05425753], decimal=6)


def test_centroid():
    np.testing.assert_allclose(SL2.centroid, [19.90399089, -13.87953919,  11.16228634])


def test_side_chain_idx():
    SL3 = xl.dSpinLabel('DHC', 28, 4, gb1)
    tc = SL3.coords
    tc[:, SL3.side_chain_idx] += 5
    ans = np.load('test_data/dSL_scidx.npy')
    np.testing.assert_allclose(tc, ans)


def test_coords_setter():
    SL3 = xl.dSpinLabel('DHC', 28, 4, gb1)
    SL3.coords = SL3.coords + 5
    ans = np.load('test_data/dSL_csetter.npy')
    np.testing.assert_allclose(SL3.coords, ans)


def test_coord_set_error():
    SL3 = xl.dSpinLabel('DHC', 28, 4, gb1)
    ar = np.random.rand(5, 20, 3)
    with pytest.raises(ValueError):
        SL3.coords = ar


def test_mutate():
    gb1_Cu = xl.mutate(gb1, SL2)
    xl.save('mutate_dSL.pdb', gb1_Cu)

    with open('mutate_dSL.pdb', 'r') as f:
        test = hashlib.md5(f.read().encode('utf-8')).hexdigest()

    with open('test_data/mutate_dSL.pdb', 'r') as f:
        ans = hashlib.md5(f.read().encode('utf-8')).hexdigest()

    os.remove('mutate_dSL.pdb')

    assert test == ans

