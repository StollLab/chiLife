import os
import hashlib
import numpy as np
import pytest
import MDAnalysis as mda
import chilife as xl

protein = mda.Universe("test_data/1ubq.pdb", in_memory=True)
gb1 = mda.Universe("test_data/4wh4.pdb", in_memory=True).select_atoms("protein and segid A")
SL2 = xl.dSpinLabel("DHC", [28, 28+4], gb1)


def test_add_dlabel():
    Energies = np.loadtxt("test_data/DHC.energies")[:, 1]
    P = np.exp(-Energies / (xl.GAS_CONST * 298))
    P /= P.sum()
    xl.create_dlibrary(
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
    os.remove('___ip4_drotlib.zip')

def test_distance_distribution():
    r = np.linspace(15, 50, 256)
    SL1 = xl.SpinLabel("R1M", 6, gb1)
    dd = xl.distance_distribution(SL1, SL2, r)
    d = r[np.argmax(dd)]
    p = np.max(dd)
    assert abs(d - 26.529411764705884) < 1e-7
    assert abs(p - 0.3404778733692674) < 1e-7


def test_centroid():
    np.testing.assert_allclose(SL2.centroid, [19.90399089, -13.87953919, 11.16228634])


def test_side_chain_idx():
    SL3 = xl.dSpinLabel("DHC", [28, 32], gb1)
    tc = SL3.coords
    tc[:, SL3.side_chain_idx] += 5
    ans = np.load("test_data/dSL_scidx.npy")
    np.testing.assert_almost_equal(tc, ans, decimal=5)


def test_coords_setter():
    SL3 = xl.dSpinLabel("DHC", [28, 32], gb1)
    SL3.coords = SL3.coords + 5
    ans = np.load("test_data/dSL_csetter.npy")
    np.testing.assert_almost_equal(SL3.coords, ans, decimal=5)


def test_coord_set_error():
    SL3 = xl.dSpinLabel("DHC", [28, 32], gb1)
    ar = np.random.rand(5, 20, 3)
    with pytest.raises(ValueError):
        SL3.coords = ar


def test_mutate():
    gb1_Cu = xl.mutate(gb1, SL2)
    xl.save("mutate_dSL.pdb", gb1_Cu)

    with open("mutate_dSL.pdb", "r") as f:
        test = hashlib.md5(f.read().encode("utf-8")).hexdigest()

    with open("test_data/mutate_dSL.pdb", "r") as f:
        ans = hashlib.md5(f.read().encode("utf-8")).hexdigest()

    os.remove("mutate_dSL.pdb")

    assert test == ans


def test_add_dlabel2():
    Energies = np.loadtxt('test_data/HCS.energies')

    P = np.exp(-Energies/ (xl.GAS_CONST * 278))
    P /= P.sum()

    dihedral_atoms = [[["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "ND1"]],
                      [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "ND1"]]]

    xl.create_dlibrary('___', 'test_data/HCS.pdb', site=2, increment=2, dihedral_atoms=dihedral_atoms, spin_atoms='Cu1')

    SL1 = xl.dSpinLabel("___", [6, 8], gb1)
    r = np.linspace(15, 80, 256)
    dd = xl.distance_distribution(SL1, SL2, r, sigma=0.5)

    assert r[np.argmax(dd)] == 23.41176470588235
    os.remove('___ip2_drotlib.zip')

def test_single_chain_error():
    with pytest.raises(ValueError):
        xl.create_dlibrary(libname='___',
                           pdb='test_data/chain_broken_dlabel.pdb',
                           increment=2,
                           dihedral_atoms=[[['N', 'CA', 'C13', 'C5'],
                                       ['CA', 'C13', 'C5', 'C6']],
                                      [['N', 'CA', 'C12', 'C2'],
                                       ['CA', 'C12', 'C2', 'C3']]],
                           site=15,
                           spin_atoms='Cu1')

def test_restraint_weight():
    SL3 = xl.dSpinLabel("DHC", [28, 32], gb1, restraint_weight=0.5)
    ans = np.array([0.4291847, 0.40492998, 0.16588532])

    np.testing.assert_allclose(SL3.weights, ans)
    assert SL2.weights != SL3.weights
