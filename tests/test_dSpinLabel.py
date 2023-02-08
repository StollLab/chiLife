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
        sites=(2, 6),
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
    assert abs(p - 0.353804035205349) < 1e-7


def test_centroid():
    np.testing.assert_almost_equal(SL2.centroid, [19.691814, -13.956343,  10.883043], decimal=5)


def test_side_chain_idx():
    SL3 = xl.dSpinLabel("DHC", [28, 32], gb1)
    ans = np.array(['CB', 'CG', 'CD2', 'ND1', 'NE2', 'CE1', 'CB', 'CG', 'CD2', 'ND1',
                    'NE2', 'CE1', 'Cu1', 'O3', 'O1', 'O6', 'N5', 'C11', 'C9', 'C14',
                    'C8', 'C7', 'C10', 'O2', 'O4', 'O5'], dtype='<U3')
    np.testing.assert_equal(SL3.atom_names[SL3.side_chain_idx], ans)



def test_coords_setter():
    SL3 = xl.dSpinLabel("DHC", [28, 32], gb1)
    old = SL3.coords.copy()
    ans = old + 5
    SL3.coords += 5
    np.testing.assert_almost_equal(SL3.coords, ans)


def test_coord_set_error():
    SL3 = xl.dSpinLabel("DHC", [28, 32], gb1)
    ar = np.random.rand(5, 20, 3)
    with pytest.raises(ValueError):
        SL3.coords = ar


def test_mutate():
    SL2 = xl.dSpinLabel('DHC', (28, 32), gb1, min_method='Powell')
    gb1_Cu = xl.mutate(gb1, SL2)
    xl.save("mutate_dSL.pdb", gb1_Cu)

    with open("mutate_dSL.pdb", "r") as f:
        test = hashlib.md5(f.read().encode("utf-8")).hexdigest()

    with open("test_data/mutate_dSL.pdb", "r") as f:
        ans = hashlib.md5(f.read().encode("utf-8")).hexdigest()

    os.remove("mutate_dSL.pdb")

    assert test == ans


def test_single_chain_error():
    with pytest.raises(ValueError):
        xl.create_dlibrary(libname='___',
                           pdb='test_data/chain_broken_dlabel.pdb',
                           sites=(15, 17),
                           dihedral_atoms=[[['N', 'CA', 'C13', 'C5'],
                                       ['CA', 'C13', 'C5', 'C6']],
                                      [['N', 'CA', 'C12', 'C2'],
                                       ['CA', 'C12', 'C2', 'C3']]],
                           spin_atoms='Cu1')


def test_restraint_weight():
    SL3 = xl.dSpinLabel("DHC", [28, 32], gb1, restraint_weight=0.5)
    ans = np.array([0.579035, 0.420965])

    np.testing.assert_almost_equal(SL3.weights, ans, decimal=3)
    assert np.any(SL2.weights != SL3.weights)


def test_alternate_increment():
    with pytest.warns():
        with pytest.raises(RuntimeError):
            xl.dSpinLabel("DHC", (15, 44), gb1)

    SL2 = xl.dSpinLabel("DHC", (12, 37), gb1)
    np.testing.assert_almost_equal(SL2.spin_centers, [[26.0352109,  0.8579504,  3.0164344]], decimal=4)


def test_min_method():
    SL2 = xl.dSpinLabel("DHC", (28, 32), gb1, min_method='Powell')
    np.testing.assert_almost_equal(SL2.spin_centers, [[19.10902013, -14.5674808, 13.18468778]])
