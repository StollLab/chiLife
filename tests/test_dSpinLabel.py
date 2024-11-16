import os
import hashlib
import numpy as np
import pytest
import MDAnalysis as mda
import chilife as xl

protein = mda.Universe("test_data/1ubq.pdb", in_memory=True)
gb1 = mda.Universe("test_data/4wh4.pdb", in_memory=True).select_atoms("protein and segid A")
SL2 = xl.dSpinLabel("DHC", [28, 28+4], gb1, rotlib='test_data/DHC')


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
            [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "ND1"], ['CD2', 'NE2', 'Cu1', 'N5']],
            [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "ND1"], ['CD2', 'NE2', 'Cu1', 'N5']],
        ],
        spin_atoms=["Cu1"],
    )

    # Test that chain operators were reset
    dSL = xl.dSpinLabel('___', (28, 32), protein)

    os.remove('___ip4_drotlib.zip')

    spin_center_ans = np.array([[44.46482086, 25.11999893, 12.35381317],
                                [44.35407639, 24.42422485, 11.61791611]])
    Eans =np.array([-2.80119358, -2.33511906, -0.21797025, -6.21705528, -2.50485578,
                    -7.42736075, 87.1990342 ,  0.93290375, -1.83151097, -1.18059185,
                    -6.61134315, -6.70879944])

    np.testing.assert_almost_equal(dSL.spin_centers, spin_center_ans, decimal=5)
    np.testing.assert_almost_equal(dSL.atom_energies.sum(axis=1), Eans, decimal=3)

def test_add_dlabel_shared_atom_names():
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

    # Test that chain operators were reset
    with pytest.raises(RuntimeError):
        dSL = xl.dSpinLabel('___', (28, 32), protein)

    dSL = xl.dSpinLabel('___', (28, 32), protein, minimize=False)

    # Ensure cst idx are aligned
    assert np.all(dSL.RE1.atom_names[dSL.cst_idx1] == dSL.RE2.atom_names[dSL.cst_idx2])

    # And that the duplicate name atom sare not
    test = np.linalg.norm(np.diff(dSL.RE1.coords[:, dSL.RE1.atom_names == 'ND1'], axis=1), axis=2)
    ans = np.array([[6.650147 ],
                    [6.6462107],
                    [6.754248 ]])
    np.testing.assert_almost_equal(test, ans, decimal=6)

    os.remove('___ip4_drotlib.zip')


def test_distance_distribution():
    r = np.linspace(15, 50, 256)
    SL1 = xl.SpinLabel("R1M", 6, gb1)
    dd = xl.distance_distribution(SL1, SL2, r)
    d = r[np.argmax(dd)]
    p = np.max(dd)
    assert abs(d - 26.529411764705884) < 1e-7
    assert abs(p - 0.3548311024322747) < 1e-7


def test_centroid():
    np.testing.assert_almost_equal(SL2.centroid, [19.69981126, -13.95217125,  10.8655417], decimal=5)


def test_side_chain_idx():
    SL3 = xl.dSpinLabel("DHC", [28, 32], gb1, rotlib='test_data/DHC')
    ans = np.array(['CB', 'CG', 'CD2', 'ND1', 'NE2', 'CE1', 'CB', 'CG', 'CD2', 'ND1',
                    'NE2', 'CE1', 'Cu1', 'O3', 'O1', 'O6', 'N5', 'C11', 'C9', 'C14',
                    'C8', 'C7', 'C10', 'O2', 'O4', 'O5'], dtype='<U3')
    np.testing.assert_equal(SL3.atom_names[SL3.side_chain_idx], ans)



def test_coords_setter():
    SL3 = xl.dSpinLabel("DHC", [28, 32], gb1, rotlib='test_data/DHC')
    old = SL3.coords.copy()
    ans = old + 5
    SL3.coords += 5
    np.testing.assert_almost_equal(SL3.coords, ans)


def test_coord_set_error():
    SL3 = xl.dSpinLabel("DHC", [28, 32], gb1, rotlib='test_data/DHC')
    ar = np.random.rand(5, 20, 3)
    with pytest.raises(ValueError):
        SL3.coords = ar


def test_mutate():
    SL2 = xl.dSpinLabel('DHC', (28, 32), gb1, min_method='Powell', rotlib='test_data/DHC')
    gb1_Cu = xl.mutate(gb1, SL2)
    xl.save("mutate_dSL.pdb", gb1_Cu)

    with open("mutate_dSL.pdb", "r") as f:
        test = hashlib.md5(f.read().encode("utf-8")).hexdigest()

    with open("test_data/mutate_dSL.pdb", "r") as f:
        ans = hashlib.md5(f.read().encode("utf-8")).hexdigest()

    os.remove("mutate_dSL.pdb")

    assert test == ans


def test_single_chain_error():
    with pytest.raises(RuntimeError):
        xl.create_dlibrary(libname='___',
                           pdb='test_data/chain_broken_dlabel.pdb',
                           sites=(15, 17),
                           dihedral_atoms=[[['N', 'CA', 'C13', 'C5'],
                                       ['CA', 'C13', 'C5', 'C6']],
                                      [['N', 'CA', 'C12', 'C2'],
                                       ['CA', 'C12', 'C2', 'C3']]],
                           spin_atoms='Cu1')


def test_restraint_weight():
    SL3 = xl.dSpinLabel("DHC", [28, 32], gb1, restraint_weight=5, rotlib='test_data/DHC')
    ans = np.array([0.51175099, 0.48824901])

    np.testing.assert_almost_equal(SL3.weights, ans, decimal=3)
    assert np.any(SL2.weights != SL3.weights)


def test_alternate_increment():
    with pytest.warns():
        with pytest.raises(RuntimeError):
            xl.dSpinLabel("DHC", (15, 44), gb1, rotlib='test_data/DHC')

    SL2 = xl.dSpinLabel("DHC", (12, 37), gb1, rotlib='test_data/DHC')
    ans = np.array([[25.50056398,  1.27502619,  3.3972164 ],
                    [24.45518418,  2.0136011 ,  3.74825233],
                    [24.57566352,  1.97824049,  3.73004511],
                    [24.63698352,  1.94542836,  3.70611475],
                    [24.52183104,  2.00650002,  3.74023405]])
    np.testing.assert_almost_equal(SL2.spin_centers, ans, decimal=4)


def test_min_method():
    SL2 = xl.dSpinLabel("DHC", (28, 32), gb1, min_method='Powell', rotlib='test_data/DHC')
    ans = np.array([[ 18.6062595, -14.7057183,  12.0624657],
                    [ 18.5973142, -14.7182378,  12.0220757]])

    np.testing.assert_allclose(SL2.spin_centers, ans)


def test_no_min():
    SL2 = xl.dSpinLabel("DHC", (28, 32), gb1, minimize=False, rotlib='test_data/DHC')

    bb_coords = gb1.select_atoms('resid 28 32 and name N CA C O').positions
    bb_idx = np.argwhere(np.isin(SL2.atom_names, SL2.backbone_atoms)).flatten()

    for conf in SL2.coords:
        # Decimal = 1 because bisect alignment does not place exactly by definition
        np.testing.assert_almost_equal(conf[bb_idx], bb_coords, decimal=1)


def test_trim_false():
    SL1 = xl.dSpinLabel('DHC', (28, 32), gb1, trim=False, rotlib='test_data/DHC')
    SL3 = xl.dSpinLabel('DHC', (28, 32), gb1, eval_clash=False, rotlib='test_data/DHC')

    assert len(SL1) == len(SL3)
    most_probable = np.sort(SL1.weights)[::-1][:len(SL2)]
    most_probable /= most_probable.sum()
    np.testing.assert_almost_equal(SL2.weights, most_probable)
    assert np.any(np.not_equal(SL1.weights, SL3.weights))

def test_dmin_callback():
    vals = []
    ivals = []
    def my_callback(val, i):
        vals.append(val)
        ivals.append(i)

    SL1 = xl.dRotamerEnsemble('DHC', (28, 32), gb1, minimize=False, rotlib='test_data/DHC')
    SL1.minimize(callback=my_callback)

    assert len(vals) > 0
    assert len(ivals) > 0


def test_ig_waters():
    ub = xl.fetch('1ubq')
    SL1 = xl.dRotamerEnsemble('DHC', (28, 32), ub, ignore_waters=False, rotlib='test_data/DHC')
    ans = np.load('test_data/dignore_waters.npy')
    np.testing.assert_almost_equal(SL1.coords, ans, decimal=3)



def test_dihedral_atoms():
    ans = [['N', 'CA', 'CB', 'CG'],
           ['CA', 'CB', 'CG', 'ND1'],
           ['CD2', 'NE2', 'Cu1', 'O1'],
           ['N', 'CA', 'CB', 'CG'],
           ['CA', 'CB', 'CG', 'ND1'],
           ['CD2', 'NE2', 'Cu1', 'O1']]

    assert np.all(SL2.dihedral_atoms == ans)


def test_dihedrals():
    ans = np.array([[-2.97070555,  2.20229457, -2.5926639 , -1.21838379, -2.07361964, -2.74030998],
                    [-2.94511378,  2.18546192, -2.59115287, -1.2207707 , -2.05741481, -2.73392102]])

    np.testing.assert_almost_equal(SL2.dihedrals, ans)
