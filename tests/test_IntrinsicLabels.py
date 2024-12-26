import os, hashlib
import numpy as np
import chilife as xl
import MDAnalysis as mda

ubq = mda.Universe('test_data/1ubq.pdb')
r = np.linspace(15, 70, 256)


def test_creation():
    sele = ubq.select_atoms('resnum 59')

    # Spin density -- Bleifuss et al. Biochemistry 2001. -- Why don't the spin densities sum to 1? IDK Ask Bleifuss.
    ISL = xl.IntrinsicLabel("TYX", sele, spin_atoms={'CG': 0.38, 'CE1': 0.25, 'CE2': 0.25, 'OH': 0.29})
    SL1 = xl.SpinLabel('R1M', 28, ubq)
    P = xl.distance_distribution(ISL, SL1, r)

    assert r[np.argmax(P)] - 21.470588235294116 < 1e-6
    assert np.max(P) - 0.25610254327499965 < 1e-6


def test_save():
    sele = ubq.select_atoms('resnum 59')
    ISL = xl.IntrinsicLabel("TYX", sele, spin_atoms={'CG': 0.38, 'CE1': 0.25, 'CE2': 0.25, 'OH': 0.29})
    xl.save('test_data/ISL_test.pdb', ISL)

    with open('test_data/ISL.pdb', 'r') as f:
        ans = hashlib.md5(f.read().encode("utf-8")).hexdigest()

    with open('test_data/ISL_test.pdb', 'r') as f:
        test = hashlib.md5(f.read().encode("utf-8")).hexdigest()

    os.remove('test_data/ISL_test.pdb')

    assert test == ans


def test_single_atom():
    azurin = xl.fetch('1e67')
    Zn = azurin.select_atoms('segid D and resnum 129')
    ISL = xl.IntrinsicLabel("ZCU", Zn, spin_atoms="ZN")
    np.testing.assert_allclose(ISL.spin_centers, [[ 5.59399986, 31.97200012, 39.02500153]])


def test_ag_spin_atoms():
    azurin = xl.fetch('1e67')
    Zn = azurin.select_atoms('segid D and resnum 129')
    ISL = xl.IntrinsicLabel("ZCU", Zn, spin_atoms=Zn)
    np.testing.assert_allclose(ISL.spin_centers, [[ 5.59399986, 31.97200012, 39.02500153]])