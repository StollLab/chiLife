import os
import sys
import hashlib
from pathlib import Path
import pytest
import numpy as np

import chilife as xl
from chilife.LigandEnsemble import remap_sdf

_is_macos = sys.platform == "darwin"

protein = xl.load_protein("test_data/3gkz.pdb")


def test_from_sdf():
    LE = xl.LigandEnsemble.from_sdf("test_data/test_la_subject.sdf", use_H=False)
    with np.load("test_data/test_from_sdf.npz") as f:
        ans = dict(f)

    np.testing.assert_almost_equal(LE.coords, ans["coords"])
    np.testing.assert_almost_equal(LE.weights, ans["weights"])
    np.testing.assert_almost_equal(LE.dihedrals, ans["dihedrals"])
    np.testing.assert_equal(LE.dihedral_atoms, ans["dihedral_atoms"])


@pytest.mark.skipif(
    _is_macos,
    reason="Clash filtering results differ on macOS ARM due to BLAS differences",
)
def test_from_sdf2():
    LE = xl.LigandEnsemble.from_sdf(
        "test_data/test_la_subject.sdf", site=500, protein=protein, use_H=False
    )

    with np.load("test_data/test_from_sdf2.npz") as f:
        ans = dict(f)

    np.testing.assert_almost_equal(LE.coords, ans["coords"])
    np.testing.assert_almost_equal(LE.weights, ans["weights"])
    np.testing.assert_almost_equal(LE.dihedrals, ans["dihedrals"])
    np.testing.assert_equal(LE.dihedral_atoms, ans["dihedral_atoms"])


@pytest.mark.skipif(
    _is_macos, reason="Sampling results differ on macOS ARM due to BLAS differences"
)
def test_sample_from_sdf():
    np.random.seed(0)
    LE = xl.LigandEnsemble.from_sdf(
        "test_data/test_la_subject.sdf",
        site=500,
        protein=protein,
        sample=500,
        use_H=False,
    )

    with np.load("test_data/test_sample_from_sdf.npz") as f:
        ans = dict(f)

    np.testing.assert_almost_equal(LE.coords, ans["coords"])
    np.testing.assert_almost_equal(LE.weights, ans["weights"])
    np.testing.assert_almost_equal(LE.dihedrals, ans["dihedrals"])


def test_sample_no_alignment_method():
    np.random.seed(0)
    LE = xl.LigandEnsemble.from_sdf(
        "test_data/test_sample_no_alignment.sdf",
        site=500,
        protein=protein,
        sample=500,
        alignment_method=None,
    )
    with np.load("test_data/test_sample_LE_no_aln.npz") as f:
        ans = dict(f)

    np.testing.assert_almost_equal(LE.coords, ans["coords"])
    np.testing.assert_almost_equal(LE.weights, ans["weights"])
    np.testing.assert_almost_equal(LE.dihedrals, ans["dihedrals"])


def test_remap_sdf():
    file_name = "test_data/test_la_subject.sdf"
    data = xl.read_sdf(file_name)
    H_mask = np.array([a["element"] != "H" for a in data[0]["atoms"]])
    new_data = remap_sdf(data[0], H_mask)
    xl.write_sdf(new_data, "tmp.sdf")

    with open("test_data/test_remap_sdf_ans.sdf", "r") as f:
        ans = hashlib.md5(f.read().encode("utf-8")).hexdigest()

    with open("tmp.sdf", "r") as f:
        test = hashlib.md5(f.read().encode("utf-8")).hexdigest()

    os.remove("tmp.sdf")


@pytest.mark.skipif(
    _is_macos,
    reason="Clash filtering results differ on macOS ARM due to BLAS differences",
)
def test_spin_atoms():
    LE = xl.SpinLigand.from_sdf("test_data/dAdo.sdf", site=500, protein=protein)
    assert LE.spin_atoms[0] == "C1"

    spin_centers_ans = np.array(
        [
            [-7.35865026, -6.82140565, 17.081305],
            [-6.82096181, -7.84232728, 16.7139629],
            [-7.26680359, -6.46974427, 17.04729826],
            [-7.35944675, -6.0480977, 17.06029795],
            [-7.11066296, -6.45882481, 17.10317445],
            [-6.37372279, -5.9977177, 15.91669753],
            [-7.0460257, -7.2452747, 17.08448347],
            [-1.20554346, -3.27410407, 14.51944261],
            [-7.11535408, -7.35621504, 16.60623721],
            [-7.1975133, -6.91233867, 17.54012882],
            [-1.65077128, -4.22739542, 14.93372774],
            [-6.57534758, -7.42153722, 16.20850711],
            [-7.08465914, -7.03338525, 16.41745372],
            [-1.57373798, -4.08089548, 14.53360959],
            [-7.34505641, -6.7657256, 16.9571458],
        ]
    )
    np.testing.assert_almost_equal(LE.spin_centers, spin_centers_ans)


def test_repack():
    np.random.seed(0)
    LE = xl.LigandEnsemble.from_sdf(
        "test_data/test_sample_no_alignment.sdf",
        site=500,
        protein=protein,
        sample=500,
        alignment_method=None,
    )

    u, dE = xl.repack(protein, LE, off_rotamer=True, repetitions=10)

    with np.load("test_data/test_repack_LE.npz") as f:
        ans = dict(f)

    np.testing.assert_almost_equal(
        u.universe.trajectory.coordinate_array, ans["coords"], decimal=5
    )
    np.testing.assert_almost_equal(dE, ans["dE"], decimal=5)


def test_spin_ligand():
    np.random.seed(0)
    prot = xl.load_protein("test_data/7o1o.pdb")
    SL1 = xl.SpinLigand.from_sdf(
        "test_data/dAdo.sdf",
        "SAM",
        408,
        prot,
        trim=False,
        alignment_method=None,
        sample=500,
    )

    SL2 = xl.dSpinLabel("DCN", (91, 95), prot)
    r = np.linspace(15, 80, 256)
    P = xl.distance_distribution(SL1, SL2, r)
    assert P[48] - 0.3848342141639043 < 1e-7


@pytest.mark.parametrize("file_name", list(Path("test_data/random_sdfs").glob("*.sdf")))
def test_random_sdf(file_name):
    xl.LigandEnsemble.from_sdf(file_name)


def test_set_dihedrals():
    LE = xl.LigandEnsemble.from_sdf("test_data/dAdo.sdf")
    ans = np.load("test_data/LE_set_dihedral.npy")
    LE.dihedrals = [[180], [0]]
    np.testing.assert_almost_equal(LE.coords, ans)


def test_set_coords():
    coords = np.load("test_data/LE_set_dihedral.npy")
    LE = xl.LigandEnsemble.from_sdf("test_data/dAdo.sdf")
    LE.coords = coords
    np.testing.assert_almost_equal(LE.dihedrals, [[180], [0]])


def test_set_dihedral_sigmas1():
    LE = xl.LigandEnsemble.from_sdf("test_data/random_sdfs/5R5.sdf")
    LE.set_dihedral_sampling_sigmas(25)

    assert np.all(LE.sigmas == 25.0)


@pytest.mark.skipif(
    _is_macos,
    reason="SDF output differs on macOS ARM due to BLAS differences in sampling",
)
def test_to_rotlib():
    np.random.seed(0)
    LE = xl.LigandEnsemble.from_sdf("test_data/random_sdfs/5R5.sdf", sample=50)
    LE.to_rotlib()

    LE2 = xl.LigandEnsemble("LIG", rotlib="LIG_from_sdf")
    np.testing.assert_almost_equal(LE.coords, LE2.coords)

    LE3 = xl.LigandEnsemble("LIG", site=500, protein=protein, rotlib="LIG_from_sdf")
    LE3.to_sdf("to_rotlib.sdf")

    with open("test_data/to_rotlib.sdf", "r") as f:
        ans_str = f.read()
        ans_str = ans_str.replace("1.2.2", "version")
        ans = hashlib.md5(ans_str.encode("utf-8")).hexdigest()

    with open("to_rotlib.sdf", "r") as f:
        test_str = f.read()
        test_str = test_str.replace(xl.__version__, "version")
        test = hashlib.md5(test_str.encode("utf-8")).hexdigest()

    assert ans == test

    os.remove("LIG_from_sdf_rotlib.npz")
    os.remove("to_rotlib.sdf")


def test_copy():
    LE = xl.LigandEnsemble.from_sdf("test_data/random_sdfs/5R5.sdf", sample=50)
    LE2 = LE.copy()

    np.testing.assert_almost_equal(LE.coords, LE2.coords)

    LE.coords, _ = LE.sample(50, off_rotamer=True)

    assert LE.coords.shape != LE2.coords.shape
