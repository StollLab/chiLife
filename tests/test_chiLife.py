import pickle, hashlib, os
import numpy as np
import pytest
import MDAnalysis as mda
import chilife
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path

labels = ["R1M", "R7M", "V1M", "M1M", "I1M"]
atom_names = ["C", "H", "N", "O", "S", "SE", "Br"]
rmin_params = [2.0000, 1.27, 1.8500, 1.7000, 2.0000, 2.0000, 1.9800]
eps_params = [-0.110, -0.022, -0.200, -0.120, -0.450, -0.450, -0.320]
U = mda.Universe("test_data/m1omp.pdb")
protein = U.select_atoms("protein")
r = np.linspace(0, 100, 2 ** 8)
old_ef = lambda protein, ensemble: chilife.get_lj_rep(protein, ensemble, forgive=0.8)

with open('test_data/test_from_MMM.pkl', 'rb') as f:
    from_mmm_Ps, from_mmm_rotlibs = pickle.load(f)

from_mmm_r = from_mmm_Ps.pop('r')
omp = mda.Universe('test_data/1omp.pdb', in_memory=True)
anf = mda.Universe('test_data/1anf.pdb', in_memory=True)
from_mmm_SLs = {}
for key in from_mmm_rotlibs:
    label, site, state = key
    mbp = omp if state == 'Apo' else anf
    from_mmm_SLs[key] = chilife.SpinLabel.from_mmm(label, site, protein=mbp)

# get permutations for distance_distribution() tests
kws = []
args = []
for SL in labels:
    site1, site2 = 25, 73
    chain = "A"
    SL1 = chilife.SpinLabel(SL, site1, protein, chain)
    SL2 = chilife.SpinLabel(SL, site2, protein, chain)
    [args.append([SL1, SL2]) for i in range(5)]

    kws.append({"r": np.linspace(0, 80, 1024)})
    kws.append({"r": np.linspace(20, 80, 256)})
    kws.append({'use_spin_centers': False})
    kws.append({'sigma': 2})
    kws.append({'sigma': 0.5, 'use_spin_centers': False})

ans = []
with np.load("test_data/get_dd_tests.npz") as f:
    for i in range(25):
        ans.append(f[f"arr_{i}"])

sasa_0 = set((R.resnum, R.segid) for R in protein.residues)
with open("test_data/SASA_30.pkl", "rb") as f:
    sasa_30 = pickle.load(f)
sasa_ans = [sasa_30, sasa_0]


def test_unfiltered_dd():
    SL1 = type("PseudoLabel", (object,), {"spin_coords": None, "weights": None})
    SL2 = type("PseudoLabel", (object,), {"spin_coords": None, "weights": None})
    SL1.spin_centers = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    SL2.spin_centers = np.array([[0.0, 0.0, 20.0], [0.0, 40.0, 0.0], [60.0, 0.0, 0.0]])
    SL1.weights = np.array([0.5, 0.5])
    SL2.weights = np.array([0.1, 0.3, 0.6])

    y = chilife.pair_dd(SL1, SL2, r=r)
    y_ans = np.load("test_data/pwdd.npy")

    np.testing.assert_almost_equal(y_ans, y)


# def test_spin_pop2():
#     SL1 = type("PseudoLabel", (object,), {"spin_coords": None, "weights": None, 'spin_weights': None})
#     SL2 = type("PseudoLabel", (object,), {"spin_coords": None, "weights": None, 'spin_weights': None})
#     SL1.spin_coords = np.array([[[0.0, 0.0, 0.0], [0.0, 0.0, 10.0]]])
#
#     SL2.spin_coords = np.array([[[0.0, 0.0, 20.0], [0.0, 0.0, 24.0]],
#                                 [[0.0, 40.0, 0.0], [0.0, 40.0, 4.0]],
#                                 [[60.0, 0.0, 0.0], [60.0, 0.0, 4.0]]])
#
#     SL1.spin_weights = np.array([0.1, 0.9])
#     SL2.spin_weights = np.array([0.5, 0.5])
#
#     SL1.weights = np.array([0.5, 0.5])
#     SL2.weights = np.array([0.1, 0.3, 0.6])
#
#     y = chilife.pair_dd(SL1, SL2, r=r, spin_populations=True)
#
#     y_ans = np.load("test_data/pwdd.npy")
#     plt.plot(r, y)
#     plt.plot(r, y_ans)
#     plt.show()
#
#     np.testing.assert_almost_equal(y_ans, y)


def test_get_lj_rmin():
    assert np.all(rmin_params == chilife.get_lj_rmin(atom_names))


def test_get_lj_eps():
    assert np.all(eps_params == chilife.get_lj_eps(atom_names))


@pytest.mark.parametrize("args, kws, expected", zip(args, kws, ans))
def test_distance_distribution(args, kws, expected):
    kws.setdefault('r', r)
    P_ans = expected
    P = chilife.distance_distribution(*args, **kws)

    # Stored values normalized to 1 while distance_distribution normalizes to r
    np.testing.assert_almost_equal(P, P_ans)


def test_distance_distribution_dep():
    ans = np.load('test_data/dependent_dist.npy')
    SL1 = chilife.SpinLabel('R1M', 295, anf)
    SL2 = chilife.dSpinLabel('DHC', (233, 237), anf)
    r = np.linspace(2.5, 35, 256)
    P1 = chilife.distance_distribution(SL1, SL2, r)
    P2 = chilife.distance_distribution(SL1, SL2, r, dependent=True)

    assert np.sum(np.abs(P1 - P2)) > 0.2
    np.testing.assert_almost_equal(P2, ans)


def test_spin_populations():
    SL1 = chilife.SpinLabel('R1M', 211, protein)
    SL2 = chilife.SpinLabel('R1M', 275, protein)

    dd2 = chilife.distance_distribution(SL1, SL2, r, use_spin_centers=False)

    assert np.abs(dd2.max() - 0.1299553885263942) < 1e-7
    assert np.abs(r[np.argmax(dd2)] - 50.19607843137255) < 1e-7


# def test_get_dd_uq():
#     SL1 = chilife.SpinLabel("R1M", 238, protein=protein, sample=1000)
#     SL2 = chilife.SpinLabel("R1M", 20, protein=protein, sample=1000)
#
#     print(SL1.weights.sum(), SL2.weights.sum())
#     P = chilife.distance_distribution(SL1, SL2, r=r, uq=500)
#
#     mean = P.mean(axis=0)
#     # mean /= np.trapz(mean, r)
#     std = P.std(axis=0)
#     print(mean.shape, std.shape)
#
#     plt.plot(r, mean)
#     plt.fill_between(r, (mean - std / 2).clip(0), mean + std / 2, alpha=0.5)
#     plt.show()


# def test_get_dd_uq2():
#     import pickle
#
#     with open("test_data/SL1.pkl", "rb") as f:
#         SL1 = pickle.load(f)
#
#     with open("test_data/SL2.pkl", "rb") as f:
#         SL2 = pickle.load(f)
#
#     print(SL1.spin_coords, SL2)
#     P = chilife.distance_distribution(SL1, SL2, r=r, uq=500)
#
#     mean = P.mean(axis=0)
#     # mean /= np.trapz(mean, r)
#     std = P.std(axis=0)
#     print(mean.shape, std.shape)
#
#     plt.plot(r, mean)
#     plt.fill_between(r, (mean - std / 2).clip(0), mean + std / 2, alpha=0.5)
#     plt.show()


@pytest.mark.parametrize("cutoff, expected", zip([30, 0], sasa_ans))
def test_sas_res(cutoff, expected):
    sasres = chilife.get_sas_res(protein, cutoff)
    assert sasres == expected


def test_fetch_PDB():
    U1 = mda.Universe("test_data/m1omp.pdb", in_memory=True)
    U2 = chilife.fetch("1omp")

    assert np.all(U1.atoms.positions == U2.atoms.positions)


def test_fetch_AF():
    U = chilife.fetch('AF-O34208')
    ans = np.array([[-38.834, -52.705, 45.698],
                    [-38.5, -54.236, 47.741],
                    [-38.903, -51.597, 46.216],
                    [-38.123, -55.685, 48.115],
                    [-38.65, -56.141, 49.474],
                    [-38.411, -57.378, 49.85],
                    [-39.277, -55.422, 50.23],
                    [-39.532, -53.088, 44.618],
                    [-40.791, -52.617, 43.986],
                    [-42.008, -52.46, 44.944]])
    np.testing.assert_allclose(U.atoms.positions[100:110], ans)


IDs = ["1anf", "1omp.pdb", "3TU3"]
fNames = ["1anf.pdb", "1omp.pdb", "3TU3.pdb"]


@pytest.mark.parametrize("pdbid, names", zip(IDs, fNames))
def test_fetch2(pdbid, names):
    chilife.fetch(pdbid, save=True)
    with open(f"test_data/m{names}", "r") as f:
        ans = hashlib.md5(f.read().encode("utf-8")).hexdigest()
    with open(f"{names}", "r") as f:
        test = hashlib.md5(f.read().encode("utf-8")).hexdigest()
    assert ans == test
    os.remove(names)


def test_repack():
    np.random.seed(1000)
    protein = mda.Universe("test_data/1ubq.pdb", in_memory=True).select_atoms("protein")
    SL = chilife.SpinLabel("R1C", site=28, protein=protein)

    traj1, deltaE1 = chilife.repack(protein, SL, repetitions=10, repack_radius=10)
    traj2, deltaE2 = chilife.repack(protein, SL, repetitions=10, off_rotamer=True, repack_radius=10)

    t1coords = traj1.universe.trajectory.coordinate_array
    t2coords = traj2.universe.trajectory.coordinate_array

    with np.load("test_data/repack_ans.npz") as f:
        t1ans = f["traj1"]
        t2ans = f["traj2"]

    np.testing.assert_almost_equal(t1coords, t1ans, decimal=5)
    np.testing.assert_almost_equal(t2coords, t2ans, decimal=5)


def test_create_library():
    chilife.create_library(
        libname="___",
        resname="___",
        pdb="test_data/trt_sorted.pdb",
        dihedral_atoms=[['N', 'CA', 'CB', 'SG'],
                        ['CA', 'CB', 'SG', 'SD'],
                        ["CB", "SG", "SD", "CAD"],
                        ["SG", "SD", "CAD", "CAE"],
                        ["SD", "CAD", "CAE", "OAC"]],
        spin_atoms="CAQ",
        permanent=True
    )

    SL = chilife.SpinLabel('___', 238, protein)
    np.testing.assert_allclose(SL.spin_centers, np.array([[-24.44447989, 15.0841676, 14.75408798]]))

    struct = mda.Universe('test_data/trt_sorted.pdb')
    pos = struct.atoms[:3].positions
    SL.to_site(pos)
    assert "___" in chilife.USER_LIBRARIES
    np.testing.assert_almost_equal(SL.coords[0, 5:], struct.atoms.positions[5:], decimal=3)

    chilife.remove_library('___', prompt=False)
    assert "___" not in chilife.USER_LIBRARIES

    os.remove('____rotlib.npz')


def test_single_chain_error():
    with pytest.raises(ValueError):
        chilife.create_dlibrary(libname='___',
                                pdb='test_data/chain_broken_dlabel.pdb',
                                sites=(15, 17),
                                dihedral_atoms=[[['N', 'CA', 'C13', 'C5'],
                                                 ['CA', 'C13', 'C5', 'C6']],
                                                [['N', 'CA', 'C13', 'C5'],
                                                 ['CA', 'C13', 'C5', 'C6']]],
                                spin_atoms='Cu1')


def test_set_params():
    chilife.set_lj_params("uff")
    assert 3.851 == chilife.get_lj_rmin("C")
    assert -0.105 == chilife.get_lj_eps("C")
    assert chilife.get_lj_rmin("join_protocol")[()] == chilife.join_geom
    chilife.set_lj_params("charmm")


@pytest.mark.parametrize('key', from_mmm_rotlibs.keys())
def test_from_MMM(key):
    SL = from_mmm_SLs[key]
    ans_coords, ans_weights = from_mmm_rotlibs[key]

    np.testing.assert_almost_equal(SL.weights, ans_weights, decimal=6)
    np.testing.assert_almost_equal(SL.coords, ans_coords, decimal=4)


@pytest.mark.parametrize('key', from_mmm_Ps.keys())
def test_MMM_dd(key):
    label, site1, site2, state = key
    Pans = from_mmm_Ps[key]
    SL1 = from_mmm_SLs[label, site1, state]
    SL2 = from_mmm_SLs[label, site2, state]

    Ptest = chilife.distance_distribution(SL1, SL2, from_mmm_r)
    np.testing.assert_almost_equal(Ptest, Pans, decimal=6)


def test_save():
    L20R1 = chilife.SpinLabel("R1C", 20, protein)
    S238T = chilife.RotamerEnsemble("THR", 238, protein)
    A318DHC = chilife.dSpinLabel("DHC", [318, 322], protein)

    chilife.save(L20R1, S238T, A318DHC, protein, KDE=False)

    with open(f"test_data/test_save.pdb", "r") as f:
        ans = hashlib.md5(f.read().encode("utf-8")).hexdigest()

    with open("No_Name_Protein_many_labels.pdb", "r") as f:
        test = hashlib.md5(f.read().encode("utf-8")).hexdigest()

    os.remove("No_Name_Protein_many_labels.pdb")

    assert ans == test


def test_save_fail():
    with pytest.raises(TypeError):
        chilife.save("tmp", np.array([1, 2, 3]))


def test_use_local_library():
    weights = np.loadtxt(f'test_data/R1M.weights')
    dihedral_names = np.loadtxt(f'test_data/R1M.chi', dtype='U').tolist()
    spin_atoms = ['N1', 'O1']
    chilife.create_library('RXL', f'test_data/R1M.pdb',
                           dihedral_atoms=dihedral_names,
                           spin_atoms=spin_atoms,
                           weights=weights)

    SLAns = chilife.SpinLabel('R1M')
    for name in ('RXL', 'RXL_rotlib', 'RXL_rotlib.npz'):
        SL = chilife.SpinLabel(name)
        np.testing.assert_almost_equal(SL.coords, SLAns.coords, decimal=5)
        np.testing.assert_almost_equal(SL.weights, SLAns.weights)

    assert 'RXL' not in chilife.USER_LIBRARIES

    os.remove('RXL_rotlib.npz')
    chilife.read_rotlib.cache_clear()


def test_add_new_default_library():
    weights = np.ones(216) / 216
    oweights = np.loadtxt(f'test_data/R1M.weights')
    dihedral_names = np.loadtxt(f'test_data/R1M.chi', dtype='U').tolist()
    spin_atoms = ['N1', 'O1']
    chilife.create_library('RXL', f'test_data/R1M.pdb', resname='R1M',
                           dihedral_atoms=dihedral_names, spin_atoms=spin_atoms,
                           weights=weights, permanent=True, default=True)
    os.remove('RXL_rotlib.npz')

    # Check that added label is now the default
    SL = chilife.SpinLabel('R1M')
    assert SL.rotlib == 'RXL'
    assert np.all(SL.weights == weights)

    # Check that the old default is still available and has not changed
    SL2 = chilife.SpinLabel('R1M', rotlib='R1M')
    assert SL2.rotlib == 'R1M'
    assert np.all(oweights == SL2.weights)
    assert np.all(SL2.weights != weights)

    # Check that, after removing the new default, the old default becomes the actual default again
    chilife.remove_library('RXL', prompt=False)
    SL = chilife.SpinLabel('R1M')
    assert np.all(weights != SL.weights)
    assert np.all(oweights == SL.weights)


def test_use_secondary_label():
    weights = np.ones(216) / 216
    oweights = np.loadtxt(f'test_data/R1M.weights')
    dihedral_names = np.loadtxt(f'test_data/R1M.chi', dtype='U').tolist()
    spin_atoms = ['N1', 'O1']
    chilife.create_library('RXL', f'test_data/R1M.pdb', resname='R1M',
                           dihedral_atoms=dihedral_names, spin_atoms=spin_atoms,
                           weights=weights, permanent=True)
    os.remove('RXL_rotlib.npz')

    # Check that new rotlib is available
    SLsecond = chilife.SpinLabel('R1M', rotlib='RXL')
    assert SLsecond.rotlib == 'RXL'
    assert np.all(SLsecond.weights == weights)

    # Check that default label still acts as default
    SLdefault = chilife.SpinLabel('R1M')
    assert SLdefault.rotlib == 'R1M'
    assert np.all(oweights == SLdefault.weights)
    assert np.all(SLdefault.weights != weights)

    chilife.remove_library('RXL', prompt=False)


def test_add_library_fail():
    with pytest.raises(NameError):
        weights = np.loadtxt(f'test_data/R1M.weights')
        dihedral_names = np.loadtxt(f'test_data/R1M.chi', dtype='U').tolist()
        spin_atoms = ['N1', 'O1']
        chilife.create_library('R1M', f'test_data/R1M.pdb', resname='R1M',
                               dihedral_atoms=dihedral_names, spin_atoms=spin_atoms,
                               weights=weights, permanent=True)
    os.remove('R1M_rotlib.npz')


def test_add_rotlib_dir():
    chilife.add_rotlib_dir('test_data/usr_rtlb')
    SL = chilife.SpinLabel('XYZ', 211, protein)
    chilife.remove_rotlib_dir('test_data/usr_rtlb')
    with pytest.raises(NameError):
        SL = chilife.SpinLabel('XYZ', 211, protein)


def test_list_rotlibs():
    chilife.list_available_rotlibs()


def test_create_library_diff():
    spin_atoms = ['N1', 'O1']
    dihedral_atoms = [['N', 'CA', 'CB', 'SG'],
                      ['CA', 'CB', 'SG', 'CD'],
                      ['CB', 'SG', 'CD', 'C3'],
                      ['SG', 'CD', 'C3', 'C4']]

    chilife.create_library('R3A', 'test_data/R3A_Ensemble.pdb', site=2, dihedral_atoms=dihedral_atoms,
                           spin_atoms=spin_atoms)
    os.remove('R3A_rotlib.npz')


def test_chilife_mplstyle():
    mlp_stylelib_path = Path(mpl.get_configdir()) / 'stylelib'

    style_files = list(Path("../mplstyles/").glob("*.mplstyle"))
    assert len(style_files) > 0
    for style_file in style_files:
        test_file = mlp_stylelib_path / style_file.name
        print(test_file)
        assert test_file.exists()

    plt.style.use('chiLife')