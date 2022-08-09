import pickle, hashlib, os
import numpy as np
import pytest
import MDAnalysis as mda
import chiLife
import matplotlib.pyplot as plt

labels = ["R1M", "R7M", "V1M", "M1M", "I1M"]
atom_names = ["C", "H", "N", "O", "S", "SE", "Br"]
rmin_params = [2.0000, 1.27, 1.8500, 1.7000, 2.0000, 2.0000, 1.9800]
eps_params = [-0.110, -0.022, -0.200, -0.120, -0.450, -0.450, -0.320]
U = mda.Universe("test_data/m1omp.pdb")
protein = U.select_atoms("protein")
r = np.linspace(0, 100, 2**8)
old_ef = lambda protein, rotlib: chiLife.get_lj_rep(protein, rotlib, forgive=0.8)

# get permutations for get_dd() tests
kws = []
args = []
for SL in labels:
    site1, site2 = 25, 73
    chain = "A"
    SL1 = chiLife.SpinLabel(SL, site1, protein, chain, energy_func=old_ef)
    SL2 = chiLife.SpinLabel(SL, site2, protein, chain, energy_func=old_ef)
    [args.append([SL1, SL2]) for i in range(6)]

    kws.append({"prune": True})
    kws.append({"r": 80, "prune": True})
    kws.append({"r": (20, 80), "prune": True})
    kws.append({"r": r, "prune": True})
    kws.append({"prune": True})
    kws.append({"prune": 0.0001})

ans = []
with np.load("test_data/get_dd_tests.npz", allow_pickle=True) as f:
    for i in range(30):
        ans.append(f[f"arr_{i}"])

sasa_0 = set((R.resnum, R.segid) for R in protein.residues)
with open("test_data/SASA_30.pkl", "rb") as f:
    sasa_30 = pickle.load(f)
sasa_ans = [sasa_30, sasa_0]


def test_unfiltered_dd():
    SL1 = type("PseudoLabel", (object,), {"spin_coords": None, "weights": None})
    SL2 = type("PseudoLabel", (object,), {"spin_coords": None, "weights": None})
    SL1.spin_coords = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    SL2.spin_coords = np.array([[0.0, 0.0, 20.0], [0.0, 40.0, 0.0], [60.0, 0.0, 0.0]])
    SL1.weights = np.array([0.5, 0.5])
    SL2.weights = np.array([0.1, 0.3, 0.6])

    y = chiLife.unfiltered_dd(SL1, SL2, r=r)

    y_ans = np.load("test_data/pwdd.npy")
    plt.plot(y)
    plt.plot(y_ans)
    plt.show()

    np.testing.assert_almost_equal(y_ans, y)


@pytest.mark.parametrize("label", labels)
def test_read_rotamer_library(label):
    data = chiLife.read_sl_library(label)
    print(data.keys())
    data = [
        data[key]
        for key in data
        if key not in ("internal_coords", "sigmas", "_rdihedrals", "_rsigmas")
    ]
    print(len(data))
    for i, array in enumerate(data):
        assert np.all(array == np.load(f"test_data/{label}_{i}.npy", allow_pickle=True))


def test_get_lj_rmin():
    assert np.all(rmin_params == chiLife.get_lj_rmin(atom_names))


def test_get_lj_eps():
    assert np.all(eps_params == chiLife.get_lj_eps(atom_names))


def test_filter_by_weight():
    w1 = np.array([0, 0.001, 0.01, 0.1, 0.9])
    w2 = w1.copy()

    weights, idx = chiLife.filter_by_weight(w1, w2)

    ans_idx = np.array([[2, 4], [3, 3], [3, 4], [4, 2], [4, 3], [4, 4]])
    ans_weights = np.array([0.009, 0.01, 0.09, 0.009, 0.09, 0.81])
    np.testing.assert_almost_equal(ans_idx, idx)
    np.testing.assert_almost_equal(ans_weights, weights)


def test_filtered_dd():
    NO1 = np.array([[0.0, 0.0, 0.0]])
    NO2 = np.array([[0.0, 0.0, 20.0], [0.0, 40.0, 0.0], [60.0, 0.0, 0.0]])
    w1 = [1.0]
    w2 = [0.1, 0.3, 0.6]

    weights, idx = chiLife.filter_by_weight(w1, w2)
    NO1, NO2 = NO1[idx[:, 0]], NO2[idx[:, 1]]
    y = chiLife.filtered_dd(NO1, NO2, weights, r)
    ans_y = np.load("test_data/get_dist_dist.npy")
    np.testing.assert_almost_equal(ans_y, y)


@pytest.mark.parametrize("args, kws, expected", zip(args, kws, ans))
def test_get_dd(args, kws, expected):
    y_ans = expected
    y = chiLife.get_dd(*args, **kws)

    # Stored values normalized to 1 while get_dd normalizes to r
    np.testing.assert_almost_equal(y_ans, y / y.sum())


def test_get_dd_uq():
    SL1 = chiLife.SpinLabel("R1M", 238, protein=protein, sample=1000)
    SL2 = chiLife.SpinLabel("R1M", 20, protein=protein, sample=1000)

    print(SL1.weights.sum(), SL2.weights.sum())
    P = chiLife.get_dd(SL1, SL2, r=r, uq=500)

    mean = P.mean(axis=0)
    # mean /= np.trapz(mean, r)
    std = P.std(axis=0)
    print(mean.shape, std.shape)

    plt.plot(r, mean)
    plt.fill_between(r, (mean - std / 2).clip(0), mean + std / 2, alpha=0.5)
    plt.show()


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
#     P = chiLife.get_dd(SL1, SL2, r=r, uq=500)
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
    sasres = chiLife.get_sas_res(protein, cutoff)
    assert sasres == expected


def test_fetch():
    U1 = mda.Universe("test_data/m1omp.pdb", in_memory=True)
    U2 = chiLife.fetch("1omp")

    assert np.all(U1.atoms.positions == U2.atoms.positions)


IDs = ["1anf", "1omp.pdb", "3TU3"]
fNames = ["1anf.pdb", "1omp.pdb", "3TU3.pdb"]


@pytest.mark.parametrize("pdbid, names", zip(IDs, fNames))
def test_fetch2(pdbid, names):
    chiLife.fetch(pdbid, save=True)
    with open(f"test_data/m{names}", "r") as f:
        ans = hashlib.md5(f.read().encode("utf-8")).hexdigest()
    with open(f"{names}", "r") as f:
        test = hashlib.md5(f.read().encode("utf-8")).hexdigest()
    assert ans == test
    os.remove(names)


def test_repack():
    np.random.seed(1000)
    protein = chiLife.fetch("1ubq").select_atoms("protein")
    SL = chiLife.SpinLabel("R1C", site=28, protein=protein)

    traj1, deltaE1 = chiLife.repack(protein, SL, repetitions=50)
    traj2, deltaE2 = chiLife.repack(protein, SL, repetitions=50, off_rotamer=True)

    t1coords = traj1.universe.trajectory.coordinate_array
    t2coords = traj2.universe.trajectory.coordinate_array

    with np.load("test_data/repack_ans.npz") as f:
        t1ans = f["traj1"]
        t2ans = f["traj2"]

    np.testing.assert_almost_equal(t1coords, t1ans, decimal=5)
    np.testing.assert_almost_equal(t2coords, t2ans, decimal=5)


def test_repack_add_atoms():
    omp = mda.Universe("test_data/1omp_H.pdb")
    SL1 = chiLife.SpinLabel('R1M', 20, omp, use_H=True)
    SL2 = chiLife.SpinLabel('R1M', 211, omp, use_H=True)
    traj, de = chiLife.repack(omp, SL1, SL2, repetitions=10)


def test_add_label():
    chiLife.add_label(
        name="TRT",
        pdb="test_data/trt_sorted.pdb",
        dihedral_atoms=[['N', 'CA', 'CB', 'SG'],
                        ['CA', 'CB', 'SG', 'SD'],
                        ["CB", "SG", "SD", "CAD"],
                        ["SG", "SD", "CAD", "CAE"],
                        ["SD", "CAD", "CAE", "OAC"]],
        spin_atoms="CAQ",
    )

    assert "TRT" in chiLife.USER_LABELS
    assert "TRT" in chiLife.SPIN_ATOMS


def test_set_params():
    chiLife.set_lj_params("uff")
    assert 3.851 == chiLife.get_lj_rmin("C")
    assert -0.105 == chiLife.get_lj_eps("C")
    assert chiLife.get_lj_rmin("join_protocol")[()] == chiLife.join_geom
    chiLife.set_lj_params("charmm")


def test_MMM():
    omp = chiLife.fetch("1omp")
    SL1 = chiLife.SpinLabel.from_mmm("R1M", 238, protein=omp)

    ans = [
        0.007018871557921462,
        0.007611321209884533,
        0.00874935025052478,
        0.010485333838458726,
        0.028454490756688582,
        0.05022215305253955,
        0.05461865851164981,
        0.05994927468416553,
        0.09271821128592427,
        0.09877155092812952,
        0.17535252824480543,
        0.18778707975414155,
        0.2182611759251661,
    ]

    np.testing.assert_almost_equal(SL1.weights, ans[::-1])

    chiLife.set_lj_params("charmm")


def test_save():
    L20R1 = chiLife.SpinLabel("R1C", 20, protein)
    S238T = chiLife.RotamerLibrary("THR", 238, protein)
    A318DHC = chiLife.dSpinLabel("DHC", [318, 322], protein)

    chiLife.save(L20R1, S238T, A318DHC, protein, KDE=False)

    with open(f"test_data/test_save.pdb", "r") as f:
        ans = hashlib.md5(f.read().encode("utf-8")).hexdigest()

    with open("No_Name_Protein_many_labels.pdb", "r") as f:
        test = hashlib.md5(f.read().encode("utf-8")).hexdigest()

    os.remove("No_Name_Protein_many_labels.pdb")

    assert ans == test


def test_save_fail():
    with pytest.raises(TypeError):
        chiLife.save("tmp", np.array([1, 2, 3]))

# class TestProtein:
#
#     def test_load(self):
#         protein = chiLife.Protein.from_pdb('test_data/3tu3.pdb')
#
#     def test_load_multistate(self):
#         protein = chiLife.Protein.from_pdb('test_data/2d21.pdb')
