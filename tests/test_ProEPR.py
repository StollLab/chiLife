import pickle, hashlib, os
import numpy as np
import pytest
import MDAnalysis as mda
import ProEPR
import matplotlib.pyplot as plt

labels = ['R1M', 'R7M', 'V1M', 'M1M', 'I1M']
atom_names = ["C", "H", "N", "O", "S", "SE", "Br"]
rmin_params = [2.0000, 0.2245, 1.8500, 1.7000, 2.0000, 2.0000, 1.9800]
eps_params = [-0.110, -0.046, -0.200, -0.120, -0.450, -0.450, -0.320]
U = mda.Universe('test_data/m1omp.pdb')
protein = U.select_atoms('protein')
r = np.linspace(0, 100, 2 ** 8)

# get permutations for get_dd() tests
kws = []
args = []
for SL in labels:
    site1, site2 = 25, 73
    chain = 'A'
    SL1 = ProEPR.SpinLabel(SL, site1, chain, protein)
    SL2 = ProEPR.SpinLabel(SL, site2, chain, protein)
    [args.append([SL1, SL2]) for i in range(6)]

    kws.append({'prune': True})
    kws.append({'r': 80, 'prune': True})
    kws.append({'r': (20, 80), 'prune': True})
    kws.append({'r': r, 'prune': True})
    kws.append({'prune': True})
    kws.append({'prune': 0.0001})

ans = []
with np.load('test_data/get_dd_tests.npz', allow_pickle=True) as f:
    for i in range(30):
        ans.append(f[f'arr_{i}'])

sasa_0 = set((R.resnum, R.segid) for R in protein.residues)
with open('test_data/SASA_30.pkl', 'rb') as f:
    sasa_30 = pickle.load(f)
sasa_ans = [sasa_30, sasa_0]


def test_unfiltered_dd():
    SL1 = type('PseudoLabel', (object,), {'spin_coords': None, 'weights': None})
    SL2 = type('PseudoLabel', (object,), {'spin_coords': None, 'weights': None})
    SL1.spin_coords = np.array([[0., 0., 0.], [0., 0., 1.]])
    SL2.spin_coords = np.array([[0., 0., 20.], [0., 40., 0.], [60., 0., 0.]])
    SL1.weights = np.array([0.5, 0.5])
    SL2.weights = np.array([0.1, 0.3, 0.6])

    y = ProEPR.unfiltered_dd(SL1, SL2, r=r)

    y_ans = np.load('test_data/pwdd.npy')
    plt.plot(y)
    plt.plot(y_ans)
    plt.show()

    np.testing.assert_almost_equal(y_ans, y)


@pytest.mark.parametrize('label', labels)
def test_read_rotamer_library(label):
    data = ProEPR.read_sl_library(label)
    print(data.keys())
    data = [data[key] for key in data if key not in ('internal_coords', 'sigmas')]
    print(len(data))
    for i, array in enumerate(data):
        assert np.all(array == np.load(f'test_data/{label}_{i}.npy', allow_pickle=True))


def test_get_lj_rmin():
    assert np.all(rmin_params == ProEPR.get_lj_rmin(atom_names))


def test_get_lj_eps():
    assert np.all(eps_params == ProEPR.get_lj_eps(atom_names))


def test_filter_by_weight():
    w1 = np.array([0, 0.001, 0.01, 0.1, 0.9])
    w2 = w1.copy()

    weights, idx = ProEPR.filter_by_weight(w1, w2)

    ans_idx = np.array([[2, 4], [3, 3], [3, 4], [4, 2], [4, 3], [4, 4]])
    ans_weights = np.array([0.009, 0.01, 0.09, 0.009, 0.09, 0.81])
    np.testing.assert_almost_equal(ans_idx, idx)
    np.testing.assert_almost_equal(ans_weights, weights)


def test_filtered_dd():
    NO1 = np.array([[0., 0., 0.]])
    NO2 = np.array([[0., 0., 20.], [0., 40., 0.], [60., 0., 0.]])
    w1 = [1.0]
    w2 = [0.1, 0.3, 0.6]

    weights, idx = ProEPR.filter_by_weight(w1, w2)
    NO1, NO2 = NO1[idx[:, 0]], NO2[idx[:, 1]]
    y = ProEPR.filtered_dd(NO1, NO2, weights, r)
    ans_y = np.load('test_data/get_dist_dist.npy')
    np.testing.assert_almost_equal(ans_y, y)


@pytest.mark.parametrize('args, kws, expected', zip(args, kws, ans))
def test_get_dd(args, kws, expected):
    y_ans = expected
    y = ProEPR.get_dd(*args, **kws)

    np.testing.assert_almost_equal(y_ans, y)

def test_det_dd_uq():
    SL1 = ProEPR.SpinLabel('R1M', 238, protein=protein, sample=10000)
    SL2 = ProEPR.SpinLabel('R1M', 20, protein=protein, sample=10000)

    print(SL1.weights.sum(), SL2.weights.sum())
    P = ProEPR.get_dd(SL1, SL2, r=r, uq=1000)

    mean = P.mean(axis=0)
    # mean /= np.trapz(mean, r)
    std = P.std(axis=0)
    print(mean.shape, std.shape)

    plt.plot(r, mean)
    plt.fill_between(r, (mean - std/2).clip(0), mean + std/2, alpha=0.5)
    plt.show()


def test_det_dd_uq2():
    import pickle
    with open('test_data/SL1.pkl', 'rb') as f:
        SL1 = pickle.load(f)

    with open('test_data/SL2.pkl', 'rb') as f:
        SL2 = pickle.load(f)

    print(SL1.spin_coords, SL2)
    P = ProEPR.get_dd(SL1, SL2, r=r, uq=1000)

    mean = P.mean(axis=0)
    # mean /= np.trapz(mean, r)
    std = P.std(axis=0)
    print(mean.shape, std.shape)

    plt.plot(r, mean)
    plt.fill_between(r, (mean - std/2).clip(0), mean + std/2, alpha=0.5)
    plt.show()


@pytest.mark.parametrize('cutoff, expected', zip([30, 0], sasa_ans))
def test_sas_res(cutoff, expected):
    sasres = ProEPR.get_sas_res(protein, cutoff)
    assert sasres == expected


def test_fetch():
    U1 = mda.Universe('test_data/m1omp.pdb', in_memory=True)
    U2 = ProEPR.fetch('1omp')

    assert np.all(U1.atoms.positions == U2.atoms.positions)


IDs = ['1anf', '1omp.pdb', '3TU3']
fNames = ['1anf.pdb', '1omp.pdb', '3TU3.pdb']


@pytest.mark.parametrize('pdbid, names', zip(IDs, fNames))
def test_fetch2(pdbid, names):
    ProEPR.fetch(pdbid, save=True)
    with open(f'test_data/m{names}', 'r') as f:
        ans = hashlib.md5(f.read().encode('utf-8')).hexdigest()
    with open(f'{names}', 'r') as f:
        test = hashlib.md5(f.read().encode('utf-8')).hexdigest()
    assert ans == test
    os.remove(names)


def test_repack():
    np.random.seed(1000)
    protein = ProEPR.fetch('1ubq').select_atoms('protein')
    SL = ProEPR.SpinLabel('R1C', site=28, protein=protein, use_bbdep=True)

    traj1, deltaE1, _ = ProEPR.repack(protein, SL, repetitions=100)
    traj2, deltaE2, _ = ProEPR.repack(protein, SL, repetitions=100, off_rotamer=True)

    t1coords = traj1.universe.trajectory.coordinate_array[-10:]
    t2coords = traj2.universe.trajectory.coordinate_array[-10:]

    with np.load('test_data/repack_ans.npz') as f:
        t1ans = f['traj1']
        t2ans = f['traj2']

    np.testing.assert_almost_equal(t1coords, t1ans, decimal=5)
    np.testing.assert_almost_equal(t2coords, t2ans, decimal=5)


def test_repack2():
    SL1 = ProEPR.SpinLabel('R1C', site=28, protein=protein, use_bbdep=True)
    SL2 = ProEPR.SpinLabel('R1C', site=48, protein=protein, use_bbdep=True)
    ProEPR.repack(protein, SL1, SL2, repetitions=10)

def test_repack3():
    protein = ProEPR.fetch('1anf')
    SL1 = ProEPR.SpinLabel('R1C', site=28, protein=protein, use_bbdep=True)
    SL2 = ProEPR.SpinLabel('R1C', site=48, protein=protein, use_bbdep=True)
    ProEPR.repack(protein, SL1, SL2, repetitions=10)


def test_add_label():
    ProEPR.add_label(name='TRT',
                     pdb='test_data/trt_sorted.pdb',
                     dihedral_atoms=[['CB', 'SG', 'SD', 'CAD'],
                                     ['SG', 'SD', 'CAD', 'CAE'],
                                     ['SD', 'CAD', 'CAE', 'OAC']],
                     spin_atoms='CAQ')

    assert 'TRT' in ProEPR.USER_LABELS
    assert 'TRT' in ProEPR.SPIN_ATOMS


def test_set_params():
    ProEPR.set_lj_params('uff')
    assert 3.851 == ProEPR.get_lj_rmin('C')
    assert -0.105 == ProEPR.get_lj_eps('C')
    assert ProEPR.get_lj_rmin('join_protocol')[()] == ProEPR.join_geom
    ProEPR.set_lj_params('charmm')


def test_MMM():
    omp = ProEPR.fetch('1omp')
    SL1 = ProEPR.SpinLabel.from_mmm('R1M', 238, protein=omp)

    ans = [0.007018871557921462, 0.007611321209884533, 0.00874935025052478, 0.010485333838458726, 0.028454490756688582,
           0.05022215305253955, 0.05461865851164981, 0.05994927468416553, 0.09271821128592427, 0.09877155092812952,
           0.17535252824480543, 0.18778707975414155, 0.2182611759251661]

    np.testing.assert_almost_equal(SL1.weights, ans[::-1])

    ProEPR.set_lj_params('charmm')

# class TestProtein:
#
#     def test_load(self):
#         protein = ProEPR.Protein.from_pdb('test_data/3tu3.pdb')
#
#     def test_load_multistate(self):
#         protein = ProEPR.Protein.from_pdb('test_data/2d21.pdb')
