import hashlib, os, pickle
from functools import partial
import numpy as np
import ProEPR
import pytest

ubq = ProEPR.fetch('1ubq')
U = ProEPR.fetch('1omp')
hashes = {}
with open('test_data/hashes.txt', 'r') as f:
    for line in f:
        if len(line.split()) == 2:
            a, b = line.split()
            hashes[a] = b


def test_from_mda():
    res16 = U.residues[16]
    rotlib = ProEPR.RotamerLibrary.from_mda(res16)
    rotlib.save_pdb('test_data/test_from_MDA.pdb')

    with open('test_data/ans_from_MDA.pdb', 'r') as f:
        ans = hashlib.md5(f.read().encode('utf-8')).hexdigest()

    with open('test_data/test_from_MDA.pdb', 'r') as f:
        test = hashlib.md5(f.read().encode('utf-8')).hexdigest()

    os.remove('test_data/test_from_MDA.pdb')
    assert ans == test


def test_with_sample():
    np.random.seed(200)

    SL = ProEPR.SpinLabel('R1C', 28, ubq, sample=2000, energy_func=partial(ProEPR.get_lj_rep, forgive=0.8))

    with open('test_data/withsample.pkl', 'rb') as f:
        SLans = pickle.load(f)

    np.testing.assert_almost_equal(SL.coords, SLans.coords)
    np.testing.assert_almost_equal(SL.weights, SLans.weights)
    np.testing.assert_almost_equal(SL.dihedrals, SLans.dihedrals)
    assert len(SL.coords) == len(SL.weights) == len(SL.dihedrals) == len(SL.internal_coords)

def test_user_label():
    SL = ProEPR.SpinLabel('TRT', 28, ubq, 'A')
    ProEPR.save('test_data/1ubq_28TRT.pdb', SL, protein='test_data/1ubq.pdb', KDE=False)
    ans = hashes['1ubq_28TRT.pdb']

    with open('test_data/1ubq_28TRT.pdb', 'rb') as f:
        test = hashlib.md5(f.read()).hexdigest()

    os.remove('test_data/1ubq_28TRT.pdb')

    assert test == ans


def test_save_pkl():
    Lys = ProEPR.RotamerLibrary('LYS')

    with open('test_data/Lys.pkl', 'wb') as f:
        pickle.dump(Lys, f)

    with open('test_data/Lys.pkl', 'rb') as f:
        reload = pickle.load(f)

    os.remove('test_data/Lys.pkl')

    assert Lys == reload


def test_save_pkl_2():
    res16 = U.residues[16]
    rotlib = ProEPR.RotamerLibrary.from_mda(res16)
    with open('test_data/res16.pkl', 'wb') as f:
        pickle.dump(rotlib, f)

    with open('test_data/res16.pkl', 'rb') as f:
        reload = pickle.load(f)

    assert rotlib == reload
    os.remove('test_data/res16.pkl')


def test_sample():
    np.random.seed(200)
    ubq = ProEPR.fetch('1ubq')
    K48 = ProEPR.RotamerLibrary.from_mda(ubq.residues[47])
    coords, weight = K48.sample(off_rotamer=True)

    wans = 0.0037019925960148086
    cans = np.array([[20.7640662,  27.90259646, 22.59445096],
                     [21.54999924, 26.79599953, 23.13299942],
                     [23.02842996, 27.03777002, 22.93937713],
                     [23.46033135, 28.06053558, 22.38830811],
                     [21.11010724, 25.47090289, 22.45317528],
                     [21.14878991, 25.52995343, 20.90566928],
                     [20.5939163,  24.28947355, 20.19950785],
                     [21.46383773, 23.07557031, 20.55152114],
                     [20.74675938, 21.83900512, 20.19455276]])

    np.testing.assert_almost_equal(coords, cans)
    assert weight == wans


def test_multisample():
    ubq = ProEPR.fetch('1ubq')
    R1M = ProEPR.SpinLabel.from_wizard('R1M', site=48, protein=ubq, to_find=10000)


@pytest.mark.parametrize('res', ProEPR.SUPPORTED_RESIDUES)
def test_lib_distribution_persists(res):
    if res in list(ProEPR.SUPPORTED_LABELS) + list(ProEPR.USER_LABELS):
        L1 = ProEPR.SpinLabel(res)
        L2 = ProEPR.SpinLabel(res, sample=100)
    else:
        L1 = ProEPR.RotamerLibrary(res)
        L2 = ProEPR.RotamerLibrary(res, sample=100)

    np.testing.assert_almost_equal(L1._rdihedrals, L2._rdihedrals)
    np.testing.assert_almost_equal(L1._rkappas, L2._rkappas)
    np.testing.assert_almost_equal(L1._weights, L2._weights)

methods = ['rosetta', 'bisect', 'mmm', 'fit']
@pytest.mark.parametrize(('method'), methods)
def test_superimposition_method(method):
    if method == 'fit':
        with pytest.raises(NotImplementedError) as e_info:
            SL = ProEPR.SpinLabel('R1C', site=28, protein=ubq, superimposition_method=method)
    else:
        SL = ProEPR.SpinLabel('R1C', site=28, protein=ubq,
                              superimposition_method=method,
                              energy_func=partial(ProEPR.get_lj_rep, forgive=0.8))
        ProEPR.save(f'A28R1_{method}_superimposition.pdb', SL, 'test_data/1ubq.pdb', KDE=False)

        with open(f'A28R1_{method}_superimposition.pdb', 'rb') as f:
            test = hashlib.md5(f.read()).hexdigest()

        with open(f'test_data/A28R1_{method}_superimposition.pdb', 'rb') as f:
            ans = hashlib.md5(f.read()).hexdigest()

        os.remove(f'A28R1_{method}_superimposition.pdb')
        assert ans == test
