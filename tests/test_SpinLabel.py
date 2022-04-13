import hashlib, os, pickle
from functools import partial
import numpy as np
from scipy.spatial import cKDTree
import pytest
import chiLife

labels = ['R1M', 'R7M', 'V1M', 'M1M', 'I1M']
ubq = chiLife.fetch('1ubq')
U = chiLife.fetch('1omp')
protein = U.select_atoms('protein')

hashes = {}
with open('test_data/hashes.txt', 'r') as f:
    for line in f:
        if len(line.split()) == 2:
            a, b = line.split()
            hashes[a] = b
efunc = partial(chiLife.get_lj_rep, forgive=0.8)
kwinput = [{'label': l, 'energy_func': efunc} for l in labels] + \
          [{'label': l, 'site': 25, 'chain': 'A', 'energy_func': efunc} for l in labels] + \
          [{'label': l, 'site': 25, 'chain': 'A', 'protein': protein, 'energy_func': efunc} for l in labels] + \
          [{'label': l, 'site': 25, 'chain': 'A', 'protein': U, 'energy_func': efunc} for l in labels]
ansinput = [f'test_data/{label}_basic.npz' for label in labels] * 2 + \
           [f'test_data/{label}_advanced.npz' for label in labels] * 2


@pytest.mark.parametrize(('kwinput', 'afile'), zip(kwinput, ansinput))
def test_MMM_construction(kwinput, afile):
    SL1 = chiLife.SpinLabel(superimposition_method='MMM', **kwinput)

    with np.load(afile) as f:
        coords = f['coords']
        weights = f['weights']

    np.testing.assert_almost_equal(weights, SL1.weights, decimal=5)
    np.testing.assert_almost_equal(coords, SL1.coords, decimal=5)


def test_bbdep_construction1():
    efunc = partial(chiLife.get_lj_energy, cap=np.inf)
    SL1 = chiLife.SpinLabel('R1C', site=211, chain='A', protein=U, energy_func=efunc, forgive=0.5)
    SL2 = chiLife.SpinLabel('R1C', site=295, chain='A', protein=U, energy_func=efunc, forgive=0.5)

    chiLife.save('test_data/bbdep_test.pdb', SL1, SL2, protein='test_data/1omp.pdb')

    with open('test_data/bbdep_test.pdb', 'rb') as f:
        test = hashlib.md5(f.read()).hexdigest()

    os.remove('test_data/bbdep_test.pdb')

    assert hashes['bbdep_test.pdb'] == test


@pytest.mark.parametrize('label', labels)
def test_eq(label):
    SL1 = chiLife.SpinLabel(label)
    SL2 = chiLife.SpinLabel(label, site=25)
    SL3 = chiLife.SpinLabel(label, site=25, chain='A')
    SL4 = chiLife.SpinLabel(label, site=25, chain='A', protein=protein)
    assert SL1 == SL2 == SL3 != SL4


@pytest.mark.parametrize('label', labels)
def test_lib2site(label):
    site = np.array([[-7.882, -3.264, -26.954],
                     [-8.528, -3.130, -28.260],
                     [-7.611, -2.318, -29.157]])

    lib = chiLife.SpinLabel(label)
    lib._to_site(site)
    ans = np.load(f'test_data/{label}_lib2site.npy')
    np.testing.assert_almost_equal(ans, lib.coords)


@pytest.mark.parametrize('label', labels)
def test_evaluate_clashes(label):
    site1, chain = 25, 'A'

    environment = U.select_atoms(f'protein and not (segid {chain} and resid {site1})')
    environment_tree = cKDTree(environment.positions)

    lib = chiLife.SpinLabel(label, site1, chain=chain, superimposition_method='mmm', energy_func=efunc)
    lib._to_site(U.select_atoms(f'protein and segid {chain} and resid {site1} and name N CA C').positions)
    lib.evaluate_clashes(environment, environment_tree, temp=298)
    with np.load(f'test_data/{label}_label.npz') as f:
        np.testing.assert_almost_equal(f['coords'], lib.coords, decimal=5)
        np.testing.assert_almost_equal(f['weights'], lib.weights, decimal=5)


def test_from_wizard():
    np.random.seed(200)
    SL = chiLife.SpinLabel.from_wizard('TRT', 28, ubq, to_find=10)

    with open('test_data/from_wiz.pkl', 'rb') as f:
        SLans = pickle.load(f)

    np.testing.assert_almost_equal(SL.coords, SLans.coords, decimal=5)
    np.testing.assert_almost_equal(SL.weights, SLans.weights)
    np.testing.assert_almost_equal(SL.dihedrals, SLans.dihedrals)
    assert len(SL) == len(SL.dihedrals) == len(SL.weights) == len(SL.coords)

def test_minimize():

    SL1 = chiLife.SpinLabel('R1C', 238, protein=U, eval_clash=False)
    SL1.minimize()