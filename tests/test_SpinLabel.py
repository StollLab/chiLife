import hashlib, os, pickle
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import MDAnalysis as mda
import pytest
import chilife

labels = ["R1M", "R7M", "V1M", "M1M", "I1M"]
ubq = mda.Universe("test_data/1ubq.pdb", in_memory=True)
U = mda.Universe("test_data/1omp.pdb", in_memory=True)
protein = U.select_atoms("protein")

hashes = {}
with open("test_data/hashes.txt", "r") as f:
    for line in f:
        if len(line.split()) == 2:
            a, b = line.split()
            hashes[a] = b
efunc = partial(chilife.get_lj_rep, forgive=0.8)
kwinput = (
    [{"label": l, "energy_func": efunc} for l in labels]
    + [{"label": l, "site": 25, "chain": "A", "energy_func": efunc} for l in labels]
    + [
        {"label": l, "site": 25, "chain": "A", "protein": protein, "energy_func": efunc}
        for l in labels
    ]
    + [
        {"label": l, "site": 25, "chain": "A", "protein": U, "energy_func": efunc}
        for l in labels
    ]
)


def test_bbdep_construction1():
    efunc = chilife.ljEnergyFunc(chilife.get_lj_energy, forgive=0.5, cap=np.inf)
    SL1 = chilife.SpinLabel("R1C", site=211, chain="A", protein=U, energy_func=efunc)
    SL2 = chilife.SpinLabel("R1C", site=295, chain="A", protein=U, energy_func=efunc)

    chilife.save("bbdep.pdb", SL1, SL2, protein_path="test_data/1omp.pdb")

    with open("bbdep.pdb", "r") as f:
        test = hashlib.md5(f.read().encode('utf-8')).hexdigest()

    with open("test_data/bbdep.pdb", "r") as f:
        ans = hashlib.md5(f.read().encode('utf-8')).hexdigest()

    os.remove("bbdep.pdb")

    assert test == ans


@pytest.mark.parametrize("label", labels)
def test_eq(label):
    SL1 = chilife.SpinLabel(label)
    SL2 = chilife.SpinLabel(label, site=25)
    SL3 = chilife.SpinLabel(label, site=25, chain="A")
    SL4 = chilife.SpinLabel(label, site=25, chain="A", protein=protein)
    assert SL1 == SL2 == SL3 != SL4


@pytest.mark.parametrize("label", labels)
def test_lib2site(label):
    site = np.array([[-7.882, -3.264, -26.954],
                     [-8.528, -3.130, -28.260],
                     [-7.611, -2.318, -29.157]])

    lib = chilife.SpinLabel(label)
    lib.to_site(site)
    ans = np.load(f"test_data/{label}_lib2site.npy")
    np.testing.assert_almost_equal(ans, lib._coords)


@pytest.mark.parametrize("label", labels)
def test_add_protein(label):

    site1, chain = 25, "A"
    lib = chilife.SpinLabel(label, site1, chain=chain)
    lib.protein = U
    lib.protein_setup()

    with np.load(f"test_data/{label}_label.npz") as f:
        np.testing.assert_almost_equal(f["coords"], lib._coords, decimal=5)
        np.testing.assert_almost_equal(f["weights"], lib.weights)


def test_from_wizard():
    np.random.seed(200)
    SL = chilife.SpinLabel.from_wizard("TRT", 28, ubq, to_find=10)
    for ic, dihedral in zip(SL.internal_coords, SL.dihedrals):
        np.testing.assert_allclose(np.rad2deg(ic.get_dihedral(1, SL.dihedral_atoms)), dihedral)


    with np.load("test_data/from_wiz.npz") as f:
        SLans = dict(f)

    np.testing.assert_almost_equal(SL.coords, SLans['coords'], decimal=5)
    np.testing.assert_almost_equal(SL.weights, SLans['weights'])
    np.testing.assert_almost_equal(np.cos(np.deg2rad(SL.dihedrals)), np.cos(np.deg2rad(SLans["dihedrals"])), decimal=5)
    assert len(SL) == len(SL.dihedrals) == len(SL.weights) == len(SL.coords)
    assert len(SL) == len(SL.dihedrals) == len(SL.weights) == len(SL.coords)


def test_minimize():

    SL1 = chilife.SpinLabel("R1M", 20, protein=U, minimize=True)
    SL2 = chilife.SpinLabel("R1M", 238, protein=U, minimize=True)

    r = np.linspace(15, 80, 256)
    P = chilife.distance_distribution(SL1, SL2, r)
    assert np.max(P) - 0.20643375571027256 <= 1e-2
    assert np.argmax(P) == 65


def test_spin_center_array_dim():
    SL1 = chilife.SpinLabel('R1M', 5, ubq)
    assert SL1.spin_centers.shape == (1, 3)

def test_no_altoloc():
    GR1G = mda.Universe('test_data/GR1G.gro')
    assert not hasattr(GR1G._topology, "altLocs")
    SL = chilife.SpinLabel('R1M', 2, GR1G)
    assert hasattr(GR1G._topology, "altLocs")


