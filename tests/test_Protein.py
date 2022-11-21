import os, hashlib
import numpy as np
import pytest
import chilife
from chilife.Protein import Protein, parse_paren, Residue, ResidueSelection, Segment, SegmentSelection, AtomSelection
import MDAnalysis as mda

prot = Protein.from_pdb('test_data/1omp_H.pdb')
mda_prot = mda.Universe('test_data/1omp_H.pdb')

def test_from_pdb():

    ans = np.array([[ -1.924, -20.646, -23.898],
                    [ -1.825, -19.383, -24.623],
                    [ -1.514, -19.5  , -26.1  ],
                    [ -2.237, -20.211, -26.799],
                    [ -0.696, -18.586, -23.89 ]])

    np.testing.assert_almost_equal(prot.coords[21:26], ans)


def test_from_multistate_pdb():
    prot = Protein.from_pdb('test_data/DHC.pdb')


def test_parse_paren1():
    P = parse_paren('(name CA or name CB)')
    assert P == ['name CA or name CB']


def test_parse_paren2():
    P = parse_paren('(resnum 32 and name CA) or (resnum 33 and name CB)')
    assert len(P) == 3


def test_parse_paren3():
    P = parse_paren('resnum 32 and name CA or (resnum 33 and name CB)')
    assert len(P) == 2


def test_parse_paren4():
    P = parse_paren('resname LYS ARG PRO and (name CA or (name CB and resname PRO)) or resnum 5')
    assert len(P) == 3


def test_parse_paren5():
    P = parse_paren('(name CA or (name CB and resname PRO) or name CD)')
    assert len(P) == 3


def test_select_or():
    m1 = prot.select_atoms('name CA or name CB')
    ans_mask = (prot.names == 'CA') + (prot.names == 'CB')

    np.testing.assert_almost_equal(m1.mask, ans_mask)
    assert np.all(np.isin(m1.names, ['CA', 'CB']))
    assert not np.any(np.isin(prot.names[~m1.mask], ['CA', 'CB']))


def test_select_and_or_and():

    m1 = prot.select_atoms('(resnum 32 and name CA) or (resnum 33 and name CB)')
    ans_mask = (prot.resnums == 32) * (prot.names == 'CA') + (prot.resnums == 33) * (prot.names == 'CB')

    np.testing.assert_almost_equal(m1.mask, ans_mask)
    assert np.all(np.isin(m1.names, ['CA', 'CB']))
    assert np.all(np.isin(m1.resnums, [32, 33]))


def test_select_and_not():
    m1 = prot.select_atoms('resnum 32 and not name CA')
    ans_mask = (prot.resnums == 32) * (prot.names != 'CA')

    np.testing.assert_almost_equal(m1.mask, ans_mask)
    assert not np.any(np.isin(m1.names, ['CA']))
    assert np.all(np.isin(m1.resnums, [32]))


def test_select_name_and_resname():
    m1 = prot.select_atoms("name CB and resname PRO")
    ans_mask = (prot.names == 'CB') * (prot.resnames == 'PRO')

    np.testing.assert_almost_equal(m1.mask, ans_mask)
    assert np.all(np.isin(m1.names, ['CB']))
    assert np.all(np.isin(m1.resnames, ['PRO']))


def test_select_complex():
    m1 = prot.select_atoms('resname LYS ARG PRO and (name CA or (type C and resname PRO)) or resnum 5')

    resnames = np.isin(prot.resnames, ['LYS', 'ARG', 'PRO'])
    ca_or_c_pro = ((prot.names == 'CA') + ((prot.atypes == 'C') * (prot.resnames == 'PRO' )))
    resnum5 = (prot.resnums == 5)
    ans_mask = resnames * ca_or_c_pro + resnum5

    np.testing.assert_almost_equal(m1.mask, ans_mask)
    assert not np.any(np.isin(prot.resnums[~m1.mask], 5))


def test_select_range():
    m1 = prot.select_atoms('resnum 5-15')
    m2 = prot.select_atoms('resnum 5:15')
    ans_mask = np.isin(prot.resnums, list(range(5, 16)))
    np.testing.assert_almost_equal(m1.mask, ans_mask)
    np.testing.assert_almost_equal(m2.mask, ans_mask)


features = ("atomids", "names", "altlocs", "resnames", "chains", "resnums", "occupancies", "bs", "atypes", "charges")
@pytest.mark.parametrize('feature', features)
def test_AtomSelection_features(feature):
    m1 = prot.select_atoms('resname LYS ARG PRO and (name CA or (type C and resname PRO)) or resnum 5')
    A = prot.__getattr__(feature)[m1.mask]
    B = m1.__getattr__(feature)

    assert np.all(A == B)


def test_byres():
    waters = prot.select_atoms('byres name OH2 or resname HOH')
    assert np.all(np.unique(waters.resixs) == np.arange(370, 443, dtype=int))


def test_unary_not():
    asel = prot.select_atoms('not resid 5')
    ans = prot.resnums != 5
    assert np.all(asel.mask == ans)

def test_around():
    asel = prot.select_atoms('around 5 resnum 5')
    mdasel = mda_prot.select_atoms('around 5 resnum 5')

    assert np.all(np.sort(asel.atoms.names) == np.sort(mdasel.atoms.names))
    assert np.all(asel.atoms.residues.resnums == mdasel.atoms.residues.resnums)

def test_subsel():
    asel = prot.select_atoms('around 5 resnum 5')
    bsel = prot.select_atoms('resname LEU around 5 resnum 5')
    a2sel = asel.select_atoms('resname LEU')
    np.testing.assert_almost_equal(bsel.coords, a2sel.coords)

def test_SL_Protein():
    omp = chilife.Protein.from_pdb('test_data/1omp.pdb')
    ompmda = mda.Universe('test_data/1omp.pdb')

    SL1 = chilife.SpinLabel('R1M', 20, omp)
    SL2 = chilife.SpinLabel('R1M', 20, ompmda)

    assert SL1 == SL2


def test_trajectory():
    p = chilife.Protein.from_pdb('test_data/2klf.pdb')
    ans = np.array([[41.068,  2.309,  0.296],
                    [39.431,  5.673,  2.669],
                    [39.041,  6.607,  2.738],
                    [39.385,  5.41 ,  3.503],
                    [35.887,  7.796,  3.6  ],
                    [34.155,  6.464,  8.545],
                    [39.874,  3.395,  2.951],
                    [35.945,  5.486,  1.808],
                    [39.028,  4.272,  1.541],
                    [40.395,  3.852,  9.391]])
    test = []
    for _ in p.trajectory:
        test.append(p.coords[0])
    test = np.array(test)

    np.testing.assert_allclose(test, ans)


def test_trajectory_set():
    p = chilife.Protein.from_pdb('test_data/2klf.pdb')
    p.trajectory[5]
    np.testing.assert_allclose([34.155,  6.464,  8.545], p.coords[0])


def test_selection_traj():
    p = chilife.Protein.from_pdb('test_data/2klf.pdb')
    s = p.select_atoms('resi 10 and name CB')
    np.testing.assert_allclose(s.coords, [16.569, -1.406, 11.158])
    p.trajectory[5]
    np.testing.assert_allclose(s.coords, [16.195, -1.233, 11.306])


def test_ResidueSelection():
    p = chilife.Protein.from_pdb('test_data/1omp.pdb')
    r = p.residues[10:12]
    assert isinstance(r, ResidueSelection)
    assert len(r) == 2
    assert np.all(r.resnums == [11, 12])

    r2 = p.residues[10]
    assert isinstance(r2, Residue)
    assert r2.resname == 'ILE'
    assert r2.resnum == 11


def test_save_Protein():
    p = chilife.Protein.from_pdb('test_data/1omp.pdb')
    chilife.save('my_protein.pdb', p)

    with open('test_data/test_save_xlprotein.pdb', 'r') as f:
        ans = hashlib.md5(f.read().encode("utf-8")).hexdigest()

    with open("my_protein.pdb", "r") as f:
        test = hashlib.md5(f.read().encode("utf-8")).hexdigest()

    os.remove('my_protein.pdb')
    assert test == ans


def test_bool_index_atomsel():
    bindex = prot.resnames == 'LYS'
    x = prot.atoms[bindex]
    assert len(x) == 768
    assert np.all(x.resname == 'LYS')


def test_save_trajectory():
    traj = chilife.Protein.from_pdb('mda_traj.pdb')
    chilife.save('xlsavetraj.pdb', traj)

    with open('test_data/xlsavetraj.pdb', 'r') as f:
        ans = hashlib.md5(f.read().encode("utf-8")).hexdigest()

    with open("xlsavetraj.pdb", "r") as f:
        test = hashlib.md5(f.read().encode("utf-8")).hexdigest()

    os.remove('xlsavetraj.pdb')
    assert test == ans



def test_trajectory_iter():

    traj = chilife.Protein.from_pdb('mda_traj.pdb')
    assert not np.all(traj.trajectory.coords[0] == traj.trajectory.coords[1])

    prev_coords = np.zeros_like(traj.coords)
    for ts in traj.trajectory:
        this_coords = traj.coords
        assert not np.all(this_coords == prev_coords)
        prev_coords = this_coords.copy()


def test_xl_protein_repack():

    p = chilife.Protein.from_pdb('test_data/1omp.pdb')
    pmda = mda.Universe('test_data/1omp.pdb')
    SL = chilife.SpinLabel('R1M', 238, p)
    np.random.seed(3000)
    trajmda, dEmda = chilife.repack(pmda, SL, repetitions=100, temp=300, off_rotamer=True)
    np.random.seed(3000)
    traj, dE = chilife.repack(p, SL, repetitions=100, temp=300, off_rotamer=True)

    print('SHUOLD !')

    # chilife.save('mda_traj.pdb', trajmda, mode='w')
    chilife.save('chilife_traj.pdb', traj, mode='w')
    chilife.save('mda_traj.pdb', traj, mode='w')
    ans = np.array([ 4.29882146e-03,  5.38218655e-01, -7.73797120e-06,  6.96190997e-02,
                     3.16121282e-02,  5.45366181e-01, -7.34109371e-01, -6.96171902e-02,
                    -8.19977337e-04, -5.58844086e-03])

    np.testing.assert_almost_equal(dE, ans, decimal=5)


def test_same_as_mda():
    RE1 = chilife.RotamerEnsemble('ILE', 116, mda_prot)
    RE2 = chilife.RotamerEnsemble('ILE', 116, prot)

    np.testing.assert_almost_equal(RE1.coords, RE2.coords)
    np.testing.assert_almost_equal(RE1.atom_energies, RE2.atom_energies)
    np.testing.assert_almost_equal(RE1.weights, RE2.weights)

def test_xl_protein_mutate():
    p = chilife.Protein.from_pdb('test_data/1omp.pdb')
    SL = chilife.SpinLabel('R1M', 238, p)
    pSL = chilife.mutate(p, SL)

    chilife.save('test_mutate.pdb', pSL)

    with open('test_data/mutate_xlprotein.pdb', 'r') as f:
        ans = hashlib.md5(f.read().encode("utf-8")).hexdigest()

    with open("test_mutate.pdb", "r") as f:
        test = hashlib.md5(f.read().encode("utf-8")).hexdigest()

    os.remove('test_mutate.pdb')
    assert ans == test


def test_xl_protein_from_mda():
    assert False


def test_xl_protein_from_pose():
    assert False


def test_sl_form_xl_traj():
    assert False

def test_backbone_to_site():
    # Noticed that when using chilife.Protein, the backbone O was being placed incorrectly because the atom selection
    # coordinates were being returned as 1D not 2D. Added logic to fix it but need to add more rigerous tests
    assert False