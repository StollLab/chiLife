import os, hashlib, pickle
import numpy as np
import pytest
import chilife
from chilife.MolSys import MolSys, parse_paren, Residue, ResidueSelection
import MDAnalysis as mda

prot = MolSys.from_pdb('test_data/1omp_H.pdb')
mda_prot = mda.Universe('test_data/1omp_H.pdb')


def test_from_pdb():
    ans = np.array([[-1.924, -20.646, -23.898],
                    [-1.825, -19.383, -24.623],
                    [-1.514, -19.5, -26.1],
                    [-2.237, -20.211, -26.799],
                    [-0.696, -18.586, -23.89]])

    np.testing.assert_almost_equal(prot.coords[21:26], ans)


def test_from_multistate_pdb():
    prot = MolSys.from_pdb('test_data/DHC.pdb')


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

    np.testing.assert_allclose(m1.mask, np.argwhere(ans_mask).T[0])
    assert np.all(np.isin(m1.names, ['CA', 'CB']))
    mm = np.ones(prot.n_atoms, dtype=bool)
    mm[m1.mask] = False
    assert not np.any(np.isin(prot.names[mm], ['CA', 'CB']))


def test_select_and_or_and():
    m1 = prot.select_atoms('(resnum 32 and name CA) or (resnum 33 and name CB)')
    ans_mask = (prot.resnums == 32) * (prot.names == 'CA') + (prot.resnums == 33) * (prot.names == 'CB')

    np.testing.assert_almost_equal(m1.mask, np.argwhere(ans_mask).T[0])
    assert np.all(np.isin(m1.names, ['CA', 'CB']))
    assert np.all(np.isin(m1.resnums, [32, 33]))


def test_select_and_not():
    m1 = prot.select_atoms('resnum 32 and not name CA')
    ans_mask = (prot.resnums == 32) * (prot.names != 'CA')

    np.testing.assert_almost_equal(m1.mask, np.argwhere(ans_mask).T[0])
    assert not np.any(np.isin(m1.names, ['CA']))
    assert np.all(np.isin(m1.resnums, [32]))


def test_select_name_and_resname():
    m1 = prot.select_atoms("name CB and resname PRO")
    ans_mask = (prot.names == 'CB') * (prot.resnames == 'PRO')

    np.testing.assert_almost_equal(m1.mask, np.argwhere(ans_mask).T[0])
    assert np.all(np.isin(m1.names, ['CB']))
    assert np.all(np.isin(m1.resnames, ['PRO']))


def test_select_complex():
    m1 = prot.select_atoms('resname LYS ARG PRO and (name CA or (type C and resname PRO)) or resnum 5')

    resnames = np.isin(prot.resnames, ['LYS', 'ARG', 'PRO'])
    ca_or_c_pro = ((prot.names == 'CA') + ((prot.atypes == 'C') * (prot.resnames == 'PRO')))
    resnum5 = (prot.resnums == 5)
    ans_mask = resnames * ca_or_c_pro + resnum5

    np.testing.assert_almost_equal(m1.mask, np.argwhere(ans_mask).T[0])
    assert not np.any(np.isin(prot.resnums[~m1.mask], 5))


def test_select_range():
    m1 = prot.select_atoms('resnum 5-15')
    m2 = prot.select_atoms('resnum 5:15')
    ans_mask = np.isin(prot.resnums, list(range(5, 16)))
    np.testing.assert_almost_equal(m1.mask, np.argwhere(ans_mask).T[0])
    np.testing.assert_almost_equal(m2.mask, np.argwhere(ans_mask).T[0])


features = ("atomids", "names", "altlocs", "resnames", "chains", "resnums", "occupancies", "bs", "atypes", "charges")


@pytest.mark.parametrize('feature', features)
def test_AtomSelection_features(feature):
    m1 = prot.select_atoms('resname LYS ARG PRO and (name CA or (type C and resname PRO)) or resnum 5')
    A = prot.__getattr__(feature)[m1.mask]
    B = m1.__getattr__(feature)

    assert np.all(A == B)


def test_byres():
    waters = prot.select_atoms('byres name OH2 or resname HOH')
    assert np.all(np.unique(waters.resixs) == np.arange(371, 444, dtype=int))


def test_unary_not():
    asel = prot.select_atoms('not resid 5')
    ans = prot.resnums != 5
    assert np.all(asel.mask == np.argwhere(ans).T[0])


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
    omp = chilife.MolSys.from_pdb('test_data/1omp.pdb')
    ompmda = mda.Universe('test_data/1omp.pdb')

    SL1 = chilife.SpinLabel('R1M', 20, omp)
    SL2 = chilife.SpinLabel('R1M', 20, ompmda)

    assert SL1 == SL2


def test_trajectory():
    p = chilife.MolSys.from_pdb('test_data/2klf.pdb')
    ans = np.array([[41.068, 2.309, 0.296],
                    [39.431, 5.673, 2.669],
                    [39.041, 6.607, 2.738],
                    [39.385, 5.41, 3.503],
                    [35.887, 7.796, 3.6],
                    [34.155, 6.464, 8.545],
                    [39.874, 3.395, 2.951],
                    [35.945, 5.486, 1.808],
                    [39.028, 4.272, 1.541],
                    [40.395, 3.852, 9.391]])
    test = []
    for _ in p.trajectory:
        test.append(p.coords[0])
    test = np.array(test)

    np.testing.assert_allclose(test, ans)


def test_trajectory_set():
    p = chilife.MolSys.from_pdb('test_data/2klf.pdb')
    p.trajectory[5]
    np.testing.assert_allclose([34.155, 6.464, 8.545], p.coords[0])


def test_selection_traj():
    p = chilife.MolSys.from_pdb('test_data/2klf.pdb')
    s = p.select_atoms('resi 10 and name CB')
    np.testing.assert_allclose(s.coords, [[16.569, -1.406, 11.158]])
    p.trajectory[5]
    np.testing.assert_allclose(s.coords, [[16.195, -1.233, 11.306]])


def test_ResidueSelection():
    p = chilife.MolSys.from_pdb('test_data/1omp.pdb')
    r = p.residues[10:12]
    assert isinstance(r, ResidueSelection)
    assert len(r) == 2
    assert np.all(r.resnums == [11, 12])

    r2 = p.residues[10]
    assert isinstance(r2, Residue)
    assert r2.resname == 'ILE'
    assert r2.resnum == 11


def test_save_Protein():
    p = chilife.MolSys.from_pdb('test_data/1omp.pdb', sort_atoms=True)
    p._fname = "my_protein"
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
    assert len(x) == 769
    assert np.all(x.resnames == 'LYS')


def test_save_trajectory():
    traj = chilife.MolSys.from_pdb('test_data/xlsavetraj.pdb')
    traj._fname = 'xlsavetraj'
    chilife.save('xlsavetraj.pdb', traj)

    with open('test_data/xlsavetraj.pdb', 'r') as f:
        ans = hashlib.md5(f.read().encode("utf-8")).hexdigest()

    with open("xlsavetraj.pdb", "r") as f:
        test = hashlib.md5(f.read().encode("utf-8")).hexdigest()

    os.remove('xlsavetraj.pdb')
    assert test == ans


def test_trajectory_iter():
    traj = chilife.MolSys.from_pdb('test_data/test_traj_iter.pdb')
    assert not np.all(traj.trajectory.coords[0] == traj.trajectory.coords[1])

    prev_coords = np.zeros_like(traj.coords)
    for ts in traj.trajectory:
        this_coords = traj.coords
        assert not np.all(this_coords == prev_coords)
        prev_coords = this_coords.copy()


def test_xl_protein_repack():

    np.random.seed(2)
    protein = chilife.MolSys.from_pdb("test_data/1ubq.pdb").select_atoms("protein")
    SL = chilife.SpinLabel("R1M", site=28, protein=protein)

    traj1, deltaE1 = chilife.repack(protein, SL, repetitions=10, repack_radius=10)
    traj2, deltaE2 = chilife.repack(protein, SL, repetitions=10, off_rotamer=True, repack_radius=10)

    t1coords = traj1.universe.trajectory.coordinate_array
    t2coords = traj2.universe.trajectory.coordinate_array

    with np.load("test_data/repack_ans.npz") as f:
        t1ans = f["traj1"]
        t2ans = f["traj2"]

    np.testing.assert_almost_equal(t1coords, t1ans, decimal=4)
    np.testing.assert_almost_equal(t2coords, t2ans, decimal=4)


def test_same_as_mda():
    RE1 = chilife.RotamerEnsemble('ILE', 116, mda_prot)
    RE2 = chilife.RotamerEnsemble('ILE', 116, prot)

    np.testing.assert_almost_equal(RE1.coords, RE2.coords, decimal=5)
    np.testing.assert_almost_equal(RE1.weights, RE2.weights, decimal=5)

    SL1 = chilife.SpinLabel('R1M', 238, mda_prot)
    SL2 = chilife.SpinLabel('R1M', 238, prot)

    np.testing.assert_almost_equal(SL1.coords, SL2.coords, decimal=5)
    np.testing.assert_almost_equal(SL1.weights, SL2.weights, decimal=5)


def test_xl_protein_mutate():
    p = chilife.MolSys.from_pdb('test_data/1omp.pdb', sort_atoms=True)
    SL = chilife.SpinLabel('R1M', 238, p)
    pSL = chilife.mutate(p, SL)
    pSL._fname = 'test_mutate'
    chilife.save('mutate_xlprotein.pdb', pSL)

    with open('test_data/mutate_xlprotein.pdb', 'r') as f:
        ans = hashlib.md5(f.read().encode("utf-8")).hexdigest()

    with open("mutate_xlprotein.pdb", "r") as f:
        test = hashlib.md5(f.read().encode("utf-8")).hexdigest()

    os.remove('mutate_xlprotein.pdb')
    assert ans == test


def test_re_form_xl_traj():
    traj = MolSys.from_pdb('test_data/xlsavetraj.pdb')
    SL = chilife.RotamerEnsemble.from_trajectory(traj, 236, burn_in=0)
    ans_dihedrals = np.array([[-175.57682647, -23.12323366],
                              [-175.57682647, 5.02763709],
                              [-175.57682647, 63.16005353],
                              [-175.57682647, 63.65448686],
                              [-175.57682647, 89.75074833]])

    assert len(SL) == 5
    np.testing.assert_almost_equal(SL.dihedrals, ans_dihedrals)


def test_backbone_selection():
    PhiSel = prot.residues[40].phi_selection()
    PsiSel = prot.residues[40].psi_selection()
    phi = np.rad2deg(chilife.get_dihedral(PhiSel.positions))
    psi = np.rad2deg(chilife.get_dihedral(PsiSel.positions))

    assert phi == -67.45821389595835
    assert psi == 135.37370203341246


# def test_xl_protein_from_pose():
#     assert False

def test_atom_sel_getitem():
    res = prot.residues[10].atoms[0:4]
    assert np.all(res.resnums == 11)
    np.testing.assert_equal(res.names, ['N', 'CA', 'C', 'O'])


def test_residue_sel_getitem():
    res = prot.residues[10:20]
    res10 = res.residues[0]

    assert res10.resid == 11
    assert np.all(res10.atoms.ix == np.arange(147, 166))
    assert np.all(res.residues[3:5].resnums == [14, 15])


def test_copy():
    prot2 = prot.copy()
    np.testing.assert_equal(prot2.coords, prot.coords)
    assert prot2.coords is not prot

def test_setattr1():
    prot2 = prot.copy()
    prot2.names = 'test'
    assert np.all(prot2.names == 'test')


def test_setattr2():
    prot2 = prot.copy()

    asel = prot2.select_atoms('resname ARG')
    mask = np.argwhere(prot2.resnames == 'ARG').flatten()

    asel.resnames = 'TES'
    assert np.all(asel.resnames == 'TES')
    assert np.all(prot2.resnames[mask] == 'TES')


def test_pickle():
    mol = MolSys.from_pdb('test_data/PPII_Capped.pdb')
    with open('tmp.pkl', 'wb') as f:
        pickle.dump(mol, f)

    del mol
    with open('tmp.pkl', 'rb') as f:
        mol = pickle.load(f)

    ans = MolSys.from_pdb('test_data/PPII_Capped.pdb')

    assert mol is not ans

    np.testing.assert_equal(mol.names, ans.names)
    np.testing.assert_equal(mol.resnames, ans.resnames)
    np.testing.assert_equal(mol.coords, ans.coords)


def test_pickle_selection():
    mol = MolSys.from_pdb('test_data/PPII_Capped.pdb').atoms
    with open('tmp.pkl', 'wb') as f:
        pickle.dump(mol, f)

    del mol
    with open('tmp.pkl', 'rb') as f:
        mol = pickle.load(f)

    ans = MolSys.from_pdb('test_data/PPII_Capped.pdb').atoms

    assert mol is not ans

    np.testing.assert_equal(mol.names, ans.names)
    np.testing.assert_equal(mol.resnames, ans.resnames)
    np.testing.assert_equal(mol.coords, ans.coords)


def test_from_atomsel():
    # MDAnalysis test
    atomsel = mda_prot.select_atoms('resnum 30-150')
    new_prot = MolSys.from_atomsel(atomsel)

    np.testing.assert_equal(new_prot.positions, atomsel.positions)
    np.testing.assert_equal(new_prot.ix + 435, atomsel.ix)
    assert len(new_prot.trajectory) == len(atomsel.universe.trajectory)

    # chiLife MolSys test
    atomsel = prot.select_atoms('resnum 30-150')
    new_prot = MolSys.from_atomsel(atomsel)

    np.testing.assert_equal(new_prot.positions, atomsel.positions)
    np.testing.assert_equal(new_prot.ix + 435, atomsel.ix)
    assert len(new_prot.trajectory) == len(atomsel.universe.trajectory)


def test_ResidueSelection_iter():
    prot = MolSys.from_pdb('test_data/1a2w.pdb')
    chain_B = prot.select_atoms('chain B')
    for res in chain_B.residues:
        assert res.chain == 'B'
        assert np.all(res.atoms.chains == 'B')
        

def test_ResidueSelection_iter2():
    protein = chilife.MolSys.from_pdb("test_data/1ubq.pdb").select_atoms("protein")
    ids = [res.resid for res in protein.residues]
    np.testing.assert_equal(ids, np.arange(1,77))



