import os, hashlib
import pickle
import random
from functools import partial
import numpy as np
import pytest
import MDAnalysis as mda

import chilife

pdbids = ["1ubq", "1a2w", '1az5']
ubq = mda.Universe("test_data/1ubq.pdb", in_memory=True)
resis = [
    (key, -90, 160)
    for key in chilife.dihedral_defs
    if key
    not in (("R1B", "R1C", "CYR1", "MTN", "ALA", "GLY", 'HID', 'HIE', 'ASH', 'GLH', 'CYM', 'LYN', 'HIP') +
            tuple(chilife.USER_LIBRARIES) +
            tuple(chilife.USER_dLIBRARIES))
]


@pytest.mark.parametrize("res", resis)
def test_read_dunbrack(res):
    res, phi, psi = res

    dlib = chilife.read_bbdep(res, -70, 90)


    with np.load(f"test_data/{res}_{phi}_{psi}.npz", allow_pickle=True) as f:
        dlib_ans = {key: f[key] for key in f if key != "allow_pickle"}

    for key in dlib_ans:
        if dlib_ans[key].dtype not in [np.dtype(f"<U{i}") for i in range(1, 5)]:
            np.testing.assert_almost_equal(dlib_ans[key], dlib[key], decimal=5)
        else:
            assert np.all(np.char.equal(dlib_ans[key], dlib[key]))


def test_sort_pdb():
    pdbfile = "test_data/trt.pdb"
    x = chilife.sort_pdb(pdbfile)

    linest = [line for line in x if line[76:79].strip() != 'H']
    test = hashlib.md5("".join(linest).encode("utf-8")).hexdigest()

    with open("test_data/trt_sorted.pdb", "r") as f:
        lines = f.readlines()
        linesa = [line for line in lines if line[76:79].strip() != 'H']
        ans = hashlib.md5("".join(linesa).encode("utf-8")).hexdigest()

    assert test == ans


def test_sort_pdb2():
    x = chilife.sort_pdb("test_data/SL_GGAGG.pdb")

    with open("test_data/SL_GGAGG_tmp.pdb", "w") as f:
        for line in x:
            f.write(line)

    U = mda.Universe("test_data/SL_GGAGG_tmp.pdb", in_memory=True)
    os.remove("test_data/SL_GGAGG_tmp.pdb")

    # check if atom order compatible with internal coordinates system.
    ICs = chilife.MolSysIC.from_atoms(U, preferred_dihedrals=[["C", "N", "CA", "C"]])


def test_sort_H():
    x = chilife.sort_pdb('../src/chilife/data/rotamer_libraries/residue_pdbs/lys.pdb')
    H_names = [xx[12:16] for xx in x[9:]]
    np.testing.assert_equal(H_names, [' H  ', ' HA ', '3HB ', '2HB ',
                                      '3HG ', '2HG ', '3HD ', '2HD ',
                                      '2HE ', '3HE ', '3HZ ', '2HZ ',
                                      '1HZ '])


def test_sort_residue():
    lines = chilife.sort_pdb('test_data/TUM.pdb', aln_atoms=['N3', 'C4', 'C5'])
    with open('test_data/TUM_srtd.pdb', 'r') as f:
        alines = f.readlines()

    test = hashlib.md5("".join(lines[0]).encode("utf-8")).hexdigest()
    ans = hashlib.md5("".join(alines).encode("utf-8")).hexdigest()
    assert test == ans


def test_sort_many_models():
    x = chilife.sort_pdb("test_data/msort.pdb")
    with open("test_data/msort_tmp.pdb", "w") as f:
        for i, struct in enumerate(x):
            f.write(f"MODEL {i}\n")
            f.writelines(struct)
            f.write("ENDMDL\n")

    with open("test_data/msort_tmp.pdb", "r") as f:
        heavy_test = [line for line in f.readlines() if line[76:79].strip() != 'H']

        test = hashlib.md5("".join(heavy_test).encode("utf-8")).hexdigest()

    with open("test_data/msort_ans.pdb", "r") as f:
        heavy_ans = [line for line in f.readlines() if line[76:79].strip() != 'H']

        ans = hashlib.md5("".join(heavy_ans).encode("utf-8")).hexdigest()

    os.remove('test_data/msort_tmp.pdb')

    assert test == ans


def test_mutate():
    protein = mda.Universe("test_data/1ubq.pdb", in_memory=True).select_atoms("protein")
    SL = chilife.SpinLabel("R1C", site=28, protein=protein)

    labeled_protein = chilife.mutate(protein, SL)
    ub1_A28R1 = mda.Universe("test_data/1ubq_A28R1.pdb", in_memory=True)

    np.testing.assert_almost_equal(ub1_A28R1.atoms.positions, labeled_protein.atoms.positions, decimal=3)


def test_mutate2():
    protein = mda.Universe('test_data/1ubq.pdb', in_memory=True).select_atoms("protein")
    SL1 = chilife.SpinLabel("R1C", site=28, protein=protein)
    SL2 = chilife.SpinLabel("R1C", site=48, protein=protein,)

    labeled_protein = chilife.mutate(protein, SL1, SL2)
    ub1_A28R1_K48R1 = mda.Universe("test_data/ub1_A28R1_K48R1.pdb")
    np.testing.assert_almost_equal(ub1_A28R1_K48R1.atoms.positions, labeled_protein.atoms.positions, decimal=3)


def test_mutate3():
    protein = mda.Universe("test_data/1omp_H.pdb").select_atoms("protein")
    SL1 = chilife.SpinLabel("R1C", site=238, protein=protein, use_H=True)
    SL2 = chilife.SpinLabel("R1C", site=311, protein=protein, use_H=True)

    labeled_protein = chilife.mutate(protein, SL1, SL2)
    assert len(labeled_protein.atoms) != len(protein.atoms)


def test_mutate4():
    protein = mda.Universe("test_data/1omp_H.pdb").select_atoms("protein")
    D41G = chilife.RotamerEnsemble("GLY", 41, protein=protein, eval_clash=False)
    S238A = chilife.RotamerEnsemble("ALA", 238, protein=protein, eval_clash=False)
    mPro = chilife.mutate(protein, D41G, S238A, add_missing_atoms=False)
    D41G_pos = mPro.select_atoms("resnum 41").positions
    S238A_pos = mPro.select_atoms("resnum 238").positions
    np.testing.assert_almost_equal(D41G._coords[0], D41G_pos, decimal=6)
    np.testing.assert_almost_equal(S238A._coords[0], S238A_pos, decimal=6)


def test_mutate5():
    PPII = mda.Universe('test_data/PolyProII.pdb')
    TOC = chilife.SpinLabel('TOC', 8, PPII)

    PPIIm = chilife.mutate(PPII, TOC)
    assert PPIIm.residues[-1].resname == 'NHH'

def test_mutate6():
    SL = chilife.SpinLabel("R1C", site=28, protein=ubq)
    labeled_protein = chilife.mutate(ubq, SL, rotamer_index=10)

    # Make sure the newly mutated protein has the coordinates of the rotamer selected
    np.testing.assert_allclose(labeled_protein.select_atoms(SL.selstr).positions, SL.coords[10])

    # Make sure that is not the same as the first rotamer.
    assert np.max(labeled_protein.select_atoms(SL.selstr).positions - SL.coords[0]) > 1

def test_mutate7():
    SL = chilife.SpinLabel("R1C", site=28, protein=ubq)
    labeled_protein = chilife.mutate(ubq, SL, rotamer_index='all')

    assert len(labeled_protein.trajectory) == len(SL)

    label_atoms =labeled_protein.select_atoms(SL.selstr)
    for i, ts in enumerate(labeled_protein.trajectory):
        np.testing.assert_allclose(label_atoms.positions, SL.coords[i])

def test_mutate8():
    SL1 =  chilife.SpinLabel("R1C", site=28, protein=ubq)
    SL2 = chilife.RotamerEnsemble("LYS", site=48, protein=ubq)
    labeled_protein = chilife.mutate(ubq, SL1, SL2, rotamer_index='all')

    assert len(labeled_protein.trajectory) == max(len(SL1), len(SL2))

    for i, ts in enumerate(labeled_protein.trajectory):
        for SL in  (SL1, SL2):
            label_atoms = labeled_protein.select_atoms(SL.selstr)
            if len(SL) <= i:
                np.testing.assert_allclose(label_atoms.positions, SL.coords[-1])
            else:
                np.testing.assert_allclose(label_atoms.positions, SL.coords[i])


def test_mutate9():

    prot = chilife.load_protein("test_data/ubq_ccap.pdb")
    K = chilife.RotamerEnsemble("LYS", site=76, protein=prot, use_H=True)
    mut = chilife.mutate(prot, K)


def test_add_missing_atoms():
    protein = mda.Universe("test_data/1omp.pdb", in_memory=True).select_atoms("protein")
    new_prot = chilife.mutate(protein)
    assert len(new_prot.atoms) != len(protein.atoms)
    assert len(new_prot.atoms) == 2877


@pytest.mark.parametrize(
    "res", set(chilife.dihedral_defs.keys()) -
           {"CYR1", "MTN", "R1M", "R1C", 'HID', 'HIE', 'ASH', 'GLH', 'CYM', 'LYN', 'HIP'} -
           chilife.USER_dLIBRARIES -
           chilife.USER_LIBRARIES
)
def test_sort_and_internal_coords(res):
    pdbfile = chilife.RL_DIR / f"residue_pdbs/{res.lower()}.pdb"
    lines = chilife.sort_pdb(str(pdbfile))
    anames = [line[13:16] for line in lines if "H" not in line[12:16]]

    with open(pdbfile, "r") as f:
        ans = [
            line[13:16]
            for line in f.readlines()
            if line.startswith("ATOM")
            if "H" not in line[12:16]
        ]

    assert anames == ans


def test_get_min_topol():
    with open('test_data/DHC.pdb', 'r') as f:
        lines = f.readlines()

    all_lines = []
    for line in lines:
        if line.startswith('MODEL'):
            newlines = []
        elif line.startswith(('ATOM', 'HETATOM')):
            newlines.append(line)
        elif line.startswith('ENDMDL'):
            all_lines.append(newlines)

    min_bonds = chilife.get_min_topol(all_lines)
    ans = {(0, 1), (0, 2), (0, 6), (2, 3), (2, 4), (2, 5), (6, 7), (6, 16), (7, 8), (7, 10), (7, 17), (8, 9), (8, 23),
           (10, 11), (10, 18), (10, 19), (11, 12), (11, 13), (12, 15), (12, 20), (13, 14), (13, 21), (14, 15), (14, 22),
           (15, 77), (23, 24), (23, 27), (24, 25), (24, 28), (24, 29), (25, 26), (25, 30), (30, 31), (30, 34), (31, 32),
           (31, 35), (31, 36), (32, 33), (32, 37), (37, 38), (37, 41), (38, 39), (38, 42), (38, 43), (39, 40), (39, 44),
           (44, 45), (44, 54), (45, 46), (45, 48), (45, 55), (46, 47), (46, 61), (48, 49), (48, 56), (48, 57), (49, 50),
           (49, 51), (50, 53), (50, 58), (51, 52), (51, 60), (52, 53), (52, 59), (53, 77), (61, 62), (61, 63), (64, 73),
           (64, 77), (65, 75), (66, 75), (66, 77), (67, 73), (68, 71), (68, 72), (68, 74), (68, 77), (69, 76), (70, 76),
           (71, 73), (71, 78), (71, 79), (72, 76), (72, 82), (72, 83), (74, 75), (74, 80), (74, 81)}

    assert min_bonds == ans


def test_sort_nonuni_topol():
    srtd = chilife.sort_pdb('test_data/test_nonuni_top.pdb', uniform_topology=False)

    with open('test_data/nonuni_topol_ans.pkl', 'rb') as f:
        ans = pickle.load(f)

    for test_model, ans_model in zip(srtd, ans):
        for test_line, ans_line in zip(test_model, ans_model):
            assert test_line == ans_line


def test_get_angle():
    sel = ubq.select_atoms('resnum 23 and name N CA C')
    ang = chilife.get_angle(sel.positions)
    assert ang - np.deg2rad(110.30299155404542) < 1e-6


def test_get_angles():
    Ns = ubq.select_atoms('name N').positions
    CAs = ubq.select_atoms('name CA').positions
    Cs = ubq.select_atoms('name C').positions

    angs = chilife.get_angles(Ns, CAs, Cs)
    ans = [chilife.get_angle([N, CA, C]) for N, CA, C in zip(Ns, CAs, Cs)]
    np.testing.assert_almost_equal(angs, ans)


def test_get_dihedral():
    sel = ubq.select_atoms('(resnum 23 and name N CA C) or (resnum 24 and name N)')
    dihe = chilife.get_dihedral(sel.positions)
    assert dihe - np.deg2rad(37.20999032936247) < 1e-6


def test_get_dihedrals():
    N1s = ubq.select_atoms('name N').positions[:-1]
    CAs = ubq.select_atoms('name CA').positions[:-1]
    Cs = ubq.select_atoms('name C').positions[:-1]
    N2s = ubq.select_atoms('name N').positions[1:]

    dihes = chilife.get_dihedrals(N1s, CAs, Cs, N2s)
    ans = [chilife.get_dihedral([N1, CA, C, N2]) for N1, CA, C, N2 in zip(N1s, CAs, Cs, N2s)]
    np.testing.assert_almost_equal(dihes, ans, decimal=6)


def test_guess_mobile_dihedrals():
    label = chilife.MolSys.from_pdb('test_data/TEP.pdb')

    ICs = chilife.MolSysIC.from_atoms(label)
    mbd = chilife.guess_mobile_dihedrals(ICs)

    ans = [['N', 'CA', 'CB', 'CB2'],
           ['CA', 'CB', 'CB2', 'NG'],
           ['ND', 'CE3', 'CZ3', 'C31'],
           ['CZ1', 'C11', 'C12', 'O12'],
           ['C11', 'C12', 'N12', 'C13'],
           ['C12', 'N12', 'C13', 'C14'],
           ['N12', 'C13', 'C14', 'C16'],
           ['C14', 'C15', 'C18', 'C19'],
           ['C14', 'C15', 'C20', 'C21'],
           ['C16', 'C17', 'C22', 'C23'],
           ['C16', 'C17', 'C24', 'C25']]

    np.testing.assert_equal(mbd, ans)


def test_get_site_volume():
    vols = [chilife.get_site_volume(site, ubq) for site in ubq.residues.resnums]
    ans = np.load('test_data/test_get_site_volume.npy')
    np.testing.assert_almost_equal(vols, ans)


def test_make_peptide():
    pep = chilife.make_peptide("ACDEF[R1M]GHIKL<COC1=CC=C(C=C1)CC(C(=O)O)N>MNPQR[R1C]STVWY")

    test = pep.positions.copy()
    ans = np.load('test_data/test_mk_pep.npy')

    ans -= np.mean(ans, axis=0)

    test -= np.mean(test, axis=0)
    mx = np.linalg.inv(test.T @ test) @ test.T @ ans
    test = test @ mx
    diff = test - ans
    rms = np.sqrt(np.sum(diff * diff) / len(test))

    assert rms < 3

def test_make_strand():
    phi = np.ones(22) * -140
    psi = np.ones(22) * 130
    omega = np.ones(22) * -178

    pep = chilife.make_peptide("ACDEF[R1M]GHIKL<COC1=CC=C(C=C1)CC(C(=O)O)N>MNPQR[R1C]STVWY", phi, psi, omega)
    pepIC = chilife.MolSysIC.from_atoms(pep)
    phi_idx = pepIC.phi_idxs(np.arange(2, 23))
    psi_idx = pepIC.psi_idxs(np.arange(1, 22))
    tphi = np.rad2deg(pepIC.z_matrix[phi_idx, 2])
    tpsi  = np.rad2deg(pepIC.z_matrix[psi_idx, 2])
    np.testing.assert_allclose(phi[1:], tphi)
    np.testing.assert_allclose(psi[:-1], tpsi)


def test_make_pep_w_cap():
    pep = chilife.make_peptide("[ACE]A[R1M]A[NME]")
    ans = np.load('test_data/ACE_AR1A_NME.npy')
    np.testing.assert_almost_equal(pep.positions, ans, decimal=4)

def test_make_pep_w_cap2():
    phi = np.array([-61, -63])
    psi = np.array([-40, -42])
    omega = np.array([-176, 176])
    pep = chilife.make_peptide("A[R1M]A", phi=phi, psi=psi, omega=omega)
    ans = np.load('test_data/mkpep2.npy')
    np.testing.assert_almost_equal(pep.positions, ans, decimal=4)


def test_make_pep_w_cap3():
    phi = np.array([-60, -62, -64])
    psi = np.array([-39, -41, -43])
    omega = np.array([-175, 180, 175])

    pep = chilife.make_peptide("[ACE]A[R1M]A[NME]", phi=phi, psi=psi, omega=omega)
    ans = np.load('test_data/test_mkpep3.npy')
    np.testing.assert_almost_equal(pep.positions, ans, decimal=4)





def test_mkpep_template():
    ATII = chilife.fetch('6JOD').select_atoms('segid B')
    pep = chilife.make_peptide("DRV[TOC]IHPF", template=ATII)

    ATII_IC = chilife.MolSysIC.from_atoms(ATII)
    phi_idx = ATII_IC.phi_idxs(np.arange(2, 23))
    psi_idx = ATII_IC.psi_idxs(np.arange(1, 22))
    phi = np.rad2deg(ATII_IC.z_matrix[phi_idx, 2])
    psi  = np.rad2deg(ATII_IC.z_matrix[psi_idx, 2])

    pep_IC = chilife.MolSysIC.from_atoms(pep)
    phi_idx = pep_IC.phi_idxs(np.arange(2, 23))
    psi_idx = pep_IC.psi_idxs(np.arange(1, 22))
    tphi = np.rad2deg(pep_IC.z_matrix[phi_idx, 2])
    tpsi  = np.rad2deg(pep_IC.z_matrix[psi_idx, 2])

    np.testing.assert_allclose(phi, tphi)
    np.testing.assert_allclose(psi, tpsi)


def test_parsed_sequence():
    pseq = chilife.parse_sequence("[ACE]AaNIE<cccn>[NHH]")
    for a, b in zip(pseq, ['ACE', 'ALA', 'ALA', 'ASN', 'ILE', 'GLU', 'cccn', 'NHH']):
        assert a == b


s2rkeys = ["COC1=CC=C(C=C1)CC(C(=O)O)N",
           "CC1CC=NC1C(=O)NCCCCC(C(=O)O)N",
           "CC(C)(C)OC(=O)N[C@@H](CCCCN)C(O)=O",
           "C(CC1=CC=C(C=C1)OCCCCC#C)(C(=O)O)N"]
s2rans = [f"test_data/s2rm{i+1}.npy" for i in range(len(s2rkeys))]
@pytest.mark.parametrize('key, ans', zip(s2rkeys, s2rans))
def test_smiles2residue(key, ans):
    m1 = chilife.smiles2residue(key, randomSeed=0)
    test_mask = np.argsort(m1.names)
    test = m1.positions[test_mask]

    answer = np.load(ans)
    answer -= np.mean(answer, axis=0)

    test -= np.mean(test, axis=0)
    mx = np.linalg.inv(test.T @ test) @ test.T @ answer
    test = test @ mx
    diff = test - answer
    rms = np.sqrt(np.sum(diff * diff) / len(test))

    assert rms < 3


def test_smiles2residue2():
    m1 = chilife.smiles2residue("C1C[C@H]2[C@@H]([C@@H]2C1)C(=O)C[C@H]([NH3+])C(=O)[O-]", randomSeed=0)
    test_mask = np.argsort(m1.names)
    test = m1.positions[test_mask]

    answer = np.load('test_data/s2rm5.npy')
    answer -= np.mean(answer, axis=0)

    test -= np.mean(test, axis=0)
    mx = np.linalg.inv(test.T @ test) @ test.T @ answer
    test = test @ mx
    diff = test - answer
    rms = np.sqrt(np.sum(diff * diff) / len(test))

    assert rms < 3

def test_smiles2residue3():
    m1 = chilife.smiles2residue("CCc1c([nH+]c(nc1NC[C@@H](C(=O)[O-])[NH3+])C)C")
    assert True


def test_append_cap():
    pep = chilife.MolSys.from_pdb('test_data/test_make_peptide.pdb')
    new_pep = chilife.append_cap(pep, 'ace')
    new_pep = chilife.append_cap(new_pep, 'nme')

    ace_coords = np.array([[-0.62455994,  0.61491044,  0.946],
                           [ 0.02253714,  1.18988467,  1.832],
                           [-2.13832676,  0.62232832,  0.958],
                           [-2.5042362 ,  1.63777023,  0.907],
                           [-2.51699797,  0.07177849,  0.109],
                           [-2.50306232,  0.16268392,  1.864]])

    assert new_pep.residues[0].resname == 'ACE'
    assert new_pep.residues[0].resnum == 0
    np.testing.assert_allclose(new_pep.residues[0].positions, ace_coords)

    nhh_coords = np.array([[21.63728595, 20.70101107, 18.63843541],
                           [21.81199224, 22.11387508, 18.92988522],
                           [20.84481362, 20.39311986, 18.09318996],
                           [21.42355845, 22.33956019, 19.91116725],
                           [22.86200865, 22.37037609, 18.90219429],
                           [21.28635322, 22.70933948, 18.19786284]])


    assert new_pep.residues[-1].resname == 'NME'
    assert new_pep.residues[-1].resnum == 24
    np.testing.assert_allclose(new_pep.residues[-1].positions, nhh_coords)


def test_add_cap():
    chilife.store_cap('ZZZ', 'test_data/ACE_A_NME.pdb', 'N')
    chilife.store_cap('XXX', 'test_data/ACE_A_NME.pdb', 'C')

    assert (chilife.RL_DIR / f'cap_pdbs/XXX.pdb').exists()
    assert (chilife.RL_DIR / f'cap_pdbs/ZZZ.pdb').exists()

    os.remove(chilife.RL_DIR / f'cap_pdbs/XXX.pdb')
    os.remove(chilife.RL_DIR / f'cap_pdbs/ZZZ.pdb')