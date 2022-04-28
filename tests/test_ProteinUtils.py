import os, hashlib
from functools import partial
import numpy as np
import pytest
import MDAnalysis as mda
import chiLife

pdbids = ['1ubq', '1a2w']
ubq = chiLife.fetch('1ubq')
resis = [(key, -90, 160) for key in chiLife.dihedral_defs
         if key not in (chiLife.SUPPORTED_LABELS + ('R1B', 'R1C', 'CYR1', 'MTN', 'ALA', 'GLY'))]
ICs = chiLife.get_internal_coords(ubq)
gd_kwargs = [{'resi': 28, 'atom_list': ['C', 'N', 'CA', 'C']},
             {'resi': 28, 'atom_list': [['C', 'N', 'CA', 'C'], ['N', 'CA', 'C', 'N']]}]
gd_ans = [-1.1540443794802524, np.array([-1.15404438, -0.66532042])]

@pytest.mark.parametrize('res', resis)
def test_read_dunbrack( res):
    res, phi, psi = res

    dlib = chiLife.read_bbdep(res, -70, 90)

    with np.load(f'test_data/{res}_{phi}_{psi}.npz', allow_pickle=True) as f:
        dlib_ans = {key: f[key] for key in f if key != 'allow_pickle'}

    for key in dlib_ans:
        if dlib_ans[key].dtype not in [np.dtype(f'<U{i}') for i in range(1, 5)]:
            np.testing.assert_almost_equal(dlib_ans[key], dlib[key])
        else:
            assert np.all(np.char.equal(dlib_ans[key], dlib[key]))


@pytest.mark.parametrize('pdbid', pdbids)
def test_get_internal_coordinates(pdbid):
    protein = chiLife.fetch(pdbid).select_atoms('protein and not altloc B')
    protein2 = protein.universe.copy().select_atoms('protein and not altloc B')
    ICs = chiLife.get_internal_coords(protein)

    coords = ICs.to_cartesian()
    protein2.atoms.positions = coords

    # Calculate max absolute deviation
    aa_MaxAD = np.linalg.norm(protein2.atoms.positions - protein.atoms.positions, axis=1).max()

    assert aa_MaxAD < 0.1  # Angstroms


def test_icset_dihedral():
    ICs = chiLife.get_internal_coords(ubq)
    ICs.set_dihedral([-np.pi / 2, np.pi / 2], 72, [['C', 'N', 'CA', 'C'], ['N', 'CA', 'C', 'N']])
    coords = ICs.to_cartesian()
    ubq2 = ubq.universe.copy().select_atoms('protein and not altloc B')
    ubq2.atoms.positions = coords

    ans = mda.Universe('test_data/1ubq.pdb')
    np.testing.assert_almost_equal(ans.atoms.positions, ubq2.atoms.positions, decimal=3)


def test_sort_pdb():
    pdbfile = 'test_data/trt.pdb'
    x = chiLife.sort_pdb(pdbfile)
    with open('test_data/trt_tmp.pdb', 'w') as f:
        for line in x:
            f.write(line)

    with open('test_data/trt_tmp.pdb', 'r') as f:
        test = hashlib.md5(f.read().encode('utf-8')).hexdigest()

    with open('test_data/trt_sorted.pdb', 'r') as f:
        ans = hashlib.md5(f.read().encode('utf-8')).hexdigest()

    os.remove('test_data/trt_tmp.pdb')
    assert test == ans


def test_sort_pdb2():
    x = chiLife.sort_pdb('test_data/GR1G.pdb')

    with open('test_data/GR1G_tmp.pdb', 'w') as f:
        for line in x:
            f.write(line)

    with open('test_data/GR1G_tmp.pdb', 'rb') as f:
        test = hashlib.md5(f.read()).hexdigest()

    os.remove('test_data/GR1G_tmp.pdb')


def test_sort_pdb3():
    x = chiLife.sort_pdb('test_data/SL_GGAGG.pdb')

    with open('test_data/SL_GGAGG_tmp.pdb', 'w') as f:
        for line in x:
            f.write(line)

    U = mda.Universe('test_data/SL_GGAGG_tmp.pdb', in_memory=True)
    os.remove('test_data/SL_GGAGG_tmp.pdb')
    ICs = chiLife.get_internal_coords(U, preferred_dihedrals=[['C', 'N', 'CA', 'C']])


def test_mutate():
    protein = chiLife.fetch('1ubq').select_atoms('protein')
    SL = chiLife.SpinLabel('R1C', site=28, protein=protein, energy_func=partial(chiLife.get_lj_rep, forgive=0.8))

    labeled_protein = chiLife.mutate(protein, SL)
    ub1_A28R1 = mda.Universe('test_data/1ubq_A28R1.pdb')

    np.testing.assert_almost_equal(ub1_A28R1.atoms.positions, labeled_protein.atoms.positions, decimal=3)


def test_mutate2():
    protein = chiLife.fetch('1ubq').select_atoms('protein')
    SL1 = chiLife.SpinLabel('R1C', site=28, protein=protein, energy_func=partial(chiLife.get_lj_rep, forgive=0.8))
    SL2 = chiLife.SpinLabel('R1C', site=48, protein=protein, energy_func=partial(chiLife.get_lj_rep, forgive=0.8))

    labeled_protein = chiLife.mutate(protein, SL1, SL2)
    ub1_A28R1_K48R1 = mda.Universe('test_data/ub1_A28R1_K48R1.pdb')
    np.testing.assert_almost_equal(ub1_A28R1_K48R1.atoms.positions, labeled_protein.atoms.positions, decimal=3)


def test_mutate3():
    protein = mda.Universe('test_data/1omp_H.pdb').select_atoms('protein')
    SL1 = chiLife.SpinLabel('R1C', site=238, protein=protein, use_H=True)
    SL2 = chiLife.SpinLabel('R1C', site=311, protein=protein, use_H=True)

    labeled_protein = chiLife.mutate(protein, SL1, SL2)
    assert len(labeled_protein.atoms) != len(protein.atoms)

def test_mutate4():
    protein = mda.Universe('test_data/1omp_H.pdb').select_atoms('protein')
    D41G = chiLife.RotamerLibrary('GLY', 41, protein=protein)
    S238A = chiLife.RotamerLibrary('ALA', 238, protein=protein)
    mPro = chiLife.mutate(protein, D41G, S238A, add_missing_atoms=False)
    D41G_pos = mPro.select_atoms('resnum 41').positions
    S238A_pos = mPro.select_atoms('resnum 238').positions
    np.testing.assert_almost_equal(D41G.coords[0], D41G_pos, decimal=6)
    np.testing.assert_almost_equal(S238A.coords[0], S238A_pos, decimal=6)



@pytest.mark.parametrize(['inp', 'ans'], zip(gd_kwargs, gd_ans))
def test_get_dihedral(inp, ans):
    dihedral = ICs.get_dihedral(**inp)
    np.testing.assert_almost_equal(dihedral, ans)


def test_ProteinIC_save_pdb():
    protein = mda.Universe('test_data/alphabetical_peptide.pdb').select_atoms('protein')
    
    uni_ics = chiLife.get_internal_coords(protein)
    uni_ics.save_pdb('test_data/postwrite_alphabet_peptide.pdb')

    with open('test_data/postwrite_alphabet_peptide.pdb', 'r') as f:
        test = hashlib.md5(f.read().encode('utf-8')).hexdigest()

    with open('test_data/alphabetical_peptide.pdb', 'r') as f:
        truth = hashlib.md5(f.read().encode('utf-8')).hexdigest()

    os.remove('test_data/postwrite_alphabet_peptide.pdb')
    assert test == truth


def test_has_clashes():
    assert not ICs.has_clashes()
    ICs.set_dihedral(np.pi/2, 35, ['N', 'CA', 'C', 'N'])
    assert ICs.has_clashes()


def test_add_missing_atoms():
    protein = chiLife.fetch('1omp').select_atoms('protein')
    new_prot = chiLife.mutate(protein)
    assert len(new_prot.atoms) != len(protein.atoms)
    assert len(new_prot.atoms) == 2877

def test_get_internal_coords():
    polypro = mda.Universe('test_data/PPII_Capped.pdb')
    polyproIC = chiLife.get_internal_coords(polypro)

@pytest.mark.parametrize('res', chiLife.SUPPORTED_RESIDUES)
def test_sort_and_internal_coords(res):
    pass
