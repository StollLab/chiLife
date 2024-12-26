import hashlib, os, pickle
from itertools import chain
import pytest
import numpy as np
import MDAnalysis as mda
import chilife as xl

pdbids = ["1ubq", "1a2w", '1az5']
ubq = mda.Universe("test_data/1ubq.pdb", in_memory=True)
mbp = mda.Universe("test_data/2klf.pdb", in_memory=True)

ubqIC = xl.MolSysIC.from_atoms(ubq)
mbpIC = xl.MolSysIC.from_atoms(mbp)


gd_kwargs = [
    {"resi": 28, "atom_list": ["C", "N", "CA", "C"]},
    {"resi": 28, "atom_list": [["C", "N", "CA", "C"], ["N", "CA", "C", "N"]]},
]
gd_ans = [-1.1540451, np.array([-1.1540451, -0.6653183])]


def test_from_prot():

    with np.load('test_data/ubq_zmats.npz') as f:
        zmat_ans = f['zmat']
        zmat_idx_ans = f['zmat_idxs']


    non_nan_idx = ubqIC.z_matrix_idxs > 0
    np.testing.assert_equal(ubqIC.z_matrix_idxs[non_nan_idx], zmat_idx_ans[non_nan_idx])
    np.testing.assert_almost_equal(ubqIC.trajectory.coordinate_array[:, ubqIC.non_nan_idxs],
                                   zmat_ans[:, ubqIC.non_nan_idxs], decimal=6)

    for i, idxs in enumerate(ubqIC.z_matrix_idxs):
        idxt = [val for val in idxs if val != -2147483648]
        assert i == idxt[0]


def test_from_prot_traj():

    with np.load('test_data/mbp_zmats.npz') as f:
        zmat_ans = f['zmat']
        zmat_idx_ans = f['zmat_idxs']

    np.testing.assert_equal(mbpIC.z_matrix_idxs[mbpIC.non_nan_idxs], zmat_idx_ans[mbpIC.non_nan_idxs])
    np.testing.assert_almost_equal(mbpIC.trajectory.coordinate_array[:, mbpIC.non_nan_idxs],
                                   zmat_ans[:, mbpIC.non_nan_idxs], decimal=6)


@pytest.mark.parametrize("pdbid", pdbids)
def test_get_internal_coordinates(pdbid):
    protein = mda.Universe(f"test_data/{pdbid}.pdb", in_memory=True).select_atoms("protein and not altloc B")
    ICs = xl.MolSysIC.from_atoms(protein, ignore_water=False)
    np.testing.assert_almost_equal(ICs.coords, protein.atoms.positions, decimal=4)

    protein = xl.MolSys.from_pdb(f"test_data/{pdbid}.pdb").select_atoms("protein and not altloc B")
    ICs = xl.MolSysIC.from_atoms(protein, ignore_water=False)
    np.testing.assert_almost_equal(ICs.coords, protein.atoms.positions, decimal=4)


def test_copy():
    my_copy = ubqIC.copy()

    assert np.all(my_copy.z_matrix_idxs == ubqIC.z_matrix_idxs)
    assert my_copy.z_matrix_idxs is not ubqIC.z_matrix_idxs

    assert np.all(ubqIC.trajectory.coordinate_array == my_copy.trajectory.coordinate_array)
    assert ubqIC.trajectory.coordinate_array is not my_copy.trajectory.coordinate_array
    for v1, v2 in zip(my_copy.chain_operators.values(), ubqIC.chain_operators.values()):
        for v11, v22 in zip(v1.values(), v2.values()):
            assert np.all(v11 == v22)


def test_save_pdb():
    protein = mda.Universe("test_data/alphabetical_peptide.pdb").select_atoms("protein")

    uni_ics = xl.MolSysIC.from_atoms(protein)
    xl.save("alphabetical_peptide.pdb", uni_ics)

    with open("alphabetical_peptide.pdb", "r") as f:
        test = hashlib.md5(f.read().encode("utf-8")).hexdigest()

    with open("test_data/alphabetical_peptide.pdb", "r") as f:
        truth = hashlib.md5(f.read().encode("utf-8")).hexdigest()

    os.remove("alphabetical_peptide.pdb")
    assert test == truth


def test_has_clashes():
    local_ubqIC = ubqIC.copy()
    assert not local_ubqIC.has_clashes()
    local_ubqIC.set_dihedral(np.pi / 2, 35, ["N", "CA", "C", "N"])
    assert local_ubqIC.has_clashes()


def test_set_dihedral():
    local_ubqIC = ubqIC.copy()
    local_ubqIC.set_dihedral([-np.pi / 2, np.pi / 2], 72, [["C", "N", "CA", "C"], ["N", "CA", "C", "N"]])

    ans = mda.Universe("test_data/test_icset.pdb")
    np.testing.assert_almost_equal(ans.atoms.positions, local_ubqIC.coords, decimal=3)


def test_set_dihedral2():
    lys = mda.Universe('../src/chilife/data/rotamer_libraries/residue_pdbs/lys.pdb')

    ICs = xl.MolSysIC.from_atoms(lys)
    ICs.set_dihedral(np.pi / 2, 1, ["N", "CA", "CB", "CG"])

    print('DSF')


@pytest.mark.parametrize(["inp", "ans"], zip(gd_kwargs, gd_ans))
def test_get_dihedral(inp, ans):
    ubqIC = xl.MolSysIC.from_atoms(ubq)
    dihedral = ubqIC.get_dihedral(**inp)
    np.testing.assert_almost_equal(dihedral, ans, decimal=4)



def test_polypro():
    polypro = mda.Universe("test_data/PPII_Capped.pdb")
    polyproIC = xl.MolSysIC.from_atoms(polypro)
    np.testing.assert_equal(polyproIC.z_matrix_names[12], ['CD', 'CG', 'CB', 'CA'])


def test_PRO():
    pro = mda.Universe("../src/chilife/data/rotamer_libraries/residue_pdbs/pro.pdb")
    pro_ic = xl.MolSysIC.from_atoms(pro)

    np.testing.assert_equal(pro_ic.z_matrix_names[6], ["CD", "CG", "CB", "CA"])

def test_PRO2():
    zmat_idxs = ubqIC.chain_res_name_map['A', 37, 'N', 'CA']
    np.testing.assert_equal(ubqIC.z_matrix_names[zmat_idxs[0]], ["C", "CA", "N", "C"] )


def test_set_coords():
    R1A = mda.Universe("test_data/R1A.pdb")
    R1A_IC = xl.MolSysIC.from_atoms(R1A)
    R1A_IC_c = R1A_IC.copy()
    R1A_IC_c.set_dihedral([np.pi/2, -np.pi/2, np.pi/2], 1, [['N', 'CA', 'CB', 'SG' ],
                                                            ['CA', 'CB', 'SG', 'SD'],
                                                            ['CB', 'SG', 'SD', 'CE']])

    coords = R1A_IC_c.coords

    R1A_IC.coords = coords
    np.testing.assert_allclose(R1A_IC.z_matrix, R1A_IC_c.z_matrix, rtol=1e-5)
    np.testing.assert_almost_equal(R1A_IC.coords, R1A_IC_c.coords, decimal=6)


def test_nonbonded():
    with open('test_data/ic_nb.pkl', 'rb') as f:
        nb_ans = pickle.load(f)

    np.testing.assert_equal(ubqIC.nonbonded, nb_ans)


def test_get_zmat_idxs():
    R1A = mda.Universe("test_data/R1A.pdb")
    R1A_IC = xl.MolSysIC.from_atoms(R1A)
    idxs = R1A_IC.get_z_matrix_idxs(1, ['CB', 'SG', 'SD', 'CE'])

    np.testing.assert_equal(idxs, 7)

    with pytest.raises(RuntimeError):
        idxs = R1A_IC.get_z_matrix_idxs(1, ['CE', 'SD', 'SG', 'CB'])


def test_phi_idxs():
    idxs = ubqIC.phi_idxs(range(4, 11,))
    vals = ubqIC.z_matrix[idxs, -1]
    ans = [ubqIC.get_dihedral(i, ('C', 'N', 'CA', 'C')) for i in range(4, 11)]

    np.testing.assert_almost_equal(vals, ans, decimal=5)

    idx = ubqIC.phi_idxs(1)
    assert len(idx) == 0


def test_psi_idxs():
    idxs = ubqIC.psi_idxs(range(4, 11,))
    vals = ubqIC.z_matrix[idxs, -1]
    ans = [ubqIC.get_dihedral(i, ('N', 'CA', 'C', 'N')) for i in range(4, 11)]

    np.testing.assert_almost_equal(vals, ans, decimal=4)

    idx = ubqIC.psi_idxs(76)
    assert len(idx) == 0

    idxs = ubqIC.psi_idxs()
    assert 0 not in idxs




def test_phi_psi_idxs_multichain():
    prot = xl.fetch('1a2w').select_atoms('protein')
    ICs = xl.MolSysIC.from_atoms(prot)

    with pytest.raises(ValueError):
        ICs.psi_idxs(10)

    with pytest.raises(ValueError):
        ICs.psi_idxs(10)

    A = ICs.psi_idxs(10, chain='A')
    B = ICs.phi_idxs(10, chain='A')

    assert A[0] == 80
    assert B[0] == 71

    A = ICs.psi_idxs(10, chain='B')
    B = ICs.phi_idxs(10, chain='B')

    assert A[0] == 1031
    assert B[0] == 1022

def test_chi_idxs():
    idxs = ubqIC.chi_idxs(range(4, 11, ))
    idxs = list(chain.from_iterable(idxs))
    vals = ubqIC.z_matrix[idxs, -1]
    ans = np.array([-1.0589304 ,  1.70940745, -3.13784337, -3.06411982,  3.0752275 ,
                    -3.00272059, -3.04088092,  1.33890903, -1.18974721, -2.89442134,
                     1.23405945])

    np.testing.assert_almost_equal(vals, ans, decimal=6)

def test_ic_pref_dihe():

    mol = mda.Universe('test_data/test_ic_pref_dihe.pdb', in_memory=True)
    dih = [['N', 'CA', 'CB', 'CB2'],
           ['CA', 'CB', 'CB2', 'NG'],
           ['ND', 'CE3', 'CZ3', 'C31']]

    IC = xl.MolSysIC.from_atoms(mol, preferred_dihedrals=dih)
    IC.set_dihedral(np.pi / 2, 1, ['ND', 'CE3', 'CZ3', 'C31'])

    sister_dihe_atom_coords = IC.coords[IC.atom_names == 'C36'].flat
    np.testing.assert_almost_equal(sister_dihe_atom_coords, [2.1495337, -3.4064763, -0.9388056], decimal=6)


def test_pickle():
    mol1 = mda.Universe('test_data/PPII_Capped.pdb', in_memory=True)
    mol2 = xl.MolSys.from_pdb('test_data/PPII_Capped.pdb')
    for mol in (mol1, mol2):
        prot2 = mol

        with open('tmp.pkl', 'wb') as f:
            pickle.dump(prot2, f)

        del prot2
        del mol

        with open('tmp.pkl', 'rb') as f:
            prot2 = pickle.load(f)

    os.remove('tmp.pkl')


def test_chain_operators():
    LYS = mda.Universe('../src/chilife/data/rotamer_libraries/residue_pdbs/lys.pdb')
    pic = xl.MolSysIC.from_atoms(LYS)

    mx = np.array([[0.38281548,  0.9238248,   0.],
                   [0.92382485, -0.38281548,  0.],
                   [0.,          0., -0.99999994]])
    ori = np.array([-0.449, -3.572, -1.666])

    np.testing.assert_almost_equal(pic.chain_operators[0]['mx'], mx)
    np.testing.assert_almost_equal(pic.chain_operators[0]['ori'], ori)


def test_iter():
    zdata = []
    cdata = []
    for ic in mbpIC:
        cdata.append(ic.coords.copy())
        zdata.append(ic.z_matrix.copy())

    zans = []
    cans = []
    for ts in mbpIC.trajectory:
        cans.append(mbpIC.coords.copy())
        zans.append(mbpIC.z_matrix.copy())

    assert np.sum(np.abs(cdata[0] - cdata[1])) > 0
    np.testing.assert_equal(cdata, cans)
    np.testing.assert_equal(zdata, zans)

def test_load_new():
    pass

def test_use_frames():
    pass