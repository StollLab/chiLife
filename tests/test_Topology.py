import pickle
import pytest
import MDAnalysis
import chilife as xl
from chilife.Topology import *
from MDAnalysis.topology.guessers import guess_angles, guess_dihedrals

ubq = xl.MolSys.from_pdb('test_data/1ubq.pdb', sort_atoms=True)
mbp = MDAnalysis.Universe("test_data/2klf.pdb", in_memory=True)
ubqu = MDAnalysis.Universe('test_data/1ubq.pdb')


def test_guess_bonds():
    bonds = xl.guess_bonds(mbp.atoms.positions, mbp.atoms.types)
    dist = np.linalg.norm(mbp.atoms.positions[bonds[:, 0]] - mbp.atoms.positions[bonds[:, 1]], axis=1)

    ans = np.load('test_data/bonds.npy')
    ans_dist = np.load('test_data/bond_dist.npy')
    sorted_args = np.lexsort((ans[:, 0], ans[:, 1]))

    np.testing.assert_equal(bonds, ans[sorted_args])
    np.testing.assert_allclose(dist, ans_dist[sorted_args])


@pytest.mark.parametrize('prot', [ubqu, mbp])
def test_get_angle_defs(prot):
    bonds = xl.guess_bonds(prot.atoms.positions, prot.atoms.types)
    prot.add_bonds(bonds)
    graph1 = ig.Graph(n=len(prot.atoms), edges=bonds)
    angles = set(get_angle_defs(graph1))

    angles2 = set(guess_angles(prot.bonds))

    assert angles == angles2


@pytest.mark.parametrize('prot', [ubqu, mbp])
def test_get_dihedral_defs(prot):
    bonds = xl.guess_bonds(prot.atoms.positions, prot.atoms.types)
    prot.add_bonds(bonds)

    graph1 = ig.Graph(n=len(prot.atoms), edges=bonds)
    dihedrals = set(get_dihedral_defs(graph1))

    prot.add_angles(guess_angles(prot.bonds))
    dihedrals2 = set(guess_dihedrals(prot.angles))

    assert dihedrals == dihedrals2


def test_construction():
    bonds = xl.guess_bonds(ubq.coords, ubq.atypes)
    top = Topology(ubq, bonds)
    for key, val in top.dihedrals_by_resnum.items():

        names = tuple(ubq.atoms[idx].name for idx in val)
        assert names == key[2:]

        resnum = ubq.atoms[val[-2]].resnum
        assert resnum == key[1]

        chain = ubq.atoms[val[-1]].segid
        assert chain == key[0]


def test_get_z_matrix():
    mbp_bonds = xl.guess_bonds(mbp.atoms.positions, mbp.atoms.types)
    top = Topology(mbp, mbp_bonds)

    test_idxs = top.get_zmatrix_dihedrals()

    with open('test_data/top_idxs.pkl', 'rb') as f:
        ans_idxs = pickle.load(f)

    for a, b in zip(test_idxs, ans_idxs):
        np.testing.assert_equal(a, b)


def test_dihedrals_by_atom():
    mbp_bonds = xl.guess_bonds(mbp.atoms.positions, mbp.atoms.types)
    top = Topology(mbp, mbp_bonds)


    with open('test_data/mbp_dihedrals.pkl', 'rb') as f:
        ans = pickle.load(f)

    assert top.dihedrals == ans

    with open('test_data/dihedrals_by_atom.pkl', 'rb') as f:
        ans = pickle.load(f)

    assert top.dihedrals_by_atoms == ans


def test_has_rings():
    Y59 = ubq.select_atoms('resnum 59')
    N60 = ubq.select_atoms('resnum 60')
    Y59_top = Topology(Y59, xl.guess_bonds(Y59.positions, Y59.types))
    N60_top = Topology(N60, xl.guess_bonds(N60.positions, N60.types))

    assert Y59_top.has_rings
    assert not N60_top.has_rings


def test_cycle_idx():
    Y59 = ubq.select_atoms('resnum 59')
    N60 = ubq.select_atoms('resnum 60')
    Y59_top = Topology(Y59, xl.guess_bonds(Y59.positions, Y59.types))
    N60_top = Topology(N60, xl.guess_bonds(N60.positions, N60.types))

    np.testing.assert_equal(Y59_top.ring_idxs, [[5, 6, 7, 8, 9, 10]])
    assert N60_top.ring_idxs == []
