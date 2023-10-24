import pickle
import MDAnalysis
import chilife as xl
from chilife.Topology import *
from MDAnalysis.topology.guessers import guess_angles, guess_dihedrals

ubq = xl.Protein.from_pdb('test_data/1ubq.pdb', sort_atoms=True)
mbp = MDAnalysis.Universe("test_data/2klf.pdb", in_memory=True)

ubqu = MDAnalysis.Universe('test_data/1ubq.pdb')
bonds = xl.guess_bonds(ubq.coords, ubq.atypes)
ubqu.add_bonds(bonds)
graph1 = ig.Graph(n=len(ubq.atoms), edges=bonds)

def test_get_angle_defs():
    angles = set(get_angle_defs(graph1))
    angles2 = set(guess_angles(ubqu.bonds))

    assert angles == angles2


def test_get_dihedral_defs():
    dihedrals = set(get_dihedral_defs(graph1))
    ubqu.add_angles(guess_angles(ubqu.bonds))
    dihedrals2 = set(guess_dihedrals(ubqu.angles))

    assert dihedrals == dihedrals2


def test_construction():
    top = Topology(ubq, bonds)
    for key, val in top.dihedrals_by_resnum.items():

        names = tuple(ubq.atoms[idx].name for idx in val)
        assert names == key[2:]

        resnum = ubq.atoms[val[-2]].resnum
        assert resnum == key[1]

        chain = ubq.atoms[val[-1]].segid
        assert chain == key[0]

def test_get_z_matrix():
    mbp_bonds = xl.guess_bonds(mbp.atoms.positions, mbp.atoms.names)
    top = Topology(mbp, mbp_bonds)

    test_idxs = top.get_zmatrix_dihedrals()
    ans_idxs = np.load('test_data/top_idxs.npy', allow_pickle=True)
    np.testing.assert_equal(test_idxs, ans_idxs)

def test_dihedrals_by_atom():
    mbp_bonds = xl.guess_bonds(mbp.atoms.positions, mbp.atoms.names)
    top = Topology(mbp, mbp_bonds)
    with open('test_data/dihedrals_by_atom.pkl', 'rb') as f:
        ans = pickle.load(f)
    assert top.dihedrals_by_atoms == ans
