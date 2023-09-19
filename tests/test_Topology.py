import MDAnalysis
import igraph as ig
import chilife as xl
from chilife.Topology import *
from MDAnalysis.topology.guessers import guess_angles, guess_dihedrals

ubq = xl.Protein.from_pdb('test_data/1ubq.pdb')
ubqu = MDAnalysis.Universe('test_data/1ubq.pdb')
bonds = xl.guess_bonds(ubq.coords, ubq.atypes)
ubqu.add_bonds(bonds)
graph1 = ig.Graph(n = len(ubq.atoms), edges=bonds)

def test_get_angle_defs():
    angles = set(get_angle_defs(graph1))
    angles2 = set(guess_angles(ubqu.bonds))

    assert angles == angles2


def test_get_dihedral_defs():
    dihedrals = set(get_dihedral_defs(graph1))
    dihedrals2 = set(guess_dihedrals(ubqu.bonds))

    assert dihedrals == dihedrals2


def test_construction():
    top = Topology(ubq, bonds)
    for key, val in top.dihedrals_by_resnum.items():
        names = tuple(ubq.atoms[idx].name for idx in val)
        resnum = ubq.atoms[val[-1]].resnum
        chain = ubq.atoms[val[-1]].segid
        assert chain == key[0]
        assert resnum == key[1]
        assert names == key[2:]
