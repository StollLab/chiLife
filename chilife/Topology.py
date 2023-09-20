from typing import List, Tuple
from itertools import combinations, product

import numpy as np
import igraph as ig


class Topology:

    def __init__(self, mol, bonds, **kwargs):

        mol = mol.atoms
        self.atoms = mol.atoms
        self.atom_names = self.atoms.names
        self.atom_idxs = np.arange(len(mol))
        self.bonds = bonds

        self.graph = kwargs.get('graph', self._make_graph())
        self.angles = kwargs.get('angles', get_angle_defs(self.graph))
        self.dihedrals = kwargs.get('dihedrals', get_dihedral_defs(self.graph))

        self.dihedrals_by_bonds = {}
        self.dihedrals_by_atoms = {}
        self.dihedrals_by_resnum = {}

        for dihe in self.dihedrals:
            b, c, e = dihe[1:]
            bc_list = self.dihedrals_by_bonds.setdefault((b, c), [])
            bc_list.append(dihe)

            e_list = self.dihedrals_by_atoms.setdefault(e, [])
            e_list.append(dihe)
            n1, n2, n3, n4 = self.atom_names[list(dihe)]
            r1 = self.atoms[c].resnum
            c1 = self.atoms[c].segid
            self.dihedrals_by_resnum[c1, r1, n1, n2, n3, n4] = dihe

    def get_zmatrix_dihedrals(self):
        zmatrix_dihedrals = []
        hold = []
        for key in self.atom_idxs:
            if key not in self.dihedrals_by_atoms:
                hold.append(key)
                continue

            if len(hold) > 0:
                for i, elem in enumerate(hold):
                    zmatrix_dihedrals.append(hold[:i + 1])
                hold = []

            runner_ups = []
            for dihe in self.dihedrals_by_atoms[key]:
                if dihe[0] < dihe[1] < dihe[2] < dihe[3]:
                    zmatrix_dihedrals.append(list(dihe))
                    break
                elif max(dihe[:3]) < dihe[3]:
                    runner_ups.append(dihe)
            else:
                if len(runner_ups) > 0:
                    zmatrix_dihedrals.append(list(runner_ups[0]))

        return zmatrix_dihedrals


    def _make_graph(self):
        return ig.Graph(n=len(self.atom_idxs), edges=self.bonds)


def get_angle_defs(graph: ig.Graph) -> Tuple[Tuple[int, int, int]]:
    """

    Parameters
    ----------
    graph:

    Returns
    -------

    """
    angles = []
    for node in graph.vs.indices:
        neighbors = tuple(graph.neighbors(node))
        if len(neighbors) > 1:
            atom_angles = [(min(c), node, max(c)) for c in combinations(neighbors, 2)]
            angles += [tuple(a) for a in atom_angles]

    return tuple(angles)


def get_dihedral_defs(graph):
    dihedrals = []
    for a, b in graph.get_edgelist():
        a_neighbors = tuple(graph.neighbors(a))
        b_neighbors = tuple(graph.neighbors(b))

        if len(a_neighbors) > 1 and len(b_neighbors) > 1:

            bond_dihedrals = [(aa, a, b, bb) if aa < bb else (bb, b, a, aa)
                                             for aa, bb in product(a_neighbors, b_neighbors)
                                             if all(idx not in (aa, bb) for idx in (a, b))]

            dihedrals += [tuple(a) for a in bond_dihedrals]

    return tuple(dihedrals)
