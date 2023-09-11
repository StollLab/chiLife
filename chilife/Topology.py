from typing import List, Tuple
from itertools import combinations, product

import numpy as np
import igraph as ig


class Topology:

    def __init__(self, atoms, bonds, **kwargs):
        self.atoms = atoms
        self.bonds = bonds

        self.graph = kwargs.get('graph', self._make_graph())
        self.angles = kwargs.get('angles', get_angle_defs(self.graph))
        self.dihedrals = kwargs.get('dihedrals', get_dihedral_defs(self.graph))

        # self.dihedrals = kwargs.get('dihedrals', get_dihedral_defs(self.graph))

    def _make_graph(self):
        return ig.Graph(n=len(self.atoms), edges=self.bonds)


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

            bond_dihedrals = [(min(c), a, b, max(c)) for c in product(a_neighbors, b_neighbors)
                                                  if all(idx not in c for idx in (a, b))]
            dihedrals += [tuple(a) for a in bond_dihedrals]

    return tuple(dihedrals)
