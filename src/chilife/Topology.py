from typing import List, Tuple
from itertools import combinations, product

import numpy as np
import igraph as ig


class Topology:
    """
    Topology class

    Parameters
    ----------
    mol : MDAnalysis.Universe, MDAnalysis.AtomGroup, chiLife.MolecularSystemBase
        Molecular system from which to define a topology.
    bonds : ArrayLike
        Array of tuples defining all the bonds of the molecule.
    kwargs : dict
        Additional keyword arguments, usually used to speed up construction by providing precomputed values for the
        topology attributes.

        graph : igraph.Graph
            A Graph object of the  molecule topology.
        angles : ArrayLike
            Array defining all bond-angles of the molecule.
        dihedrals : ArrayLike
            Array defining all dihedral angles of the molecule.

    """


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

    @property
    def ring_idxs(self):
        fund_cycles = self.graph.fundamental_cycles()
        cyverts = set()
        for cycle in fund_cycles:
            for edge in self.graph.es(cycle):
                cyverts.update(edge.tuple)

        return sorted(cyverts)

    @property
    def has_rings(self):
        if self.ring_idxs == []:
            return False
        else:
            return True


    def get_zmatrix_dihedrals(self):
        """
        Get the dihedral definitions for the z-matrix.

        Returns
        -------
        zmatrix_dihedrals : ArrayLike

        """
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
        """Create an igraph.Graph form the topology"""
        return ig.Graph(n=len(self.atom_idxs), edges=self.bonds)

    def update_resnums(self):
        """Update the residue numbers of each atom with respect to the dihedral that they belong to."""
        self.dihedrals_by_resnum = {}
        for dihe in self.dihedrals:
            c = dihe[2]
            n1, n2, n3, n4 = self.atom_names[list(dihe)]
            r1 = self.atoms[c].resnum
            c1 = self.atoms[c].segid
            self.dihedrals_by_resnum[c1, r1, n1, n2, n3, n4] = dihe


def get_angle_defs(graph: ig.Graph) -> Tuple[Tuple[int, int, int]]:
    """
    Get all angle definitions for the topology defined by the graph.

    Parameters
    ----------
    graph: igraph.Graph
        A graph of the molecular topology.

    Returns
    -------
    angles : Tuple[Tuple[int, int, int]]
        Tuple containing tuples defining all angles of the molecule/molecular system.
    """
    angles = []
    for node in graph.vs.indices:
        neighbors = tuple(graph.neighbors(node))
        if len(neighbors) > 1:
            atom_angles = [(min(c), node, max(c)) for c in combinations(neighbors, 2)]
            angles += [tuple(a) for a in atom_angles]

    return tuple(angles)


def get_dihedral_defs(graph):
    """
    Get all dihedral definitions for the topology defined by the graph.

    Parameters
    ----------
    graph: igraph.Graph
        A graph of the molecular topology.

    Returns
    -------
    dihedrals : Tuple[Tuple[int, int, int, int]]
        Tuple containing tuples defining all dihedrals of the molecule/molecular system.
    """

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
