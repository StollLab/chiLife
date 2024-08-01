from typing import List, Tuple, Set
from itertools import combinations, product
from operator import itemgetter

import numpy as np
from numpy.typing import ArrayLike
from scipy.spatial import cKDTree

import igraph as ig

from .globals import bond_hmax_dict
from .math_utils import simple_cycle_vertices

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
        self.bonds_any_atom = {}
        for b in self.bonds:
            for at in b:
                self.bonds_any_atom.setdefault(at, []).append(b)

        self.graph = kwargs.get('graph', self._make_graph())
        self.angles = kwargs.get('angles', get_angle_defs(self.graph))
        self.angles_any_atom = {}
        for a in self.angles:
            for at in a:
                self.angles_any_atom.setdefault(at, []).append(a)

        self.dihedrals = kwargs.get('dihedrals', get_dihedral_defs(self.graph))
        self.degree = []
        self.dihedrals_by_bonds = {}
        self.dihedrals_by_atoms = {}
        self.dihedrals_any_atom = {}
        self.dihedrals_by_resnum = {}

        for dihe in self.dihedrals:
            b, c, e = dihe[1:]
            bc_list = self.dihedrals_by_bonds.setdefault((b, c), [])
            bc_list.append(dihe)

            e_list = self.dihedrals_by_atoms.setdefault(e, [])
            e_list.append(dihe)

            for at in dihe:
                self.dihedrals_any_atom.setdefault(at, []).append(dihe)

            n1, n2, n3, n4 = self.atom_names[list(dihe)]
            r1 = self.atoms[c].resnum
            c1 = self.atoms[c].segid
            self.dihedrals_by_resnum[c1, r1, n1, n2, n3, n4] = dihe

    @property
    def ring_idxs(self):
        """Indices of atoms that are a part of one or more rings."""
        return simple_cycle_vertices(self.graph)

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
            elif 0 < len(hold) < 3:
                hold.append(key)
                continue
            else:
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
                    minrup = np.argmin(np.sum(np.abs(np.diff(runner_ups, axis=1)), axis=1))
                    zmatrix_dihedrals.append(list(runner_ups[minrup]))

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
                                             if all(idx not in (aa, bb) for idx in (a, b)) and aa != bb]

            dihedrals += [tuple(a) for a in bond_dihedrals]

    return tuple(dihedrals)


def get_min_topol(lines: List[List[str]],
                  forced_bonds: set = None) -> Set[Tuple[int, int]]:
    """ Git the minimum topology shared by all the states/models a PDB ensemble. This is to ensure a consistent
    internal coordinate system between all conformers of an ensemble even when there are minor differences in topology.
    e.g. when dHis-Cu-NTA has the capping ligand in different bond orientations.

    Parameters
    ----------
    lines : List[List[str]]
        List of lists corresponding to individual states/models of a pdb file. All models must have the same stoma in
        the same order and only the coordinates should differ.
    forced_bonds : set
        A set of bonds to that must be used regardless even if the bond lengths are not physically reasonable.
    Returns
    -------
    minimal_bond_set : Set[Tuple[int, int]]
        A set of tuples holding the indices of atom pairs which are thought to be bonded in all states/models.
    """
    bonds_list = []
    if isinstance(lines[0], str):
        lines = [lines]

    # Get bonds for all structures
    for struct in lines:
        coords = np.array([(line[30:38], line[38:46], line[46:54]) for line in struct], dtype=float)
        atypes = np.array([line[76:78].strip() for line in struct])
        pairs = guess_bonds(coords, atypes)
        bonds = set(tuple(pair) for pair in pairs)
        bonds_list.append(bonds)

    # Get the shared bonds between all structures.
    minimal_bond_set = set.intersection(*bonds_list)
    # Include any forced bonds
    if forced_bonds is not None:
        minimal_bond_set |= forced_bonds

    return minimal_bond_set


def guess_bonds(coords: ArrayLike, atom_types: ArrayLike) -> np.ndarray:
    """ Given a set of coordinates and their atom types (elements) guess the bonds based off an empirical metric.

    Parameters
    ----------
    coords : ArrayLike
        Array of three-dimensional coordinates of the atoms of a molecule or set of molecules for which you would like
        to guess the bonds of.
    atom_types : ArrayLike
        Array of the element symbols corresponding to the atoms of ``coords``

    Returns
    -------
    bonds : np.ndarray
        An array of the atom index pairs corresponding to the atom pairs that are thought ot form bonds.
    """
    atom_types = np.array([a.title() for a in atom_types])
    kdtree = cKDTree(coords)
    pairs = kdtree.query_pairs(4., output_type='ndarray')
    pair_names = [tuple(x) for x in atom_types[pairs].tolist()]
    bond_lengths = itemgetter(*pair_names)(bond_hmax_dict)
    a_atoms = pairs[:, 0]
    b_atoms = pairs[:, 1]

    dist = np.linalg.norm(coords[a_atoms] - coords[b_atoms], axis=1)
    bonds = pairs[dist < bond_lengths]
    sorted_args = np.lexsort((bonds[:, 0], bonds[:, 1]))
    return bonds[sorted_args]


def neighbors(edges, node):
    """
    Given a graph defined by edges and a node, find all neighbors of that node.

    Parameters
    ----------
    edges : ArrayLike
        Array of tuples defining all edges between nodes
    node : int
        The node of the graph for which to find neighbors.

    Returns
    -------
    nbs : ArrayLike
        Neighbor nodes.
    """
    nbs = []
    for edge in edges:
        if node not in edge:
            continue
        elif node == edge[0]:
            nbs.append(edge[1])
        elif node == edge[1]:
            nbs.append(edge[0])
    return nbs


def modified_bfs_edges(edges, root, bb_idxs):
    """
    Breadth first search of nodes given a set of edges
    Parameters
    ----------
    edges : ArrayLike
        Array of tuples defining edges between nodes.
    root : int
        Starting (root) node to begin the breadth first search at.

    Yields
    ------
    parent : int
        The node from which the children node stem
    child: List[int]
        All children node of parent.
    """
    nodes = np.unique(edges)

    depth_limit = len(nodes)
    seen = {root}

    n = len(nodes)
    depth = 0
    neigh = neighbors(edges, root)
    # Prioritize side chains
    neigh1 = [n for n in neigh if n not in bb_idxs]
    neigh2 = [n for n in neigh if n in bb_idxs]

    for neigh in neigh1, neigh2:
        next_parents_children = [(root, neigh)]
        while next_parents_children and depth < depth_limit:
            this_parents_children = next_parents_children
            next_parents_children = []
            for parent, children in this_parents_children:
                for child in children:
                    if child not in seen:
                        seen.add(child)
                        next_parents_children.append((child, neighbors(edges, child)))
                        yield parent, child
                if len(seen) == n:
                    return
            depth += 1
