from typing import List, Tuple, Set
from collections import defaultdict
from itertools import product, combinations
from operator import itemgetter
from enum import Enum

import numpy as np
from numpy.typing import ArrayLike
from scipy.spatial import cKDTree

import igraph as ig

import chilife.io
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

    def __init__(self, mol, bonds, bond_types=None, **kwargs):
        mol = mol.atoms
        self.atoms = mol.atoms
        self.atom_names = self.atoms.names
        self.atom_idxs = np.arange(len(mol))
        self.bonds = bonds
        self.bond_types = (
            bond_types
            if bond_types is not None
            else np.array([BondType.UNSPECIFIED for bond in self.bonds])
        )
        self.bonds_any_atom = {}
        for b in self.bonds:
            for at in b:
                self.bonds_any_atom.setdefault(at, []).append(b)

        self.graph = kwargs.get("graph", self._make_graph())
        self.angles = kwargs.get("angles", get_angle_defs(self.graph))

        self.angles_any_atom = {}
        for a in self.angles:
            for at in a:
                self.angles_any_atom.setdefault(at, []).append(a)

        self.dihedrals = (
            kwargs["dihedrals"]
            if "dihedrals" in kwargs
            else get_dihedral_defs(self.graph)
        )
        self.degree = []
        self.dihedrals_by_bonds = defaultdict(list)
        self.dihedrals_by_atoms = defaultdict(list)
        self.dihedrals_any_atom = defaultdict(list)
        self.dihedrals_by_resnum = {}

        resnums = self.atoms.resnums
        segids = self.atoms.segids
        for dihe in self.dihedrals:
            b, c, e = dihe[1:]
            self.dihedrals_by_bonds[(b, c)].append(dihe)
            self.dihedrals_by_atoms[e].append(dihe)

            for at in dihe:
                self.dihedrals_any_atom[at].append(dihe)

            n1, n2, n3, n4 = self.atom_names[list(dihe)]
            r1 = resnums[c]
            c1 = segids[c]
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
                    zmatrix_dihedrals.append(hold[: i + 1])
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
                    minrup = np.argmin(
                        np.sum(np.abs(np.diff(runner_ups, axis=1)), axis=1)
                    )
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


def get_angle_defs(graph: ig.Graph) -> np.ndarray:
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
            angles += [a for a in atom_angles]

    return np.array(angles, dtype=int)


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
        a_neighbors = graph.neighbors(a)
        b_neighbors = graph.neighbors(b)

        if len(a_neighbors) > 1 and len(b_neighbors) > 1:
            bond_dihedrals = [
                (aa, a, b, bb) if aa < bb else (bb, b, a, aa)
                for aa, bb in product(a_neighbors, b_neighbors)
                if all(idx not in (aa, bb) for idx in (a, b)) and aa != bb
            ]

            dihedrals += [a for a in bond_dihedrals]

    return np.array(dihedrals, dtype=int)


def get_min_topol(
    lines: List[List[str]], forced_bonds: set = None
) -> Set[Tuple[int, int]]:
    """Git the minimum topology shared by all the states/models a PDB ensemble. This is to ensure a consistent
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
        coords = np.array(
            [(line[30:38], line[38:46], line[46:54]) for line in struct], dtype=float
        )
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
    """Given a set of coordinates and their atom types (elements) guess the bonds based off an empirical metric.

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
    pairs = kdtree.query_pairs(4.0, output_type="ndarray")
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


class BondType(Enum):
    UNSPECIFIED = 0
    SINGLE = 1
    DOUBLE = 2
    TRIPLE = 3
    QUADRUPLE = 4
    QUINTUPLE = 5
    HEXTUPLE = 6
    ONEANDAHALF = 7
    TWOANDAHALF = 8
    THREANDAHALF = 9
    FOURANDAHALF = 10
    FIVEANDAHALF = 11
    AROMATIC = 12
    IONIC = 13
    HYDROGEN = 14
    THREECENTER = 15
    DATIVEONE = 16
    DATIVE = 17
    DATIVEL = 18
    DATIVER = 19
    OTHER = 20
    ZERO = 21
    POLYMERIC = 22
    PI = 23
    DELOCALIZED = 24


class Bond:
    def __init__(self, bond, bond_type=BondType.UNSPECIFIED):
        self.bond_idx = bond


def bonds_from_ccd_data(molsys, ccd_data):
    bonds, bond_types, bond_chiral = [], [], []
    for res in molsys._residues:
        if res.resname not in ccd_data:
            continue

        if "chem_comp_bond" in ccd_data[res.resname]:
            res_bond_data = ccd_data[res.resname]["chem_comp_bond"]
            res_bond_data_listed = {
                k: v if isinstance(v, (tuple, list)) else (v,)
                for k, v in res_bond_data.items()
            }
            mymap = {
                name: index
                for name, index in zip(np.atleast_1d(res.names), np.atleast_1d(res.ix))
            }

            for a1, a2, bond_type, aromatic, stereo in zip(
                res_bond_data_listed["atom_id_1"],
                res_bond_data_listed["atom_id_2"],
                res_bond_data_listed["value_order"],
                res_bond_data_listed["pdbx_aromatic_flag"],
                res_bond_data_listed["pdbx_stereo_config"],
            ):
                i1 = mymap.get(a1, None)
                i2 = mymap.get(a2, None)
                if i1 is None or i2 is None:
                    continue

                bonds.append([i1, i2])
                bond_types.append(chilife.io.CIF_BOND_TO_XL[bond_type.lower()])
                bond_chiral.append(stereo)

        if "chem_comp" in ccd_data[res.resname]:
            res_ccd_data = ccd_data[res.resname]["chem_comp"]
            linkage_key = (
                "type"
                if "type" in res_ccd_data
                else "link type"
                if "link_type" in res_ccd_data
                else None
            )
            if (pres := res.previous_residue()) is not None and linkage_key:
                link_type = res_ccd_data["type"].lower()
                if link_type in POLYMER_LINKAGE_TYPES:
                    a1, a2, btype, bchiral = POLYMER_LINKAGE_TYPES[link_type]
                    i1 = pres.ix[pres.names == a1].flat[0]
                    i2 = res.ix[res.names == a2].flat[0]
                    bonds.append([i1, i2])
                    bond_types.append(btype)
                    bond_chiral.append(bchiral)

    bonds = np.array(bonds)
    bond_types = np.array(bond_types)
    bond_chiral = np.array(bond_chiral)
    if len(bonds) == 0:
        bonds = None
        bond_types = None
        bond_chiral = None

    return bonds, bond_types, bond_chiral


def numpyify_ccd(ccd_data):
    numpyified_ccd_data = {}
    for resname, ccd_info in ccd_data.items():
        numpyified_ccd_data[resname] = {}
        if "chem_comp_bond" in ccd_info:
            res_bond_data = ccd_info["chem_comp_bond"]
            numpyified_ccd_data[resname]["bond"] = {
                "paired_names": np.array(
                    [
                        res_bond_data_listed["atom_id_1"],
                        res_bond_data_listed["atom_id_2"],
                    ]
                ),
                "bond_order": np.array(res_bond_data_listed["value_order"]),
                "bond_stereo": np.array(res_bond_data_listed["pdbx_stereo_config"]),
            }

        if "chem_comp_bond" in ccd_info:
            res_bond_data = ccd_info["chem_comp_bond"]
            numpyified_ccd_data[resname]["bond"] = {
                "paired_names": np.array(
                    [
                        res_bond_data_listed["atom_id_1"],
                        res_bond_data_listed["atom_id_2"],
                    ]
                ),
                "bond_order": np.array(res_bond_data_listed["value_order"]),
                "bond_stereo": np.array(res_bond_data_listed["pdbx_stereo_config"]),
            }


# Linkage data is atom name of the previous residue followed by atom name of the current residue followed by bond type
POLYMER_LINKAGE_TYPES = {
    "l-peptide linking": ["C", "N", BondType.SINGLE, "N"],
    "peptide linking": ["C", "N", BondType.SINGLE, "N"],
    "d-peptide linking": ["C", "N", BondType.SINGLE, "N"],
}
