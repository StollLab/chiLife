from typing import Union, Tuple, List, Set
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike
from scipy.spatial.distance import cdist
import igraph as ig

from .globals import atom_order, nataa_codes, natnu_codes, mm_backbones
from .Topology import get_min_topol, guess_bonds, modified_bfs_edges


def sort_pdb(pdbfile: Union[str, List[str], List[List[str]]],
             uniform_topology: bool = True,
             index: bool = False,
             bonds: Union[ArrayLike, Set] = set(),
             **kwargs) -> Union[List[str], List[List[str]], List[int]]:
    """Read ATOM lines of a pdb and sort the atoms according to chain, residue index, backbone atoms and side chain atoms.
    Side chain atoms are sorted by distance to each other/backbone atoms with atoms closest to the backbone coming
    first and atoms furthest from the backbone coming last. This sorting is essential to making internal-coordinates
    with consistent and preferred dihedral definitions.

    Parameters
    ----------
    pdbfile : str, List[str], List[List[str]]
        Name of the PDB file, a list of strings containing ATOM lines of a PDB file or a list of lists containing
        ATOM lines of a PDB file, where each sublist corresponds to a state/model of a multi-state pdb.
    uniform_topology: bool
        When given a multi-state pdb, assume that all states have the same topology (bonds) as the first state.
    index: bool :
         Return the sorted index rather than the sorted lines.
    bonds: ArrayLike :
         When sorting the PDB, use the provided bond list to as the topology rather than guessing the bonds.
    kwargs : dict
        Additional keyword arguments.

        aln_atoms : ArrayLike
            Only used with create_library. Array of 3 atom names indicating which atoms will be the alignment atoms for
            the library. These atoms will be used to determine which atoms constitute the backbone and side chain atoms.
    Returns
    -------
    lines : List[str], List[List[str]]
        Sorted list of strings corresponding to the ATOM entries of a PDB file.
    """
    if isinstance(pdbfile, (str, Path)):
        with open(pdbfile, "r") as f:
            lines = f.readlines()

        start_idxs = []
        end_idxs = []
        connect = []
        lines = [line for line in lines if line.startswith(('MODEL', 'ENDMDL', 'CONECT', 'ATOM', 'HETATM'))]
        for i, line in enumerate(lines):
            if line.startswith('MODEL'):
                start_idxs.append(i + 1)
            elif line.startswith("ENDMDL"):
                end_idxs.append(i)
            elif line.startswith('CONECT'):
                connect.append(line)

        # Use connect information for bonds if present
        if connect != [] and bonds == set():
            connect, _, _ = parse_connect(connect)
            kwargs['additional_bonds'] = kwargs.get('additional_bonds', set()) | connect

        # If it's a multi-state pdb...
        if start_idxs != []:

            if uniform_topology:
                # Assume that all states have the same topology as the first
                idxs = _sort_pdb_lines(lines[start_idxs[0]:end_idxs[0]], bonds=bonds, index=True, **kwargs)

            else:
                # Calculate the shared topology and force it
                atom_lines = [lines[s:e] for s, e in zip(start_idxs, end_idxs)]
                min_bonds_list = get_min_topol(atom_lines, forced_bonds=bonds)
                idxs = _sort_pdb_lines(lines[start_idxs[0]:end_idxs[0]], bonds=min_bonds_list, index=True, **kwargs)

            if isinstance(idxs, tuple):
                idxs, bonds = idxs

            lines[:] = [[lines[idx + start][:6] + f"{i + 1:5d}" + lines[idx + start][11:]
                         for i, idx in enumerate(idxs)]
                        for start in start_idxs]

            if kwargs.get('return_bonds', False):
                lines = lines, bonds
        else:
            lines = _sort_pdb_lines(lines, bonds=bonds, index=index, **kwargs)

    elif isinstance(pdbfile, list):
        lines = _sort_pdb_lines(pdbfile, bonds=bonds, index=index, **kwargs)

    return lines


def _sort_pdb_lines(lines, bonds=None, index=False, **kwargs) -> \
        Union[List[str], List[int], Tuple[list[str], List[Tuple[int]]]]:
    """
    Helper function to sort PDB ATOM and HETATM lines based off of the topology of the topology of the molecule.

    Parameters
    ----------
    lines : List[str]
        A list of the PDB ATOM and HETATM lines.
    bonds : Set[Tuple[int]]
        A Set of tuples of atom indices corresponding to atoms ( `lines` ) that are bound to each other. Passing this
        keyword argument will prevent chiLife from guessing the bonds. To guess most bonds but strictly enforce a few
        use he `additional_bonds` keyword argument.
    index : bool
        If True a list of atom indices will be returned
    **kwargs : dict
        Additional keyword arguments.
        return_bonds : bool
            Return bond indices as well, usually only used when letting the function guess the bonds.
        additional_bonds: set(tuple(int))
            Additional bonds (index pairs) that will be added to the bonds list. Used when letting the function guess
            the bonds, but
        aln_atoms : ArrayLike
            Only used with create_library. Array of 3 atom names indicating which atoms will be the alignment atoms for
            the library. These atoms will be used to determine which atoms constitute the backbone and side chain atoms.

    Returns
    -------
    lines : List[str | int]
        The sorted lines or indices corresponding to the sorted lines.
    bonds: Set[Tuple[int]]
        A set of tuples containing pars of indices corresponding to the atoms bound to in lines.
    """

    waters = [line for line in lines if line[17:20] in ('SOL', 'HOH')]
    water_idx = [idx for idx, line in enumerate(lines) if line[17:20] in ('SOL', 'HOH')]
    lines = [line for line in lines if line.startswith(("ATOM", "HETATM")) and line[17:20] not in ('SOL', 'HOH')]
    n_atoms = len(lines)
    index_key = {line[6:11]: i for i, line in enumerate(lines)}

    # Presort
    lines.sort(key=atom_sort_key)
    presort_idx_key = {line[6:11]: i for i, line in enumerate(lines)}
    presort_bond_key = {index_key[line[6:11]]: i for i, line in enumerate(lines)}

    coords = np.array([[float(line[30:38]), float(line[38:46]), float(line[46:54])] for line in lines])
    atypes = np.array([line[76:78].strip() for line in lines])
    anames = np.array([line[12:17].strip() for line in lines])

    if bonds:
        input_bonds = {tuple(b) for b in bonds}
        presort_bonds = set(tuple(sorted((presort_bond_key[b1], presort_bond_key[b2]))) for b1, b2 in bonds)
    else:
        bonds = guess_bonds(coords, atypes)
        presort_bonds = set(tuple(sorted((b1, b2))) for b1, b2 in bonds)
        if kwargs.get('additional_bonds', set()) != set():
            presort_bonds.union(kwargs['additional_bonds'])

    # get residue groups
    chain, resi, resn = lines[0][21], int(lines[0][22:26].strip()), lines[0][17:20].strip()
    start = 0
    resdict = {}
    for curr, pdb_line in enumerate(lines):

        if chain != pdb_line[21] or resi != int(pdb_line[22:26].strip()):
            resdict[chain, resi] = start, curr, resn
            start = curr
            chain, resi, resn = pdb_line[21], int(pdb_line[22:26].strip()), pdb_line[17:20].strip()

    resdict[chain, resi] = start, curr + 1, resn
    midsort_key = []
    for key in resdict:
        start, stop, resn = resdict[key]
        midsort_key += sort_residue(coords, atypes[start:stop], anames[start:stop],
                                    resn, presort_bonds, start, kwargs.get('aln_atoms', None))

    lines[:] = [lines[i] for i in midsort_key]
    lines.sort(key=atom_sort_key)

    if 'input_bonds' not in locals():
        input_bonds = presort_bonds
        idxmap = {presort_idx_key[line[6:11]]: i for i, line in enumerate(lines)}
    else:
        idxmap = {index_key[line[6:11]]: i for i, line in enumerate(lines)}

    # Return line indices if requested
    if index:
        str_lines = lines
        lines = [index_key[line[6:11]] for line in lines] + water_idx

    # Otherwise make new indices
    else:
        lines = [line[:6] + f"{i + 1:5d}" + line[11:] for i, line in enumerate(lines)] + waters

    if kwargs.get('return_bonds', False):
        bonds = {tuple(sorted((idxmap[a], idxmap[b]))) for a, b in input_bonds}
        return lines, bonds

    return lines


def sort_residue(coords, atypes, anames, resname, presort_bonds, start, aln_atoms=None):
    """
    Sort atoms of a residue.

    Parameters
    ----------
    coords : ArrayLike
        cartesian coordinates of each atom.
    atypes : ArrayLike
        Element symbols of each atom.
    anames : ArrayLike
        names of each atom
    resname : str
        Name of the residue
    presort_bonds : ArrayLike
        Array of index pairs corresponding to bonded atom pairs. Index is based on the indices of atoms before sorting.
    start : int
        The index of the first atom of the residue in the greater protein environment.
    aln_atoms : List
        List of atom indices defining the "origin" of the residue. These will be used to define backbone and side chain
        atoms.

    Returns
    -------
    sorted_args : List[int]
        Indices of the residue atoms based on the topology and the backbone/aln atoms.
    """

    stop = start + len(anames)
    n_heavy = np.sum(atypes != 'H')
    res_coords = coords[start:stop]

    # Adjust bond atom indices to reference local index
    bonds = np.array([(a - start, b - start) for a, b in presort_bonds
                      if (start <= a < stop) and (start <= b < stop)])

    # Get all nearest neighbors and sort by distance for generating graph
    distances = np.linalg.norm(res_coords[bonds[:, 0]] - res_coords[bonds[:, 1]], axis=1)
    distances = np.around(distances, decimals=3)

    # Sort bonds by distance
    idx_sort = np.lexsort((bonds[:, 0], bonds[:, 1], distances))
    pairs = bonds[idx_sort]

    # Create graph
    graph = ig.Graph(edges=pairs)

    # Determine if this residue fits aln_atoms criteria. aln_atoms must be defined, present, and adjacent in the graph
    if aln_atoms is not None:
        if len(aln_atoms) != 3:
            raise ValueError('aln_atoms must have length 3')

        aln_cen = np.argwhere(aln_atoms[1] == anames).flat[0]
        neighbor_idx = np.argwhere(np.isin(anames, aln_atoms[::2])).flatten()
        aln_present = np.all(np.isin(aln_atoms, anames))
        aln_adjacent = np.all(np.isin(neighbor_idx, graph.neighbors(aln_cen)))
        use_aln = aln_adjacent and aln_present
    else:
        neighbor_idx = []
        use_aln = False

    # let downstream functions decided backbone
    if use_aln:
        sorted_args = [np.argwhere(anames == a).flat[0] for a in aln_atoms]
        root_idx = np.argwhere(anames == aln_atoms[1]).flat[0]

    # Otherwise attempt to find canonical backbone definitions
    else:
        # guess possible backbone atoms for the residue type with the given atom names
        bb_candidates = get_bb_candidates(anames, resname)

        # Use candidates if they are defined
        if bb_candidates:
            aname_lst = anames.tolist()
            sorted_args = [aname_lst.index(name) for name in bb_candidates if name in aname_lst]

        # Otherwise, if there are no preceding residues start from the first atom
        elif start == 0:
            sorted_args = [0]

        # Otherwise use the atom bound to the nearest residue.
        else:
            for a, b in presort_bonds:
                # Find the atom bonded to a previous residue
                if a < start and start <= b < stop and atypes[b-start] != 'H':
                    sorted_args = [b - start]
                    break
            # Otherwise get the closest to any previous atom
            else:
                dist_mat = cdist(coords[:start], coords[start:stop])
                sorted_args = [np.squeeze(np.argwhere(dist_mat == dist_mat.min()))[0]]

        # If using a defined backbone, use the known root atoms for the BFS search below
        bb_names = anames[sorted_args]

        # TODO: Need a better way to guess the root atom from the backbone and side chains.
        for name in ["CA", "C1'"]:
            if name in bb_names:
                root_idx = np.argwhere(bb_names == name).flat[0]
                root_idx = sorted_args[root_idx]
                break
        # Unless there are no known root atoms, then just use the first atom
        else:
            root_idx = sorted_args[0]

    # Skip BFS sorting if heavy atoms are sorted, i.e. the backbone is known and there is no side chain (e.g. GLY)
    if len(sorted_args) != n_heavy:

        graph = ig.Graph(edges=pairs)

        if root_idx not in graph.vs.indices:
            raise RuntimeError(f'The connectivity of one or more residue is not valid')

        # Use dfs verts for backbone
        if use_aln:
            sorted_args += get_backbone_atoms(graph, root_idx, neighbor_idx, sorted_args=sorted_args)

        # Use bfs verts for
        # bfsv, si, bfse = graph.bfs(root_idx)
        # [v for v in bfsv if v not in sorted_args]
        sidechain_nodes = [v[1] for v in modified_bfs_edges(pairs, root_idx, sorted_args) if v[1] not in sorted_args]

        # Check for disconnected parts of residue
        if not graph.is_connected():
            for g in graph.connected_components():
                if np.any([arg in g for arg in sorted_args]):
                    continue
                free_nodes = [idx for idx in sidechain_nodes if atypes[start + idx] != 'H']
                g_nodes = [idx for idx in g if atypes[start + idx] != 'H']
                near_root = cdist(res_coords[free_nodes], res_coords[g_nodes]).argmin()

                yidx = near_root % len(g_nodes)
                subnodes, _, _ = graph.bfs(g_nodes[yidx])
                sidechain_nodes += list(subnodes)

    elif stop - start > n_heavy:
        # Assumes  non-heavy atoms come after the heavy atoms, which should be true because of the pre-sort
        sidechain_nodes = list(range(n_heavy, n_heavy + (stop - start - len(sorted_args))))
    else:
        sidechain_nodes = []

    sorted_args += sidechain_nodes

    # get any leftover atoms (eg HN)
    if len(sorted_args) != stop - start:
        for idx in range(stop - start):
            if idx not in sorted_args:
                sorted_args.append(idx)

    return [x + start for x in sorted_args]


def get_bb_candidates(anames, resname):
    """
    Function to guess the backbone atom names for a set of atom names residue name.
    Parameters
    ----------
    anames : ArrayLike
        Array of atom names of the residue.
    resname : str
        Residue name

    Returns
    -------
    bb_candidates : ArrayLike
        Array of backbone atom names.

    """
    if resname in natnu_codes:
        bb_candidates = mm_backbones['nu']
    elif resname in nataa_codes:
        bb_candidates = mm_backbones['aa']
    elif np.isin(mm_backbones['aa'], anames).sum() >= 3:
        bb_candidates = mm_backbones['aa']
    elif np.isin(mm_backbones['nu'], anames).sum() >= 3:
        bb_candidates = mm_backbones['nu']
    elif resname in mm_backbones:
        bb_candidates = mm_backbones[resname]
    else:
        bb_candidates = []

    return bb_candidates


def get_backbone_atoms(graph, root_idx, neighbor_idx, **kwargs):
    """
    Function to get backbone atom indices from a graph using a depth first search (dfs), given a root index and a list
    of neighboring indices that define  the starting branches of the backbone atoms (i.e. not sidechain atoms).

    Parameters
    ----------
    graph: igraph.Graph
        The graph to search for backbone atoms.
    root_idx: int
        The root index of the graph that separates the side chain and backbone atoms.
    neighbor_idx: indices of the neighbors of the `root_idx` that start the dfs of the backbone atoms.

    kwargs: dict
        Additional keyword arguments to pass to the depth first search.

        sorted_args: List[int]
            A sorted list of atom indices dictating backbone atoms that have already been defined

    Returns
    -------
    backbone_vs: List[int]
        list of sorted backbone atom indices.

    """

    sorted_args = kwargs.get('sorted_args', [])

    # Start stemming from root atom
    neighbors = graph.neighbors(root_idx)
    backbone_vs = []
    sidechain_vs = []
    bblist_order = {}
    dfsv, dfse = graph.dfs(root_idx)
    for idx in dfsv:
        if idx == root_idx:
            continue
        elif idx in neighbor_idx:
            bblist_order[idx] = len(backbone_vs)
            active_list = backbone_vs
        elif idx in neighbors:

            active_list = sidechain_vs

        if idx not in sorted_args:
            active_list.append(idx)

    bblist_order['end'] = len(backbone_vs)
    bbidx_slices = {}
    items = iter(bblist_order)
    k1 = next(items)
    for k2 in items:
        bbidx_slices[k1] = bblist_order[k1], bblist_order[k2]
        k1 = k2
    a, b = neighbor_idx
    return backbone_vs[slice(*bbidx_slices[b])] + backbone_vs[slice(*bbidx_slices[a])]


def atom_sort_key(pdb_line: str) -> Tuple[str, int, int]:
    """Assign a base rank to sort atoms of a pdb.

    Parameters
    ----------
    pdb_line : str
        ATOM line from a pdb file as a string.

    Returns
    -------
    tuple :
        chain_id, resid, name_order.
        ordered ranking of atom for sorting the pdb.
    """
    chain_id = pdb_line[21]
    res_name = pdb_line[17:20].strip()
    resid = int(pdb_line[22:26].strip())
    atom_name = pdb_line[12:17].strip()
    atom_type = pdb_line[76:79].strip()
    if res_name == "ACE":
        if atom_type != 'H' and atom_name not in ('CH3', 'C', 'O'):
            raise ValueError(f'"{atom_name}" is not canonical name of an ACE residue atom. \n'
                             f'Please rename to "CH3", "C", or "O"')
        name_order = (
            {"CH3": 0, "C": 1, "O": 2}.get(atom_name, 4) if atom_type != "H" else 5
        )

    else:
        name_order = atom_order.get(atom_name, 4) if atom_type != "H" else atom_order.get(atom_name, 7)

    return chain_id, resid, name_order


def parse_connect(connect: List[str]) -> Tuple[Set[Tuple[int]]]:
    """
    Parse PDB CONECT information to get a list covalent bonds, hydrogen bonds and ionic bonds.

    Parameters
    ----------
    connect : List[str]
        A list of strings that are the CONECT lines from a PDB file.

    Returns
    -------
    c_bonds : Set[Tuple[int]]
        Set of atom index pairs corresponding to atoms that are bound covalently.
    h_bonds : Set[Tuple[int]]
        Set of atom index pairs corresponding to atoms that are hydrogen bound.
    ci_bonds : Set[Tuple[int]]
        Set of atom index pairs corresponding to atoms that are bound ionically.
    """
    c_bonds, h_bonds, i_bonds = set(), set(), set()
    for line in connect:
        line = line.ljust(61)
        a0 = int(line[6:11])
        c_bonds |= {tuple(sorted((a0 - 1, int(b) - 1))) for b in line[11:31].split()}
        h_bonds |= {tuple(sorted((a0 - 1, int(b) - 1))) for b in (line[31:41].split() + line[46:56].split())}
        i_bonds |= {tuple(sorted((a0 - 1, int(b) - 1))) for b in (line[41:46], line[56:61]) if not b.isspace()}

    return c_bonds, h_bonds, i_bonds
