import numpy as np


def assign_atom_names(mol):
    """
    Arbitrarily assign atom names to a ligand. Specifically used to apply atom names to atoms read from an SDF file.

    Parameters
    ----------
    mol: Dict
        Dictionary of molecular information as read from an SDF file using :func:`~chilife.io.read_sdf`

    Returns
    -------
    names: List[str]
        List of atom names.
    """
    record = {}
    names = []
    for atom in mol["atoms"]:
        element = atom["element"]
        number = record.setdefault(element, 1)
        record[element] += 1
        names.append(f"{element}{number}")

    return names


def remap_sdf(sdf_data, mask):
    """
    Reassign atom indices of SDF data when some atoms have been removed. This is usually performed when removing
    hydrogen atoms.

    .. note:: This function does **not** remap atom property indices or indices of any field other than atoms and bonds.

    Parameters
    ----------
    sdf_data: Dict
        Dictionary of SDF data as created by :func:`~chilife.io.read_sdf`
    mask: ArrayLike
        Array of boolean values corresponding to the atoms of the sdf_data molecule that wil be kept (``True``) and
        removed (``False``).

    Returns
    -------
    new_data: Dict
        Dictionary of molecular information as created by :func:`~chilife.io.read_sdf` no longer containing the atoms
        corresponding to the ``False`` entries of ``mask``.
    """
    index_map = {a: i for i, a in enumerate(np.argwhere(mask).flatten())}
    new_data = {}
    for key, value in sdf_data.items():
        if key == "atoms":
            new_atoms = []
            for i, atom in enumerate(value):
                if i not in index_map:
                    continue
                new_atom = {k: v for k, v in atom.items()}
                new_atom["index"] = index_map[i]
                new_atoms.append(new_atom)
            new_data[key] = new_atoms

        elif key == "bonds":
            new_bonds = []
            for bond in value:
                if bond["idx1"] in index_map and bond["idx2"] in index_map:
                    new_bond = {k: v for k, v in bond.items()}
                    new_bond["idx1"] = index_map[bond["idx1"]]
                    new_bond["idx2"] = index_map[bond["idx2"]]
                    new_bonds.append(new_bond)
            new_data[key] = new_bonds
        else:
            new_data[key] = value

    # Update atom and bond counts
    new_data["n_atoms"] = len(new_data["atoms"])
    new_data["n_bonds"] = len(new_data["bonds"])

    # Add notice to user that other atom indices may be incorrect
    new_data["comment"] += " | See ATOM_REMAP_NOTICE below"
    new_data["ATOM REMAP NOTICE"] = (
        "NOTICE: The atom indices of this sdf file have been remapped. This means that atom index information "
        "outside of the bond and atom sections may be incorrect."
    )

    return new_data