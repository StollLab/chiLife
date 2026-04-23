from __future__ import annotations

import weakref
from collections.abc import Iterable
import numbers
import operator
from functools import partial, update_wrapper

import MDAnalysis

from .ligand_utils import assign_atom_names
from .Topology import Topology, BondType, bonds_from_ccd_data
import numpy as np
from numpy.typing import ArrayLike
from scipy.spatial import cKDTree

import chilife

# TODO:
#   Behavior: AtomSelections should have orders to be enforced when indexing.

masked_properties = (
    "record_types",
    "atomids",
    "names",
    "altlocs",
    "altLocs",
    "resnames",
    "resnums",
    "icodes",
    "resids",
    "resis",
    "chains",
    "occupancies",
    "bs",
    "segs",
    "segids",
    "atypes",
    "types",
    "elements",
    "charges",
    "chiral",
    "ix",
    "resixs",
    "resindices",
    "segixs",
    "segindices",
    "_Atoms",
    "_Residues",
    "atoms",
)

singles = {
    "record_type": "record_types",
    "name": "names",
    "altloc": "altlocs",
    "altLoc": "altlocs",
    "atype": "atypes",
    "element": "elements",
    "type": "atypes",
    "resn": "resnums",
    "resname": "resnames",
    "resnum": "resnums",
    "resid": "resids",
    "icode": "icodes",
    "resi": "resis",
    "chain": "chains",
    "segid": "segids",
    "charge": "charges",
    "bfactor": "bs",
    "qfactor": "occupancies",
    "occupancy": "occupancies",
    "segix": "segixs",
    "resix": "resixs",
}


class MolecularSystemBase:
    """Base class for molecular systems containing attributes universal to all subclasses"""

    def __getattr__(self, key):
        if not isinstance(self, MolSys) and "molsys" not in self.__dict__:
            return self.__getattribute__(key)
        elif key in singles:
            return np.squeeze(
                self.molsys.__getattribute__(singles[key])[self.mask]
            ).flat[0]
        elif isinstance(self, MolSys) and key not in self.__dict__:
            return self.__getattribute__(key)
        elif key == "trajectory":
            return self.molsys.trajectory
        elif key == "atoms":
            return self.molsys.__getattribute__(key)[self.mask]
        elif key in masked_properties or key in singles:
            return np.squeeze(self.molsys.__getattribute__(key)[self.mask])
        else:
            return self.molsys.__getattribute__(key)

    def __setattr__(self, key, value):
        if key in ("molsys", "mask"):
            super(MolecularSystemBase, self).__setattr__(key, value)
        elif key in singles:
            self.molsys.__getattribute__(singles[key])[self.mask] = value
        elif isinstance(self, MolSys) and key not in self.__dict__:
            super(MolecularSystemBase, self).__setattr__(key, value)
        elif key == "trajectory":
            self.molsys.__dict__["trajectory"] = value
        elif key in masked_properties or key in singles:
            self.molsys.__getattribute__(key)[self.mask] = value
        else:
            super(MolecularSystemBase, self).__setattr__(key, value)

    def __add__(self, other):
        if other.molsys is self.molsys:
            new_mask = np.unique(np.concatenate([self.mask, other.mask]))
            return AtomSelection(self.molsys, new_mask)

        else:
            return concat_molsys(self.atoms, other.atoms)

    def __sub__(self, other):
        if other.molsys is self.molsys:
            tmask = ~np.isin(self.mask, other.mask)
            new_mask = self.mask[tmask]
            return AtomSelection(self.molsys, new_mask)
        else:
            raise RuntimeError("You can not subtract atoms from a different molsys.")

    def select_atoms(self, selstr):
        """
        Select atoms from :class:`~MolSys` or :class:`~AtomSelection` based on selection a selection string, similar
        to the `select_atoms` method from `MDAnalysis <https://docs.mdanalysis.org/stable/documentation_pages/selections.html>`_

        Parameters
        ----------
        selstr : str
            Selection string

        Returns
        -------
        atoms : :class:`~AtomSelection`
            Selected atoms
        """

        mask = process_statement(selstr, self.logic_keywords, self.molsys_keywords)
        if hasattr(self, "mask"):
            t_mask = np.zeros(self.molsys.n_atoms, dtype=bool)
            t_mask[self.mask] = True
            mask *= t_mask

        mask = np.argwhere(mask).T[0]
        return AtomSelection(self.molsys, mask)

    def __getitem__(self, item):
        if np.issubdtype(type(item), np.integer):
            relidx = self.molsys.ix[self.mask][item]
            return Atom(self.molsys, relidx)

        if isinstance(item, slice):
            return AtomSelection(self.molsys, self.mask[item])

        elif isinstance(item, (np.ndarray, list, tuple)):
            item = np.asarray(item)
            if len(item) == 1 or item.sum() == 1:
                return Atom(self.molsys, self.mask[item])
            elif item.dtype == bool:
                item = np.argwhere(item).T[0]
                return AtomSelection(self.molsys, item)
            else:
                return AtomSelection(self.molsys, self.mask[item])

        elif hasattr(item, "__iter__"):
            if all([np.issubdtype(type(x), int) for x in item]):
                return AtomSelection(self.molsys, self.mask[item])

        raise TypeError(
            "Only integer, slice type, and boolean mask arguments are supported at this time"
        )

    @property
    def fname(self):
        """File name or functional name of the molecular system"""
        return self.molsys._fname

    @fname.setter
    def fname(self, val):
        self.molsys._fname = val

    @property
    def positions(self):
        """3-Dimensional cartesian coordinates of the atoms in the molecular system"""
        return self.coords

    @positions.setter
    def positions(self, val):
        self.coords = np.asarray(val)

    @property
    def coords(self):
        """3-dimensional cartesian coordinates of the atoms in the molecular system"""
        return self.trajectory.coords[self.trajectory.frame, self.mask]

    @coords.setter
    def coords(self, val):
        self.trajectory.coords[self.trajectory.frame, self.mask] = np.asarray(val)

    @property
    def n_atoms(self):
        """ "Number of atoms in the molecular system"""
        return len(self.mask)

    @property
    def n_residues(self):
        """Number of residues in the molecular system"""
        return len(np.unique(self.resixs[self.mask]))

    @property
    def n_chains(self):
        """Number of chains in the molecular system"""
        return len(np.unique(self.chains[self.mask]))

    @property
    def n_segments(self):
        """Number of segments in the molecular system"""
        return len(np.unique(self.segments[self.mask]))

    @property
    def residues(self):
        """:class:`~ResidueSelection` for all residues of the molecular system"""
        return ResidueSelection(self.molsys, self.mask)

    @property
    def segments(self):
        """:class:`~SegmentSelection` for all segments (chains) of the molecular system"""
        return SegmentSelection(self.molsys, self.mask)

    @property
    def logic_keywords(self):
        """Dictionary of logic keywords (strings) pertaining to :meth:`atom_selection` method. Keywords are mapped
        to the associated logic function"""
        return self.molsys._logic_keywords

    @property
    def molsys_keywords(self):
        """Dictionary of molecular system keywords (strings) pertaining to :meth:`atom_selection` method. Keywords are
        mapped to the associated molecular system attributes"""
        return self.molsys._molsys_keywords

    @property
    def universe(self):
        """Wrapper attribute to allow molecular systems to behave like MDAnalysis AtomGroup objects"""
        return self.molsys

    @property
    def bonds(self):
        """Array of atom indices specifying bonds between atoms. Note that the bonds are indexed with respect to the
        parent molecular system and are not specific to any atom, residue or chain selection"""
        bonds = self._bonds[self._bond_mask] if self._bonds is not None else None
        return bonds

    @bonds.setter
    def bonds(self, val):
        raise AttributeError(
            "Bonds cannot be set explicitly. Please use the `add_bond(s)` and `remove_bond(s)` methods."
        )

    @property
    def bond_atom_names(self):
        """Array of atom names of bonded atoms corresponding to those in :meth:`~MolecularSystemBase.bonds`"""
        return np.array([self.molsys.names[b] for b in self.bonds])

    @property
    def bond_types(self):
        """Array of chiLife BondType objects (enumerations) specifying the type of bond between atoms in
        :meth:`~MolecularSystemBase.bonds`"""
        if self.bonds is None or self._bond_mask is None:
            return None

        bond_types = self._bond_types[self._bond_mask]
        return bond_types

    @bond_types.setter
    def bond_types(self, val):
        raise AttributeError(
            "Bond types cannot be set explicitly. Please use the `add_bond(s)` and `remove_bond(s)` methods."
        )

    @property
    def bond_chiral(self):
        """chirality of the bonds between atoms specified in :meth:`~MolecularSystemBase.bonds`"""
        if self.bonds is None or self._bond_mask is None:
            return None
        bond_chiral = self.molsys._bond_chiral[self._bond_mask]
        return bond_chiral

    def _check_mol_sys_for_bonds(self):
        """
        Helper function to check if the current object has any bonds
        """
        if not isinstance(self, MolSys):
            raise AttributeError(
                "Bonds and bond types can only be set on `MolSys` classes and not atom/residue/chain selections"
            )

    def add_bond(self, bond, bond_type=None):
        """
        Add singular bond between two atoms.

        Parameters
        ----------
        bond : ArrayLike[int]
            size 2 array of integers specifying bonds between atom indices of the parent :class:`MolSys`.
        bond_type : ArrayLike[BondType] | None
            Array of chilife BondType instances. If not specified ``BondType.UNSPECIFIED`` will be used.
        """
        self._check_mol_sys_for_bonds()
        bond_type = bond_type or BondType.UNSPECIFIED

        if self._check_overwrite_bond(bond, bond_type):
            return None
        elif len(self._bonds) == 0:
            self._bonds = np.array([bond])
            self._bond_types = np.array([bond_type])
        else:
            self._bonds = np.concatenate((self._bonds, [bond]))
            self._bond_types = np.concatenate((self._bond_types, [bond_type]))

        self._bond_mask = np.arange(len(self._bonds))
        self.topology = Topology(self, self._bonds, self._bond_types)

    def _check_overwrite_bond(self, bond, bond_type):
        if self.bonds is None:
            return False

        if len(self.bonds) == 0:
            return False

        exists = np.argwhere(np.all(self._bonds == bond, axis=1)).flatten()
        exists_reversed = np.argwhere(
            np.all(self._bonds == bond[::-1], axis=1)
        ).flatten()
        if exists.size > 0:
            if bond_type is not None:
                self._bond_types[exists[0]] = bond_type

        elif exists_reversed.size > 0:
            if bond_type is not None:
                self._bond_types[exists_reversed[0]] = bond_type

        else:
            return False

        return True

    def add_bonds(self, bonds, bond_type=None, bond_chiral=None, update_topology=True):
        """
        Add multiple bonds between atoms.

        Parameters
        ----------
        bonds : ArrayLike[int]
            N by 2 array of integers specifying bonds between atom indices of the parent :class:`MolSys`.
        bond_type : ArrayLike[BondType] | None
            Array of chilife BondType instances. Must be the same length as ``bonds``. If not specified
            ``BondType.UNSPECIFIED`` will be used.
        bond_chiral : ArrayLike[str] | None
            chirality of the bonds.
        update_topology : bool
            Toggle updating the attached MolSys Topology.
        """

        self._check_mol_sys_for_bonds()
        bonds = np.atleast_1d(bonds)
        bond_type = (
            bond_type if bond_type is not None else [BondType.UNSPECIFIED] * len(bonds)
        )
        bond_chiral = bond_chiral if bond_chiral is not None else ["?"] * len(bonds)

        if not isinstance(bond_type, Iterable):
            bond_type = [bond_type] * len(bonds)
        if not isinstance(bond_chiral, Iterable):
            bond_chiral = [bond_chiral] * len(bonds)

        if len(bonds) != len(bond_type):
            raise AttributeError(
                "Failed to add bonds to the MolSys because the number of bonds and bond_types do not match."
            )
        if len(bonds) != len(bond_chiral):
            raise AttributeError(
                "Failed to add bonds to the MolSys because the number of bonds and bond_chiral do not match."
            )

        new_bonds, new_bond_types, new_bond_chiral = [], [], []
        for b, btype, chiral in zip(bonds, bond_type, bond_chiral):
            if not self._check_overwrite_bond(b, btype):
                new_bonds.append(b)
                new_bond_types.append(btype)
                new_bond_chiral.append(chiral)

        # Convert to numpy arrays
        new_bonds = np.array(new_bonds)
        new_bond_types = np.array(new_bond_types)
        new_bond_chiral = np.array(new_bond_chiral)

        if self.bonds is None or len(self.bonds) == 0:
            self._bonds = new_bonds
            self._bond_types = new_bond_types
            self._bond_chiral = new_bond_chiral
        else:
            self._bonds = np.concatenate((self._bonds, new_bonds))
            self._bond_types = np.concatenate((self._bond_types, new_bond_types))
            self._bond_chiral = np.concatenate((self._bond_chiral, new_bond_chiral))

        # Update persistent residue/segement objects
        if hasattr(self, "_residue_list"):
            for res in self._residues:
                res._update_bond_mask()
        if hasattr(self, "_segments_list"):
            for seg in self._segments:
                seg._update_bond_mask()

        self._bond_mask = np.arange(len(self._bonds))

        if update_topology:
            self.topology = Topology(self, self._bonds, self._bond_types)

    def remove_bond(self, bond):
        """
        Remove a singular bond between two atoms.

        Parameters
        ----------
        bond : ArrayLike[int]
            Array of two integers specifying atoms involved in the bond that is to be removed.
        """
        self.remove_bonds([bond])
        self._bond_mask = np.arange(len(self._bonds))

    def remove_bonds(self, bonds=None):
        """
        Remove multiple bonds between atoms.

        Parameters
        ----------
        bonds : ArrayLike[int]
            N by 2 array of integers specifying the parent :class:`MolSys` atom indices of the atoms that are to be
            un-bonded.
        """
        if bonds is None:
            self._bonds = np.array([[]])
            self._bond_types = np.array([])
            self._bond_mask = np.array([], dtype=bool)
            return None

        self._check_mol_sys_for_bonds()
        mask = np.ones(len(bonds), dtype=bool)
        for bond in bonds:
            exists = np.all(self._bonds == bond, axis=1)
            exists_reversed = np.all(self._bonds == bond[::-1], axis=1)

            if exists.size > 0:
                mask *= ~exists
            elif exists_reversed:
                mask *= ~exists_reversed

        self._bonds = self.bonds[mask]
        self._bond_types = self.bond_types[mask]
        self._bond_mask = np.arange(len(self._bonds))

    def _update_bond_mask(self, bond_mask=None):
        if bond_mask is not None:
            self._bond_mask = bond_mask
        elif self.molsys._bonds is None:
            self._bond_mask = None
        else:
            self._bond_mask = np.argwhere(
                np.isin(self.molsys._bonds, self.mask).all(axis=1)
            ).flatten()

    def copy(self):
        """Creates a deep copy of the underlying MolSys and returns an :class:`AtomSelection` from the new copy
        matching the current selection if it exists."""
        p2 = self.molsys.copy()
        return AtomSelection(p2, self.mask.copy())

    def __iter__(self):
        """Iterate over all atoms of the MolSys"""
        for idx in self.mask:
            yield self.molsys.atoms[idx]

    def __eq__(self, value):
        """Checks if two molecular systems are equal by making sure they have the same ``self.molsys`` and the same
        mask"""
        return self.molsys is value.molsys and self.mask == value.mask

    def write_pdb(self, filename):
        pass

    def write_cif(self, filename, use_bio_ccd: bool = True):
        """
        Write a CIF file of the current ``MolSys``. Note that if there are bonds or formal charges this information will
        be preserved in the _chem_comp_atom and _chem_comp_bond fields.

        Parameters
        ----------
        filename : str | Path
            File name or ``Path`` to write the cif file to.
        use_bio_ccd : bool
            Use the chilife ``bio_ccd`` to assign bond information to residues where this information is not already
            present. If the residue is not in the ``bio_ccd`` and does not have bond information in the ``MolSys``
            objectno CCD information will be written.
        """
        segids, stype = [], []
        poly_ids, strand_ids, poly_type = [], [], []
        poly_asym, poly_auth_num, poly_icodes = [], [], []
        poly_seq_eids, poly_seq_hetero, poly_seq_mon_id, poly_seq_num = [], [], [], []
        nonpoly_asym_id, nonpoly_auth_seq, nonpoly_eid = [], [], []
        nonpoly_mon_id, nonpoly_icode, nonpoly_seq_num, nonpoly_sid = [], [], [], []

        for i, seg in enumerate(self.segments):
            segids.append(str(i + 1))
            if len(seg.residues) > 1:
                stype.append("polymer")
                if any(x.resname in chilife.nataa_codes for x in seg.residues):
                    poly_type.append("polypeptide")
                elif any(x.resname in chilife.natnu_codes for x in seg.residues):
                    poly_type.append("nucleotide")
                else:
                    poly_type.append("unknown")

                poly_ids.append(str(i + 1))
                strand_ids.append(str(seg.segid))

                poly_seq_num.extend([str(x) for x in range(1, len(seg.residues) + 1)])
                poly_seq_mon_id.extend([res.resname for res in seg.residues])
                poly_seq_hetero.extend(
                    [
                        "y" if np.any(res.atoms.record_type == "HETATM") else "n"
                        for res in seg.residues
                    ]
                )
                poly_seq_eids.extend([poly_ids[-1]] * len(seg.residues))
                poly_asym.extend([str(res.segid) for res in seg.residues])
                poly_auth_num.extend([str(res.resnum) for res in seg.residues])
                poly_icodes.extend(
                    [res.icode if res.icode.strip() else "." for res in seg.residues]
                )

            else:
                res = seg.residues[0]
                nonpoly_eid.append(str(i + 1))
                nonpoly_asym_id.append(str(seg.segid))
                nonpoly_auth_seq.append(str(res.resnum))
                nonpoly_mon_id.append(res.resname)
                nonpoly_icode.append(res.icode if res.icode != "" else ".")
                nonpoly_seq_num.append(str(1))
                nonpoly_sid.append(str(seg.segid))
                stype.append("non-polymer")

        entites = self.segindices
        atom_site_data = struct_to_cif_atom_site(self)

        local_ccd = {}
        for res in self._residues:
            if res.resname in local_ccd:
                continue
            elif res.resname in chilife.bio_ccd and use_bio_ccd:
                local_ccd[res.resname] = chilife.bio_ccd[res.resname]
            else:
                local_ccd[res.resname] = res_to_ccd(res)

        ccd_list = sorted(local_ccd.keys())
        local_ccd = chilife.io.join_ccd_info(ccd_list, local_ccd)

        cif_data = {
            self.name: {
                "entry": {"id": self.name},
                "atom_type": {
                    "symbol": [str(e) for e in sorted(np.unique(self.elements))]
                },
                "audit_author": {
                    "name": ["chiLife", "Maxx Tessmer", "Stefan Stoll"],
                    "pdbx_ordinal": ["1", "2", "3"],
                },
                "citation": {
                    "book_publisher": "?",
                    "country": "US",
                    "id": "primary",
                    "journal_full": "PLoS Computational Biology",
                    "journal_id_ISSN": "1553-734X",
                    "journal_volume": "19",
                    "journal_issue": "3",
                    "journal_page_first": "e1010834",
                    "journal_page_last": "e1010834",
                    "title": "chiLife: An open-source Python package for in silico spin labeling and integrative protein modeling",
                    "year": "2023",
                },
                "citation_author": {
                    "citation_id": ["primary", "primary"],
                    "name": ["Maxx Tessmer", "Stefan Stoll"],
                    "ordinal": ["1", "2"],
                },
                "entity": {
                    "id": segids,
                    "pdbx_description": ["."] * len(segids),
                    "type": stype,
                },
                "entity_poly": {
                    "entity_id": poly_ids,
                    "pdbx_strand_id": strand_ids,
                    "pdbx_type": poly_type,
                },
                "entity_poly_seq": {
                    "entity_id": poly_seq_eids,
                    "hetero": poly_seq_hetero,
                    "mon_id": poly_seq_mon_id,
                    "num": poly_seq_num,
                },
                "ma_data": {
                    "content_type": "model_coordinates",
                    "id": 1,
                    "name": "Model",
                },
                "pdbx_nonpoly_scheme": {
                    "asym_id": nonpoly_asym_id,
                    "auth_seq_num": nonpoly_auth_seq,
                    "entity_id": nonpoly_eid,
                    "mon_id": nonpoly_mon_id,
                    "pdb_ins_code": nonpoly_icode,
                    "pdb_seq_num": nonpoly_seq_num,
                    "pdb_strand_id": nonpoly_sid,
                },
                "pdbx_poly_seq_scheme": {
                    "asym_id": poly_asym,
                    "auth_seq_num": poly_auth_num,
                    "entity_id": poly_seq_eids,
                    "hetero": poly_seq_hetero,
                    "mon_id": poly_seq_mon_id,
                    "pdb_ins_code": poly_icodes,
                    "pdb_seq_num": poly_seq_num,
                    "pdb_strand_id": poly_asym,
                    "seq_id": poly_seq_num,
                },
                "software": {
                    "classification": "model building",
                    "date": "2023-03-31",
                    "description": "Protein modeling",
                    "name": "chiLife",
                    "pdbx_ordinal": "1",
                    "type": "python package",
                    "version": chilife.__version__,
                },
                "struct_asym": {
                    "entity_id": poly_ids,
                    "id": strand_ids,
                },
                "atom_site": atom_site_data,
                **local_ccd,
            }
        }

        # Delete empty entries
        del_list = []
        for k1, v1 in cif_data[self.name].items():
            bools = []
            for k2, v2 in v1.items():
                bools.append(bool(v2))

            if np.all(~np.array(bools)):
                del_list.append(k1)

        for key in del_list:
            del cif_data[self.name][key]

        chilife.io.write_cif(filename, cif_data)

    def _make_sdf_data(self):
        sdf_data = []

        properties = {}

        if len(chg_mask := np.argwhere(self.charges != 0).flatten().tolist()) > 0:
            properties["CHG"] = {
                "atom_ids": chg_mask,
                "charges": self.charges[chg_mask].tolist(),
            }

        for ts in self.trajectory:
            sdf_dict = {
                "name": self.name,
                "software": f"chiLife {chilife.__version__}",
                "comment": "",
                "n_atoms": self.n_atoms,
                "n_bonds": len(self.bonds),
                "chiral": int(np.any(~np.isin(self.chiral, ("?", "N")))),
                "format": "v2000",
                "atoms": [
                    {
                        "index": i,
                        "xyz": atom.position.tolist(),
                        "element": atom.element,
                        "mass difference": 0,
                        "charge": 0,
                        "stereo": 0,
                        "valence": 0,
                    }
                    for i, atom in enumerate(self.atoms)
                ],
                "bonds": [
                    {
                        "idx1": a1,
                        "idx2": a2,
                        "type": chilife.io.XL_BOND_TO_SDF[btype],
                        "stereo": 0,
                    }
                    for (a1, a2), btype in zip(self.bonds, self.bond_types)
                ],
                "properties": properties,
            }

            sdf_data.append(sdf_dict)

        return sdf_data

    def write_sdf(self, filename):
        """
        Write an SDF file from the current ``MolSys`` object. Note there is no support for sdf style chirality,
        isotopes, valence, RAD, ISO, RBC, SUB, UNS,

        Parameters
        ----------
        filename : str | Path
            Name of the file
        """
        sdf_data = self._make_sdf_data()
        chilife.io.write_sdf(sdf_data, filename)


class MolSys(MolecularSystemBase):
    """
    An object containing all attributes of a molecular system.

    Parameters
    ----------
    atomids : np.ndarray
        Array of atom indices where each index is unique to each atom in the molecular system.
    names : np.ndarray
        Array of atom names, e.g. 'N', 'CA' 'C'.
    altlocs : np.ndarray
        Array of alternative location identifiers for each atom.
    resnames : np.ndarray
        Array of residue names for each atom.
    resnums : np.ndarray
        Array of residue numbers for each atom.
    chains : np.ndarray
        Array of chain identifiers for each atom.
    trajectory : np.ndarray
        Cartesian coordinates of each atom for each state of an ensemble or for each timestep of a trajectory.
    occupancies : np.ndarray
        Occupancy or q-factory for each atom.
    bs : np.ndarray
        thermal, or b-factors for each atom.
    segs : np.ndarray
        Segment/chain IDs for each atom.
    atypes : np.ndarray
        Atom/element types.
    charges : np.ndarray
        Formal or partial charges assigned to each atom.
    chiral : nd.array | None
        Chirality specifier for each atom.
    resindices: np.ndarray | None
        Array of the residue indices each atom belongs to. resindices are 0-indexed.
    segindices: np.ndarray | None
        Array of the segment index each atom belongs to. segindices are 0-indexed.
    bonds : ArrayLike | None
        Array of atom ID pairs corresponding to all atom bonds in the system.
    bond_types : ArrayLike | None
        Array of BondType objects corresponding to all bonds in the system.
    bond_chiral: ArrayLike | None
        Array specifying the chirality (if any) of each bond.
    name : str
        Name of molecular system
    use_ccd: dict | None
        A dictionary containing ccd information needed to generate topological information. This can still be used if
        bonds are provided but some are missing.
    """

    def __init__(
        self,
        record_types: np.ndarray,
        atomids: np.ndarray,
        names: np.ndarray,
        altlocs: np.ndarray,
        resnames: np.ndarray,
        resnums: np.ndarray,
        icodes: np.ndarray,
        chains: np.ndarray,
        trajectory: np.ndarray,
        occupancies: np.ndarray,
        bs: np.ndarray,
        segs: np.ndarray,
        atypes: np.ndarray,
        charges: np.ndarray,
        chiral: np.ndarray | None = None,
        resindices: np.ndarray | None = None,
        segindices: np.ndarray | None = None,
        bonds: np.ndarray | None = None,
        bond_types: np.ndarray | None = None,
        bond_chiral: np.ndarray | None = None,
        name: str = "Noname_MolSys",
        use_ccd: dict | None = None,
    ):
        # self.molsys = weakref.proxy(self)
        self.record_types = record_types.copy()
        self.atomids = atomids.copy()
        self.names = names.copy().astype("U4")
        self.altlocs = altlocs.copy()
        self.resnames = resnames.copy()
        self.resnums = resnums.copy()
        self.icodes = icodes.copy()
        self.chains = chains.copy()
        self.trajectory = Trajectory(trajectory.copy(), self)
        self.occupancies = occupancies.copy()
        self.bs = bs.copy()
        self.segs = segs.copy() if np.any(segs != "") else chains.copy()
        self.atypes = atypes.copy()
        self.charges = charges.copy()
        self.chiral = (
            chiral.copy() if chiral is not None else np.array(["?"] * len(atomids))
        )
        self._fname = name

        self.ix = np.arange(len(self.atomids))
        self.mask = np.arange(len(self.atomids))

        # resix_borders = np.nonzero(np.r_[1, np.diff(self.resnums)[:-1]])
        # resix_borders = np.append(resix_borders, [self.n_atoms])
        # resixs = []
        # for i in range(len(resix_borders) - 1):
        #     dif = resix_borders[i + 1] - resix_borders[i]
        #     resixs.append(np.ones(dif, dtype=int) * i)

        if resindices is None:
            residues, resixs = [], []
            i = -1
            for aidx in range(self.n_atoms):
                residue_id = self.resnums[aidx], self.icodes[aidx], self.chains[aidx]
                if residue_id not in residues:
                    residues.append(residue_id)
                    i += 1

                resixs.append(i)

            self.resixs = np.array(resixs)
        else:
            self.resixs = resindices

        self.resindices = self.resixs

        if segindices is None:
            segix_map = {v: i for i, v in enumerate(np.unique(self.segs))}
            self.segixs = np.array([segix_map[k] for k in self.segs])
        else:
            self.segixs = segindices
        self.segindices = self.segixs

        self.is_protein = np.array(
            [res in chilife.SUPPORTED_RESIDUES for res in resnames]
        )

        self._molsys_keywords = {
            "id": self.atomids,
            "record_type": self.record_types,
            "name": self.names,
            "altloc": self.altlocs,
            "resname": self.resnames,
            "resnum": self.resnums,
            "resix": self.resixs,
            "resindex": self.resixs,
            "icode": self.icodes,
            "chain": self.chains,
            "occupancies": self.occupancies,
            "b": self.bs,
            "chiral": self.chiral,
            "segid": self.segs,
            "segix": self.segixs,
            "segindex": self.segixs,
            "type": self.atypes,
            "charges": self.charges,
            "resi": self.resnums,
            "resid": self.resnums,
            "i.": self.resnums,
            "s.": self.segs,
            "c.": self.chains,
            "r.": self.resnames,
            "n.": self.names,
            "resn": self.resnames,
            "q": self.occupancies,
            "elem": self.atypes,
            "element": self.atypes,
            "protein": self.is_protein,
            "_len": self.n_atoms,
        }

        self._logic_keywords = {
            "and": operator.mul,
            "or": operator.add,
            "not": unot,
            "!": unot,
            "<": operator.lt,
            ">": operator.gt,
            "==": operator.eq,
            "<=": operator.le,
            ">=": operator.ge,
            "!=": operator.ne,
            "byres": partial(byres, molsys=self.molsys),
            "within": update_wrapper(partial(within, molsys=self.molsys), within),
            "around": update_wrapper(partial(within, molsys=self.molsys), within),
            "point": update_wrapper(partial(point, molsys=self.molsys), point),
        }

        # Aliases
        self.resns = self.resnames
        self.resis = self.resnums
        self.altLocs = self.altlocs
        self.resids = self.resnums
        self.segids = self.segs
        self.types = self.atypes
        self.elements = self.atypes
        self.bfactors = self.bs

        self._bonds = None
        self._bond_types = None
        self._bond_mask = np.array([], dtype=bool)
        self._bond_chiral = None

        add_ccd_bonds = False
        if use_ccd is not None:
            if use_ccd is True:
                use_ccd = chilife.bio_ccd

            unique_resnames = np.unique(resnames)
            unique_resnames = [
                res
                for res in unique_resnames
                if (res in use_ccd) and ("chem_comp_atom" in use_ccd[res])
            ]
            atom_chiral_mapping = {
                (res, aid): c
                for res in unique_resnames
                for aid, c in zip(
                    use_ccd[res]["chem_comp_atom"]["atom_id"],
                    use_ccd[res]["chem_comp_atom"]["pdbx_stereo_config"],
                )
            }

            chiral = np.array(
                [
                    atom_chiral_mapping.get((res, atom_id), "?")
                    for res, atom_id in zip(resnames, names)
                ]
            )

            chiral_mask = self.chiral == "?"
            self.chiral[chiral_mask] = chiral[chiral_mask]
            ccd_bonds, ccd_bond_types, ccd_bond_chiral = bonds_from_ccd_data(
                self, use_ccd
            )

            if bonds is None:
                bonds = ccd_bonds
                bond_types = ccd_bond_types
                bond_chiral = ccd_bond_chiral
            else:
                add_ccd_bonds = True

        # Add Topology
        if bonds is not None:
            self._bonds = bonds.copy()
            self._bond_types = (
                bond_types.copy()
                if bond_types is not None
                else np.array([BondType.UNSPECIFIED] * len(bonds))
            )
            self._bond_mask = np.arange(len(self._bonds))
            if bond_chiral is not None and any(bond_chiral):
                self._bond_chiral = bond_chiral.copy()
            else:
                self._bond_chiral = np.array(["?"] * len(bonds))

            if hasattr(self, "_residue_list"):
                for res in self._residues:
                    res._update_bond_mask()
            if hasattr(self, "_segments_list"):
                for seg in self._segments:
                    seg._update_bond_mask()

        if add_ccd_bonds:
            self.add_bonds(
                ccd_bonds, ccd_bond_types, ccd_bond_chiral, update_topology=False
            )

        # Atoms creation is required to be last
        self.atoms = AtomSelection(self, self.mask, self._bond_mask)
        self.topology = (
            Topology(self, self._bonds, self._bond_types) if bonds is not None else None
        )

    @classmethod
    def from_pdb(cls, file_name, sort_atoms=False, **kwargs):
        """
        Reads a pdb file and returns a MolSys object

        Parameters
        ----------
        file_name : str, Path
            Path to the PDB file to read.
        sort_atoms : bool
            Presort atoms for consisted internal coordinate definitions
        kwargs: dict
            Additional arguments to be passed to the :class:`MolSys` constructor.
        Returns
        -------
        cls : MolSys
            Molecular system object of the PDB structure.
        """

        pdb_dict = chilife.read_pdb(file_name, sort_atoms=sort_atoms)

        # Add chain IDs if not present.
        chain_data = pdb_dict["chains"]
        used_chains = np.unique(chain_data)
        if "" in used_chains:
            available_chains = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            for new_chain in available_chains:
                if new_chain not in used_chains:
                    break
            chain_data[chain_data == ""] = new_chain

        return cls(**pdb_dict, **kwargs)

    @classmethod
    def from_cif(cls, file_name):
        """
        Reads a cif file and returns a MolSys object

        Parameters
        ----------
        file_name : str, Path
            Path to the CIF file to read.
        Returns
        -------
        cls : MolSys
            Molecular system object of the CIF structure.
        """
        cif_data = chilife.io.read_cif(file_name)

        if len(cif_data) > 1:
            raise RuntimeError(
                "More than one data entry found in the cif file. Please seperate into multiple files or join as "
                "separate chains in one entry"
            )

        name = list(cif_data.keys())[0]
        cif_data = cif_data[name]
        atom_site_data = cif_data["atom_site"]

        model_id = np.array([atom_site_data["pdbx_PDB_model_num"]]).flatten()
        model_ids = np.unique(model_id)
        n_models = len(model_ids)
        n_atoms = np.sum(model_id == model_ids[0])
        anames = np.array(atom_site_data["label_atom_id"][:n_atoms])

        for i in model_ids:
            if np.sum(model_id == i) != n_atoms:
                raise RuntimeError(
                    "This CIF file contains multiple structure models with different numbers of atoms. chiLife "
                    "currently only supports reading CIF files where all models have the exact same number of atoms."
                )
            model_anames = np.array(atom_site_data["label_atom_id"])[model_id == i]
            if not np.all(model_anames == anames):
                raise RuntimeError(
                    "This CIF file contains multiple structure models with different atom orders. chiLife currently "
                    "only supports reading CIF files where all models have the exact same atom orders."
                )
        record_types = np.array(atom_site_data["group_PDB"][:n_atoms])
        atomids = np.array(atom_site_data["id"][:n_atoms])
        altlocs = np.array(
            ["" if a == "." else a for a in atom_site_data["label_alt_id"][:n_atoms]]
        )

        atypes = np.array(atom_site_data["type_symbol"][:n_atoms])
        resnames = np.array(atom_site_data["label_comp_id"][:n_atoms])
        resnums = np.array(atom_site_data["auth_seq_id"][:n_atoms], dtype=int)
        icodes = np.array(atom_site_data["pdbx_PDB_ins_code"][:n_atoms])
        icodes[icodes == "?"] = ""

        segs = np.char.array(atom_site_data["label_asym_id"][:n_atoms]) + np.char.array(
            atom_site_data["label_entity_id"][:n_atoms]
        )

        occupancies = np.array(atom_site_data["occupancy"][:n_atoms], dtype=float)
        bs = np.array(atom_site_data["B_iso_or_equiv"][:n_atoms], dtype=float)
        chains = np.array(atom_site_data["auth_asym_id"][:n_atoms])
        charges = np.array(atom_site_data["pdbx_formal_charge"][:n_atoms], dtype="U3")
        charges[charges == "?"] = "nan"
        charges = charges.astype(float)

        traj = np.empty((n_models, n_atoms, 3))
        for i, id in enumerate(model_ids):
            sl = slice(i * n_atoms, (i + 1) * n_atoms)
            xs = np.array(atom_site_data["Cartn_x"][sl], dtype=float)
            ys = np.array(atom_site_data["Cartn_y"][sl], dtype=float)
            zs = np.array(atom_site_data["Cartn_z"][sl], dtype=float)
            traj[i] = np.dstack([xs, ys, zs])

        # Get ccd
        ccd_data = chilife.io.create_ccd_dicts(cif_data)

        msys = cls(
            record_types,
            atomids,
            anames,
            altlocs,
            resnames,
            resnums,
            icodes,
            chains,
            traj,
            occupancies,
            bs,
            segs,
            atypes,
            charges,
            name=name,
            use_ccd=ccd_data,
        )
        msys._cif_data = cif_data

        return msys

    @classmethod
    def from_sdf(cls, file_name):
        """
        Reads a sdf file and returns a MolSys object

        Parameters
        ----------
        file_name : str, Path
            Path to the sdf file to read.
        Returns
        -------
        cls : MolSys
            Molecular system object of the sdf structure.
        """
        sdf_data = chilife.io.read_sdf(file_name)
        mol0 = sdf_data[0]
        name = mol0["name"] or "LIG"
        atom_elements_0 = np.array([a["element"] for a in mol0["atoms"]])
        n_atoms = len(atom_elements_0)
        res_name = mol0["name"][:3] if mol0["name"] != "" else "LIG"

        record_types = np.array(["HETATM"] * n_atoms)
        atomids = np.array([a["index"] for a in mol0["atoms"]])
        anames = np.array(assign_atom_names(mol0))
        altlocs = np.array([""] * n_atoms)
        resnames = np.array([res_name] * n_atoms)
        resnums = np.ones(n_atoms, dtype=int)
        icodes = np.array([""] * n_atoms)
        chains = np.array(["A"] * n_atoms)
        occupancies = np.ones(n_atoms)
        bs = np.zeros(n_atoms)
        segs = np.array(["A"] * n_atoms)
        atypes = atom_elements_0

        # CHG block takes precidence over ATOM block charges
        if "CHG" in mol0["properties"]:
            charges = np.zeros(n_atoms)
            chg_idxs = mol0["properties"]["CHG"]["atom_ids"]
            chg_values = mol0["properties"]["CHG"]["charges"]
            charges[chg_idxs] = chg_values
        else:
            charges = np.array(
                [
                    chilife.io.SDF_ATOM_CHARGE_TO_INT[a["charge"]]
                    for a in sdf_data[0]["atoms"]
                ]
            )

        bonds, bond_types = [], []
        for bond in mol0["bonds"]:
            bonds.append([bond["idx1"], bond["idx2"]])
            bond_types.append(chilife.io.SDF_BOND_TO_XL[bond["type"]])
        bonds, bond_types = np.array(bonds), np.array(bond_types)

        trajectory = []
        for struct_dict in sdf_data:
            atom_elements_i = np.array([a["element"] for a in struct_dict["atoms"]])
            if not np.all(atom_elements_i == atom_elements_0):
                raise RuntimeError(
                    "This structures in this sdf file contain different atoms, suggesting they are not all the same "
                    "molecule. chiLife currently only supports reading sdf files as MolSys objects if all molecules in "
                    "the sdf are the same with only atom positions differing."
                )

            coords = [a["xyz"] for a in struct_dict["atoms"]]
            trajectory.append(coords)
        trajectory = np.array(trajectory)

        msys = cls(
            record_types,
            atomids,
            anames,
            altlocs,
            resnames,
            resnums,
            icodes,
            chains,
            trajectory,
            occupancies,
            bs,
            segs,
            atypes,
            charges,
            bonds=bonds,
            bond_types=bond_types,
            name=name,
        )

        msys._sdf_data = sdf_data

        return msys

    @classmethod
    def from_arrays(
        cls,
        anames: ArrayLike,
        atypes: ArrayLike,
        resnames: ArrayLike,
        resindices: ArrayLike,
        resnums: ArrayLike,
        segindices: ArrayLike,
        record_types: ArrayLike = None,
        altlocs: ArrayLike = None,
        icodes: ArrayLike = None,
        segids: ArrayLike = None,
        trajectory: ArrayLike = None,
        **kwargs,
    ) -> MolSys:
        """
        Create a MolSys object from a minimal set of arrays.

        Parameters
        ----------
        anames : ArrayLike
            Array of atom names.
        atypes : ArrayLike
            Array of atom types/elements.
        resnames : ArrayLike
            Array of residue names for each atom.
        resindices : ArrayLike
            Array of residue indices for each atom.
        resnums : ArrayLike
            Array of residue numbers for each atom.
        segindices : ArrayLike
            Array of segment indices for each atom.
        record_types : ArrayLike | None
            Array of record types for each atom, e.g. ATOM or HETATM
        altlocs: ArrayLike | None
            Array of alternate location specifies for each atom.
        icodes: ArrayLike | None
            Array of insertion codes for each atom.
        segids : ArrayLike
            Array of segment IDs for each atom.
        trajectory : ArrayLike
            Array of cartesian coordinates for each atom and each state of an ensemble or frame of a trajectory.
        kwargs : dict
            Dictionary of additional arguments to pass the the :class:`MolSys` constructor.
        Returns
        -------
        cls : MolSys
            Molecular system object of the PDB structure.
        """
        anames = np.asarray(anames)
        atypes = np.asarray(atypes)
        resindices = np.asarray(resindices)
        segindices = np.asarray(segindices)

        n_atoms = len(anames)
        atomids = np.arange(n_atoms)
        altlocs = np.array([""] * n_atoms) if altlocs is None else np.array(altlocs)
        icodes = np.array([""] * n_atoms) if icodes is None else np.array(icodes)

        if len(resnums) != n_atoms:
            resnums = np.array([resnums[residx] for residx in resindices])
        else:
            resnums = np.asarray(resnums)

        if len(resnames) != n_atoms:
            resnames = np.array([resnames[residx] for residx in resindices])
        else:
            resnames = np.asarray(resnames)

        if len(icodes) != n_atoms:
            icodes = np.array([icodes[residx] for residx in resindices])
        else:
            icodes = np.asarray(icodes)

        if len(segindices) != n_atoms:
            segindices = np.array([segindices[x] for x in resindices])

        if segids is None:
            segids = np.array(["ABCDEFGHIJKLMNOPQRSTUVWXYZ"[i] for i in segindices])
        elif len(segids) != n_atoms:
            segids = np.array([segids[i] for i in segindices])
        else:
            segids = np.asarray(segids)

        chains = segids

        if trajectory is None:
            trajectory = np.empty((1, n_atoms, 3))
        elif len(trajectory.shape) == 2:
            trajectory = trajectory[None, ...]

        if record_types is None:
            record_types = np.array(
                [
                    "ATOM" if rnme in chilife.SUPPORTED_RESIDUES else "HETATM"
                    for rnme in resnames
                ]
            )

        occupancies = np.ones(n_atoms)
        bs = np.ones(n_atoms)
        charges = kwargs.pop("charges", np.zeros(n_atoms))

        return cls(
            record_types,
            atomids,
            anames,
            altlocs,
            resnames,
            resnums,
            icodes,
            chains,
            trajectory,
            occupancies,
            bs,
            segids,
            atypes,
            charges,
            **kwargs,
        )

    @classmethod
    def from_atomsel(cls, atomsel, frames=None):
        """
        Create a new MolSys object from an existing :class:`~AtomSelection` or
        `MDAnalysis.AtomGroup <https://userguide.mdanalysis.org/stable/atomgroup.html>`_ object.

        Parameters
        ----------
        atomsel : :class:`~AtomSelection`, MDAnalysis.AtomGroup
            The :class:`~AtomSelection` or MDAnalysis.AtomGroup from which to create the new MolSys object.
        frames : int, Slice, ArrayLike
            Frame index, slice or array of frame indices corresponding to the frames of the atom selection you wish to
            extract into a new :class:`~MolSys` object.

        Returns
        -------
        cls : MolSys
            Molecular system object of the PDB structure.
        """

        if isinstance(atomsel, (MDAnalysis.AtomGroup, AtomSelection)):
            U = atomsel.universe
        elif isinstance(atomsel, MDAnalysis.Universe):
            U = atomsel
            atomsel = U.atoms

        if frames is None:
            frames = slice(0, len(U.trajectory))

        record_types = (
            atomsel.record_types if hasattr(atomsel, "record_types") else None
        )
        icodes = atomsel.icodes if hasattr(atomsel, "icodes") else None
        anames = atomsel.names
        atypes = atomsel.types
        resnames = atomsel.resnames
        resnums = atomsel.resnums
        ridx_map = {num: i for i, num in enumerate(np.unique(atomsel.resindices))}
        resindices = np.array([ridx_map[num] for num in atomsel.resindices])
        segids = atomsel.segids
        sidx_map = {num: i for i, num in enumerate(np.unique(atomsel.segindices))}
        segindices = np.array([sidx_map[num] for num in atomsel.segindices])

        if (
            isinstance(atomsel, MDAnalysis.AtomGroup)
            and hasattr(atomsel, "bonds")
            and len(atomsel.bonds) > 0
        ):
            bonds = atomsel.bonds.indices
            bond_types = np.array([BondType.UNSPECIFIED] * len(bonds))
        elif isinstance(atomsel, MolecularSystemBase) and atomsel.bonds is not None:
            bonds = atomsel.bonds
            bond_types = atomsel.bond_types
        else:
            bonds = None
            bond_types = None

        # remap bond indices
        if bonds is not None:
            k = {idx: i for i, idx in enumerate(atomsel.ix)}
            bonds = np.array(
                [[k[a1], k[a2]] for a1, a2 in bonds if a1 in k and a2 in k], dtype=int
            )

        if hasattr(U.trajectory, "coordinate_array"):
            trajectory = U.trajectory.coordinate_array[frames][:, atomsel.ix, :]
        else:
            trajectory = []
            for ts in U.trajectory[frames]:
                trajectory.append(atomsel.positions)
            trajectory = np.array(trajectory)

        return cls.from_arrays(
            anames,
            atypes,
            resnames,
            resindices,
            resnums,
            segindices,
            segids=segids,
            trajectory=trajectory,
            record_types=record_types,
            icodes=icodes,
            bonds=bonds,
            bond_types=bond_types,
        )

    @classmethod
    def from_rdkit(cls, mol):
        """
        Create a MolSys from an rdkit Mol object with embedded conformers.

        Parameters
        ----------
        mol : rdkit.Chem.rdchem.Mol
            An rdkit Mol object.

        Returns
        -------
        cls: MolSys
            Molecular system object of the PDB structure.
        """
        atypes = np.array([a.GetSymbol() for a in mol.GetAtoms()])
        anames = np.array([a + str(i) for i, a in enumerate(atypes)])
        resnames = np.array(["UNK" for _ in anames])
        resindices = np.array([0] * len(anames))
        resnums = np.array([1] * len(anames))
        segindices = np.array([0] * len(anames))
        segids = np.array(["A"] * len(anames))
        trajectory = np.array([conf.GetPositions() for conf in mol.GetConformers()])
        bonds = np.array(
            [[b.GetBeginAtomIdx(), b.GetEndAtomIdx()] for b in mol.GetBonds()]
        )
        bond_types = np.array([BondType(b.GetBondType()) for b in mol.GetBonds()])
        return cls.from_arrays(
            anames,
            atypes,
            resnames,
            resindices,
            resnums,
            segindices,
            segids=segids,
            trajectory=trajectory,
            bonds=bonds,
            bond_types=bond_types,
        )

    @classmethod
    def from_openMM(cls, simulation):
        """
        Create a MolSys from an openMM Simulation object.

        Parameters
        ----------
        simulation : openmm.simulation.Simulation
            The simulation object to create the MolSys object from.

        Returns
        -------
        sys : chilife.MolSys
            chiLife molecular system as a clone from
        """
        try:
            from openmm.unit import angstrom
        except:
            raise RuntimeError(
                "You must have the optional openMM dependency installed to use from_openMM"
            )

        top = simulation.topology
        atypes = np.array([a.element.symbol for a in top.atoms()])
        anames = np.array([a.name for a in top.atoms()])
        resnames = np.array([a.residue.name for a in top.atoms()])
        resindices = np.array([a.residue.index for a in top.atoms()])
        resnums = np.array([int(a.residue.id) for a in top.atoms()])
        segindices = np.array([a.residue.chain.index for a in top.atoms()])
        segids = np.array([a.residue.chain.id for a in top.atoms()])
        trajectory = (
            simulation.context.getState(getPositions=True)
            .getPositions(asNumpy=True)
            .value_in_unit(angstrom)
        )
        bonds = np.array([[bond.atom1.index, bond.atom2.index] for bond in top.bonds()])
        # bond_types = np.array([  for bond in top.bonds()])

        sys = cls.from_arrays(
            anames,
            atypes,
            resnames,
            resindices,
            resnums,
            segindices,
            segids=segids,
            trajectory=trajectory,
            bonds=bonds,
            # bond_types=bond_types,
        )
        return sys

    def copy(self):
        """Create a deep copy of the MolSys."""

        if hasattr(self, "bonds"):
            bonds = self.bonds
            bond_types = self.bond_types
        else:
            bonds = None
            bond_types = None

        return MolSys(
            record_types=self.record_types,
            atomids=self.atomids,
            names=self.names,
            altlocs=self.altlocs,
            resnames=self.resnames,
            resnums=self.resnums,
            icodes=self.icodes,
            chains=self.chains,
            trajectory=self.trajectory.coords,
            occupancies=self.occupancies,
            bs=self.bs,
            segs=self.segs,
            atypes=self.atypes,
            charges=self.charges,
            bonds=bonds,
            bond_types=bond_types,
            name=self.names,
        )

    def load_new(self, coordinates):
        """
        Load a new set of 3-dimensional coordinates into the MolSys.

        Parameters
        ----------
        coordinates : ArrayLike
            Set of new coordinates to load into the MolSys
        """
        self.trajectory.load_new(coordinates=coordinates)

    @property
    def _atoms(self):
        if not hasattr(self, "_Atoms"):
            self._Atoms = np.array(
                [Atom(weakref.proxy(self), i) for i in range(self.n_atoms)]
            )
        return self._Atoms

    @property
    def _residues(self):
        if not hasattr(self, "_residue_list"):
            _, first_res_ix = np.unique(self.resixs, return_index=True)
            self._residue_list = [
                Residue(weakref.proxy(self), [ix]) for ix in first_res_ix
            ]

        return self._residue_list

    @property
    def _segments(self):
        if not hasattr(self, "_segments_list"):
            _, first_seg_ix = np.unique(self.segixs, return_index=True)
            self._segments_list = [
                Segment(weakref.proxy(self), ix) for ix in first_seg_ix
            ]
        return self._segments_list

    @property
    def molsys(self):
        return self


class Trajectory:
    """
    Object containing and managing 3-dimensional coordinates of :class:`~MolSys` objects, particularly when there are
    multiple states or time steps.

    Parameters
    ----------
    coordinates : np.ndarray
        3-dimensional coordinates for all atoms and all states/frames of the associated :class:`~MolSys` object.
    molsys : MolSys
        The associated :class:`~MolSys` object.
    timestep : float
        The time step or change in time between frames/states of the trajectory/ensemble.
    """

    def __init__(self, coordinates: np.ndarray, molsys: MolSys, timestep: float = 1):
        self.molsys = molsys
        self.timestep = timestep
        self.coords = coordinates
        self.coordinate_array = coordinates
        self._frame = 0
        self.time = np.arange(0, len(self.coords)) * self.timestep

    def load_new(self, coordinates):
        """
        Load a new set (trajectory or ensemble) of 3-dimensional coordinates into the Trajectory.

        Parameters
        ----------
        coordinates : ArrayLike
            Set of new coordinates to load into the MolSys
        """
        self.coords = coordinates
        self.coordinate_array = coordinates
        self.time = np.arange(0, len(self.coords)) * self.timestep

    def __getitem__(self, item):
        if isinstance(item, (slice, list, np.ndarray)):
            return TrajectoryIterator(self, item)

        elif isinstance(item, numbers.Integral):
            self._frame = item
            return self.time[self._frame]

    def __iter__(self):
        return iter(TrajectoryIterator(self))

    def __len__(self):
        return len(self.time)

    @property
    def frame(self):
        """The frame number the trajectory is currently on"""
        return self._frame

    @frame.setter
    def frame(self, value):
        self._frame = value


class TrajectoryIterator:
    """An iterator object for looping over :class:`~Trajectory` objects."""

    def __init__(self, trajectory, arg=None):
        self.trajectory = trajectory
        if arg is None:
            self.indices = range(len(self.trajectory.time))
        elif isinstance(arg, slice):
            self.indices = range(*arg.indices(len(self.trajectory.time)))
        elif isinstance(arg, (np.ndarray, list, tuple)):
            if np.issubdtype(arg.dtype, np.integer):
                self.indices = arg
            elif np.dtype == bool:
                self.indices = np.argwhere(arg)
            else:
                raise TypeError(f"{arg.dtype} is not a valid indexor for a trajectory")
        else:
            raise TypeError(f"{arg} is not a valid indexor for a trajectory")

    def __iter__(self):
        for frame in self.indices:
            yield self.trajectory[frame]

    def __len__(self):
        return len(self.indices)


def process_statement(statement, logickws, subjectkws):
    """
    Parse a selection string to identify subsets of atoms that satisfy the given conditions.

    Parameters
    ----------
    statement : str
        The selection string to parse.
    logickws : dict
        Dictionary of logic keywords (strings) mapped to the associated logic function.
    subjectkws : dict
        Dictionary of subject keywords (strings) mapped to the associated subject attributes.

    Returns
    -------
    mask : np.ndarray
        Array of subject indices that satisfy the given conditions.
    """
    sub_statements = parse_paren(statement)

    mask = np.ones(subjectkws["_len"], dtype=bool)
    operation = None

    for stat in sub_statements:
        if stat.startswith("("):
            tmp = process_statement(stat, logickws, subjectkws)
            mask = operation(mask, tmp) if operation else tmp
            continue

        stat_split = stat.split()
        if stat_split[0] not in logickws and stat_split[0] not in subjectkws:
            raise ValueError(
                f"Invalid selection keyword: {stat_split[0]}. All selections statements and substatements must start "
                f"with a valid selection keyword"
            )

        subject = None
        values = []
        while len(stat_split) > 0:
            if stat_split[0] in subjectkws:
                subject = subjectkws[stat_split.pop(0)]
                values = []

            elif stat_split[0] in logickws:
                if subject is not None:
                    values = np.array(values, dtype=subject.dtype)
                    tmp = process_sub_statement(subject, values)
                    subject, values = None, []
                    mask = operation(mask, tmp) if operation else tmp

                # Get next operation
                operation, stat_split = build_operator(stat_split, logickws)

            else:
                values += parse_value(stat_split.pop(0))

        if subject is not None:
            values = np.array(values, dtype=subject.dtype)
            tmp = process_sub_statement(subject, values)
            mask = operation(mask, tmp) if operation else tmp
            operation = None

        elif not values and hasattr(operation, "novals"):
            mask = operation(mask, np.ones_like(mask, dtype=bool))

    return mask


def parse_paren(string):
    """
    Break down a given string to parenthetically defined subcomponents

    Parameters
    ----------
    string : str
        The string to break down.

    Returns
    -------
    results : List[str]
        List of parenthetically defined subcomponents.
    """
    stack = 0
    start_idx = 0
    end_idx = 0
    results = []

    for i, c in enumerate(string):
        if c == "(":
            if stack == 0:
                start_idx = i + 1
                if i != 0:
                    results.append(string[end_idx:i].strip())
            stack += 1
        elif c == ")":
            # pop stack
            stack -= 1

            if stack == 0:
                results.append(f"({string[start_idx:i].strip()})")
                start_idx = end_idx
                end_idx = i + 1

    if end_idx != len(string) - 1:
        results.append(string[end_idx:].strip())

    results = [res for res in results if res != ""]
    if len(results) == 1 and results[0].startswith("("):
        results = parse_paren(results[0][1:-1])

    if stack != 0:
        raise RuntimeError(
            "The provided statement is missing a parenthesis or has an extra one."
        )
    return results


def check_operation(operation, stat_split, logickws):
    """
    Given an advanced operation, identify the additional arguments being passed to the advanced operation and create
    a simple operation that defined specifically for the provided arguments.
    Parameters
    ----------
    operation : callable
        A function or callable object that operates on molecular system attributes.
    stat_split : List[str]
        the arguments associated with the operation
    logickws : dict
        Dictionary of logic keywords (strings) mapped to the associated logic function.

    Returns
    -------
    operation : callable
        A simplified version of the provided operation now accounting for the user provided parameters.
    """
    advanced_operators = (
        logickws["within"],
        logickws["around"],
        logickws["byres"],
        logickws["point"],
    )
    if operation in advanced_operators:
        outer_operation = logickws["and"]
        args = [stat_split.pop(i) for i in range(1, 1 + operation.nargs)]
        operation = partial(operation, *args)

        def toperation(a, b, outer_operation, _io):
            return outer_operation(a, _io(b))

        operation = partial(toperation, outer_operation=outer_operation, _io=operation)

    return operation


def build_operator(stat_split, logickws):
    """
    A function to combine multiple sequential operators into one.

    Parameters
    ----------
    stat_split : List[str]
        The list of string arguments passed by the user to define the sequence of operations
    logickws : dict
        Dictionary of logic keywords (strings) mapped to the associated logic function.

    Returns
    -------
    operation : callable
        A compression of sequential operations combined into one.
    split_stat : List[str]
        Any remaining arguments passed by the user that were not used to define ``operation``
    """
    operation = logickws["and"]
    unary_operators = (logickws["not"], logickws["byres"])
    binary_operators = (logickws["and"], logickws["or"])
    advanced_operators = (logickws["within"], logickws["around"], logickws["point"])

    while _io := logickws.get(stat_split[0], False):
        if _io in unary_operators:

            def toperation(a, b, operation, _io):
                return operation(a, _io(b))

            operation = partial(toperation, operation=operation, _io=_io)
            stat_split = stat_split[1:]

        elif _io in advanced_operators:
            stat_split = stat_split[1:]
            args = [stat_split.pop(0) for i in range(_io.nargs)]
            _io = partial(_io, *args)

            def toperation(a, b, operation, _io):
                return operation(a, _io(b))

            operation = partial(toperation, operation=operation, _io=_io)

            if hasattr(_io.func, "novals"):
                operation.novals = _io.func.novals

        elif _io in binary_operators:
            if operation != logickws["and"]:
                raise RuntimeError(
                    "Cannot have two binary logical operators in succession"
                )
            operation = _io
            stat_split = stat_split[1:]

        if len(stat_split) == 0:
            break

    return operation, stat_split


def parse_value(value):
    """
    Helper function to parse values that may have slices.

    Parameters
    ----------
    value : str, int, ArrayLike[int]
        Values to parse.

    Returns
    -------
    return_value : List
        A list of values that satisfy the input.

    """
    return_value = []
    if "-" in value:
        start, stop = [int(x) for x in value.split("-")]
        return_value += list(range(start, stop + 1))
    elif ":" in value:
        start, stop = [int(x) for x in value.split(":")]
        return_value += list(range(start, stop + 1))
    else:
        return_value.append(value)

    return return_value


def process_sub_statement(subject, values):
    if subject.dtype == bool:
        return subject

    return np.isin(subject, values)


class AtomSelection(MolecularSystemBase):
    """
    An object containing a group of atoms from a :class:`~MolSys` object

    Parameters
    ----------
    molsys : :class:`~MolSys`
        The :class:`~MolSys` object from which the AtomSelection belongs to.
    mask : np.ndarray
        An array of atom indices that defines the atoms of ``molsys`` that make up the atom selection.
    """

    def __new__(cls, *args):
        if len(args) == 2:
            molsys, mask = args
        else:
            return object.__new__(cls)

        if isinstance(mask, int):
            return Atom(molsys, mask)
        else:
            return object.__new__(cls)

    def __init__(self, molsys, mask, bond_mask=None):
        self.molsys = molsys
        self.mask = mask
        self._update_bond_mask(bond_mask)

    def __getitem__(self, item):
        if np.issubdtype(type(item), np.integer):
            relidx = self.molsys.ix[self.mask][item]
            return self.molsys._atoms[relidx]

        if isinstance(item, slice):
            return AtomSelection(self.molsys, self.mask[item])

        elif isinstance(item, (np.ndarray, list, tuple)):
            item = np.asarray(item)
            if len(item) == 1 or item.sum() == 1:
                return self.molsys._atoms[self.mask[item][0]]
            elif item.dtype == bool:
                item = np.argwhere(item).T[0]
                return AtomSelection(self.molsys, item)
            else:
                return AtomSelection(self.molsys, self.mask[item])

        elif hasattr(item, "__iter__"):
            if all([np.issubdtype(type(x), int) for x in item]):
                return AtomSelection(self.molsys, self.mask[item])

        raise TypeError(
            "Only integer, slice type, and boolean mask arguments are supported at this time"
        )

    def __len__(self):
        return len(self.mask)


class ResidueSelection(MolecularSystemBase):
    """
    An object containing a group of residues and all their atoms from a :class:`~MolSys` object. Note that if one atom of
    a residue is present in the AtomSelection used to create the group, all atoms of that residue will be present in
    the ResidueSelection object.

    .. note::
        Unlike a regular :class:`~AtomSelection` object, the ``resnames``, ``resnums``, ``segids`` and ``chains``
        attributes of this object are tracked with respect to residue not atom and changes made to these attributes
        on this object will not be present in the parent object and vice versa.

    Parameters
    ----------
    molsys : :class:`~MolSys`
        The :class:`~MolSys` object from which the AtomSelection belongs to.
    mask : np.ndarray
        An array of atom indices that defines the atoms of ``molsys`` that make up the atom selection.
    bond_mask : np.ndarray
        boolean array specifying which bonds belong to the ``ResidueSelection``.
    """

    def __init__(self, molsys, mask, bond_mask=None):
        resixs = np.unique(molsys.resixs[mask])
        self.mask = np.argwhere(np.isin(molsys.resixs, resixs)).flatten()
        self.molsys = molsys

        _, self.first_ix = np.unique(molsys.resixs[self.mask], return_index=True)

        if len(self.mask) > 1:
            self.resixs = self.resixs[self.first_ix].flatten()
            self.resnames = self.resnames[self.first_ix].flatten()
            self.resnums = self.resnums[self.first_ix].flatten()
            self.segids = self.segids[self.first_ix].flatten()
            self.chains = self.chains[self.first_ix].flatten()

        self._update_bond_mask(bond_mask)

    def __setattr__(self, key, value):
        self.__dict__[key] = value

    def __getitem__(self, item):
        resixs = np.unique(self.molsys.resixs[self.mask])
        new_resixs = resixs[item]
        new_mask = np.argwhere(np.isin(self.molsys.resixs, new_resixs)).flatten()

        if np.issubdtype(type(item), int):
            return self.molsys._residues[new_resixs]
        elif isinstance(item, slice):
            return ResidueSelection(self.molsys, new_mask)
        elif hasattr(item, "__iter__"):
            if all([np.issubdtype(type(x), int) for x in item]):
                return ResidueSelection(self.molsys, new_mask)

        raise TypeError(
            "Only integer and slice type arguments are supported at this time"
        )

    def __len__(self):
        return len(self.first_ix)

    def __iter__(self):
        for resix in self.resixs:
            yield self.molsys._residues[resix]


class SegmentSelection(MolecularSystemBase):
    """
    An object containing a group of segments and all their atoms from a :class:`~MolSys` object. Note that if one atom
    of a segment is present in the AtomSelection used to create the group, all atoms of that segment will be present in
    the SegmentSelection object.

    .. note::
        Unlike a regular :class:`~AtomSelection` object, the ``segids`` and ``chains``
        attributes of this object are tracked with respect to segments not atoms and changes made to these attributes
        on this object will not be present in the parent object and vice versa.

    Parameters
    ----------
    molsys : :class:`~MolSys`
        The :class:`~MolSys` object from which the AtomSelection belongs to.
    mask : np.ndarray
        An array of atom indices that defines the atoms of ``molsys`` that make up the atom selection.
    bond_mask : np.ndarray
        boolean array specifying which bonds belong to the ``SegmentSelection``.
    """

    def __init__(self, molsys, mask, bond_mask=None):
        self.seg_ixs = np.unique(molsys.segixs[mask])
        self.mask = np.argwhere(np.isin(molsys.segixs, self.seg_ixs)).flatten()
        self.molsys = molsys

        _, self.first_ix = np.unique(molsys.segixs[self.mask], return_index=True)

        self.segids = self.segids[self.first_ix].flatten()
        self.chains = self.chains[self.first_ix].flatten()
        self._update_bond_mask(bond_mask)

    def __setattr__(self, key, value):
        self.__dict__[key] = value

    def __getitem__(self, item):
        segixs = np.unique(self.molsys.segixs[self.mask])
        new_segixs = segixs[item]
        new_mask = np.argwhere(np.isin(self.molsys.segixs, new_segixs))

        if np.issubdtype(type(item), int):
            return self.molsys._segments[new_segixs]
        elif isinstance(item, slice):
            return SegmentSelection(self.molsys, new_mask)
        elif hasattr(item, "__iter__"):
            if all([np.issubdtype(type(x), int) for x in item]):
                return SegmentSelection(self.molsys, new_mask)

        raise TypeError(
            "Only integer and slice type arguments are supported at this time"
        )

    def __len__(self):
        return len(self.first_ix)

    def __iter__(self):
        for new_segix in self.seg_ixs:
            yield self.molsys._segments[new_segix]


class Atom(MolecularSystemBase):
    """
    An object containing information for a single atom.

    Parameters
    ----------
    molsys : :class:`~MolSys`
        The :class:`~MolSys` object from which the Atom belongs to.
    mask : np.ndarray
        The index that defines the atoms of ``molsys`` that make up the atom selection.
    """

    def __init__(self, molsys, mask):
        self.molsys = molsys
        self.index = mask
        self.mask = mask

    @property
    def position(self):
        """Cartesian coordinates of the atom in angstroms."""
        return self.molsys.coords[self.index]

    @property
    def bonds(self):
        """list of :class:`AtomSelection` objects containing the atoms for each bond that this atom is involved in"""
        idxs = self.topology.bonds_any_atom[self.mask]
        return [AtomSelection(self.molsys, idx) for idx in idxs]

    @property
    def angles(self):
        """list of :class:`AtomSelection` objects containing the atoms for each angle that this atom is involved in"""
        idxs = self.topology.angles_any_atom[self.mask]
        return [AtomSelection(self.molsys, idx) for idx in idxs]

    @property
    def dihedrals(self):
        """list of :class:`AtomSelection` objects containing the atoms for each dihedral that this atom is involved
        in"""
        idxs = self.topology.dihedrals_any_atom[self.mask]
        return [AtomSelection(self.molsys, idx) for idx in idxs]

    @property
    def bonded_atoms(self):
        """:class:`AtomSelection` object containing all atoms that this atom is bonded to. Does not contain this atom"""
        all_idxs = np.unique(np.concatenate(self.topology.bonds_any_atom[self.mask]))
        all_idxs = np.array([idx for idx in all_idxs if idx != self.mask])
        return AtomSelection(self.molsys, all_idxs)


class Residue(MolecularSystemBase):
    """
    An object representing a single residue.

    Parameters
    ----------
    molsys : :class:`~MolSys`
        The :class:`~MolSys` object from which the Residue belongs to.
    mask : int
        An array of atom indices that defines the atoms of the residue
    bond_mask : np.ndarray
        boolean array specifying which bonds belong to the ``Residue``.
    """

    def __init__(self, molsys, mask, bond_mask=None):
        resix = np.unique(molsys.resixs[mask])[0]
        self.mask = np.where(molsys.resixs == resix)[0]
        self.molsys = molsys
        self._update_bond_mask(bond_mask)
        self.resindex = resix
        self.segindex = molsys.segixs[self.mask][0]

    def __len__(self):
        return len(self.mask)

    def phi_selection(self):
        """
        Get an :class:`~AtomSelection` of the atoms defining the Phi backbone dihedral angle of the residue.

        Returns
        -------
        sel : AtomSelection
            An :class:`~AtomSelection` object containing the atoms that make up the Phi backbone dihedral angle of
            the selected residue.
        """

        prev = self.previous_residue()
        if prev is None:
            return None

        mask_phi = [
            prev.ix[prev.names == "C"].flat[0],
            self.ix[self.names == "N"].flat[0],
            self.ix[self.names == "CA"].flat[0],
            self.ix[self.names == "C"].flat[0],
        ]

        sel = self.molsys.atoms[mask_phi]
        return sel if len(sel) == 4 else None

    def psi_selection(self):
        """
        Get an :class:`~AtomSelection` of the atoms defining the Psi backbone dihedral angle of the residue.

        Returns
        -------
        sel : AtomSelection
            An :class:`~AtomSelection` object containing the atoms that make up the Psi backbone dihedral angle of
            the selected residue.
        """

        nex = self.next_residue()
        if nex is None:
            return None

        mask_psi = [
            self.ix[self.names == "N"].flat[0],
            self.ix[self.names == "CA"].flat[0],
            self.ix[self.names == "C"].flat[0],
            nex.ix[nex.names == "N"].flat[0],
        ]
        sel = self.molsys.atoms[mask_psi]
        return sel if len(sel) == 4 else None

    def previous_residue(self, ignore_chain_breaks=False):
        resix = self.resix - 1
        res = self.molsys._residues[resix] if resix >= 0 else None

        if ignore_chain_breaks or res is None:
            return res

        elif self.resnum - 1 == res.resnum and self.segix == res.segix:
            return res
        else:
            return None

    def next_residue(self, ignore_chain_breaks=False):
        resix = self.resix + 1
        res = (
            self.molsys._residues[resix] if resix <= self.molsys.resixs.max() else None
        )

        if ignore_chain_breaks or res is None:
            return res

        elif self.resnum + 1 == res.resnum and self.segix == res.segix:
            return res
        else:
            return None


class Segment(MolecularSystemBase):
    """
    An object representing a single segment of a molecular system.

    Parameters
    ----------
    molsys : MolSys
        The :class:`~MolSys` object from which the segment belongs to.
    mask : int
        An array of atom indices that defines the atoms of the segment.
    """

    def __init__(self, molsys, mask):
        segix = np.unique(molsys.segixs[mask])[0]
        self.mask = np.where(molsys.segixs == segix)[0]
        self.molsys = molsys
        self.segindex = segix

        self._update_bond_mask(mask)


def byres(mask, molsys):
    """Advanced operator to select atoms by residue"""
    residues = np.unique(molsys.resixs[mask])
    mask = np.isin(molsys.resixs, residues)
    return mask


def unot(mask):
    """Unitary `not` operator"""
    return ~mask


def within(distance, mask, molsys):
    """
    Advanced logic operator to identify atoms within a user defined distance of another selection.

    Parameters
    ----------
    distance : float
        The distance window defining which atoms will be included in the selection.
    mask : np.ndarray
        A boolean array defining a subset of  atoms of a :class:`~MolSys` from which the distance cutoff will be
        measured by.
    molsys : :class:`~MolSys`
        A chiLife :class:`~MolSys` from which to select atoms from .

    Returns
    -------
    out_mask : np.ndarray
        A boolean array defining a subset of  atoms of a :class:`~MolSys` that are within the user defined distance
        of the user defined selection.
    """
    distance = float(distance)
    tree1 = cKDTree(molsys.coords)
    tree2 = cKDTree(molsys.molsys.coords[mask])
    results = np.concatenate(tree2.query_ball_tree(tree1, distance))
    results = np.unique(results)
    out_mask = np.zeros_like(mask, dtype=bool)
    out_mask[results] = True
    out_mask = out_mask * ~mask
    return out_mask


within.nargs = 1


def point(x, y, z, distance, mask, molsys):
    """
    Advanced logic operator to identify atoms within a user defined distance of a point in cartesian space.

    Parameters
    ----------
    x : float
        The x coordinate of the point in cartesian space.
    y : float
        The y coordinate of the point in cartesian space.
    z : float
        The z coordinate of the point in cartesian space.
    distance : float
        The distance window defining which atoms will be included in the selection.
    mask : np.ndarray
        A boolean array defining a subset of  atoms of a :class:`~MolSys` from which the distance cutoff will be
        measured by.
    molsys : :class:`~MolSys`
        A chiLife :class:`~MolSys` from which to select atoms from .

    Returns
    -------
    out_mask : np.ndarray
        A boolean array defining a subset of  atoms of a :class:`~MolSys` that are within the user defined distance
        of the user defined selection.
    """

    tree1 = cKDTree(molsys.coords[mask])
    args = tree1.query_ball_point(np.array([x, y, z], dtype=float), distance)
    out_mask = np.zeros_like(mask, dtype=bool)
    out_mask[args] = True

    return out_mask


point.nargs = 4
point.novals = True


def concat_molsys(*systems):
    """
    Function to concatenate two or more :class:`~MolSys` objects into a single :class:`~MolSys` object. Atoms will be
    concatenated in the order they are placed in the list.

    Parameters
    ----------
    systems : List[MolSys]
        List of :class:`~MolSys` objects to concatenate.

    Returns
    -------
    mol : MolSys
        The concatenated `~MolSys` object.
    """
    for sys in systems:
        if not isinstance(sys, MolecularSystemBase):
            raise TypeError("Can only concatenate MolecularSystemBase instances")

    atom_ids = []
    record_types = []
    anames = []
    altlocs = []
    atypes = []
    resnames = []
    resnums = []
    resindices = []
    icodes = []
    chains = []
    occupancies = []
    bs = []
    segs = []
    trajectory = []
    segindices = []
    segids = []
    charges = []
    chiral = []
    bonds = []
    bond_types = []
    bond_chiral = []
    names = []
    n_atoms = 0
    n_residues = 0
    n_chains = 0

    traj_len = len(systems[0].trajectory)

    has_bonds = any(sys.bonds is not None for sys in systems)

    for sys in systems:
        record_types.append(sys.record_types)
        atom_ids.append(sys.atomids)
        anames.append(sys.names)
        altlocs.append(sys.altlocs)
        resnames.append(sys.resnames)
        resnums.append(sys.resnums)
        icodes.append(sys.icodes)
        chains.append(sys.chains)
        occupancies.append(sys.occupancies)
        bs.append(sys.bs)
        segs.append(sys.segids)
        atypes.append(sys.atypes)
        charges.append(sys.charges)
        chiral.append(sys.chiral)

        if len(sys.trajectory) != traj_len:
            raise AttributeError(
                "Atom selections must have the same trajectory lengths to be combined"
            )

        trajectory.append(sys.trajectory.coordinate_array[:, sys.mask])

        # New bonds
        if not has_bonds:
            bonds = None
            bond_types = None
            bond_chiral = None
        elif sys.bonds is None:
            bonds.append(np.array([[]]))
            bond_types.append(np.array([]))
            bond_chiral.append(np.array([]))
        else:
            old_atom_ixs = sys.ix
            atom_ix_map = {old: new + n_atoms for new, old in enumerate(old_atom_ixs)}
            b_mask = np.all(np.isin(sys.bonds, old_atom_ixs), axis=1)

            sys_i_bonds = sys.bonds[b_mask]
            new_bonds = [[atom_ix_map[a], atom_ix_map[b]] for a, b in sys_i_bonds]
            bonds.append(new_bonds)
            bond_types.append(sys.bond_types[b_mask])
            bond_chiral.append(sys.bond_chiral[b_mask])

        # New resinidces starting from 0
        new_resindices = np.zeros_like(sys.resindices)
        for i, res in enumerate(sys.residues):
            mask = sys.resindices == res.resindex
            new_resindices[mask] = i + n_residues
        resindices.append(new_resindices)

        # New segindices starting from 0
        new_segindices = np.zeros_like(sys.segindices)
        for i, seg in enumerate(sys.segments):
            mask = sys.segindices == seg.segindex
            new_segindices[mask] = i + n_chains
        segindices.append(new_segindices)

        n_atoms += len(sys.atoms)
        n_residues += len(sys.residues)
        n_chains += len(sys.segments)
        names.append(sys.name)

    record_types = np.concatenate(record_types)
    atom_ids = np.concatenate(atom_ids)
    anames = np.concatenate(anames)
    altlocs = np.concatenate(altlocs)
    resnames = np.concatenate(resnames)
    resnums = np.concatenate(resnums)
    icodes = np.concatenate(icodes)
    chains = np.concatenate(chains)
    trajectory = np.concatenate(trajectory, axis=1)
    occupancies = np.concatenate(occupancies)
    bs = np.concatenate(bs)
    segs = np.concatenate(segs)
    atypes = np.concatenate(atypes)
    charges = np.concatenate(charges)
    chiral = np.concatenate(chiral) if np.any([x is not None for x in chiral]) else None
    resindices = np.concatenate(resindices)
    segindices = np.concatenate(segindices)

    if has_bonds:
        bonds = np.concatenate(bonds)
        bond_types = np.concatenate(bond_types)
        bond_chiral = np.concatenate(bond_chiral)

    if len(names) <= 3:
        name = "_and_".join(names)
    else:
        name = f"concatenation of {len(systems)} atom selections"

    return MolSys(
        record_types,
        atom_ids,
        anames,
        altlocs,
        resnames,
        resnums,
        icodes,
        chains,
        trajectory,
        occupancies,
        bs,
        segs,
        atypes,
        charges,
        chiral,
        resindices,
        segindices,
        bonds,
        bond_types,
        bond_chiral,
        name,
    )


def res_to_ccd(res):
    """
    Create a CCD entry for a :class:`Residue`.

    Parameters
    ----------
    res : Residue
        The residue to create the CCD entry for.

    Returns
    -------
    ccd_data : dict
        Dictionary containing the ``chem_comp_atom``, ``chem_comp_bond`` data derived from the atoms ond bonds of the
        :class:`Residue`.
    """

    chem_comp_atom = {
        "comp_id": [str(res.resname)] * len(res.atoms),
        "atom_id": res.names.tolist(),
        "type_symbol": res.elements.tolist(),
        "pdbx_aromatic_flag": ["?"] * len(res.atoms),
        "pdbx_stereo_config": res.chiral.tolist(),
    }

    bond_atom_names = res.bond_atom_names
    bond_types = [chilife.io.XL_BOND_TO_CIF[btype] for btype in res.bond_types]
    if len(bond_atom_names) > 0:
        chem_comp_bond = {
            "comp_id": [str(res.resname)] * len(res.bonds),
            "atom_id_1": bond_atom_names[:, 0].tolist(),
            "atom_id_2": bond_atom_names[:, 1].tolist(),
            "value_order": bond_types,
            "pdbx_aromatic_flag": ["?"] * len(res.bonds),
            "pdbx_stereo_config": res.bond_chiral.tolist(),
        }
    else:
        chem_comp_bond = {
            "comp_id": [],
            "atom_id_1": [],
            "atom_id_2": [],
            "value_order": [],
            "pdbx_aromatic_flag": [],
            "pdbx_stereo_config": [],
        }

    ccd_data = {
        "chem_comp_atom": chem_comp_atom,
        "chem_comp_bond": chem_comp_bond,
    }

    return ccd_data


def struct_to_cif_atom_site(struct):
    """
    Helper function to generate the cif ``_atom_site`` entry for a cif file.

    Parameters
    ----------
    struct : MolecularSystemBase
        The structure or structure to be converted into the ``_atom_site`` entry.

    Returns
    -------
    atom_site_dict : dict
        Dictionary containing the ``_atom_site`` entry for a cif file.
    """
    n_frames = len(struct.trajectory)
    atom_group = np.tile(struct.record_types, n_frames).tolist()
    atom_id = np.tile(struct.atomids.astype(str), n_frames).tolist()
    type_symbol = np.tile(struct.atypes, n_frames).tolist()
    label_atom_id = np.tile(struct.names, n_frames).tolist()
    label_alt_id = np.tile(
        [x if x.strip() else "." for x in struct.altlocs], n_frames
    ).tolist()
    label_comp_id = np.tile(struct.resnames, n_frames).tolist()
    label_asym_id = np.tile(struct.segids, n_frames).tolist()
    label_entity_id = np.tile((struct.segindices + 1).astype(str), n_frames).tolist()
    label_seq_id = np.tile((struct.resindices + 1).astype(str), n_frames).tolist()
    pdbx_PDB_ins_code = np.tile(
        [x if x.strip() else "?" for x in struct.icodes], n_frames
    ).tolist()
    occupancy = np.tile(struct.occupancies.astype(str), n_frames).tolist()
    B_iso_or_equiv = np.tile(struct.bs.astype(str), n_frames).tolist()
    auth_seq_id = np.tile(struct.resnums.astype(str), n_frames).tolist()
    auth_asym_id = np.tile(struct.chains, n_frames).tolist()
    pdbx_PDB_model_num = np.concatenate(
        [[str(i + 1)] * struct.n_atoms for i in range(n_frames)]
    ).tolist()

    cartn_x, cartn_y, cartn_z = [], [], []

    for i, ts in enumerate(struct.trajectory):
        cartn_x.extend(struct.positions[:, 0].astype(str).tolist())
        cartn_y.extend(struct.positions[:, 1].astype(str).tolist())
        cartn_z.extend(struct.positions[:, 2].astype(str).tolist())

    atom_site_dict = {
        "group_PDB": atom_group,
        "id": atom_id,
        "type_symbol": type_symbol,
        "label_atom_id": label_atom_id,
        "label_alt_id": label_alt_id,
        "label_comp_id": label_comp_id,
        "label_asym_id": label_asym_id,
        "label_entity_id": label_entity_id,
        "label_seq_id": label_seq_id,
        "pdbx_PDB_ins_code": pdbx_PDB_ins_code,
        "cartn_x": cartn_x,
        "cartn_y": cartn_y,
        "cartn_z": cartn_z,
        "occupancy": occupancy,
        "B_iso_or_equiv": B_iso_or_equiv,
        "auth_seq_id": auth_seq_id,
        "auth_asym_id": auth_asym_id,
        "pdbx_PDB_model_num": pdbx_PDB_model_num,
    }

    return atom_site_dict
