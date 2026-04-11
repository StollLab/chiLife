import re
from pathlib import Path
from copy import deepcopy
import warnings
from itertools import combinations

import numpy as np
from numpy.typing import ArrayLike
from scipy.spatial import cKDTree

import igraph as ig
import MDAnalysis as mda

import chilife.io as io
import chilife.scoring as scoring

from .protein_utils import new_chain, FreeAtom, guess_chain
from .ligand_utils import assign_atom_names, remap_sdf
from .alignment_methods import ligand_alignment_methods, linear_alignment
from .MolSys import MolSys, MolecularSystemBase
from .MolSysIC import MolSysIC
from .base_classes import Ensemble
from .math_utils import simple_cycle_vertices

default_energy_func = scoring.ljEnergyFunc()


class LigandEnsemble(Ensemble):
    """Create new ``LigandEnsemble`` object.

    Parameters
    ----------
    res : string
        3-character name of desired residue, e.g. SAM.
    site : int
        :class:`~MolSys` residue number to attach library to.
    protein : MDAnalysis.Universe, MDAnalysis.AtomGroup, MolSys
        Object containing all protein information (coords, atom types, etc.)
    chain : str
        :class:`~MolSys` chain identifier to attach spin label to.
    rotlib : str
        Rotamer library to use for constructing the ``LigandEnsemble``
    **kwargs : dict
        exclude_nb_interactions: int:
            When calculating internal clashes, ignore 1-``exclude_nb_interactions`` interactions and below. Defaults
            to ignore 1-3 interactions, i.e. atoms that are connected by 2 bonds or fewer will not have a steric
            effect on each other.
        eval_clash : bool
            Switch to turn clash evaluation on (True) and off (False).
        forcefield: str
            Name of the forcefield you wish to use to parameterize atoms for the energy function. Currently, supports
            `charmm` and `uff`. NOTE: chilife forcefields currently only parameterize clashes and not electrostatics
            or bonded interactions.
        energy_func : callable
            Python function or callable object that takes a protein or an :class:`Ensemble` object as input and
            returns an energy value (kcal/mol) for each atom of each rotamer in the ensemble. See also
            :mod:`Scoring <chiLife.scoring>` . Defaults to a capped lennard-jones repulsive energy function.
        temp : float
           Temperature to use when running ``energy_func``. Only used if ``energy_func`` accepts a temperature
           argument  Defaults to 298 K.
        clash_radius : float
           Cutoff distance (angstroms) for inclusion of atoms in clash evaluations. This distance is measured from
           ``clash_ori`` Defaults to the longest distance between any two atoms in the ensemble plus 5
           angstroms.
        clash_ori : str
           Atom selection to use as the origin when finding atoms within the ``clash_radius``. Defaults to 'cen',
           the centroid of the ensemble heavy atoms.
        ignore_wateres : bool
            Determines whether water molecules (resnames HOH and OH2) are ignored during clash evaluation.
            Defaults to True.
        protein_tree : Scipy.spatial.cKDTree
           KDTree of atom positions for fast distance calculations and neighbor detection. Defaults to None
        trim: bool
            When true, the lowest `trim_tol` fraction of rotamers in the ensemble will be removed.
        trim_tol: float
            Tolerance for trimming rotamers from the ensemble. ``trim_tol=0.005`` means the bottom 0.5% of rotamers
            will be removed.
        alignment_method : str
           Method to use when attaching or aligning the ensemble with the specified site of the :class:`~MolSys`.
           Defaults to ``'svd'`` which aligns the principal components of each ligand in the ensemble with the
           specified residue of the associated :class:`MolSys`.
        sample : int, bool
           Argument to use the off-rotamer sampling method. If ``False`` or ``0`` the off-rotamer sampling method
           will not be used. If ``int`` the ensemble will be generated with that many off-rotamer samples.
        dihedral_sigmas : float, numpy.ndarray
           Standard deviations of dihedral angles (degrees) for off rotamer sampling. Can be a single number for
           isotropic sampling, a vector to define each dihedral individually or a matrix to define a value for
           each rotamer and each dihedral. Setting this value to np.inf will force uniform (accessible volume)
           sampling. Defaults to 35 degrees.
        weighted_sampling : bool
           Determines whether the rotamer ensemble is sampled uniformly or based off of their intrinsic weights.
           Defaults to ``False``.
        use_H : bool
           Determines if hydrogen atoms are used or not. Defaults to ``True``.
    """

    def __init__(self, res, site=None, protein=None, chain=None, rotlib=None, **kwargs):
        self.res = res
        if site is None and protein is not None:
            raise ValueError(
                "A protein has been provided but a site has not. If you wish to construct an ensemble "
                "associated with a protein you must include the site you wish to model."
            )
        elif site is None:
            site = 1
            icode = ""
        elif isinstance(site, str):
            site_split = re.split(r"(\D+)", site)
            site = site_split[0]
            icode = site_split[1] if len(site_split) > 1 else ""
        else:
            icode = ""

        self.site = int(site)
        self.icode = icode
        self.resnum = self.site
        self.chain = chain if chain is not None else guess_chain(protein, self.site)
        self.selstr = f"resid {self.site} and segid {self.chain} and not altloc B"
        self.nataa = ""
        self.input_kwargs = kwargs
        self._sdf_data = kwargs.pop("_sdf_data", None)
        self.__dict__.update(assign_defaults(kwargs))
        self.input_kwargs.pop("energy_func", None)
        self.protein = protein

        # Convert string arguments for alignment_method to respective function
        if isinstance(self.alignment_method, str):
            self.alignment_method = ligand_alignment_methods[self.alignment_method]

        lib = self.get_lib(rotlib)
        self.__dict__.update(lib)
        self._weights = self.weights / self.weights.sum()

        if len(self.sigmas) != 0 and "dihedral_sigmas" not in kwargs:
            self.sigmas[self.sigmas == 0] = self.dihedral_sigmas
        else:
            self.set_dihedral_sampling_sigmas(self.dihedral_sigmas)

        self._rsigmas = np.deg2rad(self.sigmas)
        with np.errstate(divide="ignore"):
            self._rkappas = 1 / self._rsigmas**2

        self._rkappas = np.clip(self._rkappas, 0, 1e16)

        # Remove hydrogen atoms unless otherwise specified
        if not self.use_H:
            self.H_mask = self.atom_types != "H"
            self._coords = self._coords[:, self.H_mask]
            self.atom_types = self.atom_types[self.H_mask]
            self.atom_names = self.atom_names[self.H_mask]

        self.ic_mask = np.argwhere(
            np.isin(self.internal_coords.atom_names, self.atom_names)
        ).flatten()
        self._lib_coords = self._coords.copy()
        self._lib_dihedrals = self._dihedrals.copy()
        self._lib_IC = self.internal_coords

        if self.clash_radius is None:
            self.clash_radius = (
                np.linalg.norm(self.clash_ori - self.coords, axis=-1).max() + 5
            )

        self._graph = ig.Graph(edges=self.bonds)
        self.aidx, self.bidx = [list(x) for x in zip(*self.non_bonded)]
        self.side_chain_idx = np.arange(len(self.atom_names))

        # Allocate variables for clash evaluations
        self.atom_energies = None
        self.clash_ignore_idx = None
        self.partition = 1

        # Assign a name to the label
        self.name = self.rotlib
        if self.site is not None:
            self.name = self.nataa + str(self.site) + self.rotlib
        if self.chain not in ("A", None):
            self.name += f"_{self.chain}"

        self.update(no_lib=True)

        # Store atom information as atom objects
        self.atoms = [
            FreeAtom(name, atype, idx, self.res, self.site, coord)
            for idx, (coord, atype, name) in enumerate(
                zip(self._coords[0], self.atom_types, self.atom_names)
            )
        ]

    @classmethod
    def from_sdf(
        cls, file_name, res="LIG", site=None, protein=None, chain=None, **kwargs
    ):
        """
        A function to create a LigandEnsemble from a SDF file.

        Parameters
        ----------
        file_name : str
            The name of the SDF file.

        Returns
        -------
        cls : LigandEnsemble
            The LigandEnsemble object.

        """

        chain = (
            chain
            if chain is not None
            else guess_chain(protein, site)
            if site is not None
            else new_chain(protein)
        )
        site = 1 if site is None else site
        sdf_data = io.read_sdf(file_name)
        kwargs["_sdf_data"] = sdf_data
        coords = np.array([[atom["xyz"] for atom in mol["atoms"]] for mol in sdf_data])

        # Assuming all mols are identical
        mol = sdf_data[0]
        n_atoms = len(mol["atoms"])
        n_conformers = len(coords)

        atom_names = np.array(assign_atom_names(mol))
        atom_types = np.array([atom["element"] for atom in mol["atoms"]])
        resnames = np.array([res] * n_atoms)
        resindices = np.zeros(n_atoms) * site
        resnums = np.ones(n_atoms)
        segindices = np.zeros(n_atoms)
        record_types = np.array(["HETATM"] * n_atoms)
        segids = np.array([chain] * n_atoms)
        bonds = np.array([[bond["idx1"], bond["idx2"]] for bond in mol["bonds"]])
        bond_types = np.array([bond["type"] for bond in mol["bonds"]])

        btype_map = {tuple(bond): btype for bond, btype in zip(bonds, bond_types)}
        G = ig.Graph(len(atom_names), bonds)

        # Dont consider bonds to hydrogen atoms.
        H_idxs = np.argwhere(atom_types == "H")
        mobile_bonds = [bond for bond in bonds if ~np.any(np.isin(bond, H_idxs))]

        # Map heavy atom degrees
        neighbors = {}
        for bond in mobile_bonds:
            for b in bond:
                if b not in neighbors:
                    neighbors[b] = []

                nieghbor = bond[0] if b == bond[1] else bond[1]
                neighbors[b].append(nieghbor)

        # Only consider bonds that aren't part of the same ring
        ring_idxs = simple_cycle_vertices(G)
        mobile_bonds = [
            bond
            for bond in mobile_bonds
            if ~np.any([np.all(np.isin(bond, r)) for r in ring_idxs])
        ]

        # Only look at single bonds
        mobile_bonds = [bond for bond in mobile_bonds if btype_map[tuple(bond)] == 1]

        # Don't consider terminal mobile bonds (e.g. methyls or hydroxyls) or
        significant_bonds = []
        for bond in mobile_bonds:
            if len(neighbors[bond[0]]) == 1:
                continue
            if len(neighbors[bond[1]]) == 1:
                continue
            significant_bonds.append(bond)

        # Find a suitable root, make sure its only bonded to one atom that is at max degree 3
        for root, neigh in neighbors.items():
            if len(neigh) == 1 and len(neighbors[neigh[0]]) < 4:
                break

        # DFS to sort atoms
        sort_idx, layer_idx, parents = G.bfs(root)

        # Move hydrogen atoms to end
        v = {idx: int(atom_types[idx] == "H") for idx in sort_idx}
        sort_idx = sorted(sort_idx, key=lambda x: v[x])

        # Ensure at least 4 consecutive atoms starting the ligand so the 4th atom doesn't need an improper
        if (
            parents[sort_idx[2]] == parents[sort_idx[3]]
            and parents[sort_idx[4]] != sort_idx[3]
        ):
            sort_idx[3], sort_idx[4] = sort_idx[4], sort_idx[3]

        # Create maps so original order can be preserved
        forward_map = {n: i for i, n in enumerate(sort_idx)}
        back_map = np.argsort(sort_idx)

        # Sort the atoms
        atom_names = atom_names[sort_idx]
        atom_types = atom_types[sort_idx]
        coords = coords[:, sort_idx]
        bonds = np.array([[forward_map[k1], forward_map[k2]] for k1, k2 in bonds])

        # Create MolSys
        molsys = MolSys.from_arrays(
            atom_names,
            atom_types,
            resnames,
            resindices,
            resnums.astype(int),
            segindices.astype(int),
            record_types,
            segids=segids,
            trajectory=coords,
            bonds=np.array(bonds),
            bond_types=np.array(
                [io.SDF_BOND_TO_XL[bond_type] for bond_type in bond_types]
            ),
        )

        # Create internal coords
        ICs = MolSysIC.from_atoms(molsys, bonds=bonds, bond_types=bond_types)

        # Identify mobile dihedrals
        dihedral_idxs = []
        new_neighbors = {
            forward_map[k]: [forward_map[vi] for vi in v] for k, v in neighbors.items()
        }
        new_significant_bonds = [
            [forward_map[x] for x in bond] for bond in significant_bonds
        ]
        for a, b in new_significant_bonds:
            if a > b:
                a, b = b, a

            for a1 in new_neighbors[a]:
                if a1 < a:
                    break
            for b1 in new_neighbors[b]:
                if b1 > b:
                    break

            dihedral_idxs.append([a1, a, b, b1])

        dihedral_idxs = [sorted(lst) for lst in dihedral_idxs]
        dihedral_idxs = sorted(dihedral_idxs, key=sum)
        mobile_dihedrals = [
            [atom_names[xi] for xi in dihedral] for dihedral in dihedral_idxs
        ]

        weights = kwargs.pop("weights", np.ones(n_conformers) / n_conformers)
        dihedrals = np.array([ic.get_dihedral(1, mobile_dihedrals) for ic in ICs])

        if len(dihedrals.shape) == 1:
            dihedrals = dihedrals[:, None]

        sigmas = kwargs.get("sigmas", np.array([]))

        lib = {
            "rotlib": f"{res}_from_sdf",
            "resname": res,
            "coords": np.atleast_3d(coords),
            "internal_coords": ICs,
            "weights": weights,
            "atom_types": atom_types,
            "atom_names": atom_names,
            "dihedral_atoms": np.array(mobile_dihedrals),
            "dihedrals": np.rad2deg(dihedrals),
            "_rdihedrals": dihedrals,
            "sigmas": sigmas,
            "_rsigmas": np.deg2rad(sigmas),
            "type": "chilife ligand ensemble library from an SDF file",
            "_back_map": back_map,
            "_forward_map": forward_map,
            "format_version": 0.0,
        }

        # Name atoms
        return cls(res, site, protein, chain, rotlib=lib, **kwargs)

    @classmethod
    def from_pdb(
        cls, file_name, res="LIG", site=None, protein=None, chain=None, **kwargs
    ):
        raise NotImplementedError

    @classmethod
    def from_rdkit(
        cls, rdmol, res="LIG", site=None, protein=None, chain=None, **kwargs
    ):
        raise NotImplementedError

    def to_sdf(self, file_name):
        """
        Write LigandEnsemble to SDF file.

        Parameters
        ----------
        file_name: str | Path
            Name or Path object of the SDF file.
        """
        if self._sdf_data is None:
            sdf_data = self.internal_coords.protein._make_sdf_data()[0]
        else:
            sdf_data = self._sdf_data[0]

        if not self.use_H:
            sdf_data = remap_sdf(sdf_data, self.H_mask)

        if hasattr(self, "_back_map"):
            coords = self.coords[:, self._backmap]
        else:
            coords = self.coords

        mols = []
        for conf in coords:
            mol = {k: v for k, v in sdf_data.items()}
            atoms = []
            for atom, xyz in zip(mol["atoms"], conf):
                new_atom = {k: v for k, v in atom.items() if k != "xyz"}
                new_atom["xyz"] = xyz
                atoms.append(new_atom)

            mol["atoms"] = atoms
            mols.append(mol)

        io.write_sdf(mols, file_name)

    def update(self, no_lib=False):
        """
        Reanalyze the ensemble in place. This is generally used to update the library if manually changing positions
        of the library or the associated protein/molecular system object. If the ensemble was made with off-rotamer
        sampling, new conformations will be sampled. If using the rotamer library method, the whole library will be
        used by default (``no_lib=False``). If you wish to use just the conformers that were retrained after trimming
        the original structure, set ``no_lib=True``.

        Parameters
        ----------
        no_lib: bool
            Argument to toggle using the current ensemble as a library or using the underlying rotamer library. If
            ``no_lib=True`` chiLife will not use the underlying rotamer library.
        """
        # Sample from library if requested
        if self._sample_size and len(self.dihedral_atoms) > 0:
            mask = self.sigmas.sum(axis=0) == 0

            # Draw samples
            self._coords, self.weights, self.internal_coords = self.sample(
                self._sample_size, off_rotamer=~mask, return_dihedrals=True
            )

            dihedrals = np.asarray(
                [
                    self.internal_coords.get_dihedral(1, self.dihedral_atoms)
                    for ts in self.internal_coords.trajectory
                ]
            )
            self._dihedrals = np.rad2deg(dihedrals)

        elif not no_lib:
            # Reset to library
            self._coords = self._lib_coords.copy()
            self._dihedrals = self._lib_dihedrals.copy()
            self.internal_coords = self._lib_IC
            self.weights = self._weights.copy()

        if self.protein is not None:
            self.protein_setup()
        else:
            self.ICs_to_site()

    def init_protein(self):
        """
        Helper function to initialize important parameters for interacting with the associated protein/molecular system
        """
        if isinstance(self.protein, (mda.AtomGroup, mda.Universe)):
            if not hasattr(self.protein.universe._topology, "altLocs"):
                self.protein.universe.add_TopologyAttr(
                    "altLocs", np.full(len(self.protein.universe.atoms), "")
                )

        # Position library at selected residue
        self.resindex = self.protein.select_atoms(self.selstr).resindices[0]
        self.segindex = self.protein.select_atoms(self.selstr).segindices[0]
        if self.ignore_waters:
            self._protein = self.protein.select_atoms(
                "not (byres name OH2 or resname HOH)"
            )
        else:
            self._protein = self.protein.atoms

        self.protein_tree = cKDTree(self._protein.atoms.positions)

        # Delete cached lennard jones parameters if they exist.
        if hasattr(self, "ermin_ij"):
            del self.ermin_ij
            del self.eeps_ij

    def protein_setup(self):
        """
        perform necessary operations to align the rotamer library to the specified site, analyze clashes with the
        associated protein/molecular system, trim high energy rotamers, and adjust weights accordingly.
        """
        self.to_site()

        # Get weight of current or closest rotamer
        clash_ignore_idx = self.protein.select_atoms(
            f"resid {self.site} and segid {self.chain}"
        ).ix
        self.clash_ignore_idx = np.argwhere(
            np.isin(self.protein.ix, clash_ignore_idx)
        ).flatten()

        protein_clash_idx = self.protein_tree.query_ball_point(
            self.clash_ori, self.clash_radius
        )
        self.protein_clash_idx = [
            idx for idx in protein_clash_idx if idx not in self.clash_ignore_idx
        ]

        if hasattr(self.energy_func, "prepare_system"):
            self.energy_func.prepare_system(self)

        # Guess which conformer the native residue is if it matches the structure of the LigandEnsemble
        if self._coords.shape[1] == len(self.clash_ignore_idx):
            RMSDs = np.linalg.norm(
                self._coords
                - self.protein.atoms[self.clash_ignore_idx].positions[None, :, :],
                axis=(1, 2),
            )

            idx = np.argmin(RMSDs)
            self.current_weight = self.weights[idx]

        else:
            self.current_weight = 0

        # Evaluate external clash energies and reweigh rotamers
        if self._minimize and self.eval_clash:
            raise RuntimeError(
                "Both `minimize` and `eval_clash` options have been selected, but they are incompatible."
                "Please select only one. Also note that minimize performs its own clash evaluations so "
                "eval_clash is not necessary."
            )
        elif self.eval_clash:
            self.evaluate()

        elif self._minimize:
            self.minimize()

    def sample(self, n=1, off_rotamer=False, **kwargs):
        """Randomly sample a rotamer in the library.

        Parameters
        ----------
        n : int
             number of rotamers to sample.
        off_rotamer : bool | List[bool]
            If False the rotamer library used to construct the LigandEnsemble will be sampled using the exact
            dihedrals. If True, all mobile dihedrals will undergo minor perturbations. A list indicates a per-dihedral
            switch to turn on/off off-rotamer sampling of a particular dihedrals, corresponding to
            ``self.dihedral_atoms`` e.g. ``off_rotamer = [False, False, False, True, True]``  R1 will only sample
            the last two mobile dihedral angles.
        **kwargs : dict
            return_dihedrals : bool
                If True, sample will return a MolSysIC object of the sampled rotamer
            remove_clashing : bool
                If True, samples that have internal clashing atoms will be removed

        Returns
        -------
        coords : ArrayLike
            The 3D coordinates of the sampled rotamer(s)
        new_weights : ArrayLike
            New weights (relative) of the sampled rotamer(s)
        ICs : List[chilife.MolSysIC] (Optional)
            Internal coordinate (MolSysIC) objects of the rotamer(s).
        """

        if not self.weighted_sampling:
            idx = np.random.randint(len(self._weights), size=n)
        else:
            idx = np.random.choice(len(self._weights), size=n, p=self._weights)

        if not hasattr(off_rotamer, "__len__"):
            off_rotamer = (
                [off_rotamer] * len(self.dihedral_atoms)
                if len(self.dihedral_atoms) > 0
                else [True]
            )
        else:
            off_rotamer = off_rotamer[: len(self.dihedral_atoms)]

        if not any(off_rotamer):
            coords, weights = (
                np.squeeze(self._lib_coords[idx]),
                np.squeeze(self._weights[idx]),
            )

            if self.alignment_method is not None:
                target_pos = self.protein.select_atoms(
                    f"resid {self.site} and segid {self.chain} and not altloc B"
                ).positions
                mx, ori = self.alignment_method(target_pos)

                # if self.alignment_method not in {'fit', 'rosetta'}:
                old_mx, old_ori = self.alignment_method(coords)
                old_ori, old_mx = old_ori, old_mx.transpose(0, 2, 1)

                self._coords -= old_ori[:, None, :]
                cmx = old_mx @ mx

                coords = np.einsum("ijk,ikl->ijl", coords, cmx) + ori

            return coords, weights

        if hasattr(self, "internal_coords"):
            returnables = self._off_rotamer_sample(idx, off_rotamer, **kwargs)
            return [
                np.squeeze(x) if isinstance(x, np.ndarray) else x for x in returnables
            ]

        elif len(self.dihedral_atoms) == 0:
            raise RuntimeError(
                "This ensemble has no rotatable bonds and therefore can't sample new conformations"
            )
        else:
            raise AttributeError(
                f"{type(self)} objects require both internal coordinates ({type(self)}.internal_coords "
                f"attribute) and dihedral standard deviations ({type(self)}.sigmas attribute) to "
                f"perform off rotamer sampling"
            )

    def _off_rotamer_sample(self, idx, off_rotamer, **kwargs):
        """Perform off rotamer sampling. Primarily a helper function for :meth:`~LigandEnsemble.sample`

        Parameters
        ----------
        idx : int
             Index of the rotamer of the rotamer library that will be perturbed.
        off_rotamer : List[bool]
             A list indicates a per-dihedral bools indicating which dihedral angles of ``self.dihedral_atoms`` will
             have off rotamer sampling. ``self.dihedral_atoms`` e.g. ``off_rotamer = [False, False, False, True,
             True]`` will only sample the last two mobile dihedral angles.
        **kwargs : dict
            return_dihedrals : bool
                If True, sample will return a :class:`~MolSysIC` object of the sampled rotamer

        Returns
        -------
        coords : ArrayLike
            The 3D coordinates of the sampled rotamer(s)
        new_weights : ArrayLike
            New weights (relative) of the sampled rotamer(s)
        ICs : List[chilife.MolSysIC] (Optional)
            Internal coordinate (:class:`~MolSysIC`) objects of the rotamer(s).
        """

        # Use accessible volume sampling if only provided a single rotamer
        if len(self._weights) == 1 or np.all(np.isinf(self.sigmas)):
            new_dihedrals = np.random.random((len(idx), len(off_rotamer))) * 2 * np.pi
            new_weights = np.ones(len(idx))

        #  sample from von mises near rotamer unless more information is provided
        elif self.skews is None:
            new_dihedrals = np.random.vonmises(
                self._rdihedrals[idx][:, off_rotamer],
                self._rkappas[idx][:, off_rotamer],
            )
            new_weights = np.ones(len(idx))
        # Sample from skewednorm if skews are provided
        # else:
        #     deltas = skewnorm.rvs(a=self.skews[idx], loc=self.locs[idx], scale=self.sigmas[idx])
        #     pdf = skewnorm.pdf(deltas, a=self.skews[idx], loc=self.locs[idx], scale=self.sigmas[idx])
        #     new_dihedrals = np.deg2rad(self.dihedrals[idx] + deltas)
        #     new_weight = pdf.prod()

        else:
            new_weights = np.ones(len(idx))

        new_weights = self._weights[idx] * new_weights

        z_matrix = self._lib_IC.batch_set_dihedrals(
            idx, new_dihedrals, 1, self.dihedral_atoms[off_rotamer]
        )
        fit_coords = self._lib_coords[idx]

        ICs = self.internal_coords.copy()
        ICs.load_new(z_matrix)
        coords = ICs.protein.trajectory.coordinate_array[:, self.ic_mask]

        # Realign new conformer with old conformer with basic RMSD alignment
        if self.alignment_method is None:
            ic_ori = coords.mean(axis=1)
            mx, ori = linear_alignment(coords, fit_coords)
            coords = (
                ICs.protein.trajectory.coordinate_array - ic_ori[:, None, ...]
            ) @ mx + ori

            nori = ori - np.einsum("ij,ijk->ik", ic_ori, mx)

            ICs._chain_operators = [{0: {"mx": m, "ori": o}} for m, o in zip(mx, nori)]
            ICs.protein.trajectory.coordinate_array = coords
            coords = coords[:, self.ic_mask]

        if kwargs.setdefault("remove_clashing", 2):
            clash_dist = (
                2
                if isinstance(kwargs["remove_clashing"], bool)
                else kwargs["remove_clashing"]
            )
            # Remove structures with internal clashes
            dist = np.linalg.norm(coords[:, self.aidx] - coords[:, self.bidx], axis=2)
            sidx = np.atleast_1d(
                np.squeeze(np.argwhere(np.all(dist > clash_dist, axis=1)))
            )

            ICs.use_frames(sidx)
            coords = coords[sidx]
            new_weights = new_weights[sidx]

        if kwargs.setdefault("return_dihedrals", False):
            return coords, new_weights, ICs
        else:
            return coords, new_weights

    def evaluate(self):
        """Place rotamer ensemble on protein site and recalculate rotamer weights."""

        # Calculate external energies
        energies = self.energy_func(self)

        # Calculate total weights (combining internal and external)
        self.weights, self.partition = scoring.reweight_rotamers(
            energies, self.temp, self.weights
        )

        # Remove low-weight rotamers from ensemble
        if self._do_trim:
            self.trim_rotamers()

    def trim_rotamers(self, keep_idx: ArrayLike = None) -> None:
        """
        Remove rotamers with small weights from ensemble

        Parameters
        ----------
        keep_idx : ArrayLike[int]
            Indices of rotamers to keep. If None rotamers will be trimmed based off of ``self.trim_tol``
        """
        if keep_idx is None:
            arg_sort_weights = np.argsort(self.weights)[::-1]
            sorted_weights = self.weights[arg_sort_weights]
            cumulative_weights = np.cumsum(sorted_weights)
            cutoff = np.maximum(
                1, len(cumulative_weights[cumulative_weights < 1 - self.trim_tol])
            )
            keep_idx = arg_sort_weights[:cutoff]

        if len(self.weights) == len(self.internal_coords):
            self.internal_coords = self.internal_coords.copy()
            self.internal_coords.use_frames(keep_idx)

        self._coords = self._coords[keep_idx]
        self._dihedrals = self._dihedrals[keep_idx]
        self.weights = self.weights[keep_idx]
        if self.atom_energies is not None:
            self.atom_energies = self.atom_energies[keep_idx]

        # normalize weights
        self.weights /= self.weights.sum()

    def to_rotlib(
        self,
        libname: str = None,
        description: str = None,
        comment: str = None,
        reference: str = None,
    ) -> None:
        """
        Saves the current :class:`~LigandEnsemble` as a rotamer library file.

        Parameters
        ----------
        libname : str
            Name of the rotamer library.
        description : str
            Description of the rotamer library.
        comment : str
            Additional comments about the rotamer library.
        reference : str
            References associated with the rotamer library.
        """
        if libname is None:
            libname = self.name.lstrip("0123456789.- ")

        if description is None:
            description = (
                "Rotamer library made with chiLife using `to_rotlib` method"
                "of a LigandEnsemble."
            )

        ICs = self.internal_coords.copy()

        # Remove chain operators to align all labels on backbone
        coords = ICs.protein.trajectory.coordinate_array

        lib = {
            "rotlib": libname,
            "resname": self.res,
            "coords": coords,
            "internal_coords": ICs,
            "weights": self.weights,
            "atom_types": self.atom_types.copy(),
            "atom_names": self.atom_names.copy(),
            "dihedral_atoms": self.dihedral_atoms,
            "dihedrals": self.dihedrals,
            "sigmas": self.sigmas,
            "type": "chilife rotamer library",
            "description": description,
            "comment": comment,
            "reference": reference,
            "format_version": 1.4,
        }

        if hasattr(self, "spin_atoms"):
            lib["spin_atoms"] = self.spin_atoms
            lib["spin_weights"] = self.spin_weights

        np.savez(Path().cwd() / f"{libname}_rotlib.npz", **lib, allow_pickle=True)

    def __eq__(self, other):
        """Equivalence measurement between a spin label and another object. A SpinLabel cannot be equivalent to a
        non-SpinLabel Two SpinLabels are considered equivalent if they have the same coordinates and weights"""

        if not isinstance(other, LigandEnsemble):
            return False
        elif self._coords.shape != other._coords.shape:
            return False

        return np.all(np.isclose(self._coords, other._coords)) and np.all(
            np.isclose(self.weights, other.weights)
        )

    def set_site(self, site):
        """Assign ``LigandEnsemble`` to a new residue number or "site". If there is an associated protein/molecular
        system the ``LigandEnsemble`` will be moved to that site. If not, only the LigandEnsemble residue number will
        be changed.

        Parameters
        ----------
        site : int
            Residue number to assign to the ``LigandEnsemble``.
        """
        self.site = site
        self.to_site()

    def copy(self, site: int = None, chain: str = None, rotlib: dict = None):
        """Create a deep copy of the ``LigandEnsemble`` object. Assign new site, chain or rotlib  if desired. Useful
        when labeling homo-oligomers or proteins with repeating units/domains.

        Parameters
        ----------
        site : int
             New site number for the copy.
        chain : str
             New chain identifier for the copy.
        rotlib : dict
            New (base) rotamer library for the dict.

        Returns
        -------
        new_copy : chilife.LigandEnsemble
            A deep copy of the original RotamerLibrary
        """
        new_copy = self._base_copy(self._rotlib)
        for item in self.__dict__:
            if isinstance(item, np.ndarray) or item == "internal_coords":
                new_copy.__dict__[item] = self.__dict__[item].copy()
            elif item not in ("_protein", "_lib_IC", "_rotlib"):
                new_copy.__dict__[item] = deepcopy(self.__dict__[item])
            elif self.__dict__[item] is None:
                new_copy.protein = None
            else:
                new_copy.__dict__[item] = self.__dict__[item]

        if site is not None:
            new_copy.site = site
        if chain is not None:
            new_copy.chain = chain
        return new_copy

    def _base_copy(self, rotlib=None):
        """
        Helper function for :meth:`~LigandEnsemble.copy`. Creates a copy of the current LigandEnsemble object with only
        residue, site, chain, and rotlib information.

        Parameters
        ----------
        rotlib: str | Path
            The name or ``Path`` object specifying the new rotamer library.

        Returns
        -------
        LigandEnsemble
            The new base copy of the ``LigandEnsemble`` object.

        """
        return LigandEnsemble(self.res, self.site, rotlib=rotlib, chain=self.chain)

    def to_site(self, target_pos: ArrayLike = None) -> None:
        """
        Method that performs the actual placement of the ``LigandEnsemble`` at a given site. Alignment will be
        performed using the alignment method specified during object construction.

        Parameters
        ----------
        target_pos: ArrayLike
            Optional array of points that specifies the site. If not provided, the ``self.site`` residue of the
            associated protein/molecular system will be used.
        """

        if self.alignment_method is None:
            return None

        if target_pos is None:
            target_pos = self.protein.select_atoms(
                f"resid {self.site} and segid {self.chain} and not altloc B"
            ).positions

        mx, ori = self.alignment_method(target_pos)

        old_mx, old_ori = self.alignment_method(self.coords)
        old_ori, old_mx = old_ori, old_mx.transpose(0, 2, 1)

        self._coords -= old_ori[:, None, :]
        cmx = old_mx @ mx

        self._coords = np.einsum("ijk,ikl->ijl", self._coords, cmx) + ori
        self.ICs_to_site()

    def ICs_to_site(self):
        """Modify the internal coordinates to be aligned with the site that the LigandEnsemble is attached to"""

        mx, ori = self.alignment_method(
            self.internal_coords.protein.trajectory.coordinate_array
        )

        self.ic_ori, self.ic_mx = ori, np.swapaxes(mx, -1, -2)

        new_mxs = self.ic_mx @ self.mx

        new_ori = np.einsum("ij,ikj->ij", -self.ic_ori, new_mxs) + self.origin
        op = [{0: {"mx": m, "ori": o}} for m, o in zip(new_mxs, new_ori)]

        self.internal_coords.chain_operators = op

    def get_lib(self, rotlib):
        """
        Helper function to find the rotlib and load it into memory.

        Parameters
        ----------
        rotlib : str | Path
            The rotlib name or file location.

        Returns
        -------
        lib : dict
            The rotamer library data as a dictionary.
        """
        was_none = True if rotlib is None else False
        rotlib = self.res if rotlib is None else rotlib

        # If the rotlib isn't a dict try to figure out where it is
        if not isinstance(rotlib, dict):
            trotlib = io.get_possible_rotlibs(
                rotlib, suffix="rotlib", extension=".npz", was_none=was_none
            )
            if trotlib:
                rotlib = trotlib
            else:
                raise NameError(
                    f"There is no rotamer library called {rotlib} in this directory or in chilife"
                )

            lib = io.read_library(rotlib, None, None)

        # If rotlib is a dict, assume that dict is in the format that read_library`would return
        else:
            lib = rotlib

        # Copy mutable library values (mostly np arrays) so that instances do not share data
        lib = {
            key: value.copy() if hasattr(value, "copy") else value
            for key, value in lib.items()
        }

        # Modify library to be appropriately used with self.__dict__.update
        self._rotlib = {
            key: value.copy()
            if hasattr(value, "copy") and key != "internal_coords"
            else value
            for key, value in lib.items()
        }

        lib["_coords"] = lib.pop("coords").copy()
        lib["_dihedrals"] = lib.pop("dihedrals").copy()
        if "skews" not in lib:
            lib["skews"] = None

        return lib

    def set_dihedral_sampling_sigmas(self, value):
        """
        Helper function to assign new sigmas for off-rotamer sampling.

        Parameters
        ----------
        value : float | ArrayLike[float]
            The sigma value(s) to assign. A single float will assign all dihedrals of all rotamers to the same sigma
            (isotropic). An array with as many elements as dihedrals will assign different sigmas to each dihedral
            (Anisotropic). An array of dimensions n_rots by n_dihedrals will assign different anisotropic sigmas to
            different rotamers.
        """
        value = np.asarray(value)

        if value.shape == ():
            self.sigmas = (
                np.ones((len(self._weights), len(self.dihedral_atoms))) * value
            )
        elif len(value) == len(self.dihedral_atoms) and len(value.shape) == 1:
            self.sigmas = np.tile(value, (len(self._weights), 1))
        elif value.shape == (len(self._weights), len(self.dihedral_atoms)):
            self.sigmas = value.copy()
        else:
            raise ValueError(
                "`dihedral_sigmas` must be a scalar, an array the length of the `self.dihedral atoms` or "
                "an array with the shape of (len(self.weights), len(self.dihedral_atoms))"
            )

    @property
    def bonds(self):
        """Set of atom indices corresponding to the atoms that form covalent bonds"""

        if not hasattr(self, "_bonds"):
            icmask_map = {x: i for i, x in enumerate(self.ic_mask)}
            self._bonds = np.array(
                [
                    (icmask_map[a], icmask_map[b])
                    for a, b in self.internal_coords.bonds
                    if a in icmask_map and b in icmask_map
                ]
            )

        return self._bonds

    @bonds.setter
    def bonds(self, inp):
        self._bonds = set(tuple(i) for i in inp)
        idxs = np.arange(len(self.atom_names))
        all_pairs = set(combinations(idxs, 2))
        self._non_bonded = all_pairs - self._bonds

    @property
    def non_bonded(self):
        """Set of atom indices corresponding to the atoms that are not covalently bonded. Also excludes atoms that
        have 1-n non-bonded interactions where ``n=self._exclude_nb_interactions`` . By default, 1-3 interactions are
        excluded"""
        if not hasattr(self, "_non_bonded"):
            pairs = {
                v.index: [
                    path
                    for path in self._graph.get_all_shortest_paths(v)
                    if len(path) <= (self._exclude_nb_interactions)
                ]
                for v in self._graph.vs
            }
            pairs = {(a, c) for a in pairs for b in pairs[a] for c in b if a < c}
            all_pairs = set(combinations(range(len(self.atom_names)), 2))
            self._non_bonded = all_pairs - pairs

        return sorted(list(self._non_bonded))

    @non_bonded.setter
    def non_bonded(self, inp):
        self._non_bonded = set(tuple(i) for i in inp)
        idxs = np.arange(len(self.atom_names))
        all_pairs = set(combinations(idxs, 2))
        self._bonds = all_pairs - self._non_bonded

    def __len__(self):
        """Number of rotamers in the ensemble"""
        return len(self.coords)

    @property
    def clash_ori(self):
        """Location to use as 'center' when searching for atoms with potential clashes"""
        if isinstance(self._clash_ori_inp, (np.ndarray, list)):
            if len(self._clash_ori_inp) == 3:
                return self._clash_ori_inp
        elif isinstance(self._clash_ori_inp, str):
            if self._clash_ori_inp in ["cen", "centroid"]:
                return self.centroid
            elif (ori_name := self._clash_ori_inp.upper()) in self.atom_names:
                return np.squeeze(self._coords[0][ori_name == self.atom_names])
        else:
            raise ValueError(
                f"Unrecognized clash_ori option {self._clash_ori_inp}. Please specify a 3D vector, an "
                f"atom name or `centroid`"
            )

        return self._clash_ori

    @clash_ori.setter
    def clash_ori(self, inp):
        self._clash_ori_inp = inp

    @property
    def centroid(self):
        """Get the centroid of the whole ensemble."""
        return self._coords.mean(axis=(0, 1))

    @property
    def coords(self):
        """Cartesian coordinates of all rotamers in the ensemble."""
        return self._coords

    @coords.setter
    def coords(self, coords):
        # TODO: Add warning about removing isomers

        # Allow users to input a single rotamer
        coords = coords if coords.ndim == 3 else coords[None, :, :]

        # Check if atoms match
        if coords.shape[1] == len(self.side_chain_idx):
            tmp = np.array([self._coords[0].copy() for _ in coords])
            tmp[:, self.side_chain_idx] = coords
            coords = tmp

        if coords.shape[1] != len(self.atoms):
            raise ValueError(
                "The number of atoms in the input array does not match the number of atoms of the residue"
            )

        self._coords = coords
        self.internal_coords = self.internal_coords.copy()
        self.internal_coords.set_cartesian_coords(coords, self.ic_mask)

        # Check if they are all at the same site
        self._dihedrals = np.rad2deg(
            [ic.get_dihedral(1, self.dihedral_atoms) for ic in self.internal_coords]
        )

        if len(self.dihedral_atoms) == 1:
            self._dihedrals = self.dihedrals[:, None]

        # Apply uniform weights
        self.weights = np.ones(len(self._dihedrals))
        self.weights /= self.weights.sum()

    @property
    def dihedrals(self):
        """Values of the (mobile) dihedral angles of all rotamers in the ensemble"""
        return self._dihedrals

    @dihedrals.setter
    def dihedrals(self, dihedrals):
        dihedrals = np.asarray(dihedrals)
        dihedrals = dihedrals if dihedrals.ndim == 2 else dihedrals[None, :]
        if dihedrals.shape[1] != self.dihedrals.shape[1]:
            raise ValueError(
                "The input array does not have the correct number of dihedrals"
            )

        if len(dihedrals) != self.dihedrals.shape[0]:
            warnings.warn(
                "WARNING: Setting dihedrals in this fashion will set all bond lengths and angles to that "
                "of the first rotamer in the library effectively removing stereo-isomers from the ensemble. It "
                "will also set all weights to ."
            )

            # Apply uniform weights
            self.weights = np.ones(len(dihedrals))
            self.weights /= self.weights.sum()
            idxs = [0 for _ in range(len(dihedrals))]
        else:
            idxs = np.arange(len(dihedrals))

        self._dihedrals = dihedrals
        z_matrix = self.internal_coords.batch_set_dihedrals(
            idxs, np.deg2rad(dihedrals), 1, self.dihedral_atoms
        )
        self.internal_coords = self.internal_coords.copy()
        self.internal_coords.load_new(z_matrix)
        self._coords = self.internal_coords.protein.trajectory.coordinate_array.copy()[
            :, self.ic_mask
        ]

    @property
    def mx(self):
        """The rotation matrix to rotate a residue from the local coordinate frame to the current residue. The local
        coordinate frame is defined by the alignment method"""
        if self.alignment_method is None:
            mx, ori = linear_alignment(
                self.internal_coords.protein.trajectory.coordinate_array[
                    :, self.ic_mask
                ],
                self.coords,
            )
        else:
            mx, ori = self.alignment_method(self.coords)
        return mx

    @property
    def origin(self):
        """Origin of the local coordinate frame"""

        if self.alignment_method is None:
            mx, ori = linear_alignment(
                self.internal_coords.protein.trajectory.coordinate_array[
                    :, self.ic_mask
                ],
                self.coords,
            )
        else:
            mx, ori = self.alignment_method(self.coords)

        return ori

    @property
    def protein(self):
        return self._protein

    @protein.setter
    def protein(self, protein):
        if hasattr(protein, "atoms") and isinstance(
            protein.atoms, (mda.AtomGroup, MolecularSystemBase)
        ):
            self._protein = protein
            self.init_protein()
            self.nataa = self.protein.select_atoms(self.selstr)[0].resname

        elif protein is None:
            self._protein = protein
        else:
            raise ValueError(
                "The input protein must be an instance of MDAnalysis.Universe, MDAnalysis.AtomGroup, or "
                "chilife.MolSys"
            )


def assign_defaults(kwargs):
    """
    Helper function to assign default values to kwargs that have not been explicitly assigned by the user. Also
    checks to make sure that the user provided kwargs are real kwargs.
    Parameters
    ----------
    kwargs : dict
        Dictionary of user supplied keyword arguments.

    Returns
    -------
    kwargs : dict
        Dictionary of user supplied keyword arguments augmented with defaults for all kwargs the user did not supply.
    """

    # Make all string arguments lowercase
    for key, value in kwargs.items():
        if isinstance(value, str):
            kwargs[key] = value.lower()

    # Default parameters
    defaults = {
        "protein_tree": None,
        "temp": 298,
        "clash_radius": None,
        "_clash_ori_inp": kwargs.pop("clash_ori", "cen"),
        "alignment_method": "svd",
        "dihedral_sigmas": 35,
        "weighted_sampling": False,
        "eval_clash": True if not kwargs.get("minimize", False) else False,
        "use_H": True,
        "_exclude_nb_interactions": kwargs.pop("exclude_nb_interactions", 3),
        "_sample_size": kwargs.pop("sample", False),
        "energy_func": default_energy_func,
        "_minimize": kwargs.pop("minimize", False),
        "min_method": "L-BFGS-B",
        "_do_trim": kwargs.pop("trim", True),
        "trim_tol": 0.005,
        "ignore_waters": True,
    }

    # Overwrite defaults
    kwargs_filled = {**defaults, **kwargs}

    # Make sure there are no unused parameters
    if len(kwargs_filled) != len(defaults):
        raise TypeError(
            f"Got unexpected keyword argument(s): "
            f'{", ".join(key for key in kwargs if key not in defaults)}'
        )

    return kwargs_filled.items()
