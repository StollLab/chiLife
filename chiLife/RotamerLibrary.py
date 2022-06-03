from copy import deepcopy
import pickle
import logging
import numpy as np
from itertools import combinations
from scipy.spatial import cKDTree
import scipy.optimize as opt
import MDAnalysis as mda
import chiLife


class RotamerLibrary:

    backbone_atoms = ["H", "N", "CA", "HA", "C", "O"]

    def __init__(self, res, site=1, protein=None, chain=None, **kwargs):
        """
        Create new RotamerLibrary object.

        :param res: string
            Name of desired residue, e.g. R1A.
        :param site: int
            Protein residue number to attach library to.
        :param protein: MDAnalysis.Universe, MDAnalysis.AtomGroup
            Object containing all protein information (coords, atom types, etc.)
        :param chain: str
            Protein chain identifier to attach spin label to.
        """

        self.res = res
        self.protein = protein
        self.site = int(site)
        self.chain = chain if chain is not None else guess_chain(self.protein, self.site)
        self.selstr = f"resid {self.site} and segid {self.chain} and not altloc B"
        self.__dict__.update(assign_defaults(kwargs))

        lib = self.get_lib()
        self.__dict__.update(lib)
        self._weights = self.weights / self.weights.sum()

        if len(self.sigmas) == 0:
            self.sigmas = (
                np.ones((len(self._weights), len(self.dihedral_atoms)))
                * self.dihedral_sigma
            )

        self.sigmas[self.sigmas == 0] = self.dihedral_sigma
        self._rsigmas = np.deg2rad(self.sigmas)
        self._rkappas = 1 / self._rsigmas**2
        self.ic_mask = self.atom_names != "Z"

        # Remove hydrogen atoms unless otherwise specified
        if not self.use_H:
            self.H_mask = self.atom_types != "H"
            self.ic_mask *= self.H_mask
            self._coords = self._coords[:, self.H_mask]
            self.atom_types = self.atom_types[self.H_mask]
            self.atom_names = self.atom_names[self.H_mask]

        # Parse important indices
        self.backbone_idx = np.argwhere(np.isin(self.atom_names, ["N", "CA", "C"]))
        self.side_chain_idx = np.argwhere(
            np.isin(self.atom_names, RotamerLibrary.backbone_atoms, invert=True)
        ).flatten()

        if self.sample_size and len(self.dihedral_atoms) > 0:
            # Get list of non-bonded atoms before overwritig
            a, b = [list(x) for x in zip(*self.non_bonded)]

            # Perform sapling
            self._coords = np.tile(self._coords[0], (self.sample_size, 1, 1))
            self._coords, self.weights, self.internal_coords = self.sample(
                self.sample_size, off_rotamer=True, return_dihedrals=True
            )

            self._dihedrals = np.asarray(
                [IC.get_dihedral(1, self.dihedral_atoms) for IC in self.internal_coords]
            )
            # Remove structures with internal clashes
            dist = np.linalg.norm(self._coords[:, a] - self._coords[:, b], axis=2)
            sidx = np.atleast_1d(np.squeeze(np.argwhere(np.all(dist > 2, axis=1))))
            self.internal_coords = self.internal_coords[sidx]
            self._dihedrals = np.rad2deg(self._dihedrals[sidx])
            self._coords, self.weights = self._coords[sidx], self.weights[sidx]

        # Allocate variables for clash evaluations
        self.atom_energies = None
        self.clash_ignore_coords = None
        self.clash_ignore_idx = None
        self.partition = 1

        # Assign a name to the label
        self.name = self.res
        if self.site is not None:
            self.name = str(self.site) + self.res
        if self.chain is not None:
            self.name += f"_{self.chain}"

        # Create arrays of  LJ potential params
        if len(self.side_chain_idx) > 0:
            self.rmin2 = chiLife.get_lj_rmin(self.atom_types[self.side_chain_idx])
            self.eps = chiLife.get_lj_eps(self.atom_types[self.side_chain_idx])

        if hasattr(self.protein, "atoms") and isinstance(
            self.protein.atoms, mda.AtomGroup
        ):
            self.protein_setup()

        # Store atom information as atom objects
        self.atoms = [
            chiLife.Atom(name, atype, idx, self.res, self.site, coord)
            for idx, (coord, atype, name) in enumerate(
                zip(self._coords[0], self.atom_types, self.atom_names)
            )
        ]

    @classmethod
    def from_pdb(
        cls,
        pdb_file,
        res=None,
        site=None,
        protein=None,
        chain=None,
    ):
        # TODO: create pdb reader and impliment from_pdb method
        return cls(res, site, protein, chain)

    @classmethod
    def from_mda(cls, residue, **kwargs):
        """
        Create RotamerLibrary from the MDAnalysis.Residue
        :param residue: MDAnalysis.Residue

        :return: RotamerLibrary
        """

        res = residue.resname
        site = residue.resnum
        chain = residue.segid
        protein = residue.universe
        return cls(res, site, protein, chain, **kwargs)

    @classmethod
    def from_trajectory(
        cls, traj, site, chain=None, energy=None, burn_in=100, **kwargs
    ):
        chain = guess_chain(traj, site) if chain is None else chain

        if hasattr(traj.universe._topology, "altLocs"):
            res = traj.select_atoms(f"segid {chain} and resnum {site} and not altloc B")
        else:
            res = traj.select_atoms(f"segid {chain} and resnum {site}")

        resname = res.residues[0].resname

        coords = []
        for ts in traj.universe.trajectory[burn_in:]:
            coords.append(res.atoms.positions)
        coords = np.array(coords)

        _, unique_idx, non_unique_idx = np.unique(
            coords, axis=0, return_inverse=True, return_index=True
        )
        coords = coords[unique_idx]

        prelib = cls(
            resname, site, chain=chain, protein=traj, eval_clash=False, **kwargs
        )
        prelib._coords = np.atleast_3d(coords)

        if energy is not None:
            energy = energy[burn_in:]  # - energy[burn_in]
            energy = np.array(
                [energy[non_unique_idx == idx].mean() for idx in range(len(unique_idx))]
            )
            T = kwargs.setdefault("T", 1)
            pi = np.exp(-energy / (chiLife.GAS_CONST * T))
            pi /= pi.sum()
        else:
            pi = np.ones(len(traj.universe.trajectory[burn_in:]))
            pi = np.array(
                [pi[non_unique_idx == idx].sum() for idx in range(len(unique_idx))]
            )
            pi /= pi.sum()

        prelib.weights = pi

        dihedrals = []
        masks = [np.isin(prelib.atom_names, ddef) for ddef in prelib.dihedral_atoms]
        for i in range(len(prelib.weights)):
            dihedrals.append([chiLife.get_dihedral(coords[i][mask]) for mask in masks])

        dihedrals = np.rad2deg(np.array(dihedrals))
        prelib._dihedrals = dihedrals
        prelib.backbone_to_site()

        return prelib

    def __eq__(self, other):
        """Equivalence measurement between a spin alebl and another object. A SpinLabel cannot be equivalent to a
        non-SpinLabel Two SpinLabels are considered equivalent if they have the same coordinates and weights
        TODO: Future consideration -- different orders (argsort weights)"""

        if not isinstance(other, RotamerLibrary):
            return False
        elif self._coords.shape != other._coords.shape:
            return False

        return np.all(np.isclose(self._coords, other._coords)) and np.all(
            np.isclose(self.weights, other.weights)
        )

    def set_site(self, site):
        """
        Assign SpinLabel to a site.
        :param site: int
            residue number to assign SpinLabel to.
        """
        self.site = site
        self.to_site()

    def copy(self, site=None, chain=None):
        """Create a deep copy of the spin label. Assign new site and chain information if desired"""
        new_copy = deepcopy(self)
        if site is not None:
            new_copy.site = site
        if chain is not None:
            new_copy.chain = chain
        return new_copy

    def __deepcopy__(self, memodict={}):
        new_copy = chiLife.RotamerLibrary(self.res, self.site)
        for item in self.__dict__:
            if item != "protein":
                new_copy.__dict__[item] = deepcopy(self.__dict__[item])
            elif self.__dict__[item] is None:
                new_copy.protein = None
            else:
                new_copy.protein = self.protein.copy()
        return new_copy

    def to_site(self, site_pos=None):
        """Move spin label to new site
        :param site_pos: array-like
            3x3 array of ordered backbone atom coordinates of new site (N CA C)
        """
        if site_pos is None:
            N, CA, C = chiLife.parse_backbone(self, kind="global")
        else:
            N, CA, C = site_pos

        mx, ori = chiLife.global_mx(N, CA, C, method=self.superimposition_method)

        # if self.superimposition_method not in {'fit', 'rosetta'}:
        N, CA, C = chiLife.parse_backbone(self, kind="local")
        old_ori, ori_mx = chiLife.local_mx(N, CA, C, method=self.superimposition_method)
        self._coords -= old_ori
        mx = ori_mx @ mx

        self._coords = np.einsum("ijk,kl->ijl", self._coords, mx) + ori

        self.ICs_to_site()

    def ICs_to_site(self):
        ic_backbone = np.squeeze(self.internal_coords[0].coords[:3])
        self.ic_ori, self.ic_mx = chiLife.local_mx(
            *ic_backbone, method=self.superimposition_method
        )
        m2m3 = self.ic_mx @ self.mx
        op = {}
        for segid in self.internal_coords[0].chain_operators:
            new_mx = self.internal_coords[0].chain_operators[segid]["mx"] @ m2m3
            new_ori = (
                self.internal_coords[0].chain_operators[segid]["ori"] - self.ic_ori
            ) @ m2m3 + self.ori
            op[segid] = {"mx": new_mx, "ori": new_ori}

        for IC in self.internal_coords:
            IC.chain_operators = op

    def backbone_to_site(self):
        # Keep protein backbone dihedrals for oxygen and hydrogens
        for atom in ["H", "O"]:
            mask = self.atom_names == atom
            if any(mask) and self.protein is not None:
                pos = self.protein.select_atoms(
                    f"segid {self.chain} and resnum {self.site} "
                    f"and name {atom} and not altloc B"
                ).positions
                if len(pos) > 0:
                    self._coords[:, mask] = pos[0]

    def sample(self, n=1, off_rotamer=False, **kwargs):
        """Randomly sample a rotamer in the library."""
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
            return np.squeeze(self._coords[idx]), np.squeeze(self.weights[idx])

        if len(self.dihedral_atoms) == 0:
            return self._coords, self.weights, self.internal_coords
        elif hasattr(self, "internal_coords"):
            returnables = zip(
                *[self._off_rotamer_sample(iidx, off_rotamer, **kwargs) for iidx in idx]
            )
            return [np.squeeze(x) for x in returnables]

        else:
            raise AttributeError(
                f"{type(self)} objects require both internal coordinates ({type(self)}.internal_coords "
                f"attribute) and dihedral standard deviations ({type(self)}.sigmas attribute) to "
                f"perform off rotamer sampling"
            )

    def _off_rotamer_sample(self, idx, off_rotamer, **kwargs):
        """

        :param idx:
        :return:
        """
        new_weight = 0
        if len(self.sigmas) > 0:
            new_dihedrals = np.random.vonmises(
                self._rdihedrals[idx, off_rotamer], self._rkappas[idx, off_rotamer]
            )
            # diff1 = (new_dihedrals - self._rdihedrals[idx, off_rotamer]) * self._rkappas[idx, off_rotamer]
            # diff = (diff1 @ diff1) / len(diff1)
            new_weight = self._weights[idx]  # * np.exp(-0.5 * 5e-2 * diff)
        else:
            new_dihedrals = np.random.random(len(off_rotamer)) * 2 * np.pi
            new_weight = 1.0

        internal_coord = self.internal_coords[idx].copy().set_dihedral(
            new_dihedrals, 1, self.dihedral_atoms[off_rotamer]
        )

        coords = internal_coord.to_cartesian()
        coords = coords[self.ic_mask]

        if kwargs.setdefault("return_dihedrals", False):
            return coords, new_weight, internal_coord
        else:
            return coords, new_weight

    def update_weight(self, weight):
        self.current_weight = weight

    def minimize(self):
        def objective(dihedrals, ic):
            coords = ic.set_dihedral(dihedrals, 1, self.dihedral_atoms).to_cartesian()
            temp_rotlib._coords = np.atleast_3d([coords[self.ic_mask]])
            return self.energy_func(temp_rotlib.protein, temp_rotlib)

        temp_rotlib = self.copy()
        for i, IC in enumerate(self.internal_coords):
            d0 = IC.get_dihedral(1, self.dihedral_atoms)
            lb = d0 - np.deg2rad(self.dihedral_sigma) / 2
            ub = d0 + np.deg2rad(self.dihedral_sigma) / 2
            bounds = np.c_[lb, ub]

            xopt = opt.minimize(
                objective, x0=self._rdihedrals[i], args=IC, bounds=bounds
            )

            self._coords[i] = IC.to_cartesian()[self.ic_mask]
            self.weights[i] = self.weights[i] * np.exp(
                -xopt.fun / (chiLife.GAS_CONST * 298)
            )

        self.weights /= self.weights.sum()
        self.trim()

    def trim(self, tol=0.005):
        """Remove insignificant rotamers from Library"""
        old_len = len(self.weights)
        arg_sort_weights = np.argsort(self.weights)[::-1]
        sorted_weights = self.weights[arg_sort_weights]
        cumulative_weights = np.cumsum(sorted_weights)
        cutoff = np.maximum(1, len(cumulative_weights[cumulative_weights < 1 - tol]))

        self._coords = self._coords[arg_sort_weights[:cutoff]]
        self._dihedrals = self._dihedrals[arg_sort_weights[:cutoff]]
        self.weights = self.weights[arg_sort_weights[:cutoff]]
        if len(arg_sort_weights) == len(self.internal_coords):
            self.internal_coords = [
                self.internal_coords[x] for x in arg_sort_weights[:cutoff]
            ]
        self.weights /= self.weights.sum()

        logging.info(
            f"{len(self.weights)} of {old_len} {self.res} rotamers make up at least {100 * (1-tol):.2f}% of "
            f"rotamer density for site {self.site}"
        )

    def centroid(self):
        """get the centroid of the whole rotamer library."""
        return self._coords.mean(axis=(0, 1))

    def evaluate(self):
        """place spin label on protein site and recalculate rotamer weights"""

        if self.protein_tree is None:
            self.protein_tree = cKDTree(self.protein.atoms.positions)

        rotamer_energies = self.energy_func(self.protein, self)
        rotamer_probabilities = np.exp(
            -rotamer_energies / (chiLife.GAS_CONST * self.temp)
        )

        self.weights, self.partition = chiLife.reweight_rotamers(
            rotamer_probabilities, self.weights, return_partition=True
        )
        logging.info(f"Relative partition function: {self.partition:.3}")

        self.trim()

    def save_pdb(self, name=None):
        if name is None:
            name = self.name
        chiLife.save_rotlib(name, self.atoms, self._coords)

    @property
    def backbone(self):
        """Backbone coordinates of the spin label"""
        return np.squeeze(self._coords[0][self.backbone_idx])

    def get_lib(self):
        """
        Parse backbone information from protein and fetch the appropriate rotamer library

        :return:
        """

        PhiSel, PsiSel = None, None
        if self.protein is not None:
            # get site backbone information from protein structure
            PhiSel = (
                self.protein.select_atoms(f"resnum {self.site} and segid {self.chain}")
                .residues[0]
                .phi_selection()
            )
            PsiSel = (
                self.protein.select_atoms(f"resnum {self.site} and segid {self.chain}")
                .residues[0]
                .psi_selection()
            )

        # Default to helix backbone if none provided
        Phi = (
            None
            if PhiSel is None
            else np.rad2deg(chiLife.get_dihedral(PhiSel.positions))
        )
        Psi = (
            None
            if PsiSel is None
            else np.rad2deg(chiLife.get_dihedral(PsiSel.positions))
        )
        self.Phi, self.Psi = Phi, Psi
        # Get library
        logging.info(f"Using backbone dependent library with Phi={Phi}, Psi={Psi}")
        lib = chiLife.read_library(self.res, Phi, Psi)
        lib = {key: value.copy() for key, value in lib.items()}
        if 'internal_coords' in lib:
            lib['internal_coords'] = [a.copy() for a in lib['internal_coords']]
        lib['_coords'] = lib.pop('coords')
        lib['_dihedrals'] = lib.pop('dihedrals')
        return lib

    def protein_setup(self):
        # Position library at selected residue
        self.resindex = self.protein.select_atoms(self.selstr).residues[0].resindex
        self.segindex = self.protein.select_atoms(self.selstr).residues[0].segindex
        self.protein = self.protein.select_atoms("not (byres name OH2 or resname HOH)")

        if self.protein_tree is None:
            self.protein_tree = cKDTree(self.protein.atoms.positions)

        self.to_site()
        self.backbone_to_site()

        # Get weight of current or closest rotamer
        self.clash_ignore_idx = self.protein.select_atoms(
            f"resid {self.site} and segid {self.chain}"
        ).ix

        protein_clash_idx = self.protein_tree.query_ball_point(
            self.clash_ori, self.clash_radius
        )
        self.protein_clash_idx = [
            idx for idx in protein_clash_idx if idx not in self.clash_ignore_idx
        ]

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

        if self.eval_clash:
            self.evaluate()

    @property
    def bonds(self):
        if not hasattr(self, "_bonds"):
            bond_tree = cKDTree(self._coords[0])
            bonds = bond_tree.query_pairs(2.0)
            self._bonds = set(tuple(sorted(bond)) for bond in bonds)
        return list(sorted(self._bonds))

    @bonds.setter
    def bonds(self, inp):
        self._bonds = set(tuple(i) for i in inp)
        idxs = np.arange(len(self.atom_names))
        all_pairs = set(combinations(idxs, 2))
        self._non_bonded = all_pairs - self._bonds

    @property
    def non_bonded(self):
        if not hasattr(self, "_non_bonded"):
            idxs = np.arange(len(self.atom_names))
            all_pairs = set(combinations(idxs, 2))
            self._non_bonded = all_pairs - set(self.bonds)

        return sorted(list(self._non_bonded))

    @non_bonded.setter
    def non_bonded(self, inp):
        self._non_bonded = set(tuple(i) for i in inp)
        idxs = np.arange(len(self.atom_names))
        all_pairs = set(combinations(idxs, 2))
        self._bonds = all_pairs - self._non_bonded

    def __len__(self):
        return len(self.weights)

    @property
    def clash_ori(self):
        if isinstance(self._clash_ori_inp, (np.ndarray, list)):
            if len(self._clash_ori_inp) == 3:
                return self._clash_ori_inp
        elif isinstance(self._clash_ori_inp, str):
            if self._clash_ori_inp in ["cen", "centroid"]:
                return self.centroid()
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
    def coords(self):
        return self._coords

    @coords.setter
    def coords(self, coords):
        # Allow users to input a single rotamer
        value = coords if coords.ndim == 3 else coords[None, :, :]

        # Check if atoms match
        if coords.shape[1] == len(self.side_chain_idx):
            tmp = np.array([self._coords[0].copy() for _ in coords])
            tmp[:, self.side_chain_idx] = coords
            coords = tmp

        if coords.shape[1] != len(self.atoms):
            raise ValueError('The number of atoms in the input array does not match the number of atoms of the residue')

        self._coords = coords
        self.ICs_to_site()

        if coords.shape[1] != len(self.internal_coords[0].coords):
            tmp = np.array([np.empty_like(self.internal_coords[0].coords) for _ in coords])
            tmp[:] = np.nan
            tmp[:, self.ic_mask] = coords
            coords = tmp

        self.internal_coords = [self.internal_coords[0].copy() for _ in coords]
        for ic, val in zip(self.internal_coords, coords):
            ic.coords = val

        self._dihedrals = np.rad2deg(
            [IC.get_dihedral(1, self.dihedral_atoms) for IC in self.internal_coords]
        )

        # Apply uniform weights
        self.weights = np.ones(len(self._dihedrals))
        self.weights /= self.weights.sum()

    @property
    def dihedrals(self):
        return self._dihedrals

    @property
    def mx(self):
        mx, ori = chiLife.global_mx(
            *np.squeeze(self.backbone), method=self.superimposition_method
        )
        return mx

    @property
    def ori(self):
        return np.squeeze(self.backbone[1])


def assign_defaults(kwargs):

    # Make all string arguments lowercase
    for key, value in kwargs.items():
        if isinstance(value, str):
            kwargs[key] = value.lower()

    # Default parameters
    defaults = {
        "protein_tree": None,
        "forgive": 1.0,
        "temp": 298,
        "clash_radius": 14.0,
        "_clash_ori_inp": kwargs.pop("clash_ori", "cen"),
        "superimposition_method": "bisect",
        "dihedral_sigma": 35,
        "weighted_sampling": False,
        "eval_clash": False,
        "use_H": False,
        "sample_size": kwargs.pop("sample", False),
        "energy_func": chiLife.get_lj_rep,
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


def guess_chain(protein, site):
    if protein is None:
        chain = "A"
    elif len(set(protein.segments.segids)) == 1:
        chain = protein.segments.segids[0]
    elif np.isin(protein.residues.resnums, site).sum() == 0:
        raise ValueError(f"Residue {site} is not present on the provided protein")
    elif np.isin(protein.residues.resnums, site).sum() == 1:
        chain = protein.select_atoms(f"resid {site}").segids[0]
    else:
        raise ValueError(
            f"Residue {site} is present on more than one chain. Please specify the desired chain"
        )
    return chain
