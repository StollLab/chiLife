from copy import deepcopy
import logging
import numpy as np
from itertools import combinations
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
import scipy.optimize as opt
import MDAnalysis as mda
import chiLife


class RotamerLibrary:

    backbone_atoms = ['H', 'N', 'CA', 'HA', 'C', 'O']

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
        self.chain = chain if chain is not None else self.guess_chain()
        self.selstr = f'resid {self.site} and segid {self.chain} and not altloc B'
        self.__dict__.update(assign_defaults(kwargs))

        lib = self.get_lib()
        self.__dict__.update(lib)
        self._weights = self.weights / self.weights.sum()

        if len(self.sigmas) == 0:
            self.sigmas = np.ones((len(self._weights), len(self.dihedral_atoms))) * self.dihedral_sigma

        self.sigmas[self.sigmas == 0] = self.dihedral_sigma
        self._rsigmas = np.deg2rad(self.sigmas)
        self._rkappas = 1 / self._rsigmas ** 2
        self.ic_mask = self.atom_names != 'Z'

        # Remove hydrogen atoms unless otherwise specified
        if not self.use_H:
            self.H_mask = self.atom_types != 'H'
            self.ic_mask *= self.H_mask
            self.coords = self.coords[:, self.H_mask]
            self.atom_types = self.atom_types[self.H_mask]
            self.atom_names = self.atom_names[self.H_mask]

        # Parse important indices
        self.backbone_idx = np.argwhere(np.isin(self.atom_names, ['N', 'CA', 'C']))
        self.side_chain_idx = np.argwhere(np.isin(self.atom_names, RotamerLibrary.backbone_atoms, invert=True)).flatten()

        if self.sample_size and len(self.dihedral_atoms) > 0:
            # Get list of non-bonded atoms before overwritig
            a, b = [list(x) for x in zip(*self.non_bonded)]

            # Perform sapling
            self.coords = np.tile(self.coords[0], (self.sample_size, 1, 1))
            self.mx, self.ori = np.eye(3), np.array([0, 0, 0])
            self.coords, self.weights, self.internal_coords = self.sample(self.sample_size,
                                                                          off_rotamer=True,
                                                                          return_dihedrals=True)

            self.dihedrals = np.asarray([IC.get_dihedral(1, self.dihedral_atoms) for IC in self.internal_coords])
            # Remove structures with internal clashes
            dist = np.linalg.norm(self.coords[:, a] - self.coords[:, b], axis=2)
            sidx = np.atleast_1d(np.squeeze(np.argwhere(np.all(dist > 2, axis=1))))
            self.internal_coords = self.internal_coords[sidx]
            self.dihedrals = np.rad2deg(self.dihedrals[sidx])
            self.coords, self.weights = self.coords[sidx], self.weights[sidx]

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
            self.name += f'_{self.chain}'

        # Create arrays of  LJ potential params
        if len(self.side_chain_idx) > 0:
            self.rmin2 = chiLife.get_lj_rmin(self.atom_types[self.side_chain_idx])
            self.eps = chiLife.get_lj_eps(self.atom_types[self.side_chain_idx])

        if hasattr(self.protein, 'atoms') and isinstance(self.protein.atoms, mda.AtomGroup):
            self.protein_setup()

        # Store atom information as atom objects
        self.atoms = [chiLife.Atom(name, atype, idx, self.res, self.site, coord) for idx, (coord, atype, name) in
                      enumerate(zip(self.coords[0], self.atom_types, self.atom_names))]

        self.mx, self.ori = chiLife.global_mx(*np.squeeze(self.backbone), method=self.superimposition_method)

    @classmethod
    def from_pdb(cls, pdb_file, res=None, site=None, protein=None, chain=None,):
        #TODO: create pdb reader and impliment from_pdb method
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
    def from_trajectory(cls, traj, site, energy=None, burn_in=100,  **kwargs):

        if hasattr(traj.universe._topology, 'altLocs'):
            res = traj.select_atoms(f'resnum {site} and not altloc B')
        else:
            res = traj.select_atoms(f'resnum {site}')

        chain = res.atoms.segids[0]
        resname = res.residues[0].resname

        coords = []
        for ts in traj.universe.trajectory[burn_in:]:
             coords.append(res.atoms.positions)
        coords = np.array(coords)

        _, unique_idx, non_unique_idx = np.unique(coords, axis=0, return_inverse=True, return_index=True)
        coords = coords[unique_idx]

        prelib = cls(resname, site, chain=chain, protein=traj, **kwargs)
        prelib.coords = np.atleast_3d(coords)

        if energy is not None:
            energy = energy[burn_in:]  # - energy[burn_in]
            energy = np.array([energy[non_unique_idx == idx].mean() for idx in range(len(unique_idx))])
            T = kwargs.setdefault('T', 1)
            pi = np.exp(-energy / (chiLife.GAS_CONST * T))
            pi /= pi.sum()
        else:
            pi = np.ones(len(traj.universe.trajectory[burn_in:]))
            pi = np.array([pi[non_unique_idx == idx].sum() for idx in range(len(unique_idx))])
            pi /= pi.sum()

        prelib.weights = pi

        dihedrals = []
        masks = [np.isin(prelib.atom_names, ddef) for ddef in prelib.dihedral_atoms]
        for i in range(len(prelib.weights)):
            dihedrals.append([chiLife.get_dihedral(coords[i][mask]) for mask in masks])

        dihedrals = np.array(dihedrals)
        prelib.dihedrals = dihedrals

        return prelib

    def __eq__(self, other):
        """Equivalence measurement between a spin alebl and another object. A SpinLabel cannot be equivalent to a
        non-SpinLabel Two SpinLabels are considered equivalent if they have the same coordinates and weights
        TODO: Future consideration -- different orders (argsort weights)"""

        if not isinstance(other, RotamerLibrary):
            return False
        elif self.coords.shape != other.coords.shape:
            return False

        return np.all(np.isclose(self.coords, other.coords)) and np.all(np.isclose(self.weights, other.weights))

    def set_site(self, site):
        """
        Assign SpinLabel to a site.
        :param site: int
            residue number to assign SpinLabel to.
        """
        self.site = site
        self._to_site()

    def copy(self, site=None, chain=None):
        """Create a deep copy of the spin label. Assign new site and chain information if desired"""
        new_copy = deepcopy(self)
        if site is not None:
            new_copy.site=site
        if chain is not None:
            new_copy.chain = chain
        return new_copy

    def __deepcopy__(self, memodict={}):
        new_copy = chiLife.RotamerLibrary(self.res, self.site, self.chain)
        for item in self.__dict__:
            if item != 'protein':
                new_copy.__dict__[item] = deepcopy(self.__dict__[item])
            elif self.__dict__[item] is None:
                new_copy.protein = None
            else:
                new_copy.protein = self.protein.copy()
        return new_copy

    def _to_site(self, site_pos=None):
        """ Move spin label to new site
        :param site_pos: array-like
            3x3 array of ordered backbone atom coordinates of new site (N CA C)
        """
        if site_pos is None:
            N, CA, C = chiLife.parse_backbone(self, kind='global')
        else:
            N, CA, C = site_pos

        mx, ori = chiLife.global_mx(N, CA, C, method=self.superimposition_method)

        # if self.superimposition_method not in {'fit', 'rosetta'}:
        N, CA, C = chiLife.parse_backbone(self, kind='local')
        old_ori, ori_mx = chiLife.local_mx(N, CA, C, method=self.superimposition_method)
        self.coords -= old_ori
        mx = ori_mx @ mx

        self.coords = np.einsum('ijk,kl->ijl', self.coords, mx) + ori

        # Keep protein backbone dihedrals for oxygen and hydrogens
        for atom in ['H', 'O']:
            mask = self.atom_names == atom
            if any(mask) and self.protein is not None:
                pos = self.protein.select_atoms(f'segid {self.chain} and resnum {self.site} '
                                                f'and name {atom} and not altloc B').positions
                if len(pos) > 0:
                    self.coords[:, mask] = pos[0]

        self.mx, self.ori = chiLife.global_mx(*np.squeeze(self.backbone), method=self.superimposition_method)
        self.ICs_to_site()

    def ICs_to_site(self):
        ic_backbone = np.squeeze(self.internal_coords[0].coords[:3])
        self.ic_ori, self.ic_mx = chiLife.local_mx(*ic_backbone, method=self.superimposition_method)
        m2m3 = self.ic_mx @ self.mx
        op = {}
        for segid in self.internal_coords[0].chain_operators:
            new_mx = self.internal_coords[0].chain_operators[segid]['mx'] @ m2m3
            new_ori = (self.internal_coords[0].chain_operators[segid]['ori'] - self.ic_ori) @ m2m3 + self.ori
            op[segid] = {'mx': new_mx, 'ori': new_ori}

        for IC in self.internal_coords:
            IC.chain_operators = op

    def sample(self, n=1, off_rotamer=False, **kwargs):
        """Randomly sample a rotamer in the library."""
        if not self.weighted_sampling:
            idx = np.random.randint(len(self._weights), size=n)
        else:
            idx = np.random.choice(len(self._weights), size=n, p=self._weights)

        if not hasattr(off_rotamer, '__len__'):
            off_rotamer = [off_rotamer] * len(self.dihedral_atoms) if len(self.dihedral_atoms) > 0 else [True]
        else:
            off_rotamer = off_rotamer[:len(self.dihedral_atoms)]

        if not any(off_rotamer):
            return np.squeeze(self.coords[idx]), np.squeeze(self.weights[idx])

        if len(self.dihedral_atoms) == 0:
            return self.coords, self.weights, self.internal_coords
        elif hasattr(self, 'internal_coords'):
            returnables = zip(*[self._off_rotamer_sample(iidx, off_rotamer, **kwargs) for iidx in idx])
            return [np.squeeze(x) for x in returnables]

        else:
            raise AttributeError(f'{type(self)} objects require both internal coordinates ({type(self)}.internal_coords '
                                 f'attribute) and dihedral standard deviations ({type(self)}.sigmas attribute) to '
                                 f'perform off rotamer sampling')

    def _off_rotamer_sample(self, idx, off_rotamer, **kwargs):
        """

        :param idx:
        :return:
        """
        new_weight = 0
        if len(self.sigmas) > 0:
            new_dihedrals = np.random.vonmises(self._rdihedrals[idx, off_rotamer], self._rkappas[idx, off_rotamer])
            # diff1 = (new_dihedrals - self._rdihedrals[idx, off_rotamer]) * self._rkappas[idx, off_rotamer]
            # diff = (diff1 @ diff1) / len(diff1)
            new_weight = self._weights[idx]  # * np.exp(-0.5 * 5e-2 * diff)
        else:
            new_dihedrals = np.random.random(len(off_rotamer)) * 2 * np.pi
            new_weight = 1.

        internal_coord = self.internal_coords[idx].set_dihedral(new_dihedrals, 1, self.dihedral_atoms[off_rotamer])

        coords = internal_coord.to_cartesian()
        coords = coords[self.ic_mask]

        if kwargs.setdefault('return_dihedrals', False):
            return coords, new_weight, internal_coord
        else:
            return coords, new_weight

    def update_weight(self, weight):
        self.current_weight = weight

    def minimize(self):

        protein_clash_idx = self.protein_tree.query_ball_point(self.centroid(), self.clash_radius)
        protein_clash_idx = [idx for idx in protein_clash_idx if idx not in self.clash_ignore_idx]

        # Calculate rmin and epsilon for all atoms in protein that may clash
        rmin_ij, eps_ij = chiLife.get_lj_params(self.rmin2, self.eps,
                                                self.protein.atoms.types[protein_clash_idx], 1, forgive=1)

        a, b = [list(x) for x in zip(*self.non_bonded)]
        a_eps = chiLife.get_lj_eps(self.atom_types[a])
        a_radii = chiLife.get_lj_rmin(self.atom_types[a])
        b_eps = chiLife.get_lj_eps(self.atom_types[b])
        b_radii = chiLife.get_lj_rmin(self.atom_types[b])

        ab_lj_eps = np.sqrt(np.outer(a_eps, b_eps).reshape((-1)))
        ab_lj_radii = np.add.outer(a_radii, b_radii).reshape(-1)

        def objective(dihedrals, ic):
            coords = ic.set_dihedral(dihedrals, 1, self.dihedral_atoms).to_cartesian()
            coords = coords[self.ic_mask]

            dist = cdist(coords[self.side_chain_idx], self.protein.atoms.positions[protein_clash_idx]).ravel()
            dist2 = np.linalg.norm(coords[a] - coords[b], axis=1)
            with np.errstate(divide='ignore'):
                E1 = self.energy_func(dist[dist < 10], rmin_ij[dist < 10], eps_ij[dist < 10]).sum()
                E2 = self.energy_func(dist2, ab_lj_radii, ab_lj_eps).sum()

            E = E1 + E2
            return E

        for i, IC in enumerate(self.internal_coords):
            d0 = IC.get_dihedral(1, self.dihedral_atoms)
            lb = d0 - np.deg2rad(self.dihedral_sigma) / 2
            ub = d0 + np.deg2rad(self.dihedral_sigma) / 2
            bounds = np.c_[lb, ub]
            xopt = opt.minimize(objective, x0=self._rdihedrals[i], args=IC, bounds=bounds)

            self.coords[i] = IC.to_cartesian()[self.ic_mask]
            self.weights[i] = self.weights[i] * np.exp(-xopt.fun / (chiLife.GAS_CONST * 298))

        self.weights /= self.weights.sum()
        self.trim()

    def evaluate_clashes(self, environment, environment_tree=None, ignore_idx=None, temp=298):
        """
        Measure lennard-jones clashes of the spin label in a given environment and reweight/trim rotamers of the
        SpinLabel.
        :param environment: MDAnalysis.Universe, MDAnalysis.AtomGroup
            The protein environment to be considered when evaluating clashes.
        :param environment_tree: cKDtree
            k-dimensional tree of atom coordinates of the environment.
        :param ignore_idx: array-like
            list of atom coordinate indices to ignore when evaluating clashes. Usually the native amino acid at the
            SpinLable site.
        :param temp: float
            Temperature to consider when re-weighting rotamers.
        """

        if environment_tree is None:
            environment_tree = cKDTree(environment.positions)

        probabilities = chiLife.evaluate_clashes(ori=self.clash_ori, label_library=self.coords[:, self.side_chain_idx],
                                                 label_lj_rmin2=self.rmin2, label_lj_eps=self.eps,
                                                 environment=environment, environment_tree=environment_tree,
                                                 ignore_idx=ignore_idx, temp=temp, energy_func=self.energy_func,
                                                 clash_radius=self.clash_radius, forgive=self.forgive)

        self.weights, self.partition = chiLife.reweight_rotamers(probabilities, self.weights, return_partition=True)
        logging.info(f'Relative partition function: {self.partition:.3}')

    def trim(self, tol=0.005):
        """ Remove insignificant rotamers from Library"""
        old_len = len(self.weights)
        arg_sort_weights = np.argsort(self.weights)[::-1]
        sorted_weights = self.weights[arg_sort_weights]
        cumulative_weights = np.cumsum(sorted_weights)
        cutoff = np.maximum(1, len(cumulative_weights[cumulative_weights < 1 - tol]))

        self.coords = self.coords[arg_sort_weights[:cutoff]]
        self.dihedrals = self.dihedrals[arg_sort_weights[:cutoff]]
        self.weights = self.weights[arg_sort_weights[:cutoff]]
        if len(arg_sort_weights) == len(self.internal_coords):
            self.internal_coords = [self.internal_coords[x] for x in arg_sort_weights[:cutoff]]
        self.weights /= self.weights.sum()

        logging.info(f'{len(self.weights)} of {old_len} {self.res} rotamers make up at least {100 * (1-tol):.2f}% of '
                     f'rotamer density for site {self.site}')

    def centroid(self):
        """get the centroid of the whole rotamer library."""
        return self.coords.mean(axis=(0, 1))

    def evaluate(self):
        """place spin label on protein site and recalculate rotamer weights"""

        if self.protein_tree is None:
            self.protein_tree = cKDTree(self.protein.atoms.positions)
        self.evaluate_clashes(self.protein, self.protein_tree, ignore_idx=self.clash_ignore_idx)
        self.trim()

    def save_pdb(self, name=None):
        if name is None:
            name = self.name
        chiLife.save_rotlib(name, self.atoms, self.coords)

    @property
    def backbone(self):
        """Backbone coordinates of the spin label"""
        return np.squeeze(self.coords[0][self.backbone_idx])

    def get_lib(self):
        """
        Parse backbone information from protein and fetch the appropriate rotamer library

        :return:
        """

        PhiSel, PsiSel = None, None
        if self.protein is not None:
            # get site backbone information from protein structure
            PhiSel = self.protein.select_atoms(f'resnum {self.site} and segid {self.chain}').residues[0].phi_selection()
            PsiSel = self.protein.select_atoms(f'resnum {self.site} and segid {self.chain}').residues[0].psi_selection()

        # Default to helix backbone if none provided
        Phi = None if PhiSel is None else chiLife.get_dihedral(PhiSel.positions)
        Psi = None if PsiSel is None else chiLife.get_dihedral(PsiSel.positions)

        # Get library
        logging.info(f'Using backbone dependent library with Phi={Phi}, Psi={Psi}')
        lib = chiLife.read_library(self.res, Phi, Psi)
        return deepcopy(lib)

    def guess_chain(self):
        if self.protein is None:
            chain = 'A'
        elif len(nr_segids := set(self.protein.segments.segids)) == 1:
            chain = self.protein.segments.segids[0]
        elif np.isin(self.protein.residues.resnums, self.site).sum() == 0:
            raise ValueError(f'Residue {self.site} is not present on the provided protein')
        elif np.isin(self.protein.residues.resnums, self.site).sum() == 1:
            chain = self.protein.select_atoms(f'resid {self.site}').segids[0]
        else:
            raise ValueError(f'Residue {self.site} is present on more than one chain. Please specify the desired chain')
        return chain

    def protein_setup(self):
        # Position library at selected residue
        self.resindex = self.protein.select_atoms(self.selstr).residues[0].resindex
        self.segindex = self.protein.select_atoms(self.selstr).residues[0].segindex
        self.protein = self.protein.select_atoms('not (byres name OH2 or resname HOH)')

        self._to_site()

        # Get weight of current or closest rotamer
        self.clash_ignore_idx = \
            self.protein.select_atoms(f'resid {self.site} and segid {self.chain}').ix

        if self.coords.shape[1] == len(self.clash_ignore_idx):
            RMSDs = np.linalg.norm(self.coords - self.protein.atoms[self.clash_ignore_idx].positions[None, :, :],
                                   axis=(1, 2))
            idx = np.argmin(RMSDs)
            self.current_weight = self.weights[idx]
        else:
            self.current_weight = 0

        if self.eval_clash:
            self.evaluate()

    @property
    def bonds(self):
        if not hasattr(self, '_bonds'):
            bond_tree = cKDTree(self.coords[0])
            bonds = bond_tree.query_pairs(2.)
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
        if not hasattr(self, '_non_bonded'):
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
            if self._clash_ori_inp in ['cen', 'centroid']:
                return self.centroid()
            elif (ori_name := self._clash_ori_inp.upper()) in self.atom_names:
                return np.squeeze(self.coords[0][ori_name == self.atom_names])
        else:
            raise ValueError(f'Unrecognized clash_ori option {self._clash_ori_inp}. Please specify a 3D vector, an '
                             f'atom name or `centroid`')

        return self._clash_ori

    @clash_ori.setter
    def clash_ori(self, inp):
        self._clash_ori_inp = inp


def assign_defaults(kwargs):

    # Make all string arguments lowercase
    for key, value in kwargs.items():
        if isinstance(value, str):
            kwargs[key] = value.lower()

    # Default parameters
    defaults = {'protein_tree': None,
                'forgive': 1.,
                'clash_radius': 14.,
                '_clash_ori_inp': kwargs.pop('clash_ori', 'cen'),
                'superimposition_method': 'bisect',
                'dihedral_sigma': 35,
                'weighted_sampling': False,
                'eval_clash': False,
                'use_H': False,
                'sample_size': kwargs.pop('sample', False),
                'energy_func': chiLife.get_lj_rep}

    # Overwrite defaults
    kwargs_filled = {**defaults, **kwargs}

    # Make sure there are no unused parameters
    if len(kwargs_filled) != len(defaults):
        raise TypeError(f'Got unexpected keyword argument(s): '
                        f'{", ".join(key for key in kwargs if key not in defaults)}')

    return kwargs_filled.items()
