import igraph as ig
from copy import deepcopy
from itertools import combinations
import logging
import warnings
import inspect
from functools import partial
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
import scipy.optimize as opt
import MDAnalysis as mda

import chilife.io as io
import chilife.RotamerEnsemble as re
import chilife.scoring as scoring

from chilife.protein_utils import make_mda_uni
from .MolSysIC import MolSysIC

class dRotamerEnsemble:
    """Create new dRotamerEnsemble object.

        Parameters
        ----------
        res : string
            3-character name of desired residue, e.g. RXA.
        site : tuple[int, int]
            MolSys residue numbers to attach the bifunctional library to.
        protein : MDAnalysis.Universe, MDAnalysis.AtomGroup
            Object containing all protein information (coords, atom types, etc.)
        chain : str
            MolSys chain identifier to attach the bifunctional ensemble to.
        rotlib : str
            Rotamer library to use for constructing the RotamerEnsemble
        **kwargs : dict
            restraint_weight: float
                Force constant (kcal/mol/A^2) for calculating energetic penalty of restraint satisfaction, i.e. the
                alignment of the overlapping atoms of the two mono-functional subunits of the bifunctional label.
            torsion_weight: float
                Force constant (kcal/mol/radian^2) for calculating energetic penalty of the deviation from rotamer
                starting dihedral angles.
            minimize: bool
                Switch to turn on/off minimization. During minimization each rotamer is optimized in dihedral space
                with respect to alignment of the "cap" atoms of the two mono-functional subunits, internal clashes
                and deviation from the starting conformation in dihedral space.
            min_method: str
                Name of the minimization algorithm to use. All ``scipy.optimize.minimize`` algorithms are available
                and include: ‘Nelder-Mead’, ‘Powell’, ‘CG’, ‘BFGS’, ‘Newton-CG’, ‘L-BFGS-B’, ‘TNC’, ‘COBYLA’, ‘SLSQP’,
                ‘trust-constr’, ‘dogleg’, ‘trust-ncg’, ‘trust-exact’, ‘trust-krylov’, and custom.
            exclude_nb_interactions: int:
                When calculating internal clashes, ignore 1-``exclude_nb_interactions`` interactions and below. Defaults
                to ignore 1-3 interactions, i.e. atoms that are connected by 2 bonds or fewer will not have a steric
                effect on each other.
            eval_clash : bool
                Switch to turn clash evaluation on (True) and off (False).
            forcefield: str
                Name of the forcefield you wish to use to parameterize atoms for the energy function. Currently,
                supports `charmm` and `uff`
            energy_func : callable
               Python function or callable object that takes a protein or a RotamerEnsemble object as input and
               returns an energy value (kcal/mol) for each rotamer in the ensemble. See also
               :mod:`Scoring <chiLife.scoring>`. Defaults to a capped lennard-jones potentail.
            temp : float
               Temperature to use when running ``energy_func``. Only used if ``energy_func`` accepts a temperature
               argument  Defaults to 298 K.
            clash_radius : float
               Cutoff distance (angstroms) for inclusion of atoms in clash evaluations. This distance is measured from
               ``clash_ori`` Defaults to the longest distance between any two atoms in the rotamer ensemble plus 5
               angstroms.
            clash_ori : str
               Atom selection to use as the origin when finding atoms within the ``clash_radius``. Defaults to 'cen',
               the centroid of the rotamer ensemble heavy atoms.
            protein_tree : Scipy.spatial.cKDTree
               KDTree of atom positions for fast distance calculations and neighbor detection. Defaults to None
            trim: bool
                When true, the lowest `trim_tol` fraction of rotamers in the ensemble will be removed.
            trim_tol: float
                Tolerance for trimming rotamers from the ensemble. trim_tol=0.005 means the bottom 0.5% of rotamers
                will be removed.
            alignment_method : str
               Method to use when attaching or aligning the rotamer ensemble backbone with the protein backbone.
               Defaults to ``'bisect'`` which aligns the CA atom, the vectors bisecting the N-CA-C angle and the
               N-CA-C plane.
            dihedral_sigmas : float, numpy.ndarray
               Standard deviations of dihedral angles (degrees) for off rotamer sampling. Can be a single number for
               isotropic sampling, a vector to define each dihedral individually or a matrix to define a value for
               each rotamer and each dihedral. Setting this value to np.inf will force uniform (accessible volume)
               sampling. Defaults to 35 degrees.
            weighted_sampling : bool
               Determines whether the rotamer ensemble is sampled uniformly or based off of their intrinsic weights.
               Defaults to False.
            use_H : bool
               Determines if hydrogen atoms are used or not. Defaults to False.

        Attributes
        ----------
        res : string
            3-character name of desired residue, e.g. RXA.
        site1 : int
            MolSys residue number of the first attachment site.
        site2 : int
            MolSys residue number of the second attachment site.
        increment : int
            Number of residues  between two attachment sites.
        protein : MDAnalysis.Universe, MDAnalysis.AtomGroup, :class:`~MolSys`
            Object containing all protein information (coords, atom types, etc.)
        chain : str
            Chain identifier of the site the bifunctional ensemble is attached to.
        name : str
            Name of the ensemble. Usually include the native amino acid, the site number and the label that was attached
            changing the name will change the object name when saving in a PDB.
        RE1 : RotamerEnsemble
            Monofunctional ensemble subunit attached to the first site.
        RE2 : RotamerEnsemble
            Monofunctional ensemble subunit attached to the second site.
    """


    backbone_atoms = ["H", "N", "CA", "HA", "C", "O"]

    def __init__(self, res, sites, protein=None, chain=None, rotlib=None, **kwargs):
        self.res = res
        self.site1, self.icode1, self.site2, self.icode2 = proc_sites(sites)

        self.site = self.site1
        self.increment = self.site2 - self.site1
        self.kwargs = kwargs

        self.protein = protein
        self.chain = chain if chain is not None else self.guess_chain()

        self.input_kwargs = kwargs
        self.__dict__.update(dassign_defaults(kwargs))

        self.get_lib(rotlib)
        self.create_ensembles()

        self.RE1.backbone_to_site()
        self.RE2.backbone_to_site()

        self.cst_idx1 = np.where(self.RE1.atom_names[None, :] == self.csts[:, None])[1]
        self.cst_idx2 = np.where(self.RE2.atom_names[None, :] == self.csts[:, None])[1]

        _, idx1 = np.unique(self.cst_idx1, return_index=True)
        _, idx2 = np.unique(self.cst_idx2, return_index=True)

        self.cst_idx1 = self.cst_idx1[np.sort(idx1)]
        self.cst_idx2 = self.cst_idx2[np.sort(idx2)]

        for i in range(1, len(self.cst_idx2)):
            if self.RE2.atom_names[self.cst_idx2[i]] == self.RE2.atom_names[self.cst_idx2[i-1]]:
                self.cst_idx2[i - 1], self.cst_idx2[i] = self.cst_idx2[i], self.cst_idx2[i - 1]

        self.rl1mask = np.argwhere(~np.isin(self.RE1.atom_names, self.csts)).flatten()
        self.rl2mask = np.argwhere(~np.isin(self.RE2.atom_names, self.csts)).flatten()

        self.name = self.res
        if self.site1 is not None:
            self.name = f"{self.RE1.nataa}{self.site1}-{self.RE2.nataa}{self.site2}{self.res}"
        if self.chain is not None:
            self.name += f"_{self.chain}"

        self.selstr = f"resid {self.site1} {self.site2} and segid {self.chain} and not altloc B"

        self._graph = ig.Graph(edges=self.bonds)

        if self.clash_radius is None:
            self.clash_radius = np.linalg.norm(self.clash_ori - self.coords, axis=-1).max() + 5

        self.protein_setup()
        self.sub_labels = (self.RE1, self.RE2)

    @property
    def weights(self):
        """Array of the fractions of rotamer populations for each rotamer in the library."""
        return self.RE1.weights

    @weights.setter
    def weights(self, value):
        self.RE1.weights = value
        self.RE2.weights = value

    @property
    def coords(self):
        """The 3D cartesian coordinates of each atom of each rotamer in the library."""
        ovlp = (self.RE1.coords[:, self.cst_idx1] + self.RE2.coords[:, self.cst_idx2]) / 2
        return np.concatenate([self.RE1._coords[:, self.rl1mask], self.RE2._coords[:, self.rl2mask], ovlp], axis=1)

    @coords.setter
    def coords(self, value):
        if value.shape[1] != len(self.atom_names):
            raise ValueError(
                f"The provided coordinates do not match the number of atoms of this ensemble ({self.res})"
            )

        self.RE1._coords[:, self.rl1mask] = value[:, :len(self.rl1mask)]
        self.RE2._coords[:, self.rl2mask] = value[:, len(self.rl1mask):len(self.rl1mask) + len(self.rl2mask)]
        self.RE1._coords[:, self.cst_idx1] = value[:, len(self.rl1mask) + len(self.rl2mask):]
        self.RE2._coords[:, self.cst_idx2] = value[:, len(self.rl1mask) + len(self.rl2mask):]

    @property
    def _lib_coords(self):
        ovlp = (self.RE1._lib_coords[:, self.cst_idx1] + self.RE2._lib_coords[:, self.cst_idx2]) / 2
        return np.concatenate([self.RE1._lib_coords[:, self.rl1mask],
                               self.RE2._lib_coords[:, self.rl2mask], ovlp], axis=1)

    @property
    def atom_names(self):
        """The names of each atom in the rotamer"""
        return np.concatenate((self.RE1.atom_names[self.rl1mask],
                               self.RE2.atom_names[self.rl2mask],
                               self.RE1.atom_names[self.cst_idx1]))

    @property
    def atom_types(self):
        """The element or atom type of each atom in the rotamer."""
        return np.concatenate((self.RE1.atom_types[self.rl1mask],
                               self.RE2.atom_types[self.rl2mask],
                               self.RE1.atom_types[self.cst_idx1]))

    @property
    def dihedral_atoms(self):
        """Four atom sets defining each flexible dihedral of the side chain"""
        return np.concatenate([self.RE1.dihedral_atoms, self.RE2.dihedral_atoms])

    @property
    def dihedrals(self):
        """Dihedral angle values of each dihedral defined in :py:attr::`~dihedral_atoms` for each rotamer in the
        library"""
        return np.concatenate([self.RE1.dihedrals, self.RE2.dihedrals], axis=-1)

    @property
    def centroid(self):
        """The geometric center of all atoms of all rotamers in the rotamer library"""
        return self.coords.mean(axis=(0, 1))

    @property
    def clash_ori(self):
        """The origin used to determine if an external atom will be considered for clashes using the
        ``clash_radius`` property of the ensemble"""

        if isinstance(self._clash_ori_inp, (np.ndarray, list)):
            if len(self._clash_ori_inp) == 3:
                return self._clash_ori_inp

        elif isinstance(self._clash_ori_inp, str):
            if self._clash_ori_inp in ["cen", "centroid"]:
                return self.centroid

            elif (ori_name := self._clash_ori_inp.upper()) in self.atom_names:
                return np.squeeze(self.coords[0][ori_name == self.atom_names])

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
    def side_chain_idx(self):
        """Indices of the atoms that correspond to the side chain atoms (e.g. CB, CG, etc. and not N, CA, C)"""
        if not hasattr(self, '_side_chain_idx'):
            self._side_chain_idx = np.argwhere(
                np.isin(self.atom_names, dRotamerEnsemble.backbone_atoms, invert=True)
            ).flatten()

        return self._side_chain_idx

    @property
    def bonds(self):
        """Array of intra-label atom pairs indices that are covalently bonded."""
        if not hasattr(self, "_bonds"):
            bonds = []

            for bond in self.RE1.bonds:
                bndin = np.isin(bond, self.rl1mask)
                if np.all(bndin):
                    bonds.append(bond)
                elif np.any(bndin):
                    bonds.append([bond[0], np.argwhere(self.atom_names == self.RE1.atom_names[bond[1]]).flat[0]])
                else:
                    bonds.append([np.argwhere(self.atom_names == self.RE1.atom_names[bond[0]]).flat[0],
                                  np.argwhere(self.atom_names == self.RE1.atom_names[bond[1]]).flat[0]])

            for bond in self.RE2.bonds:
                bndin = np.isin(bond, self.rl2mask)
                if np.all(bndin):
                    bonds.append([b + len(self.rl1mask) for b in bond])
                elif not bndin[1]:
                    bonds.append([bond[0] + len(self.rl1mask),
                                  np.argwhere(self.atom_names == self.RE2.atom_names[bond[1]]).flat[0]])

            self._bonds = np.array(sorted(set(map(tuple, bonds))), dtype=int)

        return self._bonds

    @bonds.setter
    def bonds(self, inp):
        """
        Set of intra-label bonded pairs.
        Parameters
        ----------
        inp : ArrayLike
            List of atom ID pairs that are bonded
        """
        self._bonds = set(tuple(i) for i in inp)
        idxs = np.arange(len(self.atom_names))
        all_pairs = set(combinations(idxs, 2))
        self._non_bonded = all_pairs - self._bonds

    @property
    def non_bonded(self):
        """ Array of indices of intra-label non-bonded atom pairs. Primarily used for internal clash evaluation when
        sampling the dihedral space"""

        if not hasattr(self, "_non_bonded"):
            pairs = {v.index: [path for path in self._graph.get_all_shortest_paths(v) if
                           len(path) <= (self._exclude_nb_interactions)] for v in self._graph.vs}
            pairs = {(a, c) for a in pairs for b in pairs[a] for c in b if a < c}
            all_pairs = set(combinations(range(len(self.atom_names)), 2))
            self._non_bonded = all_pairs - pairs

        return sorted(list(self._non_bonded))

    @non_bonded.setter
    def non_bonded(self, inp):
        """
        Create a set of non-bonded atom pair indices within a single side chain
        Parameters
        ----------
        inp : ArrayLike
            List of atom ID pairs that are not bonded
        """

        self._non_bonded = set(tuple(i) for i in inp)
        idxs = np.arange(len(self.atom_names))
        all_pairs = set(combinations(idxs, 2))
        self._bonds = all_pairs - self._non_bonded

    def protein_setup(self):
        if isinstance(self.protein, (mda.AtomGroup, mda.Universe)):
            if not hasattr(self.protein.universe._topology, "altLocs"):
                self.protein.universe.add_TopologyAttr('altLocs', np.full(len(self.protein.universe.atoms), ""))
        if self.ignore_waters:
            self.protein = self.protein.select_atoms("not (byres name OH2 or resname HOH)")
        else:
            self.protein = self.protein.atoms

        clash_ignore_idx = self.protein.select_atoms(f"resid {self.site1} {self.site2} and segid {self.chain}").ix
        self.clash_ignore_idx = np.argwhere(np.isin(self.protein.ix, clash_ignore_idx)).flatten()
        self.resindex = self.protein.select_atoms(self.selstr).residues[0].resindex
        self.segindex = self.protein.select_atoms(self.selstr).residues[0].segindex

        if self.protein_tree is None:
            self.protein_tree = cKDTree(self.protein.atoms.positions)

        protein_clash_idx = self.protein_tree.query_ball_point(
            self.clash_ori, self.clash_radius
        )
        self.protein_clash_idx = [
            idx for idx in protein_clash_idx if idx not in self.clash_ignore_idx
        ]

        self.aidx, self.bidx = [list(x) for x in zip(*self.non_bonded)]
        if hasattr(self.energy_func, 'prepare_system'):
            self.energy_func.prepare_system(self)

        if self._minimize:
            self.minimize()

        if self.eval_clash:
            self.evaluate()

    def guess_chain(self):
        """
        Function to guess the chain based off of the attached protein and residue number.

        Returns
        -------
        chain : str
            The chain name
        """
        if self.protein is None:
            chain = "A"
        elif len(set(self.protein.segments.segids)) == 1:
            chain = self.protein.segments.segids[0]
        elif np.isin(self.protein.residues.resnums, self.site1).sum() == 0:
            raise ValueError(
                f"Residue {self.site1} is not present on the provided protein"
            )
        elif np.isin(self.protein.residues.resnums, self.site1).sum() == 1:
            chain = self.protein.select_atoms(f"resid {self.site1}").segids[0]
        else:
            raise ValueError(
                f"Residue {self.site1} is present on more than one chain. Please specify the desired chain"
            )
        return chain

    def get_lib(self, rotlib):
        """
        Specil function to get a rotamer library for a dRotamerEnsemble and apply all attributes to the object
        instance. ``rotlib`` can be a residue name, a file name, or a dictionary containing all  the standard entries
        of a rotamer library.

        Parameters
        ----------
        rotlib : Union[Path, str, dict]
            The name of the residue, the Path to the rotamer library file or a dictionary containing all entries of a
            rotamer library
        """

        # If given a dictionary use that as the rotlib
        if isinstance(rotlib, dict):
            if tuple(sorted(rotlib.keys())) != ('csts', 'libA', 'libB'):
                raise RuntimeError('Non-file dRotamerEnsemble rotlibs must be dictionaries consisting of exactle 3 '
                                   'entries: `csts`, `libA` and `libB`')
            self.csts = rotlib['csts']
            self.libA, self.libB = rotlib['libA'], rotlib['libB']
            self.kwargs["eval_clash"] = False
            return None

        rotlib = self.res if rotlib is None else rotlib
        if 'ip' not in rotlib:
            rotlib += f'ip{self.increment}'

        # Check if any exist
        rotlib_path = io.get_possible_rotlibs(rotlib, suffix='drotlib', extension='.zip')

        if rotlib_path is None:
            # Check if libraries exist but for different i+n
            rotlib_path = io.get_possible_rotlibs(rotlib.replace(f'ip{self.increment}', ''),
                                                       suffix='drotlib',
                                                       extension='.zip',
                                                       return_all=True)

            if rotlib_path is None:
                raise NameError(f'There is no rotamer library called {rotlib} in this directory or in chilife')

            else:
                warnings.warn(f'No rotamer library found for the given increment (ip{self.increment}) but rotlibs '
                              f'were found for other increments. chiLife will combine these rotlib to model '
                              f'this site1 pair and but they may not be accurate! Because there is no information '
                              f'about the relative weighting of different rotamer libraries all weights will be '
                              f'set to 1/len(rotlib)')

        if isinstance(rotlib_path, Path):
            libA, libB, csts = io.read_library(rotlib_path)

        elif isinstance(rotlib_path, list):

            cctA, cctB, ccsts = {}, {}, {}
            libA, libB, csts = io.read_library(rotlib_path[0])
            unis = []

            for lib in (libA, libB):
                names, types = lib['atom_names'], libA['atom_types']
                residxs = np.zeros(len(names), dtype=int)
                resnames, resids = np.array([self.res]), np.array([1])
                segidx = np.array([0])

                uni = make_mda_uni(names, types, resnames, residxs, resids, segidx)
                unis.append(uni)

            for p in rotlib_path:
                tlibA, tlibB, tcsts = io.read_library(p)
                for lib, tlib, cct, uni in zip((libA, libB), (tlibA, tlibB), (cctA, cctB), unis):

                    # Libraries must have the same atom order
                    if not np.all(np.isin(tlib['atom_names'], lib['atom_names'])) and \
                            np.all(tlib['dihedral_atoms'] == lib['dihedral_atoms']):
                        raise ValueError(f'Rotlibs {rotlib_path[0].stem} and {p.stem} are not compatable. You may'
                                         f'need to rename one of them.')

                    # Map coordinates
                    ixmap = [np.argwhere(tlib['atom_names'] == aname).flat[0] for aname in lib['atom_names']]
                    cct.setdefault('coords', []).append(tlib['coords'][:, ixmap])

                    # Create new internal coords if they are defined differently
                    lib_ic, tlib_ic = lib['internal_coords'], tlib['internal_coords']
                    if np.any(lib_ic.atom_names != tlib_ic.atom_names):
                        uni.load_new(cct['coords'][-1])
                        tlib_ic = MolSysIC.from_atoms(uni, lib['dihedral_atoms'], lib_ic.bonds)

                    tlib['zmats'] = tlib_ic.trajectory.coordinate_array
                    cct.setdefault('dihedrals', []).append(tlib['dihedrals'])
                    cct.setdefault('zmats', []).append(tlib['zmats'])

            for field in ('dihedrals', 'coords', 'zmats'):
                libA[field] = np.concatenate(cctA[field])
                libB[field] = np.concatenate(cctB[field])

            libA['internal_coords'].load_new(libA.pop('zmats'))
            libB['internal_coords'].load_new(libB.pop('zmats'))
            libA['weights'] = libB['weights'] = np.ones(len(libA['coords'])) / len(libA['coords'])

        self.csts = csts
        self.libA, self.libB = libA, libB
        self.kwargs["eval_clash"] = False

    def create_ensembles(self):
        """Creates monofunctional components of the bifunctional rotamer ensemble."""

        self.RE1 = re.RotamerEnsemble(self.res,
                                   self.site1,
                                   self.protein,
                                   self.chain,
                                   self.libA,
                                   **self.kwargs)

        self.RE2 = re.RotamerEnsemble(self.res,
                                   self.site2,
                                   self.protein,
                                   self.chain,
                                   self.libB,
                                   **self.kwargs)


    def save_pdb(self, name: str = None):
        """
        Save a PDB file of the ensemble

        Parameters
        ----------
        name: str
            Name of the PDB file
        """
        if name is None:
            name = self.name + ".pdb"
        if not name.endswith(".pdb"):
            name += ".pdb"

        io.save(name, self.RE1, self.RE2)

    def minimize(self, callback=None):
        """
        Minimize rotamers in dihedral space in the current context. Performed by default unless the ``minimize=False``
        keyword argument is used during construction. Note that the minimization method is controlled by the

        Parameters
        ----------
        callback: Callable
            A callable function to be passed as the ``scipy.optimize.minimize`` function. See the
            `scipy documentation <https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html>`_
            for details.
        """

        scores = [self._min_one(i, ic1, ic2, callback=callback) for i, (ic1, ic2) in
                  enumerate(zip(self.RE1.internal_coords, self.RE2.internal_coords))]

        scores = np.asarray(scores)
        SSEs = np.linalg.norm(self.RE1.coords[:, self.cst_idx1] - self.RE2.coords[:, self.cst_idx2], axis=2).sum(axis=1)
        MSD = SSEs / len(self.csts)
        MSDmin = MSD.min()

        if MSDmin > 0.1:
            warnings.warn(f'The minimum MSD of the cap is {MSD.min()}, this may result in distorted label. '
                          f'Check that the structures make sense.')

        if MSDmin > 0.25:
            raise RuntimeError(f'chiLife was unable to connect residues {self.site1} and {self.site2} with {self.res}. '
                               f'Please double check that this is the intended labeling site. It is likely that these '
                               f'sites are too far apart.')
        self.cap_MSDs = MSD
        self.RE1.backbone_to_site()
        self.RE2.backbone_to_site()
        self.score_base = scores.min()
        scores -= scores.min()
        self.rotamer_scores = scores + self.score_base
        self.weights *= np.exp(-scores / (scoring.GAS_CONST * self.temp) / np.exp(-scores).sum())
        self.weights /= self.weights.sum()

    def _objective(self, dihedrals, ic1, ic2):
        """
        Objective function to optimize for each rotamer in the ensemble.

        Parameters
        ----------
        dihedrals: ArrayLike
            Dihedral values

        ic1, ic2: chiLife.MolSysIC
            Internal coordinates object for the two mono functional subunits of the bifunctional label.

        Returns
        -------
        score: float
            Rotamer energy score for the current conformation
        """

        ic1.set_dihedral(dihedrals[: len(self.RE1.dihedral_atoms)], 1, self.RE1.dihedral_atoms)
        coords1 = ic1.to_cartesian()[self.RE1.ic_mask]

        ic2.set_dihedral(dihedrals[-len(self.RE2.dihedral_atoms):], 1, self.RE2.dihedral_atoms)
        coords2 = ic2.to_cartesian()[self.RE2.ic_mask]

        diff = np.linalg.norm(coords1[self.cst_idx1] - coords2[self.cst_idx2], axis=1)
        ovlp = (coords1[self.cst_idx1] + coords2[self.cst_idx2]) / 2
        coords = np.concatenate([coords1[self.rl1mask], coords2[self.rl2mask], ovlp], axis=0)
        r = np.linalg.norm(coords[self.aidx] - coords[self.bidx], axis=1)

        # Faster to compute lj here
        lj = self.irmin_ij / r
        lj = lj * lj * lj
        lj = lj * lj

        # attractive forces are needed, otherwise this term will perpetually push atoms apart
        internal_energy = self.ieps_ij * (lj * lj - 2 * lj)
        score = (diff @ diff) * self.restraint_weight / len(diff) + internal_energy.sum()

        return score

    def _min_one(self, i, ic1, ic2, callback=None):
        """
        Helper function to use when dispatching minimization jobs or each rotamer.

        Parameters
        ----------
        i: int
            rotamer index
        ic1, ic2: chiLife.MolSysIC
            Internal coordinates object for the two mono functional subunits of the bifunctional label.
        callback: Callable
            A callable function to be passed as the ``scipy.optimize.minimize`` function. See the
            `scipy documentation <https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html>`_
            for details.

        Returns
        -------
        score: float
            Energy score of the minimized rotamer.
        """

        if callback is not None:
            if 'i' in inspect.signature(callback).parameters:
                callback = partial(callback, i=i)


        d0 = np.concatenate([ic1.get_dihedral(1, self.RE1.dihedral_atoms),
                             ic2.get_dihedral(1, self.RE2.dihedral_atoms)])

        lb = d0 - np.pi  # np.deg2rad(40)
        ub = d0 + np.pi  # np.deg2rad(40) #
        bounds = np.c_[lb, ub]
        xopt = opt.minimize(self._objective, x0=d0, args=(ic1, ic2),
                            bounds=bounds, method=self.min_method,
                            callback=callback)
        self.RE1._coords[i] = ic1.coords[self.RE1.H_mask]
        self.RE2._coords[i] = ic2.coords[self.RE2.H_mask]
        tors = d0 - xopt.x
        tors = np.arctan2(np.sin(tors), np.cos(tors))
        tors = np.sqrt(tors @ tors)

        return xopt.fun + tors * self.torsion_weight

    def trim_rotamers(self):
        """Remove low probability rotamers from the ensemble. All rotamers accounting for the population less than
        ``self.trim_tol`` will be removed."""

        arg_sort_weights = np.argsort(self.weights)[::-1]
        sorted_weights = self.weights[arg_sort_weights]
        cumulative_weights = np.cumsum(sorted_weights)
        cutoff = np.maximum(1, len(cumulative_weights[cumulative_weights < 1 - self.trim_tol]))
        keep_idx = arg_sort_weights[:cutoff]
        if hasattr(self, 'cap_MSDs'):
            self.cap_MSDs = self.cap_MSDs[keep_idx]
            self.rotamer_scores = self.rotamer_scores[keep_idx]

        if hasattr(self, 'rot_clash_energy'):
            self.rot_clash_energy = self.rot_clash_energy[keep_idx]
        self.RE1.trim_rotamers(keep_idx=keep_idx)
        self.RE2.trim_rotamers(keep_idx=keep_idx)

    def evaluate(self):
        """Place rotamer ensemble on protein site and recalculate rotamer weights."""
        # Calculate external energies
        energies = self.energy_func(self)
        self.rot_clash_energy = energies
        # Calculate total weights (combining internal and external)
        self.weights, self.partition = scoring.reweight_rotamers(energies, self.temp, self.weights)
        logging.info(f"Relative partition function: {self.partition:.3}")

        # Remove low-weight rotamers from ensemble
        if self._do_trim:
            self.trim_rotamers()

    def __len__(self):
        """Return the length of the rotamer library (number of rotamers)"""
        return len(self.RE1.coords)

    def copy(self):
        """Create a deep copy of the dRotamerEnsemble object"""
        new_copy = dRotamerEnsemble(self.res, (self.site1, self.site2), chain=self.chain,
                                    protein=self.protein,
                                    rotlib={'csts': self.csts, 'libA': self.libA, 'libB': self.libB},
                                    minimize=False,
                                    eval_clash=False)
        for item in self.__dict__:
            if isinstance(self.dict[item], np.ndarray):
                new_copy.__dict__[item] = self.__dict__[item].copy()

            elif item == 'RE1':
                new_copy.__dict__[item] == self.__dict__[item].copy(rotlib=self.libA)

            elif item == 'RE2':
                new_copy.__dict__[item] == self.__dict__[item].copy(rotlib=self.libB)

            elif item == 'protein':
                pass

            else:
                new_copy.__dict__[item] = deepcopy(self.__dict__[item])
        return new_copy

def proc_sites(sites):
    sites = sorted(sites)
    new_sites = []
    for site in sites:
        if isinstance(site, str):
            site_split = re.split('(\D+)',site)
            site = site_split[0]
            icode = site_split[1] if len(site_split) > 1 else ""
        else:
            icode = ""
        new_sites.append(site)
        new_sites.append(icode)

    return new_sites

def dassign_defaults(kwargs):
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
        "eval_clash": True,
        "temp": 298,
        "clash_radius": None,
        "protein_tree": None,
        "_clash_ori_inp": kwargs.pop("clash_ori", "cen"),

        "alignment_method": "bisect",
        "dihedral_sigmas": 25,

        "use_H": False,
        "_exclude_nb_interactions": kwargs.pop('exclude_nb_interactions', 3),

        "energy_func": scoring.ljEnergyFunc(scoring.get_lj_energy, 'charmm', forgive=0.95)
,
        "_minimize": kwargs.pop('minimize', True),
        "min_method": 'L-BFGS-B',
        "_do_trim": kwargs.pop('trim', True),
        "trim_tol": 0.005,

        "restraint_weight": kwargs.pop('restraint_weight', 222),
        "torsion_weight": 5,

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