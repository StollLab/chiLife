from copy import deepcopy
from functools import partial
import inspect
import logging
import numpy as np
from itertools import combinations
from scipy.spatial import cKDTree
from scipy.stats import skewnorm
import scipy.optimize as opt
import MDAnalysis as mda
import chiLife


class RotamerLibrary:
    """Create new RotamerLibrary object.

    Attributes
    ----------
    name : str
        String identifying RotamerLibrary. Defaults to [site][res]_[chain] but chan be changed by the user if desired.
        This string will be used to name structures when saving PDBs.
    res : string
        3-character name of desired residue, e.g. R1A.
    site : int
        Protein residue number to attach library to.
    protein : MDAnalysis.Universe, MDAnalysis.AtomGroup
        Object containing all protein information (coords, atom types, etc.)
    protein_tree : scipy.spatial.cKDTree
        K-dimensional tree of the protein coordinates used internally for fast neighbor and distance calculations.
    chain : str
        Protein chain identifier to attach spin label to. Defaults to 'A' if no protein is provided.
    coords : numpy.ndarray
        Cartesian coordinates of each rotamer in the ensemble.
    internal_coords : List[ProteinIC]
        List of internal coordinate objects for each rotamer in the ensemble.
    atom_types : np.ndarray
        Array of atom types, usually just element symbols, of each atom in the rotamer.
    atom_names : np.ndarray
        Array of atom names for each atom in the rotamer.
    selstr : str
        Selection string that can be used with MDAnalysis ``select_atom`` method to select the site that the
        RotamerLibrary is attached to.
    input_kwargs : dict
        Stored copy of the keword arguments used when the object was created. ``input_kwargs`` is generally used for
        generating similar RotamerLibraries.
    eval_clash : bool
        Boolean argument to instruct the RotamerLibrary to evaluate clashes and trim rotamers on construction. If False
        The starting RotamerLibrary is attached to the specified site and no clashes are evaluated. Post-construction
        clash evaluations can be performed using the ``evaluate`` method. Defaults to False.
    energy_func : Callable
        Desired energy function to be used by the ``evaluate`` method when called by the user or when ``eval_clash``
        is set to True. ``energy_func`` must be a callable object and accept two arguments: ``protein`` and ``rotlib``
        Deafults to ``chiLife.get_lj_rep``.
    clash_ignore_idx : np.ndarray
        Array of atom indices of the ``protein`` attribute that should be ignored when the ``evaluate`` method is
        invoked. i.e. the atom indices of the native residue the RotamerLibrary is being attached to.
    forgive : float
        The "forgive-factor" to be used by the ``energy_func`` when the ``evaluate`` meth is invoked. If ``energy_func``
        does not use a forgive-factor this value will be ignored. Defaults to 1.0.
    temp : float
        Temperature (in Kelvin) to use when re-evaluating rotamer weights using ``energy_func``. Defaults to 298 K.
    clash_radius : float
        Cutoff distance (angstroms) for inclusion of atoms in clash evaluations. This distance is measured from
        ``clash_ori`` Defaults to the longest distance between any two atoms in the rotamer library plus 5 angstroms.
    superimposition_method : str, callable
        Method to use when attaching or aligning the rotamer library backbone with the protein backbone. Defaults to
        ``'bisect'`` which aligns the CA atom, the vectors bisecting the N-CA-C angle and the N-CA-C plane.
    dihedral_sigmas : float, numpy.ndarray
        Standard deviations of dihedral angles (degrees) for off rotamer sampling. Can be a single numebr for isotropic
        sampling, a vector to define each dihedral individually or a matrix to define a value for each rotamer and each
        dihedral. Defaults to 35 degrees)
    weighted_sampling : bool
        Determines whether the rotamer library is sampled uniformly or based off of their intrinsic weights. Defaults
        to False.
    use_H : bool
        Determines if hydrogen atoms are used or not. Defaults to False.
    atom_energies : numpy.ndarray
        Per atom energy (kcal/mol) value calculated suing ``self.energy_func``.
    partition : float
        Partition function value indicative of how much the ensemble has changed with respect to the original rotamer
        library. Only useful when not performing sampling.
    """

    backbone_atoms = ["H", "N", "CA", "HA", "C", "O"]

    def __init__(self, res, site=1, protein=None, chain=None, **kwargs):
        """Create new RotamerLibrary object.

        Parameters
        ----------
        res : string
            3-character name of desired residue, e.g. R1A.
        site : int
            Protein residue number to attach library to.
        protein : MDAnalysis.Universe, MDAnalysis.AtomGroup
            Object containing all protein information (coords, atom types, etc.)
        chain : str
            Protein chain identifier to attach spin label to.
        **kwargs : dict
            protein_tree : Scipy.spatial.cKDTree
                KDTree of atom positions for fast distance calculations and neighbor detection. Defaults to None
            forgive : float
                Softening factor to be passed to ``energy_func``. Only used if ``energy_func`` uses a softening factor.
                Defaults to 1.0. See :mod:`Scoring <chiLife.scoring>` .
            temp : float
                Temperature to use when running ``energy_func``. Only used if ``energy_func`` accepts a temperature
                argument  Defaults to 298 K.
            clash_radius : float
                Cutoff distance (angstroms) for inclusion of atoms in clash evaluations. This distance is measured from
                ``clash_ori`` Defaults to the longest distance between any two atoms in the rotamer library plus 5
                angstroms.
            clash_ori : str
                Atom selection to use as the origin when finding atoms within the ``clash_radius``. Defaults to 'cen',
                the centroid of the rotamer library heavy atoms.
            superimposition_method : str
                Method to use when attaching or aligning the rotamer library backbone with the protein backbone.
                Defaults to ``'bisect'`` which aligns the CA atom, the vectors bisecting the N-CA-C angle and the
                N-CA-C plane.
            dihedral_sigmas : float, numpy.ndarray
                Standard deviations of dihedral angles (degrees) for off rotamer sampling. Can be a single numebr for
                isotropic sampling, a vector to define each dihedral individually or a matrix to define a value for
                each rotamer and each dihedral. Setting this value to np.inf will force uniform (accessible volume)
                sampling. Defaults to 35 degrees.
            weighted_sampling : bool
                Determines whether the rotamer library is sampled uniformly or based off of their intrinsic weights.
                Defaults to False.
            use_H : bool
                Determines if hydrogen atoms are used or not. Defaults to False.
            eval_clash : bool
            sample : int, bool
                Argument to use the off-rotamer sampling method. If ``False`` or ``0`` the off-rotamer sampling method
                will not be used. If ``int`` the ensemble will be generated with that many off-rotamer samples.
            energy_func : callable
                Python function or callable object that takes a protein and a RotamerLibrary object as input and
                reutrns an energy value (kcal/mol) for each atom of each rotamer in the ensemble. See also
                :mod:`Scoring <chiLife.scoring>` . Defaults to :mod:`chiLife.get_lj_rep <chiLife.get_lj_rep>` .
        """

        self.res = res
        self.site = int(site)
        self.protein = protein
        self.nataa = ""
        self.chain = chain if chain is not None else guess_chain(self.protein, self.site)
        self.selstr = f"resid {self.site} and segid {self.chain} and not altloc B"
        self.input_kwargs = kwargs
        self.__dict__.update(assign_defaults(kwargs))

        # Check if superimposition method requires rotlib argument or not
        if isinstance(self.superimposition_method, str):
            self.superimposition_method = chiLife.superimpositions[self.superimposition_method]

        lib = self.get_lib()
        self.__dict__.update(lib)
        self._weights = self.weights / self.weights.sum()

        if len(self.sigmas) != 0 and 'dihedral_sigmas' not in kwargs:
            self.sigmas[self.sigmas == 0] = self.dihedral_sigmas
        else:
            self.set_dihedral_sampling_sigmas(self.dihedral_sigmas)

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

        self._lib_coords = self._coords.copy()
        self._lib_IC = self.internal_coords

        if self.clash_radius is None:
            self.clash_radius = np.linalg.norm(self.clash_ori - self.coords, axis=-1).max() + 5

        # Parse important indices
        self.backbone_idx = np.argwhere(np.isin(self.atom_names, ["N", "CA", "C"]))
        self.side_chain_idx = np.argwhere(
            np.isin(self.atom_names, RotamerLibrary.backbone_atoms, invert=True)
        ).flatten()

        if self._sample_size and len(self.dihedral_atoms) > 0:
            # Get list of non-bonded atoms before overwritig
            a, b = [list(x) for x in zip(*self.non_bonded)]

            # Perform sapling
            self._coords = np.tile(self._coords[0], (self._sample_size, 1, 1))
            self._coords, self.weights, self.internal_coords = self.sample(
                self._sample_size, off_rotamer=True, return_dihedrals=True
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
            self.protein.atoms, (mda.AtomGroup, chiLife.BaseSystem)
        ):
            self.protein_setup()
            resname = self.protein.atoms[self.clash_ignore_idx[0]].resname
            self.nataa = chiLife.nataa_codes.get(resname, resname)

        # Store atom information as atom objects
        self.atoms = [
            chiLife.FreeAtom(name, atype, idx, self.res, self.site, coord)
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
        """

        Parameters
        ----------
        pdb_file :
            
        res :
             (Default value = None)
        site :
             (Default value = None)
        protein :
             (Default value = None)
        chain :
             (Default value = None)

        Returns
        -------

        """
        # TODO: create pdb reader and impliment from_pdb method
        return cls(res, site, protein, chain)

    @classmethod
    def from_mda(cls, residue, **kwargs):
        """Create RotamerLibrary from the MDAnalysis.Residue

        Parameters
        ----------
        residue :
            MDAnalysis.Residue
        **kwargs :
            

        Returns
        -------
        type
            RotamerLibrary

        """

        res = residue.resname
        site = residue.resnum
        chain = residue.segid
        protein = residue.universe
        return cls(res, site, protein, chain, **kwargs)

    @classmethod
    def from_trajectory(cls, traj, site, chain=None, energy=None, burn_in=100, **kwargs):
        """

        Parameters
        ----------
        traj :
            
        site :
            
        chain :
             (Default value = None)
        energy :
             (Default value = None)
        burn_in :
             (Default value = 100)
        **kwargs :
            

        Returns
        -------

        """

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

        _, unique_idx, non_unique_idx = np.unique(coords, axis=0, return_inverse=True, return_index=True)
        coords = coords[unique_idx]

        prelib = cls(resname, site, chain=chain, protein=traj, eval_clash=False, **kwargs)
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
        """Equivalence measurement between a spin label and another object. A SpinLabel cannot be equivalent to a
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
        """Assign SpinLabel to a site.

        Parameters
        ----------
        site :
            int
            residue number to assign SpinLabel to.

        Returns
        -------

        """
        self.site = site
        self.to_site()

    def copy(self, site=None, chain=None):
        """Create a deep copy of the spin label. Assign new site and chain information if desired

        Parameters
        ----------
        site :
             (Default value = None)
        chain :
             (Default value = None)

        Returns
        -------

        """
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

        Parameters
        ----------
        site_pos :
            array-like
            3x3 array of ordered backbone atom coordinates of new site (N CA C) (Default value = None)

        Returns
        -------

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
        """ """
        # Update chain operators
        ic_backbone = np.squeeze(self.internal_coords[0].coords[:3])

        if self.superimposition_method.__name__ == 'fit_superimposition':
            N, CA, C = chiLife.parse_backbone(self, kind="global")
            ic_backbone = np.array([[ic_backbone[0], N[1]],
                                    [ic_backbone[1], CA[1]],
                                    [ic_backbone[2], C[1]]])

        self.ic_ori, self.ic_mx = chiLife.local_mx(
            *ic_backbone, method=self.superimposition_method
        )
        m2m3 = self.ic_mx @ self.mx
        op = {}
        for segid in self.internal_coords[0].chain_operators:
            new_mx = self.internal_coords[0].chain_operators[segid]["mx"] @ m2m3
            new_ori = (
                self.internal_coords[0].chain_operators[segid]["ori"] - self.ic_ori
            ) @ m2m3 + self.origin
            op[segid] = {"mx": new_mx, "ori": new_ori}

        for IC in self.internal_coords:
            IC.chain_operators = op

        # Update backbone conf
        alist = ["O"] if not self.use_H else ["H", 'O']
        for atom in alist:
            mask = self.internal_coords[0].atom_names == atom
            if any(mask) and self.protein is not None:

                ICatom = self.internal_coords[0].atoms[self.internal_coords[0].atom_names == atom][0]
                dihe_def = tuple(reversed(ICatom.atom_names))

                p = self.protein.select_atoms(
                    f"segid {self.chain} and resnum {self.site} and name {' '.join(dihe_def)} and not altloc B"
                ).positions

                if len(p) > 4:
                    # Guess that the closest H to the nitrogen is the H we are looking for
                    HN_idx = np.argmin(np.linalg.norm(p[3:] - p[0], axis=1)) + 3
                    p = p[[0, 1, 2, HN_idx]]
                elif len(p) == 3:
                    continue
                if atom == 'H':
                    # Reorder
                    p = p[[2, 1, 0, 3]]

                dihe = chiLife.get_dihedral(p)
                ang = chiLife.get_angle(p[1:])
                bond = np.linalg.norm(p[-1] - p[-2])
                idx = np.squeeze(np.argwhere(mask))

                if atom == 'O':
                    additional_idxs = list(self.internal_coords[0].atom_dict['dihedrals'][1][(1, ('CA', 'C', 'O'))].values())

                for IC in self.internal_coords:
                    delta = IC.zmats[1][idx][2] - dihe
                    IC.zmats[1][idx] = bond, ang, dihe
                    if atom == "O":
                        IC.zmats[1][additional_idxs, 2] -= delta

    def backbone_to_site(self):
        """ """
        # Keep protein backbone dihedrals for oxygen and hydrogens
        for atom in ["H", "O"]:
            mask = self.atom_names == atom
            if any(mask) and self.protein is not None:
                pos = self.protein.select_atoms(
                    f"segid {self.chain} and resnum {self.site} "
                    f"and name {atom} and not altloc B"
                ).positions
                if pos.shape == (1, 3):
                    self._coords[:, mask] = np.squeeze(pos)
                elif pos.shape == (3,):
                    self._coords[:, mask] = np.squeeze(pos)
                elif pos.shape[0] > 1:
                    idx = np.argmin(np.linalg.norm(pos - self.backbone[0], axis=1))
                    self._coords[:, mask] = np.squeeze(pos[idx])
                else:
                    pass

    def sample(self, n=1, off_rotamer=False, **kwargs):
        """Randomly sample a rotamer in the library.

        Parameters
        ----------
        n :
             (Default value = 1)
        off_rotamer :
             (Default value = False)
        **kwargs :
            

        Returns
        -------

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
            if not np.allclose(np.squeeze(self._lib_coords[0, self.backbone_idx]), self.backbone):
                N, CA, C = self.backbone
                mx, ori = chiLife.global_mx(N, CA, C, method=self.superimposition_method)

                N, CA, C = np.squeeze(self._lib_coords[0, self.backbone_idx])
                old_ori, ori_mx = chiLife.local_mx(N, CA, C, method=self.superimposition_method)

                self._lib_coords -= old_ori
                mx = ori_mx @ mx

                self._lib_coords = np.einsum("ijk,kl->ijl", self._lib_coords, mx) + ori

                for atom in ["H", "O"]:
                    mask = self.atom_names == atom
                    if any(mask) and self.protein is not None:
                        pos = self.protein.select_atoms(
                            f"segid {self.chain} and resnum {self.site} "
                            f"and name {atom} and not altloc B"
                        ).positions
                        if len(pos) > 0:
                            self._lib_coords[:, mask] = pos[0]


            return np.squeeze(self._lib_coords[idx]), np.squeeze(self._weights[idx])

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

        Parameters
        ----------
        idx :
            return:
        off_rotamer :
            
        **kwargs :
            

        Returns
        -------

        """
        new_weight = 0
        # Use accessible volume sampling if only provided a single rotamer
        if len(self._weights) == 1 or np.all(np.isinf(self.sigmas)):
            new_dihedrals = np.random.random(len(off_rotamer)) * 2 * np.pi
            new_weight = 1.0

        #  sample from von mises near rotamer unless more information is provided
        elif self.skews is None:
            new_dihedrals = np.random.vonmises(self._rdihedrals[idx, off_rotamer], self._rkappas[idx, off_rotamer])
            new_weight = 1.0
        # Sample from skewednorm if skews are provided
        else:
            deltas = skewnorm.rvs(a=self.skews[idx], loc=self.locs[idx], scale=self.sigmas[idx])
            pdf = skewnorm.pdf(deltas, a=self.skews[idx], loc=self.locs[idx], scale=self.sigmas[idx])
            new_dihedrals = np.deg2rad(self.dihedrals[idx] + deltas)
            new_weight = pdf.prod()

        new_weight = self._weights[idx] * new_weight
        dihedrals = self._rdihedrals[idx].copy()
        dihedrals[off_rotamer] = new_dihedrals
        int_coord = self._lib_IC[idx].copy().set_dihedral(dihedrals, 1, self.dihedral_atoms)

        coords = int_coord.coords[self.ic_mask]

        if kwargs.setdefault("return_dihedrals", False):
            return coords, new_weight, int_coord
        else:
            return coords, new_weight

    def update_weight(self, weight):
        """

        Parameters
        ----------
        weight :
            

        Returns
        -------

        """
        self.current_weight = weight

    def minimize(self):
        """ """
        def objective(dihedrals, ic):
            """

            Parameters
            ----------
            dihedrals :
                
            ic :
                

            Returns
            -------

            """
            coords = ic.set_dihedral(dihedrals, 1, self.dihedral_atoms).to_cartesian()
            temp_rotlib._coords = np.atleast_3d([coords[self.ic_mask]])
            return self.energy_func(temp_rotlib.protein, temp_rotlib)

        temp_rotlib = self.copy()
        for i, IC in enumerate(self.internal_coords):
            d0 = IC.get_dihedral(1, self.dihedral_atoms)
            lb = d0 - np.deg2rad(self.sigmas[i])
            ub = d0 + np.deg2rad(self.sigmas[i])
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
        """Remove insignificant rotamers from Library

        Parameters
        ----------
        tol :
             (Default value = 0.005)

        Returns
        -------

        """
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
        """

        Parameters
        ----------
        name :
             (Default value = None)

        Returns
        -------

        """
        if name is None:
            name = self.name
        chiLife.save_rotlib(name, self.atoms, self._coords)

    @property
    def backbone(self):
        """Backbone coordinates of the spin label"""
        return np.squeeze(self._coords[0][self.backbone_idx])

    def get_lib(self):
        """Parse backbone information from protein and fetch the appropriate rotamer library
        
        :return:

        Parameters
        ----------

        Returns
        -------

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

        if 'skews' not in lib:
            lib['skews'] = None

        return lib

    def protein_setup(self):
        """ """
        # Position library at selected residue
        self.resindex = self.protein.select_atoms(self.selstr).resindices[0]
        self.segindex = self.protein.select_atoms(self.selstr).segindices[0]
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
        """ """
        if not hasattr(self, "_bonds"):
            bond_tree = cKDTree(self._coords[0])
            bonds = bond_tree.query_pairs(2.0)
            self._bonds = set(tuple(sorted(bond)) for bond in bonds)
        return list(sorted(self._bonds))

    @bonds.setter
    def bonds(self, inp):
        """

        Parameters
        ----------
        inp :
            

        Returns
        -------

        """
        self._bonds = set(tuple(i) for i in inp)
        idxs = np.arange(len(self.atom_names))
        all_pairs = set(combinations(idxs, 2))
        self._non_bonded = all_pairs - self._bonds

    @property
    def non_bonded(self):
        """ """
        if not hasattr(self, "_non_bonded"):
            idxs = np.arange(len(self.atom_names))
            all_pairs = set(combinations(idxs, 2))
            self._non_bonded = all_pairs - set(self.bonds)

        return sorted(list(self._non_bonded))

    @non_bonded.setter
    def non_bonded(self, inp):
        """

        Parameters
        ----------
        inp :
            

        Returns
        -------

        """
        self._non_bonded = set(tuple(i) for i in inp)
        idxs = np.arange(len(self.atom_names))
        all_pairs = set(combinations(idxs, 2))
        self._bonds = all_pairs - self._non_bonded

    def __len__(self):
        return len(self.weights)

    @property
    def clash_ori(self):
        """ """
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
        """

        Parameters
        ----------
        inp :
            

        Returns
        -------

        """
        self._clash_ori_inp = inp

    @property
    def coords(self):
        """ """
        return self._coords

    @coords.setter
    def coords(self, coords):
        """

        Parameters
        ----------
        coords :
            

        Returns
        -------

        """
        # TODO: Add warning about removing isomers

        # Allow users to input a single rotamer
        coords = coords if coords.ndim == 3 else coords[None, :, :]

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
        """ """
        return self._dihedrals

    @dihedrals.setter
    def dihedrals(self, dihedrals):
        """

        Parameters
        ----------
        dihedrals :
            

        Returns
        -------

        """
        # TODO: Add warning about removing isomers

        dihedrals = dihedrals if dihedrals.ndim == 2 else dihedrals[None, :]
        if dihedrals.shape[1] != self.dihedrals.shape[1]:
            raise ValueError('The input array does not have the correct number of dihedrals')

        self._dihedrals = dihedrals
        self.internal_coords = [self.internal_coords[0].copy().set_dihedral(np.deg2rad(dihedral),
                                                                            1,
                                                                            self.dihedral_atoms)
                                for dihedral in dihedrals]
        self._coords = np.array([ic.coords[self.ic_mask] for ic in self.internal_coords])
        self.backbone_to_site()

        # Apply uniform weights
        self.weights = np.ones(len(self._dihedrals))
        self.weights /= self.weights.sum()

    @property
    def mx(self):
        """ """
        mx, ori = chiLife.global_mx(*chiLife.parse_backbone(self, kind="local"), method=self.superimposition_method)
        return mx

    @property
    def origin(self):
        """ """
        return np.squeeze(self.backbone[1])

    def get_sasa(self):
        """ """
        atom_radii = chiLife.get_lj_rmin(self.atom_types)
        if self.protein is not None:
            environment_coords = self.protein.atoms[self.protein_clash_idx].positions
            environment_radii = chiLife.get_lj_rmin(self.protein.atoms[self.protein_clash_idx].types)
        else:
            environment_coords = np.empty((0, 3))
            environment_radii = np.empty(0)

        SASAs = chiLife.numba_utils.get_sasa(self.coords, atom_radii, environment_coords, environment_radii)

        return np.array(SASAs)

    def set_dihedral_sampling_sigmas(self, value):
        """

        Parameters
        ----------
        value :
            

        Returns
        -------

        """
        value = np.asarray(value)

        if value.shape == ():
            self.sigmas = (np.ones((len(self._weights), len(self.dihedral_atoms))) * value)
        elif len(value) == len(self.dihedral_atoms) and len(value.shape) == 1:
            self.sigmas = np.tile(value, (len(self._weights), 1))
        elif value.shape == (len(self._weights), len(self.dihedral_atoms)):
            self.sigmas = value.copy()
        else:
            raise ValueError('`dihedral_sigmas` must be a scalar, an array the length of the `self.dihedral atoms` or '
                             'an array with the shape of (len(self.weights), len(self.dihedral_atoms))')


def assign_defaults(kwargs):
    """

    Parameters
    ----------
    kwargs :
        

    Returns
    -------

    """

    # Make all string arguments lowercase
    for key, value in kwargs.items():
        if isinstance(value, str):
            kwargs[key] = value.lower()

    # Default parameters
    defaults = {
        "protein_tree": None,
        "forgive": 1.0,
        "temp": 298,
        "clash_radius": None,
        "_clash_ori_inp": kwargs.pop("clash_ori", "cen"),
        "superimposition_method": "bisect",
        "dihedral_sigmas": 35,
        "weighted_sampling": False,
        "eval_clash": False,
        "use_H": False,
        "_sample_size": kwargs.pop("sample", False),
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
    """

    Parameters
    ----------
    protein :
        
    site :
        

    Returns
    -------

    """
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
