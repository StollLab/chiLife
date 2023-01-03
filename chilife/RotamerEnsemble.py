from copy import deepcopy
from pathlib import Path
import logging
import numpy as np
from itertools import combinations
from scipy.spatial import cKDTree
from scipy.stats import skewnorm
import scipy.optimize as opt
import MDAnalysis as mda
import chilife
from .numba_utils import batch_ic2cart


class RotamerEnsemble:
    """Create new RotamerEnsemble object.

    Attributes
    ----------
    name : str
        String identifying RotamerEnsemble. Defaults to [site][res]_[chain] but chan be changed by the user if desired.
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
        RotamerEnsemble is attached to.
    input_kwargs : dict
        Stored copy of the keword arguments used when the object was created. ``input_kwargs`` is generally used for
        generating similar RotamerEnsembles.
    eval_clash : bool
        Boolean argument to instruct the RotamerEnsemble to evaluate clashes and trim rotamers on construction. If False
        The starting RotamerEnsemble is attached to the specified site and no clashes are evaluated. Post-construction
        clash evaluations can be performed using the ``evaluate`` method. Defaults to False.
    energy_func : Callable
        Desired energy function to be used by the ``evaluate`` method when called by the user or when ``eval_clash``
        is set to True. ``energy_func`` must be a callable object and accept two arguments: ``protein`` and ``ensemble``
        Deafults to ``chiLife.get_lj_rep``.
    clash_ignore_idx : np.ndarray
        Array of atom indices of the ``protein`` attribute that should be ignored when the ``evaluate`` method is
        invoked. i.e. the atom indices of the native residue the RotamerEnsemble is being attached to.
    forgive : float
        The "forgive-factor" to be used by the ``energy_func`` when the ``evaluate`` meth is invoked. If ``energy_func``
        does not use a forgive-factor this value will be ignored. Defaults to 1.0.
    temp : float
        Temperature (in Kelvin) to use when re-evaluating rotamer weights using ``energy_func``. Defaults to 298 K.
    clash_radius : float
        Cutoff distance (angstroms) for inclusion of atoms in clash evaluations. This distance is measured from
        ``clash_ori`` Defaults to the longest distance between any two atoms in the rotamer ensemble plus 5 angstroms.
    alignment_method : str, callable
        Method to use when attaching or aligning the rotamer ensemble backbone with the protein backbone. Defaults to
        ``'bisect'`` which aligns the CA atom, the vectors bisecting the N-CA-C angle and the N-CA-C plane.
    dihedral_sigmas : float, numpy.ndarray
        Standard deviations of dihedral angles (degrees) for off rotamer sampling. Can be a single numebr for isotropic
        sampling, a vector to define each dihedral individually or a matrix to define a value for each rotamer and each
        dihedral. Defaults to 35 degrees)
    weighted_sampling : bool
        Determines whether the rotamer library is sampled uniformly or based off of their intrinsic weights to generate 
        the RotamerEnsemble. Defaults
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

    def __init__(self, res, site=None, protein=None, chain=None, rotlib=None, **kwargs):
        """Create new RotamerEnsemble object.

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
        rotlib : str
            Rotamer library to use for constructing the RotamerEnsemble
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
                ``clash_ori`` Defaults to the longest distance between any two atoms in the rotamer ensemble plus 5
                angstroms.
            clash_ori : str
                Atom selection to use as the origin when finding atoms within the ``clash_radius``. Defaults to 'cen',
                the centroid of the rotamer ensemble heavy atoms.
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
            eval_clash : bool
            sample : int, bool
                Argument to use the off-rotamer sampling method. If ``False`` or ``0`` the off-rotamer sampling method
                will not be used. If ``int`` the ensemble will be generated with that many off-rotamer samples.
            energy_func : callable
                Python function or callable object that takes a protein and a RotamerEnsemble object as input and
                reutrns an energy value (kcal/mol) for each atom of each rotamer in the ensemble. See also
                :mod:`Scoring <chiLife.scoring>` . Defaults to :mod:`chiLife.get_lj_rep <chiLife.get_lj_rep>` .
        """

        self.res = res
        if site is None and protein is not None:
            raise ValueError('A protein has been provided but a site has not. If you wish to construct an ensemble '
                             'associated with a protein you must include the site you wish to model.')
        elif site is None:
            site = 1

        self.site = int(site)
        self.protein = protein
        self.nataa = ""
        self.chain = chain if chain is not None else guess_chain(self.protein, self.site)
        self.selstr = f"resid {self.site} and segid {self.chain} and not altloc B"
        self.input_kwargs = kwargs
        self.__dict__.update(assign_defaults(kwargs))

        # Convert string arguments for alignment_method to respective function
        if isinstance(self.alignment_method, str):
            self.alignment_method = chilife.alignment_methods[self.alignment_method]

        lib = self.get_lib(rotlib)
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
            np.isin(self.atom_names, RotamerEnsemble.backbone_atoms, invert=True)
        ).flatten()

        # Sample from library if requested
        if self._sample_size and len(self.dihedral_atoms) > 0:
            # Get list of non-bonded atoms before overwriting
            a, b = [list(x) for x in zip(*self.non_bonded)]

            # Draw samples
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
        self.name = self.rotlib
        if self.site is not None:
            self.name = str(self.site) + self.rotlib
        if self.chain not in ('A', None):
            self.name += f"_{self.chain}"

        # Create arrays of LJ potential params
        if len(self.side_chain_idx) > 0:
            self.rmin2 = chilife.get_lj_rmin(self.atom_types[self.side_chain_idx])
            self.eps = chilife.get_lj_eps(self.atom_types[self.side_chain_idx])

        if hasattr(self.protein, "atoms") and isinstance(
            self.protein.atoms, (mda.AtomGroup, chilife.BaseSystem)
        ):
            self.protein_setup()
            resname = self.protein.atoms[self.clash_ignore_idx[0]].resname
            self.nataa = chilife.nataa_codes.get(resname, resname)
            self.name = self.nataa + self.name

        # Store atom information as atom objects
        self.atoms = [
            chilife.FreeAtom(name, atype, idx, self.res, self.site, coord)
            for idx, (coord, atype, name) in enumerate(
                zip(self._coords[0], self.atom_types, self.atom_names)
            )
        ]

    def __str__(self):
        return (
            f"Rotamer ensemble with {np.size(self.weights)} members\n"
            f"  Name: {self.name}\n"
            f"  Label: {self.res}\n"
            f"  Site: {self.site}\n"
        )

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
        """Create RotamerEnsemble from the MDAnalysis.Residue

        Parameters
        ----------
        residue :
            MDAnalysis.Residue
        **kwargs :
            

        Returns
        -------
        type
            RotamerEnsemble

        """

        res = residue.resname
        site = residue.resnum
        chain = residue.segid
        protein = residue.universe if isinstance(residue, mda.core.groups.Residue) else residue.protein
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
        if burn_in >= len(traj.universe.trajectory):
            raise ValueError("Burn in is longer than the provided trajectory.")

        chain = guess_chain(traj, site) if chain is None else chain

        altlocs = True
        if isinstance(traj, (mda.AtomGroup, mda.Universe)):
            if not hasattr(traj.universe._topology, "altLocs"):
                res = traj.select_atoms(f"segid {chain} and resnum {site}")
                altlocs = False

        if altlocs:
            res = traj.select_atoms(f"segid {chain} and resnum {site} and not altloc B")

        resname = res.residues[0].resname
        dihedral_defs = kwargs.get('dihedral_atoms', chilife.dihedral_defs.get(resname, ()))

        traj = traj.universe if isinstance(traj, mda.AtomGroup) else traj
        coords = np.array([res.atoms.positions for ts in traj.trajectory[burn_in:]])

        _, unique_idx, non_unique_idx = np.unique(coords, axis=0, return_inverse=True, return_index=True)
        coords = coords[unique_idx]

        if energy is not None:
            energy = energy[burn_in:]  # - energy[burn_in]
            energy = np.array([energy[non_unique_idx == idx].mean() for idx in range(len(unique_idx))])
            T = kwargs.setdefault("temp", 298)
            pi = np.exp(-energy / (chilife.GAS_CONST * T))
        else:
            pi = np.ones(len(traj.trajectory[burn_in:]))
            pi = np.array([pi[non_unique_idx == idx].sum() for idx in range(len(unique_idx))])

        pi /= pi.sum()
        weights = pi

        ICs = [chilife.get_internal_coords(res.atoms) for ts in traj.trajectory[unique_idx]]

        for ic in ICs:
            ic.shift_resnum(-(site - 1))

        dihedrals = np.array([ic.get_dihedral(1, dihedral_defs) for ic in ICs])
        sigmas = kwargs.get('sigmas', np.array([]))

        lib = {'rotlib': f'{resname}_from_traj',
               'resname': resname,
               'coords': np.atleast_3d(coords),
               'internal_coords': ICs,
               'weights': weights,
               'atom_types': res.types.copy(),
               'atom_names': res.names.copy(),
               'dihedral_atoms': dihedral_defs,
               'dihedrals': np.rad2deg(dihedrals),
               '_dihedrals': dihedrals.copy(),
               '_rdihedrals': dihedrals,
               'sigmas': sigmas,
               '_rsigmas': np.deg2rad(sigmas),
               'type': 'chilife rotamer library from a trajectory',
               'format_version': 1.0}

        if 'spin_atoms' in kwargs:
            spin_atoms = kwargs.pop('spin_atoms')
            if isinstance(spin_atoms, (list, tuple, np.ndarray)):
                spin_atoms = {sa: 1 / len(spin_atoms) for sa in spin_atoms}
            elif isinstance(spin_atoms, dict):
                pass
            else:
                raise TypeError('the `spin_atoms` kwarg must be a dict, list or tuple')

            lib['spin_atoms'] = np.array(list(spin_atoms.keys()))
            lib['spin_weights'] = np.array(list(spin_atoms.values()))
        kwargs.setdefault('eval_clash', False)
        return cls(resname, site, traj, chain, lib, **kwargs)

    def to_rotlib(self, libname=None):
        if libname is None:
            libname = self.name

        lib = {'rotlib': libname,
               'resname': self.res,
               'coords': self.coords,
               'internal_coords': self.internal_coords,
               'weights': self.weights,
               'atom_types': self.atom_types.copy(),
               'atom_names': self.atom_names.copy(),
               'dihedral_atoms': self.dihedral_atoms,
               'dihedrals': self.dihedrals,
               'sigmas': self.sigmas,
               'type': 'chilife rotamer library',
               'format_version': 1.0}

        if hasattr(self, 'spin_atoms'):
            lib['spin_atoms'] = self.spin_atoms
            lib['spin_weights'] = self.spin_weights

        np.savez(Path().cwd() / f'{libname}_rotlib.npz', **lib, allow_pickle=True)

    def __eq__(self, other):
        """Equivalence measurement between a spin label and another object. A SpinLabel cannot be equivalent to a
        non-SpinLabel Two SpinLabels are considered equivalent if they have the same coordinates and weights
        TODO: Future consideration -- different orders (argsort weights)"""

        if not isinstance(other, RotamerEnsemble):
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
        new_copy = chilife.RotamerEnsemble(self.res, self.site)
        for item in self.__dict__:
            if isinstance(item, np.ndarray):
                new_copy.__dict__[item] = self.__dict__[item].copy()
            if item != "protein":
                new_copy.__dict__[item] = deepcopy(self.__dict__[item])
            elif self.__dict__[item] is None:
                new_copy.protein = None
            else:
                new_copy.protein = self.protein
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
            N, CA, C = chilife.parse_backbone(self, kind="global")
        else:
            N, CA, C = site_pos

        mx, ori = chilife.global_mx(N, CA, C, method=self.alignment_method)

        # if self.alignment_method not in {'fit', 'rosetta'}:
        N, CA, C = chilife.parse_backbone(self, kind="local")
        old_ori, ori_mx = chilife.local_mx(N, CA, C, method=self.alignment_method)
        self._coords -= old_ori
        mx = ori_mx @ mx

        self._coords = np.einsum("ijk,kl->ijl", self._coords, mx) + ori

        self.ICs_to_site()

    def ICs_to_site(self):
        """ """
        # Update chain operators
        ic_backbone = np.squeeze(self.internal_coords[0].coords[:3])

        if self.alignment_method.__name__ == 'fit_alignment':
            N, CA, C = chilife.parse_backbone(self, kind="global")
            ic_backbone = np.array([[ic_backbone[0], N[1]],
                                    [ic_backbone[1], CA[1]],
                                    [ic_backbone[2], C[1]]])

        self.ic_ori, self.ic_mx = chilife.local_mx(
            *ic_backbone, method=self.alignment_method
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

                dihe = chilife.get_dihedral(p)
                ang = chilife.get_angle(p[1:])
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

                #####Should not be computing this every time
                N, CA, C = self.backbone
                mx, ori = chilife.global_mx(N, CA, C, method=self.alignment_method)
                N, CA, C = np.squeeze(self._lib_coords[0, self.backbone_idx])
                old_ori, ori_mx = chilife.local_mx(N, CA, C, method=self.alignment_method)
                ######

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
                            self._lib_coords[:, mask] = pos[0] if len(pos.shape) > 1 else pos

            return np.squeeze(self._lib_coords[idx]), np.squeeze(self._weights[idx])

        if len(self.dihedral_atoms) == 0:
            return self._coords, self.weights, self.internal_coords
        elif hasattr(self, "internal_coords"):
            returnables = self._off_rotamer_sample(idx, off_rotamer, **kwargs)
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
            new_dihedrals = np.random.random(len(idx), len(off_rotamer)) * 2 * np.pi
            new_weights = np.ones(len(idx))

        #  sample from von mises near rotamer unless more information is provided
        elif self.skews is None:
            new_dihedrals = np.random.vonmises(self._rdihedrals[idx][:, off_rotamer],
                                               self._rkappas[idx][:, off_rotamer])
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

        ICs = [self._lib_IC[iidx].copy() for iidx in idx]
        ICs = [IC.set_dihedral(new_dihedrals[i], 1, self.dihedral_atoms[off_rotamer]) for i, IC in enumerate(ICs)]

        Zmat_idxs = np.array([IC.zmat_idxs[1] for IC in ICs])
        Zmats = np.array([IC.zmats[1] for IC in ICs])
        coords = batch_ic2cart(Zmat_idxs, Zmats)
        mx, ori = ICs[0].chain_operators[1]["mx"], ICs[0].chain_operators[1]["ori"]
        coords = np.einsum("ijk,kl->ijl", coords, mx) + ori
        for i, IC in enumerate(ICs):
            IC._coords = coords[i]
        coords = coords[:, self.ic_mask]

        if kwargs.setdefault("return_dihedrals", False):
            return coords, new_weights, ICs
        else:
            return coords, new_weights

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
            temp_ensemble._coords = np.atleast_3d([coords[self.ic_mask]])
            return self.energy_func(temp_ensemble)

        temp_ensemble = self.copy()
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
                -xopt.fun / (chilife.GAS_CONST * 298)
            )

        self.weights /= self.weights.sum()
        self.trim_rotamers()

    def trim_rotamers(self):
        """Remove rotamers with small weights from ensemble"""
        arg_sort_weights = np.argsort(self.weights)[::-1]
        sorted_weights = self.weights[arg_sort_weights]
        cumulative_weights = np.cumsum(sorted_weights)
        cutoff = np.maximum(1, len(cumulative_weights[cumulative_weights < 1 - self.trim_tol]))
        keep_idx = arg_sort_weights[:cutoff]

        self._coords = self._coords[keep_idx]
        self._dihedrals = self._dihedrals[keep_idx]
        self.weights = self.weights[keep_idx]
        if len(arg_sort_weights) == len(self.internal_coords):
            self.internal_coords = [self.internal_coords[x] for x in keep_idx]

        # Renormalize weights
        self.weights /= self.weights.sum()

        logging.info(
            f"{len(self.weights)} of {len(self._weights)} {self.res} rotamers make up at least "
            f"{100 * (1-self.trim_tol):.2f}% of rotamer density for site {self.site}"
        )

    def centroid(self):
        """Get the centroid of the whole rotamer ensemble."""
        return self._coords.mean(axis=(0, 1))

    def evaluate(self):
        """Place rotamer ensemble on protein site and recalculate rotamer weights."""

        # Calculate external energies
        energies = self.energy_func(self)

        # Calculate total weights (combining internal and external)
        self.weights, self.partition = chilife.reweight_rotamers(energies, self.temp, self.weights)
        logging.info(f"Relative partition function: {self.partition:.3}")

        # Remove low-weight rotamers from ensemble
        self.trim_rotamers()

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
        chilife.save_ensemble(name, self.atoms, self._coords)

    @property
    def backbone(self):
        """Backbone coordinates of the spin label"""
        return np.squeeze(self._coords[0][self.backbone_idx])

    def get_lib(self, rotlib):
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
            else np.rad2deg(chilife.get_dihedral(PhiSel.positions))
        )
        Psi = (
            None
            if PsiSel is None
            else np.rad2deg(chilife.get_dihedral(PsiSel.positions))
        )
        self.Phi, self.Psi = Phi, Psi

        # Get library
        logging.info(f"Using backbone dependent library with Phi={Phi}, Psi={Psi}")
        cwd = Path().cwd()

        was_none = True if rotlib is None else False
        rotlib = self.res if rotlib is None else rotlib

        # If the rotlib isn't a dict try to figure out where it is
        if not isinstance(rotlib, dict):
            # Assemble a list of possible rotlib paths
            possible_rotlibs = [Path(rotlib),
                                cwd / rotlib,
                                cwd / (rotlib + '.npz'),
                                cwd / (rotlib + '_rotlib.npz')]
            possible_rotlibs += list(Path.cwd().glob(f'*{rotlib}*.npz'))
            # Check if any exist
            for possible_file in possible_rotlibs:
                if possible_file.exists() and possible_file.suffix == '.npz':
                    rotlib = possible_file
                    break
            #  If non exist
            else:
                # if rotlib wasnt specified look for default  in chilife/permanent libraries
                if rotlib in chilife.USER_LIBRARIES and was_none:
                    rotlib = chilife.RL_DIR / 'user_rotlibs' / (chilife.rotlib_defaults[self.res][0] + '_rotlib.npz')
                # If rotlib was specified look for a rotlib with that name
                elif rotlib in chilife.USER_LIBRARIES:
                    rotlib = chilife.RL_DIR / 'user_rotlibs' / (rotlib + '_rotlib.npz')
                # If rotlib is not in USER_LIBRARIES or SUPPOERTED LABELS throw and error.
                elif rotlib not in chilife.SUPPORTED_RESIDUES:
                    raise NameError(f'There is no rotamer library called {rotlib} in this directory or in chilife')

            # Read the library
            lib = chilife.read_library(rotlib, Phi, Psi)

        # If rotlib is a dict, assume that dict is in the format that read_library`would return
        else:
            lib = rotlib

        # Copy mutable library values (mostly np arrays) so that instances do not share data
        lib = {key: value.copy() if hasattr(value, 'copy') else value for key, value in lib.items()}

        # Perform a sanity check to ensure every necessary entry is present
        if isinstance(rotlib, Path) and not all(x in lib for x in chilife.rotlib_formats[lib['format_version']]):
            raise ValueError('The rotamer library does not contain all the required entries for the format version')

        # Deep copy (mutable)  internal coords.
        lib['internal_coords'] = [a.copy() for a in lib['internal_coords']]

        # Modify library to be appropriately used with self.__dict__.update
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

        # Evaluate external clash energies and reweight rotamers
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
        mx, ori = chilife.global_mx(*chilife.parse_backbone(self, kind="local"), method=self.alignment_method)
        return mx

    @property
    def origin(self):
        """ """
        return np.squeeze(self.backbone[1])

    def get_sasa(self):
        """ """
        atom_radii = chilife.get_lj_rmin(self.atom_types)
        if self.protein is not None:
            environment_coords = self.protein.atoms[self.protein_clash_idx].positions
            environment_radii = chilife.get_lj_rmin(self.protein.atoms[self.protein_clash_idx].types)
        else:
            environment_coords = np.empty((0, 3))
            environment_radii = np.empty(0)

        SASAs = chilife.numba_utils.get_sasa(self.coords, atom_radii, environment_coords, environment_radii)

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
        "alignment_method": "bisect",
        "dihedral_sigmas": 35,
        "weighted_sampling": False,
        "eval_clash": False,
        "use_H": False,
        "_sample_size": kwargs.pop("sample", False),
        "energy_func": chilife.get_lj_rep,
        "trim_tol": 0.005,
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
