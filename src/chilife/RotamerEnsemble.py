import inspect
import re
import warnings
from copy import deepcopy
from functools import partial
from pathlib import Path
import numpy as np
from numpy.typing import ArrayLike
from itertools import combinations
from scipy.spatial import cKDTree
import igraph as ig
from scipy.stats import skewnorm, circstd
import scipy.optimize as opt
import MDAnalysis as mda


import chilife.io as io
import chilife.scoring as scoring

from .globals import SUPPORTED_RESIDUES, nataa_codes, dihedral_defs, ralt_prot_states
from .scoring import get_lj_rep, GAS_CONST
from .numba_utils import get_sasa as nu_getsasa
from .alignment_methods import alignment_methods, parse_backbone, local_mx, global_mx
from .protein_utils import FreeAtom, guess_mobile_dihedrals, get_dihedral, get_angle
from .pdb_utils import get_bb_candidates, get_backbone_atoms
from .MolSys import MolSys, MolecularSystemBase
from .MolSysIC import MolSysIC

default_energy_func = scoring.ljEnergyFunc()

class RotamerEnsemble:
    """Create new RotamerEnsemble object.

    Parameters
    ----------
    res : string
        3-character name of desired residue, e.g. R1A.
    site : int
        MolSys residue number to attach library to.
    protein : MDAnalysis.Universe, MDAnalysis.AtomGroup
        Object containing all protein information (coords, atom types, etc.)
    chain : str
        MolSys chain identifier to attach spin label to.
    rotlib : str
        Rotamer library to use for constructing the RotamerEnsemble
    **kwargs : dict
        minimize: bool
            Switch to turn on/off minimization. During minimization each rotamer is optimized in dihedral space
            with respect to both internal and external clashes, and deviation from the starting conformation in
            dihedral space.
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
            Name of the forcefield you wish to use to parameterize atoms for the energy function. Currently, supports
            `charmm` and `uff`
        energy_func : callable
            Python function or callable object that takes a protein or a RotamerEnsemble object as input and
            returns an energy value (kcal/mol) for each atom of each rotamer in the ensemble. See also
            :mod:`Scoring <chiLife.scoring>` . Defaults to a capped lennard-jones repulsive energy function.
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
        ignore_wateres : bool
            Determines whether water molecules (resnames HOH and OH2) are ignored during clash evaluation.
            Defaults to True.
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
           Defaults to False.
        use_H : bool
           Determines if hydrogen atoms are used or not. Defaults to False.
       """


    def __init__(self, res, site=None, protein=None, chain=None, rotlib=None, **kwargs):


        self.res = res
        if site is None and protein is not None:
            raise ValueError('A protein has been provided but a site has not. If you wish to construct an ensemble '
                             'associated with a protein you must include the site you wish to model.')
        elif site is None:
            site = 1
            icode = ""
        elif isinstance(site, str):
            site_split = re.split(r'(\D+)',site)
            site = site_split[0]
            icode = site_split[1] if len(site_split) > 1 else ""
        else:
            icode = ""


        self.site = site
        self.icode = icode
        self.resnum = self.site
        self.chain = chain if chain is not None else guess_chain(protein, self.site)
        self.selstr = f"resid {self.site} and segid {self.chain} and not altloc B"
        self.nataa = ""
        self.input_kwargs = kwargs
        self.__dict__.update(assign_defaults(kwargs))
        self.input_kwargs.pop('energy_func', None)
        self.protein = protein

        # Convert string arguments for alignment_method to respective function
        if isinstance(self.alignment_method, str):
            self.alignment_method = alignment_methods[self.alignment_method]

        lib = self.get_lib(rotlib)
        self.__dict__.update(lib)
        self._weights = self.weights / self.weights.sum()

        if len(self.sigmas) != 0 and 'dihedral_sigmas' not in kwargs:
            self.sigmas[self.sigmas == 0] = self.dihedral_sigmas
        else:
            self.set_dihedral_sampling_sigmas(self.dihedral_sigmas)

        self._rsigmas = np.deg2rad(self.sigmas)
        with np.errstate(divide='ignore'):
            self._rkappas = 1 / self._rsigmas**2

        self._rkappas = np.clip(self._rkappas, 0, 1e16)

        # Remove hydrogen atoms unless otherwise specified
        if not self.use_H:
            self.H_mask = self.atom_types != "H"
            self._coords = self._coords[:, self.H_mask]
            self.atom_types = self.atom_types[self.H_mask]
            self.atom_names = self.atom_names[self.H_mask]

        self.ic_mask = np.argwhere(np.isin(self.internal_coords.atom_names, self.atom_names)).flatten()
        self._lib_coords = self._coords.copy()
        self._lib_dihedrals = self._dihedrals.copy()
        self._lib_IC = self.internal_coords

        if self.clash_radius is None:
            self.clash_radius = np.linalg.norm(self.clash_ori - self.coords, axis=-1).max() + 5

        # Parse important indices
        self.aln_idx = np.squeeze(np.argwhere(np.isin(self.atom_names, self.aln_atoms)))
        self.backbone_idx = np.squeeze(np.argwhere(np.isin(self.atom_names, self.backbone_atoms)))
        self.side_chain_idx = np.argwhere(np.isin(self.atom_names, self.backbone_atoms, invert=True)).flatten()

        self._graph = ig.Graph(edges=self.bonds)
        self.aidx, self.bidx = [list(x) for x in zip(*self.non_bonded)]

        # Allocate variables for clash evaluations
        self.atom_energies = None
        self.clash_ignore_idx = None
        self.partition = 1

        # Assign a name to the label
        self.name = self.rotlib
        if self.site is not None:
            self.name = self.nataa + str(self.site) + self.rotlib
        if self.chain not in ('A', None):
            self.name += f"_{self.chain}"

        self.update(no_lib=True)

        # Store atom information as atom objects
        self.atoms = [
            FreeAtom(name, atype, idx, self.res, self.site, coord)
            for idx, (coord, atype, name) in enumerate(
                zip(self._coords[0], self.atom_types, self.atom_names)
            )
        ]

    def __str__(self):
        return (
            f"Rotamer ensemble with {np.size(self.weights)} members\n" +
            f"  Name: {self.name}\n" +
            f"  Label: {self.res}\n" +
            f"  Site: {self.site}\n" +
            f"  Dihedral definitions:\n" +
            f"\n".join([f'    {d}' for d in self.dihedral_atoms]) +
            f"\n"
        )

    def __repr__(self):
        return str(self)

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
        # TODO: create pdb reader and implement from_pdb method
        raise NotImplementedError('Creation of a RotamerEnsemble from a PDB file is not yet implemented.')

        return cls(res, site, protein, chain)

    @classmethod
    def from_mda(cls, residue, **kwargs):
        """Create RotamerEnsemble from the MDAnalysis.Residue for chiLife.Residue

        Parameters
        ----------
        residue : MDAnalysis.Residue, chiLife.Residue
            A Residue object from which to create the RotamerEnsemble.
        **kwargs : dict
            Additional keyword arguments to use for creating the RotamerEnsemble.

        Returns
        -------
        cls : RotamerEnsemble
            A RotamerEnsemble object created from the provided Residue object.
        """

        res = residue.resname


        if kwargs.get('use_H', False) and res in ralt_prot_states:
            if res != 'HIS':
                res = ralt_prot_states[res].get(len(residue.atoms), res)
            elif len(residue.atoms) == 18:
                res = 'HIP'
            elif len([x for x in residue.atoms.names if 'HE' in x]) == 2:
                res = 'HIE'
            elif len([x for x in residue.atoms.names if 'HD' in x]) == 2:
                res = 'HID'
            else:
                res = 'HIS'


        site = residue.resid
        chain = residue.segid
        protein = residue.universe
        return cls(res, site, protein, chain, **kwargs)

    @classmethod
    def from_trajectory(cls, traj, site, chain=None, energy=None, burn_in=0, **kwargs):
        """
        Create a RotamerEnsemble object from a trajectory.

        Parameters
        ----------
        traj : MDAnalysis.Universe, MDAnalysis.AtomGroup, chiLife.MolecularSystemBase
            The trajectory over which the RotamerEnsemble is to be created.
        site : res
            Residue number of the residue to create a rotamer ensemble for.
        chain : str
            Chain identifier of the residue to create a RotamerEnsemble for.
        energy : ArrayLike
            Potential energy of the residue at each frame of the trajectory in kcal/mol
        burn_in : int
             Number of frames to skip from the trajectory. Used to skip burn-in period when performing MCMC sampling.
        **kwargs : dict
            Additional keyword arguments to use for creating the RotamerEnsemble.

            dihedral_atoms : list
                A list of atoms defining the mobile dihedral angles of the residue. Primarily used when creating an
                ensemble of a residue type chiLife is unaware of (see :ref:`mobile_dihedrals`
            temp : float
                Temperature to use when evaluating conformational populations from user supplied ``energy`` argument.
            spin_atoms : list, dict
                A list or dict defining the spin atoms of the label and their relative populations (see
                :ref:`spin_atoms`).

        Returns
        -------
        cls : RotamerEnsemble
            A RotamerEnsemble object created from the provided trajectory.
        """
        if burn_in >= len(traj.universe.trajectory):
            raise ValueError("Burn in is longer than the provided trajectory.")

        chain = guess_chain(traj, site) if chain is None else chain

        if isinstance(traj, (mda.AtomGroup, mda.Universe)):
            if not hasattr(traj.universe._topology, "altLocs"):
                traj.add_TopologyAttr('altLocs', ["A"] * len(traj.atoms))

        res = traj.select_atoms(f"segid {chain} and resid {site} and not altloc B")

        resname = res.residues[0].resname
        ddefs = kwargs.pop('dihedral_atoms', dihedral_defs.get(resname, ()))

        traj = traj.universe if isinstance(traj, mda.AtomGroup) else traj
        coords = np.array([res.atoms.positions for ts in traj.trajectory[burn_in:]])

        _, unique_idx, non_unique_idx = np.unique(coords, axis=0, return_inverse=True, return_index=True)
        coords = coords[unique_idx]

        if energy is not None:
            energy = energy[burn_in:]  # - energy[burn_in]
            energy = np.array([energy[non_unique_idx == idx].mean() for idx in range(len(unique_idx))])
            T = kwargs.setdefault("temp", 298)
            pi = np.exp(-energy / (scoring.GAS_CONST * T))
        else:
            pi = np.ones(len(traj.trajectory[burn_in:]))
            pi = np.array([pi[non_unique_idx == idx].sum() for idx in range(len(unique_idx))])

        pi /= pi.sum()
        weights = pi

        if not kwargs.setdefault('use_H', True):
            res = res.select_atoms('not type H')

        res = MolSys.from_atomsel(res, frames = unique_idx)

        ICs = MolSysIC.from_atoms(res)
        ICs.shift_resnum(-(site - 1))

        if ddefs == ():
            ddefs = guess_mobile_dihedrals(ICs)

        dihedrals = np.array([ic.get_dihedral(1, ddefs) for ic in ICs])
        sigmas = kwargs.get('sigmas', np.array([]))

        bb_candidates = get_bb_candidates(res.names, resname)
        aln_atoms = kwargs.get('ln_atoms', [])
        if not bb_candidates and not aln_atoms:
            raise RuntimeError('No suitable backbone candidates were found. Please use from_trajectory on residues with'
                               'standard backbone atom names or specify alignment backbone atoms with the aln_atoms '
                               'keyword argument.')
        elif aln_atoms:
            graph = res.topology.graph
            root_idx = np.argwhere(res.names == aln_atoms[1]).flat[0]
            aname_lst = ICs.atom_names.tolist()
            neighbor_idx = [aname_lst.index(a) for a in aln_atoms[::2]]
            backbone_atoms = get_backbone_atoms(graph, root_idx, neighbor_idx)

        else:
            backbone_atoms = [b for b in bb_candidates if b in res.names]
            aln_atoms = ['N', 'CA', 'C'] if np.all(np.isin(['N', 'CA', 'C'], backbone_atoms)) else ["O4'", "C1'", "C2'"]

        lib = {'rotlib': f'{resname}_from_traj',
               'resname': resname,
               'coords': np.atleast_3d(coords),
               'internal_coords': ICs,
               'weights': weights,
               'atom_types': res.types.copy(),
               'atom_names': res.names.copy(),
               'dihedral_atoms': ddefs,
               'dihedrals': np.rad2deg(dihedrals),
               'aln_atoms': aln_atoms,
               'backbone_atoms': backbone_atoms,
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

        elif lib_path := io.get_possible_rotlibs(resname, suffix='rotlib', extension='.npz'):
            with np.load(lib_path) as f:
                if 'spin_atoms' in f:
                    lib['spin_atoms'] = f['spin_atoms']
                    lib['spin_weights'] = f['spin_weights']

        kwargs.setdefault('eval_clash', False)
        kwargs.setdefault('_match_backbone', False)
        return cls(resname, site, chain=chain, rotlib=lib, **kwargs)

    def update(self, no_lib=False):
        # Sample from library if requested
        if self._sample_size and len(self.dihedral_atoms) > 0:
            mask = self.sigmas.sum(axis=0) == 0

            # Draw samples
            coords, weights, internal_coords = self.sample(self._sample_size, off_rotamer=~mask, return_dihedrals=True)

            # Remove structures with internal clashes
            dist = np.linalg.norm(coords[:, self.aidx] - coords[:, self.bidx], axis=2)
            sidx = np.atleast_1d(np.squeeze(np.argwhere(np.all(dist > 2, axis=1))))
            self.internal_coords = internal_coords
            self.internal_coords.use_frames(sidx)
            dihedrals = np.asarray(
                [self.internal_coords.get_dihedral(1, self.dihedral_atoms) for ts in self.internal_coords.trajectory]
            )
            self._dihedrals = np.rad2deg(dihedrals)
            self._coords, self.weights = coords[sidx], weights[sidx]

        elif not no_lib:
            # Reset to library
            self._coords = self._lib_coords.copy()
            self._dihedrals = self._lib_dihedrals.copy()
            self.internal_coords = self._lib_IC
            self.weights = self._weights.copy()

        if self.protein is not None:
            self.protein_setup()
        else:
            self.ICs_to_site(self.origin, self.mx)

    def protein_setup(self):
        self.to_site()

        # Get weight of current or closest rotamer
        clash_ignore_idx = self.protein.select_atoms(f"resid {self.site} and segid {self.chain}").ix
        self.clash_ignore_idx = np.argwhere(np.isin(self.protein.ix, clash_ignore_idx)).flatten()
        protein_clash_idx = self.protein_tree.query_ball_point(self.clash_ori, self.clash_radius)
        self.protein_clash_idx = [idx for idx in protein_clash_idx if idx not in self.clash_ignore_idx]

        if hasattr(self.energy_func, 'prepare_system'):
            self.energy_func.prepare_system(self)

        if self._coords.shape[1] == len(self.clash_ignore_idx):
            RMSDs = np.linalg.norm(
                self._coords - self.protein.atoms[self.clash_ignore_idx].positions[None, :, :],
                axis=(1, 2)
            )

            idx = np.argmin(RMSDs)
            self.current_weight = self.weights[idx]
        else:
            self.current_weight = 0

        # Evaluate external clash energies and reweigh rotamers
        if self._minimize and self.eval_clash:
            raise RuntimeError('Both `minimize` and `eval_clash` options have been selected, but they are incompatible.'
                               'Please select only one. Also note that minimize performs its own clash evaluations so '
                               'eval_clash is not necessary.')
        elif self.eval_clash:
            self.evaluate()

        elif self._minimize:
            self.minimize()

    def to_rotlib(self,
                  libname: str = None,
                  description: str = None,
                  comment: str = None,
                  reference: str = None) -> None:
        """
        Saves the current RotamerEnsemble as a rotamer library file.

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
            libname = self.name.lstrip('0123456789.- ')

        if description is None:
            description = (f'Rotamer library made with chiLife using `to_rotlib` method'
                           f'of a rotamer ensemble.')
        ICs = self.internal_coords.copy()

        # Remove chain operators to align all labels on backbone
        # ICs.chain_operators = None
        coords = ICs.protein.trajectory.coordinate_array
        bb_idx = np.argwhere(np.isin(ICs.atom_names, ['N', 'CA', 'C'])).flatten()
        for i in range(len(coords)):
            ori, mx = local_mx(*coords[i, bb_idx])
            coords[i] = (coords[i] - ori) @ mx

        lib = {'rotlib': libname,
               'resname': self.res,
               'coords': coords,
               'internal_coords': ICs,
               'weights': self.weights,
               'atom_types': self.atom_types.copy(),
               'atom_names': self.atom_names.copy(),
               'dihedral_atoms': self.dihedral_atoms,
               'dihedrals': self.dihedrals,
               'aln_atoms': self.aln_atoms,
               'backbone_atoms': self.backbone_atoms,
               'sigmas': self.sigmas,
               'type': 'chilife rotamer library',
               'description': description,
               'comment': comment,
               'reference': reference,
               'format_version': 1.4}

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
        """Assign RotamerLibrary to a site.

        Parameters
        ----------
        site : int
            Residue number to assign SpinLabel to.
        """
        self.site = site
        self.to_site()

    def copy(self, site: int = None, chain: str = None, rotlib: dict = None):
        """Create a deep copy of the spin label. Assign new site, chain or rotlib  if desired. Useful when labeling
        homo-oligomers or proteins with repeating units/domains.

        Parameters
        ----------
        site : int
             New site number for the copy.
        chain : str
             New chain identifier for the copy.
        rotlib : dict
            New (base) rotamer library for the dict. Used primarily when copying dRotamerEnsembels.

        Returns
        -------
        new_copy : chilife.RotamerEnsemble
            A deep copy of the original RotamerLibrary
        """
        new_copy = self._base_copy(self._rotlib)
        for item in self.__dict__:
            if isinstance(item, np.ndarray) or item == 'internal_coords':
                new_copy.__dict__[item] = self.__dict__[item].copy()
            elif item not in  ("_protein", '_lib_IC', '_rotlib'):
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
        return RotamerEnsemble(self.res, self.site, rotlib=rotlib, chain=self.chain)

    def to_site(self, site_pos: ArrayLike = None) -> None:
        """Move spin label to new site

        Parameters
        ----------
        site_pos : ArrayLike
            3x3 array of ordered backbone atom coordinates of new site (N CA C) (Default value = None)
        """
        if site_pos is None:
            bbs = parse_backbone(self, kind="global")
        else:
            bbs = site_pos

        if len(bbs) < 3 :
            raise RuntimeError(f'The residue/rotlib you have selected {self.res}, does not share a compatible backbone '
                               f'with the residue you are trying to label. Check the site and rotlib and try again.\n'
                               f'The label backbone atoms: {self.backbone_atoms}')

        mx, ori = global_mx(*bbs, method=self.alignment_method)

        # if self.alignment_method not in {'fit', 'rosetta'}:
        bbs = parse_backbone(self, kind="local")
        old_ori, ori_mx = local_mx(*bbs, method=self.alignment_method)
        self._coords -= old_ori
        cmx = ori_mx @ mx

        self._coords = np.einsum("ijk,kl->ijl", self._coords, cmx) + ori
        self.ICs_to_site(ori, mx)
        if self._match_backbone:
            self.backbone_to_site()

    def ICs_to_site(self, cori, cmx):
        """ Modify the internal coordinates to be aligned with the site that the RotamerEnsemble is attached to"""

        # Update chain operators
        ic_backbone = np.squeeze(self.internal_coords.coords[:3])

        if self.alignment_method.__name__ == 'fit_alignment':
            N, CA, C = parse_backbone(self, kind="global")
            ic_backbone = np.array([[ic_backbone[0], N[1]],
                                    [ic_backbone[1], CA[1]],
                                    [ic_backbone[2], C[1]]])

        self.ic_ori, self.ic_mx = local_mx(*ic_backbone, method=self.alignment_method)
        m2m3 = self.ic_mx @ self.mx
        op = {}

        new_mx = self.internal_coords.chain_operators[0]["mx"] @ m2m3
        new_ori = (self.internal_coords.chain_operators[0]["ori"] - self.ic_ori) @ m2m3 + self.origin
        op[0] = {"mx": new_mx, "ori": new_ori}

        self.internal_coords.chain_operators = [op]

        # Update backbone conf
        alist = ["O"] if not self.use_H else ["H", 'O']
        for atom in alist:
            mask = self.internal_coords.atom_names == atom
            if any(mask) and self.protein is not None:
                idx = np.argwhere(self.internal_coords.atom_names == atom).flat[0]
                dihe_def = self.internal_coords.z_matrix_names[idx]
                p = self.protein.select_atoms(
                    f"segid {self.chain} and resid {self.site} and name {' '.join(dihe_def)} and not altloc B"
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

                dihe = get_dihedral(p)
                ang = get_angle(p[1:])
                bond = np.linalg.norm(p[-1] - p[-2])
                idx = np.squeeze(np.argwhere(mask))
                chain = self.internal_coords.chains[0]
                resnum = self.internal_coords.atoms[0].resid
                if atom == 'O' and (tag := (chain, resnum, 'C', 'CA')) in self.internal_coords.chain_res_name_map:
                    additional_idxs = self.internal_coords.chain_res_name_map[tag]

                delta = self.internal_coords.trajectory.coordinate_array[:, idx, 2] - dihe
                self.internal_coords.trajectory.coordinate_array[:, idx] = bond, ang, dihe
                if atom == "O" and 'additional_idxs' in locals():
                    self.internal_coords.trajectory.coordinate_array[:, additional_idxs, 2] -= delta[:, None]

    def backbone_to_site(self):
        """Modify backbone atoms to match the site that the RotamerEnsemble is attached to """
        # Keep protein backbone dihedrals for oxygen and hydrogen atoms

        for atom in self.backbone_atoms:

            if atom in self.aln_atoms: continue  # TODO update test to change this
            mask = self.atom_names == atom
            if any(mask) and self.protein is not None:

                pos = self.protein.select_atoms(
                    f"segid {self.chain} and resid {self.site} "
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
        n : int
             number of rotamers to sample.
        off_rotamer : bool | List[bool]
            If False the rotamer library used to construct the RotamerEnsemble will be sampled using the exact
            dihedrals. If True, all mobile dihedrals will undergo minor perturbations. A list indicates a per-dihedral
            switch to turn on/off off-rotamer sampling of a particular dihedrals, corresponding to `self.dihedral_atoms`
            e.g. `off_+rotamer = [False, False, False, True, True]` for R1 will only sample off-rotamers for χ4 and χ5.
        **kwargs : dict
            return_dihedrals : bool
                If True, sample will return a MolSysIC object of the sampled rotamer

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
            if not np.allclose(np.squeeze(self._lib_coords[0, self.backbone_idx]), self.backbone):

                #####Should not be computing this every time
                N, CA, C = self.aln
                mx, ori = global_mx(N, CA, C, method=self.alignment_method)
                N, CA, C = np.squeeze(self._lib_coords[0, self.aln_idx])
                old_ori, ori_mx = local_mx(N, CA, C, method=self.alignment_method)
                ######

                self._lib_coords -= old_ori
                mx = ori_mx @ mx

                self._lib_coords = np.einsum("ijk,kl->ijl", self._lib_coords, mx) + ori
                for atom in ["H", "O"]:
                    mask = self.atom_names == atom
                    if any(mask) and self.protein is not None:
                        pos = self.protein.select_atoms(
                            f"segid {self.chain} and resid {self.site} "
                            f"and name {atom} and not altloc B"
                        ).positions
                        if len(pos) > 0:
                            self._lib_coords[:, mask] = pos[0] if len(pos.shape) > 1 else pos

            return np.squeeze(self._lib_coords[idx]), np.squeeze(self._weights[idx])

        if len(self.dihedral_atoms) == 0:
            return self._coords, self.weights, self.internal_coords
        elif hasattr(self, "internal_coords"):
            returnables = self._off_rotamer_sample(idx, off_rotamer, **kwargs)
            return [np.squeeze(x) if isinstance(x, np.ndarray) else x for x in returnables]

        else:
            raise AttributeError(
                f"{type(self)} objects require both internal coordinates ({type(self)}.internal_coords "
                f"attribute) and dihedral standard deviations ({type(self)}.sigmas attribute) to "
                f"perform off rotamer sampling"
            )

    def _off_rotamer_sample(self, idx, off_rotamer, **kwargs):
        """Perform off rotamer sampling. Primarily a helper function for `RotamerEnsemble.sample()`

        Parameters
        ----------
        idx : int
             Index of the rotamer of the rotamer library that will be perturbed.
        off_rotamer : List[bool]
             A list indicates a per-dihedral bools indicating which dihedral angles of `self.dihedral_atoms` will
             have off rotamer sampling. e.g. `off_+rotamer = [False, False, False, True, True]` for R1 will only sample
             off-rotamers for χ4 and χ5.
        **kwargs : dict
            return_dihedrals : bool
                If True, sample will return a MolSysIC object of the sampled rotamer

        Returns
        -------
        coords : ArrayLike
            The 3D coordinates of the sampled rotamer(s)
        new_weights : ArrayLike
            New weights (relative) of the sampled rotamer(s)
        ICs : List[chilife.MolSysIC] (Optional)
            Internal coordinate (MolSysIC) objects of the rotamer(s).
        """

        # Use accessible volume sampling if only provided a single rotamer
        if len(self._weights) == 1 or np.all(np.isinf(self.sigmas)):
            new_dihedrals = np.random.random((len(idx), len(off_rotamer))) * 2 * np.pi
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

        z_matrix = self._lib_IC.batch_set_dihedrals(idx, new_dihedrals, 1, self.dihedral_atoms[off_rotamer])
        ICs = self.internal_coords.copy()
        ICs.load_new(z_matrix)
        coords = ICs.protein.trajectory.coordinate_array[:, self.ic_mask]

        if kwargs.setdefault("return_dihedrals", False):
            return coords, new_weights, ICs
        else:
            return coords, new_weights

    def update_weight(self, weight: float) -> None:
        """
         Function to assign `self.current_weight`, which is the estimated weight of the rotamer currently occupying the
         attachment site on `self.protein`. This is only relevant if the residue type on `self.protein` is the same as
         the  RotamerLibrary.

        Parameters
        ----------
        weight : float
            New weight for the current residue.
        """
        self.current_weight = weight

    def minimize(self, callback=None):
        """
        Minimize the energy of the rotamers in dihedral space and re-weight with respect to the new energies.

        Parameters
        ----------
        callback : callable
            A callable function to be passed as the ``scipy.optimize.minimize`` function. See the
            `scipy documentation <https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html>`_
            for details.
        """
        dummy = self.copy()

        scores = np.array([self._min_one(i, ic, dummy, callback=callback) for i, ic in enumerate(self.internal_coords)])
        scores -= scores.min()

        self.weights *= np.exp(-scores / (GAS_CONST * self.temp) / np.exp(-scores).sum())
        self.weights /= self.weights.sum()
        if self._do_trim:
            self.trim_rotamers()

    def _objective(self, dihedrals, ic1, dummy):
        """
        Objective function for the minimization procedure.

        Parameters
        ----------
        dihedrals : ArrayLike
            Array of dihedral angles to generate a score for.
        ic1 : chilife.MolSysIC
            Internal coordinate objects
        dummy : chilife.RotamerEnsemble
            A dummy ensemble to use for applying dihedrals to and testing the score so that the parent rotamer ensemble
            does not need to be modified.

        Returns
        -------
        energy : float
            The "energy" score of the rotamer with the provided dihedrals.
        """

        ic1.set_dihedral(dihedrals[: len(self.dihedral_atoms)], 1, self.dihedral_atoms)
        coords = ic1.to_cartesian()[self.ic_mask]
        dummy._coords = np.atleast_3d([coords[self.ic_mask]])
        r = np.linalg.norm(coords[self.aidx] - coords[self.bidx], axis=1)

        # Faster to compute lj here
        lj = self.irmin_ij / r
        lj = lj * lj *lj
        lj = lj * lj

        # attractive forces are needed, otherwise this term will perpetually push atoms apart
        internal_energy = self.ieps_ij * (lj * lj - 2 * lj)
        external_energy = self.energy_func(dummy)
        energy = external_energy.sum() + internal_energy.sum()
        return energy

    def _min_one(self, i, ic, dummy, callback=None):
        """
        Perform a single minimization on a member of the underlying rotamer library in dihedral space.

        Parameters
        ----------
        i : int
            index of the rotamer (of the underlying library) to be minimized.
        ic : chilife.MolSysIC
            Internal coordinate object of the rotamer to be minimized.
        dummy : chilife.RotamerEnsemble
            A dummy ensemble to perform manipulations on while minimizing.

        Returns
        -------
        energy : float
            The final minimized "energy" of the objective function plus a modifier based on the deviations from the
            parent rotamer in the rotamer library.
        """
        if callback is not None:
            if 'i' in inspect.signature(callback).parameters:
                callback = partial(callback, i=i)

        d0 = ic.get_dihedral(1, self.dihedral_atoms)

        lb = d0 - np.deg2rad(40)
        ub = d0 + np.deg2rad(40)  #
        bounds = np.c_[lb, ub]
        xopt = opt.minimize(self._objective, x0=d0, args=(ic, dummy),
                            bounds=bounds, method=self.min_method,
                            callback=callback)
        self._coords[i] = ic.coords[self.ic_mask]
        tors = d0 - xopt.x
        tors = np.arctan2(np.sin(tors), np.cos(tors))
        tors = np.sqrt(tors @ tors)
        energy = xopt.fun + tors
        return energy

    def trim_rotamers(self, keep_idx: ArrayLike = None) -> None:
        """
        Remove rotamers with small weights from ensemble

        Parameters
        ----------
        keep_idx : ArrayLike[int]
            Indices of rotamers to keep. If None rotamers will be trimmed based off of `self.trim_tol`.
        """
        if keep_idx is None:
            arg_sort_weights = np.argsort(self.weights)[::-1]
            sorted_weights = self.weights[arg_sort_weights]
            cumulative_weights = np.cumsum(sorted_weights)
            cutoff = np.maximum(1, len(cumulative_weights[cumulative_weights < 1 - self.trim_tol]))
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

    @property
    def centroid(self):
        """Get the centroid of the whole rotamer ensemble."""
        return self._coords.mean(axis=(0, 1))

    def evaluate(self):
        """Place rotamer ensemble on protein site and recalculate rotamer weights."""

        # Calculate external energies
        energies = self.energy_func(self)

        # Calculate total weights (combining internal and external)
        self.weights, self.partition = scoring.reweight_rotamers(energies, self.temp, self.weights)

        # Remove low-weight rotamers from ensemble
        if self._do_trim:
            self.trim_rotamers()

    def save_pdb(self, name: str = None) -> None:
        """
        Save a pdb file containing the rotamer ensemble.

        Parameters
        ----------
        name : str | None
            Name of the file. If `None` self.name will be used.
        """
        io.save(name, self)

    @property
    def backbone(self):
        """Backbone coordinates of the spin label"""
        return np.squeeze(self._coords[0][self.backbone_idx])

    @property
    def aln(self):
        return np.squeeze(self._coords[0][self.aln_idx])


    @property
    def side_chain(self):
        return np.squeeze(self._coords[0][self.side_chain_idx])


    def get_lib(self, rotlib):
        """Parse backbone information from protein and fetch the appropriate rotamer library.

        Parameters
        ----------
            rotlib : str | dict
                The name of the rotlib or a dictionary containing all the information required for a rotamer library.
        Returns
        -------
        lib : dict
            Dictionary containing all underlying attributes of a rotamer library.
        """

        PhiSel, PsiSel = None, None
        if self.protein is not None:
            # get site backbone information from protein structure
            PhiSel = (
                self.protein.select_atoms(f"resid {self.site} and segid {self.chain}")
                .residues[0]
                .phi_selection()
            )
            PsiSel = (
                self.protein.select_atoms(f"resid {self.site} and segid {self.chain}")
                .residues[0]
                .psi_selection()
            )

        # Default to helix backbone if none provided
        Phi = (
            None
            if PhiSel is None
            else np.rad2deg(get_dihedral(PhiSel.positions))
        )
        Psi = (
            None
            if PsiSel is None
            else np.rad2deg(get_dihedral(PsiSel.positions))
        )
        self.Phi, self.Psi = Phi, Psi

        # Get library
        cwd = Path().cwd()

        was_none = True if rotlib is None else False
        rotlib = self.res if rotlib is None else rotlib

        # If the rotlib isn't a dict try to figure out where it is
        if not isinstance(rotlib, dict):
            trotlib = io.get_possible_rotlibs(rotlib, suffix='rotlib', extension='.npz', was_none=was_none)
            if trotlib:
                rotlib = trotlib
            elif rotlib not in SUPPORTED_RESIDUES:
                raise NameError(f'There is no rotamer library called {rotlib} in this directory or in chilife')

            lib = io.read_library(rotlib, Phi, Psi)

        # If rotlib is a dict, assume that dict is in the format that read_library`would return
        else:
            lib = rotlib

        # Copy mutable library values (mostly np arrays) so that instances do not share data
        lib = {key: value.copy() if hasattr(value, 'copy') else value for key, value in lib.items()}

        # Perform a sanity check to ensure every necessary entry is present
        if isinstance(rotlib, Path) and not all(x in lib for x in io.rotlib_formats[lib['format_version']]):
            raise ValueError('The rotamer library does not contain all the required entries for the format version')

        # Modify library to be appropriately used with self.__dict__.update
        self._rotlib = {key: value.copy() if hasattr(value, 'copy') and key != 'internal_coords' else value for key, value in lib.items()}

        lib['_coords'] = lib.pop('coords').copy()
        lib['_dihedrals'] = lib.pop('dihedrals').copy()
        if 'skews' not in lib:
            lib['skews'] = None
        return lib



    def init_protein(self):
        if isinstance(self.protein, (mda.AtomGroup, mda.Universe)):
            if not hasattr(self.protein.universe._topology, "altLocs"):
                self.protein.universe.add_TopologyAttr('altLocs', np.full(len(self.protein.universe.atoms), ""))

        # Position library at selected residue
        self.resindex = self.protein.select_atoms(self.selstr).resindices[0]
        self.segindex = self.protein.select_atoms(self.selstr).segindices[0]
        if self.ignore_waters:
            self._protein = self.protein.select_atoms("not (byres name OH2 or resname HOH)")
        else:
            self._protein = self.protein.atoms

        self.protein_tree = cKDTree(self._protein.atoms.positions)

        # Delete cached lennard jones parameters if they exist.
        if hasattr(self, 'ermin_ij'):
            del self.ermin_ij
            del self.eeps_ij


    @property
    def bonds(self):
        """Set of atom indices corresponding to the atoms that form covalent bonds"""

        if not hasattr(self, "_bonds"):
            icmask_map = {x: i for i, x in enumerate(self.ic_mask)}
            self._bonds = np.array([(icmask_map[a], icmask_map[b]) for a, b in self.internal_coords.bonds
                                   if a in icmask_map and b in icmask_map])
            
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
        have 1-n non-bonded interactions where `n=self._exclude_nb_interactions` . By default, 1-3 interactions are
        excluded"""
        if not hasattr(self, "_non_bonded"):
            pairs = {v.index: [path for path in self._graph.get_all_shortest_paths(v) if
                           len(path) <= (self._exclude_nb_interactions)] for v in self._graph.vs}
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
            raise ValueError('The number of atoms in the input array does not match the number of atoms of the residue')

        self._coords = coords
        self.internal_coords = self.internal_coords.copy()
        self.internal_coords.set_cartesian_coords(coords, self.ic_mask)

        # Check if they are all at the same site
        self._dihedrals = np.rad2deg([ic.get_dihedral(1, self.dihedral_atoms) for ic in self.internal_coords])

        # Apply uniform weights
        self.weights = np.ones(len(self._dihedrals))
        self.weights /= self.weights.sum()

    @property
    def dihedrals(self):
        """Values of the (mobile) dihedral angles of all rotamers in the ensemble"""
        return self._dihedrals

    @dihedrals.setter
    def dihedrals(self, dihedrals):

        dihedrals = dihedrals if dihedrals.ndim == 2 else dihedrals[None, :]
        if dihedrals.shape[1] != self.dihedrals.shape[1]:
            raise ValueError('The input array does not have the correct number of dihedrals')

        if len(dihedrals) != self.dihedrals.shape[0]:

            warnings.warn('WARNING: Setting dihedrals in this fashion will set all bond lengths and angles to that '
                          'of the first rotamer in the library effectively removing stereo-isomers from the ensemble. It '
                          'will also set all weights to .')

            # Apply uniform weights
            self.weights = np.ones(len(dihedrals))
            self.weights /= self.weights.sum()
            idxs = [0 for _ in range(len(dihedrals))]
        else:
            idxs = np.arange(len(dihedrals))

        self._dihedrals = dihedrals
        z_matrix = self.internal_coords.batch_set_dihedrals(idxs, np.deg2rad(dihedrals), 1, self.dihedral_atoms)
        self.internal_coords = self.internal_coords.copy()
        self.internal_coords.load_new(z_matrix)
        self._coords = self.internal_coords.protein.trajectory.coordinate_array.copy()[:, self.ic_mask]
        if self._match_backbone:
            self.backbone_to_site()



    @property
    def mx(self):
        """The rotation matrix to rotate a residue from the local coordinate frame to the current residue. The local
        coordinate frame is defined by the alignment method"""
        mx, ori = global_mx(*parse_backbone(self, kind="local"), method=self.alignment_method)
        return mx

    @property
    def origin(self):
        """Origin of the local coordinate frame"""
        return np.squeeze(self.aln[1])

    @property
    def CB(self):
        """The coordinates of the β-carbon atom of the RotamerEnsemble"""
        if 'CB' not in self.atom_names:
            raise ValueError("There is no CB atom in this side chain")
        cbidx = np.argwhere(self.atom_names == 'CB').flat[0]
        return self.coords[:, cbidx].mean(axis=0)

    @property
    def protein(self):
        return self._protein

    @protein.setter
    def protein(self, protein):

        if hasattr(protein, "atoms") and isinstance(protein.atoms, (mda.AtomGroup, MolecularSystemBase)):
            self._protein = protein
            self.init_protein()
            resname = self.protein.select_atoms(self.selstr)[0].resname
            self.nataa = nataa_codes.get(resname, resname)

        elif protein is None:
            self._protein = protein
        else:
            raise ValueError("The input protein must be an instance of MDAnalysis.Universe, MDAnalysis.AtomGroup, or "
                             "chilife.MolSys")

    def intra_fit(self):
        """Align all rotamers of the ensemble to the ensemble backbone"""

        target = self.aln

        tmx, tori = self.alignment_method(*target)
        bbs = np.squeeze(self.coords[:,self.aln_idx])
        mxs, oris = [np.array(x) for x in zip(*[self.alignment_method(*bb) for bb in bbs])]
        mxs = mxs.transpose(0, 2, 1) @ tmx

        self._coords = (self.coords - oris[:, None, :]) @ mxs + tori[None, None, :]
        self.ICs_to_site(tori, tmx)


    def get_sasa(self):
        """Calculate the solvent accessible surface area (SASA) of each rotamer in the protein environment."""

        atom_radii = self.energy_func.get_lj_rmin(self.atom_types)
        if self.protein is not None:
            environment_coords = self.protein.atoms[self.protein_clash_idx].positions
            environment_radii = self.energy_func.get_lj_rmin(self.protein.atoms[self.protein_clash_idx].types)
        else:
            environment_coords = np.empty((0, 3))
            environment_radii = np.empty(0)

        SASAs = nu_getsasa(self.coords, atom_radii, environment_coords, environment_radii)

        return np.array(SASAs)

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
        "alignment_method": "bisect",
        "dihedral_sigmas": 35,
        "weighted_sampling": False,
        "eval_clash": True if not kwargs.get('minimize', False) else False,
        "use_H": False,
        '_match_backbone': True,
        "_exclude_nb_interactions": kwargs.pop('exclude_nb_interactions', 3),
        "_sample_size": kwargs.pop("sample", False),
        "energy_func": default_energy_func,
        "_minimize": kwargs.pop('minimize', False),
        "min_method": 'L-BFGS-B',
        "_do_trim": kwargs.pop('trim', True),
        "trim_tol": 0.005,
        "ignore_waters": True
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
    Reads chain from protein or makes an educated guess on which chain a particular site resides on for a given
    Protein/Universe/AtomGroup.

    Parameters
    ----------
    protein : mda.Universe | mda.AtomGroup | chilife.MolSys
        The protein being labeled.
    site :  int
        The residue being labeled.
    Returns
    -------
    chain : str
        Best guess for the chain on which the selected residue resides.

    """
    site = int(site)
    if protein is None:
        chain = "A"
    elif len(set(protein.segments.segids)) == 1:
        chain = protein.segments.segids[0]
    elif np.isin(protein.residues.resids, site).sum() == 0:
        raise ValueError(f"Residue {site} is not present on the provided protein")
    elif np.isin(protein.residues.resnums, site).sum() == 1:
        chain = protein.select_atoms(f"resid {site}").segids[0]
    else:
        raise ValueError(
            f"Residue {site} is present on more than one chain. Please specify the desired chain"
        )
    return chain
