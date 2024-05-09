from typing import Tuple, Dict, Union, BinaryIO, TextIO, Protocol
import warnings
import os
import urllib
import MDAnalysis
from numpy.typing import ArrayLike
from collections import defaultdict
from hashlib import sha256
from pathlib import Path
import pickle
import shutil
from io import StringIO, BytesIO
import zipfile

import numpy as np
from scipy.stats import gaussian_kde
from memoization import cached, suppress_warnings
import MDAnalysis as mda

import chilife
from .RotamerEnsemble import RotamerEnsemble
from .SpinLabel import SpinLabel
from .dRotamerEnsemble import dRotamerEnsemble
from .IntrinsicLabel import IntrinsicLabel
from .MolSys import MolecularSystemBase
from .MolSysIC import MolSysIC

#                 ID    name   res  chain resnum      X     Y      Z      q      b              elem
fmt_str = "ATOM  {:5d} {:^4s} {:3s} {:1s}{:4d}    {:8.3f}{:8.3f}{:8.3f}{:6.2f}{:6.2f}          {:>2s}  \n"


def read_distance_distribution(file_name: str) -> Tuple[np.ndarray, np.ndarray]:
    """Reads a DEER distance distribution file in the DeerAnalysis or similar format.

    Parameters
    ----------
    file_name : str
        File name of distance distribution

    Returns
    -------
    r: np.ndarray
        Domain of distance distribution in the same units as the file.
    p: np.ndarray
        Probability density over r normalized such that the integral of p over r is 1.
    """

    # Load DA file
    data = np.loadtxt(file_name)

    # Convert nm to angstroms
    r = data[:, 0] * 10

    # Extract distance domain coordinates
    p = data[:, 1]
    return r, p


def hash_file(file: Union[Path, BinaryIO]):
    """
    Helper function for hashing rotamer library files

    Parameters
    ----------
    file : str, Path, BinaryIO
        The name, Path or IOBytes of the file to hash

    Returns
    -------
    hashstring : str
        file hash
    """
    hash = sha256()
    with open(file, 'rb') as f:
        while True:
            block = f.read(hash.block_size)
            if not block:
                break
            hash.update(block)

    return hash.hexdigest()


suppress_warnings()
@cached(custom_key_maker=hash_file)
def read_rotlib(rotlib: Union[Path, BinaryIO] = None) -> Dict:
    """Reads RotamerEnsemble for stored spin labels.

    Parameters
    ----------
    rotlib : Path, BinaryIO
        Path object pointing to the rotamer library. Or a BytesIO object containing the rotamer library file
    Returns
    -------
    lib: dict
        Dictionary of SpinLabel rotamer ensemble attributes including coords, weights, dihedrals etc.

    """
    with np.load(rotlib, allow_pickle=True) as files:
        if files['format_version'] <= 1.1:
            raise RuntimeError('The rotlib that was provided is an old version that is not compatible with your '
                               'version of chiLife. You can either remake the rotlib, or use the update_rotlib.py '
                               'script provided in the chilife scripts directory to update this rotamer library to the '
                               'new format.')

        lib = dict(files)

    del lib["allow_pickle"]

    if "sigmas" not in lib:
        lib["sigmas"] = np.array([])
    lib['internal_coords'] = lib['internal_coords'].item()
    lib["_rdihedrals"] = np.deg2rad(lib["dihedrals"])
    lib["_rsigmas"] = np.deg2rad(lib["sigmas"])
    lib['rotlib'] = str(lib['rotlib'])
    lib['type'] = str(lib['type'])
    lib['format_version'] = float(lib['format_version'])
    return lib


@cached(custom_key_maker=hash_file)
def read_drotlib(rotlib: Path) -> Tuple[dict]:
    """Reads RotamerEnsemble for stored spin labels.

        Parameters
        ----------
        rotlib : Path
            The Path to the rotamer library file.
        Returns
        -------
        lib: Tuple[dict]
            Dictionaries of rotamer library attributes including sub_residues .
        """

    with zipfile.ZipFile(rotlib, 'r') as archive:
        for f in archive.namelist():
            if 'csts' in f:
                with archive.open(f) as of:
                    csts = np.load(of)
            elif f[-12] == 'A':
                with archive.open(f) as of:
                    libA = read_rotlib.__wrapped__(of)
            elif f[-12] == 'B':
                with archive.open(f) as of:
                    libB = read_rotlib.__wrapped__(of)

    return libA, libB, csts


@cached
def read_bbdep(res: str, Phi: int, Psi: int) -> Dict:
    """Read the Dunbrack rotamer library for the provided residue and backbone conformation.

    Parameters
    ----------
    res : str
        3-letter residue code
    Phi : int
        Backbone Phi dihedral angle for the provided residue
    Psi : int
        Backbone Psi dihedral angle for the provided residue

    Returns
    -------
    lib: dict
        Dictionary of arrays containing rotamer library information in cartesian and dihedral space
    """
    lib = {}
    Phi, Psi = str(Phi), str(Psi)

    # Read residue internal coordinate structure
    with open(chilife.RL_DIR / f"residue_internal_coords/{res.lower()}_ic.pkl", "rb") as f:
        ICs = pickle.load(f)

    atom_types = ICs.atom_types.copy()
    atom_names = ICs.atom_names.copy()

    maxchi = 5 if res in chilife.SUPPORTED_BB_LABELS else 4
    nchi = np.minimum(len(chilife.dihedral_defs[res]), maxchi)

    if res not in ("ALA", "GLY"):
        library = "R1C.lib" if res in chilife.SUPPORTED_BB_LABELS else "ALL.bbdep.rotamers.lib"
        start, length = chilife.rotlib_indexes[f"{res}  {Phi:>4}{Psi:>5}"]

        with open(chilife.RL_DIR / library, "rb") as f:
            f.seek(start)
            rotlib_string = f.read(length).decode()
            s = StringIO(rotlib_string)
            s.seek(0)
            data = np.genfromtxt(s, usecols=range(maxchi + 4, maxchi + 5 + 2 * maxchi))

        lib["weights"] = data[:, 0]
        lib["dihedrals"] = data[:, 1: nchi + 1]
        lib["sigmas"] = data[:, maxchi + 1: maxchi + nchi + 1]
        dihedral_atoms = chilife.dihedral_defs[res][:nchi]

        # Calculate cartesian coordinates for each rotamer
        z_matrix = ICs.batch_set_dihedrals(np.zeros(len(lib['dihedrals']), dtype=int), np.deg2rad(lib['dihedrals']), 1, dihedral_atoms)
        ICs._chain_operators = ICs._chain_operators[0]
        ICs.load_new(np.array(z_matrix))
        internal_coords = ICs.copy()
        coords = ICs.protein.trajectory.coordinate_array.copy()

    else:
        lib["weights"] = np.array([1])
        lib["dihedrals"], lib["sigmas"], dihedral_atoms = [], [], []
        coords = ICs.to_cartesian()[None, ...]
        internal_coords = ICs.copy()

    # Get origin and rotation matrix of local frame
    mask = np.in1d(atom_names, ["N", "CA", "C"])
    ori, mx = chilife.local_mx(*coords[0, mask])

    # Set coords in local frame and prepare output
    coords -= ori

    lib["coords"] = np.einsum('ijk,kl->ijl', coords, mx)
    lib["internal_coords"] = internal_coords
    lib["atom_types"] = np.asarray(atom_types, dtype=str)
    lib["atom_names"] = np.asarray(atom_names, dtype=str)
    lib["dihedral_atoms"] = np.asarray(dihedral_atoms, dtype=str)
    lib["_rdihedrals"] = np.deg2rad(lib["dihedrals"])
    lib["_rsigmas"] = np.deg2rad(lib["sigmas"])
    lib['rotlib'] = res

    # hacky solution to experimental backbone dependent rotlibs
    if res == 'R1C':
        lib['spin_atoms'] = np.array(['N1', 'O1'])
        lib['spin_weights'] = np.array([0.5, 0.5])

    return lib


def read_library(rotlib: str, Phi: float = None, Psi: float = None) -> Dict:
    """Function to read rotamer libraries and bifunctional rotamer libraries as dictionaries.

    Parameters
    ----------
    rotlib : str, Path
        3 letter residue code or path the rotamer library file.
    Phi : float
        Backbone Phi dihedral angle for the provided residue. Only applicable for canonical amino acids.
    Psi : float
        Backbone Psi dihedral angle for the provided residue. Only applicable for canonical amino acids.

    Returns
    -------
    lib: dict
        Dictionary of arrays containing rotamer library information in cartesian and dihedral space.
    """
    backbone_exists = Phi and Psi

    if backbone_exists:
        Phi = int((Phi // 10) * 10)
        Psi = int((Psi // 10) * 10)
    # Use helix backbone if not otherwise specified
    else:
        Phi, Psi = -60, -50

    if isinstance(rotlib, Path):
        if rotlib.suffix == '.npz':
            return read_rotlib(rotlib)
        elif rotlib.suffix == '.zip':
            return read_drotlib(rotlib)
        else:
            raise ValueError(f'{rotlib.name} is not a valid rotamer library file type')
    elif isinstance(rotlib, str):
        return read_bbdep(rotlib, Phi, Psi)
    else:
        raise ValueError(f'{rotlib} is not a valid rotamer library')


def save(
        file_name: str,
        *molecules: Union[RotamerEnsemble, MolecularSystemBase, mda.Universe, mda.AtomGroup, str],
        protein_path: Union[str, Path] = None,
        mode: str = 'w',
        **kwargs,
) -> None:
    """Save a pdb file of the provided labels and proteins

    Parameters
    ----------
    file_name : str
        Desired file name for output file. Will be automatically made based off of protein name and labels if not
        provided.
    *molecules : RotmerLibrary, chiLife.MolSys, mda.Universe, mda.AtomGroup, str
        Object(s) to save. Includes RotamerEnsemble, SpinLabels, dRotamerEnsembles, proteins, or path to protein pdb.
        Can add as many as desired, except for path, for which only one can be given.
    protein_path : str, Path
        The Path to a pdb file to use as the protein object.
    mode : str
        Which mode to open the file in. Accepts 'w' or 'a' to overwrite or append.
    **kwargs :
        Additional arguments to pass to ``write_labels``
    """

    if isinstance(file_name, tuple(molecule_class.keys())):
        molecules = list(molecules)
        molecules.insert(0, file_name)
        file_name = None

    # Check for filename at the beginning of args
    tmolecules = defaultdict(list)
    for mol in molecules:
        mcls = [val for key, val in molecule_class.items() if isinstance(mol, key)]
        if mcls == []:
            raise TypeError(f'{type(mol)} is not a supported type for this function. '
                            'chiLife can only save objects of the following types:\n'
                            + ''.join(f'{key.__name__}\n' for key in molecule_class) +
                            'Please check that your input is compatible')

        tmolecules[mcls[0]].append(mol)
    molecules = tmolecules

    # Ensure only one protein path was provided (for now)
    protein_path = [] if protein_path is None else [protein_path]
    if len(protein_path) > 1:
        raise ValueError('More than one protein path was provided. C')
    elif len(protein_path) == 1:
        protein_path = protein_path[0]
    else:
        protein_path = None

    # Create a file name from protein information
    if file_name is None:
        if protein_path is not None:
            f = Path(protein_path)
            file_name = f.name[:-4]
        else:
            file_name = ""
            for protein in molecules['molcart']:
                if getattr(protein, "filename", None):
                    file_name += ' ' + Path(protein.filename).name
                    file_name = file_name[:-4]

        if file_name == '' and len(molecules['molcart']) > 0:
            file_name = "No_Name_Protein"

        # Add spin label information to file name
        if 0 < len(molecules['rotens']) < 3:
            naml = [file_name] + [rotens.name for rotens in molecules['rotens']]
            file_name = "_".join(naml)
            file_name = file_name.strip('_')
        elif len(molecules['rotens']) >= 3:
            file_name += "_many_labels"

        file_name += ".pdb"
        file_name = file_name.strip()

    if protein_path is not None:
        print(protein_path, file_name)
        shutil.copy(protein_path, file_name)
        pdb_file = open(file_name, 'a+')
    else:
        pdb_file = open(file_name, mode)

    for protein in molecules['molcart']:
        write_protein(pdb_file, protein)

    for ic in molecules['molic']:
        write_ic(pdb_file, ic)

    if len(molecules['rotens']) > 0:
        write_labels(pdb_file, *molecules['rotens'], **kwargs)


def fetch(accession_number: str, save: bool = False) -> MDAnalysis.Universe:
    """Fetch pdb file from the protein data bank or the AlphaFold Database and optionally save to disk.

    Parameters
    ----------
    accession_number : str
        4 letter structure PDB ID or alpha fold accession number. Note that AlphaFold accession numbers must begin with
        'AF-'.
    save : bool
        If true the fetched PDB will be saved to the disk.

    Returns
    -------
    U : MDAnalysis.Universe
        MDAnalysis Universe object of the protein corresponding to the provided PDB ID or AlphaFold accession number

    """
    accession_number = accession_number.split('.pdb')[0]
    pdb_name = accession_number + '.pdb'

    if accession_number.startswith('AF-'):
        print(f"https://alphafold.ebi.ac.uk/files/{accession_number}-F1-model_v3.pdb")
        urllib.request.urlretrieve(f"https://alphafold.ebi.ac.uk/files/{accession_number}-F1-model_v3.pdb", pdb_name)
    else:
        urllib.request.urlretrieve(f"http://files.rcsb.org/download/{pdb_name}", pdb_name)


    U = mda.Universe(pdb_name, in_memory=True)

    if not save:
        os.remove(pdb_name)

    return U


def load_protein(struct_file: Union[str, Path],
                 *traj_file: Union[str, Path]) -> MDAnalysis.AtomGroup:
    """

    Parameters
    ----------
    struct_file : Union[TextIO, str, Path]
        Name, Path or TextIO object referencing the structure file (e.g. pdb, gro, psf)
    *traj_file : Union[TextIO, str, Path] (optional)
        Name, Path or TextIO object(s) referencing the trajectory file (e.g. pdb, xtc, dcd)


    Returns
    -------
    protein: MDAnalysis.AtomGroup
        An MDA AtomGroup object containing the protein structure and trajectory. The object is always loaded into
        memory to allow coordinate manipulations.
    """

    if traj_file != []:
        traj_file = [str(file) for file in traj_file]
        protein = mda.Universe(str(struct_file), *traj_file, in_memory=True)
    else:
        protein = mda.Universe(struct_file, in_memory=True)

    return protein



def write_protein(pdb_file: TextIO, protein: Union[mda.Universe, mda.AtomGroup, MolecularSystemBase]) -> None:
    """
    Helper function to write protein PDBs from MDAnalysis and MolSys objects.

    Parameters
    ----------
    pdb_file : TextIO
        File to save the protein to
    protein : MDAnalysis.Universe, MDAnalysis.AtomGroup, MolSys
        MDAnalysis or MolSys object to save
    """

    # Change chain identifier if longer than 1
    available_segids = iter('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
    for seg in protein.segments:
        if len(seg.segid) > 1:
            seg.segid = next(available_segids)
    if isinstance(protein, (mda.AtomGroup, mda.Universe)):
        traj = protein.universe.trajectory
        name = Path(protein.universe.filename) if protein.universe.filename is not None else Path(pdb_file.name)
        name = name.name
    else:
        traj = protein.trajectory
        name = protein.fname if hasattr(protein, 'fname') else None

    if name is None:
        name = Path(pdb_file.name).name

    name = name[:-4] if name.endswith(".pdb") else name

    pdb_file.write(f'HEADER {name}\n')
    for mdl, ts in enumerate(traj):
        pdb_file.write(f"MODEL {mdl}\n")
        [
            pdb_file.write(
                fmt_str.format(
                    atom.index,
                    atom.name,
                    atom.resname[:3],
                    atom.segid,
                    atom.resnum,
                    *atom.position,
                    1.00,
                    1.0,
                    atom.type,
                )
            )
            for atom in protein.atoms
        ]
        pdb_file.write("TER\n")
        pdb_file.write("ENDMDL\n")


def write_ic(pdb_file: TextIO, ic: MolSysIC) -> None:
    """
    Write a :class:`~MolSysIC` internal coordinates object to a pdb file.

    Parameters
    ----------
    pdb_file : TextIO
        open file or io object.
    ic: MolSysIC
        chiLife internal coordinates object.
    """

    pdb_file.write('MODEL\n')
    for i, (atom, coord) in enumerate(zip(ic.atoms, ic.coords)):
        pdb_file.write(fmt_str.format(i + 1, atom.name, atom.resname, atom.segid, atom.resnum,
                                     coord[0], coord[1], coord[2],
                                     1.0, 1.0, atom.type))
    pdb_file.write('ENDMDL\n')


def write_labels(pdb_file: TextIO, *args: SpinLabel, KDE: bool = True, sorted: bool = True) -> None:
    """Lower level helper function for saving SpinLabels and RotamerEnsembles. Loops over SpinLabel objects and appends
    atoms and electron coordinates to the provided file.

    Parameters
    ----------
    pdb_file : str
        File name to write to.
    *args: SpinLabel, RotamerEnsemble
        The RotamerEnsemble or SpinLabel objects to be saved.
    KDE: bool
        Perform kernel density estimate smoothing on rotamer weights before saving. Usefull for uniformly weighted
        RotamerEnsembles or RotamerEnsembles with lots of rotamers
    sorted : bool
        Sort rotamers by weight before saving.
    Returns
    -------
    None
    """

    # Write spin label models
    for k, label in enumerate(args):
        pdb_file.write(f"HEADER {label.name}\n")

        # Save models in order of weight

        sorted_index = np.argsort(label.weights)[::-1] if sorted else np.arange(len(label.weights))
        norm_weights = label.weights / label.weights.max()
        if isinstance(label, dRotamerEnsemble):
            sites = np.concatenate([np.ones(len(label.RE1.atoms), dtype=int) * int(label.site1),
                                    np.ones(len(label.RE2.atoms), dtype=int) * int(label.site2)])
        else:
            sites = [atom.resi for atom in label.atoms]
        for mdl, (conformer, weight) in enumerate(
                zip(label.coords[sorted_index], norm_weights[sorted_index])
        ):
            pdb_file.write(f"MODEL {mdl}\n")

            [
                pdb_file.write(
                    fmt_str.format(
                        i,
                        label.atom_names[i],
                        label.res[:3],
                        label.chain,
                        sites[i],
                        *conformer[i],
                        weight,
                        1.00,
                        label.atom_types[i],
                    )
                )
                for i in range(len(label.atom_names))
            ]
            pdb_file.write("TER\n")
            pdb_file.write("ENDMDL\n")

    # Write electron density at electron coordinates
    for k, label in enumerate(args):
        if not hasattr(label, "spin_centers"):
            continue
        if not np.any(label.spin_centers):
            continue

        pdb_file.write(f"HEADER {label.name}_density\n")
        spin_centers = np.atleast_2d(label.spin_centers)

        if KDE and len(spin_centers) > 5:
            try:
                # Perform gaussian KDE to determine electron density
                gkde = gaussian_kde(spin_centers.T, weights=label.weights)

                # Map KDE density to pseudoatoms
                vals = gkde.pdf(spin_centers.T)

            except:
                vals = label.weights

        else:
            vals = label.weights
        norm_weights = vals / vals.max()
        [
            pdb_file.write(
                fmt_str.format(
                    i,
                    "NEN",
                    label.label[:3],
                    label.chain,
                    int(label.site),
                    *spin_centers[i],
                    norm_weights[i],
                    1.00,
                    "N",
                )
            )
            for i in range(len(norm_weights))
        ]

        pdb_file.write("TER\n")


def write_atoms(file: TextIO,
                atoms: Union[MDAnalysis.Universe, MDAnalysis.AtomGroup, MolecularSystemBase],
                coords: ArrayLike = None) -> None:
    """
    Write a set of atoms to a file in PDB format without any prefix or suffix, i.e. Not wrapped in `MODEL` or `ENDMDL`

    Parameters
    ----------
    file : TextIO
        A writable file (TextIO) object. The file that the atoms will be written to.
    atoms : MDAnalysis.Universe, MDAnalysis.AtomGroup, MolecularSystemBase
        Object containing atoms to be written. May also work with duck-typed set of atoms that are iterable and each
        atom has properties ``index``, ``name``,
    coords : ArrayLike
        Array of atom coordinates corresponding to atoms
    """

    if isinstance(atoms, MDAnalysis.Universe):
        atoms = atoms.atoms

    if coords is None:
        coords = atoms.positions

    for atom, coord in zip(atoms, coords):
        file.write(
            f"ATOM  {atom.index + 1:5d} {atom.name:^4s} {atom.resname:3s} {'A':1s}{atom.resnum:4d}    "
            f"{coord[0]:8.3f}{coord[1]:8.3f}{coord[2]:8.3f}{1.0:6.2f}{1.0:6.2f}          {atom.type:>2s}  \n"
        )


molecule_class = {RotamerEnsemble: 'rotens',
                  dRotamerEnsemble: 'rotens',
                  IntrinsicLabel: 'rotens',
                  mda.Universe: 'molcart',
                  mda.AtomGroup: 'molcart',
                  MolecularSystemBase: 'molcart',
                  MolSysIC: 'molic'}


rotlib_formats = {1.0: (
    'rotlib',  #
    'resname',
    'coords',
    'internal_coords',
    'weights',
    'atom_types',
    'atom_names',
    'dihedrals',
    'dihedral_atoms',
    'type',
    'format_version'
)}

rotlib_formats[1.1] = *rotlib_formats[1.0], 'description', 'comment', 'reference'
rotlib_formats[1.2] = rotlib_formats[1.0]
rotlib_formats[1.3] = rotlib_formats[1.2]


