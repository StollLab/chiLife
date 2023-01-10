from typing import Tuple, Dict, Union, BinaryIO, TextIO
from collections import defaultdict
from hashlib import sha256
from pathlib import Path
import pickle
import shutil
from io import StringIO, BytesIO
import zipfile

import numpy as np
from scipy.stats import gaussian_kde
from memoization import cached
import MDAnalysis as mda

import chilife
from .RotamerEnsemble import RotamerEnsemble
from .SpinLabel import SpinLabel
from .dRotamerEnsemble import dRotamerEnsemble
from .Protein import MolecularSystem
from .ProteinIC import ProteinIC

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
    hash = sha256()
    with open(file, 'rb') as f:
        while True:
            block = f.read(hash.block_size)
            if not block:
                break
            hash.update(block)

    return hash.hexdigest()


#
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
        lib = dict(files)

    del lib["allow_pickle"]

    if "sigmas" not in lib:
        lib["sigmas"] = np.array([])

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
            Path to the rotamer library file.
        Returns
        -------
        lib: Tuple[dict]
            Dictionaries of rotamer library attributes including sub_residues .

        """

    with zipfile.ZipFile(rotlib, 'r') as archive:
        for f in archive.namelist():
            if 'csts' in f:
                with archive.open(f) as of:
                    with np.load(of) as fc:
                        csts = dict(fc)
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
        Protein backbone Phi dihedral angle for the provided residue
    Psi : int
        Protein backbone Psi dihedral angle for the provided residue

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
        coords = []
        internal_coords = []
        for r in lib["dihedrals"]:
            ICn = ICs.copy().set_dihedral(np.deg2rad(r), 1, atom_list=dihedral_atoms)

            coords.append(ICn.to_cartesian())
            internal_coords.append(ICn)

    else:
        lib["weights"] = np.array([1])
        lib["dihedrals"], lib["sigmas"], dihedral_atoms = [], [], []
        coords = [ICs.to_cartesian()]
        internal_coords = [ICs.copy()]

    # Get origin and rotation matrix of local frame
    mask = np.in1d(atom_names, ["N", "CA", "C"])
    ori, mx = chilife.local_mx(*coords[0][mask])

    # Set coords in local frame and prepare output
    lib["coords"] = np.array([(coord - ori) @ mx for coord in coords])
    lib["internal_coords"] = internal_coords
    lib["atom_types"] = np.asarray(atom_types)
    lib["atom_names"] = np.asarray(atom_names)
    lib["dihedral_atoms"] = np.asarray(dihedral_atoms)
    lib["_rdihedrals"] = np.deg2rad(lib["dihedrals"])
    lib["_rsigmas"] = np.deg2rad(lib["sigmas"])
    lib['rotlib'] = res

    # hacky solution to experimental backbone dependent rotlibs
    if res == 'R1C':
        lib['spin_atoms'] = np.array(['N1', 'O1'])
        lib['spin_weights'] = np.array([0.5, 0.5])

    return lib


def read_library(rotlib: str, Phi: float = None, Psi: float = None) -> Dict:
    """Generalized wrapper function to aid selection of rotamer library reading function.

    Parameters
    ----------
    rotlib : str, Path
        3 letter residue code
    Phi : float
        Protein backbone Phi dihedral angle for the provided residue
    Psi : float
        Protein backbone Psi dihedral angle for the provided residue
    Returns
    -------
    lib: dict
        Dictionary of arrays containing rotamer library information in cartesian and dihedral space
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
        *molecules: Union[RotamerEnsemble, MolecularSystem, mda.Universe, mda.AtomGroup, str],
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
    *molecules : RotmerLibrary, chiLife.Protein, mda.Universe, mda.AtomGroup, str
        Object(s) to save. Includes RotamerEnsemble, SpinLabels, dRotamerEnsembles, proteins, or path to protein pdb.
        Can add as many as desired, except for path, for which only one can be given.
    protein_path : str, Path
        Path to a pdb file to use as the protein object.
    **kwargs :
        Additional arguments to pass to ``write_labels``

    Returns
    -------
    None
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

        if file_name == '':
            file_name = "No_Name_Protein"

        # Add spin label information to file name
        if 0 < len(molecules['rotens']) < 3:
            for rotens in molecules['rotens']:
                file_name += f"_{rotens.site}{rotens.res}"
        else:
            file_name += "_many_labels"

        file_name += ".pdb"

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



def write_protein(pdb_file: TextIO, protein: Union[mda.Universe, mda.AtomGroup]) -> None:
    """Helper function to write protein pdbs from mdanalysis objects.

    Parameters
    ----------
    pdb_file : TextIO
        File to save the protein to
    protein : MDAnalysis.Universe, MDAnalysis.AtomGroup
        MDAnalyiss object to save

    Returns
    -------
    None
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
        name = protein.fname

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


def write_ic(pdbfile: TextIO, ic: chilife.ProteinIC) -> None:
    """
    Write a chilife.ProteinIC internal coordinates object to a pdb file.
      Parameters
    ----------
    file : TextIO
        open file or io object.
    ic: chilife.ProteinIC
        chiLife internal coordinates object.

    Returns
    -------
    None
    """
    pdbfile.write('MODEL\n')
    for atom, coord in zip(ic.atoms, ic.coords):
        pdbfile.write(fmt_str.format(atom.index + 1, atom.name, atom.resn, 'A', atom.resi,
                                     coord[0], coord[1], coord[2],
                                     1.0, 1.0, atom.atype))
    pdbfile.write('ENDMDL\n')


def write_labels(pdb_file: TextIO, *args: SpinLabel, KDE: bool = True, sorted: bool = True) -> None:
    """Lower level helper function for saving SpinLabels and RotamerEnsembles. Loops over SpinLabel objects and appends
    atoms and electron coordinates to the provided file.

    Parameters
    ----------
    file : str
        File name to write to.
    *args: SpinLabel, RotamerEnsemble
        RotamerEnsemble or SpinLabel objects to be saved.
    KDE: bool
        Perform kernel density estimate smoothing on rotamer weights before saving. Usefull for uniformly weighted
        RotamerEnsembles or RotamerEnsembles with lots of rotamers
    sorted : bool
        Sort rotamers by weight befroe saving.
    Returns
    -------
    None
    """

    # Check for dSpinLables
    ensembles = []
    for arg in args:
        if isinstance(arg, chilife.RotamerEnsemble):
            ensembles.append(arg)
        elif isinstance(arg, chilife.dSpinLabel):
            ensembles.append(arg.SL1)
            ensembles.append(arg.SL2)
        else:
            raise TypeError(
                f"Cannot save {arg}. *args must be RotamerEnsemble SpinLabel or dSpinLabal objects"
            )

    # Write spin label models
    for k, label in enumerate(ensembles):
        pdb_file.write(f"HEADER {label.name}\n")

        # Save models in order of weight

        sorted_index = np.argsort(label.weights)[::-1] if sorted else np.arange(len(label.weights))
        norm_weights = label.weights / label.weights.max()

        for mdl, (conformer, weight) in enumerate(
                zip(label.coords[sorted_index], norm_weights[sorted_index])
        ):
            pdb_file.write("MODEL {}\n".format(mdl))

            [
                pdb_file.write(
                    fmt_str.format(
                        i,
                        label.atom_names[i],
                        label.res[:3],
                        label.chain,
                        int(label.site),
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
    for k, label in enumerate(ensembles):
        if not hasattr(label, "spin_centers"):
            continue
        if not np.any(label.spin_centers):
            continue

        pdb_file.write(f"HEADER {label.name}_density\n".format(label.label, k + 1))
        spin_centers = np.atleast_2d(label.spin_centers)

        if KDE and len(spin_centers) > 5:
            # Perform gaussian KDE to determine electron density
            gkde = gaussian_kde(spin_centers.T, weights=label.weights)

            # Map KDE density to pseudoatoms
            vals = gkde.pdf(spin_centers.T)

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

molecule_class = {RotamerEnsemble: 'rotens',
                  dRotamerEnsemble: 'rotens',
                  mda.Universe: 'molcart',
                  mda.AtomGroup: 'molcart',
                  MolecularSystem: 'molcart',
                  ProteinIC: 'molic'}


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
