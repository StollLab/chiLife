from __future__ import annotations
from typing import Tuple, Dict, Union, BinaryIO, TextIO
import textwrap
import warnings
import os
import urllib
import MDAnalysis
from numpy.typing import ArrayLike
from collections import defaultdict
from collections.abc import Sequence
from hashlib import sha256
from pathlib import Path
import pickle
import shutil
from io import StringIO
import zipfile

import numpy as np
from scipy.stats import gaussian_kde
from memoization import cached, suppress_warnings
import MDAnalysis as mda

import chilife.RotamerEnsemble as re
import chilife.SpinLabel as sl
import chilife.dRotamerEnsemble as dre
import chilife.LigandEnsemble as le
import chilife
from .globals import (
    dihedral_defs,
    rotlib_indexes,
    RL_DIR,
    SUPPORTED_BB_LABELS,
    USER_RL_DIR,
    rotlib_defaults,
    alt_prot_states,
)

from .alignment_methods import local_mx
from .IntrinsicLabel import IntrinsicLabel
from .MolSys import MolecularSystemBase, MolSys
from .Topology import BondType
from .MolSysIC import MolSysIC
from .pdb_utils import parse_connect, sort_pdb

#                 ID    name   res  chain resnum      X     Y      Z      q      b              elem
fmt_str = "ATOM  {:5d} {:^4s} {:3s} {:1s}{:4d}    {:8.3f}{:8.3f}{:8.3f}{:6.2f}{:6.2f}          {:>2s}  \n"
CHEM_COMP_ATOM_KEYS = (
    "comp_id",
    "atom_id",
    "type_symbol",
    "pdbx_aromatic_flag",
    "pdbx_stereo_config",
)

CHEM_COMP_BOND_KEYS = (
    "comp_id",
    "atom_id_1",
    "atom_id_2",
    "value_order",
    "pdbx_aromatic_flag",
    "pdbx_stereo_config",
)

STRUCT_CONN_KEYS = (
    "id",
    "conn_type_id",
    "pdbx_leaving_atom_flag",
    "pdbx_PDB_id",
    "ptnr1_label_asym_id",
    "ptnr1_label_comp_id",
    "ptnr1_label_seq_id",
    "ptnr1_label_atom_id",
    "pdbx_ptnr1_label_alt_id",
    "pdbx_ptnr1_PDB_ins_code",
    "pdbx_ptnr1_standard_comp_id",
    "ptnr1_symmetry",
    "ptnr2_label_asym_id",
    "ptnr2_label_comp_id",
    "ptnr2_label_seq_id",
    "ptnr2_label_atom_id",
    "pdbx_ptnr2_label_alt_id",
    "pdbx_ptnr2_PDB_ins_code",
    "ptnr1_auth_asym_id",
    "ptnr1_auth_comp_id",
    "ptnr1_auth_seq_id",
    "ptnr2_auth_asym_id",
    "ptnr2_auth_comp_id",
    "ptnr2_auth_seq_id",
    "ptnr2_symmetry",
    "pdbx_ptnr3_label_atom_id",
    "pdbx_ptnr3_label_seq_id",
    "pdbx_ptnr3_label_comp_id",
    "pdbx_ptnr3_label_asym_id",
    "pdbx_ptnr3_label_alt_id",
    "pdbx_ptnr3_PDB_ins_code",
    "details",
    "pdbx_dist_value",
    "pdbx_value_order",
    "pdbx_role",
)

# CIF `_struct_conn.conn_type_id` values that are treated as bonds in chiLife.
STRUCT_CONN_BOND_TYPES = ("covale", "disulf", "metalc")

# Element symbols treated as "non-metals" for picking the `covale` vs `metalc`
# conn_type_id on write. Everything else resolves to `metalc`.
NONMETAL_ELEMENTS = frozenset(
    {
        "H",
        "D",
        "C",
        "N",
        "O",
        "P",
        "S",
        "SE",
        "F",
        "CL",
        "BR",
        "I",
        "B",
        "AT",
        "HE",
        "NE",
        "AR",
        "KR",
        "XE",
        "RN",
    }
)

CIF_BOND_TO_XL = {
    "arom": BondType.AROMATIC,
    "delo": BondType.DELOCALIZED,
    "doub": BondType.DOUBLE,
    "pi": BondType.PI,
    "poly": BondType.POLYMERIC,
    "quad": BondType.QUADRUPLE,
    "sing": BondType.SINGLE,
    "trip": BondType.TRIPLE,
    "metalc": BondType.DATIVE,
    "disulf": BondType.UNSPECIFIED,
}

XL_BOND_TO_CIF = {v: k for k, v in CIF_BOND_TO_XL.items()}

SDF_BOND_TO_XL = {
    1: BondType.SINGLE,
    2: BondType.DOUBLE,
    3: BondType.TRIPLE,
    4: BondType.AROMATIC,
    5: BondType.UNSPECIFIED,
    6: BondType.SINGLE,
    7: BondType.DOUBLE,
    8: BondType.UNSPECIFIED,
}

XL_BOND_TO_SDF = {v: k for k, v in SDF_BOND_TO_XL.items() if k not in (5, 6, 7)}

SDF_ATOM_CHARGE_TO_INT = {
    0: 0,
    1: 3,
    2: 2,
    3: 1,
    4: 0,
    5: -1,
    6: -2,
    7: -3,
}


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
    r = data[:, 0]
    if max(r) < 15:
        r *= 10

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
    with open(file, "rb") as f:
        while True:
            block = f.read(hash.block_size)
            if not block:
                break
            hash.update(block)

    return hash.hexdigest()


def get_possible_rotlibs(
    rotlib: str,
    suffix: str,
    extension: str,
    return_all: bool = False,
    was_none: bool = False,
) -> Union[Path, None]:
    """
    Search all known rotlib directories and the current working directory for rotlib(s) that match the provided
    information.

    Parameters
    ----------
    rotlib: str
        Fullname, base name or partial name of the rotamer libraries to search for.
    suffix: str
        possible suffixes the rotamer library may have, e.g. ip2 to indicate an i+2 rotamer library.
    extension: str
        filetype extension to look for. This will be either `npz` for monofunctional rotlibs or zip for bifunctional
        rotlibs.
    return_all: bool
        By default, only the first found rotlib will be returned unless ``return_all = True``, in which case
    was_none: bool
        For internal use only.

    Returns
    -------
    rotlib: Path, List[Path], None
        The path to the found rotlib or a list of paths to the rotlibs that match the search criteri, or ``None`` if
        no rotlibs are found.
    """
    cwd = Path.cwd()
    sufplusex = "_" + suffix + extension
    # Assemble a list of possible rotlib paths starting in the current directory
    possible_rotlibs = [
        Path(rotlib),
        cwd / rotlib,
        cwd / (rotlib + extension),
        cwd / (rotlib + sufplusex),
    ]

    possible_rotlibs += list(cwd.glob(f"{rotlib}*{sufplusex}"))
    # Then in the user defined rotamer library directory
    for pth in USER_RL_DIR:
        possible_rotlibs += list(pth.glob(f"{rotlib}*{sufplusex}"))

    if not was_none:
        possible_rotlibs += list((RL_DIR / "user_rotlibs").glob(f"*{rotlib}*"))

    if return_all:
        rotlib = []
    for possible_file in possible_rotlibs:
        if possible_file.exists() and return_all:
            rotlib.append(possible_file)
        elif possible_file.exists() and not possible_file.is_dir():
            rotlib = possible_file
            break
    else:
        if isinstance(rotlib, str) and was_none and rotlib in rotlib_defaults:
            rotlib = RL_DIR / "user_rotlibs" / (rotlib_defaults[rotlib][0] + sufplusex)

        elif not isinstance(rotlib, list) or rotlib == []:
            rotlib = None

    # rotlib lists need to be sorted to prevent position mismatches for results with tests.
    if isinstance(rotlib, list):
        rotlib = list(set(rotlib))
        rotlib = [rot for rot in rotlib if str(rot).endswith(extension)]
        rotlib = sorted(rotlib)
    else:
        rotlib = rotlib if str(rotlib).endswith(extension) else None

    return rotlib


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
        if files["format_version"] <= 1.1:
            raise RuntimeError(
                "The rotlib that was provided is an old version that is not compatible with your "
                "version of chiLife. You can either remake the rotlib, or use the update_rotlib.py "
                "script provided in the chilife scripts directory to update this rotamer library to the "
                "new format."
            )

        lib = dict(files)
    if "allow_pickle" in lib:
        del lib["allow_pickle"]

    if "sigmas" not in lib:
        lib["sigmas"] = np.array([])
    lib["internal_coords"] = lib["internal_coords"].item()
    lib["_rdihedrals"] = np.deg2rad(lib["dihedrals"])
    lib["_rsigmas"] = np.deg2rad(lib["sigmas"])
    lib["rotlib"] = str(lib["rotlib"])
    lib["type"] = str(lib["type"])
    lib["format_version"] = float(lib["format_version"])
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

    with zipfile.ZipFile(rotlib, "r") as archive:
        for f in archive.namelist():
            if "csts" in f:
                with archive.open(f) as of:
                    csts = np.load(of)
            elif f[-12] == "A":
                with archive.open(f) as of:
                    libA = read_rotlib.__wrapped__(of)
            elif f[-12] == "B":
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

    ic_res = alt_prot_states.get(res, res)

    lib = {}
    Phi, Psi = str(Phi), str(Psi)

    # Read residue internal coordinate structure
    with open(RL_DIR / f"residue_internal_coords/{res.lower()}_ic.pkl", "rb") as f:
        ICs = pickle.load(f)

    atom_types = ICs.atom_types.copy()
    atom_names = ICs.atom_names.copy()

    maxchi = 5 if res in SUPPORTED_BB_LABELS else 4
    nchi = np.minimum(len(dihedral_defs[res]), maxchi)

    if res not in ("ALA", "GLY"):
        library = "R1C.lib" if res in SUPPORTED_BB_LABELS else "ALL.bbdep.rotamers.lib"
        start, length = rotlib_indexes[f"{ic_res}  {Phi:>4}{Psi:>5}"]

        with open(RL_DIR / library, "rb") as f:
            f.seek(start)
            rotlib_string = f.read(length).decode()
            s = StringIO(rotlib_string)
            s.seek(0)
            data = np.genfromtxt(s, usecols=range(maxchi + 4, maxchi + 5 + 2 * maxchi))

        lib["weights"] = data[:, 0]
        lib["dihedrals"] = data[:, 1 : nchi + 1]
        lib["sigmas"] = data[:, maxchi + 1 : maxchi + nchi + 1]
        dihedral_atoms = dihedral_defs[res][:nchi]

        # Calculate cartesian coordinates for each rotamer
        z_matrix = ICs.batch_set_dihedrals(
            np.zeros(len(lib["dihedrals"]), dtype=int),
            np.deg2rad(lib["dihedrals"]),
            1,
            dihedral_atoms,
        )
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
    mask = np.isin(atom_names, ["N", "CA", "C"])
    ori, mx = local_mx(*coords[0, mask])

    # Set coords in local frame and prepare output
    coords -= ori

    lib["coords"] = np.einsum("ijk,kl->ijl", coords, mx)
    lib["internal_coords"] = internal_coords
    lib["atom_types"] = np.asarray(atom_types, dtype=str)
    lib["atom_names"] = np.asarray(atom_names, dtype=str)
    lib["dihedral_atoms"] = np.asarray(dihedral_atoms, dtype=str)
    lib["_rdihedrals"] = np.deg2rad(lib["dihedrals"])
    lib["_rsigmas"] = np.deg2rad(lib["sigmas"])
    lib["rotlib"] = res
    lib["backbone_atoms"] = ["H", "N", "CA", "HA", "C", "O"]
    lib["aln_atoms"] = ["N", "CA", "C"]

    # hacky solution to experimental backbone dependent rotlibs
    if res == "R1C":
        lib["spin_atoms"] = np.array(["N1", "O1"])
        lib["spin_weights"] = np.array([0.5, 0.5])

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
        if rotlib.suffix == ".npz":
            return read_rotlib(rotlib)
        elif rotlib.suffix == ".zip":
            return read_drotlib(rotlib)
        else:
            raise ValueError(f"{rotlib.name} is not a valid rotamer library file type")
    elif isinstance(rotlib, str):
        return read_bbdep(rotlib, Phi, Psi)
    else:
        raise ValueError(f"{rotlib} is not a valid rotamer library")


def save(
    file_name: str,
    *molecules: Union[
        re.RotamerEnsemble, MolecularSystemBase, mda.Universe, mda.AtomGroup, str
    ],
    protein_path: Union[str, Path] = None,
    mode: str = "w",
    conect: bool = False,
    frames: Union[int, str, ArrayLike, None] = "all",
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
    conect : bool
        Whether to save  PDB CONECT records. Defaults to False.
    frames : int, str, ArrayLike[int], None
        Exclusively used for ``MolSys`` and ``AtomGroup`` objects. ``frames`` are the indices of the frames to save.
        Can be a single index, array of indices, 'all', or ``None``. If ``None`` is passed only the
        activate frame will be saved. If 'all' is passed, all frames are saved. Default is 'all'
    **kwargs :
        Additional arguments to pass to ``write_labels``

        write_spin_centers : bool
            Write spin centers (atoms named NEN) as a separate object with weights mapped to q-factor.

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
            raise TypeError(
                f"{type(mol)} is not a supported type for this function. "
                "chiLife can only save objects of the following types:\n"
                + "".join(f"{key.__name__}\n" for key in molecule_class)
                + "Please check that your input is compatible"
            )

        tmolecules[mcls[0]].append(mol)
    molecules = tmolecules

    # Ensure only one protein path was provided (for now)
    protein_path = [] if protein_path is None else [protein_path]
    if len(protein_path) > 1:
        raise ValueError("More than one protein path was provided. C")
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
            for protein in molecules["molcart"]:
                if getattr(protein, "filename", None):
                    file_name += " " + Path(protein.filename).name
                    file_name = file_name[:-4]

        if file_name == "" and len(molecules["molcart"]) > 0:
            file_name = "No_Name_Protein"

        # Add spin label information to file name
        if 0 < len(molecules["rotens"]) < 3:
            naml = [file_name] + [rotens.name for rotens in molecules["rotens"]]
            file_name = "_".join(naml)
            file_name = file_name.strip("_")
        elif len(molecules["rotens"]) >= 3:
            file_name += "_many_labels"

        file_name += ".pdb"
        file_name = file_name.strip()

    if protein_path is not None:
        print(protein_path, file_name)
        shutil.copy(protein_path, file_name)
        pdb_file = open(file_name, "a+")
    else:
        pdb_file = open(file_name, mode)

    used_names = {}
    for protein in molecules["molcart"]:
        if isinstance(protein, (mda.AtomGroup, mda.Universe)):
            name = (
                Path(protein.universe.filename)
                if protein.universe.filename is not None
                else Path(pdb_file.name)
            )
            name = name.name
        else:
            name = protein.fname if hasattr(protein, "fname") else None

        if name is None:
            name = Path(pdb_file.name).name

        name = name[:-4] if name.endswith(".pdb") else name
        name_ = name + str(used_names[name]) if name in used_names else name

        used_names[name] = used_names.setdefault(name, 0) + 1

        write_protein(pdb_file, protein, name_, conect=conect, frames=frames)

    for ic in molecules["molic"]:
        if isinstance(ic.protein, (mda.AtomGroup, mda.Universe)):
            name = (
                Path(ic.protein.universe.filename)
                if ic.protein.universe.filename is not None
                else Path(pdb_file.name)
            )
            name = name.name
        else:
            name = ic.protein.fname if hasattr(ic.protein, "fname") else None

        if name is None:
            name = Path(pdb_file.name).name

        name = name[:-4] if name.endswith(".pdb") else name
        name_ = name + str(used_names[name]) if name in used_names else name

        write_ic(pdb_file, ic, name=name_, conect=conect, frames=frames)

    if len(molecules["rotens"]) > 0:
        write_labels(pdb_file, *molecules["rotens"], conect=conect, **kwargs)

    pdb_file.close()


def fetch(
    accession_number: str, save: bool = False, format: str | None = None
) -> MDAnalysis.Universe:
    """Fetch pdb file from the protein data bank or the AlphaFold Database and optionally save to disk.

    Parameters
    ----------
    accession_number : str
        4 letter structure PDB ID or alpha fold accession number. Note that AlphaFold accession numbers must begin with
        'AF-'.
    save : bool
        If true the fetched PDB will be saved to the disk.
    format : str
        Format of the structure file (pdb or cif).
    Returns
    -------
    U : MDAnalysis.Universe
        MDAnalysis Universe object of the protein corresponding to the provided PDB ID or AlphaFold accession number

    """
    if format is None and accession_number.lower().endswith(".pdb"):
        format = "pdb"
    elif format is None and accession_number.lower().endswith(".cif"):
        format = "cif"
    elif format is None:
        format = "pdb"
    elif format.lower().startswith("."):
        format = format[1:].lower()
    else:
        format = format.lower()

    accession_number = accession_number.split(f".{format}")[0]
    pdb_name = accession_number + "." + format

    if accession_number.startswith("AF-"):
        print(
            f"https://alphafold.ebi.ac.uk/files/{accession_number}-F1-model_v6.{format}"
        )
        urllib.request.urlretrieve(
            f"https://alphafold.ebi.ac.uk/files/{accession_number}-F1-model_v6.{format}",
            pdb_name,
        )
    else:
        urllib.request.urlretrieve(
            f"http://files.rcsb.org/download/{pdb_name}", pdb_name
        )

    if format.lower() == "pdb":
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            U = mda.Universe(pdb_name, in_memory=True)
    elif format.lower() == "cif":
        U = MolSys.from_cif(pdb_name)

    if not save:
        os.remove(pdb_name)

    return U


def load_protein(
    struct_file: Union[str, Path], *traj_file: Union[str, Path]
) -> MDAnalysis.AtomGroup:
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
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            protein = mda.Universe(str(struct_file), *traj_file, in_memory=True)
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            protein = mda.Universe(struct_file, in_memory=True)

    return protein


def read_pdb(pdb_file: Union[str, Path], sort_atoms=False):
    """
    Read A PDB file into a dictionary containing all standard pdb atom information and bonds if CONECT records are
    present.

    Parameters
    ----------
    pdb_file : str | Path
        File name or ``Path`` to the PDB file to be read.

    Returns
    -------
    pdb_data : dict
        Dictionary of PDB data.
    """
    keys = [
        "record_types",
        "atomids",
        "names",
        "altlocs",
        "resnames",
        "chains",
        "resnums",
        "icodes",
        "coords",
        "occupancies",
        "bs",
        "segs",
        "atypes",
        "charges",
    ]
    if sort_atoms:
        lines = sort_pdb(pdb_file)
    else:
        with open(pdb_file, "r") as f:
            lines = f.readlines()

    connect = []
    atom_lines = []
    for line in lines:
        if line.startswith(("MODEL", "ENDMDL", "ATOM", "HETATM")):
            atom_lines.append(line)
        elif line.startswith("CONECT"):
            connect.append(line)

    start_idxs = []
    end_idxs = []
    for i, line in enumerate(atom_lines):
        if line.startswith("MODEL"):
            start_idxs.append(i + 1)
        elif line.startswith("ENDMDL"):
            end_idxs.append(i)

    if len(start_idxs) > 0:
        lines = [atom_lines[start:end] for start, end in zip(start_idxs, end_idxs)]
    else:
        lines = [atom_lines]

    PDB_data = [
        (
            line[:6].strip(),
            i,
            line[12:16].strip(),
            line[16:17].strip(),
            line[17:20].strip(),
            line[21:22].strip(),
            int(line[22:26]),
            line[26:27].strip(),
            (float(line[30:38]), float(line[38:46]), float(line[46:54])),
            float(line[54:60]),
            float(line[60:66]),
            line[72:73].strip(),
            line[76:78].strip(),
            line[78:80].strip(),
        )
        for i, line in enumerate(lines[0])
    ]

    pdb_dict = {key: np.array(data) for key, data in zip(keys, zip(*PDB_data))}

    # Parse connect data
    if len(connect) > 0:
        c_bonds, h_bonds, i_bonds = parse_connect(connect)
        bonds = np.array(list(c_bonds))
        pdb_dict["bonds"] = bonds
        pdb_dict["bond_types"] = np.array(
            [BondType.UNSPECIFIED for _ in range(len(bonds))]
        )

    trajectory = [pdb_dict.pop("coords")]

    if len(lines) > 1:
        for struct in lines[1:]:
            frame_coords = [(line[30:38], line[38:46], line[46:54]) for line in struct]

            if len(frame_coords) != len(PDB_data):
                raise ValueError(
                    "All models in a multistate PDB must have the same atoms"
                )

            trajectory.append(frame_coords)

    pdb_dict["trajectory"] = np.array(trajectory, dtype=float)
    pdb_dict["name"] = Path(pdb_file).name

    return pdb_dict


def read_sdf(file_name):
    """
    Reads SDF file into a dictionary structure.

    Parameters
    ----------
    file_name: str, Path
        string or Path object directing the function to the sdf file to read.

    Returns
    -------
    mols: List[dict]
        List of dictionaries containing all the sdf data for each molecule in the sdf file.
    """
    # Read in mol blocks
    with open(file_name, "r") as f:
        blocks, block = [], []
        for line in f:
            line = line.rstrip("\r\n")

            if not line == "$$$$":
                block.append(line)
            else:
                blocks.append(block)
                block = []

    mols = []
    for block in blocks:
        mol_data = {
            "name": block[0],
            "software": block[1].strip(),
            "comment": block[2].strip(),
        }
        n_atoms = block[3][:3].strip()
        n_bonds = block[3][3:6].strip()
        n_atom_lists = block[3][6:9].strip()
        chiral = block[3][12:15].strip()
        n_stexts = block[3][15:18].strip()
        properties_lines = block[3][30:33]
        format_version = block[3][33:].strip()

        mol_data["n_atoms"] = int(n_atoms)
        mol_data["n_bonds"] = int(n_bonds)
        mol_data["chiral"] = bool(int(chiral))
        mol_data["format"] = format_version

        atom_end_idx = 4 + mol_data["n_atoms"]
        bond_end_idx = atom_end_idx + mol_data["n_bonds"]
        mol_data["atoms"] = []
        for idx, line in enumerate(block[4:atom_end_idx]):
            atom = {}
            atom["index"] = idx
            atom["xyz"] = [float(line[0:10]), float(line[10:20]), float(line[20:30])]
            atom["element"] = line[31:34].strip()
            atom["mass difference"] = int(float(line[34:36]))
            atom["charge"] = int(line[36:39])
            atom["stereo"] = int(line[39:42])
            atom["valence"] = int(line[48:51]) if len(line) >= 51 else 0
            mol_data["atoms"].append(atom)

        mol_data["bonds"] = []
        for idx, line in enumerate(block[atom_end_idx:bond_end_idx]):
            bond = {}
            bond["idx1"] = int(line[0:3]) - 1  # Coerce to 0 index
            bond["idx2"] = int(line[3:6]) - 1  # Coerce to 0 index
            bond["type"] = int(line[6:9])
            bond["stereo"] = int(line[9:12])

            mol_data["bonds"].append(bond)

        properties = {}
        for idx, line in enumerate(block[bond_end_idx:]):
            if line == "M  END":
                break

            if line.startswith("M  "):
                key = line[3:6]
                lst = properties.setdefault(key, [])
                lst.append(line[6:])
            else:
                break

        for key, val in properties.items():
            properties[key] = parse_property_lines(key, val)

        mol_data["properties"] = properties
        data_end_idx = bond_end_idx + idx + 1
        data_iterator = iter(block[data_end_idx:])
        data_block = []
        for line in data_iterator:
            opened = line.startswith(">")

            while opened:
                data_block.append(line)
                line = next(data_iterator, "")
                opened = bool(line)

            if data_block:
                header, data = parse_data_block(data_block)
                mol_data[header] = data
                data_block = []

        mols.append(mol_data)

    return mols


def parse_data_block(data_block):
    """
    Helper function to parse the data block of an SDF file. Only used internally.

    Parameters
    ----------
    data_block: List[str]
        The lines of the sdf file pertaining to the data block

    Returns
    -------
    header: str
        The header of the data block
    data_string: str
        a string containing all data in the data block
    """
    data_string = "\n".join(data_block[1:])
    header_block = data_block[0][1:]
    header = header_block[header_block.find("<") + 1 : header_block.find(">")]

    return header, data_string


standard_property_vnames = {
    "CHG": "charges",
    "RAD": "multiplicities",
    "ISO": "mass",
    "RBC": "ring bond count",
    "SUB": "substitution count",
    "UNS": "unsaturated atom",
}


def parse_property_lines(property, lines):
    """
    Helper function to parse properties from sdf files.

    Parameters
    ----------
    property: str
        Property being parsed

    lines: List[str]
        The lines of the sdf file pertaining to the property block being parsed.

    Returns
    -------
    dict | List[str]
        dictionary of the atom ids mapping to the properties for each atom if `property` is known to chilife otherwise
        the input lines as provided.

    """
    if property in standard_property_vnames:
        vname = standard_property_vnames[property]
        k, v = parse_standard_property(lines)
        return {
            "atom_ids": np.array(k, dtype=int) - 1,
            vname: np.array(v, dtype=int),
        }

    else:
        return lines


def parse_standard_property(lines):
    """
    Helper function to parse standard properties from sdf files.

    Parameters
    ----------
    lines: List[str]
        The lines of the sdf file pertaining to the property block being parsed.

    Returns
    -------
    k: List
        Atom ids for which this property is relevant.
    V: List
        Quantitative value of the property.
    """
    k, v = [], []
    for line in lines:
        entries = textwrap.wrap(line, 4)
        k.extend(entries[1::2])
        v.extend(entries[2::2])

    return k, v


def standard_property_to_string(key, property):
    """
    Helper function to convert standard properties in a dictionary to a string format for writing sdf files.

    Parameters
    ----------
    key: str
        3 letter keycode of the property
    property: dict
        Property dict of a standard property as parsed by `parse_standard_property`.

    Returns
    -------
    strings: List[str]
        List of strings containing the property lines of an sdf file.
    """
    strings = []
    data_col = standard_property_vnames[key]

    n = 1
    running_str = " "
    for idx, val in zip(property["atom_ids"], property[data_col]):
        if n >= 8:
            running_str = f"M  {key}{n:>3d}" + running_str + "\n"
            strings.append(running_str)
            running_str = " "
            n = 1

        running_str = running_str + f"{str(idx + 1):>3s} {str(val):>3s}"
        n += 1

    if running_str != " ":
        running_str = f"M  {key}{n:>3d}" + running_str + "\n"
        strings.append(running_str)

    return strings


def write_sdf(sdf_data, file_name):
    """
    Function to write sdf file from sdf format in a dict as read by `read_sdf`.

    Parameters
    ----------
    sdf_data: Dict | List[Dict]
        The sdf data in chilife format
    file_name: str
        Name of the file to write the sdf data to.
    """
    if isinstance(sdf_data, dict):
        sdf_data = [sdf_data]

    lines = []

    for mol in sdf_data:
        # Headder
        lines.append(f"{mol['name']}\n")
        lines.append(f"\tchiLife {chilife.__version__}\n")
        lines.append(f"{mol['comment']}\n")
        lines.append(
            f"{mol['n_atoms']:>3}{mol['n_bonds']:>3}{0:>3}{0:>3}{mol['chiral']:>3}"
            f"{0:>3}{0:>3}{0:>3}{0:>3}{0:>3}999 V2000\n"
        )

        # Coords
        atoms = mol["atoms"]
        for atom in atoms:
            x, y, z = atom["xyz"]
            lines.append(
                f"{x:10.4f}{y:10.4f}{z:10.4f}{atom['element']:>2} {atom['mass difference']:>3}{atom['charge']:>3}"
                f"{atom['stereo']:>3}{0:>3}{0:>3}{atom['valence']:>3}"
                + f"{0:>3}" * 6
                + "\n"
            )

        # Bonds
        bonds = mol["bonds"]
        for bond in bonds:
            lines.append(
                f"{bond['idx1'] + 1:>3}{bond['idx2'] + 1:>3}{bond['type']:>3}{bond['stereo']:>3}\n"
            )

        # Write CHG RAD ISO Data
        for key, property in mol["properties"].items():
            if key in standard_property_vnames:
                lines += standard_property_to_string(key, property)

            else:
                lines += property

        lines.append("M  END\n")

        # Write data blocks
        standard_keys = {
            "name",
            "software",
            "comment",
            "n_atoms",
            "n_bonds",
            "chiral",
            "format",
            "atoms",
            "bonds",
            "properties",
        }
        data_keys = [key for key in mol if key not in standard_keys]
        for key in data_keys:
            lines.append(f"> <{key}>\n")
            lines.append(f"{mol[key]}")
            lines.append("\n\n")

        lines.append("$$$$\n")

    with open(file_name, "w") as f:
        f.writelines(lines)


def write_protein(
    pdb_file: TextIO,
    protein: Union[mda.Universe, mda.AtomGroup, MolecularSystemBase],
    name: str = None,
    conect: bool = False,
    frames: Union[int, str, ArrayLike, None] = "all",
) -> None:
    """
    Helper function to write protein PDBs from MDAnalysis and MolSys objects.

    Parameters
    ----------
    pdb_file : TextIO
        File to save the protein to
    protein : MDAnalysis.Universe, MDAnalysis.AtomGroup, MolSys
        MDAnalysis or MolSys object to save
    name : str
        Name of the protein to put in the header
    frames : int, str, ArrayLike or None
        Frames of the trajectory/ensemble to save to file
    """

    # Change chain identifier if longer than 1
    available_segids = iter("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    for seg in protein.segments:
        if len(seg.segid) > 1:
            seg.segid = next(available_segids)

    traj = protein.universe.trajectory
    pdb_file.write(f"HEADER {name}\n")
    frames = None if frames == "all" and len(traj) == 1 else frames

    if frames == "all":
        for mdl, ts in enumerate(traj):
            write_frame(pdb_file, protein.atoms, mdl)
    elif hasattr(frames, "__len__"):
        for mdl, frame in enumerate(frames):
            traj[frame]
            write_frame(pdb_file, protein.atoms, mdl)
    elif frames is not None:
        traj[frames]
        write_frame(pdb_file, protein.atoms)
    else:
        write_frame(pdb_file, protein.atoms)

    if conect:
        bonds = (
            protein.bonds.indices
            if hasattr(protein, "bonds")
            else protein.topology.bonds
            if hasattr(protein.topology, "bonds")
            else None
        )

        if bonds is not None:
            bonds_1 = bonds + 1
            write_bonds(pdb_file, bonds_1)


def write_frame(pdb_file: TextIO, atoms, frame=None, coords=None):
    """
    Helper function to write multiple conformational states of a protein to a PDB file.

    Parameters
    ----------
    pdb_file : TextIO
        IO object to write the frame to.
    atoms: chilife.MolSys.AtomSelection | MDAnalysis.AtomGroup
        Set of atoms to write to the pdb file.
    frame: int | None
        The frame number to write. Should be 0 indexed.
    coords: ArrayLike | None
        Coords to write to the pdb file. If None are provided the coords from `atoms` are used.
    """
    if frame is not None:
        pdb_file.write(f"MODEL {frame + 1}\n")

    if coords is None:
        coords = atoms.positions

    if not hasattr(atoms.universe, "occupancies"):
        atoms.universe.add_TopologyAttr(
            "occupancies", np.ones(len(atoms.universe.atoms))
        )

    if not hasattr(atoms.universe, "bfactors"):
        atoms.universe.add_TopologyAttr("bfactors", np.ones(len(atoms.universe.atoms)))

    [
        pdb_file.write(
            fmt_str.format(
                i + 1,
                atom.name,
                atom.resname[:3],
                atom.segid,
                atom.resnum,
                *coord,
                atom.occupancy,
                atom.bfactor,
                atom.type,
            )
        )
        for i, (atom, coord) in enumerate(zip(atoms, coords))
    ]

    pdb_file.write("TER\n")

    if frame is not None:
        pdb_file.write("ENDMDL\n")


def write_ic(
    pdb_file: TextIO,
    ic: MolSysIC,
    name: str = None,
    conect: bool = None,
    frames: Union[int, str, ArrayLike, None] = "all",
) -> None:
    """
    Write a :class:`~MolSysIC` internal coordinates object to a pdb file.

    Parameters
    ----------
    pdb_file : TextIO
        open file or io object.
    ic: MolSysIC
        chiLife internal coordinates object.
    name : str
        Name of the molecule to be used for the HEADER.
    conect : bool, optional
        Write PDB CONECT information to file.
    frames : int, str, ArrayLike or None
        Frames of the trajectory/ensemble to save to file
    """
    frames = None if frames == "all" and len(ic.trajectory) == 1 else frames
    pdb_file.write(f"HEADER {name}\n")
    traj = ic.trajectory
    if frames == "all":
        for mdl, ts in enumerate(traj):
            write_frame(pdb_file, ic.atoms, mdl, ic.coords)
    elif hasattr(frames, "__len__"):
        for mdl, frame in enumerate(frames):
            traj[frame]
            write_frame(pdb_file, ic.atoms, mdl, ic.coords)
    elif frames is not None:
        traj[frames]
        write_frame(pdb_file, ic.atoms, coords=ic.coords)
    else:
        write_frame(pdb_file, ic.atoms, coords=ic.coords)

    if conect:
        bonds = ic.topology.bonds
        write_bonds(pdb_file, bonds)


def write_labels(
    pdb_file: TextIO,
    *args: sl.SpinLabel,
    KDE: bool = True,
    sorted: bool = True,
    write_spin_centers: bool = True,
    conect: bool = False,
) -> None:
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
    write_spin_centers : bool
        Write spin centers (atoms named NEN) as a seperate object with weights mapped to q-factor.

    Returns
    -------
    None
    """

    # Write spin label models
    for k, label in enumerate(args):
        pdb_file.write(f"HEADER {label.name}\n")

        # Save models in order of weight

        sorted_index = (
            np.argsort(label.weights)[::-1] if sorted else np.arange(len(label.weights))
        )
        norm_weights = label.weights / label.weights.max()
        if isinstance(label, dre.dRotamerEnsemble):
            sites = np.concatenate(
                [
                    np.ones(len(label.RE1.atoms), dtype=int) * int(label.site1),
                    np.ones(len(label.RE2.atoms), dtype=int) * int(label.site2),
                ]
            )
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
        if conect:
            write_bonds(pdb_file, label.bonds)

    # Write spin density at electron coordinates
    for k, label in enumerate(args):
        if not hasattr(label, "spin_centers"):
            continue
        if not np.any(label.spin_centers):
            continue

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

        if write_spin_centers:
            pdb_file.write(f"HEADER {label.name}_density\n")
            norm_weights = vals / vals.max()
            [
                pdb_file.write(
                    fmt_str.format(
                        i,
                        "NEN",
                        label.res[:3],
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


def write_atoms(
    file: TextIO,
    atoms: Union[MDAnalysis.Universe, MDAnalysis.AtomGroup, MolecularSystemBase],
    coords: ArrayLike = None,
) -> None:
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
            f"{coord[0]:8.3f}{coord[1]:8.3f}{coord[2]:8.3f}{atom.occupancy:6.2f}{atom.bfactor:6.2f}          {atom.type:>2s}  \n"
        )


def write_bonds(pdb_file, bonds):
    """
    Write PDB CONECT information to ``pdb_file`` for the provided ``bonds``.

    Parameters
    ----------
    pdb_file: TextIO
        File to write bonds to.
    bonds: ArrayLike
        Array of the indices of atom pairs that make the bonds.
    """
    active_atom = bonds[0][0]
    line = "CONECT" + f"{active_atom:>5d}"

    lines = []
    for b in bonds:
        if b[0] != active_atom or len(line) > 30:
            lines.append(line + "\n")
            active_atom = b[0]
            line = "CONECT" + f"{active_atom:>5d}"

        line += f"{b[1]:>5d}"
    lines.append(line + "\n")
    pdb_file.writelines(lines)


molecule_class = {
    re.RotamerEnsemble: "rotens",
    dre.dRotamerEnsemble: "rotens",
    IntrinsicLabel: "rotens",
    le.LigandEnsemble: "rotens",
    mda.Universe: "molcart",
    mda.AtomGroup: "molcart",
    MolecularSystemBase: "molcart",
    MolSysIC: "molic",
}


rotlib_formats = {
    1.0: (
        "rotlib",  #
        "resname",
        "coords",
        "internal_coords",
        "weights",
        "atom_types",
        "atom_names",
        "dihedrals",
        "dihedral_atoms",
        "type",
        "format_version",
    )
}

rotlib_formats[1.1] = *rotlib_formats[1.0], "description", "comment", "reference"
rotlib_formats[1.2] = rotlib_formats[1.0]
rotlib_formats[1.3] = rotlib_formats[1.2]
rotlib_formats[1.4] = *rotlib_formats[1.3], "backbone_atoms", "aln_atoms"
rotlib_formats[1.5] = rotlib_formats[1.4]


def read_cif(file_name):
    """
    Read a CIF file into a dictionary.

    Parameters
    ----------
    file_name: str | Path
        The cif file to read.

    Returns
    -------
    cif_data: Dict
        All data contained in the cif file.
    """
    data_blocks = {}
    with open(file_name, "r") as f:
        for line in f:
            if line.startswith("data"):
                key = line.split("_")[1].strip()
                current_block = data_blocks.setdefault(key, [])
                current_block.append(line)
            elif "current_block" in locals():
                current_block.append(line)
            else:
                continue

    cif_data = {}
    for key, value in data_blocks.items():
        cif_data[key] = parse_cif_data_block(value)

    return cif_data


def parse_cif_data_block(data_block: list[str]) -> dict:
    """
    Helper function to parse cif data from a cif file.

    Parameters
    ----------
    data_block: list[str]
        List of lines from a cif file corresponding to a singular data block as defined by cif specifications.

    Returns
    -------
    parsed_block: dict
        The provided data block parsed into a dictionary.
    """
    parsed_block = {}
    data_block_iterator = iter(data_block)
    in_loop = False
    for line in data_block_iterator:
        if line.startswith("loop_"):
            in_loop = True
            subject_keys = []
            subject_values = []
        elif (line.startswith("#") or line.strip() == "") and in_loop:
            if topic_key == "chem_comp":
                breakpoint()
            for k, v in zip(subject_keys, zip(*subject_values)):
                topic_dict[k] = v
            in_loop = False

        elif in_loop:
            if line.startswith("_"):
                topic_key, subject_key = line.strip().split(".")
                topic_dict = parsed_block.setdefault(topic_key[1:], {})
                subject_keys.append(subject_key)
            else:
                svals = split_with_quotes(line)

                # Some lines are too long and wrap multiple lines.
                while len(svals) < len(subject_keys):
                    line = next(data_block_iterator)

                    # Consider a multiline field within a multiline loop item
                    if line[0] == ";":
                        value = line[1:]
                        while True:
                            line = next(data_block_iterator)
                            if line.startswith(";"):
                                break
                            value += line.strip()
                        line = value

                    svals += split_with_quotes(line)

                subject_values.append(svals)

        elif line.startswith("_"):
            kv = split_with_quotes(line)

            if len(kv) == 2:
                key, value = kv
            else:
                # Extract next line value
                key = kv.pop(0)
                value = next(data_block_iterator).strip()
                if value[0] == "'":
                    value.strip("'")

            # check for multiline entry
            if value[0] == ";":
                value = value[1:]
                while True:
                    line = next(data_block_iterator)
                    if line.startswith(";"):
                        break
                    value += line.strip()

            topic_key, subject_key = key.split(".")

            topic_dict = parsed_block.setdefault(topic_key[1:], {})
            topic_dict[subject_key] = value

    return parsed_block


def split_with_quotes(string):
    """
    Helper function to split a string by it's whitespace into a list while respecting quotes.

    Parameters
    ----------
    string:
        The string to split

    Returns
    -------
    vals: list[str]
        List of substrings obtained from splitting the parnet string.

    """
    vals = string.split()
    myit = enumerate(vals)
    replace = []

    for i, val in myit:
        if val[0] == '"' or val[0] == "'":
            sym = val[0]
            tvals = [val]
            ends_with_quote = val[-1] == sym and len(val) > 1
            j = i + 1
            while not ends_with_quote:
                _j, val = next(myit)
                j = _j + 1
                tvals.append(val)
                ends_with_quote = val[-1] == sym

            val = " ".join(tvals)
            val = val.strip(sym)
            replace.append((i, j, val))

    if replace:
        for i, j, val in reversed(replace):
            vals[i:j] = [val]

    return vals


def create_ccd_dicts(cif_data):
    """
    Creates chemical composition dictionary (CCD) dictionaries from CCD entries in a cif file.

    Parameters
    ----------
    cif_data: dict
        cif file data as parsed from :func:`~read_cif`.

    Returns
    -------
    ccd_data: dict
        The dictionary containing the CCD data from a cif file in a chilife-readable format.
    """
    ccd_data = {}
    atom_data = cif_data.get("chem_comp_atom", None)
    if atom_data is not None:
        for res, atom_id, atom_type, aromatic, stereo in zip(
            atom_data["comp_id"],
            atom_data["atom_id"],
            atom_data["type_symbol"],
            atom_data["pdbx_aromatic_flag"],
            atom_data["pdbx_stereo_config"],
        ):
            res_dict = ccd_data.setdefault(res, {})

            if "chem_comp_atom" not in res_dict:
                res_dict["chem_comp_atom"] = {
                    "comp_id": [],
                    "atom_id": [],
                    "type_symbol": [],
                    "pdbx_aromatic_flag": [],
                    "pdbx_stereo_config": [],
                }

            atom_dict = res_dict["chem_comp_atom"]
            atom_dict["comp_id"].append(res)
            atom_dict["atom_id"].append(atom_id)
            atom_dict["type_symbol"].append(atom_type)
            atom_dict["pdbx_aromatic_flag"].append(aromatic)
            atom_dict["pdbx_stereo_config"].append(stereo)

    bond_data = cif_data.get("chem_comp_bond", None)
    if bond_data is not None:
        for res, atom_id1, atom_id2, bond_order, aromatic, stereo in zip(
            bond_data["comp_id"],
            bond_data["atom_id_1"],
            bond_data["atom_id_2"],
            bond_data["value_order"],
            bond_data["pdbx_aromatic_flag"],
            bond_data["pdbx_stereo_config"],
        ):
            res_dict = ccd_data.setdefault(res, {})
            if "chem_comp_bond" not in res_dict:
                res_dict["chem_comp_bond"] = {
                    "comp_id": [],
                    "atom_id_1": [],
                    "atom_id_2": [],
                    "value_order": [],
                    "pdbx_aromatic_flag": [],
                    "pdbx_stereo_config": [],
                }

            bond_dict = res_dict["chem_comp_bond"]
            bond_dict["comp_id"].append(res)
            bond_dict["atom_id_1"].append(atom_id1)
            bond_dict["atom_id_2"].append(atom_id2)
            bond_dict["value_order"].append(bond_order)
            bond_dict["pdbx_aromatic_flag"].append(aromatic)
            bond_dict["pdbx_stereo_config"].append(stereo)

    chem_comp_data = cif_data.get("chem_comp", None)

    if chem_comp_data is not None:
        for i, res in enumerate(chem_comp_data["id"]):
            res_dict = ccd_data.setdefault(res, {})
            res_dict["chem_comp"] = {
                key: chem_comp_data[key][i] for key in chem_comp_data.keys()
            }

    return ccd_data


def parse_struct_conn_bonds(molsys, struct_conn):
    """
    Parse the ``_struct_conn`` loop of a cif file into inter-residue bonds on a ``MolSys``.

    Only rows whose ``conn_type_id`` is in :data:`STRUCT_CONN_BOND_TYPES` (``covale``, ``disulf``,
    ``metalc``) are kept. Rows whose ``ptnr*_symmetry`` is anything other than ``1_555`` / ``.`` /
    ``?`` are skipped (inter-molecular symmetry-mate contacts are not real bonds inside this
    asymmetric unit). Rows whose partners cannot be resolved to an atom in ``molsys`` are also
    skipped.

    Partners are resolved using the ``auth`` chain/resnum fields (matching
    :attr:`MolSys.chains` / :attr:`MolSys.resnums`) together with the insertion code, atom name,
    and alternate-location fields (matching :attr:`MolSys.icodes`, :attr:`MolSys.names`, and
    :attr:`MolSys.altlocs`).

    Parameters
    ----------
    molsys : MolSys
        The molecular system the bonds should be resolved against.
    struct_conn : dict
        ``struct_conn`` sub-dictionary from :func:`~read_cif` (keys map to tuples/lists of values,
        or to scalars when the CIF loop only has a single row).

    Returns
    -------
    bonds : np.ndarray
        ``(N, 2)`` array of atom indices into ``molsys``. Empty when no rows resolve.
    bond_types : np.ndarray
        ``(N,)`` array of :class:`~chilife.Topology.BondType` values, one per bond.
    """

    conn_types = struct_conn.get("conn_type_id", ())
    n_rows = len(conn_types)
    if n_rows == 0:
        return np.empty((0, 2), dtype=int), np.empty((0,), dtype=object)

    bonds, bond_types = [], []
    for row in range(n_rows):
        ctype = str(conn_types[row]).lower()
        if ctype not in STRUCT_CONN_BOND_TYPES:
            continue

        collapse_values = (None, ".", "?")
        c = struct_conn["ptnr1_auth_asym_id"][row]
        r = struct_conn["ptnr1_auth_seq_id"][row]
        i = struct_conn["pdbx_ptnr1_PDB_ins_code"][row]
        n = struct_conn["ptnr1_label_atom_id"][row]
        a = struct_conn["pdbx_ptnr1_label_alt_id"][row]
        crina1 = tuple(x if x not in collapse_values else "" for x in (c, r, i, n, a))

        c = struct_conn["ptnr2_auth_asym_id"][row]
        r = struct_conn["ptnr2_auth_seq_id"][row]
        i = struct_conn["pdbx_ptnr2_PDB_ins_code"][row]
        n = struct_conn["ptnr2_label_atom_id"][row]
        a = struct_conn["pdbx_ptnr2_label_alt_id"][row]
        crina2 = tuple(x if x not in collapse_values else "" for x in (c, r, i, n, a))

        i1, i2 = molsys.crina.get(crina1, None), molsys.crina.get(crina2, None)

        if i1 is None or i2 is None:
            continue

        order_key = struct_conn["pdbx_value_order"][row]
        btype = CIF_BOND_TO_XL.get(order_key, BondType.UNSPECIFIED)
        bond = sorted([i1, i2])

        # Do not overwrite existing bonds with BondType.UNSPECIFIED if the existing bond is specified.
        if btype == BondType.UNSPECIFIED:
            exists = np.argwhere(np.all(molsys._bonds == bond, axis=1)).flatten()
            if (
                len(exists) > 0
                and molsys._bond_types[exists[0]] != BondType.UNSPECIFIED
            ):
                continue

        bonds.append(bond)
        bond_types.append(btype)

    return np.array(bonds, dtype=int), np.array(bond_types, dtype=object)


def struct_conn_from_bonds(molsys, ccd_data=None):
    """
    Build a cif ``_struct_conn`` loop dict from the inter-residue bonds on a ``MolSys``.

    Only bonds whose two atoms belong to different residues are emitted, and the auto-inferred
    polymer peptide link (``C`` -> ``N`` between consecutive residues whose previous residue has a
    ``chem_comp.type`` in :data:`chilife.Topology.POLYMER_LINKAGE_TYPES`) is excluded so that a
    round-trip through :meth:`MolSys.from_cif` / :meth:`MolSys.write_cif` stays idempotent (the
    peptide link is re-inferred on read from the polymer sequence via
    :func:`~chilife.Topology.bonds_from_ccd_data`).

    Parameters
    ----------
    molsys : MolSys
        The molecular system to read bonds from.
    ccd_data : dict | None
        Per-residue CCD dict keyed by residue name (as consumed by
        :func:`~chilife.Topology.bonds_from_ccd_data`) used to look up ``chem_comp.type`` when
        deciding whether a bond is an implicit polymer peptide link. Falls back to
        ``chilife.bio_ccd`` when not provided.

    Returns
    -------
    struct_conn : dict | None
        Dict keyed by :data:`STRUCT_CONN_KEYS`, mapping each column name to a list of string values
        (one per bond). Returns ``None`` when no inter-residue bonds are present.
    """
    if molsys.bonds is None or len(molsys.bonds) == 0:
        return None

    counters = {}
    rows = defaultdict(list)
    for (b1, b2), btype in zip(molsys.bonds, molsys.bond_types):
        if molsys.resindices[b1] == molsys.resindices[b2]:
            continue

        elif (
            molsys.segindices[b1] == molsys.segindices[b2]
            and abs(molsys.resnums[b1] - molsys.resnums[b2]) == 1
        ):
            continue

        else:
            conn_type = _classify_struct_conn(molsys, b1, b2)
            counters[conn_type] = counters.get(conn_type, 0) + 1
            bond_id = f"{conn_type}{counters[conn_type]}"
            value_order = XL_BOND_TO_CIF.get(btype, "?")

            if value_order in ("metalc", "disulf"):
                value_order = "?"

            rows["id"].append(bond_id)
            rows["conn_type_id"].append(conn_type)
            rows["pdbx_leaving_atom_flag"].append("?")
            rows["pdbx_PDB_id"].append("?")
            _append_partner(rows, molsys, b1, which=1)
            _append_partner(rows, molsys, b2, which=2)
            rows["pdbx_ptnr3_label_atom_id"].append("?")
            rows["pdbx_ptnr3_label_seq_id"].append("?")
            rows["pdbx_ptnr3_label_comp_id"].append("?")
            rows["pdbx_ptnr3_label_asym_id"].append("?")
            rows["pdbx_ptnr3_label_alt_id"].append("?")
            rows["pdbx_ptnr3_PDB_ins_code"].append("?")
            rows["details"].append("?")
            rows["pdbx_dist_value"].append("?")
            rows["pdbx_value_order"].append(value_order)
            rows["pdbx_role"].append("?")

    if not rows["id"]:
        return None

    return rows


def _classify_struct_conn(molsys, i1, i2):
    """Pick the CIF ``conn_type_id`` for the bond between atoms ``i1`` and ``i2``."""
    n1 = str(molsys.names[i1]).strip()
    n2 = str(molsys.names[i2]).strip()
    r1 = str(molsys.resnames[i1]).strip()
    r2 = str(molsys.resnames[i2]).strip()
    if r1 == "CYS" and r2 == "CYS" and n1 == "SG" and n2 == "SG":
        return "disulf"

    e1 = str(molsys.atypes[i1]).strip().upper()
    e2 = str(molsys.atypes[i2]).strip().upper()
    if e1 not in NONMETAL_ELEMENTS or e2 not in NONMETAL_ELEMENTS:
        return "metalc"
    return "covale"


def _append_partner(rows, molsys, idx, which):
    """Append a single partner's struct_conn fields to ``rows`` for atom index ``idx``."""
    seg = str(molsys.segids[idx])
    chain = str(molsys.chains[idx])
    resname = str(molsys.resnames[idx])
    resnum = str(int(molsys.resnums[idx]))
    label_seq = str(int(molsys.resindices[idx]) + 1)
    name = str(molsys.names[idx]).strip()
    altloc_raw = str(molsys.altlocs[idx]).strip()
    altloc = altloc_raw if altloc_raw else "?"
    icode_raw = str(molsys.icodes[idx]).strip()
    icode = icode_raw if icode_raw else "?"

    rows[f"ptnr{which}_label_asym_id"].append(chain)
    rows[f"ptnr{which}_label_comp_id"].append(resname)
    rows[f"ptnr{which}_label_seq_id"].append(label_seq)
    rows[f"ptnr{which}_label_atom_id"].append(name)
    rows[f"pdbx_ptnr{which}_label_alt_id"].append(altloc)
    rows[f"pdbx_ptnr{which}_PDB_ins_code"].append(icode)
    if which == 1:
        rows["pdbx_ptnr1_standard_comp_id"].append("?")
    rows[f"ptnr{which}_symmetry"].append("1_555")
    rows[f"ptnr{which}_auth_asym_id"].append(chain)
    rows[f"ptnr{which}_auth_comp_id"].append(resname)
    rows[f"ptnr{which}_auth_seq_id"].append(resnum)


def write_cif(file_name, cif_data):
    """
    Function to write a CIF file from cif_data as read by :func:`~read_cif`

    Parameters
    ----------
    file_name: str | Path
        File to write the cif_data to.
    cif_data: Dict
        cif data to write to the file.
    """
    lines = []
    for struct in cif_data:
        lines.append(f"data_{struct}\n")
        lines.append("# \n")
        for key1, value1 in cif_data[struct].items():
            max_subkey_len = max(len(k) for k in value1.keys())
            key_length = len(key1) + 1 + max_subkey_len + 3
            loop_keys = []
            loop_values = []
            max_val_lens = []
            for key2, value2 in value1.items():
                line = f"_{key1}.{key2}".ljust(key_length)
                if isinstance(value2, str):
                    if " " in value2 and value2[0] != "'":
                        value2 = "'" + value2 + "'"

                    if len(value2) < 80:
                        line += f" {value2} \n"
                        lines.append(line)
                    elif " " in value2:
                        lines.append(line + "\n")
                        lines.append(value2 + "\n")
                    else:
                        lines.append(line + "\n")
                        prelude = ";"
                        for string in textwrap.wrap(value2, 80):
                            lines.append(prelude + string + "\n")
                            prelude = ""
                        lines.append("; \n")

                elif isinstance(value2, Sequence):
                    loop_keys.append(line + "\n")
                    value2 = [v if "'" not in v else '"' + v + '"' for v in value2]

                    value2 = [
                        v if (" " not in v) and (v[0] != "_") else "'" + v + "'"
                        for v in value2
                    ]

                    loop_values.append(value2)
                    try:
                        max_val_lens.append(max(len(x) for x in value2))
                    except ValueError:
                        print(value2)

            if loop_values:
                lines.append("loop_\n")
                for key in loop_keys:
                    lines.append(key)

                for row in zip(*loop_values):
                    line = " ".join(x.ljust(n) for x, n in zip(row, max_val_lens))
                    lines.append(line + "\n")

            lines.append("# \n")
    with open(file_name, "w") as f:
        f.writelines(lines)


def join_ccd_info(ccd_list, ccd_data=None):
    """
    Join CCD info from a list of CCD entries into a dictionary that can be used to write ccd info to a cif file.

    Parameters
    ----------
    ccd_list: List[str]
        List of CCD codes to extract from the ccd_data dictionary to be joined in a single ccd_info dict.
    ccd_data: Dict
        Superset dictionary of CCD entries that contain all the CCD info for the CCD codes specified in the ``ccd_list``
        If ``ccd_data`` is not provided the default chilife ``bio_ccd`` will be used which contains all biomolecular CCD
        entries from the PDB (as of mid 2025). If a CCD code in ``ccd_list`` is not in ``ccd_data`` it will be skipped.

    Returns
    -------
    return_ccd: Dict
        A dictionary containing all ccd data for the set of codes provided and ready to be written to a cif file.
    """
    # use default CCD if none is provided
    if ccd_data is None:
        ccd_data = chilife.bio_ccd
    # Otherwise overwrite default CCD data with what was provided and add missing CCD data
    else:
        _ccd_data = {k: v for k, v in chilife.bio_ccd.items()}
        _ccd_data.update(ccd_data)
        ccd_data = _ccd_data

    individual_CCDs = [ccd_data[ccd] for ccd in ccd_list if ccd in ccd_data]
    return_ccd = {
        "chem_comp": {},
        "chem_comp_atom": {key: [] for key in CHEM_COMP_ATOM_KEYS},
        "chem_comp_bond": {key: [] for key in CHEM_COMP_BOND_KEYS},
    }

    return_ccd["chem_comp"]["id"] = tuple(
        ccd["chem_comp"]["id"] for ccd in individual_CCDs if "chem_comp" in ccd
    )
    return_ccd["chem_comp"]["type"] = tuple(
        ccd["chem_comp"]["type"] for ccd in individual_CCDs if "chem_comp" in ccd
    )
    return_ccd["chem_comp"]["name"] = tuple(
        ccd["chem_comp"]["name"] for ccd in individual_CCDs if "chem_comp" in ccd
    )
    return_ccd["chem_comp"]["pdbx_synonyms"] = tuple(
        ccd["chem_comp"]["pdbx_synonyms"]
        for ccd in individual_CCDs
        if "chem_comp" in ccd
    )
    return_ccd["chem_comp"]["formula"] = tuple(
        ccd["chem_comp"]["formula"] for ccd in individual_CCDs if "chem_comp" in ccd
    )
    return_ccd["chem_comp"]["formula_weight"] = tuple(
        ccd["chem_comp"]["formula_weight"]
        for ccd in individual_CCDs
        if "chem_comp" in ccd
    )

    for ccd in individual_CCDs:
        # Add Atom information
        for key in CHEM_COMP_ATOM_KEYS:
            return_ccd["chem_comp_atom"][key].extend(ccd["chem_comp_atom"][key])

        if "chem_comp_bond" not in ccd:
            continue

        # Ensure all fields are tuples/lists
        ccd["chem_comp_bond"] = {
            k: v if isinstance(v, (list, tuple)) else (v,)
            for k, v in ccd["chem_comp_bond"].items()
        }

        # Add bond information
        for key in CHEM_COMP_BOND_KEYS:
            return_ccd["chem_comp_bond"][key].extend(ccd["chem_comp_bond"][key])

    # Convert to tuples
    return_ccd["chem_comp_atom"] = {
        k: tuple(v) for k, v in return_ccd["chem_comp_atom"].items()
    }
    return_ccd["chem_comp_bond"] = {
        k: tuple(v) for k, v in return_ccd["chem_comp_bond"].items()
    }

    # Add PDBx Ordinal
    return_ccd["chem_comp_atom"]["pdbx_ordinal"] = tuple(
        str(x) for x in range(1, len(return_ccd["chem_comp_atom"]["comp_id"]) + 1)
    )
    return_ccd["chem_comp_bond"]["pdbx_ordinal"] = tuple(
        str(x) for x in range(1, len(return_ccd["chem_comp_bond"]["comp_id"]) + 1)
    )

    return return_ccd
