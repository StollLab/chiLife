from typing import Tuple, Dict, Union
from pathlib import Path
import pickle
import shutil
from io import StringIO

import numpy as np
from scipy.stats import gaussian_kde
from memoization import cached
import MDAnalysis as mda

import chilife
from .RotamerEnsemble import RotamerEnsemble
from .SpinLabel import SpinLabel
from .dSpinLabel import dSpinLabel
from .Protein import Protein


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


@cached
def read_sl_library(label: str, user: bool = False) -> Dict:
    """Reads RotamerEnsemble for stored spin labels.

    Parameters
    ----------
    label : str
        3-character abbreviation for desired spin label
    user : bool
        Specifies if the library was defined by a user or if it is a precalculated library

    Returns
    -------
    lib: dict
        Dictionary of SpinLabel rotamer ensemble attributes including coords, weights, dihedrals etc.

    """
    subdir = "UserRotlibs/" if user else "MMM_RotLibs/"
    data = Path(__file__).parent / "data/rotamer_libraries/"
    with np.load(data / subdir / (label + "_rotlib.npz"), allow_pickle=True) as files:
        lib = dict(files)

    del lib["allow_pickle"]

    with open(chilife.RL_DIR / f"residue_internal_coords/{label}_ic.pkl", "rb") as f:
        IC = pickle.load(f)
        if isinstance(IC, list):
            ICn = IC
        else:
            ICn = [
                IC.copy().set_dihedral(
                    np.deg2rad(r), 1, atom_list=lib["dihedral_atoms"]
                )
                for r in lib["dihedrals"]
            ]

    lib["internal_coords"] = ICn

    if "sigmas" not in lib:
        lib["sigmas"] = np.array([])

    lib["_rdihedrals"] = np.deg2rad(lib["dihedrals"])
    lib["_rsigmas"] = np.deg2rad(lib["sigmas"])

    return lib


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
        lib["dihedrals"] = data[:, 1 : nchi + 1]
        lib["sigmas"] = data[:, maxchi + 1 : maxchi + nchi + 1]
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

    return lib


def read_library(
    res: str, Phi: float = None, Psi: float = None) -> Dict:
    """Generalized wrapper function to aid selection of rotamer library reading function.

    Parameters
    ----------
    res : str
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

    if res in chilife.SUPPORTED_LABELS and res not in chilife.SUPPORTED_BB_LABELS:
        return read_sl_library(res)
    elif res in chilife.USER_LABELS or res[:3] in chilife.USER_dLABELS:
        return read_sl_library(res, user=True)
    elif backbone_exists:
        return read_bbdep(res, Phi, Psi)
    else:
        return read_bbdep(res, -60, -50)


def save(
    file_name: str,
    *molecules: Union[RotamerEnsemble, Protein, mda.Universe, mda.AtomGroup, str],
    protein_path: Union[str, Path] = None,
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
    # Check for filename at the beginning of args
    molecules = list(molecules)
    if isinstance(file_name, (RotamerEnsemble, SpinLabel, dSpinLabel,
                              mda.Universe, mda.AtomGroup,
                              chilife.Protein, chilife.AtomSelection)):
        molecules.insert(0, file_name)
        file_name = None

    # Separate out proteins and spin labels
    proteins, labels = [], []
    protein_path = [] if protein_path is None else [protein_path]
    for mol in molecules:
        if isinstance(mol, (RotamerEnsemble, SpinLabel, dSpinLabel)):
            labels.append(mol)
        elif isinstance(mol, (mda.Universe, mda.AtomGroup, chilife.Protein, chilife.AtomSelection)):
            proteins.append(mol)
        elif isinstance(mol, (str, Path)):

            protein_path.append(mol)
        else:
            raise TypeError('chiLife can only save RotamerEnsembles and Proteins. Plese check that your input is '
                             'compatible')

    # Ensure only one protein path was provided (for now)
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
            for protein in proteins:
                if getattr(protein, "filename", None):
                    file_name += ' ' + getattr(protein, "filename", None)
                    file_name = file_name[:-4]

        if file_name == '':
            file_name = "No_Name_Protein"

        # Add spin label information to file name
        if 0 < len(labels) < 3:
            for label in labels:
                file_name += f"_{label.site}{label.res}"
        else:
            file_name += "_many_labels"

        file_name += ".pdb"

    if protein_path is not None:
        print(protein_path, file_name)
        shutil.copy(protein_path, file_name)

    for protein in proteins:
        write_protein(file_name, protein, mode='a+')

    if len(labels) > 0:
        write_labels(file_name, *labels, **kwargs)


def write_protein(file: str, protein: Union[mda.Universe, mda.AtomGroup], mode='a+') -> None:
    """Helper function to write protein pdbs from mdanalysis objects.

    Parameters
    ----------
    file : str
        Name of file to save the protein to
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

    traj = protein.universe.trajectory if isinstance(protein, mda.AtomGroup) else protein.trajectory


    fmt_str = "ATOM  {:5d} {:^4s} {:3s} {:1s}{:4d}    {:8.3f}{:8.3f}{:8.3f}{:6.2f}{:6.2f}          {:>2s}  \n"
    with open(file, mode) as f:
        f.write(f'HEADER {file.rstrip(".pdb")}\n')
        for mdl, ts in enumerate(traj):
            f.write(f"MODEL {mdl}\n")
            [
                f.write(
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
            f.write("TER\n")
            f.write("ENDMDL\n")


def write_labels(file: str, *args: SpinLabel, KDE: bool = True) -> None:
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

    fmt_str = "ATOM  {:5d} {:^4s} {:3s} {:1s}{:4d}    {:8.3f}{:8.3f}{:8.3f}{:6.2f}{:6.2f}          {:>2s}  \n"
    with open(file, "a+", newline="\n") as f:
        # Write spin label models
        f.write("\n")
        for k, label in enumerate(ensembles):
            f.write(f"HEADER {label.name}\n")

            # Save models in order of weight
            sorted_index = np.argsort(label.weights)[::-1]
            for mdl, (conformer, weight) in enumerate(
                zip(label.coords[sorted_index], label.weights[sorted_index])
            ):
                f.write("MODEL {}\n".format(mdl))

                [
                    f.write(
                        fmt_str.format(
                            i,
                            label.atom_names[i],
                            label.res[:3],
                            label.chain,
                            int(label.site),
                            *conformer[i],
                            1.00,
                            weight * 100,
                            label.atom_types[i],
                        )
                    )
                    for i in range(len(label.atom_names))
                ]
                f.write("TER\n")
                f.write("ENDMDL\n")

        # Write electron density at electron coordinates
        for k, label in enumerate(ensembles):
            if not hasattr(label, "spin_centers"):
                continue
            if not np.any(label.spin_centers):
                continue

            f.write(f"HEADER {label.name}_density\n".format(label.label, k + 1))
            spin_centers = np.atleast_2d(label.spin_centers)

            if KDE and np.all(np.linalg.eigh(np.cov(spin_centers.T))[0] > 0) and len(spin_centers) > 5:
                # Perform gaussian KDE to determine electron density
                gkde = gaussian_kde(spin_centers.T, weights=label.weights)

                # Map KDE density to pseudoatoms
                vals = gkde.pdf(spin_centers.T)
            else:
                vals = label.weights

            [
                f.write(
                    fmt_str.format(
                        i,
                        "NEN",
                        label.label[:3],
                        label.chain,
                        int(label.site),
                        *spin_centers[i],
                        1.00,
                        vals[i] * 100,
                        "N",
                    )
                )
                for i in range(len(vals))
            ]

            f.write("TER\n")

