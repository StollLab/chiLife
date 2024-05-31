from pathlib import Path
import rtoml
import pickle
from collections import defaultdict
import numpy as np

# Define useful global variables
SUPPORTED_BB_LABELS = ("R1C",)
DATA_DIR = Path(__file__).parent.absolute() / "data/"
RL_DIR = Path(__file__).parent.absolute() / "data/rotamer_libraries/"

with open(RL_DIR / 'additional_rotlib_dirs.txt', 'r') as f:
    USER_RL_DIR = [Path(x.strip()) for x in f.readlines()]

with open(RL_DIR / "defaults.toml", "r") as f:
    rotlib_defaults = rtoml.load(f)

USER_LIBRARIES = {f.name[:-11] for f in (RL_DIR / "user_rotlibs").glob("*rotlib.npz")}
USER_dLIBRARIES = {f.name[:-12] for f in (RL_DIR / "user_rotlibs").glob("*drotlib.zip")}
USER_dLIBRARIES = USER_dLIBRARIES | {f.name[:-15] for f in (RL_DIR / "user_rotlibs").glob("*drotlib.zip")}

DATA_DIR = Path(__file__).parent.absolute() / "data/"
RL_DIR = Path(__file__).parent.absolute() / "data/rotamer_libraries/"

# Define rotamer dihedral angle atoms
with open(DATA_DIR / "dihedral_defs.toml", "r") as f:
    dihedral_defs = rtoml.load(f)

SUPPORTED_RESIDUES = set(dihedral_defs.keys())
[SUPPORTED_RESIDUES.remove(lab) for lab in ("CYR1", "MTN")]

with open(RL_DIR / "RotlibIndexes.pkl", "rb") as f:
    rotlib_indexes = pickle.load(f)

with open(DATA_DIR / 'BondDefs.pkl', 'rb') as f:
    bond_hmax_dict = {key: (val + 0.4 if 'H' in key else val + 0.35) for key, val in pickle.load(f).items()}
    bond_hmax_dict = defaultdict(lambda: 0, bond_hmax_dict)


    def bond_hmax(a): return bond_hmax_dict.get(tuple(i for i in a), 0)


    bond_hmax = np.vectorize(bond_hmax, signature="(n)->()")

atom_order = {"N": 0, "CA": 1, "C": 2, "O": 3, 'H': 5}

nataa_codes = {'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
               'LYS': 'K', 'LEU': 'L', 'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 'ARG': 'R', 'SER': 'S',
               'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'}

natnu_codes = {'A', 'C', 'G', 'I', 'U', 'DA', 'DC', 'DG', 'DI', 'DT', 'DU'}

mm_backbones = {'aa': ['N', 'CA', 'C', 'O'],
                'nu': ["P",  "O5'", "C5'",  "C4'", "O4'", "C3'",  "C1'", "O3'", "C2'", "O2'", "OP1", "OP2"],
                'ACE': ['C', 'O', 'CH3']}


inataa = {val: key for key, val in nataa_codes.items()}

nataa_codes.update(inataa)
del inataa
