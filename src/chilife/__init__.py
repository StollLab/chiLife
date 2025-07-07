import logging
logging.getLogger("MDAnalysis").setLevel(logging.ERROR)

from .chilife import (distance_distribution, confidence_interval, create_library, create_dlibrary, add_rotlib_dir,
                      remove_rotlib_dir, add_library, remove_library, add_dihedral_def, add_to_defaults,
                      remove_from_defaults, list_available_rotlibs, rotlib_info, repack, add_to_toml)
from .globals import *
from .io import *
from .alignment_methods import parse_backbone, global_mx, local_mx
from .RotamerEnsemble import RotamerEnsemble
from .dRotamerEnsemble import dRotamerEnsemble
from .SpinLabel import SpinLabel
from .dSpinLabel import dSpinLabel
from .IntrinsicLabel import IntrinsicLabel
from .SpinLabelTraj import *
from .MolSys import *
from .MolSysIC import *
from .Topology import *
from .protein_utils import *
from .scoring import *


# RotamerEnsemble = RotamerEnsemble.RotamerEnsemble
# dRotamerEnsemble = dRotamerEnsemble.dRotamerEnsemble

# SpinLabel = SpinLabel.SpinLabel
# dSpinLabel = dSpinLabel.dSpinLabel

__version__ = '1.1.7'
