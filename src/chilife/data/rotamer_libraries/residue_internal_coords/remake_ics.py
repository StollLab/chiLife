import argparse
import pickle
from pathlib import Path

import MDAnalysis as mda
import chilife

parser = argparse.ArgumentParser(description="Rebuild residue *_ic.pkl files from residue_pdbs.")
parser.add_argument(
    "--force",
    action="store_true",
    help="Overwrite existing *_ic.pkl files (default: skip existing).",
)
parser.add_argument(
    "pdbs",
    nargs="*",
    help="Optional residue stems (e.g. trp ala). Default: all *.pdb in residue_pdbs.",
)
args = parser.parse_args()

pdb_directory = Path(__file__).resolve().parent.parent / "residue_pdbs"
if args.pdbs:
    residue_pdbs = [pdb_directory / f"{stem}.pdb" for stem in args.pdbs]
else:
    residue_pdbs = sorted(pdb_directory.glob("*.pdb"))

ic_directory = Path(__file__).resolve().parent

for pdb in residue_pdbs:
    new_path = ic_directory / f"{pdb.stem}_ic.pkl"
    if new_path.exists() and not args.force:
        continue

    print(pdb.stem)

    srtd = chilife.sort_pdb(pdb)
    with open(pdb, "w") as f:
        f.writelines(srtd)

    struct = mda.Universe(str(pdb), in_memory=True)
    resname = pdb.stem.upper()
    pref_d = chilife.dihedral_defs[resname]
    print(pref_d)
    ICs = chilife.MolSysIC.from_atoms(struct, preferred_dihedrals=pref_d)
    with open(new_path, "wb") as f:
        pickle.dump(ICs, f)