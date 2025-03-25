import MDAnalysis as mda
from pathlib import Path
import chilife
import pickle

pdb_directory = Path('../residue_pdbs')
residue_pdbs = pdb_directory.glob('*.pdb')

for pdb in residue_pdbs:

    new_path = Path(pdb.stem + '_ic.pkl')
    if new_path.exists():
        continue

    print(pdb.stem)

    srtd = chilife.sort_pdb(pdb)
    with open(pdb, 'w') as f:
        f.writelines(srtd)

    struct = mda.Universe(str(pdb), in_memory=True)
    pref_d = chilife.dihedral_defs[pdb.stem.upper()]
    print(pref_d)
    ICs = chilife.MolSysIC.from_atoms(struct, preferred_dihedrals=pref_d)
    with open(pdb.stem + '_ic.pkl', 'wb') as f:
        pickle.dump(ICs, f)