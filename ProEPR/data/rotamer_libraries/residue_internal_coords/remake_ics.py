import MDAnalysis as mda
from pathlib import Path
import ProEPR
import pickle

pdb_directory = Path('../residue_pdbs')
residue_pdbs = pdb_directory.glob('*.pdb')

for pdb in residue_pdbs:
    print(pdb.stem)
    if pdb.stem in ['ala', 'gly']:
        continue

    struct = mda.Universe(str(pdb))
    ICs = ProEPR.get_internal_coords(struct)
    with open(pdb.stem + '_ic.pkl', 'wb') as f:
        pickle.dump(ICs, f)