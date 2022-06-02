import MDAnalysis as mda
from pathlib import Path
import chiLife
import pickle

pdb_directory = Path('../residue_pdbs')
residue_pdbs = pdb_directory.glob('*.pdb')

for pdb in residue_pdbs:
    print(pdb.stem)

    struct = mda.Universe(str(pdb))
    ICs = chiLife.get_internal_coords(struct)
    with open(pdb.stem + '_ic.pkl', 'wb') as f:
        pickle.dump(ICs, f)