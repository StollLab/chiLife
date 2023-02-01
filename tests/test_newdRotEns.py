
import chilife as xl

gb1 = xl.fetch('2qmt')
def test_something():
    resi = 2
    label = 'DHC'
    inc = 2
    pdbname = f'{label}_ip{inc}_ens2.pdb'
    spin_atoms = 'Cu1'
    dihedral_atoms = [[["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "ND1"], ['CD2', 'NE2', 'Cu1', 'O1']],
                      [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "ND1"], ['CD2', 'NE2', 'Cu1', 'O1']]]

    xl.create_dlibrary(label, pdbname,
                       sites=(resi, resi + inc),
                       dihedral_atoms=dihedral_atoms,
                       spin_atoms=spin_atoms)

    SL = xl.dSpinLabel(label, [6, 8], gb1)
    SL.spin_coords
    xl.save(*SL.sub_labels, gb1)
    xl.save(SL, gb1)

def test_otherthing():
    spin_atoms = ['NN', 'ON']
    dihedral_atoms = [
        [["N", "CA", "CB", "SG"], ["CA", "CB", "SG", "S1L"], ["CB", "SG", "S1L", "C1L"], ["SG", "S1L", "C1L", "C1R"],
         ["S1L", "C1L", "C1R", "C1"]],
        [["N", "CA", "CB", "SG"], ["CA", "CB", "SG", "SG1"], ["CB", "SG", "SG1", "C1M"], ["SG", "SG1", "C1M", "C2R"],
         ["SG1", "C1M", "C2R", "C2"]]]

    label = 'RXA'
    inc = 2
    resi = 2
    pdbname = f'{label}_ip{inc}_ens.pdb'
    xl.create_dlibrary(label, pdbname, sites=(resi, resi + inc), dihedral_atoms=dihedral_atoms,
                       spin_atoms=spin_atoms)

    protein = xl.fetch('2zd7')
    RX1 = xl.dSpinLabel('RXA', (122, 124), protein, chain='A')
    RX2 = xl.dSpinLabel('RXA', (122, 124), protein, chain='B')
    xl.save(protein, RX1, RX2)


def test_rx():
    label = 'RXA'
    inc = 2
    resi = 2
    pdbname = f'{label}_ip{inc}_ens.pdb'
    spin_atoms = ['NN', 'ON']
    dihedral_atoms = [
        [["N", "CA", "CB", "SG"], ["CA", "CB", "SG", "S1L"], ["CB", "SG", "S1L", "C1L"], ["SG", "S1L", "C1L", "C1R"],
         ["S1L", "C1L", "C1R", "C1"]],
        [["N", "CA", "CB", "SG"], ["CA", "CB", "SG", "SG1"], ["CB", "SG", "SG1", "C1M"], ["SG", "SG1", "C1M", "C2R"],
         ["SG1", "C1M", "C2R", "C2"]]]

    xl.create_dlibrary(label, pdbname, sites=(resi, resi + inc), dihedral_atoms=dihedral_atoms, spin_atoms=spin_atoms)

    protein = xl.fetch('2zd7')
    RX1 = xl.dSpinLabel('RXA', (122, 124), protein, chain='A')
    RX2 = xl.dSpinLabel('RXA', (122, 124), protein, chain='B')