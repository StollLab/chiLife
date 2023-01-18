
import chilife as xl

gb1 = xl.fetch('2qmt')


def test_something():
    resi = 2
    label = 'DHC'
    inc = 2
    pdbname = f'{label}_ip{inc}_ens2.pdb'
    spin_atoms = 'Cu1'
    dihedral_atoms = [[["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "ND1"]],
                      [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "ND1"]]]

    xl.create_dlibrary(label, pdbname, sites=(resi, resi + inc, resi + inc + 2), dihedral_atoms=dihedral_atoms,
                       spin_atoms=spin_atoms)


    SL = xl.dSpinLabel(label, [6, 8], gb1)

    xl.save(SL, gb1)
