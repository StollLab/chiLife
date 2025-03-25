import os
import hashlib
from pathlib import Path

import MDAnalysis as mda
import numpy as np
import pytest

import chilife as xl

U = xl.load_protein("test_data/m1omp.pdb")
protein = U.select_atoms("protein")

load_protein_args = [('test_data/1ubq.pdb',),
                     (Path('test_data/1ubq.pdb'),),
                     ('test_data/traj_io.pdb', Path('test_data/traj_io.xtc')),]


@pytest.mark.parametrize('args', load_protein_args)
def test_load_protein(args):

    struct = xl.load_protein(*args)
    assert isinstance(struct, mda.Universe)

    
    if len(args) == 2:
        if (p1 := Path('test_data/.traj_io.xtc_offsets.lock')).exists():
            os.remove('test_data/.traj_io.xtc_offsets.lock')
        if (p2 := Path('test_data/.traj_io.xtc_offsets.npz')).exists():
            os.remove('test_data/.traj_io.xtc_offsets.npz')


def test_save():
    L20R1 = xl.SpinLabel("R1C", 20, protein)
    S238T = xl.RotamerEnsemble("THR", 238, protein, eval_clash=False)
    A318DHC = xl.dSpinLabel("DHC", [318, 322], protein, rotlib='test_data/DHC')

    xl.save(L20R1, S238T, A318DHC, protein, KDE=False)

    with open(f"test_data/test_save.pdb", "r") as f:
        ans = hashlib.md5(f.read().encode("utf-8")).hexdigest()

    with open("No_Name_Protein_many_labels.pdb", "r") as f:
        test = hashlib.md5(f.read().encode("utf-8")).hexdigest()

    os.remove("No_Name_Protein_many_labels.pdb")

    assert ans == test


def test_save_frame():
    U = mda.Universe('test_data/traj_io.pdb', 'test_data/traj_io.xtc')
    xl.save('tmp.pdb', U, frames=5)

    with open(f"test_data/save_frame.pdb", "r") as f:
        ans = hashlib.md5(f.read().encode("utf-8")).hexdigest()

    with open("tmp.pdb", "r") as f:
        test = hashlib.md5(f.read().encode("utf-8")).hexdigest()

    os.remove('tmp.pdb')

    assert ans == test


def test_save_ic_frame():
    U = mda.Universe('test_data/xlsavetraj.pdb')
    sele_IC = xl.MolSysIC.from_atoms(U.atoms)
    xl.save('ic_frames.pdb', sele_IC)

    with open(f"test_data/ic_frames.pdb", "r") as f:
        ans = hashlib.md5(f.read().encode("utf-8")).hexdigest()

    with open("ic_frames.pdb", "r") as f:
        test = hashlib.md5(f.read().encode("utf-8")).hexdigest()

    os.remove('ic_frames.pdb')
    assert ans == test


def test_save_fail():
    with pytest.raises(TypeError):
        xl.save("tmp", np.array([1, 2, 3]))


def test_save_multiple_groups():
    CAs = U.select_atoms('name CA')
    CBs = U.select_atoms('name CB')

    xl.save(U.atoms, CAs, CBs)

    names = ['m1omp', 'm1omp1', 'm1omp2']

    with open(f"No_Name_Protein.pdb", "r") as f:
        for line in f:
            for name in iter(names):
                sen = f"HEADER {name}\n"
                if line == sen:
                    names.remove(name)
                if len(names) ==0:
                    break
            if len(names) == 0:
                break

    assert len(names) == 0

    os.remove('No_Name_Protein.pdb')


def test_write_bonds():

    bonds = [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [1, 4], [1, 5], [1, 6], [2, 3], [2, 4], [2, 5], [2, 6]]
    with open('test_write_bonds.pdb', 'w') as f:
        xl.write_bonds(f, bonds)

    with open('test_write_bonds.pdb', 'r') as f:
        lines = "".join(f.readlines())
        thash = hashlib.md5(lines.encode("utf-8")).hexdigest()

    with open('test_data/test_write_bonds.pdb', 'r') as f:
        lines = "".join(f.readlines())
        ahash = hashlib.md5(lines.encode("utf-8")).hexdigest()

    os.remove('test_write_bonds.pdb')
    assert ahash == thash


def test_write_protein_with_bonds():
    prot1 = xl.load_protein('test_data/alphabetical_peptide.pdb')
    R1M = xl.SpinLabel('R1M')
    prot1.add_bonds(xl.guess_bonds(prot1.atoms.positions, prot1.atoms.types))

    xl.save(R1M, conect=True)


    with open('1R1M.pdb', 'r') as f:
        lines = "".join([line for line in f if line.startswith("CONECT")])
        thash = hashlib.md5(lines.encode("utf-8")).hexdigest()

    with open('test_data/cnct.pdb', 'r') as f:
        lines = "".join([line for line in f if line.startswith("CONECT")])
        ahash = hashlib.md5(lines.encode("utf-8")).hexdigest()

    assert thash == ahash

    os.remove('1R1M.pdb')