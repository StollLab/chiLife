import os
import hashlib
from pathlib import Path

import MDAnalysis
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
    assert isinstance(struct, MDAnalysis.Universe)

    
    if len(args) == 2:
        os.remove('test_data/.traj_io.xtc_offsets.lock')
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


def test_save_fail():
    with pytest.raises(TypeError):
        xl.save("tmp", np.array([1, 2, 3]))
