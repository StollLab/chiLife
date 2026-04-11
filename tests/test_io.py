import os
import hashlib
from pathlib import Path

import MDAnalysis as mda
import numpy as np
import pytest
import rtoml

import chilife as xl

U = xl.load_protein("test_data/m1omp.pdb")
protein = U.select_atoms("protein")

load_protein_args = [
    ("test_data/1ubq.pdb",),
    (Path("test_data/1ubq.pdb"),),
    ("test_data/traj_io.pdb", Path("test_data/traj_io.xtc")),
]


def test_fetch_PDB():
    U1 = mda.Universe("test_data/m1omp.pdb", in_memory=True)
    U2 = xl.fetch("1omp")

    assert np.all(U1.atoms.positions == U2.atoms.positions)


def test_fetch_CIF():
    U1 = mda.Universe("test_data/m1omp.pdb", in_memory=True)
    U2 = xl.fetch("1omp.cif")

    np.testing.assert_almost_equal(U1.atoms.positions, U2.atoms.positions, decimal=5)


def test_fetch_AF():
    U = xl.fetch("AF-O34208")
    ans = np.array(
        [
            [-38.834, -52.705, 45.698],
            [-38.5, -54.236, 47.741],
            [-38.903, -51.597, 46.216],
            [-38.123, -55.685, 48.115],
            [-38.65, -56.141, 49.474],
            [-38.411, -57.378, 49.85],
            [-39.277, -55.422, 50.23],
            [-39.532, -53.088, 44.618],
            [-40.791, -52.617, 43.986],
            [-42.008, -52.46, 44.944],
        ]
    )
    np.testing.assert_allclose(U.atoms.positions[100:110], ans)


def test_fetch_AF_CIF():
    U = xl.fetch("AF-O34208", format="cif")
    ans = np.array(
        [
            [-38.834, -52.705, 45.698],
            [-38.5, -54.236, 47.741],
            [-38.903, -51.597, 46.216],
            [-38.123, -55.685, 48.115],
            [-38.65, -56.141, 49.474],
            [-38.411, -57.378, 49.85],
            [-39.277, -55.422, 50.23],
            [-39.532, -53.088, 44.618],
            [-40.791, -52.617, 43.986],
            [-42.008, -52.46, 44.944],
        ]
    )
    np.testing.assert_allclose(U.atoms.positions[100:110], ans)


IDs = ["1anf", "1omp.pdb", "3tu3"]
fNames = ["1anf.pdb", "1omp.pdb", "3tu3.pdb"]


@pytest.mark.parametrize("pdbid, names", zip(IDs, fNames))
def test_fetch2(pdbid, names):
    xl.fetch(pdbid, save=True)
    with open(f"test_data/m{names}", "r") as f:
        ans = hashlib.md5(f.read().encode("utf-8")).hexdigest()
    with open(f"{names}", "r") as f:
        test = hashlib.md5(f.read().encode("utf-8")).hexdigest()
    assert ans == test
    os.remove(names)


@pytest.mark.parametrize("args", load_protein_args)
def test_load_protein(args):
    struct = xl.load_protein(*args)
    assert isinstance(struct, mda.Universe)

    if len(args) == 2:
        if (p1 := Path("test_data/.traj_io.xtc_offsets.lock")).exists():
            os.remove("test_data/.traj_io.xtc_offsets.lock")
        if (p2 := Path("test_data/.traj_io.xtc_offsets.npz")).exists():
            os.remove("test_data/.traj_io.xtc_offsets.npz")


def test_save():
    L20R1 = xl.SpinLabel("R1C", 20, protein)
    S238T = xl.RotamerEnsemble("THR", 238, protein, eval_clash=False)
    A318DHC = xl.dSpinLabel("DHC", [318, 322], protein, rotlib="test_data/DHC")

    xl.save(L20R1, S238T, A318DHC, protein, KDE=False)

    with open("test_data/test_save.pdb", "r") as f:
        ans = hashlib.md5(f.read().encode("utf-8")).hexdigest()

    with open("No_Name_Protein_many_labels.pdb", "r") as f:
        test = hashlib.md5(f.read().encode("utf-8")).hexdigest()

    os.remove("No_Name_Protein_many_labels.pdb")

    assert ans == test


def test_save_frame():
    U = mda.Universe("test_data/traj_io.pdb", "test_data/traj_io.xtc")
    xl.save("tmp.pdb", U, frames=5)

    with open("test_data/save_frame.pdb", "r") as f:
        ans = hashlib.md5(f.read().encode("utf-8")).hexdigest()

    with open("tmp.pdb", "r") as f:
        test = hashlib.md5(f.read().encode("utf-8")).hexdigest()

    os.remove("tmp.pdb")

    assert ans == test


def test_save_ic_frame():
    U = mda.Universe("test_data/xlsavetraj.pdb")
    sele_IC = xl.MolSysIC.from_atoms(U.atoms)
    xl.save("ic_frames.pdb", sele_IC)

    with open("test_data/ic_frames.pdb", "r") as f:
        ans = hashlib.md5(f.read().encode("utf-8")).hexdigest()

    with open("ic_frames.pdb", "r") as f:
        test = hashlib.md5(f.read().encode("utf-8")).hexdigest()

    os.remove("ic_frames.pdb")
    assert ans == test


def test_save_fail():
    with pytest.raises(TypeError):
        xl.save("tmp", np.array([1, 2, 3]))


def test_save_multiple_groups():
    CAs = U.select_atoms("name CA")
    CBs = U.select_atoms("name CB")

    xl.save(U.atoms, CAs, CBs)

    names = ["m1omp", "m1omp1", "m1omp2"]

    with open("No_Name_Protein.pdb", "r") as f:
        for line in f:
            for name in iter(names):
                sen = f"HEADER {name}\n"
                if line == sen:
                    names.remove(name)
                if len(names) == 0:
                    break
            if len(names) == 0:
                break

    assert len(names) == 0

    os.remove("No_Name_Protein.pdb")


def test_write_bonds():
    bonds = [
        [0, 1],
        [0, 2],
        [0, 3],
        [1, 2],
        [1, 3],
        [1, 4],
        [1, 5],
        [1, 6],
        [2, 3],
        [2, 4],
        [2, 5],
        [2, 6],
    ]
    with open("test_write_bonds.pdb", "w") as f:
        xl.write_bonds(f, bonds)

    with open("test_write_bonds.pdb", "r") as f:
        lines = "".join(f.readlines())
        thash = hashlib.md5(lines.encode("utf-8")).hexdigest()

    with open("test_data/test_write_bonds.pdb", "r") as f:
        lines = "".join(f.readlines())
        ahash = hashlib.md5(lines.encode("utf-8")).hexdigest()

    os.remove("test_write_bonds.pdb")
    assert ahash == thash


def test_write_protein_with_bonds():
    prot1 = xl.load_protein("test_data/alphabetical_peptide.pdb")
    R1M = xl.SpinLabel("R1M")
    prot1.add_bonds(xl.guess_bonds(prot1.atoms.positions, prot1.atoms.types))

    xl.save(R1M, conect=True)

    with open("1R1M.pdb", "r") as f:
        lines = "".join([line for line in f if line.startswith("CONECT")])
        thash = hashlib.md5(lines.encode("utf-8")).hexdigest()

    with open("test_data/cnct.pdb", "r") as f:
        lines = "".join([line for line in f if line.startswith("CONECT")])
        ahash = hashlib.md5(lines.encode("utf-8")).hexdigest()

    assert thash == ahash

    os.remove("1R1M.pdb")


def test_read_pdb():
    pdb_data = xl.read_pdb("test_data/7o1o.pdb")

    np.testing.assert_almost_equal(
        pdb_data["trajectory"][0][145], [20.626, 6.755, -19.788]
    )
    assert np.all(pdb_data["bond_types"] == xl.BondType.UNSPECIFIED)
    assert len(pdb_data["bonds"]) == len(pdb_data["bond_types"]) == 114


def test_read_sdf():
    file_name = "test_data/test_la_subject.sdf"
    data = xl.read_sdf(file_name)
    assert len(data) == 100
    assert data[0]["OTHER EXTRA DATA"] == "Here is some more extra data"

    coords = np.array([[atom["xyz"] for atom in mol["atoms"]] for mol in data])
    ans = np.load("test_data/load_sdf.npy")

    np.testing.assert_almost_equal(coords, ans)


def test_write_sdf():
    file_name = "test_data/test_la_subject.sdf"
    data = xl.read_sdf(file_name)
    xl.write_sdf(data, "temp.sdf")

    with open(file_name, "r") as f:
        ans = "".join([line for line in f.readlines() if "RDKit" not in line])
        ans = hashlib.md5(ans.encode("utf8")).hexdigest()

    with open("temp.sdf", "r") as f:
        test = "".join([line for line in f.readlines() if "chiLife" not in line])
        test = hashlib.md5(test.encode("utf8")).hexdigest()

    os.remove("temp.sdf")

    assert ans == test


def test_read_cif():
    file_name = "test_data/7o1o.cif"
    data = xl.read_cif(file_name)

    with open("test_data/test_read_cif.toml", "r") as f:
        lines = "".join(f.readlines())
        ans = hashlib.md5(lines.encode("utf8")).hexdigest()

    with open("temp.toml", "w") as f:
        rtoml.dump(data, f)

    with open("temp.toml", "r") as f:
        test = "".join(f.readlines())
        test = hashlib.md5(test.encode("utf8")).hexdigest()

    assert ans == test

    os.remove("temp.toml")


def test_write_cif():
    file_name = "test_data/7o1o.cif"
    cif_data = xl.read_cif(file_name)
    xl.write_cif("temp.cif", cif_data)

    with open("test_data/test_write_cif.cif", "r") as f:
        lines = "".join(f.readlines())
        ans = hashlib.md5(lines.encode("utf8")).hexdigest()

    with open("temp.cif", "r") as f:
        test = "".join(f.readlines())
        test = hashlib.md5(test.encode("utf8")).hexdigest()

    assert test == ans
    os.remove("temp.cif")


def test_split_with_quotes():
    string = "2 2 'Structure model' 'Database references' "
    ans_entries = ["2", "2", "Structure model", "Database references"]
    entries = xl.split_with_quotes(string)
    for test, ans in zip(entries, ans_entries):
        assert test == ans


def test_split_with_quotes2():
    line = '_chem_comp.name                                 " 1-cyclopentyl-6-[[(2R)-1-(6-fluoranyl-2-azaspiro[3.3]heptan-2-yl)-1-oxidanylidene-propan-2-yl]amino]-5H-pyrazolo[3,4-d]pyrimidin-4-one"\n'
    data = xl.split_with_quotes(line)
    assert len(data) == 2


def test_join_ccd_info():
    cif_data = xl.read_cif("test_data/7o1o.cif")["7O1O"]
    ans_ids = sorted(set(cif_data["chem_comp_atom"]["comp_id"]))

    out = xl.join_ccd_info(ans_ids)

    chem_comp = out["chem_comp"]
    for key in ["id", "name", "formula_weight"]:
        for val in chem_comp[key]:
            assert val in cif_data["chem_comp"][key]

    atom = out["chem_comp_atom"]
    bond = out["chem_comp_bond"]

    with open("test_data/join_ccd_info.toml", "r") as f:
        data = rtoml.load(f)

    for k, v in data["atom"].items():
        assert all(vi == ti for vi, ti in zip(v, atom[k]))

    for k, v in data["bond"].items():
        assert all(vi == ti for vi, ti in zip(v, bond[k]))
