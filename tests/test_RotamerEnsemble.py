import hashlib, os, pickle
from functools import partial
import numpy as np
import chilife
import pytest
import MDAnalysis as mda

ubq = chilife.fetch("1ubq")
U = chilife.fetch("1omp")
exou = chilife.fetch("3tu3")
traj = mda.Universe('test_data/xlsavetraj.pdb', in_memory=True)


def test_from_mda():
    res16 = U.residues[16]
    ensemble = chilife.RotamerEnsemble.from_mda(res16)
    ensemble.save_pdb("test_data/test_from_MDA.pdb")

    with open("test_data/ans_from_MDA.pdb", "r") as f:
        ans = hashlib.md5(f.read().encode("utf-8")).hexdigest()

    with open("test_data/test_from_MDA.pdb", "r") as f:
        test = hashlib.md5(f.read().encode("utf-8")).hexdigest()

    os.remove("test_data/test_from_MDA.pdb")
    assert ans == test


def test_with_sample():
    np.random.seed(200)

    SL = chilife.SpinLabel("R1C", 28, ubq, sample=2000)

    with np.load('test_data/withsample.npz') as f:
        ans = {key: f[key] for key in f}

    np.testing.assert_almost_equal(SL._coords, ans['coords'])
    np.testing.assert_almost_equal(SL.weights, ans['weights'])
    np.testing.assert_almost_equal(SL.dihedrals, ans['dihedrals'])

    assert len(SL._coords) == len(SL.internal_coords)


def test_user_label():
    SL = chilife.SpinLabel("TRT", 28, ubq, "A")
    chilife.save(
        "test_data/1ubq_28TRT_tmp.pdb", SL, protein_path="test_data/1ubq.pdb", KDE=False
    )

    with open("test_data/1ubq_28TRT.pdb", "r") as f:
        nohstring1 = "".join((line for line in f.readlines() if line[76:79].strip() != 'H'))
        ans = hashlib.md5(nohstring1.encode('utf-8')).hexdigest()

    with open("test_data/1ubq_28TRT_tmp.pdb", "r") as f:
        nohstring2 = "".join((line for line in f.readlines() if line[76:79].strip() != 'H'))
        test = hashlib.md5(nohstring2.encode('utf-8')).hexdigest()

    os.remove("test_data/1ubq_28TRT_tmp.pdb")

    assert test == ans


def test_save_pkl():
    Lys = chilife.RotamerEnsemble("LYS")

    with open("test_data/Lys.pkl", "wb") as f:
        pickle.dump(Lys, f)

    with open("test_data/Lys.pkl", "rb") as f:
        reload = pickle.load(f)

    os.remove("test_data/Lys.pkl")

    assert Lys == reload


def test_save_pkl_2():
    res16 = U.residues[16]
    ensemble = chilife.RotamerEnsemble.from_mda(res16)
    with open("test_data/res16.pkl", "wb") as f:
        pickle.dump(ensemble, f)

    with open("test_data/res16.pkl", "rb") as f:
        reload = pickle.load(f)

    assert ensemble == reload
    os.remove("test_data/res16.pkl")


def test_sample():
    np.random.seed(200)
    ubq = chilife.fetch("1ubq")
    K48 = chilife.RotamerEnsemble.from_mda(ubq.residues[47])
    coords, weight = K48.sample(off_rotamer=True)

    wans = 0.0037019925960148086
    cans = np.array([[20.76406623, 27.90259642, 22.59445098],
                     [21.54999924, 26.79599953, 23.13299942],
                     [23.02842999, 27.03777002, 22.93937713],
                     [23.39710135, 27.56595972, 21.90105423],
                     [21.1101072, 25.47090301, 22.45317533],
                     [21.19895831, 25.51252905, 20.90717057],
                     [21.03430238, 24.15607743, 20.2156025],
                     [22.11572213, 23.19412351, 20.72492909],
                     [22.16827211, 22.00913727, 19.85099131]])

    np.testing.assert_almost_equal(coords, cans)
    assert weight == wans


def test_multisample():
    ubq = chilife.fetch("1ubq")
    R1M = chilife.SpinLabel.from_wizard("R1M", site=48, protein=ubq, to_find=10000)


@pytest.mark.parametrize("res", chilife.SUPPORTED_RESIDUES)
def test_lib_distribution_persists(res):
    if res in chilife.USER_dLIBRARIES:
        return None
    elif res in chilife.USER_LIBRARIES:
        L1 = chilife.SpinLabel(res)
        L2 = chilife.SpinLabel(res, sample=100)
    else:
        L1 = chilife.RotamerEnsemble(res)
        L2 = chilife.RotamerEnsemble(res, sample=100)

    np.testing.assert_almost_equal(L1._rdihedrals, L2._rdihedrals)
    np.testing.assert_almost_equal(L1._rsigmas, L2._rsigmas)
    np.testing.assert_almost_equal(L1._weights, L2._weights)


methods = ["rosetta", "bisect", "mmm", "fit"]


@pytest.mark.parametrize(("method"), methods)
def test_superimposition_method(method):
    if method == "fit":
        with pytest.raises(NotImplementedError) as e_info:
            SL = chilife.SpinLabel(
                "R1C", site=28, protein=ubq, superimposition_method=method, eval_clash=False
            )
            chilife.save(SL, ubq)
    else:
        SL = chilife.SpinLabel(
            "R1C",
            site=28,
            protein=ubq,
            superimposition_method=method,
            energy_func=partial(chilife.get_lj_rep, forgive=0.8),
        )
        chilife.save(
            f"A28R1_{method}_superimposition.pdb", SL, "test_data/1ubq.pdb", KDE=False
        )

        with open(f"A28R1_{method}_superimposition.pdb", "r") as f:
            test = hashlib.md5(f.read().encode('utf-8')).hexdigest()

        with open(f"test_data/A28R1_{method}_superimposition.pdb", "r") as f:
            ans = hashlib.md5(f.read().encode('utf-8')).hexdigest()

        os.remove(f"A28R1_{method}_superimposition.pdb")
        assert ans == test


def test_catch_unused_kwargs():
    with pytest.raises(TypeError) as e_info:
        SL = chilife.SpinLabel("R1C", site=28, protein=ubq, supermposition_method="mmm")
    assert (
            str(e_info.value) == "Got unexpected keyword argument(s): supermposition_method"
    )


def test_guess_chain():
    anf = chilife.fetch("1anf")
    SL = chilife.SpinLabel.from_mmm("R1M", 20, forgive=0.9)


@pytest.mark.parametrize(("resi", "ans"), ((20, "A"), (206, "B")))
def test_guess_chain2(resi, ans):
    SL = chilife.SpinLabel("R1C", resi, exou)
    assert SL.chain == ans


@pytest.mark.parametrize("resi", (100, 344))
def test_guess_chain_fail(resi):
    with pytest.raises(ValueError) as e_info:
        SL = chilife.SpinLabel("R1C", resi, exou)


def test_mem_sample():
    SL1 = chilife.SpinLabel('R1M', 28, ubq, sample=10000)
    print('booger')


def test_label_as_library():
    R1C = chilife.RotamerEnsemble("R1C", site=28, protein=ubq)
    for ic in R1C.internal_coords:
        np.testing.assert_almost_equal(
            ic.coords[2], [38.73227962, 26.58109478, 12.6243569]
        )


def test_coord_setter0():
    R1C1 = chilife.RotamerEnsemble("R1C", site=28, protein=ubq)


def test_coord_setter():
    R1C1 = chilife.RotamerEnsemble("R1C", site=28, protein=ubq)
    R1C2 = chilife.RotamerEnsemble("R1C", site=29, protein=ubq)

    R1C1.coords = R1C2.coords

    np.testing.assert_allclose(R1C1.coords, R1C2.coords)
    np.testing.assert_allclose(R1C1.dihedrals, R1C2.dihedrals, rtol=1e-6)

    for key in 'ori', 'mx':
        np.testing.assert_allclose(R1C1.internal_coords[0].chain_operators[1][key],
                                   R1C2.internal_coords[0].chain_operators[1][key])


def test_coord_setter2():
    R1C1 = chilife.RotamerEnsemble("R1C", site=28, protein=ubq, sample=100)
    R1C2 = chilife.RotamerEnsemble("R1C", site=28, protein=ubq, sample=100)
    R1C1.coords = R1C2.coords[:, R1C2.side_chain_idx]

    np.testing.assert_allclose(R1C1.coords, R1C2.coords)
    np.testing.assert_allclose(R1C1.dihedrals, R1C2.dihedrals)


def test_dihedral_setter():
    R1C1 = chilife.RotamerEnsemble("R1C", site=28, protein=ubq, sample=100)
    R1C2 = chilife.RotamerEnsemble("R1C", site=28, protein=ubq, sample=100)
    R1C1.dihedrals = R1C2.dihedrals

    np.testing.assert_allclose(R1C1.coords, R1C2.coords)
    np.testing.assert_allclose(R1C1.dihedrals, R1C2.dihedrals)


def test_get_sasa():
    R1C = chilife.RotamerEnsemble("R1C")
    sasas = R1C.get_sasa()
    sasans = np.load('test_data/sasas.npy')
    np.testing.assert_allclose(sasas, sasans)


def test_default_dihedral_sigmas():
    rotlib = chilife.read_bbdep('R1C', -60, -50)
    SL = chilife.SpinLabel('R1C')
    np.testing.assert_allclose(rotlib['sigmas'][rotlib['sigmas'] != 0], SL.sigmas[SL.sigmas != 35.])


def test_construct_with_dihedral_sigmas():
    SL = chilife.SpinLabel('R1C', dihedral_sigmas=25)
    assert np.all(SL.sigmas == 25.)
    assert SL.sigmas.shape == (len(SL), len(SL.dihedral_atoms))


def test_construct_with_array_of_dihedral_sigmas():
    set_sigmas = [10, 20, 30, 40, 50]
    SL = chilife.SpinLabel('R1C', dihedral_sigmas=set_sigmas)
    for i in range(5):
        assert np.all(SL.sigmas[:, i] == set_sigmas[i])

    assert SL.sigmas.shape == (len(SL), len(SL.dihedral_atoms))


def test_construct_with_full_array_of_dihedral_sigmas():
    set_sigmas = np.random.rand(148, 5)
    SL = chilife.SpinLabel('R1C', dihedral_sigmas=set_sigmas)
    np.testing.assert_allclose(set_sigmas, SL.sigmas)


def test_dihedral_sigmas_fail():
    with pytest.raises(ValueError):
        SL = chilife.SpinLabel('R1C', dihedral_sigmas=[5, 2, 1])


def test_set_dihedral_sigmas():
    SL = chilife.SpinLabel('R1A')
    assert np.all(SL.sigmas == 35.)
    SL.set_dihedral_sampling_sigmas(25)
    assert np.all(SL.sigmas == 25.)


def test_sample_dihedral3():
    SL = chilife.SpinLabel('R1A', 27, ubq, sample=1000)


def test_clash_radius():
    RL = chilife.SpinLabel('TRT')
    assert RL.clash_radius == np.linalg.norm(RL.coords - RL.clash_ori, axis=-1).max() + 5


def test_nataa():
    SL = chilife.SpinLabel('R1A', 28, ubq)
    assert SL.nataa == 'A'


def test_from_trajectory():
    RE1 = chilife.RotamerEnsemble.from_trajectory(traj, 232, burn_in=0)
    RE2 = chilife.RotamerEnsemble.from_trajectory(traj, 231, burn_in=0)
    RE3 = chilife.RotamerEnsemble.from_trajectory(traj, 230, burn_in=0)

    assert len(RE1) == 1
    assert len(RE2) == 1
    assert len(RE3) == 10

    assert RE1.res == 'TRP'
    assert RE2.res == 'ALA'
    assert RE3.res == 'TRP'

    np.testing.assert_almost_equal(RE1.dihedrals, [[-72.49934635, 164.56177856]])
    np.testing.assert_almost_equal(RE2.dihedrals, [[]])
    np.testing.assert_almost_equal(RE3.dihedrals, [[-176.38805017,  -20.15226419],
                                                   [-176.38805017,  -52.83812431],
                                                   [-176.38805017, -111.39416684],
                                                   [-176.38805017,   73.81697923],
                                                   [-176.38805017, -134.54899132],
                                                   [-176.38805017,  118.2802729 ],
                                                   [-176.38805017,  164.59451953],
                                                   [  62.16344279,  -93.79344738],
                                                   [ -69.90840475,  -39.25944367],
                                                   [ -69.90840475,  161.67407146]])

    with pytest.raises(ValueError):
        RE = chilife.RotamerEnsemble.from_trajectory(traj, 232)


def test_to_rotlib():

    RE = chilife.RotamerEnsemble.from_trajectory(traj, 230, burn_in=0)
    RE.to_rotlib('Test')
    RE2 = chilife.RotamerEnsemble('TRP', rotlib='Test')
    assert RE2.res == 'TRP'
    assert len(RE2) == 10

    np.testing.assert_almost_equal(RE2.dihedrals, [[-176.38805017,  -20.15226419],
                                                   [-176.38805017,  -52.83812431],
                                                   [-176.38805017, -111.39416684],
                                                   [-176.38805017,   73.81697923],
                                                   [-176.38805017, -134.54899132],
                                                   [-176.38805017,  118.2802729 ],
                                                   [-176.38805017,  164.59451953],
                                                   [  62.16344279,  -93.79344738],
                                                   [ -69.90840475,  -39.25944367],
                                                   [ -69.90840475,  161.67407146]])

    with open(f"test_data/Test_rotlib.npz", "rb") as f:
        ans = hashlib.md5(f.read()).hexdigest()

    with open(f"Test_rotlib.npz", "rb") as f:
        test = hashlib.md5(f.read()).hexdigest()

    os.remove('Test_rotlib.npz')

    assert ans == test

