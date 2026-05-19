import numpy as np
import pytest
import MDAnalysis as mda
import chilife


r = np.linspace(1e-3, 3, 256)
eps = np.ones(len(r))
rmin = np.ones(len(r))
protein = mda.Universe("test_data/1ubq.pdb", in_memory=True)
xl_protein = chilife.MolSys.from_pdb("test_data/1ubq.pdb")
lj_funcs = [chilife.get_lj_energy, chilife.get_lj_scwrl, chilife.get_lj_rep]
lj_ans = [
    np.array(
        [
            -1.17807772,
            -1.36824056,
            -1.45138763,
            -1.15117638,
            -1.34821558,
            -1.07243325,
            -0.99806148,
            -0.79547696,
            -0.41502591,
            -0.39899395,
        ]
    ),
    np.array(
        [
            0.01217681,
            -0.14832397,
            -0.09426473,
            -0.10662389,
            0.0,
            -0.21180769,
            0.0,
            -0.00933278,
            0.0,
            0.0,
        ]
    ),
    np.array(
        [
            0.72906185,
            0.07046668,
            0.11446102,
            0.00656634,
            0.0201529,
            0.00476935,
            0.00413794,
            0.00179881,
            0.00138244,
            0.00089236,
        ]
    ),
]
lj_rot_idx = [0, 5, 5]


@pytest.mark.parametrize(("func", "ans", "rot_idx"), zip(lj_funcs, lj_ans, lj_rot_idx))
def test_lj(func, ans, rot_idx):
    f = chilife.ljEnergyFunc(func)
    RL = chilife.RotamerEnsemble("TRP", 28, protein, energy_func=f, eval_clash=True)
    np.testing.assert_almost_equal(RL.atom_energies[rot_idx], ans, decimal=6)


@pytest.mark.parametrize("func", lj_funcs)
def test_efunc(func):
    RL = chilife.RotamerEnsemble("TRP", 28, protein, eval_clash=False)
    f = chilife.ljEnergyFunc(func)
    test = f(RL)
    ans = np.load(f"test_data/{func.__name__}.npy")
    np.testing.assert_almost_equal(test, ans, decimal=5)


@pytest.mark.parametrize("func", lj_funcs)
def test_efunc_dlabel(func):
    dSL = chilife.dSpinLabel(
        "DHC", (28, 32), protein, eval_clash=False, rotlib="test_data/DHC"
    )
    f = chilife.ljEnergyFunc(func)
    test = f(dSL)
    ans = np.load(f"test_data/d{func.__name__}.npy")
    np.testing.assert_almost_equal(test, ans, decimal=3)


def test_molar_gas_constant():
    np.testing.assert_almost_equal(
        chilife.scoring.GAS_CONST, 1.987204258640832e-3, decimal=10
    )


def test_get_lj_case_sensitivity():
    ff = chilife.ljEnergyFunc()
    x = ff.get_lj_rmin(["CA", "Ca", "ca"])
    assert np.all(x == 1.367)


@pytest.mark.parametrize("prot", [protein, xl_protein])
def test_score_protein(prot):
    bonds = chilife.guess_bonds(prot.atoms.positions, prot.atoms.types)
    prot.add_bonds(bonds)

    if isinstance(prot, mda.Universe):
        prot = prot.atoms

    ff = chilife.ljEnergyFunc()
    ff.prepare_system(prot)
    score = ff(prot)

    assert score[0] - 2349.39882639 < 1e-6
