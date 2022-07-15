import numpy as np
import pytest
from chiLife.Protein import Protein, parse_paren

prot = Protein.from_pdb('test_data/1omp_H.pdb')


def test_from_pdb():

    ans = np.array([[ -0.07 , -24.223, -23.447],
                    [ -1.924, -20.646, -23.898],
                    [ -1.825, -19.383, -24.623],
                    [ -1.514, -19.5  , -26.1  ],
                    [ -2.237, -20.211, -26.799]])

    np.testing.assert_almost_equal(prot.coords[20:25], ans)


def test_from_multistate_pdb():
    prot = Protein.from_pdb('test_data/DHC.pdb')


def test_parse_paren1():
    P = parse_paren('(name CA or name CB)')
    assert P == ['name CA or name CB']


def test_parse_paren2():
    P = parse_paren('(resnum 32 and name CA) or (resnum 33 and name CB)')
    assert len(P) == 3


def test_parse_paren3():
    P = parse_paren('resnum 32 and name CA or (resnum 33 and name CB)')
    assert len(P) == 2


def test_parse_paren4():
    P = parse_paren('resname LYS ARG PRO and (name CA or (name CB and resname PRO)) or resnum 5')
    assert len(P) == 3


def test_parse_paren5():
    P = parse_paren('(name CA or (name CB and resname PRO) or name CD)')
    assert len(P) == 3


def test_select_or():
    m1 = prot.select_atoms('name CA or name CB')
    ans_mask = (prot.names == 'CA') + (prot.names == 'CB')

    np.testing.assert_almost_equal(m1.mask, ans_mask)
    assert np.all(np.isin(m1.names, ['CA', 'CB']))
    assert not np.any(np.isin(prot.names[~m1.mask], ['CA', 'CB']))


def test_select_and_or_and():

    m1 = prot.select_atoms('(resnum 32 and name CA) or (resnum 33 and name CB)')
    ans_mask = (prot.resnums == 32) * (prot.names == 'CA') + (prot.resnums == 33) * (prot.names == 'CB')

    np.testing.assert_almost_equal(m1.mask, ans_mask)
    assert np.all(np.isin(m1.names, ['CA', 'CB']))
    assert np.all(np.isin(m1.resnums, [32, 33]))


def test_select_and_not():
    m1 = prot.select_atoms('resnum 32 and not name CA')
    ans_mask = (prot.resnums == 32) * (prot.names != 'CA')

    np.testing.assert_almost_equal(m1.mask, ans_mask)
    assert not np.any(np.isin(m1.names, ['CA']))
    assert np.all(np.isin(m1.resnums, [32]))


def test_select_name_and_resname():
    m1 = prot.select_atoms("name CB and resname PRO")
    ans_mask = (prot.names == 'CB') * (prot.resnames == 'PRO')

    np.testing.assert_almost_equal(m1.mask, ans_mask)
    assert np.all(np.isin(m1.names, ['CB']))
    assert np.all(np.isin(m1.resnames, ['PRO']))


def test_select_complex():
    m1 = prot.select_atoms('resname LYS ARG PRO and (name CA or (type C and resname PRO)) or resnum 5')

    resnames = np.isin(prot.resnames, ['LYS', 'ARG', 'PRO'])
    ca_or_c_pro = ((prot.names == 'CA') + ((prot.atypes == 'C') * (prot.resnames == 'PRO' )))
    resnum5 = (prot.resnums == 5)
    ans_mask = resnames * ca_or_c_pro + resnum5

    np.testing.assert_almost_equal(m1.mask, ans_mask)
    assert not np.any(np.isin(prot.resnums[~m1.mask], 5))


def test_select_range():
    m1 = prot.select_atoms('resnum 5-15')
    m2 = prot.select_atoms('resnum 5:15')
    ans_mask = np.isin(prot.resnums, list(range(5, 16)))
    np.testing.assert_almost_equal(m1.mask, ans_mask)
    np.testing.assert_almost_equal(m2.mask, ans_mask)

features = ("atomids", "names", "altlocs", "resnames", "chains", "resnums", "occupancies", "bs", "atypes", "charges")
@pytest.mark.parametrize('feature', features)
def test_AtomSelection_features(feature):
    m1 = prot.select_atoms('resname LYS ARG PRO and (name CA or (type C and resname PRO)) or resnum 5')
    A = prot.__getattribute__(feature)[m1.mask]
    B = m1.__getattribute__(feature)

    assert np.all(A == B)
