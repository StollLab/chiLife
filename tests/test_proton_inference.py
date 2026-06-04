"""Per-residue proton placement tests using off-rotamer sampled conformers.

For each canonical amino acid, an off-rotamer sample provides a protonated ground
truth. The heavy-atom subset is used as input to the same proton-inference path as
``mutate`` (``_fill_missing_atoms_coords`` with the closest library rotamer).
"""

from __future__ import annotations

import numpy as np
import pytest
import MDAnalysis as mda

import chilife
from chilife.globals import bond_hmax, nataa_codes
from chilife.protein_utils import (
    _closest_rotamer_index,
    _expected_residue_ensemble,
    _fill_missing_atoms_coords,
)

# One representative context per canonical amino acid (PDB path, residue number).
_CANONICAL_SITES: dict[str, tuple[str, int]] = {}
_ubq = mda.Universe("test_data/1ubq.pdb", in_memory=True)
for _res in _ubq.residues:
    if _res.resname in nataa_codes and _res.resname not in _CANONICAL_SITES:
        _CANONICAL_SITES[_res.resname] = ("test_data/1ubq.pdb", int(_res.resid))

_CANONICAL_SITES["CYS"] = (str(chilife.RL_DIR / "residue_pdbs/cys.pdb"), 1)
_CANONICAL_SITES["TRP"] = (str(chilife.RL_DIR / "residue_pdbs/trp.pdb"), 1)

CANONICAL_AA = tuple(sorted(_CANONICAL_SITES))


def _sample_off_rotamer_coords(ensemble: chilife.RotamerEnsemble) -> np.ndarray:
    """Return one off-rotamer Cartesian conformer (n_atoms, 3)."""
    out = ensemble.sample(n=1, off_rotamer=True, remove_clashing=False)
    coords = out[0] if isinstance(out, (tuple, list)) else out
    coords = np.asarray(coords, dtype=float)
    if coords.ndim == 3:
        coords = coords[0]
    return coords


def _heavy_residue(protein, site: int, segid: str, coord_dict: dict, heavy_names: set):
    """Return a residue view with ensemble heavy-atom coordinates only."""
    heavy = protein.select_atoms(f"resid {site} and segid {segid}")
    for atom in heavy:
        if atom.name in heavy_names:
            atom.position = coord_dict[atom.name]
    return heavy.select_atoms(f"name {' '.join(sorted(heavy_names))}").residues[0]


def _max_h_displacement(truth_coords, inferred_coords, h_mask) -> float:
    if not np.any(h_mask):
        return 0.0
    return float(np.linalg.norm(inferred_coords[h_mask] - truth_coords[h_mask], axis=1).max())


def _bad_xh_bonds(coords, types, bonds) -> int:
    n_bad = 0
    for b1, b2 in bonds:
        if types[b1] != "H" and types[b2] != "H":
            continue
        dist = np.linalg.norm(coords[b1] - coords[b2])
        maxd = float(bond_hmax((types[b1], types[b2])))
        if dist > maxd + 0.25 or dist < 0.7:
            n_bad += 1
    return n_bad


@pytest.fixture(scope="module")
def off_rotamer_seed() -> int:
    return 2026


@pytest.mark.parametrize("resname", CANONICAL_AA)
def test_off_rotamer_proton_inference(resname, off_rotamer_seed):
    """Infer protons on heavy-only off-rotamer conformers; compare to protonated truth."""
    pdb_path, site = _CANONICAL_SITES[resname]
    np.random.seed(off_rotamer_seed + abs(hash(resname)) % 10_000)

    protein = mda.Universe(pdb_path, in_memory=True)
    segid = protein.select_atoms(f"resid {site}").segids[0]

    ensemble = chilife.RotamerEnsemble(
        resname,
        site,
        protein=protein,
        use_H=True,
        eval_clash=False,
    )
    protonated = _sample_off_rotamer_coords(ensemble)
    coord_dict = dict(zip(ensemble.atom_names, protonated))

    heavy_names = set(ensemble.atom_names[ensemble.atom_types != "H"])
    heavy_res = _heavy_residue(protein, site, segid, coord_dict, heavy_names)
    fill_ens = _expected_residue_ensemble(
        heavy_res, protein, use_H=True, ignore_waters=True
    )
    rot_idx = _closest_rotamer_index(fill_ens, heavy_res)
    _, _, inferred = _fill_missing_atoms_coords(heavy_res, fill_ens, rot_idx)

    h_mask = ensemble.atom_types == "H"
    heavy_mask = ~h_mask

    np.testing.assert_allclose(
        inferred[heavy_mask],
        protonated[heavy_mask],
        atol=1e-3,
        err_msg=f"{resname}: heavy atoms moved during proton fill",
    )

    max_h_err = _max_h_displacement(protonated, inferred, h_mask)
    bad_bonds = _bad_xh_bonds(inferred, ensemble.atom_types, ensemble.bonds)

    assert bad_bonds == 0, (
        f"{resname}: {bad_bonds} X-H bonds out of tolerance after proton fill "
        f"(max H displacement {max_h_err:.3f} A)"
    )
    np.testing.assert_allclose(
        inferred[h_mask],
        protonated[h_mask],
        atol=0.15,
        err_msg=(
            f"{resname}: inferred H differ from off-rotamer truth "
            f"(max displacement {max_h_err:.3f} A)"
        ),
    )
