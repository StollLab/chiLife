#!/usr/bin/env python
"""Rebuild residue PDB templates from RCSB CCD ideal coordinates.

Uses ``chilife.bio_ccd`` ideal coordinates, preserves chiLife atom naming and
backbone orientation from the existing residue templates, then regenerates
``residue_internal_coords/*_ic.pkl`` files.

Canonical amino acids use their own CCD entries. Protonation variants in
``alt_prot_states`` use the parent amino-acid CCD geometry (e.g. ASH from ASP)
with variant-specific atoms omitted when absent from the chiLife template.
"""

from __future__ import annotations

import argparse
import pickle
import re
from pathlib import Path

import MDAnalysis as mda
import numpy as np

import chilife
from chilife.globals import alt_prot_states, nataa_codes

RL_DIR = chilife.RL_DIR
PDB_DIR = RL_DIR / "residue_pdbs"
IC_DIR = RL_DIR / "residue_internal_coords"

SKIP_CCD_ATOMS = frozenset({"OXT", "HXT", "H2", "H3"})

# chiLife name -> CCD atom_id where automatic mapping fails
CHILIFE_TO_CCD_H = {
    "GLY": {"HA": "HA2", "3HA": "HA3"},
}

# bio_ccd three-letter codes for alt states often name unrelated compounds; use parents.
ALT_CCD_SOURCE = dict(alt_prot_states)

# Parent CCD atom_ids to drop when building an alt-state template
ALT_CCD_SKIP: dict[str, frozenset[str]] = {
    "ASH": frozenset(),
    "GLH": frozenset(),
    "LYN": frozenset({"HZ3"}),
    "CYM": frozenset({"HG"}),
    "HID": frozenset({"HE2"}),
    "HIE": frozenset({"HD1"}),
    "HIP": frozenset(),
    "TYM": frozenset({"HH"}),
}


def ccd_h_to_chilife(name: str) -> str:
    """Map a CCD/PDB hydrogen name to chiLife's ``{n}H{anchor}`` style."""
    if name in ("H", "HA", "H2", "H3", "HZ"):
        return name
    match = re.match(r"^H([A-Z]+?\d*)(\d+)$", name)
    if match:
        return f"{match.group(2)}H{match.group(1)}"
    return name


def kabsch(mobile: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return ``R, t`` such that ``mobile @ R + t`` best matches ``target``."""
    mobile = np.asarray(mobile, dtype=float)
    target = np.asarray(target, dtype=float)
    mob_cent = mobile.mean(axis=0)
    tgt_cent = target.mean(axis=0)
    mob = mobile - mob_cent
    tgt = target - tgt_cent
    H = mob.T @ tgt
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt = Vt.copy()
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    t = tgt_cent - mob_cent @ R
    return R, t


def ccd_positions(
    ccd_resname: str,
    ccd: dict,
    *,
    skip_atoms: frozenset[str] = SKIP_CCD_ATOMS,
) -> dict[str, np.ndarray]:
    """Ideal Cartesian coordinates from ``bio_ccd`` (Angstrom)."""
    if ccd_resname not in ccd:
        raise KeyError(f"{ccd_resname} not in bio_ccd")

    atoms = ccd[ccd_resname]["chem_comp_atom"]
    out: dict[str, np.ndarray] = {}
    for i, atom_id in enumerate(atoms["atom_id"]):
        if atom_id in skip_atoms:
            continue
        out[atom_id] = np.array(
            [
                float(atoms["pdbx_model_Cartn_x_ideal"][i]),
                float(atoms["pdbx_model_Cartn_y_ideal"][i]),
                float(atoms["pdbx_model_Cartn_z_ideal"][i]),
            ],
            dtype=float,
        )
    return out


def build_chilife_to_ccd_map(
    resname: str, template_names: list[str], ccd_pos: dict[str, np.ndarray]
) -> dict[str, str]:
    """Map each chiLife atom name in the template to a CCD ``atom_id``."""
    ccd_h_to_chi: dict[str, str] = {}
    for ccd_name in ccd_pos:
        if ccd_name[0] == "H" or ccd_name in ("H", "HA", "HZ"):
            ccd_h_to_chi[ccd_name] = ccd_h_to_chilife(ccd_name)

    chi_to_ccd: dict[str, str] = {}
    for name in template_names:
        if name in ccd_pos:
            chi_to_ccd[name] = name
            continue
        if name in CHILIFE_TO_CCD_H.get(resname, {}):
            chi_to_ccd[name] = CHILIFE_TO_CCD_H[resname][name]
            continue
        for ccd_name, chi_name in ccd_h_to_chi.items():
            if chi_name == name:
                chi_to_ccd[name] = ccd_name
                break
        else:
            raise KeyError(f"{resname}: no CCD atom for chiLife name {name!r}")
    return chi_to_ccd


def rebuild_residue(
    resname: str,
    ccd: dict,
    pdb_dir: Path,
    ic_dir: Path,
    dry_run: bool = False,
    *,
    ccd_source: str | None = None,
    skip_ccd_atoms: frozenset[str] | None = None,
) -> None:
    res_lower = resname.lower()
    template_path = pdb_dir / f"{res_lower}.pdb"
    if not template_path.exists():
        raise FileNotFoundError(template_path)

    template = mda.Universe(str(template_path), in_memory=True)
    template_names = [a.name.strip() for a in template.atoms]
    ref_coords = {a.name.strip(): a.position.copy() for a in template.atoms}

    ccd_resname = ccd_source or resname
    skip = SKIP_CCD_ATOMS | (skip_ccd_atoms or frozenset())
    ccd_pos = ccd_positions(ccd_resname, ccd, skip_atoms=skip)
    chi_to_ccd = build_chilife_to_ccd_map(resname, template_names, ccd_pos)

    mobile_bb = np.vstack([ccd_pos[chi_to_ccd[x]] for x in ("N", "CA", "C")])
    target_bb = np.vstack([ref_coords[x] for x in ("N", "CA", "C")])
    R, t = kabsch(mobile_bb, target_bb)

    new_coords: dict[str, np.ndarray] = {}
    for chi_name in template_names:
        ccd_name = chi_to_ccd[chi_name]
        new_coords[chi_name] = ccd_pos[ccd_name] @ R + t

    lines = []
    for i, name in enumerate(template_names):
        atom = template.atoms[i]
        coord = new_coords[name]
        lines.append(
            f"ATOM  {i + 1:5d} {name:^4s} {resname:3s} {atom.chainID:1s}{int(atom.resid):4d}    "
            f"{coord[0]:8.3f}{coord[1]:8.3f}{coord[2]:8.3f}{1.0:6.2f}{1.0:6.2f}          {atom.element:>2s}  \n"
        )

    if dry_run:
        print(f"{resname}: would write {template_path} ({len(lines)} atoms)")
        return

    try:
        sorted_lines = chilife.sort_pdb(lines)
    except (IndexError, RuntimeError):
        # Template atom order is already valid; some small protonation variants
        # fail bond guessing during sort (e.g. ASH).
        sorted_lines = lines
    template_path.write_text("".join(sorted_lines))

    struct = mda.Universe(str(template_path), in_memory=True)
    pref_d = chilife.dihedral_defs[resname]
    ICs = chilife.MolSysIC.from_atoms(struct, preferred_dihedrals=pref_d)
    ic_path = ic_dir / f"{res_lower}_ic.pkl"
    with open(ic_path, "wb") as f:
        pickle.dump(ICs, f)
    print(f"{resname}: wrote {template_path.name} and {ic_path.name}")


def _rebuild_targets(
    resnames: list[str] | None,
    *,
    canonical: bool,
    alt_states: bool,
) -> list[tuple[str, str | None, frozenset[str] | None]]:
    """Return ``(template, ccd_source, skip_ccd_atoms)`` jobs to run."""
    jobs: list[tuple[str, str | None, frozenset[str] | None]] = []
    allowed: dict[str, tuple[str | None, frozenset[str] | None]] = {}

    if canonical:
        for name in sorted(k for k in nataa_codes if len(k) == 3):
            allowed[name] = (None, None)

    if alt_states:
        for name in sorted(alt_prot_states):
            allowed[name] = (
                ALT_CCD_SOURCE[name],
                ALT_CCD_SKIP.get(name, frozenset()),
            )

    if not allowed:
        raise SystemExit("Select --canonical, --alt-states, or --all")

    if resnames:
        targets = [r.upper() for r in resnames]
        unknown = set(targets) - set(allowed)
        if unknown:
            raise SystemExit(f"Unknown residue names: {sorted(unknown)}")
    else:
        targets = sorted(allowed)

    for name in targets:
        source, skip = allowed[name]
        jobs.append((name, source, skip))
    return jobs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--res",
        nargs="*",
        help="Residue names to rebuild (default: all selected groups)",
    )
    parser.add_argument(
        "--canonical",
        action="store_true",
        help="Rebuild the 20 canonical amino acids",
    )
    parser.add_argument(
        "--alt-states",
        action="store_true",
        help="Rebuild canonical protonation variants (ASH, GLH, HID, ...)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Rebuild canonical amino acids and protonation variants",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print actions without writing files",
    )
    args = parser.parse_args()

    canonical = args.canonical or args.all
    alt_states = args.alt_states or args.all
    if not (canonical or alt_states):
        canonical = True

    with open(chilife.DATA_DIR / "bio_ccd.pkl", "rb") as f:
        ccd = pickle.load(f)

    for resname, ccd_source, skip_ccd_atoms in _rebuild_targets(
        args.res, canonical=canonical, alt_states=alt_states
    ):
        rebuild_residue(
            resname,
            ccd,
            PDB_DIR,
            IC_DIR,
            dry_run=args.dry_run,
            ccd_source=ccd_source,
            skip_ccd_atoms=skip_ccd_atoms,
        )


if __name__ == "__main__":
    main()
