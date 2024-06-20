import sys, os, pickle, json
from pathlib import Path
from dataclasses import dataclass
from pyrosetta import *
import numpy as np
import chiLife as xl
from rosetta.core.scoring.methods import WholeStructureEnergy
from pyrosetta.rosetta.core.io.raw_data import ScoreMap
from pyrosetta.rosetta.protocols.geometry import center_of_mass
from pyrosetta.rosetta.protocols.docking import setup_foldtree, DockingProtocol, calc_interaction_energy


init()
pose = pose_from_pdb('complex.pdb')
native_pose = pose_from_pdb('complex_20145.pdb')

to_centroid = SwitchResidueTypeSetMover('centroid')
to_full_atom = SwitchResidueTypeSetMover('fa_standard')

ub_offset = 629
exou_offset = 54

@dataclass
class ExpData:
    name: str
    site1: int
    site2: int
    label1: str
    label2: str
    r: np.ndarray
    P: np.ndarray

@EnergyMethod()
class RosettaDipolarDistance(WholeStructureEnergy):
    def __init__(self):
        WholeStructureEnergy.__init__(self, self.creator())

        # Load in experimental data
        with open('ExpData.pkl', 'rb') as f:
            self.data_dict = pickle.load(f)

        self.sites = []
        self.pose_sites = []
        self.labels = []
        for site1, site2 in self.data_dict:

            if site1 not in self.sites:
                self.sites.append(site1)
                self.pose_sites.append(site1 + ub_offset)
                self.labels.append(self.data_dict[(site1, site2)].label1)

            if site2 not in self.sites:
                self.sites.append(site2)
                self.pose_sites.append(site2 - exou_offset)
                self.labels.append(self.data_dict[(site1, site2)].label2)

        self.SLCache = {}
        self.iteration = 0
        self.last_score = 1e6

    def setup_for_scoring(self, pose, sf):
        # Create MDA Universe if it does not exist
        if not hasattr(self, 'mda_protein'):
            if not pose.is_fullatom():
                to_full_atom.apply(pose)
                self.mda_protein = xl.pose2mda(pose)
                to_centroid.apply(pose)
            else:
                self.mda_protein = xl.pose2mda(pose)

            self.SLCache = {site: xl.SpinLabel(label, site, self.mda_protein) 
                                for label, site in zip(self.labels, self.pose_sites)}

        # Recalculate rotamer ensembles every 10 iterations
        if self.iteration % 10 == 0 and pose.is_fullatom() and self.iteration != 0:
            # Update the universe with rosetta coordinates 
            self.mda_protein.atoms.positions = np.array([res.xyz(atom)
                                                         for res in pose.residues
                                                         for atom in range(1, res.natoms() + 1)
                                                         if res.atom_type(atom).element().strip() != 'X'])

            # Recalculate spin labels
            self.SLCache = {site: xl.SpinLabel(label, site, self.mda_protein) 
                                for label, site in zip(self.labels, self.pose_sites)}

        else:
            # Just move the spin label ensemble with the protein
            for site in self.pose_sites:
                backbone =  np.array([pose.residue(site).xyz('N'), 
                                      pose.residue(site).xyz('CA'), 
                                      pose.residue(site).xyz('C')])
                self.SLCache[site].to_site(backbone)

        self.iteration += 1

    def finalize_total_energy(self, pose, efunc, emap):

        score = 0
        for r1, r2 in self.data_dict:

            # Calculate preidcted P(r)s
            r= self.data_dict[(r1, r2)].r
            P = self.data_dict[(r1, r2)].P
            SL1, SL2 = self.SLCache[r1 + ub_offset], self.SLCache[r2 - exou_offset]
            dd = xl.distance_distribution(SL1, SL2, r).clip(0)

            # If the distance distribution is outside the provided domain (r)
            if np.any(np.isnan(dd)):

                CA_begin = pose.chain_begin(1)
                CB_begin = pose.chain_begin(2)
                CA_end = pose.chain_end(1)
                CB_end = pose.chain_end(2)

                diff = np.array(center_of_mass(pose, CA_begin, CA_end)) –
                       np.array(center_of_mass(pose, CB_begin, CB_end))

                cen_dist = np.sqrt(diff @ diff)

                # don’t alter score when calculating ddg
                if cen_dist > 1000: 
                    score = self.last_score
                    break

                # Otherwise penalize the difference between the mode distance to the cbeta distance
                else:
                    diff = np.array(pose.residue(r1).xyz('CA')) - np.array(pose.residue(r2).xyz('CA'))
                    CAdist = np.sqrt(diff @ diff)
                    score += np.abs(CAdist - r[np.argmax(P)])


            else:
                # Normalize and calculate overlap score
                P /= P.sum()
                dd /= dd.sum()

                score += -np.sum(np.minimum(P.clip(0), dd.clip(0)))

        self.last_score = score
        emap.set(self.scoreType, score)

# Create score functio and add dipolar distance term
RDD = RosettaDipolarDistance.scoreType
weight = 5
sfhr = create_score_function('docking')
sflr = create_score_function('interchain_cen')
sfhr.set_weight(RDD, weight)
sflr.set_weight(RDD, weight)

# Setup docking protocol
docking_protocol = DockingProtocol()
docking_protocol.set_highres_scorefxn(sfhr)
docking_protocol.set_lowres_scorefxn(sflr)
docking_protocol.set_partners('A_B')

# Setup for global docking
prtrbr = docking_protocol.perturber()
prtrbr.set_randomize1(True)
prtrbr.set_randomize2(True)
prtrbr.set_spin(True)

setup_foldtree(pose, "A_B", Vector1([1]))

nsuccess = 0
while nsuccess < 100:

    # Check to see if the decoy exists (checkpointing)
    newname = f'decoys/complex_{int(sys.argv[1]) * 100 + nsuccess}.pdb'
    if Path(newname).exists():
        print(f'{newname} alread exists skipping...')
        nsuccess += 1
        continue

    # Run protocol on a copy of the input pose
    pose2 = pose.clone()
    docking_protocol.apply(pose2)

    # Skip if filtered out by docking protocol
    if not pose2.is_fullatom():
        print('Pose is not FA. Skipping output')
        continue

    # compute additional score terms
    total = sfhr(pose2)
    scores = dict(list(ScoreMap.get_energies_map_from_scored_pose(pose2).items()))
    I_sc = calc_interaction_energy(pose2, sfhr, Vector1([1]))
    RMSD1 = pyrosetta.rosetta.core.scoring.CA_rmsd(native_pose, pose2)
    scores.update({'I_sc': I_sc, 'rmsd': RMSD1, 'name': newname})

    # Save to score file and pdb
    with open(f'scores_{sys.argv[1]}.sc', 'a') as f:
        json.dump(scores, f)
        f.write('\n')

    pose.dump_pdb(newname)
    nsuccess += 1
