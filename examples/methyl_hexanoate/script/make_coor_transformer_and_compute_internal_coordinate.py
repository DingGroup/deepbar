import mdtraj
from collections import defaultdict
import sys
sys.path.append("../../")
import MMFlow
import pickle
import torch
from sys import exit
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--solvent", type = str, choices = ['OBC2', 'vacuum'], required = True)
args = parser.parse_args()

mol_id = "mobley_1017962"
topology = mdtraj.load_prmtop(f"./structure/{mol_id}.prmtop")

bonds = MMFlow.utils.get_bonded_atoms(topology)
reference_particle_1 = 3 ## C4
reference_particle_2 = 2 ## C3
reference_particle_3 = 4 ## C5
coor_transformer = MMFlow.utils.CoordinateTransformer(
    bonds,
    reference_particle_1,
    reference_particle_2,
    reference_particle_3
)

with open(f"./output/{args.solvent}/coordinate_transformer.pkl", 'wb') as file_handle:
    pickle.dump(coor_transformer, file_handle)

traj_md = mdtraj.load_dcd(f"./output/{args.solvent}/traj/traj_md.dcd", topology)
ic_md, ic_md_logabsdet = coor_transformer.compute_internal_coordinate_from_xyz(torch.from_numpy(traj_md.xyz))

torch.save({'ic_md': ic_md, 'ic_md_logabsdet': ic_md_logabsdet}, f"./output/{args.solvent}/ic_md.pt")

ic_md_limits = {
    'reference_particle_3_angle_min' : ic_md.reference_particle_3_angle.min(0)[0],
    'reference_particle_3_angle_max' : ic_md.reference_particle_3_angle.max(0)[0],    
    'angle_min': ic_md.angle.min(0)[0],
    'angle_max': ic_md.angle.max(0)[0]
}

torch.save(ic_md_limits, f"./output/{args.solvent}/ic_md_limits.pt")


    
