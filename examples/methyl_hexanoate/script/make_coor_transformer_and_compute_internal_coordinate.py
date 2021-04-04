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
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import os
import math

parser = argparse.ArgumentParser()
parser.add_argument("--solvent", type = str, choices = ['OBC2', 'vacuum'], required = True)
args = parser.parse_args()

mol_id = "mobley_1017962"
topology = mdtraj.load_prmtop(f"./structure/{mol_id}.prmtop")

bonds = MMFlow.utils.get_bonded_atoms(topology)
## Pick three reference particles such that reference_particle_2 and
## reference_particle_3 are bonded with reference_particle_1.
## You will need to change the following three numbers based on specific molecules
reference_particle_1 = 3 ## C4
reference_particle_2 = 2 ## C3
reference_particle_3 = 4 ## C5
coor_transformer = MMFlow.utils.CoordinateTransformer(
    bonds,
    reference_particle_1,
    reference_particle_2,
    reference_particle_3,
    dihedral_mode = 'fork'
)

with open(f"./output/{args.solvent}/coordinate_transformer.pkl", 'wb') as file_handle:
    pickle.dump(coor_transformer, file_handle)

traj_md = mdtraj.load_dcd(f"./output/{args.solvent}/traj/traj_md.dcd", topology)
ic_md, ic_md_logabsdet = coor_transformer.compute_internal_coordinate_from_xyz(torch.from_numpy(traj_md.xyz))

ic_md_first_half = ic_md[0:len(ic_md)//2]
ic_md_logabsdet_first_half = ic_md_logabsdet[0:len(ic_md_logabsdet)//2]

ic_md_second_half = ic_md[len(ic_md)//2:]
ic_md_logabsdet_second_half = ic_md_logabsdet[len(ic_md_logabsdet)//2:]

torch.save({'ic_md': ic_md_first_half, 'ic_md_logabsdet': ic_md_logabsdet_first_half}, f"./output/{args.solvent}/ic_md_first_half.pt")
torch.save({'ic_md': ic_md_second_half, 'ic_md_logabsdet': ic_md_logabsdet_second_half}, f"./output/{args.solvent}/ic_md_second_half.pt")

# ## get dihedrals that need to be modelled as circular features
# circular_dihedral_flag = []
# for j in range(ic.dihedral.shape[1]):
#     d = ic.dihedral[:,j].clone()
    
#     if torch.max(d) - torch.min(d) < math.pi:
#         circular_dihedral_flag.append(False)
#         continue

#     d[d < 0] = d[d < 0] + 2*math.pi
#     if torch.max(d) - torch.min(d) < math.pi:
#         circular_dihedral_flag.append(False)
#         ic.dihedral[:,j] = d
#         continue
#     circular_dihedral_flag.append(True)
# circular_dihedral_flag = torch.tensor(circular_dihedral_flag)

# ic_limits = {
#     'reference_particle_1_xyz_min' : ic.reference_particle_1_xyz.min(0)[0],
#     'reference_particle_1_xyz_max' : ic.reference_particle_1_xyz.max(0)[0],
    
#     'reference_particle_2_bond_min' : ic.reference_particle_2_bond.min(0)[0],
#     'reference_particle_2_bond_max' : ic.reference_particle_2_bond.max(0)[0],
    
#     'reference_particle_3_bond_min' : ic.reference_particle_3_bond.min(0)[0],
#     'reference_particle_3_bond_max' : ic.reference_particle_3_bond.max(0)[0],
#     'reference_particle_3_angle_min' : ic.reference_particle_3_angle.min(0)[0],
#     'reference_particle_3_angle_max' : ic.reference_particle_3_angle.max(0)[0],
    
#     'bond_min': ic.bond.min(0)[0],
#     'bond_max': ic.bond.max(0)[0],    
#     'angle_min': ic.angle.min(0)[0],
#     'angle_max': ic.angle.max(0)[0],
#     'dihedral_min': ic.dihedral.min(0)[0],
#     'dihedral_max': ic.dihedral.max(0)[0],    
# }

ic_md_limits = {
    'reference_particle_3_angle_min' : ic_md.reference_particle_3_angle.min(0)[0],
    'reference_particle_3_angle_max' : ic_md.reference_particle_3_angle.max(0)[0],
    'angle_min': ic_md.angle.min(0)[0],
    'angle_max': ic_md.angle.max(0)[0]
}
torch.save(ic_md_limits, f"./output/{args.solvent}/ic_md_limits.pt")

# ic_flag = {}
# ic_flag['reference_particle_1_xyz'] = [None, None, None]
# ic_flag['reference_particle_2_bond'] = 'feature'
# ic_flag['reference_particle_3_bond'] = 'feature'
# ic_flag['reference_particle_3_angle'] = 'feature'

# ic_flag['bond'] = []
# ic_flag['angle'] = []
# ic_flag['dihedral'] = []

# for j in range(ic.bond.shape[-1]):
#     ic_flag['bond'].append('feature')
#     ic_flag['angle'].append('feature')
#     ic_flag['dihedral'].append('feature')
# ic_flag['circular_dihedral_flag'] = circular_dihedral_flag

# torch.save({'ic': ic[0:len(ic)//2],
#             'ic_limits': ic_limits,
#             'ic_flag': ic_flag,
#             'ic_logabsdet': ic_logabsdet[0:len(ic_logabsdet)//2],
#             'coor_transformer': coor_transformer,
#             'circular_dihedral_flag': circular_dihedral_flag},
#            f"./output/{args.solvent}/ic_first_half.pt")

# torch.save({'ic': ic[len(ic)//2:],
#             'ic_limits': ic_limits,
#             'ic_flag': ic_flag,
#             'ic_logabsdet': ic_logabsdet[len(ic_logabsdet)//2:],
#             'coor_transformer': coor_transformer,
#             'circular_dihedral_flag': circular_dihedral_flag},
#            f"./output/{args.solvent}/ic_second_half.pt")

# torch.save({'ic_limits': ic_limits,
#             'ic_flag': ic_flag,
#             'ic_logabsdet': ic_logabsdet,
#             'coor_transformer': coor_transformer},
#            f"./output/{args.solvent}/ic_info.pt")

dihedral = ic_md.dihedral.numpy()

os.makedirs(f"./output/{args.solvent}/plots", exist_ok = True)
with PdfPages(f"./output/{args.solvent}/plots/dihedral_dist_md.pdf") as pdf:
    for j in range(dihedral.shape[-1]):
        print(j)
        fig = plt.figure(0)
        fig.clf()
        plt.hist(dihedral[:, j], bins = 30, range = [-math.pi, math.pi], density = True, alpha = 1.0, label = 'md')
        plt.legend()
        pdf.savefig(fig)
        
