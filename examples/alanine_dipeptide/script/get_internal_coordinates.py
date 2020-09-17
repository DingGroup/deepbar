__author__ = "Xinqiang Ding <xqding@umich.edu>"
__date__ = "2020/05/14 18:35:12"

import numpy as np
import mdtraj
import sys
sys.path.append("/home/xqding/course/projectsOnGitHub/MDTorch/")
import MDTorch.utils as utils
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import simtk.openmm as omm
import pickle
from sys import exit
import os
import torch
torch.set_default_dtype(torch.double)
import math

## load topology
topology = mdtraj.load_prmtop("./structure/alanine_dipeptide.prmtop")

## make a CoordinateTransformer
particle_bonds = utils.get_bonded_atoms(topology.to_openmm())
coor_transformer = utils.CoordinateTransformer(particle_bonds)
reference_p_1, reference_p_2, reference_p_3 = 1, 0, 2

coor_transformer.set_reference_particles(reference_p_1,
                                         reference_p_2,
                                         reference_p_3)

## compute internal coordinates from trajectories
ref_p_name = "{}-{}-{}".format(reference_p_1,
                               reference_p_2,
                               reference_p_3)

os.makedirs("./output/internal_coor_{}".format(ref_p_name), exist_ok = True)
M = 25
traj = mdtraj.load_dcd(
        "./output/umbrella_sampling/traj_all.dcd", top = topology)
res = coor_transformer.compute_internal_from_traj(traj)

## define the two states
tmp = res['dihedral']/np.pi*180
state_0_flag = (tmp[:, 1] > 0) & (tmp[:, 1] <= 120)
state_1_flag = ~ state_0_flag

## load weights for each frame in the trajectory
with open("./output/umbrella_sampling/fastmbar_result.pkl", 'rb') as file_handle:
    data = pickle.load(file_handle)
    log_prob_mix = data['log_prob_mix']
prob_md = np.exp(-log_prob_mix)    

with PdfPages('./output/internal_coor_{}/multipage_pdf.pdf'.format(ref_p_name)) as pdf:
    for j in range(res['dihedral'].shape[1]):
        print(j)
        fig = plt.figure(0)
        fig.clf()
        plt.hist(res['dihedral'][:,j], 30, range = (-np.pi, np.pi), weights = prob_md)
        pdf.savefig()

## load energy
potential_energy = []
for i in range(25**2):
    with open(f"./output/umbrella_sampling/traj/potential_energy_kJ_per_mole_{i}.pkl", 'rb') as file_handle:
        potential_energy.append(pickle.load(file_handle))
potential_energy = np.concatenate(potential_energy)

exit()

## compute the jacobian of Cartesian coordinate with respect to internal coordinates    
with torch.no_grad():    
    xyz, logJ_xyz = coor_transformer.compute_xyz_from_internal(torch.tensor(res['ref_particle_1_xyz']),
                                                               torch.tensor(res['ref_particle_2_bond']),
                                                               torch.tensor(res['ref_particle_3_bond']),
                                                               torch.tensor(res['ref_particle_3_angle']),
                                                               torch.tensor(res['bond'][:,:,np.newaxis]),
                                                               torch.tensor(res['angle'][:,:,np.newaxis]),
                                                               torch.tensor(res['dihedral'][:,:,np.newaxis]))

## save internal coordinates along with energy and jocobian
os.makedirs("./output/internal_coor_{}".format(ref_p_name), exist_ok = True)
file_name = "./output/internal_coor_{}/internal_coor.pkl".format(ref_p_name)
with open(file_name, 'wb') as file_handle:
    pickle.dump({"internal_coor": res, 'potential_energy': potential_energy,
                 "prob": prob_md, 'state_0_flag': state_0_flag,
                 'state_1_flag': state_1_flag}, file_handle)
    
file_name = "./output/internal_coor_{}/logJ_xyz.pkl".format(ref_p_name) 
with open(file_name, 'wb') as file_handle:
    pickle.dump(logJ_xyz, file_handle)
        
with open("./output/internal_coor_{}/coor_transformer.pkl".format(ref_p_name), 'wb') as file_handle:
    pickle.dump(coor_transformer, file_handle)
