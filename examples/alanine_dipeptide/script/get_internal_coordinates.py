__author__ = "Xinqiang Ding <xqding@umich.edu>"
__date__ = "2020/05/14 18:35:12"

import numpy as np
import mdtraj
import sys
sys.path.append("/home/xqding/course/projectsOnGitHub/MMFlow/")
import MMFlow
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
particle_bonds = MMFlow.utils.get_bonded_atoms(topology)
reference_particle_1 = 1
reference_particle_2 = 0
reference_particle_3 = 2
coor_transformer = MMFlow.utils.CoordinateTransformer(
    particle_bonds,
    reference_particle_1,
    reference_particle_2,
    reference_particle_3
)

os.makedirs("./output/internal_coor", exist_ok = True)
M = 25
traj = mdtraj.load_dcd(
        "./output/umbrella_sampling/traj_all.dcd", top = topology, stride = 1)
with torch.no_grad():
    ic, ic_logabsdet = coor_transformer.compute_internal_from_xyz(torch.from_numpy(traj.xyz))

with open("./output/umbrella_sampling/fastmbar_result.pkl", 'rb') as file_handle:
    fastmbar_result = pickle.load(file_handle)
prob_md = fastmbar_result['log_prob_mix']

# xyz, xyz_logabsdet = coor_transformer.compute_xyz_from_internal(
#     ic.reference_particle_1_xyz,
#     ic.reference_particle_2_bond,
#     ic.reference_particle_2_polar_angle,
#     ic.reference_particle_2_azimuthal_angle,
#     ic.reference_particle_3_bond,
#     ic.reference_particle_3_polar_angle,
#     ic.reference_particle_3_azimuthal_angle,
#     ic.bond, ic.angle, ic.dihedral)

## save internal coordinates along with energy and jocobian
file_name = "./output/internal_coor/internal_coor.pkl"
with open(file_name, 'wb') as file_handle:
    pickle.dump({"ic": ic , 
                 "prob": prob_md}, file_handle)
