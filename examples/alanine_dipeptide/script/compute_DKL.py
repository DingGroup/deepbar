#!/home/xqding/apps/miniconda3/bin/python

# Created by Xinqiang Ding (xqding@umich.edu)
# at 2020/05/15 02:42:31

#SBATCH --job-name=DKL_qp
#SBATCH --time=01:00:00
#SBATCH --partition=sixteencore
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --array=0-499
#SBATCH --output=./slurm_output/DKL_qp_%A_%a.txt

import numpy as np
import pickle
import torch
torch.set_default_dtype(torch.double)
import sys
sys.path.append("/home/xqding/course/projectsOnGitHub/MDTorch")
from RealNVP import *
from MDTorch import *
import torch.distributions as distributions
import argparse
import os
import simtk.openmm as omm
import simtk.unit as unit
import simtk.openmm.app as app
from sys import exit

## parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--idx_state", type = int)
parser.add_argument("--idx_mask", type = int)
parser.add_argument("--hidden_dim", type = int)

args = parser.parse_args()
idx_state = args.idx_state
idx_mask = args.idx_mask
hidden_dim = args.hidden_dim

reference_p_1, reference_p_2, reference_p_3 = 1, 0, 2
ref_particle_name = "{}-{}-{}".format(reference_p_1,
                                      reference_p_2,
                                      reference_p_3)

#### compute DKL pq ####
## read logJ_xyz
with open("./output/internal_coor_{}/logJ_xyz.pkl".format(ref_particle_name), 'rb') as file_handle:
    logJ_xyz = pickle.load(file_handle)

with open("./output/umbrella_sampling/energy.pkl", 'rb') as file_handle:
    energy = pickle.load(file_handle)
    
T = 298.15 * unit.kelvin
kbT = unit.BOLTZMANN_CONSTANT_kB*unit.AVOGADRO_CONSTANT_NA*T
energy = [tmp/kbT for tmp in energy]
energy = np.array(energy)
pq_logp = (-1)*energy + logJ_xyz.numpy()

## read training internal coordinates
with open("./output/internal_coor_{}/internal_coor.pkl".format(ref_particle_name), 'rb') as file_handle:
    data = pickle.load(file_handle)
    internal_coor = data['internal_coor']
    dihedral_limits = data['dihedral_limits']
    prob = data['prob']
    state_0_flag = data['state_0_flag']
    state_1_flag = data['state_1_flag']

if idx_state == 0:
    state_flag = state_0_flag
elif idx_state == 1:
    state_flag = state_1_flag
else:
    raise ValueError("idx_state has to be either 0 or 1.")

pq_logp = pq_logp[state_flag]
prob = prob[state_flag]
prob = prob/np.sum(prob)

ref_particle_1_xyz = internal_coor['ref_particle_1_xyz'][state_flag]
ref_particle_2_bond = internal_coor['ref_particle_2_bond'][state_flag]
ref_particle_3_bond = internal_coor['ref_particle_3_bond'][state_flag]
ref_particle_3_angle = internal_coor['ref_particle_3_angle'][state_flag]
bond = internal_coor['bond'][state_flag]
angle = internal_coor['angle'][state_flag]
dihedral = internal_coor['dihedral'][state_flag]

saved_num_steps = np.logspace(3, 4.7, 100)
saved_num_steps = list(saved_num_steps.astype(np.int32))

idx_job = int(os.environ['SLURM_ARRAY_TASK_ID'])
idx_repeat = idx_job // len(saved_num_steps)
idx_num_step = idx_job % len(saved_num_steps)

num_steps = saved_num_steps[idx_num_step]

data = torch.load("./output/learn_with_data_{}/model_masks_{}_hidden_dim_{}/realNVP_state_{}_repeat_{}_step_{}.pt".format(
                  ref_particle_name, idx_mask, hidden_dim, idx_state, idx_repeat, num_steps),
                  map_location=torch.device('cpu'))
masks = data['masks']
input_dim = data['input_dim']

realnvp = RealNVP(masks, hidden_dim)
realnvp.load_state_dict(data['state_dict'])
realnvp.eval()

ba_md = np.concatenate([ref_particle_2_bond,
                        ref_particle_3_bond,
                        ref_particle_3_angle,
                        bond,
                        angle], axis = -1)
ba_md = ba_md.astype(np.float64)

ba_md_mean = torch.from_numpy(np.sum(ba_md * prob[:, np.newaxis], 0))
tmp = np.random.choice(ba_md.shape[0], ba_md.shape[0], p = prob)
tmp = ba_md[tmp, :]
ba_md_cov = torch.from_numpy(np.cov(tmp.T))

normal_dist_ba = distributions.MultivariateNormal(
    loc = ba_md_mean,
    covariance_matrix = ba_md_cov)
pq_logq_ba = normal_dist_ba.log_prob(torch.from_numpy(ba_md))

normal_dist_z = distributions.Normal(loc = torch.tensor([0.0]),
                                     scale = torch.tensor([1.0]))

d = dihedral.astype(np.float64)
d = torch.from_numpy(d)

with torch.no_grad():
    z, logJ = realnvp.inverse(d)
    
pq_logq_d = torch.sum(normal_dist_z.log_prob(z), -1) + logJ
pq_logq = pq_logq_ba + pq_logq_d

#### compute DKL qp ####
## read system
with open("./structure/system.xml", 'r') as file_handle:
    xml = file_handle.read()    
system = omm.XmlSerializer.deserialize(xml)

## make an integrator
T = 298.15 * unit.kelvin
fricCoef = 1./unit.picoseconds
stepsize = 2. * unit.femtoseconds
integrator = omm.LangevinIntegrator(T, fricCoef, stepsize)

## read topology and pdb file
with open("./output/internal_coor_{}/coor_transformer.pkl".format(ref_particle_name), 'rb') as file_handle:
    coor_transformer = pickle.load(file_handle)
    
## minimize initial positions
platform = omm.Platform.getPlatformByName('CPU')
context = omm.Context(system, integrator, platform)

N = pq_logq.shape[0]
ba = normal_dist_ba.sample((N,))
qp_logq_ba = normal_dist_ba.log_prob(ba)

z = torch.squeeze(normal_dist_z.sample((N, input_dim)))
with torch.no_grad():
    d, logJ = realnvp(z)
qp_log_qd = torch.sum(normal_dist_z.log_prob(z), -1) - logJ
qp_logq = qp_logq_ba + qp_log_qd

ref_particle_1_xyz = torch.from_numpy(ref_particle_1_xyz[0].astype(np.float64))
ref_particle_1_xyz = ref_particle_1_xyz.repeat(N, 1)

ref_particle_2_bond = torch.unsqueeze(ba[:,0], -1)
ref_particle_3_bond = torch.unsqueeze(ba[:,1], -1)
ref_particle_3_angle = torch.unsqueeze(ba[:,2], -1)
bond = torch.unsqueeze(ba[:, 3:(3+input_dim)], -1)
angle = torch.unsqueeze(ba[:, (3+input_dim):(3+2*input_dim)], -1)
d = torch.unsqueeze(d, -1)

xyz, logJ_xyz = coor_transformer.compute_xyz_from_internal(ref_particle_1_xyz,
                                                           ref_particle_2_bond,
                                                           ref_particle_3_bond,
                                                           ref_particle_3_angle,
                                                           bond, angle, d)

xyz = xyz.numpy()
energy = []
flag = []
for i in range(xyz.shape[0]):
    if (i + 1) % 1000 == 0:
        print(i)
        
    if np.any(np.isnan(xyz[i])):
        flag.append(False)
    else:
        context.setPositions(xyz[i])
        state = context.getState(getEnergy = True)
        potential_energy = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
        energy.append(potential_energy)
        flag.append(True)
        
energy = np.array(energy)
flag = np.array(flag)

kbT = unit.BOLTZMANN_CONSTANT_kB*unit.AVOGADRO_CONSTANT_NA*T
kbT_kJ_per_mol = kbT.value_in_unit(unit.kilojoule_per_mole)
energy = energy / kbT_kJ_per_mol
qp_logp = (-1)*energy + logJ_xyz.numpy()[flag]
qp_log_q = qp_logq[flag]

os.makedirs("./output/DKL_{}_hidden_dim_{}".format(ref_particle_name, hidden_dim), exist_ok = True)
with open("./output/DKL_{}_hidden_dim_{}/result_mask_{}_state_{}_repeat_{}_steps_{}.pkl".format(
        ref_particle_name, hidden_dim, idx_mask, idx_state, idx_repeat, num_steps), 'wb') as file_handle:
    pickle.dump({'pq_logq': pq_logq, 'pq_logp': pq_logp,
                 'qp_logq': qp_logq, 'qp_logp': qp_logp,
                 'prob': prob}, file_handle)
    
exit()

prmtop = omm.app.AmberPrmtopFile("./structure/model_amber.prmtop")    
file_handle = open("./output/sample_from_q_mask_{}_temperature_{:.2f}.dcd".format(idx_mask, temperatures[idx_temperature]), 'bw')
dcd_file = omm.app.dcdfile.DCDFile(file_handle, prmtop.topology,
                                   integrator.getStepSize())
for i in range(min(100, xyz.shape[0])):
    dcd_file.writeModel(xyz[i])
file_handle.close()    
