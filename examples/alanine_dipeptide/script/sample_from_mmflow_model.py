import pickle
import math
import numpy as np
import torch
torch.set_default_dtype(torch.float64)
from torch import distributions
import mdtraj
import simtk.openmm as omm
import simtk.openmm.app as app
import simtk.unit as unit
import sys
sys.path.append("/home/xqding/course/projectsOnGitHub/MMFlow")
from deepbar import transform, NFlow, utils
from sys import exit
import matplotlib as mpl
mpl.use("Agg")
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt

## load mmflow model
idx_step = 3019
data = torch.load(f"./output/mmflow_models/mmflow_model_{idx_step}.pt")

conditioner_net_create_fn = lambda feature_size, context_size, output_size: \
    transform.ResidualNet(feature_size,
                          context_size,
                          output_size,
                          hidden_size = data['hidden_size'],
                          num_blocks = data['num_blocks'])

mmflow = NFlow(data['feature_size'],
                data['context_size'],
                data['circular_feature_flag'],
                data['transform_feature_flag'],
                conditioner_net_create_fn,
                num_bins_circular = data['num_bins_circular'],
                num_bins_regular = data['num_bins_regular'])

mmflow.load_state_dict(data['state_dict'])

## sample from mmflow model
with open("./output/internal_coor/internal_coor.pkl", 'rb') as file_handle:
    data = pickle.load(file_handle)
    ic = data['ic']
    coor_transformer = data['coor_transformer']
    prob = data['prob']
    prob = prob / np.sum(prob)
    
ic.reference_particle_3_angle /= math.pi
ic.angle /= math.pi

feature = torch.cat([ic.reference_particle_3_angle[:, None],
                     ic.angle, ic.dihedral], -1)
feature = feature.double()
feature_size = feature.shape[-1]
circular_feature_flag = torch.zeros(feature_size)
circular_feature_flag[-ic.dihedral.shape[-1]:] = 1

context = torch.cat([ic.reference_particle_2_bond[:, None],
                     ic.reference_particle_3_bond[:, None],
                     ic.bond], -1)
context = context.double()

sample_index = np.random.choice(len(prob), 100_0000, replace = True, p = prob)
context = context[sample_index]
context_mean = torch.mean(context, 0)
context_centered = context - context_mean[None, :]
context_cov = torch.matmul(context_centered.T, context_centered) / context.shape[0]
context_dist = distributions.MultivariateNormal(loc = context_mean,
                                                covariance_matrix = context_cov)

num_samples = 10000
context_flow = context_dist.sample((num_samples,))
with torch.no_grad():
    feature_flow, log_prob = mmflow.sample_and_compute_log_prob(1, context_flow)
feature_flow = torch.squeeze(feature_flow)

with PdfPages("./output/plots/feature_dist.pdf") as pdf:
    for j in range(feature.shape[-1]):
        print(j)
        fig = plt.figure(0)
        fig.clf()
        if circular_feature_flag[j] > 0:
            plt.hist(feature[:, j].numpy(), bins = 30, range = [-math.pi, math.pi], density = True, weights = prob, alpha = 0.5, label = 'md')
            plt.hist(feature_flow[:, j].numpy(), bins = 30, range = [-math.pi, math.pi], density = True, alpha = 0.5, label = 'mmflow')            
        else:
            plt.hist(feature[:, j].numpy(), bins = 30, range = [0.0, 1.0], density = True, weights = prob, alpha = 0.5, label = 'md')
            plt.hist(feature_flow[:, j].numpy(), bins = 30, range = [0.0, 1.0], density = True, alpha = 0.5, label = 'mmflow')
        plt.legend()
        pdf.savefig(fig)

ic_flow = utils.InternalCoordinate(
    reference_particle_1_xyz =  torch.zeros(num_samples, 3),
    reference_particle_2_bond = context_flow[:, 0],
    reference_particle_3_bond = context_flow[:, 1],
    reference_particle_3_angle = feature_flow[:, 0]*math.pi,
    bond = context_flow[:, 2:],
    angle = feature_flow[:, 1:ic.angle.shape[-1]+1]*math.pi,
    dihedral = feature_flow[:, ic.angle.shape[-1]+1:]
)

xyz_flow, logabsdet = coor_transformer.compute_xyz_from_internal_coordinate(
    ic_flow.reference_particle_1_xyz,
    ic_flow.reference_particle_2_bond,
    ic_flow.reference_particle_3_bond,
    ic_flow.reference_particle_3_angle,
    ic_flow.bond,
    ic_flow.angle,
    ic_flow.dihedral
)

topology = mdtraj.load_prmtop("./structure/alanine_dipeptide.prmtop")
traj_flow = mdtraj.Trajectory(xyz_flow.numpy(), topology)
traj_flow.save_dcd("./output/traj_flow.dcd")

## openmm context
with open("./structure/system.xml", 'r') as file_handle:
    xml = file_handle.read()    
system = omm.XmlSerializer.deserialize(xml)

## setup context
platform = omm.Platform.getPlatformByName('Reference')
T = 298.15 * unit.kelvin
fricCoef = 10/unit.picoseconds
stepsize = 1 * unit.femtoseconds
integrator = omm.LangevinIntegrator(T, fricCoef, stepsize)
omm_context = omm.Context(system, integrator, platform)
kbT = unit.BOLTZMANN_CONSTANT_kB * T * unit.AVOGADRO_CONSTANT_NA

traj = mdtraj.load_dcd(
        "./output/umbrella_sampling/traj_all.dcd", top = topology, stride = 1)

index = np.random.choice(range(traj.n_frames), 10000, replace = True, p = prob)
xyz = traj.xyz[index]

log_p = []
for i in range(xyz.shape[0]):
    omm_context.setPositions(xyz[i])
    state = omm_context.getState(getEnergy = True)
    u = state.getPotentialEnergy()
    log_p.append(-u/kbT)

log_p_flow = []    
for i in range(xyz_flow.shape[0]):
    omm_context.setPositions(xyz[i])
    state = omm_context.getState(getEnergy = True)
    u = state.getPotentialEnergy()
    log_p_flow.append(-u/kbT)

fig = plt.figure(0)
fig.clf()
plt.hist(log_p, bins = 30, density = True, alpha = 0.5, label = 'log_p_md')
plt.hist(log_p_flow, bins = 30, range = [np.min(log_p) - 10, np.max(log_p) + 10], density = True, alpha = 0.5, label = 'log_p_flow')
plt.legend()
plt.savefig("./output/plots/log_p_dist.pdf")




