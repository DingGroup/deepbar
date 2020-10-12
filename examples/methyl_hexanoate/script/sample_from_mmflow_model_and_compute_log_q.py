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
sys.path.append("../../")
from MMFlow import transform, MMFlow, utils
sys.path.append("./script/")
from functions import *
from sys import exit
import matplotlib as mpl
mpl.use("Agg")
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--solvent", type = str,
                    choices = ['OBC2', 'vacuum'],
                    required = True)
parser.add_argument('--num_transforms', type = int)
parser.add_argument('--hidden_size', type = int)
args = parser.parse_args()

num_transforms = args.num_transforms
hidden_size = args.hidden_size

print(f"num_transforms: {num_transforms}, hidden_size: {hidden_size}")
mol_id = "mobley_1017962"

## load mmflow model
epoch = 499
data = torch.load(f"./output/{args.solvent}/mmflow_models_hidden_size_{hidden_size}_num_transforms_{num_transforms}/mmflow_model_epoch_{epoch}.pt")

conditioner_net_create_fn = lambda feature_size, context_size, output_size: \
    transform.ResidualNet(feature_size,
                          context_size,
                          output_size,
                          hidden_size = data['hidden_size'],
                          num_blocks = data['num_blocks'])

mmflow = MMFlow(data['feature_size'],
                data['context_size'],
                data['circular_feature_flag'],
                data['transform_feature_flag'],
                conditioner_net_create_fn,
                num_bins_circular = data['num_bins_circular'],
                num_bins_regular = data['num_bins_regular'])

mmflow.load_state_dict(data['state_dict'])

if torch.cuda.is_available():
    device = torch.device('cuda')    
mmflow = mmflow.to(device)

## sample from mmflow model
data = torch.load(f"./output/{args.solvent}/ic_md.pt")
ic_md = data['ic_md']
ic_md_logabsdet = data['ic_md_logabsdet']
ic_md_limits = torch.load(f"./output/{args.solvent}/ic_md_limits.pt")

with open(f"./output/{args.solvent}/coordinate_transformer.pkl", 'rb') as file_handle:
    coor_transformer = pickle.load(file_handle)
circular_feature_flag, feature_md, context_md, logabsdet_jacobian = ic_to_feature_and_context(ic_md, ic_md_limits)

log_q_md = ic_md_logabsdet + logabsdet_jacobian

feature_size = feature_md.shape[-1]
context_size = context_md.shape[-1]

feature_md = feature_md.double()
context_md = context_md.double()

context_mean = torch.mean(context_md, 0)
context_centered = context_md - context_mean[None, :]
context_cov = torch.matmul(context_centered.T, context_centered) / context_md.shape[0]
context_dist = distributions.MultivariateNormal(loc = context_mean,
                                                covariance_matrix = context_cov)


num_samples = 10_000
context_flow = context_dist.sample((num_samples,))
log_q_context_flow = context_dist.log_prob(context_flow)
log_q_context_md = context_dist.log_prob(context_md)

batch_size = 1024

feature_flow_list = []
log_q_feature_flow = []
for idx_batch in range(num_samples//batch_size + 1):
    print(f"idx_batch: {idx_batch}")
    with torch.no_grad():
        batch_context_flow = context_flow[idx_batch*batch_size:(idx_batch+1)*batch_size]
        feature_flow, log_prob = mmflow.sample_and_compute_log_prob(1, batch_context_flow.to(device))        
    feature_flow = torch.squeeze(feature_flow)
    feature_flow_list.append(feature_flow.cpu())
    log_prob = torch.squeeze(log_prob)
    log_q_feature_flow.append(log_prob.cpu())
    
feature_flow = torch.cat(feature_flow_list, 0)
log_q_feature_flow = torch.cat(log_q_feature_flow, 0)
log_q_flow = log_q_context_flow + log_q_context_flow

log_q_feature_md = []
for idx_batch in range(feature_md.shape[0]//batch_size + 1):
    print(f"idx_batch: {idx_batch}")
    with torch.no_grad():
        batch_context_md = context_md[idx_batch*batch_size:(idx_batch+1)*batch_size].to(device)
        batch_feature_md = feature_md[idx_batch*batch_size:(idx_batch+1)*batch_size].to(device)
        log_prob = mmflow.compute_log_prob(batch_feature_md, batch_context_md)
    log_prob = torch.squeeze(log_prob)
    log_q_feature_md.append(log_prob.cpu())
    
log_q_feature_md = torch.cat(log_q_feature_md, 0)
log_q_md = log_q_md + log_q_context_md + log_q_context_md


os.makedirs(f"./output/{args.solvent}/plots", exist_ok = True)
with PdfPages(f"./output/{args.solvent}/plots/feature_dist_hidden_size_{hidden_size}_num_transforms_{num_transforms}.pdf") as pdf:
    for j in range(feature_md.shape[-1]):
        print(j)
        fig = plt.figure(0)
        fig.clf()
        if circular_feature_flag[j] > 0:
            plt.hist(feature_md[:, j].numpy(), bins = 30, range = [-math.pi, math.pi], density = True, alpha = 0.5, label = 'md')
            plt.hist(feature_flow[:, j].numpy(), bins = 30, range = [-math.pi, math.pi], density = True, alpha = 0.5, label = 'mmflow')            
        else:
            plt.hist(feature_md[:, j].numpy(), bins = 30, range = [0.0, 1.0], density = True, alpha = 0.5, label = 'md')
            plt.hist(feature_flow[:, j].numpy(), bins = 30, range = [0.0, 1.0], density = True, alpha = 0.5, label = 'mmflow')
        plt.legend()
        pdf.savefig(fig)

ic_flow, logabsdet_jacobian = feature_and_context_to_ic(feature_flow, context_flow, ic_md_limits)
log_q_flow = log_q_flow - logabsdet_jacobian

guest_xyz_flow, logabsdet_jacobian = coor_transformer.compute_xyz_from_internal_coordinate(
    ic_flow.reference_particle_1_xyz,
    ic_flow.reference_particle_2_bond,
    ic_flow.reference_particle_3_bond,
    ic_flow.reference_particle_3_angle,
    ic_flow.bond,
    ic_flow.angle,
    ic_flow.dihedral
)
log_q_flow = log_q_flow - logabsdet_jacobian

topology = mdtraj.load_prmtop(f"./structure/{mol_id}.prmtop")
traj_flow = mdtraj.Trajectory(guest_xyz_flow.numpy(), topology)
traj_flow.save_dcd(f"./output/{args.solvent}/traj/traj_flow_hidden_size_{hidden_size}_num_transforms_{num_transforms}.dcd")


fig = plt.figure(1)
fig.clf()
plt.hist(log_q_md, bins = 30, label = "log_q_md", alpha = 0.5, density = True)
plt.hist(log_q_flow, bins = 30, label = "log_q_flow", alpha = 0.5, density = True)
plt.legend()
plt.tight_layout()
os.makedirs(f"./output/{args.solvent}/plots", exist_ok = True)
plt.savefig(f"./output/{args.solvent}/plots/log_q_hist_hidden_size_{hidden_size}_num_transforms_{num_transforms}.pdf")

os.makedirs(f"./output/{args.solvent}/energy", exist_ok = True)
with open(f"./output/{args.solvent}/energy/log_q_hidden_size_{hidden_size}_num_transforms_{num_transforms}.pkl", 'wb') as file_handle:
    pickle.dump({'log_q_md': log_q_md, 'log_q_flow': log_q_flow}, file_handle)
    
exit()
