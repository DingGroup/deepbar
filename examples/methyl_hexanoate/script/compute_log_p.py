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

mol_id = "mobley_1017962"

## openmm context
## read system
with open(f"./structure/{mol_id}.xml", 'r') as file_handle:
    xml = file_handle.read()
system = omm.XmlSerializer.deserialize(xml)

## setup context
T = 300 * unit.kelvin
integrator = omm.LangevinIntegrator(T,
                                    1/unit.picosecond,
                                    1*unit.femtosecond)
platform = omm.Platform.getPlatformByName("Reference")
omm_context = omm.Context(system, integrator, platform)
kbT = unit.BOLTZMANN_CONSTANT_kB * T * unit.AVOGADRO_CONSTANT_NA

topology = mdtraj.load_prmtop(f"./structure/{mol_id}.prmtop")
traj = mdtraj.load_dcd(f"./output/{args.solvent}/traj/traj_md.dcd",
                       top = topology, stride = 1)
xyz = traj.xyz

log_p_md = []
for i in range(xyz.shape[0]):
    if (i + 1) % 100 == 0:
        print(i)        
    omm_context.setPositions(xyz[i])
    state = omm_context.getState(getEnergy = True)
    u = state.getPotentialEnergy()
    log_p_md.append(-u/kbT)
log_p_md = np.array(log_p_md)

traj = mdtraj.load_dcd(f"./output/{args.solvent}/traj/traj_flow_hidden_size_{hidden_size}_num_transforms_{num_transforms}.dcd",
                       top = topology, stride = 1)
xyz_flow = traj.xyz
log_p_flow = []    
for i in range(xyz_flow.shape[0]):
    if (i + 1) % 100 == 0:
        print(i)           
    omm_context.setPositions(xyz_flow[i])
    state = omm_context.getState(getEnergy = True)
    u = state.getPotentialEnergy()
    log_p_flow.append(-u/kbT)
log_p_flow = np.array(log_p_flow)
    
fig = plt.figure(0)
fig.clf()
plt.hist(log_p_md, bins = 30, density = True, alpha = 0.5, label = 'log_p_md')
plt.hist(log_p_flow, bins = 30, range = [np.min(log_p_md) - 10, np.max(log_p_md) + 10], density = True, alpha = 0.5, label = 'log_p_flow')
plt.legend()
plt.savefig(f"./output/{args.solvent}/plots/log_p_dist_hidden_size_{hidden_size}_num_transforms_{num_transforms}.pdf")

with open(f"./output/{args.solvent}/energy/log_p.pkl", 'wb') as file_handle:
    pickle.dump({'log_p_md': log_p_md, 'log_p_flow': log_p_flow}, file_handle)
    

exit()



