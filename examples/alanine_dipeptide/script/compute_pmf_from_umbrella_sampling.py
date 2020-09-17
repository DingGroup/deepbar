__author__ = "Xinqiang Ding <xqding@umich.edu>"
__date__ = "2019/10/05 02:25:36"

import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import mdtraj
import math
import simtk.unit as unit
import sys
from FastMBAR import *
from sys import exit
import pickle

K = 200
psf = mdtraj.load_prmtop("./structure/alanine_dipeptide.prmtop")
M = 25

trajs = []
for idx in range(M**2):
    print(idx)
    traj = mdtraj.load_dcd(f"./output/umbrella_sampling/traj/traj_{idx}.dcd", psf)[::5]
    trajs.append(traj)
traj = mdtraj.join(trajs)
traj.save_dcd("./output/umbrella_sampling/traj_all.dcd")

theta1 = np.linspace(-math.pi, math.pi, M, endpoint = False)
theta2 = np.linspace(-math.pi, math.pi, M, endpoint = False)

thetas = mdtraj.compute_dihedrals(traj, [[6, 8, 14, 16], [14, 8, 6, 4]])
    
T = 298.15 * unit.kelvin
kbT = unit.BOLTZMANN_CONSTANT_kB * T* unit.AVOGADRO_CONSTANT_NA
kbT_kJ_per_mol = kbT.value_in_unit(unit.kilojoule_per_mole)
kbT_kcal_per_mol = kbT.value_in_unit(unit.kilocalorie_per_mole)

energy_matrix = np.zeros((M**2, thetas.shape[0]))

for index in range(M**2):
    print(index)
    theta1_index = index // M
    theta2_index = index % M

    theta1_c = theta1[theta1_index]
    theta2_c = theta2[theta2_index]

    diff1 = np.abs(thetas[:,0] - theta1_c)
    diff1 = np.minimum(diff1, 2*math.pi-diff1)

    diff2 = np.abs(thetas[:,1] - theta2_c)
    diff2 = np.minimum(diff2, 2*math.pi-diff2)
    
    energy_matrix[index, :] = 0.5*K*(diff1**2 + diff2**2)/kbT_kJ_per_mol    

M_PMF = 20
theta1_PMF = np.linspace(-math.pi, math.pi, M_PMF, endpoint = False)
theta2_PMF = np.linspace(-math.pi, math.pi, M_PMF, endpoint = False)
width = 2*math.pi / M_PMF

energy_PMF = np.zeros((M_PMF**2, energy_matrix.shape[1]))

for index in range(M_PMF**2):
    print(index)
    i = index // M_PMF
    j = index % M_PMF
    theta1c_PMF = theta1_PMF[i]
    theta2c_PMF = theta2_PMF[j]

    theta1_low = theta1c_PMF - 0.5*width
    theta1_high = theta1c_PMF + 0.5*width

    theta2_low = theta2c_PMF - 0.5*width
    theta2_high = theta2c_PMF + 0.5*width

    indicator1 = ((thetas[:,0] > theta1_low) & (thetas[:,0] <= theta1_high)) | \
                 ((thetas[:,0] + 2*math.pi > theta1_low) & (thetas[:,0] + 2*math.pi <= theta1_high)) | \
                 ((thetas[:,0] - 2*math.pi > theta1_low) & (thetas[:,0] - 2*math.pi <= theta1_high))

    indicator2 = ((thetas[:,1] > theta2_low) & (thetas[:,1] <= theta2_high)) | \
                 ((thetas[:,1] + 2*math.pi > theta2_low) & (thetas[:,1] + 2*math.pi <= theta2_high)) | \
                 ((thetas[:,1] - 2*math.pi > theta2_low) & (thetas[:,1] - 2*math.pi <= theta2_high))
    
    indicator = indicator1 & indicator2
    
    energy_PMF[index, ~indicator] = np.inf

    
#energy_matrix = np.vstack([energy_matrix, energy_PMF])
n = energy_matrix.shape[1]//energy_matrix.shape[0]
num_conf_all = np.array([n for i in range(M**2)])
fastmbar = FastMBAR(energy = energy_matrix, num_conf = num_conf_all, cuda=True, verbose = True)

prob = np.exp(-fastmbar.log_prob_mix)
tmp = thetas/np.pi*180
flag_0 = (tmp[:, 1] > 0) & (tmp[:, 1] <= 120)
flag_1 = ~ flag_0

dF = -np.log(np.sum(prob[flag_1])/np.sum(prob[flag_0]))

with open("./output/umbrella_sampling/fastmbar_result.pkl", 'wb') as file_handle:
    pickle.dump({'F': fastmbar.F, 'log_prob_mix': fastmbar.log_prob_mix,
                 'prob': prob, 'flag_0': flag_0,
                 'flag_1': flag_1, 'dF': dF}, file_handle)

PMF, _ = fastmbar.calculate_free_energies_of_perturbed_states(energy_PMF)
with open("./output/umbrella_sampling/PMF.pkl", 'wb') as file_handle:
    pickle.dump(PMF, file_handle)

with open("./output/umbrella_sampling/PMF.pkl", 'rb') as file_handle:
    PMF = pickle.load(file_handle)

# fig = plt.figure(0)
# fig.clf()
# plt.imshow(np.flipud(PMF.reshape((M_PMF, M_PMF))), extent = (-180, 180, -180, 180))
# plt.colorbar()
# plt.savefig("./output/umbrella_sampling/PMF_fast_mbar.pdf")

PMF_expanded = np.zeros((M_PMF+1, M_PMF+1))
PMF_expanded[0:M_PMF,0:M_PMF] = PMF.reshape((M_PMF, M_PMF))
PMF_expanded[0:-1,-1] = PMF.reshape((M_PMF, M_PMF))[:,0]
PMF_expanded[-1,:] = PMF_expanded[0,:]
theta1_PMF_expanded = np.array(list(theta1_PMF) + [math.pi])
theta2_PMF_expanded = np.array(list(theta2_PMF) + [math.pi])

fig = plt.figure(2)
fig.clf()
plt.contourf(theta1_PMF_expanded/math.pi*180, theta2_PMF_expanded/math.pi*180, (PMF_expanded - np.min(PMF_expanded))*kbT_kcal_per_mol , levels = 40, extent = (-180, 180, -180, 180))
plt.colorbar()
plt.xticks(np.arange(-150,150+1, 50))
plt.yticks(np.arange(-150,150+1, 50))
plt.xlabel(r"$\phi$ (degree)")
plt.ylabel(r"$\psi$ (degree)")
plt.xlim(-180, 180)
plt.ylim(-180, 180)
plt.vlines(0, -180, 180, linestyles = 'dotted')
plt.vlines(120, -180, 180, linestyles = 'dotted')
plt.tight_layout()
plt.savefig("./output/umbrella_sampling/PMF_contourf.eps")
