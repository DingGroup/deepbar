__author__ = "Xinqiang Ding <xqding@umich.edu>"
__date__ = "2020/05/16 23:13:43"

import numpy as np
import pickle
import argparse
import simtk.unit as unit
from FastMBAR import *
import torch
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from sys import exit
from collections import  defaultdict

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

saved_num_steps = np.logspace(3, 4.7, 100)
saved_num_steps = list(saved_num_steps.astype(np.int32))

F_fastmbar = defaultdict(list)
F_DKL_pq = defaultdict(list)
F_DKL_qp = defaultdict(list)

validation_loss_dict = {}
F_fastmbar = []
F_DKL_pq = []
F_DKL_qp = []

for idx_repeat in range(5):
    validation_loss = []
    for num_step in saved_num_steps:
        data = torch.load("./output/learn_with_data_{}/model_masks_{}_hidden_dim_{}/realNVP_state_{}_repeat_{}_step_{}.pt".format(ref_particle_name,
                                                                                                                                  idx_mask, hidden_dim,
                                                                                                                                  idx_state, idx_repeat,
                                                                                                                                  num_step)) 
        validation_loss.append(data['loss_validation'])
    validation_loss_dict[idx_repeat] = validation_loss

    idx_num_step = np.argmin(validation_loss)
    print(idx_num_step)
    num_steps = saved_num_steps[idx_num_step]

    with open("./output/DKL_{}_hidden_dim_{}/result_mask_{}_state_{}_repeat_{}_steps_{}.pkl".format(
            ref_particle_name, hidden_dim, idx_mask, idx_state, idx_repeat, num_steps), 'rb') as file_handle:
        data = pickle.load(file_handle)
    
    n = data['pq_logq'].shape[0]
    pq_logq = data['pq_logq'][np.random.choice(range(n), n, p = data['prob'])].numpy()

    n = data['pq_logp'].shape[0]
    pq_logp = data['pq_logp'][np.random.choice(range(n), n, p = data['prob'])]

    n = data['qp_logq'].shape[0]
    qp_logq = data['qp_logq'].numpy()

    n = data['qp_logp'].shape[0]
    qp_logp = data['qp_logp']

    logq = np.concatenate((qp_logq, pq_logq))
    logp = np.concatenate((qp_logp, pq_logp))

    energy_matrix = -np.vstack((logq, logp))
    num_conf = np.array([len(qp_logq), len(pq_logq)])
    fastmbar = FastMBAR(energy_matrix, num_conf, verbose = True)
    
    F_fastmbar.append(fastmbar.F[-1])
    F_DKL_pq.append(np.mean(pq_logp - pq_logq))
    flag = np.abs(qp_logp) < 1000
    F_DKL_qp.append(np.mean(qp_logq[flag] - qp_logp[flag]))

    # F_fastmbar[idx_temperature].append(F_fastmbar_tmp)
    # F_DKL_pq[idx_temperature].append(F_DKL_pq_tmp)
    # F_DKL_qp[idx_temperature].append(F_DKL_qp_tmp)
            
    # kbT = unit.BOLTZMANN_CONSTANT_kB * unit.AVOGADRO_CONSTANT_NA * temperatures * unit.kelvin
    # kbT_kcal_per_mol = kbT[0].value_in_unit(unit.kilocalorie_per_mole)
        
    # F_fastmbar[idx_temperature] = np.array(F_fastmbar[idx_temperature])*kbT_kcal_per_mol
    # F_DKL_pq[idx_temperature] = np.array(F_DKL_pq[idx_temperature])*kbT_kcal_per_mol
    # F_DKL_qp[idx_temperature] = np.array(F_DKL_qp[idx_temperature])*kbT_kcal_per_mol    

exit()    

for idx_temperature in [0, 11]:    
    fig = plt.figure(0)
    fig.clf()
    plt.errorbar(saved_num_steps, np.mean(F_fastmbar[idx_temperature], -1), yerr = np.std(F_fastmbar[idx_temperature], -1))
    plt.errorbar(saved_num_steps, np.mean(-F_DKL_pq[idx_temperature], -1), yerr = np.std(-F_DKL_pq[idx_temperature], -1))
    
    # plt.plot(saved_num_steps, F_fastmbar[-1], "-o")
    # plt.plot(saved_num_steps, -F_DKL_pq[-1], "-o")
    plt.xscale("log")
    plt.savefig("./output/DKL_{}_hidden_dim_{}/temperature_{:.2f}.pdf".format(ref_particle_name, hidden_dim, temperatures[idx_temperature]))

dF_fastmbar = F_fastmbar[11] - F_fastmbar[0]
dF_DKL_pq = (-F_DKL_pq[11] - (-F_DKL_pq[0]))
dF_DKL_qp = F_DKL_qp[11] - F_DKL_pq[0]

with open("./output/F_fastmbar.pkl", 'rb') as file_handle:
    F = pickle.load(file_handle)
dF_umbrella_sampling = F[-1]

fig = plt.figure(1)
fig.clf()
plt.errorbar(saved_num_steps, np.mean(dF_fastmbar, -1), yerr = np.std(dF_fastmbar, -1), label = "BAR")
plt.errorbar(saved_num_steps, np.mean(dF_DKL_pq, -1), yerr = np.std(dF_DKL_pq, -1), label = "DKL_pq")
plt.axhline(dF_umbrella_sampling, color = "black", label = "Ref")

# plt.plot(saved_num_steps, dF_fastmbar, "-o", )
# plt.plot(saved_num_steps, dF_DKL_pq, "-o", label = "DKL_pq")

plt.xscale("log")
plt.ylim(-40, -20)
plt.legend()
plt.savefig("./output/DKL_{}_hidden_dim_{}/dF.pdf".format(ref_particle_name, hidden_dim))
