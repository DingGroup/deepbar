#!/home/xqding/apps/miniconda3/bin/python

# Created by Xinqiang Ding (xqding@umich.edu)
# at 2019/11/26 16:13:12

#SBATCH --job-name=learn
#SBATCH --time=1-00:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --open-mode=truncate
#SBATCH --exclude=gollum[003-045]
#SBATCH --array=0-4
#SBATCH --output=./slurm_output/learn_%A_%a.txt

import numpy as np
import torch
torch.set_default_dtype(torch.float64)
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
import torch.distributions as distributions
import math
from sys import exit
import sys
sys.path.append("/home/xqding/course/projectsOnGitHub/MDTorch")
import MDTorch.utils as utils
sys.path.append("./script/")
from RealNVP import *
from make_masks import *
import pickle
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import time
import os
import simtk.openmm as omm
import simtk.openmm.app as app
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--idx_state", type = int)
parser.add_argument("--idx_mask", type = int)
parser.add_argument("--hidden_dim", type = int)

args = parser.parse_args()
idx_state = args.idx_state
idx_mask = args.idx_mask
hidden_dim = args.hidden_dim

#idx_repeat = int(os.environ['SLURM_ARRAY_TASK_ID'])
idx_repeat = 0
reference_p_1, reference_p_2, reference_p_3 = 1, 0, 2
ref_particle_name = "{}-{}-{}".format(reference_p_1,
                                      reference_p_2,
                                      reference_p_3)

print("idx_mask: {}".format(idx_mask))
print("ref_particle_name: {}".format(ref_particle_name))

with open("./output/internal_coor_{}/internal_coor.pkl".format(ref_particle_name), 'rb') as file_handle:
    data = pickle.load(file_handle)
    internal_coor = data['internal_coor']
    dihedral_limits = data['dihedral_limits']
    prob = data['prob']
    state_0_flag = data['state_0_flag']
    state_1_flag = data['state_1_flag']

low = [l[0] for l in dihedral_limits]
high = [l[1] for l in dihedral_limits]
    
if idx_state == 0:
    state_flag = state_0_flag
elif idx_state == 1:
    state_flag = state_1_flag
else:
    raise ValueError("idx_state has to be 0 or 1.")

dihedral = internal_coor['dihedral'][state_flag]
dihedral = dihedral.astype(np.float64)
prob = prob[state_flag]
        
num_samples = dihedral.shape[0]
num_validation_samples = int(num_samples * 0.2)
validation_sample_index = np.random.choice(range(num_samples), num_validation_samples,
                                      replace = False, p = prob / np.sum(prob))

os.makedirs("./output/learn_with_data_{}/model_masks_{}_hidden_dim_{}".format(ref_particle_name, idx_mask, hidden_dim), exist_ok = True)
validation_sample_flag = np.array([i in validation_sample_index for i in range(num_samples)])
train_sample_flag = ~validation_sample_flag

dihedral_train = dihedral[train_sample_flag]
dihedral_validation = dihedral[validation_sample_flag]

prob_train = prob[train_sample_flag] / np.sum(prob[train_sample_flag])
prob_validation = prob[validation_sample_flag] / np.sum(prob[validation_sample_flag])

## make RealNVP model and start training
with open("./output/internal_coor_{}/coor_transformer.pkl".format(ref_particle_name), 'rb') as file_handle:
    coor_transformer = pickle.load(file_handle)
    
input_dim = dihedral.shape[1]
if idx_mask == 0:
    masks = make_masks_0(input_dim, low, high)
elif idx_mask == 1:
    masks = make_masks_1(input_dim, low, high)
elif idx_mask == 2:
    masks = make_masks_2(input_dim, low, high)

realNVP = RealNVP(masks, hidden_dim)
realNVP = realNVP.cuda()

optimizer = optim.Adam(realNVP.parameters(), lr = 0.001)
#scheduler = MultiStepLR(optimizer, [1000], gamma=0.1, last_epoch=-1)

X_train = torch.from_numpy(dihedral_train)
X_train = X_train.cuda()

X_validation = torch.from_numpy(dihedral_validation)
X_validation = X_validation.cuda()
prob_validation = torch.from_numpy(prob_validation)
prob_validation = prob_validation.cuda()

normal_dist = distributions.Normal(loc = X_train.new_tensor([0.0]), scale = X_train.new_tensor([1.0]))

saved_num_steps = np.logspace(3, 4.7, 100)
saved_num_steps = list(saved_num_steps.astype(np.int32))
num_steps = np.max(saved_num_steps)

loss_record = []
batch_size = 20480

exit()

start_time = time.time()
for idx_step in range(num_steps + 1):
    ## sample based on weights
    indices = np.random.choice(range(X_train.shape[0]), size = batch_size, replace = True, p = prob_train)
    batch_X = X_train[indices]
    
    ## transform data X to latent space Z
    z, logdet = realNVP.inverse(batch_X)
    
    ## calculate the negative loglikelihood of X
    loss = -torch.mean(torch.sum(normal_dist.log_prob(z), -1) + logdet)
        
    if loss.item() == np.inf:
        print("trained stopped, because loss is inf")
        exit()
        
    if np.isnan(loss.item()):
        print("trained stopped, because loss is nan")
        exit()
        
    optimizer.zero_grad()
    loss.backward()
    
    #torch.nn.utils.clip_grad_norm_(realNVP.parameters(), 10^3)
    optimizer.step()
#    scheduler.step()
    loss_record.append(loss.item())
    
    if idx_step % 10 == 0:
        lr = optimizer.param_groups[0]['lr']
        print("idx_steps: {:}, lr: {:.5f}, loss: {:.5f}".format(idx_step, lr, loss.item()), flush = True)

    if idx_step % 100 == 0:
        print("time used for 100 steps: {:.3f}".format(time.time() - start_time))
        start_time = time.time()

    os.makedirs("./output/learn_with_data_{}/model_masks_{}_hidden_dim_{}".format(ref_particle_name, idx_mask, hidden_dim), exist_ok = True)
    if idx_step in saved_num_steps:
        with torch.no_grad():
            z, logdet = realNVP.inverse(X_validation)
            loss = -(torch.sum(normal_dist.log_prob(z), -1) + logdet)
            loss = loss * prob_validation
            loss = torch.sum(loss)
            
        print("model saved at step: {} with validation_loss: {:.5f}".format(idx_step, loss.item()), flush = True)
        torch.save({'state_dict': realNVP.state_dict(),
                    'hidden_dim': hidden_dim,
                    'masks': masks,
                    'input_dim': input_dim,
                    'loss_record': loss_record,
                    'loss_validation': loss.item(),
                    'train_sample_flag': train_sample_flag,
                    'validation_sample_flag': validation_sample_flag},
                   "./output/learn_with_data_{}/model_masks_{}_hidden_dim_{}/realNVP_state_{}_repeat_{}_step_{}.pt".format(ref_particle_name,
                                                                                                                           idx_mask, hidden_dim,
                                                                                                                           idx_state, idx_repeat, idx_step))
