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
sys.path.append("/home/xqding/course/projectsOnGitHub/MMFlow")
from MMFlow import transform, MMFlow, utils
sys.path.append("./script/")
#from make_masks import *
import pickle
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import time
import os
import simtk.openmm as omm
import simtk.openmm.app as app
import argparse

with open("./output/internal_coor/internal_coor.pkl", 'rb') as file_handle:
    data = pickle.load(file_handle)
    ic = data['ic']
    prob = data['prob']

ic.angle /= math.pi
ic.reference_particle_2_polar_angle /= math.pi
ic.reference_particle_3_polar_angle /= math.pi

feature = torch.cat([ic.reference_particle_3_polar_angle[:, None],
                     ic.angle, ic.dihedral], -1)
feature_size = feature.shape[-1]
circular_feature_flag = torch.zeros(feature_size)
circular_feature_flag[-ic.dihedral.shape[-1]:] = 1

context = torch.cat([ic.reference_particle_2_bond[:, None],
                     ic.reference_particle_3_bond[:, None],
                     ic.bond], -1)
context_size = context.shape[-1]

num_transforms = 2
transform_feature_flag_list = []
stride = ic.angle.shape[-1]
for _ in range(num_transforms):
    transform_feature_flag = torch.zeros(feature_size)
    transform_feature_flag[0] = np.random.random() > 0.5
    for i in range(stride):
        if np.random.random() > 0.5:
            transform_feature_flag[i+1::stride] = 1.0
    transform_feature_flag_list.append(transform_feature_flag)
    
conditioner_net_create_fn = lambda feature_size, context_size, output_size: \
    transform.ResidualNet(feature_size,
                          context_size,
                          output_size,
                          hidden_size = 16,
                          num_blocks = 2)

mmflow = MMFlow(feature_size, context_size,
                circular_feature_flag, transform_feature_flag_list,
                conditioner_net_create_fn,
                num_bins_circular = 4, num_bins_regular = 4)

context = context.double()
feature = feature.double()

## split data into training and validation
num_samples = feature.shape[0]
num_train_samples = int(num_samples * 0.9)

train_sample_indices = set(np.random.choice(num_samples,
                                            size = num_train_samples,
                                            replace = False,
                                            p = prob / np.sum(prob)))
train_sample_flag = np.array([ i in train_sample_indices for i in range(num_samples)])

feature_train = feature[train_sample_flag]
context_train = context[train_sample_flag]

feature_validation = feature[~train_sample_flag]
context_validation = context[~train_sample_flag]

if torch.cuda.is_available():
    device = torch.device('cuda')    
mmflow = mmflow.to(device)

optimizer = optim.Adam(mmflow.parameters(), lr=1e-3)
batch_size = 2048
start_time = time.time()
loss_record = []
saved_num_steps = np.logspace(2, 4.5, 30)
saved_num_steps = list(saved_num_steps.astype(np.int32))
num_steps = np.max(saved_num_steps)

for idx_step in range(num_steps + 1):
    mmflow.train()
    optimizer.zero_grad()

    indices = np.random.choice(feature_train.shape[0], size = batch_size, replace = False)

    batch_feature = feature_train[indices].to(device)
    batch_context = context_train[indices].to(device)

    #torch.autograd.set_detect_anomaly(True)    
    log_density = mmflow.compute_log_prob(batch_feature, batch_context)
    loss = -torch.mean(log_density)
    loss.backward()
    optimizer.step()
    loss_record.append(loss.item())

    if (idx_step + 1) % 10 == 0:
        lr = optimizer.param_groups[0]['lr']
        print("idx_steps: {:}, lr: {:.5f}, loss: {:.5f}".format(idx_step, lr, loss.item()), flush = True)

    if (idx_step + 1) % 100 == 0:
        print("time used for 100 steps: {:.3f}".format(time.time() - start_time))
        start_time = time.time()

    os.makedirs(f"./output/mmflow_models", exist_ok = True)

    if idx_step in saved_num_steps:
        flow.eval()
        log_density_list = []
        with torch.no_grad():
            num_batches_validation = feature_validation.shape[0]//batch_size + 1
            for idx_batch in range(num_batches_validation):
                batch_feature = feature_validation[idx_batch*batch_size: (idx_batch+1)*batch_size].to(device)
                batch_context = context_validation[idx_batch*batch_size: (idx_batch+1)*batch_size].to(device)
                
                log_density = mmflow.log_prob(batch_feature, batch_context)
                log_density_list.append(log_density)
            log_density = torch.cat(log_density_list)
            loss = -torch.mean(log_density)

        print("model saved at step: {} with loss: {:.5f}".format(idx_step, loss.item()), flush = True)
        torch.save({'model_param_dict': model_param_dict,
                    'transform': transform,
                    'state_dict': flow.state_dict(),
                    'train_sample_flag': train_sample_flag,
                    'loss_record': loss_record,
                    'loss_validation': loss.item()},
                   f"./output/{name}/nflow_models/nflow_model_{idx_step}.pt")
        loss_validation = loss.item()
