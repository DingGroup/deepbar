import numpy as np
import torch
torch.set_default_dtype(torch.float64)
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR, ReduceLROnPlateau
import torch.distributions as distributions
import math
from sys import exit
import sys
sys.path.append("../../")
from MMFlow import transform, MMFlow, utils
sys.path.append("./script/")
from functions import *
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
parser.add_argument("--solvent", type = str,
                    choices = ['OBC2', 'vacuum'],
                    required = True)
parser.add_argument('--num_transforms', type = int)
parser.add_argument('--hidden_size', type = int)
args = parser.parse_args()

num_transforms = args.num_transforms
hidden_size = args.hidden_size

print(f"num_transforms: {num_transforms}, hidden_size: {hidden_size}")

data = torch.load(f"./output/{args.solvent}/ic_md.pt")
ic_md = data['ic_md']
ic_md_limits = torch.load(f"./output/{args.solvent}/ic_md_limits.pt")

circular_feature_flag, feature, context, _ = ic_to_feature_and_context(ic_md, ic_md_limits)

feature_size = feature.shape[-1]
context_size = context.shape[-1]

transform_feature_flag = torch.zeros(num_transforms, feature_size)
for j in range(1):
    tmp = torch.as_tensor([i % 2 for i in range(num_transforms)])
    tmp = tmp[torch.randperm(num_transforms)]    
    transform_feature_flag[:, j] = tmp
    
for j in range(1, ic_md.angle.shape[-1] + 1):
    tmp = torch.as_tensor([i % 2 for i in range(num_transforms)])
    tmp = tmp[torch.randperm(num_transforms)]
    transform_feature_flag[:, j] = tmp
    transform_feature_flag[:, j + ic_md.angle.shape[-1]] = tmp

num_blocks = 2
conditioner_net_create_fn = lambda feature_size, context_size, output_size: \
    transform.ResidualNet(feature_size,
                          context_size,
                          output_size,
                          hidden_size = hidden_size,
                          num_blocks = num_blocks)

num_bins_regular = 4
num_bins_circular = 4
mmflow = MMFlow(feature_size, context_size,
                circular_feature_flag,
                transform_feature_flag,
                conditioner_net_create_fn,
                num_bins_circular = num_bins_circular,
                num_bins_regular = num_bins_regular)

context = context.double()
feature = feature.double()

## split data into training and validation
num_samples = feature.shape[0]
num_train_samples = int(num_samples * 0.9)

train_sample_indices = set(np.random.choice(num_samples,
                                            size = num_train_samples,
                                            replace = False))
train_sample_flag = np.array([ i in train_sample_indices for i in range(num_samples)])

feature_train = feature[train_sample_flag]
context_train = context[train_sample_flag]

feature_validation = feature[~train_sample_flag]
context_validation = context[~train_sample_flag]

batch_size = 1024*2
dataset = torch.utils.data.TensorDataset(feature_train, context_train)
dataloader = torch.utils.data.DataLoader(dataset, batch_size, shuffle = True)

if torch.cuda.is_available():
    device = torch.device('cuda')    
mmflow = mmflow.to(device)

optimizer = optim.Adam(mmflow.parameters(), lr=1e-3)
num_epoch = 300
scheduler = ReduceLROnPlateau(optimizer, 'min', factor = 0.5, patience = 5, threshold = 1e-3)

loss_record = []
for epoch in range(num_epoch):
    mmflow.train()
    start_time = time.time()
    lr = optimizer.param_groups[0]['lr']
    for i, data in enumerate(dataloader, 0):
        batch_feature, batch_context = data
        batch_feature = batch_feature.to(device)
        batch_context = batch_context.to(device)

        log_density = mmflow.compute_log_prob(batch_feature, batch_context)
        loss = -torch.mean(log_density)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()        

        if (i + 1) % 1 == 0:
            print(f"epoch: {epoch:>5}, lr: {lr:.3E}, step: {i:>4}, loss: {loss.item():.3f}", flush = True)
        if (i + 1) % 100 == 0:
            print("time used for 100 steps: {:.3f}".format(time.time() - start_time), flush = True)
            start_time = time.time()

    mmflow.eval()
    log_density_list = []
    with torch.no_grad():
        num_batches_validation = feature_validation.shape[0]//batch_size + 1
        for idx_batch in range(num_batches_validation):
            batch_feature = feature_validation[idx_batch*batch_size: (idx_batch+1)*batch_size].to(device)
            batch_context = context_validation[idx_batch*batch_size: (idx_batch+1)*batch_size].to(device)                
            log_density = mmflow.compute_log_prob(batch_feature, batch_context)
            log_density_list.append(log_density)
        log_density = torch.cat(log_density_list)
        loss = -torch.mean(log_density.cpu())
        loss_record.append(loss.item())
    print(f"epoch: {epoch:>5}, validation_loss: {loss.item():.3f}", flush = True)
    
    scheduler.step(loss.item())

    os.makedirs(f"./output/{args.solvent}/mmflow_models_hidden_size_{hidden_size}_num_transforms_{num_transforms}", exist_ok = True)
    if (epoch + 1) % 10 == 0:
        torch.save({'feature_size': feature_size,
                    'context_size': context_size,
                    'circular_feature_flag': circular_feature_flag,
                    'transform_feature_flag': transform_feature_flag,
                    'hidden_size': hidden_size,
                    'num_blocks': num_blocks,
                    'num_transforms': num_transforms,
                    'num_bins_circular': num_bins_circular,
                    'num_bins_regular': num_bins_regular,
                    'state_dict': mmflow.state_dict()},
                   f"./output/{args.solvent}/mmflow_models_hidden_size_{hidden_size}_num_transforms_{num_transforms}/mmflow_model_epoch_{epoch}.pt")
exit()
