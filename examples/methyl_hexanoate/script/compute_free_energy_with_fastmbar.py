__author__ = "Xinqiang Ding <xqding@umich.edu>"
__date__ = "2020/10/09 12:53:38"

import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import pickle
from FastMBAR import FastMBAR
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

with open(f"./output/{args.solvent}/energy/log_q_hidden_size_{hidden_size}_num_transforms_{num_transforms}.pkl", 'rb') as file_handle:
    data = pickle.load(file_handle)
log_q_md = data['log_q_md']
log_q_flow = data['log_q_flow']
    
with open(f"./output/{args.solvent}/energy/log_p_hidden_size_{hidden_size}_num_transforms_{num_transforms}.pkl", 'rb') as file_handle:
    data = pickle.load(file_handle)
log_p_md = data['log_p_md']
log_p_flow = data['log_p_flow']

assert(len(log_q_md) == len(log_p_md))
assert(len(log_q_flow) == len(log_p_flow))

energy_matrix = np.zeros((2, len(log_q_flow) + len(log_q_md)))
energy_matrix[0, 0:len(log_q_flow)] = -log_q_flow
energy_matrix[0, len(log_q_flow):] = -log_q_md

energy_matrix[1, 0:len(log_p_flow)] = -log_p_flow
energy_matrix[1, len(log_p_flow):] = -log_p_md

num_conf = np.array([len(log_q_flow), len(log_q_md)])
#fastmbar = FastMBAR(energy_matrix, num_conf, verbose = True, bootstrap = True)
fastmbar = FastMBAR(energy_matrix, num_conf, verbose = True)

    
