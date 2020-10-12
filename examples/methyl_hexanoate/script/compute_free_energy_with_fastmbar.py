__author__ = "Xinqiang Ding <xqding@umich.edu>"
__date__ = "2020/10/09 12:53:38"

import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import pickle
from FastMBAR import FastMBAR

with open("./output/energy/log_q.pkl", 'rb') as file_handle:
    data = pickle.load(file_handle)
log_q_md = data['log_q_md']
log_q_flow = data['log_q_flow']
    
with open("./output/energy/log_p.pkl", 'rb') as file_handle:
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
fastmbar = FastMBAR(energy_matrix, num_conf, verbose = True, bootstrap = True)


    
