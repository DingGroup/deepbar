#!/home/xqding/apps/miniconda3/bin/python

# Created by Xinqiang Ding (xqding@umich.edu)
# at 2019/10/05 01:27:20

#SBATCH --job-name=energy
#SBATCH --time=00:20:00
#SBATCH --partition=sixteencore
#SBATCH --cpus-per-task=2
#SBATCH --array=0-624
#SBATCH --output=./slurm_output/energy_%a.log
#SBATCH --open-mode=truncate

import simtk.openmm.app  as omm_app
import simtk.openmm as omm
import simtk.unit as unit
import math
import mdtraj
import os
import numpy as np
import pickle
from sys import exit

## read system
with open("./structure/system.xml", 'r') as file_handle:
    xml = file_handle.read()    
system = omm.XmlSerializer.deserialize(xml)

## setup context
platform = omm.Platform.getPlatformByName('Reference')
T = 298.15 * unit.kelvin
fricCoef = 10/unit.picoseconds
stepsize = 1 * unit.femtoseconds
integrator = omm.LangevinIntegrator(T, fricCoef, stepsize)
context = omm.Context(system, integrator, platform)

job_index = int(os.environ['SLURM_ARRAY_TASK_ID'])
topology = mdtraj.load_prmtop("./structure/alanine_dipeptide.prmtop")

traj = mdtraj.load_dcd(f"./output/umbrella_sampling/traj/traj_{job_index}.dcd", topology)
xyz = traj.xyz

potential_energy = []
for i in range(xyz.shape[0]):
    if (i + 1) % 100 == 0:
        print(i)
    context.setPositions(xyz[i])
    state = context.getState(getEnergy = True)
    
    energy = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    potential_energy.append(energy)

potential_energy = np.array(potential_energy)

with open(f"./output/umbrella_sampling/traj/potential_energy_kJ_per_mole_{job_index}.pkl", 'wb') as file_handle:
    pickle.dump(potential_energy, file_handle)
    

