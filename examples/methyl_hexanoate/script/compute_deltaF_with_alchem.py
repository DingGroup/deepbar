import numpy as np
from FastMBAR import FastMBAR
import pickle
import os
import mdtraj
from sys import exit
import simtk.openmm as omm
import simtk.openmm.app as app
import simtk.unit as unit

mol_id = "mobley_1017962"

## openmm context
## read system
with open(f"./structure/{mol_id}_vacuum.xml", 'r') as file_handle:
    xml = file_handle.read()
system_vacuum = omm.XmlSerializer.deserialize(xml)

with open(f"./structure/{mol_id}_OBC2.xml", 'r') as file_handle:
    xml = file_handle.read()
system_OBC2 = omm.XmlSerializer.deserialize(xml)

## setup context
T = 300 * unit.kelvin
integrator_vacuum = omm.LangevinIntegrator(T,
                                    1/unit.picosecond,
                                    1*unit.femtosecond)
platform_vacuum = omm.Platform.getPlatformByName("Reference")
omm_context_vacuum = omm.Context(system_vacuum, integrator_vacuum, platform_vacuum)

integrator_OBC2 = omm.LangevinIntegrator(T,
                                    1/unit.picosecond,
                                    1*unit.femtosecond)
platform_OBC2 = omm.Platform.getPlatformByName("Reference")
omm_context_OBC2 = omm.Context(system_OBC2, integrator_OBC2, platform_OBC2)

kbT = unit.BOLTZMANN_CONSTANT_kB * T * unit.AVOGADRO_CONSTANT_NA

topology = mdtraj.load_prmtop(f"./structure/{mol_id}.prmtop")
traj_vacuum = mdtraj.load_dcd(f"./output/vacuum/traj/traj_md.dcd",
                       top = topology, stride = 1)
traj_OBC2 = mdtraj.load_dcd(f"./output/OBC2/traj/traj_md.dcd",
                       top = topology, stride = 1)



traj = traj_vacuum + traj_OBC2
xyz = traj.xyz

energy_vacuum = []
energy_OBC2 = []
for i in range(xyz.shape[0]):
    if (i + 1) % 100 == 0:
        print(i)
        
    omm_context_vacuum.setPositions(xyz[i])
    state = omm_context_vacuum.getState(getEnergy = True)
    u = state.getPotentialEnergy()
    energy_vacuum.append(u/kbT)

    omm_context_OBC2.setPositions(xyz[i])
    state = omm_context_OBC2.getState(getEnergy = True)
    u = state.getPotentialEnergy()
    energy_OBC2.append(u/kbT)
    
energy_matrix = np.array([energy_vacuum, energy_OBC2])
num_conf = np.array([len(traj_vacuum), len(traj_OBC2)])
fastmbar = FastMBAR(energy_matrix, num_conf, verbose = True)

