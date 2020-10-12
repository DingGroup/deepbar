import numpy as np
import pickle
import simtk.openmm as omm
import simtk.openmm.app as app
from sys import stdout, exit
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--solvent", type = str, choices = ['OBC2', 'vacuum'], required = True)
args = parser.parse_args()

mol_id = "mobley_1017962"

## make a system with OBC2 implicit solvent
prmtop = app.AmberPrmtopFile(f"./structure/{mol_id}.prmtop")

if args.solvent == "OBC2":
    system = prmtop.createSystem(nonbondedMethod=app.NoCutoff,
                                 rigidWater=False,
                                 implicitSolvent=app.OBC2,
                                 removeCMMotion=True)    
elif args.solvent == "vacuum":
    system = prmtop.createSystem(nonbondedMethod=app.NoCutoff,
                                 rigidWater=False,
                                 implicitSolvent=None,
                                 removeCMMotion=True)    
    
inpcrd = app.AmberInpcrdFile(f"./structure/{mol_id}.inpcrd")
with open(f"./structure/{mol_id}_{args.solvent}.xml", 'w') as file_handle:
    file_handle.write(omm.XmlSerializer.serializeSystem(system))

## contruct the context
integrator = omm.LangevinIntegrator(300, 1, 0.001)
platform = omm.Platform.getPlatformByName('Reference')
context = omm.Context(system, integrator, platform)
context.setPositions(inpcrd.getPositions())
omm.LocalEnergyMinimizer_minimize(context, 0.001)

## run simulations
os.makedirs(f"./output/{args.solvent}/traj", exist_ok = True)
dcdfile_handle = open(f"./output/{args.solvent}/traj/traj_md.dcd", 'wb')
dcdfile = app.DCDFile(dcdfile_handle, prmtop.topology, 1)

num_steps = int(1e7)
save_freq = int(1e3)
num_frames = num_steps//save_freq

for k in range(num_frames):
    if (k + 1) % 100 == 0:
        print("{} output of total {} frames".format(k, num_frames), flush = True)
        
    integrator.step(save_freq)
    state = context.getState(getPositions = True)
    positions = state.getPositions()
    dcdfile.writeModel(positions)

dcdfile_handle.close()
