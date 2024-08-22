import openmm as omm
import openmm.app as app
import os
import argparse
from sys import exit
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--solvent", type=str, choices=["OBC2", "vacuum"], required=True)
args = parser.parse_args()

name = "methyl_hexanoate"

## make a system with OBC2 implicit solvent
prmtop = app.AmberPrmtopFile(f"./structure/{name}.prmtop")

if args.solvent == "OBC2":
    system = prmtop.createSystem(
        nonbondedMethod=app.NoCutoff,
        rigidWater=False,
        implicitSolvent=app.OBC2,
        removeCMMotion=True,
    )
elif args.solvent == "vacuum":
    system = prmtop.createSystem(
        nonbondedMethod=app.NoCutoff,
        rigidWater=False,
        implicitSolvent=None,
        removeCMMotion=True,
    )

inpcrd = app.AmberInpcrdFile(f"./structure/{name}.inpcrd")
with open(f"./structure/{name}_{args.solvent}.xml", "w") as file_handle:
    file_handle.write(omm.XmlSerializer.serializeSystem(system))

## contruct the simulation
integrator = omm.LangevinMiddleIntegrator(300, 1, 0.001)
platform = omm.Platform.getPlatformByName("Reference")
simulation = app.Simulation(prmtop.topology, system, integrator, platform)
simulation.context.setPositions(inpcrd.getPositions())

## minimize the energy
simulation.minimizeEnergy(tolerance=0.001)

## run simulations
os.makedirs(f"./output/{args.solvent}/traj", exist_ok=True)

simulation.reporters.append(
    app.DCDReporter(f"./output/{args.solvent}/traj/traj_md.dcd", 1000)
)

for i in tqdm(range(100)):
    simulation.step(100_000)

exit()
