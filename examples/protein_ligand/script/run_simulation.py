#!/cluster/tufts/dinglab/shared_apps/miniconda3/envs/xd/bin/python

#SBATCH --job-name=simulation
#SBATCH --partition=dinglab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:rtx_a5000:1
#SBATCH --time=2-00:00:00
#SBATCH --output=./log/simulation.log

import openmm as mm
import openmm.app as app
import openmm.unit as unit
import numpy as np
import mdtraj
import pickle
from sys import exit
import time

prmtop = app.AmberPrmtopFile("./structure/complex.prmtop")
inpcrd = app.AmberInpcrdFile("./structure/complex.inpcrd")

system = prmtop.createSystem(
    nonbondedMethod=app.CutoffNonPeriodic,
    nonbondedCutoff=1.0 * unit.nanometers,
    constraints=None,
    implicitSolvent=app.OBC2,
    implicitSolventSaltConc=0.1 * unit.molar,
)

top = mdtraj.load_prmtop("./structure/complex.prmtop")
xyz = np.array(inpcrd.positions.value_in_unit(unit.nanometers))


def compute_pairwise_distances(xyz, atom_indices_1, atom_indices_2):
    distances = np.zeros((len(atom_indices_1), len(atom_indices_2)))
    for i, atom_index_1 in enumerate(atom_indices_1):
        for j, atom_index_2 in enumerate(atom_indices_2):
            distances[i, j] = np.linalg.norm(xyz[atom_index_1] - xyz[atom_index_2])
    return distances


## ligand heavy atoms
lig_atom_indices = top.select("resname MOL and not element H")

## protein heavy atoms by residue
res_in_contact = []
protein_atom_in_contact = []
for res in top.residues:
    if res.name == "MOL":
        continue
    res_atom_indices = top.select(f"resid {res.index} and not element H")

    distances = compute_pairwise_distances(xyz, lig_atom_indices, res_atom_indices)
    dist_min = np.min(distances)

    if dist_min < 0.5:
        print(res.index, res.name, res_atom_indices)
        res_in_contact.append(res.index)

        res_atom_indices = top.select(f"resid {res.index}")
        protein_atom_in_contact.extend(res_atom_indices)

## fix protein atoms that are not in contact in the system
lig_atom_indices = top.select("resname MOL")
flexible_atom_indices = set(lig_atom_indices) | set(protein_atom_in_contact)
for i in range(system.getNumParticles()):
    if i not in flexible_atom_indices:
        system.setParticleMass(i, 0.0)

with open('./output/flexible_atom_indices.pkl', 'wb') as f:
    pickle.dump(flexible_atom_indices, f)

integrator = mm.LangevinMiddleIntegrator(
    300 * unit.kelvin, 1.0 / unit.picosecond, 0.001 * unit.picoseconds
)

platform = mm.Platform.getPlatformByName("CUDA")

simulation = app.Simulation(prmtop.topology, system, integrator, platform)
simulation.context.setPositions(inpcrd.positions)
simulation.minimizeEnergy()


simulation.reporters.append(
    app.DCDReporter("./output/trajectory.dcd", 1000, enforcePeriodicBox=False)
)
simulation.reporters.append(
    app.StateDataReporter(
        "./output/simulation_log.csv", 1000, step=True, potentialEnergy=True, temperature=True
    )
)

start_time = time.time()
simulation.step(200_000_000)

print(f"Simulation time: {time.time() - start_time:.2f} s")

