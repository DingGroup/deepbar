import openmm.app  as app
import openmm as omm
import openmm.unit as unit
import math
from sys import exit

## make the system of alanine dipeptide
pdb = app.PDBFile('./structure/alanine_dipeptide.pdb')
forcefield = app.ForceField('amber99sb.xml', 'amber99_obc.xml')
system = forcefield.createSystem(pdb.topology, nonbondedMethod=app.NoCutoff,
                                 constraints=None, rigidWater=False)
xml = omm.XmlSerializer.serialize(system)
f = open("./structure/system.xml", 'w')
f.write(xml)
f.close()

## 
bias_torsion_1 = omm.CustomTorsionForce("0.5*k1*dtheta^2; dtheta = min(tmp, 2*pi1-tmp); tmp = abs(theta - theta1)")
bias_torsion_1.addGlobalParameter("pi1", math.pi)
bias_torsion_1.addGlobalParameter("k1", 200)
bias_torsion_1.addGlobalParameter("theta1", 0.0)
bias_torsion_1.addTorsion(6, 8, 14, 16)

bias_torsion_2 = omm.CustomTorsionForce("0.5*k2*dtheta^2; dtheta = min(tmp, 2*pi2-tmp); tmp = abs(theta - theta2)")
bias_torsion_2.addGlobalParameter("pi2", math.pi)
bias_torsion_2.addGlobalParameter("k2", 200)
bias_torsion_2.addGlobalParameter("theta2", 0.0)
bias_torsion_2.addTorsion(14, 8, 6, 4)

system.addForce(bias_torsion_1)
system.addForce(bias_torsion_2)

xml = omm.XmlSerializer.serialize(system)
f = open("./structure/system_with_bias.xml", 'w')
f.write(xml)
f.close()
