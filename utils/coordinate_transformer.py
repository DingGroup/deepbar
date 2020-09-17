__author__ = "Xinqiang Ding <xqding@umich.edu>"
__date__ = "2019/08/01 22:53:01"

import torch
import numpy as np
import mdtraj
import simtk.openmm as omm
import simtk.unit as unit
import simtk.openmm.app.topology as topology
import simtk.openmm.app.element as element
import math
from collections import deque, OrderedDict, defaultdict, namedtuple
import MDTorch.utils.functional as functional

class CoordinateTransformer():
    """ A class used for transforming internal coordinates into 
    Cartesian coordinates of a molecule.
    """
    
    def __init__(self, particle_bonds, reference_particle_1, reference_particle_2, reference_particle_3):
        """ The CoordinateTransformer class is initialized using
        the bond information of a molecule and three particles used
        as references for converting between internal coordiantes and
        Cartesian coordiantes.

        The procedure used by the class to transform internal coodinates 
        into Cartesian coordinates is as follows.
        
        First, we choose three reference particles: particle_1, particle_2,
        and particle_3. These three particles can be any particles in the
        system as long as (1) both particle_2 and particle_3 are bonded with 
        particle_1 and (2) the angle between them is not 180 degree. 

        To build the Cartesian coordinate of the system, we first
        specify the Cartesian coordinate of particle_1. The bond between 
        particle_1 and particle_2 is set to align with the z-axis. Thefore,
        the Cartesian coordinate of particle_2 can be specified using the bond
        length between particle_1 and particle_2, i.e., the x and y coordinates
        of particle_2 are the same as that of particle_1 and the z coordiante
        of particle_2 is equla to that of particle_2 plus the bond length.
        The plane defined be the three reference particles is set to be in the 
        y-z plane. Thefore, the Cartesian coordinate of particle_3 can be 
        specified using the bond length between particle_3 and particle_1 and 
        the angle among the three reference particles.

        With the Cartesian coordinates of the three reference particles defined,
        we can compute the Cartesian coordinates of other particles using their
        internal coordinates including bonds, angles and dihedral angles.
        Because internal coordinates of particles are defined with repect to 
        other particles, we need to record specify and record this information,
        i.e., for particle l, its internal coordinates are defined with respect 
        to particle i, j, and k using the bond (k-l), the angle (j-k-l) and 
        the dihedral angle (i-j-k-l). The Cartesian coordinate of particle l 
        depends on that of particles i, j, and k, so we have to compute
        the Cartesian coordinates of particles i,j, and k, before we compute
        that of particle l. The method used here to respect the dependency is
        to explore particles in the molecule via bread-first search on the 
        molecule graph with particles as nodes and bonds as edges. As we 
        visit each particle l, we record three other particles i,j, and k 
        such that particles i, j , and k are successively bonded,
        particle k is bonded with particle_l, and particles i, j, and k 
        have been visited before particle l. In addition, we also record
        the order of particles that are visited. 
        

        Parameters:
        -----------
        particle_bonds: dictionary
            A dictionary contains the information of bonds in a molecule.
            particle_bonds[i] = [j, k, l, ...] if particle i is bonded
            with particle j, k, and l. particle_bonds[i] = [] if particle
            i is not bonded with any other particles. Each particle of 
            the molecule has to appear once in the keys of particle_bonds.
        reference_particle_1: Int
            The index of the reference particle 1
        reference_particle_2: Int
            The index of the reference particle 2
        reference_particle_3: Int
            The index of the reference particle 3
        """        
        
        self.particle_bonds = particle_bonds
        self.num_particles = len(particle_bonds.keys())
        
        self.ref_particle_1 = reference_particle_1
        self.ref_particle_2 = reference_particle_2
        self.ref_particle_3 = reference_particle_3

        ## parent_particle[j] = i if particle_j is discoverd
        ## by particle_i in bread-first search
        self.parent_particle = {}
        self.parent_particle[self.ref_particle_2] = self.ref_particle_1
        self.parent_particle[self.ref_particle_3] = self.ref_particle_1
        self.parent_particle[self.ref_particle_1] = self.ref_particle_2
        
        particle_visit_flag = defaultdict(bool)
        particle_visit_flag[self.ref_particle_1] = True
        particle_visit_flag[self.ref_particle_2] = True
        particle_visit_flag[self.ref_particle_3] = True

        ## particles used to define internal coordinates
        self.bond_particle_idx = {}
        self.angle_particle_idx = {}
        self.dihedral_particle_idx = {}

        ## record particles in the order of visting in
        ## bread-first search
        self.particle_visited_in_order = []

        ## Q contains all the frontier nodes that have been explored
        ## in bread-first search
        Q = deque([])
        Q.append(self.ref_particle_3)

        ## particles directly bonded with referent particle 1 and particle 2
        ## need to be taken care of with exceptions in term of the three
        ## particles with respect to which the internal coordinates are
        ## defined
        
        ## particles bonded with self.ref_particle_1
        for p in self.particle_bonds[self.ref_particle_1]:
            if p not in [self.ref_particle_2, self.ref_particle_3]:
                self.particle_visited_in_order.append(p)
                self.parent_particle[p] = self.ref_particle_1                
                Q.append(p)
                particle_visit_flag[p] = True
                
                self.bond_particle_idx[p] = (self.ref_particle_1, p)
                self.angle_particle_idx[p] = (                    
                    self.ref_particle_2,
                    self.ref_particle_1,
                    p)
                self.dihedral_particle_idx[p] = (
                    self.ref_particle_3,
                    self.ref_particle_2,
                    self.ref_particle_1,
                    p)                    

        ## particles bonded with self.ref_particle_2
        for p in self.particle_bonds[self.ref_particle_2]:
            if p not in [self.ref_particle_1, self.ref_particle_3] and \
               particle_visit_flag[p] is False:
                self.particle_visited_in_order.append(p)
                self.parent_particle[p] = self.ref_particle_2
                Q.append(p)
                particle_visit_flag[p] = True
                
                self.bond_particle_idx[p] = (self.ref_particle_2, p)
                self.angle_particle_idx[p] = (
                    self.ref_particle_1,
                    self.ref_particle_2,
                    p)
                self.dihedral_particle_idx[p] = (
                    self.ref_particle_3,
                    self.ref_particle_1,
                    self.ref_particle_2,
                    p)
        
        ## bead-first search for all other particles
        while Q:
            pivot_p = Q.popleft()
            for p in self.particle_bonds[pivot_p]:
                if particle_visit_flag[p] is False:
                    self.particle_visited_in_order.append(p)
                    self.parent_particle[p] = pivot_p
                    Q.append(p)
                    particle_visit_flag[p] = True

                    self.bond_particle_idx[p] = (pivot_p, p)
                    self.angle_particle_idx[p] = (
                        self.parent_particle[pivot_p],
                        pivot_p,
                        p)
                    self.dihedral_particle_idx[p] = (
                        self.parent_particle[self.parent_particle[pivot_p]],
                        self.parent_particle[pivot_p],
                        pivot_p,
                        p)
                    
    def compute_xyz_from_internal(self,
                                  particle_1_xyz,
                                  particle_2_bond,
                                  particle_3_bond,
                                  particle_3_angle,
                                  bond, angle, dihedral):
        """ Compute the Cartesian coordinates of the system based on internal
        coordinates. 

        parameters:
        -----------
        particle_1_xyz: Tensor, shape = (batch_size, 3)
            the Cartensian coordinates of reference particle 1.
        particle_2_bond: Tensor, shape = (batch_size)
            the length of bond between reference particle 1 and 2
        particle_3_bond: Tensor, shape = (batch_size)
            the length of bond between reference particle 1 and 3
        particle_3_angle: Tensor, shape = (batch_size)
            the value of angle between reference particle 2, 1, and 3
        bond: Tensor, shape = (batch_size, num_particles-3)
            the bond length for particles in the system other than 
            the three reference particles
        angle: Tensor, shape = (batch_size, num_particles-3)
            the angle  for particles in the system other than 
            the three reference particles
        dihedral: Tensor, shape = (batch_size, num_particles-3)
            the dihedral angles for particles in the system other than 
            the three reference particles
        
            Note bond[:,i], angle[:,i], and dihedral[:,i] specify
            the internal coordinate of the particle 
            self.particle_visit_in_order[i]
        
        """

        ## unsqueeze some of the parameters
        particle_2_bond = torch.unsqueeze(particle_2_bond, -1)
        particle_3_bond = torch.unsqueeze(particle_3_bond, -1)
        particle_3_angle = torch.unsqueeze(particle_3_angle, -1)
        
        batch_size = particle_1_xyz.shape[0]
        
        #### xyz collects Cartesian coordinates of particles        
        xyz = {}

        ## Cartesian coordinates of the three reference particles
        xyz[self.ref_particle_1] = particle_1_xyz        
        xyz[self.ref_particle_2] = particle_1_xyz + \
            torch.cat([particle_2_bond.new_zeros(batch_size, 2), particle_2_bond], dim = -1)
            
        xyz[self.ref_particle_3] = particle_1_xyz + \
            torch.cat([
                particle_3_bond.new_zeros(batch_size, 1),
                particle_3_bond*torch.sin(particle_3_angle),
                particle_3_bond*torch.cos(particle_3_angle)
            ], dim = -1)

        log_jacobian = torch.log(torch.squeeze(particle_3_bond))

        ## Cartesian coordinates of other particles
        for idx in range(len(self.particle_visited_in_order)):
            p = self.particle_visited_in_order[idx]
            p_i, p_j, p_k, p_l = self.dihedral_particle_idx[p]
            assert(p == p_l)
            xyz[p_l], logabsdet = functional.compute_particle_xyz_from_internal(
                xyz[p_i],
                xyz[p_j],
                xyz[p_k],
                bond[:, idx],
                angle[:, idx],
                dihedral[:, idx])
            log_jacobian = log_jacobian + logabsdet

        ## collect the Cartesian coordinates of particles in the order
        ## of particle index
        particles = list(xyz.keys())
        particles.sort()
        coor = torch.stack([xyz[particle] for particle in particles], dim = 1)
        
        return coor, log_jacobian
    
    def compute_internal_from_xyz(self, xyz):
        """ Compute the internal coordinates from an xyz array

        Parameters:
        -----------
        xyz: Tensor, shape = (batch_size, num_particles, 3)
            a Tensor of xyz used to compute internal coordinates.
        
        Returns:
        --------
        output : dictionary
            a dictionary consists of internal coordinates.
        """

        ref_p_1_xyz = xyz[:, self.ref_particle_1, :]
        ref_p_2_bond = torch.squeeze(
            functional.compute_distances(
                xyz,
                torch.tensor([[self.ref_particle_1, self.ref_particle_2]])
            )
        )
        ref_p_3_bond = torch.squeeze(
            functional.compute_distances(
                xyz,
                torch.tensor([[self.ref_particle_1, self.ref_particle_3]])
            )
        )            
        ref_p_3_angle = torch.squeeze(
            functional.compute_angles(
                xyz,
                torch.tensor([[self.ref_particle_2,self.ref_particle_1,self.ref_particle_3]])
            )
        )
        
        bond = functional.compute_distances(
            xyz,
            torch.tensor([self.bond_particle_idx[p] for p in self.particle_visited_in_order])
        )
        angle = functional.compute_angles(
            xyz,
            torch.tensor([self.angle_particle_idx[p] for p in self.particle_visited_in_order])
        )

        dihedral = functional.compute_dihedrals(
            xyz,
            torch.tensor([self.dihedral_particle_idx[p] for p in self.particle_visited_in_order])
        )

        
        internal_coor = InternalCoordinate(
            reference_particle_1_xyz = ref_p_1_xyz,
            reference_particle_2_bond = ref_p_2_bond,
            reference_particle_3_bond = ref_p_3_bond,
            reference_particle_3_angle = ref_p_3_angle,
            bond = bond,
            angle = angle,
            dihedral = dihedral
        )
        
        ## compute logabsdet of the transform
        logabsdet = -torch.log(ref_p_3_bond)
        logabsdet += -torch.sum(torch.log(torch.abs(bond**2*torch.sin(math.pi - angle))), -1)

        return internal_coor, logabsdet

class InternalCoordinate():
    def __init__(self,
                 reference_particle_1_xyz = None,
                 reference_particle_2_bond = None,
                 reference_particle_3_bond = None,
                 reference_particle_3_angle = None,
                 bond = None,
                 angle = None,
                 dihedral = None
    ):
        self.reference_particle_1_xyz = reference_particle_1_xyz
        self.reference_particle_2_bond = reference_particle_2_bond
        self.reference_particle_3_bond = reference_particle_3_bond
        self.reference_particle_3_angle = reference_particle_3_angle

        self.bond = bond
        self.angle = angle
        self.dihedral = dihedral
        
    def cuda(self):
        self.reference_particle_1_xyz = self.reference_particle_1_xyz.cuda()
        self.reference_particle_2_bond = self.reference_particle_2_bond.cuda()
        self.reference_particle_3_bond = self.reference_particle_3_bond.cuda()
        self.reference_particle_3_angle = self.reference_particle_3_angle.cuda()

        self.bond = self.bond.cuda()
        self.angle = self.angle.cuda()
        self.dihedral = self.dihedral.cuda()

    def double(self):
        self.reference_particle_1_xyz = self.reference_particle_1_xyz.double()
        self.reference_particle_2_bond = self.reference_particle_2_bond.double()
        self.reference_particle_3_bond = self.reference_particle_3_bond.double()
        self.reference_particle_3_angle = self.reference_particle_3_angle.double()

        self.bond = self.bond.double()
        self.angle = self.angle.double()
        self.dihedral = self.dihedral.double()

    def float(self):
        self.reference_particle_1_xyz = self.reference_particle_1_xyz.float()
        self.reference_particle_2_bond = self.reference_particle_2_bond.float()
        self.reference_particle_3_bond = self.reference_particle_3_bond.float()
        self.reference_particle_3_angle = self.reference_particle_3_angle.float()

        self.bond = self.bond.float()
        self.angle = self.angle.float()
        self.dihedral = self.dihedral.float()

    def to_tensor(self):
        ic = torch.cat([torch.unsqueeze(self.reference_particle_2_bond, -1),
                        torch.unsqueeze(self.reference_particle_3_bond, -1),
                        torch.unsqueeze(self.reference_particle_3_angle, - 1),
                        self.bond, self.angle, self.dihedral], -1)
        return ic

    def set_from_tensor(self, ic):
        self.reference_particle_1_xyz = ic.new_zeros(ic.shape[0], 3)
        self.reference_particle_2_bond = ic[:, 0]
        self.reference_particle_3_bond = ic[:, 1]
        self.reference_particle_3_angle = ic[:, 2]

        m = (ic.shape[1] - 3) // 3
        self.bond = ic[:, 3:3+m]
        self.angle = ic[:, 3+m:3+2*m]
        self.dihedral = ic[:, 3+2*m:3+3*m]
        
    
def get_bonded_atoms(topology):
    """ Return a dictionary of bonded atoms

    Parameters
    ----------
    topology : mdtraj.Topology

    Returns:
    -------
    out: dict
        out[i] = list of atom indices that are bonded with atom i.
        If atom i is not bonded to any other atoms, out[i] = [].
    """
    
    bonded_atoms = defaultdict(list)
    for bond in topology.bonds:
        atom1_index = bond.atom1.index
        atom2_index = bond.atom2.index
        bonded_atoms[atom1_index].append(atom2_index)
        bonded_atoms[atom2_index].append(atom1_index)
    return bonded_atoms


# def internal_coor_to_xyz_system(coor_base, bonds, b, a, d):
#     num_copy = b.shape[0]
#     assert(a.shape[0] == num_copy)
#     assert(d.shape[0] == num_copy)
    
#     coor_res = {}
#     coor_res[-1] = coor_base[-1].repeat(num_copy, 1)
#     coor_res[0] = coor_base[0].repeat(num_copy, 1)
#     coor_res[1] = coor_base[1].repeat(num_copy, 1)

#     num_atoms = b.shape[1] + 2
#     assert(a.shape[1] + 2 == num_atoms)
#     assert(d.shape[1] + 2 == num_atoms)
    
#     atom_visit_flag = [0 for i in range(num_atoms)]
#     atom_visit_flag[0] = 1
#     atom_visit_flag[1] = 1

#     parent_atom = {}
#     parent_atom[0] = -1
#     parent_atom[1] = 0
    
#     log_Jacobian = 0
#     Q = deque([1])

#     ## record the atom index corresponding to bond, angle and torsion
#     b_atom_idx = {}
#     a_atom_idx = {}
#     d_atom_idx = {}
    
#     while Q:
#         current_atom = Q.popleft()
#         for atom in bonds[current_atom]:
#             if atom_visit_flag[atom] == 0:
#                 Q.append(atom)
#                 atom_visit_flag[atom] = 1
#                 parent_atom[atom] = current_atom

#                 coor_3 = coor_res[current_atom]
#                 coor_2 = coor_res[parent_atom[current_atom]]
#                 coor_1 = coor_res[parent_atom[parent_atom[current_atom]]]
#                 coor_4, logabsdet = internal_coor_to_xyz_atom(coor_1, coor_2, coor_3, b[:,atom-2,:], a[:,atom-2,:], d[:,atom-2,:])

#                 coor_res[atom] = coor_4
#                 log_Jacobian = log_Jacobian + logabsdet

#                 ## record the atom index
#                 b_atom_idx[atom-2] = (current_atom, atom)
#                 a_atom_idx[atom-2] = (parent_atom[current_atom], current_atom, atom)
#                 d_atom_idx[atom-2] = (parent_atom[parent_atom[current_atom]], parent_atom[current_atom], current_atom, atom)

#     coor_list = []
#     for i in range(num_atoms):
#         coor_list.append(coor_res[i])

#     coor_res = torch.stack(coor_list)
#     coor_res = torch.transpose(coor_res, 0, 1)

#     return coor_res, log_Jacobian, b_atom_idx, a_atom_idx, d_atom_idx


# def compute_bond_and_angle_mu_std(system, T):
#     '''
#     system: an OpenMM system
#     T: float
#     '''
#     assert type(T) is int or type(T) is float
    
#     bond_parameters, angle_parameters = get_bond_and_angle_parameters(system)
#     kbT = unit.BOLTZMANN_CONSTANT_kB * T *unit.kelvin * unit.AVOGADRO_CONSTANT_NA
#     beta = 1/kbT

#     bond_mu = {}
#     bond_std = {}
#     for k, v in bond_parameters.items():
#         bond_mu[k] = bond_parameters[k][0].value_in_unit(unit.nanometer)

#         b_k = bond_parameters[k][1]
#         sigma = (1/(beta*b_k))**(1/2)    
#         bond_std[k] = sigma.value_in_unit(unit.nanometer)        

#     angle_mu = {}
#     angle_std = {}
#     for k, v in angle_parameters.items():
#         angle_mu[k] = angle_parameters[k][0].value_in_unit(unit.radian)

#         b_k = angle_parameters[k][1]
#         sigma = (1/(beta*b_k))**(1/2)    
#         angle_std[k] = sigma.value_in_unit(unit.radian)        

#     return {'bond_mu': bond_mu, 'bond_std': bond_std, 'angle_mu': angle_mu, 'angle_std': angle_std}




# def get_bond_and_angle_parameters(system):
#     ## get bond parameters
#     for i in range(system.getNumForces()):
#         force = system.getForce(i)
#         if type(force) == omm.HarmonicBondForce:
#             break

#     bond_parameters = {}
#     for idx in range(force.getNumBonds()):
#         i, j, l, k = force.getBondParameters(idx)
#         bond_parameters[tuple([i,j])] = (l, k)
#         bond_parameters[tuple([j,i])] = (l, k)        

#     ## get angle parameters
#     for i in range(system.getNumForces()):
#         force = system.getForce(i)
#         if type(force) == omm.HarmonicAngleForce:
#             break

#     angle_parameters = {}
#     for idx in range(force.getNumAngles()):
#         i, j, k, l, m = force.getAngleParameters(idx)
#         angle_parameters[tuple([i,j,k])] = (l, m)
#         angle_parameters[tuple([k,j,i])] = (l, m)

#     return bond_parameters, angle_parameters
        

    # def compute_internal_from_traj(self, traj):
    #     """ Compute the internal coordinates from a trajectory

    #     Parameters:
    #     -----------
    #     traj: mdtraj.Trajectory
    #         a trajectoy used to compute internal coordinates.
        
    #     Returns:
    #     --------
    #     output : dictionary
    #         a dictionary consists of internal coordinates.
    #     """
        
    #     ref_p_1_xyz = traj.atom_slice([self.ref_particle_1]).xyz
    #     ref_p_1_xyz = np.squeeze(ref_p_1_xyz)
    #     ref_p_2_bond = mdtraj.compute_distances(
    #         traj, [[self.ref_particle_1, self.ref_particle_2]])
    #     ref_p_3_bond = mdtraj.compute_distances(
    #         traj, [[self.ref_particle_1, self.ref_particle_3]])
    #     ref_p_3_angle = mdtraj.compute_angles(
    #         traj, [[self.ref_particle_2, self.ref_particle_1,
    #                 self.ref_particle_3]])

    #     output = {}        
    #     output['ref_particle_1_xyz'] = torch.from_numpy(ref_p_1_xyz)
    #     output['ref_particle_2_bond'] = torch.from_numpy(ref_p_2_bond)
    #     output['ref_particle_3_bond'] = torch.from_numpy(ref_p_3_bond)
    #     output['ref_particle_3_angle'] = torch.from_numpy(ref_p_3_angle)

    #     ## if the system has only three particles, self.particle_visit_in_order
    #     ## will be an empty list
    #     if len(self.particle_visited_in_order) != 0:
    #         bond = mdtraj.compute_distances(traj,
    #                [self.bond_particle_idx[p]
    #                 for p in self.particle_visited_in_order])
    #         angle = mdtraj.compute_angles(traj,
    #                [self.angle_particle_idx[p]
    #                 for p in self.particle_visited_in_order])
    #         dihedral = mdtraj.compute_dihedrals(traj,
    #                [self.dihedral_particle_idx[p]
    #                 for p in self.particle_visited_in_order])
            
    #         output['bond'] = torch.from_numpy(bond)
    #         output['angle'] = torch.from_numpy(angle)
    #         output['dihedral'] = torch.from_numpy(dihedral)
    #     else:
    #         output['bond'] = None
    #         output['angle'] = None
    #         output['dihedral'] = None
            
        
    #     return output

# def compute_dihedral(coor):
#     b1 = coor[1] - coor[0]
#     b2 = coor[2] - coor[1]
#     b3 = coor[3] - coor[2]

#     b1 = b1/np.linalg.norm(b1)
#     b2 = b2/np.linalg.norm(b2)
#     b3 = b3/np.linalg.norm(b3)
    
#     n1 = np.cross(b1, b2)
#     n2 = np.cross(b2, b3)

#     m1 = np.cross(n1, b2)
#     x = np.inner(n1, n2)
#     y = np.inner(m1, n2)
    
#     return -np.arctan2(y, x)




# def _compute_article_xyz_from_internal(self,
#                                         particle_1_xyz,
#                                         particle_2_xyz,
#                                         particle_3_xyz,
#                                         bond, angle, dihedral):
#     """ Compute the Cartesian coordinate of a particle based on its internal
#     coordinate with respect to three other particles and the Cartesian 
#     coordinates of these three particles. This function runs a batch mode, which
#     mean that it compute the Cartensian coordinates of a batch of particles based
#     on a batch of three other particles.

#     Parameters:
#     -----------
#     particle_1_xyz: Tensor 
#         the Cartesian coordinate of atom 1.
#     particle_2_xyz: Tensor 
#         the Cartesian coordinate of atom 2.
#     particle_3_xyz: Tensor 
#         the Cartesian coordinate of atom 3.
#         Note particle_1_xyz, particle_2_xyz and particle_3_xyz have the same size: [batch_size, 3].
#     bond: Tensor
#         the length of the bond between atom 3 and atom 4
#     angle: Tensor
#         the value of the angle between atom 2, atom 3 and atom 4
#     dihedral: Tensor
#         the value of the dihedral angle between atom 1, atom 2, atom3 and atom 4
#         Note bond, angle and dihedral have the same size: [batch_size, 1]

#     Returns:
#     --------
#     particle_4_xyz: Tensor
#         the Cartensian coordiante of atom 4. Its size is [batch_size, 3]
#     J:
#     """

#     ## calculate the coordinate of the forth atom
#     u_12 = particle_2_xyz - particle_1_xyz
#     u_12 = u_12 / torch.sqrt(torch.sum(u_12 ** 2, 1, keepdim = True))

#     u_23 = particle_3_xyz - particle_2_xyz
#     u_23 = u_23 / torch.sqrt(torch.sum(u_23 ** 2, 1, keepdim = True))
#     u_34 = u_23 * torch.cos(math.pi - angle)

#     n_123 = torch.cross(u_12, u_23, dim = -1)
#     n_123 = n_123 / torch.sqrt(torch.sum(n_123**2, 1, keepdim = True))
#     u_34 = u_34 + n_123*torch.sin(math.pi-angle)*torch.sin(dihedral)


#     y = torch.cross(n_123, u_23, dim = -1)
#     y = y / torch.sqrt(torch.sum(y**2, 1, keepdim = True))
#     u_34 = u_34 + y*torch.sin(math.pi-angle)*torch.cos(dihedral)

#     particle_4_xyz = particle_3_xyz + u_34*bond

#     ## calculate the Jacobian of the transform
#     db = u_34

#     n_234 = torch.cross(u_23, u_34, dim = -1)
#     n_234 = n_234 / torch.sqrt(torch.sum(n_234**2, 1, keepdim = True))
#     da = torch.cross(n_234, u_34*bond, dim = -1)

#     dd = torch.cross(u_23, u_34*bond, dim = -1)

#     J = torch.stack([db, da, dd], dim = -1)
#     J_slogdet = torch.slogdet(J)
#     return particle_4_xyz, J_slogdet.logabsdet

# def _compute_particle_xyz_from_internal_backup(self,
#                                         particle_1_xyz,
#                                         particle_2_xyz,
#                                         particle_3_xyz,
#                                         bond, angle, dihedral):
#     """ Compute the Cartesian coordinate of a particle based on its internal
#     coordinate with respect to three other particles and the Cartesian 
#     coordinates of these three particles. This function runs a batch mode, which
#     mean that it compute the Cartensian coordinates of a batch of particles based
#     on a batch of three other particles.

#     Parameters:
#     -----------
#     particle_1_xyz: Tensor 
#         the Cartesian coordinate of atom 1.
#     particle_2_xyz: Tensor 
#         the Cartesian coordinate of atom 2.
#     particle_3_xyz: Tensor 
#         the Cartesian coordinate of atom 3.
#         Note particle_1_xyz, particle_2_xyz and particle_3_xyz have the same size: [batch_size, 3].
#     bond: Tensor
#         the length of the bond between atom 3 and atom 4
#     angle: Tensor
#         the value of the angle between atom 2, atom 3 and atom 4
#     dihedral: Tensor
#         the value of the dihedral angle between atom 1, atom 2, atom3 and atom 4
#         Note bond, angle and dihedral have the same size: [batch_size, 1]

#     Returns:
#     --------
#     particle_4_xyz: Tensor
#         the Cartensian coordiante of atom 4. Its size is [batch_size, 3]
#     J:
#     """

#     ## calculate the coordinate of the forth atom
#     wi, wj, wk = particle_1_xyz, particle_2_xyz, particle_3_xyz

#     e1 = wk - wj
#     e1 = e1 / torch.norm(e1, dim = -1, keepdim = True)

#     e3 = torch.cross(e1, wj - wi, dim = -1)
#     e3 = e3 / torch.norm(e3, dim = -1, keepdim = True)

#     e2 = torch.cross(e3, e1, dim = -1)

#     wl = wk + bond*(torch.cos(math.pi-angle)*e1 +
#                     (-1)*torch.sin(math.pi-angle)*torch.cos(dihedral) +
#                     (-1)*torch.sin(math.pi-angle)*torch.sin(dihedral))

#     particle_4_xyz = wl

#     ## calculate the Jacobian of the transform
#     db = wl - wk        
#     n = torch.cross(e1, db, dim = -1)
#     n = n / torch.norm(n, dim = -1, keepdim = True)
#     da = torch.cross(db, n, dim = -1)
#     dd = bond*n

#     J = torch.stack([db, da, dd], dim = -1)
#     J_slogdet = torch.slogdet(J)
#     return particle_4_xyz, J_slogdet.logabsdet
