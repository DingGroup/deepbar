import torch
import numpy as np
import mdtraj
import simtk.openmm as omm
import simtk.unit as unit
import simtk.openmm.app.topology as topology
import simtk.openmm.app.element as element
import math
from collections import deque, OrderedDict, defaultdict, namedtuple
import MMFlow.utils.functional as functional

class CoordinateTransformer():
    """ A class used for transforming internal coordinates into 
    Cartesian coordinates of a molecule.
    """
    
    def __init__(self,
                 particle_bonds,
                 reference_particle_1,
                 reference_particle_2,
                 reference_particle_3):
        
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
        specify the Cartesian coordinate of particle_1.        
        The Cartesian coordinate of particle_2 is
        specified in a spherical coordinate style. The bond length between
        particle_1 and particle_2 is used as the radius. The angle
        between the z-plus axis and the bond of particle_1 and particle_2
        is used as the polar angle. The angle between x-plus axis and the projection
        of the bond on the x-y plane is used as the azimuthal angle. The Cartesian
        coordinate of particle_3 is specified similary as that of particle_2.
        Note the azimuthal angle used here is from (-pi, pi] instead of (0, 2*pi] 
        as used in the standard spherical coordinate.

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
                    
    def compute_xyz_from_internal_coordinate_and_orientation(
            self,
            particle_1_xyz,
            particle_2_bond, particle_2_polar_angle, particle_2_azimuthal_angle,
            particle_3_bond, particle_3_polar_angle, particle_3_azimuthal_angle,
            bond, angle, dihedral):
        """ Compute the Cartesian coordinates of the system based on internal
        coordinates. 

        parameters:
        -----------
        particle_1_xyz: Tensor, shape = (batch_size, 3)
            the Cartensian coordinates of reference particle 1.
        particle_2_bond: Tensor, shape = (batch_size)
            the length of bond between reference particle 1 and 2
        particle_2_polar_angle: Tensor, shape = (batch_size)
            the polar angle of reference particle 2
        particle_2_azimuthal_angle: Tensor, shape = (batch_size)
            the azimuthal angle of reference particle 2
        particle_3_bond: Tensor, shape = (batch_size)
            the length of bond between reference particle 1 and 3
        particle_3_polar_angle: Tensor, shape = (batch_size)
            the polar angle of reference particle 3
        particle_3_azimuthal_angle: Tensor, shape = (batch_size)
            the azimuthal angle of reference particle 3
            Note: the azimuthal angle is defined as the angle between
            x-minus axis and the projection of the bond (between particle
            1 and particle 2 or between particle 1 and particle 3) on
            the x-y plane
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
        particle_2_polar_angle = torch.unsqueeze(particle_2_polar_angle, -1)
        particle_2_azimuthal_angle = torch.unsqueeze(particle_2_azimuthal_angle, -1)

        particle_3_bond = torch.unsqueeze(particle_3_bond, -1)
        particle_3_polar_angle = torch.unsqueeze(particle_3_polar_angle, -1)
        particle_3_azimuthal_angle = torch.unsqueeze(particle_3_azimuthal_angle, -1)
                
        batch_size = particle_1_xyz.shape[0]
        
        #### xyz collects Cartesian coordinates of particles        
        xyz = {}

        ## Cartesian coordinates of the three reference particles
        xyz[self.ref_particle_1] = particle_1_xyz
        xyz[self.ref_particle_2] = particle_1_xyz + \
            particle_2_bond * torch.cat([torch.sin(particle_2_polar_angle)*torch.cos(particle_2_azimuthal_angle),
                                         torch.sin(particle_2_polar_angle)*torch.sin(particle_2_azimuthal_angle),
                                         torch.cos(particle_2_polar_angle)], dim = -1)

        log_jacobian = torch.log(torch.squeeze(particle_2_bond)**2*torch.sin(torch.squeeze(particle_2_polar_angle)))
        
        xyz[self.ref_particle_3] = particle_1_xyz + \
            particle_3_bond * torch.cat([torch.sin(particle_3_polar_angle)*torch.cos(particle_3_azimuthal_angle),
                                         torch.sin(particle_3_polar_angle)*torch.sin(particle_3_azimuthal_angle),
                                         torch.cos(particle_3_polar_angle)], dim = -1)

        log_jacobian += torch.log(torch.squeeze(particle_3_bond)**2*torch.sin(torch.squeeze(particle_3_polar_angle)))

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

    def compute_xyz_from_internal_coordinate(
            self,
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
            the polar angle of reference particle 3
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
    
    
    def compute_internal_coordinate_and_orientation_from_xyz(self, xyz):
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

        ref_p_2_polar_angle = torch.squeeze(
            functional.compute_polar_angle(
                xyz,
                torch.tensor([[self.ref_particle_1, self.ref_particle_2]])
            )
        )

        ref_p_2_azimuthal_angle = torch.squeeze(
            functional.compute_azimuthal_angle(
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

        ref_p_3_polar_angle = torch.squeeze(
            functional.compute_polar_angle(
                xyz,
                torch.tensor([[self.ref_particle_1, self.ref_particle_3]])
            )
        )

        ref_p_3_azimuthal_angle = torch.squeeze(
            functional.compute_azimuthal_angle(
                xyz,
                torch.tensor([[self.ref_particle_1, self.ref_particle_3]])                
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
        
        internal_coor = InternalCoordinateAndOrientation(
            reference_particle_1_xyz = ref_p_1_xyz,
            reference_particle_2_bond = ref_p_2_bond,
            reference_particle_2_polar_angle = ref_p_2_polar_angle,
            reference_particle_2_azimuthal_angle = ref_p_2_azimuthal_angle,
            reference_particle_3_bond = ref_p_3_bond,
            reference_particle_3_polar_angle = ref_p_3_polar_angle,
            reference_particle_3_azimuthal_angle = ref_p_3_azimuthal_angle,
            bond = bond,
            angle = angle,
            dihedral = dihedral
        )
        
        ## compute logabsdet of the transform
        log_absdet = -torch.log(ref_p_2_bond**2*torch.sin(ref_p_2_polar_angle))
        log_absdet += -torch.log(ref_p_3_bond**2*torch.sin(ref_p_3_polar_angle))
        log_absdet += -torch.sum(torch.log(torch.abs(bond**2*torch.sin(math.pi - angle))), -1)

        return internal_coor, log_absdet

    def compute_internal_coordinate_from_xyz(self, xyz):
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
                torch.tensor([[self.ref_particle_2, self.ref_particle_1, self.ref_particle_3]])
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
        log_absdet = -torch.log(ref_p_3_bond)
        log_absdet += -torch.sum(torch.log(torch.abs(bond**2*torch.sin(math.pi - angle))), -1)

        return internal_coor, log_absdet

class InternalCoordinateAndOrientation():
    def __init__(self,
                 reference_particle_1_xyz = None,
                 reference_particle_2_bond = None,
                 reference_particle_2_polar_angle = None,
                 reference_particle_2_azimuthal_angle = None,
                 reference_particle_3_bond = None,
                 reference_particle_3_polar_angle = None,
                 reference_particle_3_azimuthal_angle = None,
                 bond = None,
                 angle = None,
                 dihedral = None
    ):
        self.reference_particle_1_xyz = reference_particle_1_xyz
        self.reference_particle_2_bond = reference_particle_2_bond
        self.reference_particle_2_polar_angle = reference_particle_2_polar_angle
        self.reference_particle_2_azimuthal_angle = reference_particle_2_azimuthal_angle
        self.reference_particle_3_bond = reference_particle_3_bond
        self.reference_particle_3_polar_angle = reference_particle_3_polar_angle
        self.reference_particle_3_azimuthal_angle = reference_particle_3_azimuthal_angle
        self.bond = bond
        self.angle = angle
        self.dihedral = dihedral
        
    def cuda(self):
        self.reference_particle_1_xyz = self.reference_particle_1_xyz.cuda()
        self.reference_particle_2_bond = self.reference_particle_2_bond.cuda()
        self.reference_particle_2_polar_angle = self.reference_particle_2_polar_angle.cuda()
        self.reference_particle_2_azimuthal_angle = self.reference_particle_2_azimuthal_angle.cuda()
        self.reference_particle_3_bond = self.reference_particle_3_bond.cuda()
        self.reference_particle_3_polar_angle = self.reference_particle_3_polar_angle.cuda()
        self.reference_particle_3_azimuthal_angle = self.reference_particle_3_azimuthal_angle.cuda()
        self.bond = self.bond.cuda()
        self.angle = self.angle.cuda()
        self.dihedral = self.dihedral.cuda()

    def double(self):
        self.reference_particle_1_xyz = self.reference_particle_1_xyz.double()
        self.reference_particle_2_bond = self.reference_particle_2_bond.double()
        self.reference_particle_2_polar_angle = self.reference_particle_2_polar_angle.double()
        self.reference_particle_2_azimuthal_angle = self.reference_particle_2_azimuthal_angle.double()
        self.reference_particle_3_bond = self.reference_particle_3_bond.double()
        self.reference_particle_3_polar_angle = self.reference_particle_3_polar_angle.double()
        self.reference_particle_3_azimuthal_angle = self.reference_particle_3_azimuthal_angle.double()
        self.bond = self.bond.double()
        self.angle = self.angle.double()
        self.dihedral = self.dihedral.double()
        

    def float(self):
        self.reference_particle_1_xyz = self.reference_particle_1_xyz.float()
        self.reference_particle_2_bond = self.reference_particle_2_bond.float()
        self.reference_particle_2_polar_angle = self.reference_particle_2_polar_angle.float()
        self.reference_particle_2_azimuthal_angle = self.reference_particle_2_azimuthal_angle.float()
        self.reference_particle_3_bond = self.reference_particle_3_bond.float()
        self.reference_particle_3_polar_angle = self.reference_particle_3_polar_angle.float()
        self.reference_particle_3_azimuthal_angle = self.reference_particle_3_azimuthal_angle.float()
        self.bond = self.bond.float()
        self.angle = self.angle.float()
        self.dihedral = self.dihedral.float()                

    def __len__(self):
        return self.reference_particle_1_xyz.shape[0]


    def __getitem__(self, index):
        ic = InternalCoordinateAndOrientation(
            self.reference_particle_1_xyz[index],
            self.reference_particle_2_bond[index],
            self.reference_particle_2_polar_angle[index],
            self.reference_particle_2_azimuthal_angle[index],
            self.reference_particle_3_bond[index],
            self.reference_particle_3_polar_angle[index],
            self.reference_particle_3_azimuthal_angle[index],
            self.bond[index],
            self.angle[index],
            self.dihedral[index])
        return ic

        
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

    def __len__(self):
        return self.reference_particle_1_xyz.shape[0]

    def __getitem__(self, index):
        ic = InternalCoordinate(self.reference_particle_1_xyz[index],
                                self.reference_particle_2_bond[index],
                                self.reference_particle_3_bond[index],
                                self.reference_particle_3_angle[index],
                                self.bond[index],
                                self.angle[index],
                                self.dihedral[index])
        return ic
