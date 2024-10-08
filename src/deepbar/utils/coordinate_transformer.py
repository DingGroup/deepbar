import torch
import numpy as np
import math
from collections import deque, defaultdict
import deepbar.utils.functional as functional
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

class CoordinateTransformer:
    """A class used for transforming internal coordinates into
    Cartesian coordinates of a molecule.
    """

    def __init__(
        self,
        particle_bonds,
        reference_particle_1,
        reference_particle_2,
        reference_particle_3,
        dihedral_mode="fork",
    ):
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
        that is used in the standard spherical coordinate.

        With the Cartesian coordinates of the three reference particles defined,
        we can compute the Cartesian coordinates of other particles using their
        internal coordinates including bonds, angles and dihedral angles.
        Because internal coordinates of particles are defined with repect to 
        other particles, we need to specify and record this information,
        i.e., for particle l, its internal coordinates are defined with respect 
        to particle i, j, and k using the bond (k-l), the angle (j-k-l) and 
        the dihedral angle (i-j-k-l). The Cartesian coordinate of particle l 
        depends on that of particles i, j, and k, so we have to compute
        the Cartesian coordinates of particles i,j, and k, before we compute
        that of particle l. The method used here to respect the dependency is
        to explore particles in the molecule via bread-first search on the 
        molecule graph with particles as nodes and bonds as edges. As we 
        visit each particle l, we record three other particles i,j, and k.
        In addition, we also record the order of particles that are visited. 

        For a given particle l, there are multiple ways to choose the particles
        i, j, and k. The prameter dihedral_mode specifies two ways.

        When dihedral_mode is 'linear', particles, i, j, and k, are chosen based
        on the parent-child releationship specified during the breadth-first search,
        i.e., particles i, j , and k are successively bonded,
        particle k is bonded with particle_l, and particles i, j, and k 
        have been visited before particle l.

        The way of chosing particles i,j,and k with dihedral_mode being 'linear' tends
        to make many dihedral angles have multimodal distribution. For instance,
        if we have the following structure

                        H1
                       /
        C3 -- C2 -- C1 -- H2
                       \\
                        H3
        
        the dihedral angles for H1, H2, and H3 will be C3-C2-C1-H1, C3-C2-C1-H2, C3-C2-C1-H3.
        If the methy group can rotate, all the three dihedral angles for H1, H2 and H3 will
        have multimodal distributions.

        One way to decrease the number of dihedral angles that have multimodal distributions 
        is to set dihedral_mode to 'fork'. It uses the dihedral angle D3-C2-C1-H1 for H1, 
        uses the dihedral H1-C2-C1-H2 for H2 and used the dihedral H1-C2-C1-H3 for H3. 
        In this way, even though the methy group can rotate, the dihedrals H1-C2-C1-H2 and 
        H1-C2-C1-H3 have monomodal distributions. Therefore, the argument dihedral_mode
        is set to 'fork' by default

        
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
        dihedral_mode: Str
            The mode used to choose the three particles (i,j,k) to specify
            the internal coordinate of particle l.            
            There are two possible modes: 'fork' or 'linear', so it should be
            set to either 'fork' or 'linear'. By default, it is 'fork'.                       
        """

        self.particle_bonds = particle_bonds
        self.num_particles = len(particle_bonds.keys())

        self.ref_particle_1 = reference_particle_1
        self.ref_particle_2 = reference_particle_2
        self.ref_particle_3 = reference_particle_3

        self.dihedral_mode = dihedral_mode
        if self.dihedral_mode not in ["linear", "fork"]:
            raise ValueError("""The argument dihedral_mode has to be either 
                             'linear' or 'fork' """)

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
                    p,
                )
                self.dihedral_particle_idx[p] = (
                    self.ref_particle_3,
                    self.ref_particle_2,
                    self.ref_particle_1,
                    p,
                )

        if self.dihedral_mode == "linear":
            ## particles bonded with self.ref_particle_2
            for p in self.particle_bonds[self.ref_particle_2]:
                if (
                    p not in [self.ref_particle_1, self.ref_particle_3]
                    and particle_visit_flag[p] is False
                ):
                    self.particle_visited_in_order.append(p)
                    self.parent_particle[p] = self.ref_particle_2
                    Q.append(p)
                    particle_visit_flag[p] = True

                    self.bond_particle_idx[p] = (self.ref_particle_2, p)
                    self.angle_particle_idx[p] = (
                        self.ref_particle_1,
                        self.ref_particle_2,
                        p,
                    )
                    self.dihedral_particle_idx[p] = (
                        self.ref_particle_3,
                        self.ref_particle_1,
                        self.ref_particle_2,
                        p,
                    )

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
                            p,
                        )
                        self.dihedral_particle_idx[p] = (
                            self.parent_particle[self.parent_particle[pivot_p]],
                            self.parent_particle[pivot_p],
                            pivot_p,
                            p,
                        )

        if self.dihedral_mode == "fork":
            ## particles bonded with self.ref_particle_2
            first_child_particle = None
            for p in self.particle_bonds[self.ref_particle_2]:
                if (
                    p not in [self.ref_particle_1, self.ref_particle_3]
                    and particle_visit_flag[p] is False
                ):
                    self.particle_visited_in_order.append(p)
                    self.parent_particle[p] = self.ref_particle_2
                    Q.append(p)
                    particle_visit_flag[p] = True

                    self.bond_particle_idx[p] = (self.ref_particle_2, p)
                    self.angle_particle_idx[p] = (
                        self.ref_particle_1,
                        self.ref_particle_2,
                        p,
                    )

                    if first_child_particle is None:
                        first_child_particle = p
                        self.dihedral_particle_idx[p] = (
                            self.ref_particle_3,
                            self.ref_particle_1,
                            self.ref_particle_2,
                            p,
                        )
                    else:
                        self.dihedral_particle_idx[p] = (
                            first_child_particle,
                            self.ref_particle_1,
                            self.ref_particle_2,
                            p,
                        )

            ## bead-first search for all other particles
            while Q:
                pivot_p = Q.popleft()

                # the first node that is discovred from pivot_p
                # it will be used in defining the dihedral angles for
                # other node that are discoved from pivot_p
                first_child_particle = None

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
                            p,
                        )

                        if first_child_particle is None:
                            first_child_particle = p
                            self.dihedral_particle_idx[p] = (
                                self.parent_particle[self.parent_particle[pivot_p]],
                                self.parent_particle[pivot_p],
                                pivot_p,
                                p,
                            )
                        else:
                            self.dihedral_particle_idx[p] = (
                                first_child_particle,
                                self.parent_particle[pivot_p],
                                pivot_p,
                                p,
                            )

    def compute_xyz_from_internal_coordinate_and_orientation(
        self,
        particle_1_xyz,
        particle_2_bond,
        particle_2_polar_angle,
        particle_2_azimuthal_angle,
        particle_3_bond,
        particle_3_polar_angle,
        particle_3_azimuthal_angle,
        bond,
        angle,
        dihedral,
    ):
        """Compute the Cartesian coordinates of the system based on internal
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
        xyz[self.ref_particle_2] = particle_1_xyz + particle_2_bond * torch.cat(
            [
                torch.sin(particle_2_polar_angle)
                * torch.cos(particle_2_azimuthal_angle),
                torch.sin(particle_2_polar_angle)
                * torch.sin(particle_2_azimuthal_angle),
                torch.cos(particle_2_polar_angle),
            ],
            dim=-1,
        )

        log_jacobian = torch.log(
            torch.abs(
                torch.squeeze(particle_2_bond) ** 2
                * torch.sin(torch.squeeze(particle_2_polar_angle))
            )
        )

        xyz[self.ref_particle_3] = particle_1_xyz + particle_3_bond * torch.cat(
            [
                torch.sin(particle_3_polar_angle)
                * torch.cos(particle_3_azimuthal_angle),
                torch.sin(particle_3_polar_angle)
                * torch.sin(particle_3_azimuthal_angle),
                torch.cos(particle_3_polar_angle),
            ],
            dim=-1,
        )

        log_jacobian += torch.log(
            torch.abs(
                torch.squeeze(particle_3_bond) ** 2
                * torch.sin(torch.squeeze(particle_3_polar_angle))
            )
        )

        ## Cartesian coordinates of other particles
        for idx in range(len(self.particle_visited_in_order)):
            p = self.particle_visited_in_order[idx]
            p_i, p_j, p_k, p_l = self.dihedral_particle_idx[p]
            assert p == p_l
            xyz[p_l], logabsdet = functional._compute_particle_xyz_from_internal(
                xyz[p_i],
                xyz[p_j],
                xyz[p_k],
                bond[:, idx],
                angle[:, idx],
                dihedral[:, idx],
            )
            log_jacobian = log_jacobian + logabsdet

        ## collect the Cartesian coordinates of particles in the order
        ## of particle index
        particles = list(xyz.keys())
        particles.sort()
        coor = torch.stack([xyz[particle] for particle in particles], dim=1)

        return coor, log_jacobian

    def compute_xyz_from_internal_coordinate(
        self,
        particle_1_xyz,
        particle_2_bond,
        particle_3_bond,
        particle_3_angle,
        bond,
        angle,
        dihedral,
    ):
        """Compute the Cartesian coordinates of the system based on internal
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

        xyz[self.ref_particle_2] = particle_1_xyz + torch.cat(
            [particle_2_bond.new_zeros(batch_size, 2), particle_2_bond], dim=-1
        )

        xyz[self.ref_particle_3] = particle_1_xyz + torch.cat(
            [
                particle_3_bond.new_zeros(batch_size, 1),
                particle_3_bond * torch.sin(particle_3_angle),
                particle_3_bond * torch.cos(particle_3_angle),
            ],
            dim=-1,
        )

        log_jacobian = torch.log(torch.squeeze(particle_3_bond))

        ## Cartesian coordinates of other particles
        for idx in range(len(self.particle_visited_in_order)):
            p = self.particle_visited_in_order[idx]
            p_i, p_j, p_k, p_l = self.dihedral_particle_idx[p]
            assert p == p_l
            xyz[p_l], logabsdet = functional._compute_particle_xyz_from_internal(
                xyz[p_i],
                xyz[p_j],
                xyz[p_k],
                bond[:, idx],
                angle[:, idx],
                dihedral[:, idx],
            )
            log_jacobian = log_jacobian + logabsdet

        ## collect the Cartesian coordinates of particles in the order
        ## of particle index
        particles = list(xyz.keys())
        particles.sort()
        coor = torch.stack([xyz[particle] for particle in particles], dim=1)

        return coor, log_jacobian

    def compute_internal_coordinate_and_orientation_from_xyz(self, xyz):
        """Compute the internal coordinates from an xyz array

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
            functional._compute_distances(
                xyz, torch.tensor([[self.ref_particle_1, self.ref_particle_2]])
            )
        )

        ref_p_2_polar_angle = torch.squeeze(
            functional._compute_polar_angle(
                xyz, torch.tensor([[self.ref_particle_1, self.ref_particle_2]])
            )
        )

        ref_p_2_azimuthal_angle = torch.squeeze(
            functional._compute_azimuthal_angle(
                xyz, torch.tensor([[self.ref_particle_1, self.ref_particle_2]])
            )
        )

        ref_p_3_bond = torch.squeeze(
            functional._compute_distances(
                xyz, torch.tensor([[self.ref_particle_1, self.ref_particle_3]])
            )
        )

        ref_p_3_polar_angle = torch.squeeze(
            functional._compute_polar_angle(
                xyz, torch.tensor([[self.ref_particle_1, self.ref_particle_3]])
            )
        )

        ref_p_3_azimuthal_angle = torch.squeeze(
            functional._compute_azimuthal_angle(
                xyz, torch.tensor([[self.ref_particle_1, self.ref_particle_3]])
            )
        )

        bond = functional._compute_distances(
            xyz,
            torch.tensor(
                [self.bond_particle_idx[p] for p in self.particle_visited_in_order]
            ),
        )
        angle = functional._compute_angles(
            xyz,
            torch.tensor(
                [self.angle_particle_idx[p] for p in self.particle_visited_in_order]
            ),
        )

        dihedral = functional._compute_dihedrals(
            xyz,
            torch.tensor(
                [self.dihedral_particle_idx[p] for p in self.particle_visited_in_order]
            ),
        )

        internal_coor = InternalCoordinateAndOrientation(
            reference_particle_1_xyz=ref_p_1_xyz,
            reference_particle_2_bond=ref_p_2_bond,
            reference_particle_2_polar_angle=ref_p_2_polar_angle,
            reference_particle_2_azimuthal_angle=ref_p_2_azimuthal_angle,
            reference_particle_3_bond=ref_p_3_bond,
            reference_particle_3_polar_angle=ref_p_3_polar_angle,
            reference_particle_3_azimuthal_angle=ref_p_3_azimuthal_angle,
            bond=bond,
            angle=angle,
            dihedral=dihedral,
        )

        ## compute logabsdet of the transform
        log_absdet = -torch.log(
            torch.abs(ref_p_2_bond**2 * torch.sin(ref_p_2_polar_angle))
        )
        log_absdet += -torch.log(
            torch.abs(ref_p_3_bond**2 * torch.sin(ref_p_3_polar_angle))
        )
        log_absdet += -torch.sum(
            torch.log(torch.abs(bond**2 * torch.sin(math.pi - angle))), -1
        )

        return internal_coor, log_absdet

    def compute_internal_coordinate_from_xyz(self, xyz):
        """Compute the internal coordinates from an xyz array

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
            functional._compute_distances(
                xyz, torch.tensor([[self.ref_particle_1, self.ref_particle_2]])
            )
        )

        ref_p_3_bond = torch.squeeze(
            functional._compute_distances(
                xyz, torch.tensor([[self.ref_particle_1, self.ref_particle_3]])
            )
        )

        ref_p_3_angle = torch.squeeze(
            functional._compute_angles(
                xyz,
                torch.tensor(
                    [[self.ref_particle_2, self.ref_particle_1, self.ref_particle_3]]
                ),
            )
        )

        bond = functional._compute_distances(
            xyz,
            torch.tensor(
                [self.bond_particle_idx[p] for p in self.particle_visited_in_order]
            ),
        )
        angle = functional._compute_angles(
            xyz,
            torch.tensor(
                [self.angle_particle_idx[p] for p in self.particle_visited_in_order]
            ),
        )

        dihedral = functional._compute_dihedrals(
            xyz,
            torch.tensor(
                [self.dihedral_particle_idx[p] for p in self.particle_visited_in_order]
            ),
        )

        internal_coor = InternalCoordinate(
            reference_particle_1_xyz=ref_p_1_xyz,
            reference_particle_2_bond=ref_p_2_bond,
            reference_particle_3_bond=ref_p_3_bond,
            reference_particle_3_angle=ref_p_3_angle,
            bond=bond,
            angle=angle,
            dihedral=dihedral,
        )

        ## compute logabsdet of the transform
        log_absdet = -torch.log(ref_p_3_bond)
        log_absdet = log_absdet - torch.sum(
            torch.log(torch.abs(bond**2 * torch.sin(math.pi - angle))), -1
        )

        return internal_coor, log_absdet


class InternalCoordinateAndOrientation:
    def __init__(
        self,
        reference_particle_1_xyz=None,
        reference_particle_2_bond=None,
        reference_particle_2_polar_angle=None,
        reference_particle_2_azimuthal_angle=None,
        reference_particle_3_bond=None,
        reference_particle_3_polar_angle=None,
        reference_particle_3_azimuthal_angle=None,
        bond=None,
        angle=None,
        dihedral=None,
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
        self.reference_particle_2_polar_angle = (
            self.reference_particle_2_polar_angle.cuda()
        )
        self.reference_particle_2_azimuthal_angle = (
            self.reference_particle_2_azimuthal_angle.cuda()
        )
        self.reference_particle_3_bond = self.reference_particle_3_bond.cuda()
        self.reference_particle_3_polar_angle = (
            self.reference_particle_3_polar_angle.cuda()
        )
        self.reference_particle_3_azimuthal_angle = (
            self.reference_particle_3_azimuthal_angle.cuda()
        )
        self.bond = self.bond.cuda()
        self.angle = self.angle.cuda()
        self.dihedral = self.dihedral.cuda()

    def double(self):
        self.reference_particle_1_xyz = self.reference_particle_1_xyz.double()
        self.reference_particle_2_bond = self.reference_particle_2_bond.double()
        self.reference_particle_2_polar_angle = (
            self.reference_particle_2_polar_angle.double()
        )
        self.reference_particle_2_azimuthal_angle = (
            self.reference_particle_2_azimuthal_angle.double()
        )
        self.reference_particle_3_bond = self.reference_particle_3_bond.double()
        self.reference_particle_3_polar_angle = (
            self.reference_particle_3_polar_angle.double()
        )
        self.reference_particle_3_azimuthal_angle = (
            self.reference_particle_3_azimuthal_angle.double()
        )
        self.bond = self.bond.double()
        self.angle = self.angle.double()
        self.dihedral = self.dihedral.double()

    def float(self):
        self.reference_particle_1_xyz = self.reference_particle_1_xyz.float()
        self.reference_particle_2_bond = self.reference_particle_2_bond.float()
        self.reference_particle_2_polar_angle = (
            self.reference_particle_2_polar_angle.float()
        )
        self.reference_particle_2_azimuthal_angle = (
            self.reference_particle_2_azimuthal_angle.float()
        )
        self.reference_particle_3_bond = self.reference_particle_3_bond.float()
        self.reference_particle_3_polar_angle = (
            self.reference_particle_3_polar_angle.float()
        )
        self.reference_particle_3_azimuthal_angle = (
            self.reference_particle_3_azimuthal_angle.float()
        )
        self.bond = self.bond.float()
        self.angle = self.angle.float()
        self.dihedral = self.dihedral.float()

    def numpy(self):
        self.reference_particle_1_xyz = self.reference_particle_1_xyz.numpy()
        self.reference_particle_2_bond = self.reference_particle_2_bond.numpy()
        self.reference_particle_2_polar_angle = (
            self.reference_particle_2_polar_angle.numpy()
        )
        self.reference_particle_2_azimuthal_angle = (
            self.reference_particle_2_azimuthal_angle.numpy()
        )
        self.reference_particle_3_bond = self.reference_particle_3_bond.numpy()
        self.reference_particle_3_polar_angle = (
            self.reference_particle_3_polar_angle.numpy()
        )
        self.reference_particle_3_azimuthal_angle = (
            self.reference_particle_3_azimuthal_angle.numpy()
        )
        self.bond = self.bond.numpy()
        self.angle = self.angle.numpy()
        self.dihedral = self.dihedral.numpy()

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
            self.dihedral[index],
        )
        return ic


class InternalCoordinate:
    def __init__(
        self,
        reference_particle_1_xyz=None,
        reference_particle_2_bond=None,
        reference_particle_3_bond=None,
        reference_particle_3_angle=None,
        bond=None,
        angle=None,
        dihedral=None,
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

    def numpy(self):
        self.reference_particle_1_xyz = self.reference_particle_1_xyz.numpy()
        self.reference_particle_2_bond = self.reference_particle_2_bond.numpy()
        self.reference_particle_3_bond = self.reference_particle_3_bond.numpy()
        self.reference_particle_3_angle = self.reference_particle_3_angle.numpy()
        self.bond = self.bond.numpy()
        self.angle = self.angle.numpy()
        self.dihedral = self.dihedral.numpy()

    def __len__(self):
        return self.reference_particle_1_xyz.shape[0]

    def __getitem__(self, index):
        ic = InternalCoordinate(
            self.reference_particle_1_xyz[index],
            self.reference_particle_2_bond[index],
            self.reference_particle_3_bond[index],
            self.reference_particle_3_angle[index],
            self.bond[index],
            self.angle[index],
            self.dihedral[index],
        )
        return ic

    def plot(self, file_name, weights=None):
        if not file_name.lower().endswith(".pdf"):
            raise ValueError("file_name needs to end with .pdf!")

        with PdfPages(file_name) as pdf:
            fig = plt.figure(figsize=(6.4 * 3, 4.8))
            fig.clf()
            plt.subplot(1, 3, 1)
            plt.hist(
                self.reference_particle_2_bond.numpy(),
                bins=50,
                density=True,
                label=f"reference_particle_2_bond",
                weights=weights,
            )
            plt.legend()

            plt.subplot(1, 3, 2)
            plt.hist(
                self.reference_particle_3_bond.numpy(),
                bins=50,
                density=True,
                label=f"reference_particle_3_bond",
                weights=weights,
            )
            plt.legend()

            plt.subplot(1, 3, 3)
            plt.hist(
                self.reference_particle_3_angle.numpy(),
                bins=50,
                density=True,
                label=f"reference_particle_3_angle",
                weights=weights,
                range=[0, np.pi],
            )
            plt.legend()

            plt.tight_layout()
            pdf.savefig(fig)
            plt.close()

            for j in range(self.bond.shape[1]):
                fig = plt.figure(figsize=(6.4 * 3, 4.8))
                fig.clf()
                plt.subplot(1, 3, 1)
                plt.hist(
                    self.bond[:, j].numpy(),
                    bins=50,
                    density=True,
                    label=f"bond: {j}",
                    weights=weights,
                )
                plt.legend()

                plt.subplot(1, 3, 2)
                plt.hist(
                    self.angle[:, j].numpy(),
                    bins=50,
                    density=True,
                    label=f"angle: {j}",
                    weights=weights,
                    range=[0, np.pi],
                )
                plt.legend()

                plt.subplot(1, 3, 3)
                plt.hist(
                    self.dihedral[:, j].numpy(),
                    bins=50,
                    density=True,
                    label=f"dihedral: {j}",
                    weights=weights,
                    range=[-np.pi, np.pi],
                )
                plt.legend()

                plt.tight_layout()
                pdf.savefig(fig)
                plt.close()
