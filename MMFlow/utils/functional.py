from collections import defaultdict
import torch
import math
import numpy.linalg as linalg
import mdtraj

def get_bonded_atoms(topology):
    """ Return a dictionary of bonded atoms

    Parameters
    ----------
    topology : mdtraj.Topology

    Returns:
    -------
    out: dict
        out[i] = list of atom indices that are bonded with atom i.
        If atom i is not bonded to any other atoms, out[i] will return
        KeyError because the keys of out does not include i.
    """
    
    bonded_atoms = defaultdict(list)
    for bond in topology.bonds:
        atom1_index = bond.atom1.index
        atom2_index = bond.atom2.index
        bonded_atoms[atom1_index].append(atom2_index)
        bonded_atoms[atom2_index].append(atom1_index)
    return dict(bonded_atoms)

def compute_particle_xyz_from_internal(particle_1_xyz,
                                       particle_2_xyz,
                                       particle_3_xyz,
                                       bond, angle, dihedral):
    """ Compute the Cartesian coordinate of a particle based on its internal
    coordinate with respect to three other particles and the Cartesian 
    coordinates of these three particles. This function runs a batch mode, which
    mean that it compute the Cartensian coordinates of a batch of particles based
    on a batch of three other particles.

    Parameters:
    -----------
    particle_1_xyz: torch.Tensor 
        the Cartesian coordinate of atom 1.
    particle_2_xyz: torch.Tensor 
        the Cartesian coordinate of atom 2.
    particle_3_xyz: torch.Tensor 
        the Cartesian coordinate of atom 3.
        Note particle_1_xyz, particle_2_xyz and particle_3_xyz should the same size: [batch_size, 3].
    bond: torch.Tensor
        the length of the bond between atom 3 and atom 4
    angle: torch.Tensor
        the value of the angle between atom 2, atom 3 and atom 4
    dihedral: torch.Tensor
        the value of the dihedral angle between atom 1, atom 2, atom3 and atom 4
        Note bond, angle and dihedral have the same size: [batch_size]

    Returns:
    --------
    particle_4_xyz: torch.Tensor
        the Cartensian coordiante of atom 4. Its size is [batch_size, 3]
    logabsdet: the logarithm of the absolute value of the determinant of
        the transformation from the internal coordinates of particle 4 
        to its Cartesian coordinates
    """

    ## calculate the coordinate of the forth atom
    wi, wj, wk = particle_1_xyz, particle_2_xyz, particle_3_xyz

    e1 = wk - wj
    e1 = e1 / torch.norm(e1, dim = -1, keepdim = True)

    e3 = torch.cross(e1, wj - wi, dim = -1)
    e3 = e3 / torch.norm(e3, dim = -1, keepdim = True)

    e2 = torch.cross(e3, e1, dim = -1)

    bond = torch.unsqueeze(bond, -1)
    angle = torch.unsqueeze(angle, -1)
    dihedral = torch.unsqueeze(dihedral, -1)
    
    dw = torch.cos(math.pi-angle)*e1 + \
         (-1)*torch.sin(math.pi-angle)*torch.cos(dihedral)*e2 + \
         (-1)*torch.sin(math.pi-angle)*torch.sin(dihedral)*e3
    
    wl = wk + bond*dw

    particle_4_xyz = wl

    ## calculate the Jacobian of the transform
    logabsdet = torch.log(torch.abs(bond**2*torch.sin(math.pi - angle)))
    logabsdet = torch.squeeze(logabsdet)
    
    return particle_4_xyz, logabsdet


def compute_distances(xyz, particle_index):
    """ Compute distances between sets of two particles.
    
    Parameters:
    -----------
    xyz: torch.Tensor 
        input tensor of shape :math:`(\text{frames} , \text{particles} , 3)`
    particle_index: torch.LongTensor
        particle index tensor of shape :math:`(\text{*} , 2)`

    """

    xyz_i = torch.index_select(xyz, 1, particle_index[:, 0])
    xyz_j = torch.index_select(xyz, 1, particle_index[:, 1])
    
    distances = torch.sqrt(torch.sum((xyz_j - xyz_i)**2, -1))
    return distances

def compute_angles(xyz, particle_index):
    """ Compute angles between sets of three particles.

    Parameters:
    -----------
    xyz: torch.Tensor 
        input tensor of shape :math:`(\text{frames} , \text{particles} , 3)`
    particle_index: torch.LongTensor
        particle index tensor of shape :math:`(\text{*} , 3)`

    """

    xyz_i = torch.index_select(xyz, 1, particle_index[:, 0])
    xyz_j = torch.index_select(xyz, 1, particle_index[:, 1])
    xyz_k = torch.index_select(xyz, 1, particle_index[:, 2])    

    v = xyz_i - xyz_j
    w = xyz_k - xyz_j

    v = v / torch.sqrt(torch.sum(v**2, -1, keepdim = True))
    w = w / torch.sqrt(torch.sum(w**2, -1, keepdim = True))

    inner_product = torch.sum(v*w, -1)
    angles = torch.acos(inner_product)

    return angles

def compute_dihedrals(xyz, particle_index):
    """ Compute dihedral angles between sets of four particles.

    Parameters:
    -----------
    xyz: torch.Tensor 
        input tensor of shape :math:`(\text{frames} , \text{particles} , 3)`
    particle_index: torch.LongTensor
        particle index tensor of shape :math:`(\text{*} , 4)`

    """

    xyz_i = torch.index_select(xyz, 1, particle_index[:, 0])
    xyz_j = torch.index_select(xyz, 1, particle_index[:, 1])
    xyz_k = torch.index_select(xyz, 1, particle_index[:, 2])
    xyz_l = torch.index_select(xyz, 1, particle_index[:, 3])    

    b1 = xyz_i - xyz_j
    b2 = xyz_j - xyz_k
    b3 = xyz_l - xyz_k

    b1_cross_b2 = torch.cross(b1, b2)
    b3_cross_b2 = torch.cross(b3, b2)

    cos_d = torch.norm(b2, dim = -1)*torch.sum(b1_cross_b2*b3_cross_b2, -1)
    sin_d = torch.sum(b2*torch.cross(b3_cross_b2, b1_cross_b2), -1)
    
    dihedrals = torch.atan2(sin_d, cos_d)

    return dihedrals

def compute_polar_angle(xyz, particle_index):
    """ Compute the polar angles of bond vectors.
    
    Parameters:
    -----------
    xyz: torch.Tensor 
        input tensor of shape :math:`(\text{frames} , \text{particles} , 3)`
    particle_index: torch.LongTensor
        particle index tensor of shape :math:`(\text{*} , 2)`
    """
    
    xyz_i = torch.index_select(xyz, 1, particle_index[:, 0])
    xyz_j = torch.index_select(xyz, 1, particle_index[:, 1])

    v = xyz_j - xyz_i
    v = v / torch.sqrt(torch.sum(v**2, -1, keepdim = True))
    z = torch.index_select(v, -1, torch.tensor([2], device = v.device))
    
    angles = torch.acos(z)

    return angles

def compute_azimuthal_angle(xyz, particle_index):
    """ Compute the azimuthal angles of bond vectors with respect to the x-plus axis.
    
    Parameters:
    -----------
    xyz: torch.Tensor 
        input tensor of shape :math:`(\text{frames} , \text{particles} , 3)`
    particle_index: torch.LongTensor
        particle index tensor of shape :math:`(\text{*} , 2)`
    """
    
    xyz_i = torch.index_select(xyz, 1, particle_index[:, 0])
    xyz_j = torch.index_select(xyz, 1, particle_index[:, 1])

    v = xyz_j - xyz_i
    
    x = torch.index_select(v, -1, torch.tensor([0], device = v.device))
    y = torch.index_select(v, -1, torch.tensor([1], device = v.device))

    angles = torch.atan2(y, x)
    
    return angles
