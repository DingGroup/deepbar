import torch
import sys
sys.path.append("../../")
from MMFlow import utils

def ic_to_feature_and_context(ic, ic_limits):
    reference_particle_3_angle = (ic.reference_particle_3_angle - ic_limits['reference_particle_3_angle_min']) / (ic_limits['reference_particle_3_angle_max'] - ic_limits['reference_particle_3_angle_min'])
    reference_particle_3_angle = reference_particle_3_angle*0.95 + 0.025

    angle = (ic.angle - ic_limits['angle_min']) / (ic_limits['angle_max'] - ic_limits['angle_min'])
    angle = angle*0.95 + 0.025
    
    feature = torch.cat([reference_particle_3_angle[:, None],
                         angle,
                         ic.dihedral], -1)
    
    feature_size = feature.shape[-1]
    circular_feature_flag = torch.zeros(feature_size)
    circular_feature_flag[-ic.dihedral.shape[-1]:] = 1
    
    context = torch.cat([ic.reference_particle_2_bond[:, None],
                         ic.reference_particle_3_bond[:, None],
                         ic.bond], -1)

    logabsdet_jacobian = torch.log(0.95/(ic_limits['reference_particle_3_angle_max']-ic_limits['reference_particle_3_angle_min'])) + \
        torch.sum(torch.log(0.95/(ic_limits['angle_max'] - ic_limits['angle_min'])))
    
    return circular_feature_flag, feature, context, logabsdet_jacobian


def feature_and_context_to_ic(feature, context, ic_limits):
    reference_particle_1_xyz = context.new_zeros((context.shape[0], 3))
    
    reference_particle_2_bond = context[:, 0]
    reference_particle_3_bond = context[:, 1]
    bond = context[:, 2:]

    reference_particle_3_angle = feature[:, 0] 
    reference_particle_3_angle = (reference_particle_3_angle - 0.025)/0.95
    reference_particle_3_angle = ic_limits['reference_particle_3_angle_min'] + reference_particle_3_angle*(ic_limits['reference_particle_3_angle_max'] - ic_limits['reference_particle_3_angle_min'])

    angle = feature[:, 1:1+bond.shape[1]]
    angle = (angle - 0.025)/0.95
    angle = ic_limits['angle_min'] + angle*(ic_limits['angle_max'] - ic_limits['angle_min'])
    
    dihedral = feature[:, 1+bond.shape[1]:]

    ic = utils.InternalCoordinate(
        reference_particle_1_xyz,
        reference_particle_2_bond,
        reference_particle_3_bond,
        reference_particle_3_angle,
        bond, angle, dihedral
    )

    logabsdet_jacobian = torch.log((ic_limits['reference_particle_3_angle_max']-ic_limits['reference_particle_3_angle_min'])/0.95) + \
        torch.sum(torch.log((ic_limits['angle_max'] - ic_limits['angle_min'])/0.95))
    
    return ic, logabsdet_jacobian
