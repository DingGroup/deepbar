import torch
import torch.nn as nn
import torch.distributions as distributions
import MMFlow.MixedRationalQuadraticCouplingTransform \
    as MixedRationalQuadraticCouplingTransform

class MMFlow(nn.Module):
    def __init__(self,
                 feature_size,
                 context_size,
                 circular_feature_flag,
                 transform_feature_flag_list,
                 conditioner_net_create_fn,
                 num_bins_circular
                 num_bins_regular):
        super(MMFlow, self).__init__()

        self.feature_size = feature_size
        self.context_size = context_size

        self.register_buffer('circular_feature_flag', torch.tensor(circular_feature_flag))
        
        transform_feature_flag = torch.tensor(
            [torch.tensor(flag) for flag in transform_feature_flag_list]
        )
        self.register_buffer('transform_feature_flag', transform_feature_flag)
        
        self.conditioner_net_create_fn = conditioner_net_create_fn
        self.num_bins_circular = num_bins_circular
        self.num_bins_regular = num_bins_regular

        ## uniform distributions are used as base distributions
        ## for circular features, the base distribution is U[-pi, pi]
        ## for regular features, the base distribution is U[0, 1]
        base_dist_low = []
        base_dist_high = []
        for flag in circular_feature_flag:
            if flag > 0:
                base_dist_low.append(-math.pi)
                base_dist_high.append(math.pi)
            else:
                base_dist_low.append(0.0)
                base_dist_high.append(1.0)
                
        self.register_buffer('base_dist_low', base_dist_low)
        self.register_buffer('base_dist_high', base_dist_high)
        
        ## make transforms
        self.transforms = nn.ModuleList([])
        for i in range(self.transform_feature_flag.shape[0]):
            self.transforms.append(
                MixedRationalQuadraticCouplingTransform(
                    feature_size = feature_size,
                    context_size = context_size,
                    circular_feature_flag = self.circular_feature_flag,
                    transform_feature_flag = self.transform_feature_flag[i],
                    conditioner_net_create_fn = self.conditioner_net_create_fn,
                    num_bins_circular = num_bins_circular,
                    num_bins_regular = num_bins_regular
                )
            )
        
        
