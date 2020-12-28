import math
import torch
import torch.nn as nn
import torch.distributions as distributions
from MMFlow.transform.coupling_transform import MixedRationalQuadraticCouplingTransform

class MMFlow(nn.Module):
    def __init__(self,
                 feature_size,
                 context_size,
                 circular_feature_flag,
                 transform_feature_flag,
                 conditioner_net_create_fn,
                 num_bins_circular,
                 num_bins_regular):
        super(MMFlow, self).__init__()

        self.feature_size = feature_size
        self.context_size = context_size

        self.register_buffer('circular_feature_flag', torch.as_tensor(circular_feature_flag))
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
                
        self.register_buffer('base_dist_low', torch.tensor(base_dist_low))
        self.register_buffer('base_dist_high', torch.tensor(base_dist_high))
        
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
        
    def forward(self, feature, context = None):
        z = feature
        logabsdet_tot = 0

        for transform in self.transforms:
            z, logabsdet = transform(z, context, inverse = False)
            logabsdet_tot = logabsdet_tot + logabsdet
            
        return z, logabsdet_tot

    def _inverse(self, z, context = None):
        x = z
        logabsdet_tot = 0

        for transform in self.transforms[-1::-1]:
            x, logabsdet = transform(x, context, inverse = True)
            logabsdet_tot = logabsdet_tot + logabsdet
                    
        return x, logabsdet_tot
            
    def compute_log_prob(self, feature, context = None):
        z, logabsdet = self.forward(feature, context)
        
        base_dist = distributions.Uniform(low = self.base_dist_low,
                                          high = self.base_dist_high)
        base_dist = distributions.Independent(base_dist, 1)
        
        log_prob = base_dist.log_prob(z) + logabsdet        
        return log_prob

    def sample_and_compute_log_prob(self, num_samples, context = None):
        base_dist = distributions.Uniform(low = self.base_dist_low,
                                          high = self.base_dist_high)
        base_dist = distributions.Independent(base_dist, 1)

        if context is None:
            z = base_dist.sample((num_samples,))
            log_prob = base_dist.log_prob(z)

            x, logabsdet = self._inverse(z, None)
            log_prob = log_prob - logabsdet

        else:
            context = context.repeat(num_samples, 1)
            
            z = base_dist.sample((context.shape[0],))
            log_prob = base_dist.log_prob(z)

            x, logabsdet = self._inverse(z, context)
            log_prob = log_prob - logabsdet
        
            x = x.reshape(num_samples, -1, x.shape[-1])
            log_prob = log_prob.reshape(num_samples, -1)
            
        return x, log_prob
