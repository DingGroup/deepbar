import math
import torch
import torch.nn as nn
from MMFlow.transform.rational_quadratic_spline import *
from MMFlow.transform.resnet import *

class MixedRationalQuadraticCouplingTransform(nn.Module):
    def __init__(self,
                 feature_size,
                 context_size,
                 circular_feature_flag,
                 transform_feature_flag,
                 conditioner_net_create_fn,
                 num_bins_circular = 5,
                 num_bins_regular = 5):
        
        super(MixedRationalQuadraticCouplingTransform, self).__init__()

        self.feature_size = feature_size
        self.context_size = context_size
        
        self.num_bins_circular = num_bins_circular
        self.num_bins_regular = num_bins_regular

        self.register_buffer('circular_feature_flag', torch.as_tensor(circular_feature_flag))
        self.register_buffer('transform_feature_flag', torch.as_tensor(transform_feature_flag))
        feature_index = torch.arange(feature_size)

        self.register_buffer(
            'identity_circular_feature_index',
            feature_index.masked_select(
                (transform_feature_flag <= 0) & (circular_feature_flag > 0))
        )

        self.register_buffer(
            'identity_regular_feature_index',
            feature_index.masked_select(
                (transform_feature_flag <= 0) & (circular_feature_flag <= 0))
        )
        
        self.register_buffer(
            'transform_circular_feature_index',
            feature_index.masked_select(
                (transform_feature_flag > 0) & (circular_feature_flag > 0))
        )

        self.register_buffer(
            'transform_regular_feature_index',
            feature_index.masked_select(
                (transform_feature_flag > 0) & (circular_feature_flag <= 0))
        )

        conditioner_net_input_size = \
            len(self.identity_circular_feature_index)*2 + \
            len(self.identity_regular_feature_index)
        conditioner_net_output_size = \
            len(self.transform_circular_feature_index)*(self.num_bins_circular*3) + \
            len(self.transform_regular_feature_index)*(self.num_bins_regular*3 + 1)
        
        self.conditioner_net = conditioner_net_create_fn(
            conditioner_net_input_size,
            context_size,
            conditioner_net_output_size
        )
        
    def forward(self, inputs, context, inverse = False):
        ## split inputs into four categories
        identity_circular_inputs = torch.index_select(inputs, -1, self.identity_circular_feature_index)
        identity_regular_inputs = torch.index_select(inputs, -1, self.identity_regular_feature_index)

        transform_circular_inputs = torch.index_select(inputs, -1, self.transform_circular_feature_index)
        transform_regular_inputs = torch.index_select(inputs, -1, self.transform_regular_feature_index)
        
        ## expand identity circular inputs and combine it with identity regular inputs
        ## to get the inputs for the conditioner network
        identity_circular_inputs_expand = torch.cat([torch.cos(identity_circular_inputs),
                                                     torch.sin(identity_circular_inputs)],
                                                     dim = -1)
        conditioner_net_inputs = torch.cat([identity_circular_inputs_expand,
                                           identity_regular_inputs],
                                          dim = -1)

        ## compute conditioner
        conditioner_net_outputs = self.conditioner_net(conditioner_net_inputs, context)
        conditioner_circular = conditioner_net_outputs[..., 0:len(self.transform_circular_feature_index)*(self.num_bins_circular*3)]
        conditioner_circular = conditioner_circular.reshape(inputs.shape[0], len(self.transform_circular_feature_index), self.num_bins_circular*3)
        conditioner_regular = conditioner_net_outputs[..., len(self.transform_circular_feature_index)*(self.num_bins_circular*3):]
        conditioner_regular = conditioner_regular.reshape(inputs.shape[0], len(self.transform_regular_feature_index), self.num_bins_regular*3+1)
        
        ## transform regular feature
        if conditioner_regular.nelement() > 0:
            unnormalized_widths = conditioner_regular[..., 0:self.num_bins_regular]
            unnormalized_heights = conditioner_regular[..., self.num_bins_regular:2*self.num_bins_regular]
            unnormalized_derivatives = conditioner_regular[..., 2*self.num_bins_regular:]

            unnormalized_widths = unnormalized_widths / np.sqrt(self.conditioner_net.hidden_size)
            unnormalized_heights = unnormalized_heights / np.sqrt(self.conditioner_net.hidden_size)

            regular_outputs, regular_logabsdet = rational_quadratic_spline(
                transform_regular_inputs,
                unnormalized_widths = unnormalized_widths,
                unnormalized_heights = unnormalized_heights,
                unnormalized_derivatives = unnormalized_derivatives,
                inverse = inverse,
                left = 0.0, right = 1.0,
                bottom = 0.0, top = 1.0)
        else:
            regular_outputs = conditioner_net_outputs.new_zeros(inputs.shape[0], len(self.transform_regular_feature_index))
            regular_logabsdet = conditioner_net_outputs.new_zeros(inputs.shape[0], len(self.transform_regular_feature_index))
        
        ## transform circular feature
        if conditioner_circular.nelement() > 0:
            unnormalized_widths = conditioner_circular[..., 0:self.num_bins_circular]
            unnormalized_heights = conditioner_circular[..., self.num_bins_circular:2*self.num_bins_circular]
            unnormalized_derivatives = conditioner_circular[..., 2*self.num_bins_circular:]
            unnormalized_derivatives = torch.cat([unnormalized_derivatives,
                                                  unnormalized_derivatives[..., 0][..., None]],
                                                 dim = -1)

            unnormalized_widths = unnormalized_widths / np.sqrt(self.conditioner_net.hidden_size)
            unnormalized_heights = unnormalized_heights / np.sqrt(self.conditioner_net.hidden_size)

            circular_outputs, circular_logabsdet = rational_quadratic_spline(
                transform_circular_inputs,
                unnormalized_widths = unnormalized_widths,
                unnormalized_heights = unnormalized_heights,
                unnormalized_derivatives = unnormalized_derivatives,
                inverse = inverse,
                left = -math.pi, right = math.pi,
                bottom = -math.pi, top = math.pi)
        else:
            circular_outputs = conditioner_net_outputs.new_zeros(inputs.shape[0], len(self.transform_circular_feature_index))
            circular_logabsdet = conditioner_net_outputs.new_zeros(inputs.shape[0], len(self.transform_circular_feature_index))
                    
        ## collect outputs
        outputs = torch.empty_like(inputs)
        outputs[:, self.identity_regular_feature_index] = identity_regular_inputs
        outputs[:, self.identity_circular_feature_index] = identity_circular_inputs
        outputs[:, self.transform_regular_feature_index] = regular_outputs
        outputs[:, self.transform_circular_feature_index] = circular_outputs

        logabsdet = torch.cat([regular_logabsdet, circular_logabsdet], -1)
        logabsdet = torch.sum(logabsdet, -1)
        return outputs, logabsdet
    
if __name__ == "__main__":
    feature_size = 10
    context_size = 5
    circular_feature_flag = torch.LongTensor([i%3 for i in range(feature_size)])
    transform_feature_flag = torch.LongTensor([i%2 for i in range(feature_size)])

    conditioner_net_create_fn = lambda input_feature_size, context_size, output_size: \
        ResidualNet(input_feature_size, context_size, output_size,
                    hidden_size = 8, num_blocks = 2)
    
    transform = MixedRationalQuadraticCouplingTransform(
        feature_size, context_size,
        circular_feature_flag, transform_feature_flag,
        conditioner_net_create_fn
    )

    batch_size = 3
    feature = torch.rand((batch_size, feature_size))
    context = torch.rand((batch_size, context_size))
    
    with torch.no_grad():
        outputs, logabsdet = transform(feature, context)
        inputs, logabsdet_i = transform(outputs, context, inverse = True)
