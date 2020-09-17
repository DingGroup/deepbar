import torch
import torch.nn as nn

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
        
        circular_feature_flag = torch.as_tensor(circular_feature_flag)
        transform_feature_flag = torch.as_tensor(transform_feature_flag)

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
        
    def forward(self, inputs, context):
        identity_circular_inputs = torch.index_select(inputs, -1, self.identity_circular_feature_index)
        identity_regular_inputs = torch.index_select(inputs, -1, self.identity_regular_feature_index)

        identity_circular_inputs_expand = torch.cat([torch.cos(identity_circular_inputs),
                                                      torch.sin(identity_circular_inputs)],
                                                     dim = -1)
        
        conditioner_net_inputs = torch.cat([identity_circular_inputs_expand,
                                           identity_regular_inputs],
                                          dim = -1)
        conditioner_net_outputs = self.conditioner_net(conditioner_nn_inputs, context)

        conditioner_circular = conditioner_net_outputs[..., 0:len(self.transform_circular_feature_index)*(self.num_bins_circular*3)]
        conditioner_regular = conditioner_net_output[..., len(self.transform_circular_feature_index)*(self.num_bins_circular*3):]

        ## transform regular feature
        unnormalized_widths = conditioner_regular[..., 0:self.num_bins_regular]
        unnormalized_heights = conditioner_regular[..., self.num_bins_regular:2*self.num_bins_regular]
        unnormalized_derivatives = conditioner_regular[..., 2*self.num_bins_regular :]

        unnormalized_widths /= np.sqrt(self.conditioner_net.hidden_features)
        unnormalized_heights /= np.sqrt(self.conditioner_net.hidden_features)
        
        
        if self.tails is None:
            spline_fn = splines.rational_quadratic_spline
            spline_kwargs = {}
        else:
            spline_fn = splines.unconstrained_rational_quadratic_spline
            spline_kwargs = {"tails": self.tails, "tail_bound": self.tail_bound}

        return spline_fn(
            inputs=inputs,
            unnormalized_widths=unnormalized_widths,
            unnormalized_heights=unnormalized_heights,
            unnormalized_derivatives=unnormalized_derivatives,
            inverse=inverse,
            min_bin_width=self.min_bin_width,
            min_bin_height=self.min_bin_height,
            min_derivative=self.min_derivative,
            **spline_kwargs
        )
        
        
        ## split output from conditioner_net_outputs and apply rational quatratic spline transform
        transform_circular_inputs = torch.index_select(inputs, -1, self.transform_circular_feature_index)
        transform_regular_inputs = torch.index_select(inputs, -1, self.transform_regular_feature_index)
        
        
if __name__ == "__main__":
    feature_size = 10
    context_size = 5
    circular_feature_flag = torch.LongTensor([i%3 for i in range(feature_size)])
    transform_feature_flag = torch.LongTensor([i%2 for i in range(feature_size)])
    transform = MixedRationalQuadraticCouplingTransform(
        feature_size, context_size,
        circular_feature_flag, transform_feature_flag
    )

    batch_size = 22
    feature = torch.randn((batch_size, feature_size))
    context = torch.randn((batch_size, context_size))
    transform(feature, context)
    
    
