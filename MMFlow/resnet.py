import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import init

class ResidualBlock(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.input_size = input_size
        self.linear_layers = nn.ModuleList(
            [nn.Linear(input_size, input_size) for _ in range(2)]
        )
        
    def forward(self, inputs):
        x = inputs
        x = self.linear_layers[0](x)
        x = F.relu(x)
        x = self.linear_layers[1](x)
        x = x + inputs
        outputs = F.relu(x)
        return outputs

class ResidualNet(nn.Module):
    def __init__( self,
                  input_feature_size,
                  context_size,
                  output_size,
                  hidden_size,
                  num_blocks=2
    ):        
        super().__init__()
        self.input_feature_size = input_feature_size
        self.context_size = context_size        
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.num_blocks = num_blocks

        input_size = input_feature_size
        if context_size is not None:
            input_size = input_feature_size + context_size
        
        self.blocks = nn.ModuleList(
            [ResidualBlock(input_size) for _ in range(num_blocks)]
        )

        self.final_layer = nn.Linear(input_size, output_size)

    def forward(self, feature, context=None):
        inputs = feature
        if self.context_size is not None:
            inputs = torch.cat([feature, context], -1)
            
        x = inputs
        for block in self.blocks:
            x = block(x)
        outputs = F.relu(self.final_layer(x))
        
        return outputs

    
