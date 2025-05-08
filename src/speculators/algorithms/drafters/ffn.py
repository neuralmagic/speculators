"""
Will define a Linear layer that can accept

Speculator config combines everything together;
Check if only need to accept draft config, or need the entire spec config
"""
from torch import nn

# Use nn.Linear directly
class Linear(nn.Module):
    # to be able to convert the constructor to accept configs
    def __init__(self, input_dim, output_dim):
        self.input_dim = input_dim
        self.output_dim = output_dim

    def forward(self, x):
        """
        """
