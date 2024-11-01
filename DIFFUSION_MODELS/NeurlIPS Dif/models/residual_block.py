import torch.nn as nn
from .swish import Swish

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.swish = Swish()
        self.relu = nn.ReLU()
    
    def forward(self, x):
        out = self.swish(self.conv1(x))
        out = self.conv2(out)
        out += x  # Residual connection
        return self.relu(out)