import torch as th
import torch.nn as nn

class Residual(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x1, x2):
        assert x1.shape == x2.shape, f'The shape of two feature maps that you want to concat must be same, but got {x1.shape}, {x2.shape}.'
        return th.cat((x1, x2), dim = 0) 

