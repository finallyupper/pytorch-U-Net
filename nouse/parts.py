import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

class Residual(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, down, up): # 3-dim assumed / tensor 
        assert down.dim() == 3 and up.dim() == 3
        s_d, s_u = down.shape[1], up.shape[1] # width of each feature map
        diff = (int)((s_d - s_u)/2)

        if diff <= 0:# pad
            down = F.pad(down, (diff, diff, diff, diff))

        elif diff > 0:
            center_crop = transforms.CenterCrop((s_u, s_u))
            down = center_crop(down)

        assert down.shape[1:] == up.shape[1:], \
            f'The size of two feature maps that you want to concat must be same, but got {down.shape}, {up.shape}.'
        
        return th.cat((down, up), dim = 0) 

