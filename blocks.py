import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torchvision.transforms import CenterCrop 

class Block(nn.Module):
      """Conv - ReLU - Conv - ReLU"""
      def __init__(self, in_channels, out_channels):
            super(Block, self).__init__()
            self.in_channels = in_channels 
            self.out_channels = out_channels 

            self.block = nn.Sequential(
                  nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                  nn.ReLU(),
                  nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                  nn.ReLU() #TODO : Add BatchNorm2d
            )

      def forward(self, x):
            return self.block(x)
     
class Down(nn.Module):
      """
      Contracting path block
      2x2Maxpooling(s=2) -> Block(Conv - ReLU - Conv - ReLU)
      """
      def __init__(self, in_channels, out_channels):
            super(Down, self).__init__()
            self.in_channels = in_channels 
            self.out_channels = out_channels 

            self.down_block = nn.Sequential(
                        nn.MaxPool2d(kernel_size=2, stride=2),
                        Block(self.in_channels, self.out_channels)
                  )
      def forward(self, x):
            return self.down_block(x) 

class Up(nn.Module):
      """
      Expanding path block
      2x2deconv - concat - Block(Conv - ReLU - Conv - ReLU)
      """
      def __init__(self, in_channels, out_channels):
            super(Up, self).__init__()
            self.in_channels = in_channels 
            self.out_channels = out_channels 
            self.deconv = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size= 2, stride = 2) # ex. (52 - 1)*2+2=104   <- W_out = (W_in - 1)xS + K
            self.block = Block(self.in_channels, self.out_channels)
            
      def forward(self, down, up):
          up = self.deconv(up)

          d_h, d_w = down.size()[2], down.size()[3] # [batch, channel, height, width]
          u_h, u_w = up.size()[2], up.size()[3]

          diff_w = (int)((d_w - u_w) / 2)
          diff_h = (int)((d_h - u_h) / 2)

          down = F.pad(down, (diff_h, u_h - diff_h, diff_w, u_w - diff_w))
          down = self.crop(u_h, u_w, down)
          assert down.size()[2] == up.size()[2] and down.size()[3] == up.size()[3], \
            f'The size of two feature maps that you want to concat must be same, but got {down.shape}, {up.shape}.'
          
          up = th.cat((down, up), dim = 1) # concat on channels (dim = 1)
          up = self.block(up)

          return up 
      
      def crop(self, u_h, u_w, down):
            center_crop = CenterCrop([u_h, u_w])
            down = center_crop(down)
            return down 

class OutUp(nn.Module):
      def __init__(self, in_channels, out_channels): 
            super(OutUp, self).__init__()
            self.in_channels = in_channels 
            self.out_channels = out_channels 
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
      def forward(self, x):
            return self.conv(x)
            
