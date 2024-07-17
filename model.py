import torch as th
import torch.nn as nn
from blocks import Block, Down, Up, OutUp

class UNET(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(UNET, self).__init__()
        self.in_channels = in_channels 
        self.num_classes = num_classes 

        self.down1 = Block(self.in_channels, 64)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 256)
        self.down4 = Down(256, 512)
        self.down5 = Down(512, 1024)

        self.up1 = Up(1024, 512)
        self.up2 = Up(512, 256)
        self.up3 = Up(256, 128)
        self.up4 = Up(128, 64)

        self.up5 = OutUp(64, self.num_classes)
        

    def forward(self, x):
        # Contracting Path
        out1 = self.down1(x)
        out2 = self.down2(out1)
        out3 = self.down3(out2)
        out4 = self.down4(out3) 
        out5 = self.down5(out4) 

        # Expanding Path
        out6 = self.up1(out4, out5)
        out7 = self.up2(out3, out6)
        out8 = self.up3(out2, out7)
        out9 = self.up4(out1, out8)
        out = self.up5(out9)
        return out 




