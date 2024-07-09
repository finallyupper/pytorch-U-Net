import torch as th
import torch.nn as nn
from blocks import down, up
from parts import Residual

class UNET(nn.Module):
    def __init__(self, num_classes):
        super(UNET, self).__init__()
        self.pool1 = nn.MaxPool2d(2, 2)

        self.down1 = down(1, 64)
        self.down2 = down(64, 128)
        self.down3 = down(128, 256)
        self.down4 = down(256, 512)
        self.down5 = down(512, 1024)

        self.up1 = up(1024, 512)
        self.up2 = up(512, 256)
        self.up3 = up(256, 128)
        self.up4 = up(128, 64)

        self.up_conv1 = nn.ConvTranspose2d(1024, 512, 2)
        self.up_conv2 = nn.ConvTranspose2d(512, 256, 2)
        self.up_conv3 = nn.ConvTranspose2d(256, 128, 2)
        self.up_conv4 = nn.ConvTranspose2d(128, 64)
        
        self.conv = nn.Conv2d(64, num_classes, 1)

    def forward(self, x):
        out1 = self.down1(x)
        
        out2 = self.pool1(out1)
        out2 = self.down2(out2)

        out3 = self.pool1(out2)
        out3 = self.down3(out3)

        out4 = self.pool1(out3)
        out4 = self.down4(out4)

        out5 = self.pool1(out4)
        out5 = self.down5(out5)

        out6 = self.up_conv1(out5)
        out6 = Residual(out4, out6) # 512 + 512 -> 1024 channels
        out6 = self.up1(out6)

        out7 = self.up_conv2(out6)
        out7 = Residual(out3, out7) # 256 + 256 -> 512 channels
        out7 = self.up2(out7) 

        out8 = self.up_conv2(out7)
        out8 = Residual(out2, out8) # 128 + 128 -> 256 channels
        out8 = self.up2(out8) 

        out9 = self.up_conv2(out8)
        out9 = Residual(out1, out9) # 64 + 64 -> 128 channels
        out9 = self.up2(out9) 

        out = self.conv(out9)

        return out 





