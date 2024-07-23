import torch

from torch import nn
from torch.nn import functional as F

class DoubleConv(nn.Module):
    
    def __init__(self, in_channels, out_channels, middle_channels=None):
        super().__init__()

        if not middle_channels:
            middle_channels = out_channels
        
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, middle_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(middle_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(middle_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)
    
class DownConv(nn.Module):
    
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.down_conv = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
            DoubleConv(in_channels, out_channels)
        )
        
    def forward(self, x):
        return self.down_conv(x)
    
class UpConv(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.double_conv= DoubleConv(in_channels, out_channels, in_channels//2)

    def pad_image(self, x1, x2):
        x_delta = abs(x1.size(dim=-1) - x2.size(dim=-1))
        x_padding = (x_delta // 2, x_delta - x_delta // 2)

        y_delta = abs(x1.size(dim=-2) - x2.size(dim=-2))
        y_padding = (y_delta // 2, y_delta - y_delta // 2)

        x1 = F.pad(x1, x_padding + y_padding)
        return x1

    def forward(self, x1: torch.Tensor, x2: torch.Tensor):
        x1 = self.upsample(x1)
        x1 = self.pad_image(x1, x2)

        x = torch.cat([x2, x1], dim=1)        
        return self.double_conv(x)
    
class OutConv(nn.Module):
    
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.out_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        
    def forward(self, x):
        return self.out_conv(x)
    
class UNet(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.double_conv = DoubleConv(in_channels, 64)
        self.down_conv1 = DownConv(64, 128)
        self.down_conv2 = DownConv(128, 256)
        self.down_conv3 = DownConv(256, 512)
        self.down_conv4 = DownConv(512, 1024 // 2)
        
        self.up_conv1 = UpConv(1024, 512 // 2)
        self.up_conv2 = UpConv(512, 256 // 2)
        self.up_conv3 = UpConv(256, 128 // 2)
        self.up_conv4 = UpConv(128, 64)

        self.out_conv = OutConv(64, out_channels)

    def forward(self, x):
        x0 = self.double_conv(x)
        
        x1 = self.down_conv1(x0)
        x2 = self.down_conv2(x1)
        x3 = self.down_conv3(x2)
        x4 = self.down_conv4(x3)

        x = self.up_conv1(x4, x3)
        x = self.up_conv2(x, x2)
        x = self.up_conv3(x, x1)
        x = self.up_conv4(x, x0)

        return self.out_conv(x)