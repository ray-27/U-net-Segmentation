
import torch
from torch.nn.modules.activation import ReLU
from torch.nn.modules.batchnorm import BatchNorm2d
from torch.nn.modules.conv import Conv2d
import torchvision.transforms.functional as TF
import torch.nn as nn

class DoubleConv(nn.Module):
    def __init__(self, in_channels,out_channels):
        super(DoubleConv,self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels,out_channels,3,1,1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels,out_channels,3,1,1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self,x):
        return self.conv(x)


class UNET(nn.Module):
    def __init__(self, in_channels=3,out_channels=1, features=[64,128,256,512]):
        super(UNET,self).__init__()

        self.ups = nn.ModuleList()
        self.down = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2,stride=2)

        #down part of the UNET
        for feature in features:
            self.down.append(DoubleConv(in_channels,feature))
            in_channels = feature

        #up part of the UNET
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(feature*2,feature,kernel_size=2,stride=2)
            )
            self.ups.append(DoubleConv(feature*2,feature))

        self.bottleneck = DoubleConv(features[-1],features[-1]*2)
        self.final_conv = nn.Conv2d(features[0],out_channels,kernel_size=1)

    def forward(self,x):
