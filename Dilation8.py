import torch
import torch.nn as nn
from collections import OrderedDict
from torchsummary import summary


class Dilation8(nn.Module):
    def __init__(self, in_channels = 3, out_channels = 1):
        super().__init__()
        
        self.frontEnd1 = Dilation8._FrontEnd(in_channels = 3, out_channels = 64, count = 2, layer_num = 1)
        self.frontEnd2 = Dilation8._FrontEnd(in_channels = 64, out_channels = 128, count = 2, layer_num = 2)
        self.frontEnd3 = Dilation8._FrontEnd(in_channels = 128, out_channels = 256, count = 3, layer_num = 3)
        self.frontEnd4 = Dilation8._FrontEnd(in_channels = 256, out_channels = 512, count = 3, layer_num = 4)
        self.frontEnd5 = Dilation8._FrontEnd(in_channels = 512, out_channels = 512, count = 3, layer_num = 5, dilation = 2)

    def forward(self, x):
        x = self.frontEnd1(x)
        x = self.frontEnd2(x)
        x = self.frontEnd3(x)
        x = self.frontEnd4(x)
        x = self.frontEnd5(x)
        return x

    @staticmethod
    def _FrontEnd(in_channels, out_channels, count = 2, layer_num = 1, kernel_size = 3, bnorm = False, dilation = 1):
        layers = OrderedDict()
        layer_num = str(layer_num)
        for layer in range(1,count+1):
            if layer > 1:
                in_channels = out_channels
            layers['conv'+layer_num+'_'+str(layer)] = nn.Conv2d(
                                                                in_channels = in_channels,
                                                                out_channels = out_channels,
                                                                kernel_size = kernel_size,
                                                                dilation = dilation
                                                                )
            layers['relu'+layer_num+'_'+str(layer)] = nn.ReLU(inplace=True)
        if dilation < 2:
            layers['pool'+layer_num+'_'] = nn.MaxPool2d(2,2)
        return nn.Sequential(layers)

