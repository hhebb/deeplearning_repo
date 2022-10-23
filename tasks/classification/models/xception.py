import torch
import numpy as np
import torchsummary
import torchvision


class Xception(torch.nn.Module):
    def __init__(self, growth_rate=12, num_layers=100, theta=0.5):
        super().__init__()

        # entry flow
        self.conv_1 = ConvBlock(in_channels=3, out_channels=32, kernel=3, stride=2)
        self.conv_2 = ConvBlock(in_channels=32, out_channels=64, kernel=3, stride=1)

        self.sep_conv_1 = SeparableConv(in_channels=64, out_channels=128)
        self.sep_conv_2 = SeparableConv(in_channels=128, out_channels=128)
        self.pool_1 = torch.nn.MaxPool2d(kernel_size=3, stride=2)

        self.sep_conv_3 = SeparableConv(in_channels=128, out_channels=256)
        self.sep_conv_4 = SeparableConv(in_channels=256, out_channels=256)
        self.pool_2 = torch.nn.MaxPool2d(kernel_size=3, stride=2)

        self.sep_conv_5 = SeparableConv(in_channels=256, out_channels=728)
        self.sep_conv_6 = SeparableConv(in_channels=728, out_channels=728)
        self.pool_3 = torch.nn.MaxPool2d(kernel_size=3, stride=2)

        # middle flow
        self.middle = torch.nn.Sequential()
        
        for i in range(8):
            seq = torch.nn.Sequential()
            for j in range(3):
                seq.add_module(f'sep_conv_m_{i}_{j}', SeparableConv(in_channels=728, out_channels=728))
            
            # self.middle.append(seq)
            self.middle.add_module(f'sep_conv_m_{i}', seq)

        # exit flow
        self.sep_conv_7 = SeparableConv(in_channels=728, out_channels=728)
        self.sep_conv_8 = SeparableConv(in_channels=728, out_channels=1024)
        self.pool_4 = torch.nn.MaxPool2d(kernel_size=3, stride=2)

        self.sep_conv_9 = SeparableConv(in_channels=1024, out_channels=1536)
        self.sep_conv_10 = SeparableConv(in_channels=1536, out_channels=2048)
        self.gap = torch.nn.AdaptiveAvgPool2d(1)
        self.flatten = torch.nn.Flatten(start_dim=1)
        self.fc = torch.nn.Linear(2048, 10)
        

    def forward(self, x):
        # entry
        x = self.conv_1(x)
        entry = self.conv_2(x)

        out_1 = self.sep_conv_1(entry)
        out_1 = self.sep_conv_2(out_1)
        out_1 = self.pool_1(out_1)

        out_2 = self.sep_conv_3(out_1)
        out_2 = self.sep_conv_4(out_2)
        out_2 = self.pool_2(out_2)

        out_3 = self.sep_conv_5(out_2)
        out_3 = self.sep_conv_6(out_3)
        out_3 = self.pool_3(out_3)
        # print(out_3.shape)
        
        # middle
        for i in range(8):
            middle = self.middle[i](out_3)

        # exit
        # print(middle.shape)
        exit = self.sep_conv_7(middle)
        exit = self.sep_conv_8(exit)
        exit = self.pool_4(exit)

        exit = self.sep_conv_9(exit)
        exit = self.sep_conv_10(exit)
        exit = self.gap(exit)
        exit = self.flatten(exit)
        exit = self.fc(exit)

        return exit


class SeparableConv(torch.nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # 1x1 conv
        self.add_module('pointwise', ConvBlock(in_channels=in_channels, out_channels=out_channels, kernel=1, stride=1, padding=0, group=1))
        # depthwise
        self.add_module('depthwise', ConvBlock(in_channels=out_channels, out_channels=out_channels, kernel=3, stride=1, padding=1, group=out_channels))


class ConvBlock(torch.nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel, stride, padding=0, group=1):
        super().__init__()
        self.add_module('bn', torch.nn.BatchNorm2d(in_channels))
        self.add_module('relu', torch.nn.ReLU(True))
        self.add_module('conv', torch.nn.Conv2d(in_channels, out_channels, kernel_size=kernel, stride=stride, padding=padding, groups=group, bias=False))
