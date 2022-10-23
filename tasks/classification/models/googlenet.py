import torch
import numpy as np
import torchsummary

class GoogleNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_1 = torch.nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.pool_1 = torch.nn.MaxPool2d(3, 2)

        self.conv_2 = torch.nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0)
        self.conv_2_2 = torch.nn.Conv2d(64, 192, kernel_size=3, stride=1, padding=1)
        self.pool_2 = torch.nn.MaxPool2d(3, 2, ceil_mode=True)

        self.inception_3_1 = Inception(192, 64, 96, 128, 16, 32, 32)
        self.inception_3_2 = Inception(256, 128, 128, 192, 32, 96, 64)
        self.pool_3 = torch.nn.MaxPool2d(3, 2, ceil_mode=True)

        self.inception_4_1 = Inception(480, 192, 96, 208, 16, 48, 64)
        self.inception_4_2 = Inception(512, 160, 112, 224, 24, 64, 64)
        self.inception_4_3 = Inception(512, 128, 128, 256, 24, 64, 64)
        self.inception_4_4 = Inception(512, 112, 144, 288, 32, 64, 64)
        self.inception_4_5 = Inception(528, 256, 160, 320, 32, 128, 128)
        self.pool_4 = torch.nn.MaxPool2d(2, 2, ceil_mode=True)

        self.inception_5_1 = Inception(832, 256, 160, 320, 32, 128, 128)
        self.inception_5_2 = Inception(832, 384, 192, 384, 48, 128, 128)

        self.aux_1 = InceptionAux(512, 10, .7)
        self.aux_2 = InceptionAux(528, 10, .7)

        self.gap = torch.nn.AdaptiveAvgPool2d(1)
        self.dropout = torch.nn.Dropout(.5, inplace=True)
        self.fc = torch.nn.Linear(1024, 10)

        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
                torch.nn.init.trunc_normal_(module.weight, mean=0.0, std=0.01, a=-1, b=2)
            elif isinstance(module, torch.nn.BatchNorm2d):
                torch.nn.init.constant_(module.weight, 1)
                torch.nn.init.constant_(module.bias, 0)


    def forward(self, x):
        out = self.conv_1(x)
        out = self.pool_1(out)

        out = self.conv_2(out)
        out = self.conv_2_2(out)
        out = self.pool_2(out)

        out = self.inception_3_1(out)
        out = self.inception_3_2(out)
        out = self.pool_3(out)
        out = self.inception_4_1(out)
        aux_1 = self.aux_1(out) if self.training else None

        out = self.inception_4_2(out)
        out = self.inception_4_3(out)
        out = self.inception_4_4(out)
        aux_2 = self.aux_2(out) if self.training else None

        out = self.inception_4_5(out)
        out = self.pool_4(out)
        out = self.inception_5_1(out)
        out = self.inception_5_2(out)

        out = self.gap(out)
        out = torch.flatten(out, 1)
        out = self.dropout(out)
        aux_3 = self.fc(out)

        # return (out, aux_1, aux_2, aux_3)
        return out


class Inception(torch.nn.Module):
    def __init__(self, in_dim, filter_1, filter_2, filter_2_2, filter_3, filter_3_2, filter_4):
        super().__init__()
        self.branch_1 = BasicConv(in_dim, filter_1, kernel_size=1, padding='same')

        self.branch_2 = BasicConv(in_channels=in_dim, out_channels=filter_2, kernel_size=1, padding='same')
        self.branch_2_2 = BasicConv(in_channels=filter_2, out_channels=filter_2_2, kernel_size=3, padding='same')

        self.branch_3 = BasicConv(in_channels=in_dim, out_channels=filter_3, kernel_size=1, padding='same')
        self.branch_3_2 = BasicConv(in_channels=filter_3, out_channels=filter_3_2, kernel_size=5, padding='same')

        self.branch_4 = torch.nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.branch_4_2 = BasicConv(in_channels=in_dim, out_channels=filter_4, kernel_size=1, padding='same')

    def forward(self, x):
        branch_1 = self.branch_1(x)
        
        branch_2 = self.branch_2(x)
        branch_2 = self.branch_2_2(branch_2)
        
        branch_3 = self.branch_3(x)
        branch_3 = self.branch_3_2(branch_3)
        
        branch_4 = self.branch_4(x)
        branch_4 = self.branch_4_2(branch_4)
        concat = torch.concat((branch_1, branch_2, branch_3, branch_4), axis=1)

        return concat


class InceptionAux(torch.nn.Module):
    def __init__(self, in_channels, num_classes, dropout=.7):
        super().__init__()
        self.pool = torch.nn.AdaptiveAvgPool2d(4)
        self.conv = torch.nn.Conv2d(in_channels, 128, kernel_size=1, stride=1, padding=0)
        self.relu = torch.nn.ReLU(True)
        self.fc_1 = torch.nn.Linear(2048, 1024)
        self.fc_2 = torch.nn.Linear(1024, num_classes)
        self.dropout = torch.nn.Dropout(dropout, True)

    def forward(self, x):
        out = self.pool(x)
        out = self.conv(out)
        out = torch.flatten(out, 1)
        out = self.fc_1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc_2(out)

        return out


class BasicConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels, **kargs):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, bias=False, **kargs)
        self.bn = torch.nn.BatchNorm2d(out_channels, eps=.001)
        self.relu = torch.nn.ReLU(True)

        self.sequence = torch.nn.Sequential(
            self.conv,
            self.bn,
            self.relu
        )

    def forward(self, x):
        x = self.sequence(x)
        return x