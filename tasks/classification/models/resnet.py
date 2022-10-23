import torch

class ResNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn = torch.nn.BatchNorm2d(64)
        self.relu = torch.nn.ReLU(inplace=True)
        self.pool = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer_1_1 = BasicBlock(64, 64)
        self.layer_1_2 = BasicBlock(64, 64, down=True)
        self.layer_2_1 = BasicBlock(128, 128)
        self.layer_2_2 = BasicBlock(128, 128, down=True)
        self.layer_3_1 = BasicBlock(256, 256)
        self.layer_3_2 = BasicBlock(256, 256, down=True)
        self.layer_4_1 = BasicBlock(512, 512)
        self.layer_4_2 = BasicBlock(512, 512)

        self.gap = torch.nn.AdaptiveAvgPool2d(1)
        self.fc = torch.nn.Linear(512, 10)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.layer_1_1(x)
        x = self.layer_1_2(x)
        x = self.layer_2_1(x)
        x = self.layer_2_2(x)
        x = self.layer_3_1(x)
        x = self.layer_3_2(x)
        x = self.layer_4_1(x)
        x = self.layer_4_2(x)

        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


class BasicBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, factor=1, down=False):
        super().__init__()
        self.conv_1 = conv_3x3(in_channels*factor, out_channels, 1)
        self.bn_1 = torch.nn.BatchNorm2d(out_channels)
        self.relu_1 = torch.nn.ReLU()
        self.conv_2 = conv_3x3(out_channels, out_channels, 1)
        self.bn_2 = torch.nn.BatchNorm2d(out_channels)
        self.relu_2 = torch.nn.ReLU()

        self.down = down
        if self.down:
            self.conv_3 = conv_3x3(out_channels, out_channels*2, 1)
            self.bn_3 = torch.nn.BatchNorm2d(out_channels*2)
            self.relu_3 = torch.nn.ReLU()

    def forward(self, x):
        out = self.conv_1(x)
        out = self.bn_1(out)
        out = self.relu_1(out)
        
        out = self.conv_2(out)
        out = self.bn_2(out)

        out += x
        out = self.relu_2(out)

        if self.down:
            out = self.conv_3(out)
            out = self.bn_3(out)
            out = self.relu_3(out)

        return out


class BottleNetBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, downsample=None):
        self.conv_1 = conv_1x1(in_channels, out_channels, 1)
        self.bn_1 = torch.nn.BatchNorm2d(out_channels)
        self.relu_1 = torch.nn.ReLU()
        
        self.conv_2 = conv_3x3(out_channels, out_channels, 1)
        self.bn_2 = torch.nn.BatchNorm2d(out_channels)
        self.relu_2 = torch.nn.ReLU()

        self.conv_3 = conv_1x1(out_channels, out_channels, 1)
        self.bn_3 = torch.nn.BatchNorm2d(out_channels)
        self.relu_3 = torch.nn.ReLU()

    def forward(self, x):
        out = self.conv_1(x)
        out = self.bn_1(out)
        out = self.relu_1(out)
        
        out = self.conv_2(out)
        out = self.bn_2(out)
        out = self.relu_2(out)

        out = self.conv_3(out)
        out = self.bn_3(out)

        out += x
        out = self.relu_3(out)

        return out


def conv_3x3(in_channels, out_channels, dilation):
    return torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False, dilation=1)

def conv_1x1(in_channels, out_channels):
    return torch.nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)

