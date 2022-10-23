import torch
import numpy as np
import torchsummary

class VGGNet(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.conv_1_1 = torch.nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding='same')
        self.relu_1_1 = torch.nn.ReLU()
        self.conv_1_2 = torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding='same')
        self.relu_1_2 = torch.nn.ReLU()
        self.pool_1 = torch.nn.MaxPool2d(2)

        self.conv_2_1 = torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding='same')
        self.relu_2_1 = torch.nn.ReLU()
        self.conv_2_2 = torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding='same')
        self.relu_2_2 = torch.nn.ReLU()
        self.pool_2 = torch.nn.MaxPool2d(2)

        self.conv_3_1 = torch.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding='same')
        self.relu_3_1 = torch.nn.ReLU()
        self.conv_3_2 = torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding='same')
        self.relu_3_2 = torch.nn.ReLU()
        self.conv_3_3 = torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding='same')
        self.relu_3_3 = torch.nn.ReLU()
        self.pool_3 = torch.nn.MaxPool2d(2)

        self.conv_4_1 = torch.nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding='same')
        self.relu_4_1 = torch.nn.ReLU()
        self.conv_4_2 = torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding='same')
        self.relu_4_2 = torch.nn.ReLU()
        self.conv_4_3 = torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding='same')
        self.relu_4_3 = torch.nn.ReLU()
        self.pool_4 = torch.nn.MaxPool2d(2)

        self.conv_5_1 = torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding='same')
        self.relu_5_1 = torch.nn.ReLU()
        self.conv_5_2 = torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding='same')
        self.relu_5_2 = torch.nn.ReLU()
        self.conv_5_3 = torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding='same')
        self.relu_5_3 = torch.nn.ReLU()
        self.pool_5 = torch.nn.MaxPool2d(2)

        self.flatten = torch.nn.Flatten()
        self.fc_1 = torch.nn.Linear(7*7*512, 4096)
        self.fc_2 = torch.nn.Linear(4096, 4096)
        self.fc_3 = torch.nn.Linear(4096, 1000)
        self.fc_4 = torch.nn.Linear(1000, 10)
        
        self.sequence = torch.nn.Sequential(
            self.conv_1_1,
            self.relu_1_1,
            self.conv_1_2,
            self.relu_1_2,
            self.pool_1,
            self.conv_2_1,
            self.relu_2_1,
            self.conv_2_2,
            self.relu_2_2,
            self.pool_2,
            self.conv_3_1,
            self.relu_3_1,
            self.conv_3_2,
            self.relu_3_2,
            self.conv_3_3,
            self.relu_3_3,
            self.pool_3,
            self.conv_4_1,
            self.relu_4_1,
            self.conv_4_2,
            self.relu_4_2,
            self.conv_4_3,
            self.relu_4_3,
            self.pool_4,
            self.conv_5_1,
            self.relu_5_1,
            self.conv_5_2,
            self.relu_5_2,
            self.conv_5_3,
            self.relu_5_3,
            self.pool_5,
            self.flatten,
            self.fc_1,
            self.fc_2,
            self.fc_3,
            self.fc_4
        )

    def forward(self, x):
        x = self.sequence(x)
        return x