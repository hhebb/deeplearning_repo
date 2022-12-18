# prerequisites
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from torchvision.utils import save_image
from torchvision.datasets import MNIST
import albumentations as A
from albumentations.pytorch import transforms


# standardization code
batch_size = 200
d_latent = 256


def sample_z(batch_size=1, d_latent=16):
    return torch.randn(batch_size, d_latent, device='cuda:0')

G = nn.Sequential(
    nn.Linear(d_latent, 64),
    nn.ReLU(),
    nn.Linear(64, 512),
    nn.ReLU(),
    nn.Linear(512, 28*28),
    nn.Sigmoid()
).to('cuda:0')

D = nn.Sequential(
    nn.Linear(28*28, 512),
    nn.LeakyReLU(),
    nn.Dropout(0.2),
    nn.Linear(512, 64),
    nn.LeakyReLU(),
    nn.Dropout(0.2),
    nn.Linear(64, 1),
    nn.Sigmoid()
).to('cuda:0')
