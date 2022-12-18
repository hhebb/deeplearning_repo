import torch
from torch import nn
from torchvision.models.segmentation import fcn_resnet50, FCN_ResNet50_Weights
from torch.autograd import Function


class FCN(nn.Module):
    def __init__(self, classes=1):
        super().__init__()

        weights = FCN_ResNet50_Weights.DEFAULT
        self.model = fcn_resnet50(num_classes=21) # weights=weights, 
        self.conv = nn.Conv2d(21, classes, 1)

    def forward(self, x):
        x = self.model(x)['out']
        x = self.conv(x)
        x = torch.sigmoid(x)

        return x


class FCN_DANN(nn.Module):
    def __init__(self):
        super().__init__()

        self.alpha = .5

        # seg
        weights = FCN_ResNet50_Weights.DEFAULT
        self.model = fcn_resnet50(num_classes=21) # 
        self.backbone = self.model.backbone
        self.fcn = self.model.classifier
        self.conv = nn.Conv2d(21, 3, 1)

        # domain
        self.domain_cls = nn.Sequential(
            nn.Conv2d(2048, 512, 3, 1, 1),
            nn.Conv2d(512, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            # nn.ReLU(),
            nn.AdaptiveMaxPool2d(1),
            nn.Flatten(start_dim=1),
            nn.Linear(128, 32),
            nn.Linear(32, 1),
        )

    def set_alpha(self, alpha):
        self.alpha = alpha

    def forward(self, x):
        # feature
        feature = self.backbone(x)['out']
        
        # seg mask
        seg = self.fcn(feature)
        seg = self.conv(seg)
        seg = nn.functional.interpolate(seg, scale_factor=8, mode='bilinear')
        seg = torch.sigmoid(seg)

        # domain classifier
        feature_inv = ReverseGrad.apply(feature, self.alpha)
        domain = self.domain_cls(feature_inv)
        # domain = torch.sigmoid(domain)

        return seg, domain


class ReverseGrad(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output * ctx.alpha
        
        return output, None
