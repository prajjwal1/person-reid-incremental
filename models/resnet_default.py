from __future__ import absolute_import

import torch
from torch import nn
from torch.nn import functional as F
import torchvision


class ResNet50(nn.Module):
    def __init__(self, **kwargs):
        super(ResNet50, self).__init__()
        resnet50 = torchvision.models.resnet50(pretrained=True)
        self.base = nn.Sequential(*list(resnet50.children())[:-2])
        self.classifier = nn.Linear(2048, 751)
        self.feat_dim = 2048 # feature dimension

    def forward(self, x):
        x = self.base(x)
        x = F.avg_pool2d(x, x.size()[2:])
        f = x.view(x.size(0), -1)
        y = self.classifier(f)
        return y

