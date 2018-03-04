import torch

import torch.nn as nn
import torch.optim as optim

import torch.nn.functional as F
from torch.autograd import Variable

import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import torchvision.models as models

import sys
import math

class Bottleneck(nn.Module):
    def __init__(self,nChannels, growRate):
        super(Bottleneck, self).__init__()
        interChannels =4*growthRate
        self.bn1 = nn.BatchNorm2d(nChannels)
        self.conv1 = nn.Conv2d(nChannels, interChannels,kernel_size=1,
                bias=False)
        self.bn2 = nn.BatchNrm2d(interChannels)
        self.conv2 = self.Conv2d(interChannels, growthRate,kernel_siz=3,
                padding=1, bias=False)

    def forward(self,x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        out = torch.cat((x,out),1)
        return out

class SingleLayer(nn.Module):
    def __init__(self, nChannels, nOutChannels):
        super(Transition, self).__init__()
        self.bn1 = nn.BatchNorm2d(nChannels)
        self.conv1 = nn.Conv2d(nChannels, nOutChannels,kernel_size=1,
                bias = False)

    def forward(self,x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = F.avg_pool2d(out,2)
        return out

class DenseNet(nn.Module):
    def __init__(self, growthRate,depth,reduction,nClasses,b0ottleneck):
        super(DenseNet,self).__init__()

        nDenseBlocks = depth-4
        if bottleneck:
            nDenseBlocks //=2

        nChannels = 2*growthRate
        self.conv1 = nn.Conv2d(3, nChannels,)
