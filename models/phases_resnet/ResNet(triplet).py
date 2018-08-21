from __future__ import absolute_import

import torch
from torch import nn
from torch.nn import functional as F 
import torchvision

__all__ = ['ResNet50']

class ResNet50(nn.Module):
    def __init__(self,num_fcs,num_classes,loss={'htri'},**kwargs):
        super(ResNet50,self).__init__()
        self.loss = loss
        resnet50 = torchvision.models.resnet50(pretrained=True)
        self.base= nn.Sequential(*list(resnet50.children())[:-2])
        self.num_fcs=num_fcs
        for i in range(num_fcs):
           setattr(self, "fc%d" %i, nn.Linear(2048,num_classes))

    def forward(self,x):
        x = self.base(x)
        x = F.avg_pool2d(x,x.size()[2:])
        f = x.view(x.size(0),-1)
        clf_outputs = {}
        num_fcs = 2
        
        for i in range(num_fcs):
            clf_outputs["fc%d" %i] = getattr(self, "fc%d" %i)(f)

        return clf_outputs