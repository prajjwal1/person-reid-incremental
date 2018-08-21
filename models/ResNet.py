from __future__ import absolute_import

import torch
from torch import nn
from torch.nn import functional as F 
import torchvision

class ResNet50(nn.Module):
    def __init__(self, **kwargs):
        super(ResNet50,self).__init__()
        resnet50 = torchvision.models.resnet50(pretrained=True)
        self.base= nn.Sequential(*list(resnet50.children())[:-2])
        setattr(self,"fc0", nn.Linear(2048,751)) #Market 1501
        setattr(self,"fc1", nn.Linear(2048,1360)) #CUHK-03          

    def forward(self,x):
        x = self.base(x)
        x = F.avg_pool2d(x,x.size()[2:])
        f = x.view(x.size(0),-1)
        clf_outputs = {}
        num_fcs = 2
        
        for i in range(num_fcs):
            clf_outputs["fc%d" %i] = getattr(self, "fc%d" %i)(f)

        return clf_outputs,f 
