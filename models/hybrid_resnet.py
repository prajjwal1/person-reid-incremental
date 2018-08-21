from __future__ import absolute_import

import torch
from torch import nn
from torch.nn import functional as F
import torchvision

def conv_layer(ni,nf,kernel_size=3,stride=1):
    return nn.Sequential(
            nn.Conv2d(ni,nf,kernel_size=kernel_size,bias=False,stride=stride,padding=kernel_size//2),
            nn.BatchNorm2d(nf,momentum=0.01),
            nn.LeakyReLU(negative_slope=0.1,inplace=True)
        )

class augmented1(nn.Module):
    def __init__(self,ni):
        super(augmented1, self).__init__()
        self.conv1 = conv_layer(ni, ni//2, kernel_size=1)
        self.conv2 = conv_layer(ni//2, ni,kernel_size=3)
        self.classifier = nn.Linear(ni*8*4,751)

    def forward(self,x):
        residual = x
        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))
        
        out += residual
        
        out = out.view(out.size(0),-1)
        return self.classifier(out)
        

class augmented2(nn.Module):
    def __init__(self,ni):
        super(augmented2, self).__init__()
        self.conv1 = conv_layer(ni, ni//2, kernel_size=1)
        self.conv2 = conv_layer(ni//2, ni, kernel_size= 3)
        self.classifier = nn.Linear(ni*8*4,1360)

    def forward(self,x):
        residual = x
        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))
        
        out += residual
        
        out = out.view(out.size(0),-1)
        return self.classifier(out)
    

class hybrid_cnn(nn.Module):
    def __init__(self,**kwargs):
        super(hybrid_cnn,self).__init__()
        resnet = torchvision.models.resnet50(pretrained=True)
        self.base = nn.Sequential(*list(resnet.children())[:-2])
        
        setattr(self,"fc0",augmented1(2048))
        setattr(self,"fc1",augmented2(2048))


    def forward(self,x):
        x = self.base(x)
        clf_outputs = {}
        num_fcs = 2
        
        for i in range(num_fcs):
            clf_outputs["fc%d" %i] = getattr(self, "fc%d" %i)(x)

        return clf_outputs,x
    
