
# coding: utf-8

# In[1]:


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


# In[2]:


class BottleNeck(nn.Module):
    def __init__(self,num_channels,g_rate):
        super(BottleNeck,self).__init__()
        interChannels = 4*g_rate
        self.batch_norm1 = nn.BatchNorm2d(num_channels)
        self.conv1 = nn.Conv2d(num_channels, interChannels,kernel_size=1,bias=False)
        
        self.batch_norm2 = nn.BatchNorm2d(interChannels)
        self.conv2 = nn.Conv2d(interChannels,g_rate,kernel_size=3,padding=1,bias=False)
        
    def forward(self,x):
        outp = self.conv1(F.relu(self.batch_norm1(x)))
        outp = self.conv2(F.relu(self.batch_norm2(outp)))
        outp = torch.cat((x,outp),1)
        return outp


# In[4]:


class SingleLayer(nn.Module):
    def __init__(self, num_channels,g_rate):
        super(SingleLayer,self).__init__()
        self.batch_norm1 = nn.BatchNorm2d(num_channels)
        self.conv1 = nn.Conv2d(num_channels,g_rate,kernel_size=3, padding=1, bias=False)
        
    def forward(self,x):
        outp = self.conv1(F.relu(self.batch_norm1(x)))
        outp = torch.cat((x,outp),1)
        return outp


# In[5]:


class Transition(nn.Module):
    def __init__(self,num_channels,num_out_channels):
        super(Transition,self).__init__()
        self.batch_norm1 = nn.BatchNorm2d(num_channels)
        self.conv1 = nn.Conv2d(num_channels,num_out_channels,kernel_size=1,bias=False)
    
    def forward(self,x):
        outp = self.conv1(F.relu(self.batch_norm1(x)))
        outp = F.avg_pool2d(out,2)
        return out


# In[7]:


class DenseNetBase(nn.Module):
    def __init__(self,g_rate,depth,reduction,num_classes,bottleneck):
        super(DenseNetBase,self).__init__()
        
        num_dense_blocks = (depth-4) // 3
        if bottleneck:
            num_dense_blocks //=2
        
        num_channels = 2*g_rate
        self.conv1 = nn.Conv2d(3,num_channels,kernel_size=3,padding=1)
        self.dense1 = self._make_dense(num_channels,g_rate,num_dense_blocks,bottleneck)
        num_channels += num_dense_blocks*g_rate
        num_out_channels = int(math.floor(num_channels*reduction))
        self.trans1 = Transition(num_channels,num_out_channels)
    
        num_channels = num_out_channels
        self.dense2 = self._make_dense(num_channels,g_rate,num_dense_blocks,bottleneck)
        num_channels += num_dense_blocks*g_rate
        num_out_channels = int(math.floor(num_channels*reduction))
        self.trans2 = Transition(num_channels,num_out_channels)
        
        num_channels = num_out_channels
        self.dense3 = self._make_dense(num_channels,g_rate,num_dense_blocks,bottleneck)
        num_channels += num_dense_blocks*g_rate
        
        self.batch_norm1 = nn.BatchNorm2d(num_channels)
        self.fc = nn.Linear(num_channels,num_classes)
        
        for m in self.modules():
            if isinstance(m,nn.Conv2d):
                n = m.kernel_size[0]*m_kernel_size[1]*m.out_channels
                m.weight.data.normal_(0,math.sqrt(2./n))
            elif isinstance(m,nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m,nn.Linear):
                m.bias.data.zero_()
    
    def _make_dense(self, num_channels,g_rate,num_dense_blocks,bottleneck):
        layers = []
        for i in range(int(num_dense_blocks)):
            if bottleneck:
                layers.append(BottleNeck(num_channels,g_rate))
            else:
                layers.append(SingleLayer(num_channels,g_rate))
            num_channels += g_rate
        return nn.Sequential(*layers)
    
    def forward(self,x):
        outp = self.conv1(x)
        outp = self.trans1(self.dense1(outp))
        outp = self.trans2(self.dense2(outp))
        outp = self.dense3(out)
        outp = torch.squeeze(F.avg_pool2d(F.relu(self.bn1(out)),8))
        return self.fc(outp)


# In[8]:


class DenseNet121(nn.Module):
    def __init__(self,num_classes, pretrained):
        super(DenseNet121,self).__init__()
        net = models.densenet.densenet121(pretrained=pretrained)
        self.features = nn.features
        self.classifier = nn.Linear(1024, num_classes)
    
    def forward(self, X):
        features = self.features(x)
        outp = F.relu(features, inplace=True)
        otp = F.avg_pool2d(out, kernel_size=7).view(features.size(0),-1)
        return self.classifier(outp)


# In[9]:


class DenseNet100(models.densenet.DenseNet):
    def __init__(self, num_classes):
        super(DenseNet100, self).__init__(g_rate=12, block_config=(16,16,16), num_init_features=24,bn_size=3,drop_rate=0.5, num_classes=num_classes)
        
    def forward(self,x):
        outp = torch.squeeze(F.avg_pool2d(F.relu(self.features(x)), 8))
        return self.classifier(outp)

