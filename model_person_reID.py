import torch
import torch.nn as nn
from torch.nn import init
from torchvision import models
from torch.autograd import Variable

def weight_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.kaiming_normal(m.weight.data,a=0,mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal(m.weight.data,a=0,mode='fan_out')
        init.constant(m.bias.data,0.0)
    elif classname.find('BatchNorm1d') != -1:
        init.normal(m.weight.data,1.0,0.02)
        init.constant(m.bias.data,0.0)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal(m.weight.data,std=0.001)
        init.constant(m.bias.data,0.0)

class net1(nn.Module):
    
    def __init__(self,num_class):
        super().__init__()
        model_ft = models.resnet50(pretrained=True)

        model_ft.avgpool = nn.AdaptiveAvgPool2d((1,1))
        num_features = model_ft.fc.in_features
        block = []
        num_bottleneck = 512
        block += [nn.Linear(num_features,num_bottleneck)]
        block += [nn.BatchNorm1d(num_bottleneck)]
        block += [nn.LeakyReLU(0.1)]  #Slope
        block += [nn.Dropout(p=0.5)] 

        block += nn.Sequential(*block)
        block.apply(weights_init)
        model_ft.fc = block
        self.model = model_ft

        classifier = []
        classifier += [nn.Linear(num_bottleneck,class_num)]
        classifier = nn.Sequential(*classifier)
        classifier.apply(weights_init_classifier)
        self.classifier = classifier

    def forward(self,x):
        x = self.model(x)
        x = self.classifier(x)
        return x
