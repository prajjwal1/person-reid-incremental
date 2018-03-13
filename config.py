
# coding: utf-8

# In[1]:


import os
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.models as models


# In[1]:


from binary_cross_entropy import binary_cross_entropy_with_logits
from DenseNet import DenseNetBase, DenseNet121,DenseNet100
from dataset import MultiLabelDataset


# In[2]:


def loss_function(multilabel):
    if multilabel:
        return binary_cross_entropy_with_logits
    return F.cross_entropy


# In[4]:
def get_dataset(name, partition, transform):
    image_folder_datasets = ['imagenet', 'food-101']
    image_dir = os.path.join('input', '%s-%s' % (name, partition))
    if name in image_folder_datasets:
        dataset = datasets.ImageFolder(image_dir, transform)
    elif name == 'cifar10':
        if partition == 'train':
            train = True
        elif partition == 'test':
            train = False
        dataset = datasets.CIFAR10(root='cifar', train=train, download=True, transform=transform)
    elif name == 'food-collage':
        csv_path = os.path.join('input', '%s-%s.csv' % (name, partition))
        dataset = MultiLabelDataset(csv_path, image_dir, transform)
    return dataset


# In[5]:


def get_network(name, num_classes, pretrained):
    if name=='densenet-base':
        assert not pretrained
        return DenseNetBase(g_rate=12,depth=100,reduction=0.5,bottleneck=True,num_classes=num_classes)
    elif name == 'densenet-121':
        return DenseNet121(num_classes,pretrained)
    elif name == 'densenet-100':
        assert not pretrained
        return DenseNet100(num_classes)

