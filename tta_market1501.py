from __future__ import print_function,absolute_import

import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn 
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from torch.nn import functional as F

import os
import sys
import time
import datetime
import argparse
import os.path as osp

import dataset_manager
from dataset_loader import ImageDataset
import transforms as tfms 
from models import hybrid_convnet2
from clr import CyclicLR
from loss import CrossEntropy,TripletLoss,DeepSuperVision
from utils import AverageMeter,Logger,save_checkpoint
from metrics import evaluate
from samplers import RandomIdentitySampler


workers=4
height=256
width=128
split_id=0
max_epoch=60
train_batch = 64
test_batch = 64
lr = 0.0003
stepsize= 60
gamma = 0.1
margin=0.3
weight_decay=5e-4
print_freq = 10
num_instances=4
eval_step=20
start_eval=0
start_epoch=0
split_id=0

PATH = 'log'

dataset_name = 'market1501'
evaluation = False
num_classes = 751   #751 for market 1501 #1360 for CUHK-03
# CUHK03 specific
cuhk03_labeled = False
use_metric_cuhk03 = False
cuhk03_classic_split = False
########################################################
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

sys.stdout = Logger(osp.join(PATH,'log_train.txt'))

print("Dataset is being initialized")


dataset = dataset_manager.init_img_dataset(
    root='data',name=dataset_name,split_id=split_id,
    cuhk03_labeled=cuhk03_labeled,cuhk03_classic_split=cuhk03_classic_split,
)


tfms_train = tfms.Compose([
    tfms.Random2DTranslation(height,width),
    tfms.RandomHorizontalFlip(),
    tfms.ToTensor(),
    tfms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225]),
])

tfms_test = tfms.Compose([
    tfms.Resize(size=(height,width)),
    tfms.ToTensor(),
    tfms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225]),
])

pin_memory = True

trainloader = DataLoader(
    ImageDataset(dataset.train,transform=tfms_train),
    sampler = RandomIdentitySampler(dataset.train,num_instances=num_instances),
    batch_size = train_batch,num_workers=workers,
    pin_memory=True,drop_last=True,
)

queryloader = DataLoader(
    ImageDataset(dataset.query,transform=tfms_test),
    batch_size=test_batch,shuffle=False,num_workers=workers,
    pin_memory=pin_memory,drop_last=False,
)

galleryloader = DataLoader(
    ImageDataset(dataset.gallery,transform=tfms_test),
    batch_size=test_batch,shuffle=False,num_workers=workers,
    pin_memory=pin_memory,drop_last=False, 
)

activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

#####################################################################
def train(epoch,model,optim,trainloader):

    losses = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    
    model.train()

    end = time.time()

    cross_entropy = CrossEntropy(num_classes = num_classes)
    triplet_loss_fn = TripletLoss(margin=margin)

    model.fc0.train(True)
    model.fc1.train(False)  

    output_fc = "fc0"
     
    model.base.train(True)

    for batch,(imgs,pids,_) in enumerate(trainloader):
        imgs,pids = imgs.cuda(), pids.cuda()

        data_time.update(time.time()-end)

        clf_outputs = model(imgs)
        f = activation['fc0.conv2']
        f = F.avg_pool2d(f,f.size()[2:])
        f = f.view(f.size(0),-1)

        fc0 = clf_outputs[output_fc] 

        if isinstance(fc0,tuple):  
            cross_entropy_loss = DeepSuperVision(cross_entropy,fc0,pids)
        else:
            cross_entropy_loss = cross_entropy(fc0,pids)
        if isinstance(f,tuple):
            triplet = DeepSuperVision(triplet_loss_fn,f,pids)
        else:
            triplet = triplet_loss_fn(f,pids)
        
        loss = cross_entropy_loss + triplet
        
        optim.zero_grad()
        loss.backward()
        optim.step()

        batch_time.update(time.time()-end)
        end = time.time()

        losses.update(loss.item(),pids.size(0))

        if (batch+1) % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                   epoch+1, batch+1, len(trainloader), batch_time=batch_time,
                   data_time=data_time, loss=losses))
######################################################################################################
def test(model, queryloader,galleryloader,ranks=[1,5,10,20]):
    batch_time = AverageMeter()

    model.eval()

    with torch.no_grad():
        qf,q_pids,q_camids = [],[],[]
        for batch,(imgs,pids,camids) in enumerate(queryloader):
            imgs = imgs.cuda()

            end = time.time()
            
            fc0_preds = model(imgs)
            
            batch_time.update(time.time()-end)

            output_fc = "fc0"       
            fc0 = fc0_preds[output_fc]
            fc0 = fc0.data.cpu()  

            qf.append(fc0)
            q_pids.extend(pids)
            q_camids.extend(camids)
        qf = torch.cat(qf,0)
        q_pids = np.asarray(q_pids)
        q_camids = np.asarray(q_camids)

        print("Extracted features for query set, obtained {}-by-{} matrix".format(qf.size(0), qf.size(1)))

        gf,g_pids,g_camids = [],[],[]
        end = time.time()
        for batch,(imgs,pids,camids) in enumerate(galleryloader):
            imgs = imgs.cuda()
            
            end = time.time()
           
            fc0_preds = model(imgs)

            output_fc = "fc0"
            fc0 = fc0_preds[output_fc]
            
            batch_time.update(time.time()-end)

            fc0 = fc0.data.cpu()
            gf.append(fc0)
            g_pids.extend(pids)
            g_camids.extend(camids)
        gf = torch.cat(gf,0)
        g_pids = np.asarray(g_pids)
        g_camids = np.asarray(g_camids)

        print("Extracted features for gallery set, obtained {}-by-{} matrix".format(gf.size(0), gf.size(1)))
        ##############################################################################################################################
    print("==> BatchTime(s)/BatchSize(img): {:.3f}/{}".format(batch_time.avg, test_batch))
    print("==> BatchTime(s)/BatchSize(img): {:.3f}/{}".format(batch_time.avg, test_batch))

    m, n = qf.size(0), gf.size(0)
    distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
              torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    distmat.addmm_(1, -2, qf, gf.t())
    distmat = distmat.numpy()

    print("Computing CMC and mAP")
    cmc, mAP = evaluate(distmat, q_pids, g_pids, q_camids, g_camids, use_metric_cuhk03=use_metric_cuhk03)

    print("Results ----------")
    print("mAP: {:.1%}".format(mAP))
    print("CMC curve")
    for r in ranks:
        print("Rank-{:<3}: {:.1%}".format(r, cmc[r-1]))
    print("------------------")

    return cmc[0]

#######################################################################################################

print("Model is being initialized")     

model = hybrid_convnet2.hybrid_cnn().to(device)
SAVED_MODEL_PATH = 'saved_models/p1.pth.tar'

checkpoint = torch.load(SAVED_MODEL_PATH)
model.load_state_dict(checkpoint['state_dict'])
start_epoch = checkpoint['epoch']

print("Model size: {:.5f}M".format(sum(p.numel() for p in model.parameters())/1000000.0))

model.fc0.conv2.register_forward_hook(get_activation('fc0.conv2'))

optim = torch.optim.SGD(model.parameters(),lr=lr,momentum=0.9,weight_decay=weight_decay)
scheduler = CyclicLR(optim,gamma=gamma,step_size=stepsize)

num_epochs = 952


start_time = time.time()
train_time=0
best_rank1 = -np.inf
best_epoch=0

if evaluation:
    print("Evaluation in Progress")
    test(model,queryloader,galleryloader)
    sys.exit(0)

print("Training of model in progress")

for epoch in range(start_epoch,num_epochs) :
    start_train_time = time.time()
    train(epoch,model,optim,trainloader)
   
    
    scheduler.batch_step()

    if (epoch == 0 or epoch == 100 or epoch == 180 or epoch == 250 or epoch == 350 or epoch == 500 or epoch == 650 or epoch == 750 or epoch == 850 or epoch==950):
        print("Testing of model in progress")
        rank1 = test(model, queryloader,galleryloader)
        best = rank1 > best_rank1
        if best:
            best_rank1 = rank1
            best_epoch = epoch+1

    
        state_dict = model.state_dict()
        save_checkpoint({
        'state_dict':state_dict,
        'rank1':rank1,
        'epoch':epoch,
    },best,osp.join(PATH,'checkpoint'+str(epoch+1)+'pth.tar'))

    print("Best Rank-1 {:.1%},acheived at epoch ".format(best_rank1,best_epoch))


        

    



