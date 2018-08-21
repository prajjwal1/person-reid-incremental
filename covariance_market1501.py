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
from models import hybrid_convnet2, hybrid_linear
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
train_batch = 32
test_batch = 32
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

#dataset_name = 'cuhk03'
dataset_name = 'dukemtmcreid'
evaluation = False
num_classes = 702    #702 for Duke MMTC   #751 for market 1501 #1360 for CUHK-03
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
    pin_memory=pin_memory,drop_last=True,
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
def Flatten(input):
    return input.contiguous().view(input.size(0), -1)


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

    model.fc0.train(False)
    model.fc1.train(True)  

    output_fc = "fc1"
     
    model.base.train(True)
    
    ################################################3
    person_per_batch = 8
    imgs_per_person = 4


    bmask = []
    l_all_pos = []
    l_all_neg = []
    pos_targets = torch.Tensor()
    neg_targets = torch.Tensor()
    C_pos = torch.zeros([train_batch,256,2,4],device=device)
    C_neg = torch.zeros([train_batch,256,2,4],device=device)
###################################

    for batch,(imgs,pids,camids) in enumerate(trainloader):
        #imgs,pids = imgs.cuda(), pids.cuda()
        pids = torch.Tensor.numpy(pids)
        camids = torch.Tensor.numpy(camids)
        uid = list(set(pids))
        mask = np.zeros([2*person_per_batch,person_per_batch*imgs_per_person])
        for i in range(len(uid)):
            sel = uid[i]
        # print(sel)
            pos = -1
            neg = -1
            k = -1
            for j in range(len(pids)):
                if (pids[j]==sel):
                    k = j
                    break
                
            for j in range(len(pids)):
                if (pids[k]==pids[j] and camids[k]!= camids[j]):   # Same IDs and diff cam IDs
                    pos = j                        #Postive        
                    break
                
            for j in range(len(pids)):
                if (pids[k]!=pids[j]):             #Negative                # Diff Cam IDs
                    neg = j
                    break

            mask[2*i][k] = 1
            mask[2*i][pos] = 1
            mask[2*i+1][k] = 1
            mask[2*i+1][neg] = 1

        bmask.append(mask)
    
        l_batch_pos = []
        l_batch_neg = []
        kl = mask  #bmask[batch]
        for i in range(len(kl)):
            l5 = []
            for j in range(len(kl[i])):
                if (kl[i][j]==1):
                    l5.append(j)
            if i %2 <1:
                l_batch_pos.append(l5)
            else:
                l_batch_neg.append(l5)
        l_all_pos.append(l_batch_pos)
        l_all_neg.append(l_batch_neg)



        data_time.update(time.time()-end)

        clf_outputs = model(imgs.cuda())
        f = activation['fc1.conv2']  #bs,2048,8,4
        f = f.permute(0,3,1,2)
        m = nn.AdaptiveAvgPool2d((256,2))
        f = m(f)
        f = f.permute(0,2,3,1)

        fc1 = clf_outputs[output_fc] 

        for i in range(len(l_batch_pos)):
            pos_idx0 = l_batch_pos[i][0]
            pos_idx1 = l_batch_pos[i][1]
        #print(f[pos_idx0].shape)
            pos_targets = torch.sub(f[pos_idx1],f[pos_idx0])
            C_pos += pos_targets
        #print(pos_targets.shape)
        #pos_targets = torch.Tensor(pos_targets)
    
        for i in range(len(l_batch_neg)):
            neg_idx0 = l_batch_neg[i][0]
            neg_idx1 = l_batch_neg[i][1]
            neg_targets = torch.sub(f[neg_idx1],f[neg_idx0])
            C_neg += neg_targets
        
        g = Flatten(C_pos)
        
        y = Flatten(C_neg)
    
        u = g-y               # (bs,2048)
        v = torch.unsqueeze(u,2)       # (64,2048,1)
        w = v.permute(0,2,1)           # (64,1,2048)
        x_net = torch.matmul(v,w)      # (64,2048,2048)   
        y  = torch.sum(x_net)
        y = F.relu(y)
        alpha = 1e-9
        beta=0
        covariance_loss = 1*(alpha*y-beta)

        pids = torch.from_numpy(pids)
        pids = pids.cuda()


        if isinstance(fc1,tuple):  
            cross_entropy_loss = DeepSuperVision(cross_entropy,fc1,pids)
        else:
            cross_entropy_loss = cross_entropy(fc1,pids)
        """
        if isinstance(f,tuple):
            triplet = DeepSuperVision(triplet_loss_fn,f,pids)
        else:
            triplet = triplet_loss_fn(f,pids)
        """
        #print("xent", cross_entropy_loss)
        #print("covariance", covariance_loss)
        loss = cross_entropy_loss + covariance_loss
        #print("xent", cross_entropy_loss)
        #print("covariance_loss", covariance_loss)

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
            
            fc1_preds = model(imgs)
            
            batch_time.update(time.time()-end)

            output_fc = "fc1"       
            fc1 = fc1_preds[output_fc]
            fc1 = fc1.data.cpu()  

            qf.append(fc1)
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
           
            fc1_preds = model(imgs)

            output_fc = "fc1"
            fc1 = fc1_preds[output_fc]
            
            batch_time.update(time.time()-end)

            fc1 = fc1.data.cpu()
            gf.append(fc1)
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
#start_epoch = checkpoint['epoch']

print("Model size: {:.5f}M".format(sum(p.numel() for p in model.parameters())/1000000.0))

model.fc1.conv2.register_forward_hook(get_activation('fc1.conv2'))

optim = torch.optim.SGD(model.parameters(),lr=lr,momentum=0.9,weight_decay=weight_decay)
scheduler = CyclicLR(optim,gamma=gamma,step_size=stepsize)

num_epochs = 1202


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

    if (epoch == 0 or epoch == 100 or epoch == 180 or epoch == 250 or epoch == 350 or epoch == 500 or epoch == 650 or epoch == 750 or epoch == 850 or epoch==950 or epoch==1100 or epoch == 1200):
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


        

    


