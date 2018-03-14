#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader

import torchvision.datasets as dset
import torchvision.models as models
import torchvision.transforms as transforms

import argparse
import math
import os
import setproctitle
import shutil
import sys

import config
import densenet
from transforms import RandomBrightness

def main():
    parser = argparse.ArgumentParser()
    #parser.add_argument('-n', '--network-name', type=str, required=True)
    #parser.add_argument('-d', '--dataset-name', type=str, required=True)
    #parser.add_argument('-c', '--num-classes', type=int, required=True)
    network_name = 'densenet-121'
    dataset_name = 'cifar10'
    num_classes = 10
    parser.add_argument('-m', '--multilabel', type=bool, default=False)
    parser.add_argument('-p', '--pretrained', type=bool, default=False)
    parser.add_argument('-l', '--load')
    parser.add_argument('--batchSz', type=int, default=64)
    parser.add_argument('--nEpochs', type=int, default=300)
    parser.add_argument('--sEpoch', type=int, default=1)
    parser.add_argument('--no-cuda', action='store_true')
    parser.add_argument('--save')
    parser.add_argument('--seed', type=int, default=50)
    parser.add_argument('--opt', type=str, default='sgd', choices=('sgd', 'adam', 'rmsprop'))
    args = parser.parse_args()

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    args.save = args.save or 'work/%s/%s' % (network_name, dataset_name)
    setproctitle.setproctitle(args.save)

    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    if os.path.exists(args.save):
        shutil.rmtree(args.save)
    os.makedirs(args.save, exist_ok=True)

    normMean = [0.49139968, 0.48215827, 0.44653124]
    normStd = [0.24703233, 0.24348505, 0.26158768]
    normTransform = transforms.Normalize(normMean, normStd)

    trainTransform = transforms.Compose([
        transforms.Scale(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
	RandomBrightness(-0.25, 0.25),
        normTransform
    ])
    testTransform = transforms.Compose([
        transforms.Scale(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normTransform
    ])

    kwargs = {'num_workers': 6, 'pin_memory': True} if args.cuda else {}
    trainLoader = DataLoader(
        config.get_dataset(dataset_name, 'train', trainTransform),
        batch_size=args.batchSz, shuffle=True, **kwargs)
    testLoader = DataLoader(
        config.get_dataset(dataset_name, 'test', testTransform),
        batch_size=args.batchSz, shuffle=False, **kwargs)

    if args.load:
        print("Loading network: {}".format(args.load))
        net = torch.load(args.load)
    else:
        net = config.get_network(network_name, num_classes, args.pretrained)
    
    #if True: # make this an optional
    #    net = torch.nn.DataParallel(net)
    
    print('  + Number of params: {}'.format(sum([p.data.nelement() for p in net.parameters()])))
    if args.cuda:
        net = net.cuda().half()

    if args.opt == 'sgd':
        optimizer = optim.SGD(net.parameters(), lr=1e-1, momentum=0.9, weight_decay=1e-4)
    elif args.opt == 'adam':
        optimizer = optim.Adam(net.parameters(), weight_decay=1e-4)
    elif args.opt == 'rmsprop':
        optimizer = optim.RMSprop(net.parameters(), weight_decay=1e-4)

    trainF = open(os.path.join(args.save, 'train.csv'), 'a')
    testF = open(os.path.join(args.save, 'test.csv'), 'a')

    for epoch in range(args.sEpoch, args.nEpochs + args.sEpoch):
        adjust_opt(args.opt, optimizer, epoch)
        train(args, epoch, net, trainLoader, optimizer, trainF)
        test(args, epoch, net, testLoader, optimizer, testF)
        torch.save(net, os.path.join(args.save, '%d.pth' % epoch))

    trainF.close()
    testF.close()

def train(args, epoch, net, trainLoader, optimizer, trainF):
    net.train()
    nProcessed = 0
    nTrain = len(trainLoader.dataset)
    for batch_idx, (data, target) in enumerate(trainLoader):
        if args.cuda:
            data = data.cuda().half()
            if args.multilabel:
                target = target.cuda().half()
            else:
                target = target.cuda().long()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = net(data)
        loss = config.get_loss_function(args.multilabel)(output, target)
        loss.backward()
        optimizer.step()
        nProcessed += len(data)
        if args.multilabel:
            pred = output.data.gt(0.5)
            tp = (pred + target.data.byte()).eq(2).sum()
            fp = (pred - target.data.byte()).eq(1).sum()
            fn = (pred - target.data.byte()).eq(-1).sum()
            tn = (pred + target.data.byte()).eq(0).sum()
            acc = (tp + tn) / (tp + tn + fp + fn)
            try:
                prec = tp / (tp + fp)
            except ZeroDivisionError:
                prec = 0.0
            try:
                rec = tp / (tp + fn)
            except ZeroDivisionError:
                rec = 0.0
            partialEpoch = epoch + batch_idx / len(trainLoader) - 1
            print('Epoch: {:.2f} [{}/{} ({:.0f}%)]\tLoss: {:.4f}\tAcc: {:.4f}\tPrec: {:.4f}\tRec: {:.4f}\tTP: {}\tFP: {}\tFN: {}\tTN: {}'.format(
                partialEpoch, nProcessed, nTrain, 100. * batch_idx / len(trainLoader),
                loss.data[0], acc, prec, rec, tp, fp, fn, tn))
            trainF.write('{},{},{},{},{}\n'.format(partialEpoch, loss.data[0], acc, prec, rec))
        else:
            pred = output.data.max(1)[1]
            incorrect = pred.ne(target.data).sum()
            err = 100.*incorrect/len(data)
            partialEpoch = epoch + batch_idx / len(trainLoader) - 1
            print('Train Epoch: {:.2f} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tError: {:.6f}'.format(
                partialEpoch, nProcessed, nTrain, 100. * batch_idx / len(trainLoader),
                loss.data[0], err))
            trainF.write('{},{},{}\n'.format(partialEpoch, loss.data[0], err))
        trainF.flush()

def test(args, epoch, net, testLoader, optimizer, testF):
    net.eval()
    test_loss = 0
    acc = prec = rec = 0
    incorrect = 0
    for data, target in testLoader:
        if args.cuda:
            data = data.cuda().half()
            if args.multilabel:
                target = target.cuda().half()
            else:
                target = target.cuda().long()
        data, target = Variable(data, volatile=True), Variable(target)
        output = net(data)
        test_loss += config.get_loss_function(args.multilabel)(output, target).data[0]
        if args.multilabel:
            pred = output.data.gt(0.5)
            tp = (pred + target.data.byte()).eq(2).sum()
            fp = (pred - target.data.byte()).eq(1).sum()
            fn = (pred - target.data.byte()).eq(-1).sum()
            tn = (pred + target.data.byte()).eq(0).sum()
            acc += (tp + tn) / (tp + tn + fp + fn)
            try:
                prec += tp / (tp + fp)
            except ZeroDivisionError:
                prec += 0.0
            try:
                rec += tp / (tp + fn)
            except ZeroDivisionError:
                rec += 0.0
        else:
            pred = output.data.max(1)[1] # get the index of the max log-probability
            incorrect += pred.ne(target.data).sum()
    test_loss /= len(testLoader)
    acc /= len(testLoader)
    prec /= len(testLoader)
    rec /= len(testLoader)
    if args.multilabel:
        print('\nTest set: Loss: {:.4f}, Acc: {:.4f}, Prec: {:.4f}, Rec: {:.4f}\n'.format(
            test_loss, acc, prec, rec))
        testF.write('{},{},{},{},{}\n'.format(epoch, test_loss, acc, prec, rec))
    else:
        nTotal = len(testLoader.dataset)
        err = 100. * incorrect / nTotal
        print('\nTest set: Average loss: {:.4f}, Error: {}/{} ({:.0f}%)\n'.format(
        test_loss, incorrect, nTotal, err))
        testF.write('{},{},{}\n'.format(epoch, test_loss, err))
    testF.flush()

def adjust_opt(optAlg, optimizer, epoch):
    if optAlg == 'sgd':
        if epoch < 150: lr = 1e-1
        elif epoch == 150: lr = 1e-2
        elif epoch == 225: lr = 1e-3
        else: return

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

if __name__=='__main__':
    main()
