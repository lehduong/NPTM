from __future__ import print_function
import argparse
import numpy as np
import os
import shutil

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

import models
import wandb
from utils import get_train_valid_loader, get_test_loader, AverageMeter
from compute_flops import print_model_param_flops


# Training settings
parser = argparse.ArgumentParser(description='PyTorch Slimming CIFAR training')
parser.add_argument('--dataset', type=str, default='cifar100',
                    help='training dataset (default: cifar100)')
parser.add_argument('--scratch', default='', type=str, metavar='PATH',
                    help='path to the pruned model')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=256, metavar='N',
                    help='input batch size for testing (default: 256)')
parser.add_argument('--epochs', type=int, default=160, metavar='N',
                    help='number of epochs to train (default: 160)')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 8)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                    help='learning rate (default: 0.1)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--save', default='./logs', type=str, metavar='PATH',
                    help='path to save prune model (default: current directory)')
parser.add_argument('--arch', default='vgg', type=str,
                    help='architecture to use')
parser.add_argument('--depth', default=16, type=int,
                    help='depth of the neural network')
parser.add_argument('--wandb_name', default='resnet_56_standard', type=str,
                    help='name of wandb run')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

wandb.init(
    name=args.wandb_name,
    project='Clean_Rethinking_PFEC',
    config={
        **vars(args)}
)


torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

if not os.path.exists(args.save):
    os.makedirs(args.save)

kwargs = {'num_workers': args.workers, 'pin_memory': True} if args.cuda else {}

train_loader, val_loader = get_train_valid_loader('../../../data',
                                                  batch_size=args.batch_size,
                                                  random_seed=args.seed,
                                                  num_workers=args.workers,
                                                  valid_size=0.1,
                                                  pin_memory=True,
                                                  dataset=args.dataset
                                                  )
test_loader = get_test_loader('../../../data',
                              batch_size=args.batch_size,
                              shuffle=False,
                              num_workers=args.workers,
                              pin_memory=True,
                              dataset=args.dataset
                              )

model = models.__dict__[args.arch](dataset=args.dataset, depth=args.depth)

if args.scratch:
    checkpoint = torch.load(args.scratch)
    model = models.__dict__[args.arch](
        dataset=args.dataset, depth=args.depth, cfg=checkpoint['cfg'])

model_ref = models.__dict__[args.arch](dataset=args.dataset, depth=args.depth)

flops_std = print_model_param_flops(model_ref, 32)
flops_small = print_model_param_flops(model, 32)
args.epochs = int(args.epochs * (flops_std / flops_small))

if args.cuda:
    model.cuda()

optimizer = optim.SGD(model.parameters(), lr=args.lr,
                      momentum=args.momentum, weight_decay=args.weight_decay)

if args.resume:
    if os.path.isfile(args.resume):
        print("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
        args.start_epoch = checkpoint['epoch']
        best_val_prec1 = checkpoint['best_prec1']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {}) Prec1: {:f}"
              .format(args.resume, checkpoint['epoch'], best_val_prec1))
    else:
        print("=> no checkpoint found at '{}'".format(args.resume))


def train(epoch):
    model.train()
    avg_loss = 0.
    train_acc = 0.
    for batch_idx, (data, target) in enumerate(train_loader):
        cur_lr = next(iter(optimizer.param_groups))['lr']
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        avg_loss += loss.item()
        pred = output.data.max(1, keepdim=True)[1]
        train_acc += pred.eq(target.data.view_as(pred)).cpu().sum()
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.1f}%)]\tLoss: {:.6f}\tLR: {:.4f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item(), cur_lr))
    avg_loss /= len(train_loader.dataset)

    return avg_loss


def validate():
    model.eval()
    test_loss = 0
    correct = 0
    top1 = AverageMeter()
    with torch.no_grad():
        for data, target in val_loader:
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            output = model(data)
            # sum up batch loss
            test_loss += F.cross_entropy(output, target, size_average=False)
            # get the index of the max log-probability
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()
            top1.update(1.0*pred.eq(target.data.view_as(
                pred)).cpu().sum()/data.size(0), data.size(0))

    test_loss /= len(val_loader.dataset)
    print('\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.1f}%)\n'.format(
        test_loss, correct, len(val_loader.dataset),
        100. * top1.avg))

    return top1.avg, test_loss


def test():
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            output = model(data)
            # sum up batch loss
            test_loss += F.cross_entropy(output, target, size_average=False)
            # get the index of the max log-probability
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.1f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    return correct / float(len(test_loader.dataset)), test_loss


def save_checkpoint(state, is_best, filepath):
    torch.save(state, os.path.join(filepath, 'checkpoint.pth.tar'))
    if is_best:
        shutil.copyfile(os.path.join(filepath, 'checkpoint.pth.tar'),
                        os.path.join(filepath, 'model_best.pth.tar'))


best_val_prec1 = 0.
best_model_perf = 0.

for epoch in range(args.start_epoch, args.epochs):
    if epoch in [int(args.epochs*0.5), int(args.epochs*0.75)]:
        for param_group in optimizer.param_groups:
            param_group['lr'] *= 0.1
    cur_lr = next(iter(optimizer.param_groups))['lr']

    # train and eval
    avg_train_loss = train(epoch)
    val_prec1, avg_val_loss = validate()
    test_prec1, avg_test_loss = test()

    # find best-val
    is_best = val_prec1 > best_val_prec1
    best_val_prec1 = max(val_prec1, best_val_prec1)
    if is_best:
        best_model_perf = test_prec1

    # log wandb
    wandb.log({'top1': test_prec1, 'best_model_perf': best_model_perf, 'val_top1': val_prec1, 'best_val': best_val_prec1, 'lr': cur_lr,
               'avg_train_loss': avg_train_loss, 'avg_val_loss': avg_val_loss, 'avg_test_loss': avg_test_loss})

    save_checkpoint({
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'best_prec1': best_val_prec1,
        'optimizer': optimizer.state_dict(),
        'cfg': model.cfg
    }, is_best, filepath=args.save)

wandb.save(os.path.join(args.save, '*'))
