import torch.nn as nn
from torchvision import datasets, transforms
from torch.autograd import Variable

import argparse
import numpy as np
import os
import time
import sys
import models as arch_module
import torch

sys.path.append('..')


# Prune settings
parser = argparse.ArgumentParser(description='PyTorch CIFAR prune')
parser.add_argument('--data', metavar='DIR',
                    default='../../../data/image-net', help='path to dataset')
parser.add_argument('--test-batch-size', type=int, default=64, metavar='N',
                    help='input batch size for testing (default: 64)')
parser.add_argument('--arch', type=str, default='resnet18',
                    help='depth of the resnet')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--model', default='', type=str, metavar='PATH',
                    help='path to the model (default: none)')
parser.add_argument('--save', default='', type=str, metavar='PATH',
                    help='path to save pruned model (default: none)')
parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',
                    help='number of data loading workers (default: 20)')
parser.add_argument('-v', default='A', type=str,
                    help='version of the pruned model')
parser.add_argument('--random_rank', dest='random_rank', action='store_true')
parser.set_defaults(random_rank=False)

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

args = parser.parse_args()
args.cuda = torch.cuda.is_available()

if not os.path.exists(args.save):
    os.makedirs(args.save)

model = arch_module.__dict__[args.arch](pretrained=True)
model = torch.nn.DataParallel(model)

if args.model:
    if os.path.isfile(args.model):
        print("=> loading checkpoint '{}'".format(args.model))
        checkpoint = torch.load(args.model, map_location='cpu')
        args.start_epoch = checkpoint['epoch']
        best_prec1 = checkpoint['best_prec1'] if 'best_prec1' in checkpoint.keys(
        ) else checkpoint['acc']
        cfg = checkpoint['cfg'] if 'cfg' in checkpoint.keys() else None
        model = arch_module.__dict__[args.arch](cfg=cfg)
        model = torch.nn.DataParallel(model)
        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}' (epoch {}) Prec1: {:f}"
              .format(args.model, checkpoint['epoch'], best_prec1))
    else:
        raise ValueError("No checkpoint found at '{}'".format(args.model))

if args.cuda:
    model.cuda()

print('Pre-processing Successful!')


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def test(model):
    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(os.path.join(args.data, 'val'), transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.test_batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    model.eval()
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    criterion = nn.CrossEntropyLoss().cuda()

    end = time.time()

    with torch.no_grad():
        for i, (data, target) in enumerate(val_loader):
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            output = model(data)
            loss = criterion(output, target)

            # get the index of the max log-probability
            pred = output.data.max(1, keepdim=True)[1]
            # measure accuracy and record loss
            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
            losses.update(loss.item(), data.size(0))
            top1.update(prec1.item(), data.size(0))
            top5.update(prec5.item(), data.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                          i, len(val_loader), batch_time=batch_time, loss=losses,
                          top1=top1, top5=top5))
    print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))
    return top1.avg, top5.avg


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

# acc, _ = test(model)


if args.arch == 'resnet18':
    if args.v == 'DCP30':
        pruning_plan = [
            ('layer1.0', 0.3), ('layer1.1', 0.3),
            ('layer2.0', 0.3), ('layer2.1', 0.3),
            ('layer3.0', 0.3), ('layer3.1', 0.3),
            ('layer4.0', 0.35), ('layer4.1', 0.35)
        ]
    elif args.v == 'DCP50':
        pruning_plan = [
            ('layer1.0', 0.5), ('layer1.1', 0.5),
            ('layer2.0', 0.5), ('layer2.1', 0.5),
            ('layer3.0', 0.5), ('layer3.1', 0.5),
            ('layer4.0', 0.55), ('layer4.1', 0.55)
        ]
    elif args.v == 'DCP70':
        pruning_plan = [
            ('layer1.0', 0.7), ('layer1.1', 0.7),
            ('layer2.0', 0.7), ('layer2.1', 0.7),
            ('layer3.0', 0.7), ('layer3.1', 0.7),
            ('layer4.0', 0.75), ('layer4.1', 0.75)
        ]
elif args.arch == 'resnet34':
    if args.v == 'B':
        pruning_plan = [
            ('layer1.0', 0.0), ('layer1.1', 0.5), ('layer1.2', 0.5),
            ('layer2.0', 0.0), ('layer2.1',
                                0.6), ('layer2.2', 0.6), ('layer2.3', 0.0),
            ('layer3.0', 0.0), ('layer3.1',
                                0.4), ('layer3.2', 0.4), ('layer3.3', 0.4),
            ('layer3.4', 0.4), ('layer3.5', 0.0),
            ('layer4.0', 0.0), ('layer4.1', 0.0), ('layer4.2', 0.0)
        ]
    elif args.v == 'Taylor82':  # 16865716
        pruning_plan = [
            ('layer1.0', 0.3), ('layer1.1', 0.3), ('layer1.2', 0.3),
            ('layer2.0', 0.3), ('layer2.1',
                                0.3), ('layer2.2', 0.3), ('layer2.3', 0.3),
            ('layer3.0', 0.3), ('layer3.1',
                                0.3), ('layer3.2', 0.3), ('layer3.3', 0.3),
            ('layer3.4', 0.3), ('layer3.5', 0.3),
            ('layer4.0', 0.3), ('layer4.1', 0.3), ('layer4.2', 0.0)
        ]
elif args.arch == 'resnet50':
    if args.v == 'DCP30':  # 16945246
        pruning_plan = [
            ('layer1.0', 0.3), ('layer1.1', 0.3), ('layer1.2', 0.3),
            ('layer2.0', 0.3), ('layer2.1',
                                0.3), ('layer2.2', 0.3), ('layer2.3', 0.3),
            ('layer3.0', 0.3), ('layer3.1',
                                0.3), ('layer3.2', 0.3), ('layer3.3', 0.3),
            ('layer3.4', 0.3), ('layer3.5', 0.3),
            ('layer4.0', 0.3), ('layer4.1', 0.3), ('layer4.2', 0.3)
        ]
    elif args.v == 'DCP50':  # 12381864
        pruning_plan = [
            ('layer1.0', 0.5), ('layer1.1', 0.5), ('layer1.2', 0.5),
            ('layer2.0', 0.5), ('layer2.1',
                                0.5), ('layer2.2', 0.5), ('layer2.3', 0.5),
            ('layer3.0', 0.5), ('layer3.1',
                                0.5), ('layer3.2', 0.5), ('layer3.3', 0.5),
            ('layer3.4', 0.5), ('layer3.5', 0.5),
            ('layer4.0', 0.5), ('layer4.1', 0.5), ('layer4.2', 0.5)
        ]
    elif args.v == 'DCP70':  # 8665318
        pruning_plan = [
            ('layer1.0', 0.7), ('layer1.1', 0.7), ('layer1.2', 0.7),
            ('layer2.0', 0.7), ('layer2.1',
                                0.7), ('layer2.2', 0.7), ('layer2.3', 0.7),
            ('layer3.0', 0.7), ('layer3.1',
                                0.7), ('layer3.2', 0.7), ('layer3.3', 0.7),
            ('layer3.4', 0.7), ('layer3.5', 0.7),
            ('layer4.0', 0.7), ('layer4.1', 0.7), ('layer4.2', 0.7)
        ]
    elif args.v == 'GAL1':  # 14532769
        pruning_plan = [
            ('layer1.0', 0.4), ('layer1.1', 0.4), ('layer1.2', 0.4),
            ('layer2.0', 0.4), ('layer2.1',
                                0.4), ('layer2.2', 0.4), ('layer2.3', 0.4),
            ('layer3.0', 0.4), ('layer3.1',
                                0.4), ('layer3.2', 0.4), ('layer3.3', 0.4),
            ('layer3.4', 0.4), ('layer3.5', 0.4),
            ('layer4.0', 0.4), ('layer4.1', 0.4), ('layer4.2', 0.4)
        ]
    elif args.v == 'GAL0.5':  # 21105070
        pruning_plan = [
            ('layer1.0', 0.14), ('layer1.1', 0.14), ('layer1.2', 0.14),
            ('layer2.0', 0.14), ('layer2.1',
                                 0.14), ('layer2.2', 0.14), ('layer2.3', 0.14),
            ('layer3.0', 0.14), ('layer3.1',
                                 0.14), ('layer3.2', 0.14), ('layer3.3', 0.14),
            ('layer3.4', 0.14), ('layer3.5', 0.14),
            ('layer4.0', 0.15), ('layer4.1', 0.15), ('layer4.2', 0.15)
        ]
    elif args.v == 'Taylor56':  # 7961423
        pruning_plan = [
            ('layer1.0', 0.7), ('layer1.1', 0.7), ('layer1.2', 0.7),
            ('layer2.0', 0.7), ('layer2.1',
                                0.7), ('layer2.2', 0.7), ('layer2.3', 0.7),
            ('layer3.0', 0.75), ('layer3.1',
                                 0.75), ('layer3.2', 0.75), ('layer3.3', 0.75),
            ('layer3.4', 0.75), ('layer3.5', 0.75),
            ('layer4.0', 0.75), ('layer4.1', 0.75), ('layer4.2', 0.75)
        ]
    elif args.v == 'Taylor72':  # 14229642
        pruning_plan = [
            ('layer1.0', 0.4), ('layer1.1', 0.4), ('layer1.2', 0.4),
            ('layer2.0', 0.4), ('layer2.1',
                                0.4), ('layer2.2', 0.4), ('layer2.3', 0.4),
            ('layer3.0', 0.4), ('layer3.1',
                                0.4), ('layer3.2', 0.4), ('layer3.3', 0.4),
            ('layer3.4', 0.4), ('layer3.5', 0.4),
            ('layer4.0', 0.42), ('layer4.1', 0.42), ('layer4.2', 0.42)
        ]
    elif args.v == 'Taylor81':  # 17913255
        pruning_plan = [
            ('layer1.0', 0.25), ('layer1.1', 0.25), ('layer1.2', 0.25),
            ('layer2.0', 0.25), ('layer2.1',
                                 0.25), ('layer2.2', 0.25), ('layer2.3', 0.25),
            ('layer3.0', 0.25), ('layer3.1',
                                 0.25), ('layer3.2', 0.25), ('layer3.3', 0.25),
            ('layer3.4', 0.25), ('layer3.5', 0.25),
            ('layer4.0', 0.27), ('layer4.1', 0.27), ('layer4.2', 0.27)
        ]
    elif args.v == 'Taylor91':  # 22420635
        pruning_plan = [
            ('layer1.0', 0.10), ('layer1.1', 0.10), ('layer1.2', 0.10),
            ('layer2.0', 0.10), ('layer2.1',
                                 0.10), ('layer2.2', 0.10), ('layer2.3', 0.10),
            ('layer3.0', 0.10), ('layer3.1',
                                 0.10), ('layer3.2', 0.10), ('layer3.3', 0.10),
            ('layer3.4', 0.10), ('layer3.5', 0.10),
            ('layer4.0', 0.10), ('layer4.1', 0.10), ('layer4.2', 0.10)
        ]
    elif args.v == 'FPGM30':  # TODO:
        pruning_plan = [
            ('layer1.0', 0.3), ('layer1.1', 0.3), ('layer1.2', 0.3),
            ('layer2.0', 0.3), ('layer2.1',
                                0.3), ('layer2.2', 0.3), ('layer2.3', 0.3),
            ('layer3.0', 0.3), ('layer3.1',
                                0.3), ('layer3.2', 0.3), ('layer3.3', 0.3),
            ('layer3.4', 0.3), ('layer3.5', 0.3),
            ('layer4.0', 0.3), ('layer4.1', 0.3), ('layer4.2', 0.3)
        ]
    elif args.v == 'FPGM40':  # TODO:
        pruning_plan = [
            ('layer1.0', 0.4), ('layer1.1', 0.4), ('layer1.2', 0.4),
            ('layer2.0', 0.4), ('layer2.1',
                                0.4), ('layer2.2', 0.4), ('layer2.3', 0.4),
            ('layer3.0', 0.4), ('layer3.1',
                                0.4), ('layer3.2', 0.4), ('layer3.3', 0.4),
            ('layer3.4', 0.4), ('layer3.5', 0.4),
            ('layer4.0', 0.4), ('layer4.1', 0.4), ('layer4.2', 0.4)
        ]
    elif args.v == 'HRankPlus11.05':  # 11004777
        pruning_plan = [
            ('layer1.0', 0.55), ('layer1.1', 0.55), ('layer1.2', 0.55),
            ('layer2.0', 0.55), ('layer2.1',
                                 0.55), ('layer2.2', 0.55), ('layer2.3', 0.55),
            ('layer3.0', 0.55), ('layer3.1',
                                 0.55), ('layer3.2', 0.55), ('layer3.3', 0.55),
            ('layer3.4', 0.55), ('layer3.5', 0.55),
            ('layer4.0', 0.58), ('layer4.1', 0.58), ('layer4.2', 0.58)
        ]
    elif args.v == 'HRankPlus15.09':  # 14987634
        pruning_plan = [
            ('layer1.0', 0.35), ('layer1.1', 0.35), ('layer1.2', 0.35),
            ('layer2.0', 0.35), ('layer2.1',
                                 0.35), ('layer2.2', 0.35), ('layer2.3', 0.35),
            ('layer3.0', 0.35), ('layer3.1',
                                 0.35), ('layer3.2', 0.35), ('layer3.3', 0.35),
            ('layer3.4', 0.35), ('layer3.5', 0.35),
            ('layer4.0', 0.4), ('layer4.1', 0.4), ('layer4.2', 0.4)
        ]
    elif args.v == 'EagleEye50':  # 13403078 FLOPs ratio: 0.4929744950026542
        pruning_plan = [
            ('layer1.0', 0.45), ('layer1.1', 0.45), ('layer1.2', 0.45),
            ('layer2.0', 0.45), ('layer2.1',
                                 0.45), ('layer2.2', 0.45), ('layer2.3', 0.45),
            ('layer3.0', 0.45), ('layer3.1',
                                 0.45), ('layer3.2', 0.45), ('layer3.3', 0.45),
            ('layer3.4', 0.45), ('layer3.5', 0.45),
            ('layer4.0', 0.45), ('layer4.1', 0.45), ('layer4.2', 0.45)
        ]
    elif args.v == 'EagleEye25':  # 7391609 FLOPs ratio: 0.24624503866204167
        pruning_plan = [
            ('layer1.0', 0.76), ('layer1.1', 0.76), ('layer1.2', 0.76),
            ('layer2.0', 0.76), ('layer2.1',
                                 0.76), ('layer2.2', 0.76), ('layer2.3', 0.76),
            ('layer3.0', 0.76), ('layer3.1',
                                 0.76), ('layer3.2', 0.76), ('layer3.3', 0.76),
            ('layer3.4', 0.76), ('layer3.5', 0.76),
            ('layer4.0', 0.80), ('layer4.1', 0.80), ('layer4.2', 0.80)
        ]
    elif args.v == 'Provable18.01':  # 20931915
        pruning_plan = [
            ('layer1.0', 0.14), ('layer1.1', 0.14), ('layer1.2', 0.14),
            ('layer2.0', 0.14), ('layer2.1',
                                 0.14), ('layer2.2', 0.14), ('layer2.3', 0.14),
            ('layer3.0', 0.14), ('layer3.1',
                                 0.14), ('layer3.2', 0.14), ('layer3.3', 0.14),
            ('layer3.4', 0.14), ('layer3.5', 0.14),
            ('layer4.0', 0.16), ('layer4.1', 0.16), ('layer4.2', 0.16)
        ]
    elif args.v == 'Provable44.04':  # 14229642
        pruning_plan = [
            ('layer1.0', 0.4), ('layer1.1', 0.4), ('layer1.2', 0.4),
            ('layer2.0', 0.4), ('layer2.1',
                                0.4), ('layer2.2', 0.4), ('layer2.3', 0.4),
            ('layer3.0', 0.4), ('layer3.1',
                                0.4), ('layer3.2', 0.4), ('layer3.3', 0.4),
            ('layer3.4', 0.4), ('layer3.5', 0.4),
            ('layer4.0', 0.42), ('layer4.1', 0.42), ('layer4.2', 0.42)
        ]
else:
    raise ValueError(
        "Expect arch to be one of [resnet18, resnet34, resnet50] but got {}".format(args.arch))


def filter_pruning(model, pruning_plan):
    # get the cfg of pruned network
    cfg = []
    parallel_instance = False
    if isinstance(model, nn.DataParallel):
        model = model.module
        parallel_instance = True
    for block_name, prune_prob in pruning_plan:
        block = model.get_block(block_name)
        conv_layers = list(filter(lambda layer: isinstance(
            layer, nn.Conv2d), block.modules()))
        out_channels = conv_layers[0].out_channels
        num_keep = int(out_channels*(1-prune_prob))
        cfg.append(num_keep)
    # construct pruned network
    new_model = arch_module.__dict__[args.arch](cfg=cfg)
    # copy weight from original network to new network
    is_last_conv_pruned = False
    mask = None  # mask of pruned layer
    for [m0, m1] in zip(model.modules(), new_model.modules()):
        if isinstance(m0, nn.Conv2d):
            is_channel_pruned = False
            pre_prune_shape = m1.weight.data.shape
            # current layer is not modified
            if (m0.in_channels == m1.in_channels) and (m0.out_channels == m1.out_channels):
                m1.weight.data = m0.weight.data.clone()
                is_last_conv_pruned = False
            # CASE 1: PREVIOUS layer is pruned
            # remove the input weights corresponding to removed filter
            # Importance: as some layer could be pruned while having prior layer pruned as well
            # hence, it's crucial to set this condition above the m0.out_channels > m1.out_channels
            # as the is_last_conv_pruned flag would be set to False and can be rewrite if aforemention situation happend
            if m0.in_channels > m1.in_channels:
                # the filter would always
                channel_idx = np.squeeze(np.argwhere(
                    np.asarray(mask.cpu().numpy())))
                if channel_idx.size == 1:
                    channel_idx = np.resize(channel_idx, (1,))
                w = m0.weight.data[:, channel_idx.tolist(), :, :].clone()
                m1.weight.data = w.clone()
                is_last_conv_pruned = False
                is_channel_pruned = True
            # CASE 2: CURRENT layer's filters are pruned
            # copy kept filter weight to new model
            if m0.out_channels > m1.out_channels:
                mask = create_l1_norm_mask(
                    m0, m1.out_channels, random_rank=args.random_rank)
                filter_idx = np.squeeze(np.argwhere(
                    np.asarray(mask.cpu().numpy())))
                if filter_idx.size == 1:
                    filter_idx = np.resize(filter_idx, (1,))
                if not is_channel_pruned:
                    # only filter is pruned
                    w = m0.weight.data[filter_idx.tolist(), :, :, :].clone()
                else:
                    # both channel and filter are pruned
                    w = m0.weight.data[filter_idx.tolist(), :, :, :].clone()
                    w = w[:, channel_idx.tolist(), :, :]
                m1.weight.data = w.clone()
                is_last_conv_pruned = True
            after_prune_shape = m1.weight.data.shape
            if pre_prune_shape != after_prune_shape:
                print(pre_prune_shape)
                print(after_prune_shape)
                print(m0)
                print(m1)
                raise Exception(
                    'Pruned weight and its prepruned weight have mismatch shape')
        # adjust batchnorm with corresponding filter
        elif isinstance(m0, nn.BatchNorm2d):
            # if last conv layer is pruned then modify the batchnorm as well
            if is_last_conv_pruned:
                filter_idx = np.squeeze(np.argwhere(
                    np.asarray(mask.cpu().numpy())))
                if filter_idx.size == 1:
                    filter_idx = np.resize(filter_idx, (1,))
                m1.weight.data = m0.weight.data[filter_idx.tolist()].clone()
                m1.bias.data = m0.bias.data[filter_idx.tolist()].clone()
                m1.running_mean = m0.running_mean[filter_idx.tolist()].clone()
                m1.running_var = m0.running_var[filter_idx.tolist()].clone()
            # if the last conv layer wasn't modified then simply copy weights
            else:
                m1.weight.data = m0.weight.data.clone()
                m1.bias.data = m0.bias.data.clone()
                m1.running_mean = m0.running_mean.clone()
                m1.running_var = m0.running_var.clone()
        # linear layer will not be pruned
        elif isinstance(m0, nn.Linear):
            m1.weight.data = m0.weight.data.clone()
            m1.bias.data = m0.bias.data.clone()
    if parallel_instance:
        new_model = nn.DataParallel(new_model)
    return new_model


def create_l1_norm_mask(layer, num_keep, random_rank=False):
    """
        Create a 1d tensor with binary value, the i-th element is set to 1
            if i-th output filter of this layer is kept, 0 otherwise
            selection criterion - l1 norm based
        :param layer: nn.Conv2d
        :param num_keep: int - number of filters that would be keep
        :return: 1D torch.tensor - mask
    """
    out_channels = layer.out_channels
    weight_copy = layer.weight.data.abs().clone().cpu().numpy()
    L1_norm = np.sum(weight_copy, axis=(1, 2, 3))
    if random_rank:
        L1_norm = np.random.random_sample(L1_norm.shape)
    arg_max = np.argsort(L1_norm)
    arg_max_rev = arg_max[::-1][:num_keep]
    # create mask
    mask = torch.zeros(out_channels)
    mask[arg_max_rev.tolist()] = 1
    return mask


if __name__ == '__main__':
    acc_top1, acc_top5 = test(model)
    newmodel = filter_pruning(model, pruning_plan)
    new_acc_top1, new_acc_top5 = test(newmodel)
    torch.save({
        'cfg': newmodel.module.cfg,
        'state_dict': newmodel.state_dict()
    },
        os.path.join(args.save, 'pruned.pth.tar'))

    print(newmodel)
    cfg = newmodel.module.cfg
    print('cfg: ', str(cfg))
    num_parameters1 = sum([param.nelement() for param in model.parameters()])
    num_parameters2 = sum([param.nelement()
                           for param in newmodel.parameters()])
    print("number of parameters: " + str(num_parameters2))
    with open(os.path.join(args.save, "prune.txt"), "w") as fp:
        fp.write("Before pruning: "+"\n")
        fp.write("acc@1: "+str(acc_top1)+"\n"+"acc@5: "+str(acc_top5)+"\n")
        fp.write("Number of parameters: \n"+str(num_parameters1)+"\n")
        fp.write("==========================================\n")
        fp.write("After pruning: "+"\n")
        fp.write("cfg :"+"\n")
        fp.write(str(cfg)+"\n")
        fp.write("acc@1: "+str(new_acc_top1)+"\n" +
                 "acc@5: "+str(new_acc_top5)+"\n")
        fp.write("Number of parameters: \n" + str(num_parameters2) + "\n")
