import argparse
import os
import random
import shutil
import time
import warnings

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torch_warmup_lr import WarmupLR
from efficientnet_pytorch import EfficientNet
import timm
import wandb
import PIL

from utils import get_conv_zero_param, get_zero_param


model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name])) + ['efficientnet_b'+str(x) for x in range(0, 8)]

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                    ' | '.join(model_names) +
                    ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 8)')
parser.add_argument('--epochs', default=10, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=1000, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--pretrained_path', default='../../../iclr2021_checkpoints/pytorch_imagenet_weights/efficientnet-b0-355c32eb.pth',
                    type=str, help='path to (EfficientNet) pretrained model (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--world-size', default=1, type=int,
                    help='number of distributed processes')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='gloo', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--percent', default=0.1, type=float)
parser.add_argument('--save', default='', type=str)
parser.add_argument('--wandb_name', default='resnet_56_standard', type=str,
                    help='name of wandb run')

parser.add_argument('--schedule', type=int, nargs='+', default=[18, 32],
                    help='Decrease learning rate at these epochs.')
parser.add_argument('--init_lr', type=float, default=0.001,
                    help='initialized learning rate when doing warm up (default: 0.001)')
parser.add_argument('--gamma', type=float, default=0.1,
                    help='gamma of step learning rate decay (default: 0.1)')
parser.add_argument('--num_warmup', type=int, default=4,
                    help='number of epochs to increase learning rate (default: 4)')
parser.add_argument('--warmup_strategy', type=str, default='cos',
                    help='warmup strategy (default: cos)')

parser.add_argument(
    '--div_factor',
    type=float,
    default=100,
    help='div factor of OneCycle Learning rate Schedule (default: 10)')

parser.add_argument(
    '--final_div_factor',
    type=float,
    default=100,
    help='final div factor of OneCycle Learning rate Schedule (default: 100)')

parser.add_argument(
    '--pct_start',
    type=float,
    default=0.1,
    help='pct_start of OneCycle Learning rate Schedule (default: 0.1)')

parser.add_argument('--use_llr', dest='use_llr', action='store_true')
parser.add_argument('--use_lr_rewind',
                    dest='use_lr_rewind', action='store_true')
parser.add_argument('--use_onecycle', dest='use_onecycle', action='store_true')
parser.set_defaults(use_llr=False)
parser.set_defaults(use_lr_rewind=False)
parser.set_defaults(use_onecycle=False)

best_prec1 = 0


def main():
    global args, best_prec1
    args = parser.parse_args()

    if args.use_llr:
        args.num_warmup = int(0.1*args.epochs)
        args.schedule = [int(0.33*0.9*args.epochs), int(0.66*0.9*args.epochs)]
    if args.use_lr_rewind:
        # in original training, learning rate is reduce by factor of 10 every 30 epochs
        args.schedule = [args.epochs - x *
                         30 for x in reversed(range(1, args.epochs//30+1))]
        args.lr *= 10**(len(args.schedule))

    # init wandb
    wandb.init(
        name=args.wandb_name,
        project='Magnitude_Weights_Pruning',
        config={
            **vars(args), 'random_rank': str.find(args.wandb_name, 'random') > -1}  # TODO: Fix this heuristic :))
    )

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if not os.path.exists(args.save):
        os.makedirs(args.save)

    args.distributed = args.world_size > 1

    if args.distributed:
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size)

    # create model
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
        if args.arch.startswith('efficientnet'):
            # model = EfficientNet.from_pretrained(
            #    args.arch, weights_path=args.pretrained_path)
            model = timm.create_model(args.arch, pretrained=True)
        else:
            model = models.__dict__[args.arch](pretrained=True)
    else:
        print("=> creating model '{}'".format(args.arch))
        if args.arch.startswith('efficientnet'):
            # model = EfficientNet.from_name(args.arch)
            model = timm.create_model(args.arch)
        else:
            model = models.__dict__[args.arch]()

    if args.gpu is not None:
        model = model.cuda(args.gpu)
    elif args.distributed:
        model.cuda()
        model = torch.nn.parallel.DistributedDataParallel(model)
    else:
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    if args.arch.startswith('efficientnet'):
        print("Using RMSprop for optimization...")
        optimizer = torch.optim.RMSprop(model.parameters(), args.lr,
                                        momentum=args.momentum,
                                        weight_decay=args.weight_decay,
                                        eps=1e-3,
                                        alpha=0.9)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            model.load_state_dict(checkpoint['state_dict'])
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(
            train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    if args.arch.find('efficientnet') > -1:
        val_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(valdir, transforms.Compose([
                transforms.Resize(256, interpolation=PIL.Image.BICUBIC),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ])),
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True)
    else:
        val_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(valdir, transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ])),
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True)

    lr_scheduler = None

    if int(args.use_llr) + int(args.use_onecycle) + int(args.use_lr_rewind) > 1:
        raise ValueError(
            "Only one method among [large learning rate restarting, onecycle, learning rate rewinding] can be used at a time...")

    if args.use_llr:
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=args.schedule, gamma=args.gamma)
        lr_scheduler = WarmupLR(lr_scheduler, init_lr=args.init_lr,
                                num_warmup=args.num_warmup, warmup_strategy=args.warmup_strategy)
        # supress warning
        optimizer.zero_grad()
        optimizer.step()
        lr_scheduler.step()  # activate warm up lr

    elif args.use_onecycle:
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, div_factor=args.div_factor,
                                                           epochs=args.epochs, steps_per_epoch=len(train_loader), pct_start=args.pct_start,
                                                           final_div_factor=args.final_div_factor)
    elif args.use_lr_rewind:
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=args.schedule, gamma=args.gamma)

    if args.evaluate:
        validate(val_loader, model, criterion)
        return

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        #####################################################################################################
        num_parameters = get_conv_zero_param(model)
        print('Zero Conv parameters: {}'.format(num_parameters))
        num_parameters = get_zero_param(model)
        print('Zero parameters: {}'.format(num_parameters))
        num_parameters = sum([param.nelement()
                              for param in model.parameters()])
        print('Parameters: {}'.format(num_parameters))
        #####################################################################################################

        cur_lr = next(iter(optimizer.param_groups))['lr']

        # train for one epoch
        ret = train(train_loader, model, criterion,
                    optimizer, epoch, lr_scheduler)

        # evaluate on validation set
        prec1, prec5, test_loss = validate(val_loader, model, criterion)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer': optimizer.state_dict(),
        }, is_best, checkpoint=args.save)

        wandb.log({'top1': prec1, 'top5': prec5,
                   'best_top1': best_prec1, 'lr': cur_lr,
                   'train_top1': ret['train_top1'].avg,
                   'train_top5': ret['train_top5'].avg,
                   'train_loss': ret['train_loss'].avg,
                   'test_loss': test_loss})

        if args.use_llr or args.use_lr_rewind:
            lr_scheduler.step()

    wandb.save(os.path.join(args.save, '*'))

    return


def train(train_loader, model, criterion, optimizer, epoch, lr_scheduler=None):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            input = input.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)

        # compute output
        output = model(input)
        loss = criterion(output, target)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()

        for k, m in enumerate(model.modules()):
            # print(k, m)
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                weight_copy = m.weight.data.abs().clone()
                mask = weight_copy.gt(0).float().cuda()
                m.weight.grad.data.mul_(mask)

        optimizer.step()
        if args.use_onecycle:
            lr_scheduler.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        cur_lr = next(iter(optimizer.param_groups))['lr']

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'LR {cur_lr:.4f}\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      data_time=data_time, loss=losses, cur_lr=cur_lr, top1=top1, top5=top5))
    return {'train_loss': losses, 'train_top1': top1, 'train_top5': top5}


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def validate(val_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            if args.gpu is not None:
                input = input.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))
            top5.update(prec5.item(), input.size(0))

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

    return top1.avg, top5.avg, losses.avg


def save_checkpoint(state, is_best, checkpoint, filename='finetuned.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(os.path.join(filepath, 'finetuned.pth.tar'),
                        os.path.join(filepath, 'finetuned_best.pth.tar'))


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
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


if __name__ == '__main__':
    main()
