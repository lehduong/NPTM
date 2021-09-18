from __future__ import division

import os
import sys
import shutil
import time
import random
import argparse
import torch
import torch.backends.cudnn as cudnn
import torchvision.datasets as dset
import torchvision.transforms as transforms
from utils import AverageMeter, RecorderMeter, time_string, convert_secs2time, get_train_valid_loader, get_test_loader
import models
import numpy as np
import wandb


model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='Trains ResNeXt on CIFAR or ImageNet',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('data_path', type=str, help='Path to dataset')
parser.add_argument('--dataset', type=str, choices=['cifar10', 'cifar100',
                                                    'imagenet', 'svhn', 'stl10'], help='Choose between Cifar10/100 and ImageNet.')
parser.add_argument('--arch', metavar='ARCH', default='resnet18', choices=model_names,
                    help='model architecture: ' + ' | '.join(model_names) + ' (default: resnext29_8_64)')
# Optimization options
parser.add_argument('--epochs', type=int, default=300,
                    help='Number of epochs to train.')
parser.add_argument('--finetune_epochs', type=int, default=50,
                    help='Number of epochs to retrain.')
parser.add_argument('--batch_size', type=int, default=128, help='Batch size.')
parser.add_argument('--learning_rate', type=float,
                    default=0.1, help='The Learning Rate.')
parser.add_argument('--momentum', type=float, default=0.9, help='Momentum.')
parser.add_argument('--decay', type=float, default=0.0005,
                    help='Weight decay (L2 penalty).')
parser.add_argument('--schedule', type=int, nargs='+',
                    default=[150, 225], help='Decrease learning rate at these epochs.')
parser.add_argument('--gammas', type=float, nargs='+', default=[
                    0.1, 0.1], help='LR is multiplied by gamma on schedule, number of gammas should be equal to schedule')
# Checkpoints
parser.add_argument('--print_freq', default=200, type=int,
                    metavar='N', help='print frequency (default: 200)')
parser.add_argument('--save_path', type=str, default='./',
                    help='Folder to save checkpoints and log.')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--start_epoch', default=0, type=int,
                    metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('--evaluate', dest='evaluate',
                    action='store_true', help='evaluate model on validation set')
# Acceleration
parser.add_argument('--ngpu', type=int, default=1, help='0 = CPU.')
parser.add_argument('--workers', type=int, default=2,
                    help='number of data loading workers (default: 2)')
# random seed
parser.add_argument('--manualSeed', type=int, help='manual seed')
# compress rate
parser.add_argument('--rate', type=float, default=0.9,
                    help='compress rate of model')
parser.add_argument('--layer_begin', type=int, default=1,
                    help='compress layer of model')
parser.add_argument('--layer_end', type=int, default=1,
                    help='compress layer of model')
parser.add_argument('--layer_inter', type=int, default=1,
                    help='compress layer of model')
parser.add_argument('--epoch_prune', type=int, default=1,
                    help='compress layer of model')
parser.add_argument('--use_state_dict', dest='use_state_dict',
                    action='store_true', help='use state dcit or not')
parser.add_argument('--wandb_name', default='resnet_56_standard', type=str,
                    help='name of wandb run')

parser.add_argument('--finetune-schedule', type=int, nargs='+', default=[18, 32],
                    help='Decrease learning rate at these epochs.')
parser.add_argument('--init_lr', type=float, default=0.0008,
                    help='initialized learning rate when doing warm up (default: 0.008)')
parser.add_argument('--gamma', type=float, default=0.1,
                    help='gamma of step learning rate decay (default: 0.1)')
parser.add_argument('--num_warmup', type=int, default=4,
                    help='number of epochs to increase learning rate (default: 4)')
parser.add_argument('--warmup_strategy', type=str, default='cos',
                    help='warmup strategy (default: cos)')

parser.add_argument(
    '--div_factor',
    type=float,
    default=125,
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
parser.add_argument('--use_onecycle', dest='use_onecycle', action='store_true')
parser.add_argument('--random_mask', dest='random_mask', action='store_true')
parser.add_argument('--finetune', dest='finetune', action='store_true')
parser.set_defaults(use_llr=False)
parser.set_defaults(use_onecycle=False)
parser.set_defaults(random_mask=False)
parser.set_defaults(finetune=False)

args = parser.parse_args()
args.use_cuda = args.ngpu > 0 and torch.cuda.is_available()

if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)
if args.use_cuda:
    torch.cuda.manual_seed_all(args.manualSeed)
cudnn.benchmark = True

wandb.init(
    name=args.wandb_name,
    project='Soft_Filters_Pruning',
    config={
            **vars(args)}
)


def main():
    # Init logger
    if not os.path.isdir(args.save_path):
        os.makedirs(args.save_path)
    log = open(os.path.join(args.save_path,
                            'log_seed_{}.txt'.format(args.manualSeed)), 'w')
    print_log('save path : {}'.format(args.save_path), log)
    state = {k: v for k, v in args._get_kwargs()}
    print_log(state, log)
    print_log("Random Seed: {}".format(args.manualSeed), log)
    print_log("python version : {}".format(
        sys.version.replace('\n', ' ')), log)
    print_log("torch  version : {}".format(torch.__version__), log)
    print_log("cudnn  version : {}".format(
        torch.backends.cudnn.version()), log)
    print_log("Compress Rate: {}".format(args.rate), log)
    print_log("Layer Begin: {}".format(args.layer_begin), log)
    print_log("Layer End: {}".format(args.layer_end), log)
    print_log("Layer Inter: {}".format(args.layer_inter), log)
    print_log("Epoch prune: {}".format(args.epoch_prune), log)
    # Init dataset
    if not os.path.isdir(args.data_path):
        os.makedirs(args.data_path)

    if args.dataset == 'cifar10':
        mean = [x / 255 for x in [125.3, 123.0, 113.9]]
        std = [x / 255 for x in [63.0, 62.1, 66.7]]
    elif args.dataset == 'cifar100':
        mean = [x / 255 for x in [129.3, 124.1, 112.4]]
        std = [x / 255 for x in [68.2, 65.4, 70.4]]
    else:
        assert False, "Unknow dataset : {}".format(args.dataset)

    if args.dataset == 'cifar10':
        num_classes = 10
    elif args.dataset == 'cifar100':
        num_classes = 100
    else:
        assert False, 'Do not support dataset : {}'.format(args.dataset)

    train_loader, val_loader = get_train_valid_loader(args.data_path,
                                                      batch_size=args.batch_size,
                                                      random_seed=args.manualSeed,
                                                      num_workers=args.workers,
                                                      valid_size=0.1,
                                                      pin_memory=True,
                                                      dataset=args.dataset
                                                      )
    test_loader = get_test_loader(args.data_path,
                                  batch_size=args.batch_size,
                                  shuffle=False,
                                  num_workers=args.workers,
                                  pin_memory=True,
                                  dataset=args.dataset
                                  )

    print_log("=> creating model '{}'".format(args.arch), log)
    # Init model, criterion, and optimizer
    net = models.__dict__[args.arch](num_classes)
    print_log("=> network :\n {}".format(net), log)

    net = torch.nn.DataParallel(net, device_ids=list(range(args.ngpu)))

    # define loss function (criterion) and optimizer
    criterion = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(net.parameters(), state['learning_rate'], momentum=state['momentum'],
                                weight_decay=state['decay'], nesterov=True)
    lr_scheduler = None

    if args.use_cuda:
        net.cuda()
        criterion.cuda()

    recorder = RecorderMeter(args.epochs+args.finetune_epochs)
    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print_log("=> loading checkpoint '{}'".format(args.resume), log)
            checkpoint = torch.load(args.resume)
            recorder = checkpoint['recorder']
            args.start_epoch = checkpoint['epoch']
            if args.use_state_dict:
                net.load_state_dict(checkpoint['state_dict'])
            else:
                net = checkpoint['state_dict']

            optimizer.load_state_dict(checkpoint['optimizer'])
            print_log("=> loaded checkpoint '{}' (epoch {})" .format(
                args.resume, checkpoint['epoch']), log)
        else:
            print_log("=> no checkpoint found at '{}'".format(args.resume), log)
    else:
        print_log(
            "=> do not use any checkpoint for {} model".format(args.arch), log)

    if args.finetune:
        args.start_epoch = args.epochs
        recorder.reset(args.finetune_epochs)

    if args.evaluate:
        time1 = time.time()
        test(test_loader, net, criterion, log)
        time2 = time.time()
        print('function took %0.3f ms' % ((time2-time1)*1000.0))
        return

    m = Mask(net, args.random_mask)

    m.init_length()

    comp_rate = args.rate
    print("-"*10+"one epoch begin"+"-"*10)
    print("the compression rate now is %f" % comp_rate)

    val_acc_1,   val_los_1 = validate(val_loader, net, criterion, log)

    print(" accu before is: %.3f %%" % val_acc_1)

    m.model = net
    m.init_mask(comp_rate)
    m.do_mask()
    net = m.model

    if args.use_cuda:
        net = net.cuda()
    val_acc,   val_loss = validate(val_loader, net, criterion, log)
    print(" accu after is: %s %%" % val_acc)

    # Main loop
    start_time = time.time()
    epoch_time = AverageMeter()

    for epoch in range(args.start_epoch, args.epochs):
        current_learning_rate = adjust_learning_rate(
            optimizer, epoch, args.gammas, args.schedule)

        need_hour, need_mins, need_secs = convert_secs2time(
            epoch_time.avg * (args.epochs-epoch))
        need_time = '[Need: {:02d}:{:02d}:{:02d}]'.format(
            need_hour, need_mins, need_secs)

        print_log('\n==>>{:s} [Epoch={:03d}/{:03d}] {:s} [learning_rate={:6.4f}]'.format(time_string(), epoch, args.epochs, need_time, current_learning_rate)
                  + ' [Best : Accuracy={:.2f}, Error={:.2f}]'.format(recorder.max_accuracy(False), 100-recorder.max_accuracy(False)), log)

        # train for one epoch
        train_acc, train_los = train(
            train_loader, net, criterion, optimizer, epoch, log, lr_scheduler)

        # evaluate on validation set
        val_acc_1,   val_los_1 = validate(val_loader, net, criterion, log)
        if (epoch % args.epoch_prune == 0 or epoch == args.epochs-1):
            m.model = net
            m.if_zero()
            m.init_mask(comp_rate)
            m.do_mask()
            m.if_zero()
            net = m.model
            if args.use_cuda:
                net = net.cuda()

        val_acc_2,   val_loss_2 = validate(val_loader, net, criterion, log)
        test_acc,   test_loss = test(test_loader, net, criterion, log)

        is_best = recorder.update(
            epoch, train_los, train_acc, val_loss_2, val_acc_2)

        wandb.log({'top1': val_acc_2, 'best_top1': recorder.max_accuracy(
            False), 'lr': current_learning_rate, 'test_top1': test_acc})

        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': net,
            'recorder': recorder,
            'optimizer': optimizer.state_dict(),
        }, is_best, args.save_path, 'checkpoint.pth.tar')

        # measure elapsed time
        epoch_time.update(time.time() - start_time)
        start_time = time.time()
        # recorder.plot_curve( os.path.join(args.save_path, 'curve.png') )

    ###############################################################################
    # Fine-tuning
    if args.finetune_epochs > 0:
        print("-"*10+"start fine-tuning now"+"-"*10)
        print("the number of finetuning epochs is %f" % args.finetune_epochs)
        start_time = time.time()
        epoch_time = AverageMeter()
        recorder.reset(args.finetune_epochs)

        # load best val model
        checkpoint = torch.load(os.path.join(
            args.save_path, 'model_best.pth.tar'))
        if args.use_state_dict:
            net.load_state_dict(checkpoint['state_dict'])
        else:
            net = checkpoint['state_dict']

        # save new mask to Mask objet
        val_acc_1,   val_los_1 = validate(val_loader, net, criterion, log)
        print(" accu of finetune model before is: %.3f %%" % val_acc_1)
        m.model = net
        m.init_mask(comp_rate)
        m.do_mask()
        net = m.model
        if args.use_cuda:
            net = net.cuda()
        val_acc,   val_loss = validate(val_loader, net, criterion, log)
        print(" accu of finetune model after is: %s %%" % val_acc)

        # create a new optimizer to reset all momentum
        # since standard retrain usually ignore statedict of optimizer of original training
        # also foster network to converge to new optima
        # TODO: Fixed hardcode hyperparam in scheduler and optimizer
        optimizer = torch.optim.SGD(net.parameters(), 0.0008, momentum=state['momentum'],
                                    weight_decay=state['decay'], nesterov=True)

        if args.use_onecycle:
            lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.1, div_factor=args.div_factor,
                                                               epochs=args.finetune_epochs, steps_per_epoch=len(train_loader), pct_start=args.pct_start,
                                                               final_div_factor=args.final_div_factor)

        for epoch in range(0, args.finetune_epochs):
            need_hour, need_mins, need_secs = convert_secs2time(
                epoch_time.avg * (args.finetune_epochs))
            need_time = '[Need: {:02d}:{:02d}:{:02d}]'.format(
                need_hour, need_mins, need_secs)

            current_learning_rate = optimizer.param_groups[0]['lr']
            print_log('\n==>>{:s} [Epoch={:03d}/{:03d}] {:s} [learning_rate={:6.4f}]'.format(time_string(), epoch, args.finetune_epochs, need_time, current_learning_rate)
                      + ' [Best : Accuracy={:.2f}, Error={:.2f}]'.format(recorder.max_accuracy(False), 100-recorder.max_accuracy(False)), log)

            # train for one epoch
            train_acc, train_loss = train(
                train_loader, net, criterion, optimizer, epoch, log, lr_scheduler)

            # make sure to not update weights equal 0 i.e. keep the mask during fine-tuning
            m.model = net
            m.if_zero()
            m.do_mask()
            m.if_zero()
            net = m.model
            if args.use_cuda:
                net = net.cuda()

            # evaluate on validation set
            val_acc, val_loss = validate(val_loader, net, criterion, log)
            test_acc, test_loss = test(test_loader, net, criterion, log)

            is_best = recorder.update(
                epoch, train_los, train_acc, val_loss, val_acc)

            wandb.log({'top1': val_acc, 'best_top1': recorder.max_accuracy(False),
                       'lr': current_learning_rate, 'test_top1': test_acc, 'train_loss': train_loss, 'val_loss': val_loss, 'test_loss': test_loss})

            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': net,
                'recorder': recorder,
                'optimizer': optimizer.state_dict(),
            }, is_best, args.save_path, 'finetuned.pth.tar')

            # measure elapsed time
            epoch_time.update(time.time() - start_time)
            start_time = time.time()
            # recorder.plot_curve( os.path.join(args.save_path, 'curve.png') )

    log.close()
    wandb.save(os.path.join(args.save_path, '*'))

# train function (forward, backward, update)


def train(train_loader, model, criterion, optimizer, epoch, log, lr_scheduler):
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

        if args.use_cuda:
            target = target.cuda()
            input = input.cuda()
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.data.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if lr_scheduler:
            lr_scheduler.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print_log('  Epoch: [{:03d}][{:03d}/{:03d}]   '
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})   '
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})   '
                      'Loss {loss.val:.4f} ({loss.avg:.4f})   '
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})   '
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})   '.format(
                          epoch, i, len(train_loader), batch_time=batch_time,
                          data_time=data_time, loss=losses, top1=top1, top5=top5) + time_string(), log)
    print_log('  **Train** Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Error@1 {error1:.3f}'.format(
        top1=top1, top5=top5, error1=100-top1.avg), log)
    return top1.avg, losses.avg


def validate(val_loader, model, criterion, log):
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            if args.use_cuda:
                target = target.cuda()
                input = input.cuda()
            input_var = torch.autograd.Variable(input)
            target_var = torch.autograd.Variable(target)

            # compute output
            output = model(input_var)
            loss = criterion(output, target_var)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
            losses.update(loss.data.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))
            top5.update(prec5.item(), input.size(0))

    print_log('  **Validate** Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Error@1 {error1:.3f}'.format(
        top1=top1, top5=top5, error1=100-top1.avg), log)

    return top1.avg, losses.avg


def test(test_loader, model, criterion, log):
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        for i, (input, target) in enumerate(test_loader):
            if args.use_cuda:
                target = target.cuda()
                input = input.cuda()
            input_var = torch.autograd.Variable(input)
            target_var = torch.autograd.Variable(target)

            # compute output
            output = model(input_var)
            loss = criterion(output, target_var)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
            losses.update(loss.data.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))
            top5.update(prec5.item(), input.size(0))

    print_log('  **Test** Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Error@1 {error1:.3f}'.format(
        top1=top1, top5=top5, error1=100-top1.avg), log)

    return top1.avg, losses.avg


def print_log(print_string, log):
    print("{}".format(print_string))
    log.write('{}\n'.format(print_string))
    log.flush()


def save_checkpoint(state, is_best, save_path, filename):
    filename = os.path.join(save_path, filename)
    torch.save(state, filename)
    if is_best:
        if filename.find('finetuned') > -1:
            bestname = os.path.join(save_path, 'model_best_finetune.pth.tar')
            shutil.copyfile(filename, bestname)
        else:
            bestname = os.path.join(save_path, 'model_best.pth.tar')
            shutil.copyfile(filename, bestname)


def adjust_learning_rate(optimizer, epoch, gammas, schedule):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.learning_rate
    assert len(gammas) == len(
        schedule), "length of gammas and schedule should be equal"
    for (gamma, step) in zip(gammas, schedule):
        if (epoch >= step):
            lr = lr * gamma
        else:
            break
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


class Mask:
    def __init__(self, model, random_mask=False):
        self.model_size = {}
        self.model_length = {}
        self.compress_rate = {}
        self.mat = {}
        self.model = model
        self.mask_index = []
        self.random_mask = random_mask

    def get_codebook(self, weight_torch, compress_rate, length):
        weight_vec = weight_torch.view(length)
        weight_np = weight_vec.cpu().numpy()

        weight_abs = np.abs(weight_np)
        weight_sort = np.sort(weight_abs)

        threshold = weight_sort[int(length * (1-compress_rate))]
        weight_np[weight_np <= -threshold] = 1
        weight_np[weight_np >= threshold] = 1
        weight_np[weight_np != 1] = 0

        print("codebook done")
        return weight_np

    def get_filter_codebook(self, weight_torch, compress_rate, length):
        """
            Return a 'codebook' determined if a filter of conv layer will be pruned or not
        """
        codebook = np.ones(length)
        if len(weight_torch.size()) == 4:
            filter_pruned_num = int(weight_torch.size()[0]*(1-compress_rate))
            weight_vec = weight_torch.view(weight_torch.size()[0], -1)
            norm2 = torch.norm(weight_vec, 2, 1)
            norm2_np = norm2.cpu().numpy()
            if self.random_mask:
                norm2_np = np.random.random_sample(norm2_np.shape)
            filter_index = norm2_np.argsort()[:filter_pruned_num]
#            norm1_sort = np.sort(norm1_np)
#            threshold = norm1_sort[int (weight_torch.size()[0] * (1-compress_rate) )]
            kernel_length = weight_torch.size(
            )[1] * weight_torch.size()[2] * weight_torch.size()[3]
            for x in range(0, len(filter_index)):
                codebook[filter_index[x] *
                         kernel_length: (filter_index[x]+1) * kernel_length] = 0

        else:
            pass
        return codebook

    def convert2tensor(self, x):
        x = torch.FloatTensor(x)
        return x

    def init_length(self):
        for index, item in enumerate(self.model.parameters()):
            self.model_size[index] = item.size()

        for index1 in self.model_size:
            for index2 in range(0, len(self.model_size[index1])):
                if index2 == 0:
                    self.model_length[index1] = self.model_size[index1][0]
                else:
                    self.model_length[index1] *= self.model_size[index1][index2]

    def init_rate(self, layer_rate):
        """
            Initialized compression rate and index of going-to-be pruned layers
        """
        for index, item in enumerate(self.model.parameters()):
            self.compress_rate[index] = 1
        for key in range(args.layer_begin, args.layer_end + 1, args.layer_inter):
            self.compress_rate[key] = layer_rate
        # different setting for  different architecture
        if args.arch == 'resnet20':
            last_index = 57
        elif args.arch == 'resnet32':
            last_index = 93
        elif args.arch == 'resnet56':
            last_index = 165
        elif args.arch == 'resnet110':
            last_index = 327
        self.mask_index = [x for x in range(0, last_index, 3)]
#        self.mask_index =  [x for x in range (0,330,3)]

    def init_mask(self, layer_rate):
        """
            Compute the mask for each layer (stored in self.mat)
        """
        self.init_rate(layer_rate)
        for index, item in enumerate(self.model.parameters()):
            if(index in self.mask_index):
                self.mat[index] = self.get_filter_codebook(
                    item.data, self.compress_rate[index], self.model_length[index])
                self.mat[index] = self.convert2tensor(self.mat[index])
                if args.use_cuda:
                    self.mat[index] = self.mat[index].cuda()
        print("mask Ready")

    def do_mask(self):
        """
            Zero out weights of network according to the mask
        """
        for index, item in enumerate(self.model.parameters()):
            if(index in self.mask_index):
                a = item.data.view(self.model_length[index])
                b = a * self.mat[index]
                item.data = b.view(self.model_size[index])
        print("mask Done")

    def if_zero(self):
        """
            Print number of nonzero weights of network
        """
        for index, item in enumerate(self.model.parameters()):
            #            if(index in self.mask_index):
            if(index == 0):
                a = item.data.view(self.model_length[index])
                b = a.cpu().numpy()

                print("number of nonzero weight is %d, zero is %d" %
                      (np.count_nonzero(b), len(b) - np.count_nonzero(b)))


if __name__ == '__main__':
    main()
