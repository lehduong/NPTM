import torch
import torch.nn as nn
from timm.utils import ApexScaler

try:
    from apex import amp
    has_apex = True
except ImportError:
    amp = None
    has_apex = False


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


def get_conv_zero_param(model):
    total = 0
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            total += torch.sum(m.weight.data.eq(0))
    return total


def get_zero_param(model):
    total = 0
    for m in model.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            total += torch.sum(m.weight.data.eq(0))
    return total


class PruningApexScaler(ApexScaler):
    def __call__(self, loss, optimizer, model):
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
        for k, m in enumerate(model.modules()):
            if isinstance(m, nn.Conv2d):
                weight_copy = m.weight.data.abs().clone()
                mask = weight_copy.gt(0).float().cuda()
                m.weight.grad.data.mul_(mask)
        optimizer.step()
