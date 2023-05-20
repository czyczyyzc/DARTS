from __future__ import print_function, absolute_import
from utils import to_torch


def accuracy(output, target, topk=(1,)):
    output, target = to_torch(output), to_torch(target)
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, dim=1)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    ret = []
    for k in topk:
        correct_k = correct[:k].contiguous().view(-1).float().sum(0)
        ret.append(correct_k.mul_(100.0 / batch_size))
    return ret
