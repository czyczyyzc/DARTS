from __future__ import print_function, absolute_import

from bisect import bisect_right


class LRScheduler(object):
    def __init__(self, optimizer, lr, num_epochs, step_size, gamma=0.1):
        self.optimizer  = optimizer
        self.lr         = lr
        self.num_epochs = num_epochs
        self.step_size  = step_size
        self.gamma      = gamma

    def step(self, epoch):
        if len(self.num_epochs) == 1 and self.step_size != -1:
            lr = self.lr * (self.gamma ** (epoch // self.step_size))
        else:
            lr = self.lr * (self.gamma ** bisect_right(self.num_epochs, epoch))
        for g in self.optimizer.param_groups:
            lr_g = lr * g.get('lr_mult', 1)
            if g['lr'] != lr_g:
                print("At epoch{}, the learning rate is changed to {}".format(epoch, lr_g))
            g['lr'] = lr_g
        return
