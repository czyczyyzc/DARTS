from __future__ import print_function, absolute_import

import time
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from utils import Bar
from utils.meters import AverageMeter
from modeling.metrics.classification import accuracy


def _concat(xs):
    return torch.cat([x.view(-1) for x in xs])


class Trainer(object):
    def __init__(self, model, netw_optimizer, arch_optimizer, unrolled=False, distributed=False):
        super(Trainer, self).__init__()
        self.model          = model
        self.netw_optimizer = netw_optimizer
        self.arch_optimizer = arch_optimizer
        self.unrolled       = unrolled
        self.distributed    = distributed

    def _backup(self):
        backup = [], []
        for v in self.model.module.netw_parameters():
            backup[0].append(copy.deepcopy(v.data))
            try:
                backup[1].append(copy.deepcopy(self.netw_optimizer.state[v]['momentum_buffer']))
            except KeyError:
                backup[1].append(torch.zeros_like(v.data))
        return backup

    def _restore(self, backup):
        for v in self.model.module.netw_parameters():
            v.data.copy_(backup[0].pop(0))
            if len(backup) > 1:
                self.netw_optimizer.state[v]['momentum_buffer'].copy_(backup[1].pop(0))

    def _backward_step_unrolled(self, data_train, target_train, data_valid, target_valid, r=1e-2):
        backup = self._backup()
        loss = F.cross_entropy(self.model(data_train), target_train)
        self.netw_optimizer.zero_grad()
        loss.backward()
        self.netw_optimizer.step()

        loss = F.cross_entropy(self.model(data_valid), target_valid)
        self.netw_optimizer.zero_grad()
        self.arch_optimizer.zero_grad()
        loss.backward()

        dalpha = [v.grad.data for v in self.model.module.arch_parameters()]
        dparam = [v.grad.data for v in self.model.module.netw_parameters()]

        R = r / _concat(dparam).norm()
        params_p, params_n = [], []
        for p, v in zip(backup[0], dparam):
            params_p.append(p.add(v, alpha=R))
            params_n.append(p.sub(v, alpha=R))

        self._restore((params_p,))
        loss = F.cross_entropy(self.model(data_train), target_train)
        self.arch_optimizer.zero_grad()
        loss.backward()
        grads_p = [v.grad.data for v in self.model.module.arch_parameters()]

        self._restore((params_n,))
        loss = F.cross_entropy(self.model(data_train), target_train)
        self.arch_optimizer.zero_grad()
        loss.backward()
        grads_n = [v.grad.data for v in self.model.module.arch_parameters()]
        grads_a = [(x - y).div_(2 * R) for x, y in zip(grads_p, grads_n)]

        lr = self.netw_optimizer.param_groups[0]['lr']
        for g, v in zip(dalpha, grads_a):
            g.sub_(v, alpha=lr)
        for v, g in zip(self.model.module.arch_parameters(), dalpha):
            v.grad.data.copy_(g)
        self.arch_optimizer.step()
        self._restore(backup)

    def _backward_step(self, data_valid, target_valid):
        logits = self.model(data_valid)
        loss = F.cross_entropy(logits, target_valid)
        self.arch_optimizer.zero_grad()
        loss.backward()
        self.arch_optimizer.step()

    def _step(self, data_train, target_train, data_valid, target_valid):
        if self.unrolled:
            self._backward_step_unrolled(data_train, target_train, data_valid, target_valid)
        else:
            self._backward_step(data_valid, target_valid)

    def train(self, train_loader, valid_loader, epoch):
        self.model.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        loss_avg = AverageMeter()
        top1_avg = AverageMeter()
        top5_avg = AverageMeter()
        end = time.time()

        bar = Bar('Processing', max=len(train_loader)) if not self.distributed or dist.get_rank() == 0 else None

        for i, (data_train, target_train) in enumerate(train_loader):
            data_time.update(time.time() - end)

            data_train, target_train = data_train.cuda(), target_train.cuda()
            data_valid, target_valid = next(iter(valid_loader))
            data_valid, target_valid = data_valid.cuda(), target_valid.cuda()

            self._step(data_train, target_train, data_valid, target_valid)

            logits = self.model(data_train)
            loss = F.cross_entropy(logits, target_train)

            self.netw_optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.module.netw_parameters(), 5.0)
            self.netw_optimizer.step()

            with torch.no_grad():
                prec1, prec5 = accuracy(logits, target_train, topk=(1, 5))
                if self.distributed:
                    temp = torch.stack([loss, prec1, prec5])
                    dist.reduce(temp, dst=0)
                    if dist.get_rank() == 0:
                        temp /= dist.get_world_size()
                    loss, prec1, prec5 = temp
                loss_avg.update(loss,  target_train.size(0))
                top1_avg.update(prec1, target_train.size(0))
                top5_avg.update(prec5, target_train.size(0))

            batch_time.update(time.time() - end)
            end = time.time()
            if bar is not None:
                bar.suffix = "Epoch: [{N_epoch}][{N_batch}/{N_size}] | Time {N_bta:.3f} | " \
                             "Loss {N_lossa:.3f} | Prec1 {N_preca1:.2f} | Prec5 {N_preca5:.2f}".format(
                    N_epoch=epoch, N_batch=i+1, N_size=len(train_loader), N_bta=batch_time.avg,
                    N_lossa=loss_avg.avg, N_preca1=top1_avg.avg, N_preca5=top5_avg.avg
                )
                bar.next()
        if bar is not None:
            bar.finish()
        return
