from __future__ import print_function, absolute_import

import os
import sys
import random
import argparse
import torch
import torch.optim
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch.utils.data.distributed
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import numpy as np
from modeling import models
from data import datasets, transforms
from data.samplers import SubsetDistributedSampler
from engine.trainer import Trainer
from engine.evaluator import Evaluator
from utils.meters import count_parameters_in_MB
from utils.logging import Logger
from utils.serialization import load_checkpoint, save_checkpoint


def argument_parser():
    parser = argparse.ArgumentParser(description='NAS with Pytorch Implementation')
    parser.add_argument('--gpu-ids', type=str, default='0')
    # data
    parser.add_argument('-d', '--dataset', type=str, default='cifar10', choices=datasets.names())
    parser.add_argument('-j', '--num-workers', type=int, default=1)
    parser.add_argument('-b', '--batch-size', type=int, default=64)
    parser.add_argument('--train-portion', type=float, default=0.5)
    parser.add_argument('--num-epochs', type=int, default=50)
    # model
    parser.add_argument('-a', '--arch', type=str, default='cnn', choices=models.names())
    parser.add_argument('--init-channels', type=int, default=16, help="number of initial channels")
    parser.add_argument('--layers', type=int, default=8, help="total number of layers")
    parser.add_argument('--unrolled', action='store_true', default=False, help="use one-step unrolled validation loss")
    # optimizer
    parser.add_argument('--netw-lr', type=float, default=0.025, help="initial learning rate")
    parser.add_argument('--netw-lr-min', type=float, default=0.001, help="minimum learning rate")
    parser.add_argument('--netw-momentum', type=float, default=0.9)
    parser.add_argument('--netw-weight-decay', type=float, default=3e-4)
    parser.add_argument('--arch-lr', type=float, default=3e-4)
    parser.add_argument('--arch-weight-decay', type=float, default=1e-3)
    # training
    parser.add_argument('--seed', type=int, default=2)
    parser.add_argument('--resume', action='store_true', default=False, help='resume from checkpoint')
    parser.add_argument('--evaluate', action='store_true', default=False, help="evaluation only")
    parser.add_argument('--print-freq', type=int, default=20)
    # misc
    working_dir = os.path.dirname(os.path.abspath(__file__))
    parser.add_argument('--data-dir', type=str, metavar='PATH', default=os.path.join(working_dir, 'temp', 'data'))
    parser.add_argument('--logs-dir', type=str, metavar='PATH', default=os.path.join(working_dir, 'temp', 'logs'))
    # distributed
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--net-card', type=str, default='', help="Name of the network card.")
    return parser


def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
    if args.net_card:
        os.environ['GLOO_SOCKET_IFNAME'] = args.net_card
        os.environ['NCCL_SOCKET_IFNAME'] = args.net_card
    args.gpu_ids = list(map(int, args.gpu_ids.split(',')))

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    cudnn.enabled = True
    cudnn.benchmark = True
    # cudnn.deterministic = True

    # Redirect print to both console and log file
    sys.stdout = Logger(os.path.join(args.logs_dir, 'log.txt'))

    args.world_size  = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1
    args.batch_size  = args.batch_size // args.world_size
    args.distributed = args.world_size > 1
    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        dist.init_process_group(backend='nccl', init_method='env://', world_size=args.world_size)
        dist.barrier()

    # Create dataloaders
    train_transforms = transforms.create(args.dataset, train=True)
    test_transforms  = transforms.create(args.dataset, train=False)

    data_root = os.path.join(args.data_dir, args.dataset)
    train_dataset = datasets.create(args.dataset, data_root, train=True,  transform=train_transforms, download=True)
    test_dataset  = datasets.create(args.dataset, data_root, train=False, transform=test_transforms,  download=True)

    indices = list(range(len(train_dataset)))
    split = int(np.floor(args.train_portion * len(train_dataset)))
    if args.distributed:
        train_sampler = SubsetDistributedSampler(shuffle=True, indices=indices[:split])
        valid_sampler = SubsetDistributedSampler(shuffle=True, indices=indices[split:])
    else:
        train_sampler = torch.utils.data.sampler.SubsetRandomSampler(indices[:split])
        valid_sampler = torch.utils.data.sampler.SubsetRandomSampler(indices[split:])

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, sampler=train_sampler,
        num_workers=args.num_workers, pin_memory=True, drop_last=True)
    valid_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, sampler=valid_sampler,
        num_workers=args.num_workers, pin_memory=True, drop_last=True)
    test_loader  = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True)

    # Create model
    model = models.create(args.arch, num_classes=len(train_dataset.classes),
                          init_channels=args.init_channels, layers=args.layers)
    if args.distributed:
        model = nn.parallel.DistributedDataParallel(
            model.cuda(), device_ids=[args.local_rank], output_device=args.local_rank)  # find_unused_parameters=True
    else:
        model = nn.DataParallel(model, device_ids=args.gpu_ids, output_device=args.gpu_ids[0]).cuda()

    if not args.distributed or args.local_rank == 0:
        print("param size {:f} MB".format(count_parameters_in_MB(model)))

    # Criterion
    # criterion = nn.CrossEntropyLoss().cuda()

    # Optimizer
    netw_optimizer = torch.optim.SGD(model.module.netw_parameters(), lr=args.netw_lr, momentum=args.netw_momentum,
                                     weight_decay=args.netw_weight_decay)
    netw_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(netw_optimizer, T_max=args.num_epochs, eta_min=args.netw_lr_min)
    arch_optimizer = torch.optim.Adam(model.module.arch_parameters(), lr=args.arch_lr, betas=(0.5, 0.999),
                                      weight_decay=args.arch_weight_decay)

    # Load from checkpoint
    start_epoch = best_prec1 = 0
    if args.resume:
        checkpoint = load_checkpoint(os.path.join(args.logs_dir, 'checkpoint.pth.tar'))
        model.module.load_state_dict(checkpoint['state_dict'])
        netw_optimizer.load_state_dict(checkpoint['netw_optimizer'])
        netw_scheduler.load_state_dict(checkpoint['netw_scheduler'])
        arch_optimizer.load_state_dict(checkpoint['arch_optimizer'])
        start_epoch = checkpoint['epoch']
        best_prec1  = checkpoint['best_prec1']
        print("=> Start epoch {}  best_prec1 {:.2f}".format(start_epoch, best_prec1))
    if args.distributed:
        dist.barrier()

    # Create Evaluator
    evaluator = Evaluator(model, args.print_freq, args.distributed)
    if args.evaluate:
        evaluator.evaluate(test_loader)
        return

    # Create Trainer
    trainer = Trainer(model, netw_optimizer, arch_optimizer, args.unrolled, args.print_freq, args.distributed)

    # Start training
    for epoch in range(start_epoch, args.num_epochs):
        # Use .set_epoch() method to reshuffle the dataset partition at every iteration
        if args.distributed:
            train_sampler.set_epoch(epoch)
            valid_sampler.set_epoch(epoch)

        trainer.train(train_loader, valid_loader, epoch)
        netw_scheduler.step()

        # evaluate on validation set
        # prec1 = evaluator.evaluate(test_loader)
        # is_best = prec1 > best_prec1
        # best_prec1 = max(prec1, best_prec1)
        is_best = True
        if not args.distributed or dist.get_rank() == 0:

            lr = netw_scheduler.get_lr()[0]
            print('epoch {:d}, lr {:.6f}'.format(epoch, lr))
            genotype = model.module.genotype()
            print(genotype)
            print(F.softmax(model.module.alphas_normal, dim=-1))
            print(F.softmax(model.module.alphas_reduce, dim=-1))

            save_checkpoint({
                'state_dict': model.module.state_dict(),
                'netw_optimizer': netw_optimizer.state_dict(),
                'netw_scheduler': netw_scheduler.state_dict(),
                'arch_optimizer': arch_optimizer.state_dict(),
                'epoch': epoch + 1,
                'best_prec1': best_prec1,
            }, is_best, fpath=os.path.join(args.logs_dir, 'checkpoint.pth.tar'))

        if args.distributed:
            dist.barrier()

    # Final test
    checkpoint = load_checkpoint(os.path.join(args.logs_dir, 'checkpoint.pth.tar'))
    model.module.load_state_dict(checkpoint['state_dict'])
    evaluator.evaluate(test_loader)
    return


if __name__ == '__main__':
    parser = argument_parser()
    main(parser.parse_args())
