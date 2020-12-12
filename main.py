from __future__ import print_function, absolute_import

import os
import sys
import pickle
import random
import argparse
import torch
import torch.optim
import torch.nn as nn
import torch.utils.data
import torch.utils.data.distributed
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import numpy as np
from modeling import models
from modeling.models.genotypes import DARTS
from data import datasets, transforms
from engine.trainer import Trainer
from engine.evaluator import Evaluator
from utils.logging import Logger
from utils.serialization import load_checkpoint, save_checkpoint


def argument_parser():
    parser = argparse.ArgumentParser(description='NAS with Pytorch Implementation')
    parser.add_argument('--gpu-ids', type=str, default='0')
    # data
    parser.add_argument('-d', '--dataset', type=str, default='cifar10', choices=datasets.names())
    parser.add_argument('-j', '--num-workers', type=int, default=4)
    parser.add_argument('-b', '--batch-size', type=int, default=96)
    parser.add_argument('--num-epochs', type=int, default=600)
    parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
    parser.add_argument('--label_smooth', type=float, default=0.1, help='label smoothing')
    # model
    parser.add_argument('-a', '--arch', type=str, default='cnn_cifar', choices=models.names())
    parser.add_argument('--init-channels', type=int, default=36, help="number of initial channels")
    parser.add_argument('--layers', type=int, default=20, help="total number of layers")
    parser.add_argument('--auxiliary', action='store_true', default=False, help='use auxiliary tower')
    parser.add_argument('--auxiliary-weight', type=float, default=0.4, help='weight for auxiliary loss')
    parser.add_argument('--drop-prob', type=float, default=0.2, help='drop path probability')
    # optimizer
    parser.add_argument('--lr', type=float, default=0.025, help="initial learning rate")
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight-decay', type=float, default=3e-4)
    parser.add_argument('--gamma', type=float, default=0.97, help='learning rate decay')
    parser.add_argument('--step-size', type=int, default=1, help='epochs between two learning rate decays')
    # training
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--resume', action='store_true', default=False, help='resume from checkpoint')
    parser.add_argument('--evaluate', action='store_true', default=False, help="evaluation only")
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
    train_transforms = transforms.create(args.dataset, train=True, cutout=args.cutout)
    test_transforms  = transforms.create(args.dataset, train=False)

    data_root = os.path.join(args.data_dir, args.dataset)
    train_dataset = datasets.create(args.dataset, data_root, train=True,  transform=train_transforms, download=True)
    test_dataset  = datasets.create(args.dataset, data_root, train=False, transform=test_transforms,  download=True)

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True)
    else:
        train_sampler = torch.utils.data.sampler.RandomSampler(train_dataset)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, sampler=train_sampler, num_workers=args.num_workers,
        pin_memory=True, drop_last=True)
    test_loader  = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True)

    # Create model
    # with open(os.path.join(args.logs_dir, 'genotype.pkl'), 'rb') as file:
    #     genotype = pickle.loads(file.read())
    #     if not args.distributed or dist.get_rank() == 0:
    #         print(genotype)
    genotype = DARTS
    print(genotype)
    norm_layer = nn.SyncBatchNorm if args.distributed else nn.BatchNorm2d
    model = models.create(args.arch, num_classes=len(train_dataset.classes), init_channels=args.init_channels,
                          layers=args.layers, genotype=genotype, auxiliary=args.auxiliary, drop_prob=args.drop_prob,
                          norm_layer=norm_layer)
    if args.distributed:
        model = nn.parallel.DistributedDataParallel(
            model.cuda(), device_ids=[args.local_rank], output_device=args.local_rank)  # find_unused_parameters=True
    else:
        model = nn.DataParallel(model).cuda()

    if not args.distributed or args.local_rank == 0:
        n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print('number of params:', n_parameters)
    # Criterion
    # criterion = nn.CrossEntropyLoss().cuda()

    # Optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

    # Load from checkpoint
    start_epoch = best_prec1 = 0
    if args.resume:
        checkpoint = load_checkpoint(os.path.join(args.logs_dir, 'checkpoint.pth.tar'))
        model.module.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        start_epoch = checkpoint['epoch']
        best_prec1  = checkpoint['best_prec1']
        print("=> Start epoch {}  best_prec1 {:.2f}".format(start_epoch, best_prec1))
    if args.distributed:
        dist.barrier()

    # Create Evaluator
    evaluator = Evaluator(model, args.distributed)
    if args.evaluate:
        evaluator.evaluate(test_loader)
        return

    # Create Trainer
    trainer = Trainer(model, optimizer, args.auxiliary_weight, args.distributed)

    # Start training
    for epoch in range(start_epoch, args.num_epochs):
        # Use .set_epoch() method to reshuffle the dataset partition at every iteration
        if args.distributed:
            train_sampler.set_epoch(epoch)

        model.drop_prob = args.drop_prob * epoch / args.num_epochs

        trainer.train(train_loader, epoch)
        scheduler.step()

        # evaluate on validation set
        # prec1 = evaluator.evaluate(test_loader)
        # is_best = prec1 > best_prec1
        # best_prec1 = max(prec1, best_prec1)
        is_best = True
        if not args.distributed or args.local_rank == 0:
            lr = scheduler.get_lr()[0]
            print('epoch {:d}, lr {:.6f}'.format(epoch, lr))
            save_checkpoint({
                'state_dict': model.module.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
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
