import sys
sys.path.append("../")
import argparse
import builtins
import math
import os
import random
import shutil
import time
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter
from lmdb_datasets import LMDBDataset
import torchvision.datasets as dset
from d2c import AutoEncoder
from d2c import builder
from d2c import loader
from d2c import utils

parser = argparse.ArgumentParser(description='AE + Contrastive Training')
parser.add_argument('--data', type=str, default='./data',
                        help='path to dataset')
parser.add_argument('-j', '--workers', default=32, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.03, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--schedule', default=[120, 160], nargs='*', type=int,
                    help='learning rate schedule (when to drop lr by 10x)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum of SGD solver')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dataset', default='cifar10', type=str,
                    help='dataset used for training')

parser.add_argument('--arch_instance', type=str, default='res_mbconv',
                        help='path to the architecture instance')
# KL annealing
parser.add_argument('--kl_anneal_portion', type=float, default=0.3,
                    help='The portions epochs that KL is annealed')
parser.add_argument('--kl_const_portion', type=float, default=0.0001,
                    help='The portions epochs that KL is constant at kl_const_coeff')
parser.add_argument('--kl_const_coeff', type=float, default=0.0001,
                    help='The constant value used for min KL coeff')
# Flow params
parser.add_argument('--num_nf', type=int, default=0,
                    help='The number of normalizing flow cells per groups. Set this to zero to disable flows.')
parser.add_argument('--num_x_bits', type=int, default=8,
                    help='The number of bits used for representing data for colored images.')
# latent variables
parser.add_argument('--num_latent_scales', type=int, default=1,
                    help='the number of latent scales')
parser.add_argument('--num_groups_per_scale', type=int, default=10,
                    help='number of groups of latent variables per scale')
parser.add_argument('--num_latent_per_group', type=int, default=20,
                    help='number of channels in latent variables per group')
parser.add_argument('--ada_groups', action='store_true', default=False,
                    help='Settings this to true will set different number of groups per scale.')
parser.add_argument('--min_groups_per_scale', type=int, default=1,
                    help='the minimum number of groups per scale.')
# encoder parameters
parser.add_argument('--num_channels_enc', type=int, default=32,
                    help='number of channels in encoder')
parser.add_argument('--num_preprocess_blocks', type=int, default=2,
                    help='number of preprocessing blocks')
parser.add_argument('--num_preprocess_cells', type=int, default=3,
                    help='number of cells per block')
parser.add_argument('--num_cell_per_cond_enc', type=int, default=1,
                    help='number of cell for each conditional in encoder')
# decoder parameters
parser.add_argument('--num_channels_dec', type=int, default=32,
                    help='number of channels in decoder')
parser.add_argument('--num_postprocess_blocks', type=int, default=2,
                    help='number of postprocessing blocks')
parser.add_argument('--num_postprocess_cells', type=int, default=3,
                    help='number of cells per block')
parser.add_argument('--num_cell_per_cond_dec', type=int, default=1,
                    help='number of cell for each conditional in decoder')
# NAS
parser.add_argument('--use_se', action='store_true', default=False,
                    help='This flag enables squeeze and excitation.')
parser.add_argument('--res_dist', action='store_true', default=False,
                    help='This flag enables squeeze and excitation.')
parser.add_argument('--cont_training', action='store_true', default=False,
                    help='This flag enables training from an existing checkpoint.')

parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')

# moco specific configs:
parser.add_argument('--moco-dim', default=128, type=int,
                    help='feature dimension (default: 128)')
parser.add_argument('--moco-k', default=65536, type=int,
                    help='queue size; number of negative keys (default: 65536)')
parser.add_argument('--moco-m', default=0.999, type=float,
                    help='moco momentum of updating key encoder (default: 0.999)')
parser.add_argument('--moco-t', default=0.07, type=float,
                    help='softmax temperature (default: 0.07)')

# options for moco v2
parser.add_argument('--mlp', action='store_true',
                    help='use mlp head')
parser.add_argument('--aug-plus', action='store_true',
                    help='use moco v2 data augmentation')
parser.add_argument('--cos', action='store_true',
                    help='use cosine lr schedule')
parser.add_argument('--lr_weight', default=1300, type=int,
                    help='loss weight')

parser.add_argument('--recon_loss_weight', default=17500, type=int,
                    help='loss weight')

parser.add_argument('--loss', type=str, default='cpc',
                    choices=['cpc', 'mc_cpc'],
                    help="cpc: original objective, mc_cpc: multi-class, multi-label extension")

parser.add_argument('--optimizer', type=str, default='sgd', choices=['sgd', 'adam', 'adamw'])


parser.add_argument('--save-dir', default='', type=str,
                    help='save location')

def main():
    args = parser.parse_args()
    if not os.path.exists(args.save_dir):
        if len(args.save_dir) > 0:
            os.makedirs(args.save_dir)

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

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()

    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node,
                 args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu

    # suppress printing if not master
    if args.multiprocessing_distributed and args.gpu != 0:
        def print_pass(*args):
            pass
        builtins.print = print_pass

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)

    arch_instance = utils.get_arch_cells(args.arch_instance)

    model_fn = lambda : AutoEncoder(args, None, arch_instance)

    if args.dataset == 'cifar10' or args.dataset == 'cifar100':
        latent_dim = 16
    elif args.dataset == "celeba_64":
        latent_dim = 32
    elif args.dataset == "celeba_256" or args.dataset == "ffhq"::
        latent_dim = 64

    if args.dataset == 'celeba_256' or args.dataset == "ffhq":
        model = builder.MoCo(model_fn, 
            args.moco_dim, args.moco_k, args.moco_m, args.moco_t, args.mlp, 
            dim_mlp = latent_dim*latent_dim*args.num_latent_per_group, dim_mlp_next=2048)
    else:
        model = builder.MoCo(model_fn, 
            args.moco_dim, args.moco_k, args.moco_m, args.moco_t, args.mlp, 
            dim_mlp = latent_dim*latent_dim*args.num_latent_per_group)


    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int(
                (args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
        # comment out the following line for debugging
        raise NotImplementedError("Only DistributedDataParallel is supported.")
    else:
        # AllGather implementation (batch shuffle, queue update, etc.) in
        # this code only supports DistributedDataParallel.
        raise NotImplementedError("Only DistributedDataParallel is supported.")

    if args.rank == 0:
        writer = SummaryWriter(logdir=args.save_dir)
    else:
        writer = None
    # define loss function (criterion) and optimizer
    if args.loss == 'cpc':
        criterion = nn.CrossEntropyLoss().cuda(args.gpu)
    elif args.loss == 'mc_cpc':
        criterion = nn.KLDivLoss(reduction='batchmean').cuda(args.gpu)
    else:
        raise NotImplementedError(args.loss + ' not implemented yet.')

    # optionally resume from a checkpoint
    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), args.lr, weight_decay=args.weight_decay)
    else:
        raise NotImplementedError

    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code
    traindir = os.path.join(args.data)

    if args.aug_plus:
        # MoCo v2's aug: similar to SimCLR https://arxiv.org/abs/2002.05709
        if args.dataset == 'cifar10' or args.dataset == 'cifar100':
            augmentation = [
                transforms.RandomResizedCrop(32),
                transforms.RandomApply([
                    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
                ], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]
        elif args.dataset == 'celeba_64':
            augmentation = [
                transforms.RandomResizedCrop(64, scale=(0.25, 1.0)),
                transforms.RandomApply([
                    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
                ], p=0.5),
                transforms.RandomGrayscale(p=0.2),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]
        elif args.dataset == 'celeba_256':
            augmentation = [
                transforms.RandomResizedCrop(256, scale=(0.25, 1.0)),
                transforms.RandomApply([
                    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
                ], p=0.5),
                transforms.RandomGrayscale(p=0.2),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]
        elif args.dataset == 'ffhq':
            augmentation = [
                transforms.RandomResizedCrop(256, scale=(0.25, 1.0)),
                transforms.RandomApply([
                    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
                ], p=0.5),
                transforms.RandomGrayscale(p=0.2),
                transforms.RandomApply([moco.loader.GaussianBlur([.1, 2.])], p=0.2),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]

    else:
        raise NotImplementedError("Moco v1 augmentation not adapted")

    if args.dataset == "cifar10" :
        train_dataset = dset.CIFAR10(
            traindir, train=True,
            transform=loader.TwoCropsTransform(
                transforms.Compose(augmentation)),
            download=True)
    elif args.dataset == "cifar100" :
        train_dataset = dset.CIFAR100(
        traindir, train=True,
        transform=loader.TwoCropsTransform(
            transforms.Compose(augmentation)),
        download=True)
    elif args.dataset == "celeba_64" :
        train_dataset = LMDBDataset(root=traindir, name='celeba64', train=True, 
            transform=loader.TwoCropsTransform(
                transforms.Compose(augmentation)), is_encoded=True)
        print("the len of dataset is ", len(train_dataset))
    elif args.dataset == "celeba_256":
        train_dataset = LMDBDataset(root=traindir, name='celeba', train=True, 
            transform=loader.TwoCropsTransform(
                transforms.Compose(augmentation)))
        print("the len of dataset is ", len(train_dataset))
    elif args.dataset == "ffhq":
        train_dataset = LMDBDataset(root=traindir, name='ffhq', train=True,
            transform=moco.loader.TwoCropsTransform(
                transforms.Compose(augmentation)))
        print("the len of dataset is ", len(train_dataset))

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(
            train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=True)
    

    print("the len of loader is ", len(train_loader))

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, args, writer)
        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                                                    and args.rank % ngpus_per_node == 0):
            if epoch % 100 == 0 or epoch == args.epochs - 1:
                save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }, is_best=False, filename=os.path.join(args.save_dir, 'checkpoint_{:04d}.pth.tar'.format(epoch)))
            save_checkpoint({'epoch': epoch+1, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()},
                    is_best=False, filename=os.path.join(args.save_dir, 'checkpoint_recent.pth.tar'))

    if writer is not None:
        writer.close()


def train(train_loader, model, criterion, optimizer, epoch, args, writer):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    losses_rec = AverageMeter('Loss_rec', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')

    if args.loss == 'mc_cpc':
        top_n = AverageMeter('Acc@n', ':6.2f')

    progress_items = [batch_time, data_time, losses, losses_rec, top1, top5]
    if args.loss == 'mc_cpc':
        progress_items += [top_n]

    progress = ProgressMeter(
        len(train_loader),
        progress_items,
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()
    n_iters = len(train_loader) * epoch

    end = time.time()

    for i, (images, _) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            images[0] = images[0].cuda(args.gpu, non_blocking=True)
            images[1] = images[1].cuda(args.gpu, non_blocking=True)

        images[0] = utils.pre_process(images[0].cuda(), args.num_x_bits)
        images[1] = utils.pre_process(images[1].cuda(), args.num_x_bits)

        # compute output
        rec_logits, output, target = model(im_q=images[0], im_k=images[1])

        rec_output = model.module.encoder_q.decoder_output(rec_logits)
        recon_loss = torch.mean(utils.reconstruction_loss(rec_output, 
            images[0], crop=False))
        loss = recon_loss/float(args.recon_loss_weight)

        if args.loss == 'cpc':
            moco_loss = criterion(output, target)
        elif args.loss == 'mc_cpc':
            logits = output.view(1, -1)
            labels = torch.zeros_like(output)
            labels[:, 0] += 1.0 / output.size(0)
            labels = labels.view(1, -1)
            moco_loss = criterion(F.log_softmax(logits, dim=1), labels)
        else:
            raise NotImplementedError(args.loss + ' not implemented.')
        loss += moco_loss
        # acc1/acc5 are (K+1)-way contrast classifier accuracy
        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(moco_loss.item(), images[0].size(0))
        losses_rec.update(recon_loss.item(), images[0].size(0))
        top1.update(acc1[0], images[0].size(0))
        top5.update(acc5[0], images[0].size(0))

        if args.loss == 'mc_cpc':
            acc_n = accuracy_n(output)
            top_n.update(acc_n.item(), images[0].size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)
            if writer is not None:
                writer.add_scalar('pretrain/acc1', top1.avg, n_iters+i)
                writer.add_scalar('pretrain/acc5', top5.avg, n_iters+i)
                writer.add_scalar('pretrain/batch_time',
                                  batch_time.avg, n_iters+i)
                writer.add_scalar('pretrain/data_time',
                                  data_time.avg, n_iters+i)
                writer.add_scalar('pretrain/loss', losses.avg, n_iters+i)
                writer.add_scalar('pretrain/rec_loss', losses_rec.avg, n_iters+i)
                if args.loss == 'mc_cpc':
                    writer.add_scalar('pretrain/top_n', top_n.avg, n_iters+i)

                n = int(np.floor(np.sqrt(images[0].size(0))))
                x_img = images[0][:n*n]
                output_img = rec_output.mean if isinstance(rec_output, torch.distributions.bernoulli.Bernoulli) else rec_output.sample()
                output_img = output_img[:n*n]
                x_tiled = utils.tile_image(x_img, n)
                output_tiled = utils.tile_image(output_img, n)
                in_out_tiled = torch.cat((x_tiled, output_tiled), dim=2)
                writer.add_image('reconstruction', in_out_tiled, n_iters+i)


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
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

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
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


def accuracy_n(output):
    """
    Computes the accuracy over the top n predictions for Nx(K+1) class, N label classification.
    We assume that the targets are size (N, K+1) where the first element is positive.
    """
    with torch.no_grad():
        top_n = output.size(0)
        k = output.size(1)
        _, pred = output.view(-1).topk(top_n, 0, True, True)
        correct = pred.fmod(k).eq(0.0)
        return correct.float().sum().mul(100.0 / top_n)


if __name__ == '__main__':
    main()

