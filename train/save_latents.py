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
import torchvision.utils as vutils
import torchvision
import torchvision.datasets as tdatasets
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

from d2c import AutoEncoder
from d2c import builder
from d2c import loader
from d2c import utils

from tensorboardX import SummaryWriter
from lmdb_datasets import LMDBDataset


join=os.path.join

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--data', type=str, default='./data',
                        help='path to dataset')
parser.add_argument('-j', '--workers', default=32, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')

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

parser.add_argument('--eval_mode', default='reconstruction', type=str,required=True,
                    choices=["reconstruction", "save_latents",
                    "save_original"],
                    help='The mode to run for evaluation')

parser.add_argument('--out_dir', default='images', type=str,
                    help='The output dir')

parser.add_argument('--latent_fname', default='', type=str,
                    help='The path containing sampled latents')
parser.add_argument('--label_fname', default='', type=str,
                    help='The path containing labels of latents')
parser.add_argument('--single_batch', action='store_true', default=False)
parser.add_argument('--dir_name', default='', type=str,
                    help='the dir from which to read images')
parser.add_argument('--nrow', default=0, type=int,
                    help='number of rows in output image')
parser.add_argument('--out_fname', default='', type=str,
                    help='output fname')
parser.add_argument('--denormalize', action='store_true', default=False,
                    help='Denormalizes while sampling')

class CropCelebA64(object):
    """ This class applies cropping for CelebA64. This is a simplified implementation of:
    https://github.com/andersbll/autoencoding_beyond_pixels/blob/master/dataset/celeba.py
    """
    def __call__(self, pic):
        new_pic = pic.crop((15, 40, 178 - 15, 218 - 30))
        return new_pic

    def __repr__(self):
        return self.__class__.__name__ + '()'

def main():
    args = parser.parse_args()
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

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
        args.world_size = ngpus_per_node * args.world_size
        mp.spawn(main_worker, nprocs=ngpus_per_node,
                 args=(ngpus_per_node, args))
    else:
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
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)

    arch_instance = utils.get_arch_cells(args.arch_instance)
    model_fn = lambda : AutoEncoder(args, None, arch_instance)

    if args.dataset == 'cifar10' or args.dataset == 'cifar100':
        latent_dim = 16
    elif args.dataset == "celeba_64":
        latent_dim = 32
    elif args.dataset == "celeba_256":
        latent_dim = 64

    if args.dataset == 'celeba_256':
        model = builder.MoCo(model_fn, 
            args.moco_dim, args.moco_k, args.moco_m, args.moco_t, False, 
            dim_mlp = latent_dim*latent_dim*args.num_latent_per_group, dim_mlp_next=2048)
    else:
        model = builder.MoCo(model_fn, 
            args.moco_dim, args.moco_k, args.moco_m, args.moco_t, False, 
            dim_mlp = latent_dim*latent_dim*args.num_latent_per_group)


    if args.distributed:
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int(
                (args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[args.gpu])
        else:
            model.cuda()
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
        raise NotImplementedError("Only DistributedDataParallel is supported.")
    else:
        raise NotImplementedError("Only DistributedDataParallel is supported.")
    

    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)

            state_dict = checkpoint['state_dict']
            for k in list(state_dict.keys()):
                if k.startswith('module.encoder_q.fc') or k.startswith('module.encoder_k.fc'):
                    print("deleting ", k)
                    del state_dict[k]

            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(state_dict)
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True
    model.eval()

    # Data loading code
    traindir = os.path.join(args.data)

    if args.dataset == "cifar10" :
        train_dataset = tdatasets.CIFAR10(
            traindir, train=True,
            transform=transforms.ToTensor(),
            download=True)
        
        val_dataset = tdatasets.CIFAR10(
            traindir, train=False,
            transform=transforms.ToTensor(),
            download=True)
    elif args.dataset == "cifar100":
        train_dataset = tdatasets.CIFAR100(
            traindir, train=True,
            transform=transforms.ToTensor(),
            download=True)
        
        val_dataset = tdatasets.CIFAR100(
            traindir, train=False,
            transform=transforms.ToTensor(),
            download=True)

    elif args.dataset == "celeba_64":
        print("the train dir is ", traindir)
        valid_transform = transforms.Compose([
            CropCelebA64(),
            transforms.Resize(64),
            transforms.ToTensor(),
        ])

        train_dataset = tdatasets.CelebA(
            traindir, split="train",
            transform=valid_transform,
            download=False)

        val_dataset = tdatasets.CelebA(
            traindir, split="valid",
            transform=valid_transform,
            download=False)

        # train_dataset = LMDBDataset(root=traindir, name='celeba64', train=True, 
        #     transform=transforms.Compose([
        #         transforms.Resize((64, 64)),
        #         transforms.ToTensor()]), is_encoded=True)
        # val_dataset = LMDBDataset(root=traindir, name='celeba64', train=False, 
        #     transform=transforms.Compose([
        #         transforms.Resize((64, 64)),
        #         transforms.ToTensor()]), is_encoded=True)
    elif args.dataset == "celeba_256":
        train_dataset = LMDBDataset(root=traindir, name='celeba', train=True, 
            transform=transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor()]))
        val_dataset = LMDBDataset(root=traindir, name='celeba', train=False, 
            transform=transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor()]))
    else:
        raise NotImplementedError("Dataset not supported")


    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, sampler=None, drop_last=False)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, sampler=None, drop_last=False)

    
    if args.eval_mode == "reconstruction":
        save_images(val_loader, model, 
            join(args.out_dir, "reconstruct_"+str(args.dataset)), args)
    elif args.eval_mode == "save_original":
        save_original_images(train_loader, join(args.out_dir, "orig_" + str(args.dataset)), args)
    elif args.eval_mode == "save_latents":
        if len(args.out_fname) == 0:
            train_fname = join(args.out_dir, "train_feats_"+str(args.dataset))
        else:
            train_fname = args.out_fname
        if "celeba" in args.dataset or "ffhq" in args.dataset:
            get_labels=False
        else:
            get_labels=True
        save_latents(train_loader, model,
         train_fname, args, rotate=True, get_labels=get_labels)

def save_latents(loader, model, out_fname, args, rotate=True, get_labels=True):

    features = []
    labels = []
    with torch.no_grad():
        for i, (x,y) in enumerate(loader):
            x = x.cuda(args.gpu, non_blocking=True)
            if rotate:
                x_rotate = torch.flip(x.clone(), dims=[-1])
                x = torch.cat((x, x_rotate), axis=0)
                if get_labels:
                    y = torch.cat((y, y), axis=0)
            x = utils.pre_process(x.cuda(), args.num_x_bits)
            feats = model.module.encoder_q(x, get_latent=True, reshape=False)
            features.extend(feats.cpu().data.numpy())
            if get_labels:
                labels.extend(y.cpu().data.numpy())
            print(i, len(loader))

        features = np.array(features)
        print(features.shape, np.min(features), np.max(features))
        data_mean = torch.from_numpy(np.mean(features, axis=(0, 2, 3), keepdims=True)).cuda()
        data_std = torch.from_numpy(np.std(features, axis=(0, 2, 3), keepdims=True)).cuda()
        latent_stats_ckpt = {'mean': data_mean, 'std': data_std}
        torch.save(latent_stats_ckpt, os.path.join(args.out_dir, 'latent_stats.ckpt'))
        np.save(out_fname, features)

        if get_labels:
            labels = np.array(labels)
            np.save(out_fname.replace(".npy", "_labels.npy"), labels)

def save_original_images(loader, out_fname, args):
    images_orig = []
    with torch.no_grad():
        for i, (x,_) in enumerate(loader):

            x = x.cuda(args.gpu, non_blocking=True)
            x = utils.pre_process(x.cuda(), args.num_x_bits)
            images_orig.extend(np.transpose(x.cpu().data.numpy()*255.0, (0, 2, 3, 1)))
            if len(images_orig) >= 50000:
                break

        images_orig = np.array(images_orig)[:50000]
        np.save(out_fname+".npy", images_orig)
        vutils.save_image(x, out_fname+ ".png")

def save_images(loader, model, out_fname, args):

    images = []
    images_orig = []
    loss_fn = nn.MSELoss(reduction='sum')
    rec_loss = 0.0
    total = 0.0

    with torch.no_grad():
        for i, (x,_) in enumerate(loader):

            x = x.cuda(args.gpu, non_blocking=True)
            x = utils.pre_process(x.cuda(), args.num_x_bits)

            rec_logits = model.module.encoder_q(x)
            rec_output = model.module.encoder_q.decoder_output(rec_logits)

            output_img = rec_output.mean if isinstance(rec_output, torch.distributions.bernoulli.Bernoulli) else rec_output.sample()

            rec_loss += loss_fn(output_img, x).item()
            total += x.size(0)

            print(i, len(loader), output_img.size(), output_img.min(), output_img.max())
            images.extend(np.transpose(output_img.cpu().data.numpy()*255.0, (0, 2, 3, 1)))
            if args.single_batch:
                break
            if len(images) >= 10000:
                break

        images = np.array(images)
        print("the mean error is ", rec_loss/float(total))
        print(images.shape, np.min(images), np.max(images))
        if not args.single_batch:
            np.save(out_fname+".npy", images)
        vutils.save_image(output_img, out_fname+".png")
        vutils.save_image(x, out_fname+ "_orig.png")

if __name__ == '__main__':
    main()
