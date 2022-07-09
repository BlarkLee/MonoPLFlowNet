import os, sys
import os.path as osp
import time
from functools import partial
import gc
import traceback

import numpy as np

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data

import transforms.transforms_monopl_self as transforms
import datasets
import models
import cmd_args
from main_utils import *

from models import EPE3DLoss
from models.chamfer_loss import chamfer_loss
from models import DepthNet_linear
from models.DepthNet_linear import *
from evaluation_monopl_kitti_ablation_monosf_multi import evaluate


def main():
    # ensure numba JIT is on
    if 'NUMBA_DISABLE_JIT' in os.environ:
        del os.environ['NUMBA_DISABLE_JIT']

    # parse arguments
    global args
    args = cmd_args.parse_args_from_yaml(sys.argv[1])

    # -------------------- logging args --------------------
    if osp.exists(args.ckpt_dir):
        to_continue = query_yes_no('Attention!!!, ckpt_dir already exists!\
                                        Whether to continue?',
                                   default=None)
        if not to_continue:
            sys.exit(1)
    os.makedirs(args.ckpt_dir, mode=0o777, exist_ok=True)

    logger = Logger(osp.join(args.ckpt_dir, 'log'))
    logger.log('sys.argv:\n' + ' '.join(sys.argv))

    os.environ['NUMBA_NUM_THREADS'] = str(args.workers)
    logger.log('NUMBA NUM THREADS\t' + os.environ['NUMBA_NUM_THREADS'])

    for arg in sorted(vars(args)):
        logger.log('{:20s} {}'.format(arg, getattr(args, arg)))
    logger.log('')

    # -------------------- dataset & loader --------------------
    if not args.evaluate:
        train_dataset = datasets.__dict__[args.dataset](
            train=True,
            transform=transforms.Augmentation(args.aug_together,
                                              args.aug_pc2,
                                              args.data_process,
                                              args.num_points,
                                              args.allow_less_points),
            gen_func=transforms.GenerateDataUnsymmetric(args),
            args=args
        )
        logger.log('train_dataset: ' + str(train_dataset))
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.workers,
            pin_memory=True,
            worker_init_fn=lambda x: np.random.seed((torch.initial_seed()) % (2 ** 32))
        )

    val_dataset = datasets.__dict__[args.dataset](
        train=False,
        transform=transforms.ProcessData(args.data_process,
                                         args.num_points,
                                         args.allow_less_points),
        gen_func=transforms.GenerateDataUnsymmetric(args),
        args=args
    )
    logger.log('val_dataset: ' + str(val_dataset))
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        worker_init_fn=lambda x: np.random.seed((torch.initial_seed()) % (2 ** 32))
    )

    # -------------------- create model --------------------
    logger.log("=>  creating model '{}'".format(args.arch))
    model = models.__dict__[args.arch](args)

    if not args.evaluate:
        init_func = partial(init_weights_multi, init_type=args.init, gain=args.gain)
        model.apply(init_func)
    logger.log(model)

    model = torch.nn.DataParallel(model).cuda()
    criterion = EPE3DLoss().cuda()

    if args.evaluate:
        torch.backends.cudnn.enabled = False
    else:
        cudnn.benchmark = True
    # https://discuss.pytorch.org/t/what-does-torch-backends-cudnn-benchmark-do/5936
    # But if your input sizes changes at each iteration,
    # then cudnn will benchmark every time a new size appears,
    # possibly leading to worse runtime performances.
    
    args.encoder = 'densenet121_bts'
    args.bts_size = 512
    args.max_depth = 35
    model_depth = DepthNet_linear.DepthModel(args)
    model_depth.decoder.apply(weights_init_xavier)
    model_depth = torch.nn.DataParallel(model_depth).cuda()
    
    gen_lattice = transforms.GenerateDataUnsymmetric(args)
    
    # -------------------- resume --------------------
    if args.resume:
        if osp.isfile(args.resume):
            logger.log("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            checkpoint['state_dict'] = {k.replace('_model', 'module'):v for k,v in checkpoint['state_dict'].items()}
            model.load_state_dict(checkpoint['state_dict'], strict=True)
        else:
            logger.log("=> no checkpoint found at '{}'".format(args.resume))
            checkpoint = None
    else:
        args.start_epoch = 0
    
    if args.depth_checkpoint_path != '':
        if os.path.isfile(args.depth_checkpoint_path):
            print("Loading depth checkpoint '{}'".format(args.depth_checkpoint_path))
            depth_checkpoint = torch.load(args.depth_checkpoint_path)
            model_depth.load_state_dict(depth_checkpoint['model'], strict=True)
        else:
            print("No depth checkpoint found at '{}'".format(args.depth_checkpoint_path))
            

    # -------------------- evaluation --------------------
    if args.evaluate:
        _, res_dict = evaluate(val_loader, model, model_depth, logger, args, gen_lattice)
        logger.close()
        return res_dict

if __name__ == '__main__':
    main()
