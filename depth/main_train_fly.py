# This implementation is modified based on BTS https://github.com/cleinc/bts/tree/master/pytorch

import time
import argparse
import datetime
import sys
import os
from functools import partial

import numpy as np

import torch
import torch.nn as nn
import torch.nn.utils as utils
import torch.nn.functional as tf

import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp

from tensorboardX import SummaryWriter

import matplotlib
import matplotlib.cm
import threading
from tqdm import tqdm

from depthnet import DepthModel
from dataloader_flyingthings3d import *

from collections import Counter



def convert_arg_line_to_args(arg_line):
    for arg in arg_line.split():
        if not arg.strip():
            continue
        yield arg


parser = argparse.ArgumentParser(fromfile_prefix_chars='@')
parser.convert_arg_line_to_args = convert_arg_line_to_args

parser.add_argument('--mode',                      type=str,   help='train or test', default='train')
parser.add_argument('--model_name',                type=str,   help='model name', default='bts_eigen_v2')
parser.add_argument('--encoder',                   type=str,   help='type of encoder, desenet121_bts, densenet161_bts, '
                                                                    'resnet101_bts, resnet50_bts, resnext50_bts or resnext101_bts',
                                                               default='densenet161_bts')
# Dataset
parser.add_argument('--dataset',                   type=str,   help='dataset to train on, kitti or nyu', default='nyu')
parser.add_argument('--flyingthings3d_path',       type=str,   help='path to flyingthings3d dataset', required=True)
parser.add_argument('--input_height',              type=int,   help='input height', default=480)
parser.add_argument('--input_width',               type=int,   help='input width',  default=640)
parser.add_argument('--max_depth',                 type=float, help='maximum depth in estimation', default=10)

parser.add_argument("--depth_ckpt", type=str, default='"/mnt/fire/runfa/SM3D/bts/pytorch/models/09_18_alllevel/model-183500-best_d1_0.95980"', help="specify the checkpoint of rcnn if trained separated")

# Log and save
parser.add_argument('--log_directory',             type=str,   help='directory to save checkpoints and summaries', default='')
parser.add_argument('--checkpoint_path',           type=str,   help='path to a checkpoint to load', default='')
parser.add_argument('--log_freq',                  type=int,   help='Logging frequency in global steps', default=100)
parser.add_argument('--save_freq',                 type=int,   help='Checkpoint saving frequency in global steps', default=500)

# Training
parser.add_argument('--fix_first_conv_blocks',                 help='if set, will fix the first two conv blocks', action='store_true')
parser.add_argument('--fix_first_conv_block',                  help='if set, will fix the first conv block', action='store_true')
parser.add_argument('--bn_no_track_stats',                     help='if set, will not track running stats in batch norm layers', action='store_true')
parser.add_argument('--weight_decay',              type=float, help='weight decay factor for optimization', default=1e-2)
parser.add_argument('--bts_size',                  type=int,   help='initial num_filters in bts', default=512)
parser.add_argument('--retrain',                               help='if used with checkpoint_path, will restart training from step zero', action='store_true')
parser.add_argument('--adam_eps',                  type=float, help='epsilon in Adam optimizer', default=1e-6)
parser.add_argument('--batch_size',                type=int,   help='batch size', default=4)
parser.add_argument('--num_epochs',                type=int,   help='number of epochs', default=50)
parser.add_argument('--learning_rate',             type=float, help='initial learning rate', default=1e-4)
parser.add_argument('--end_learning_rate',         type=float, help='end learning rate', default=-1)
parser.add_argument('--variance_focus',            type=float, help='lambda in paper: [0, 1], higher value more focus on minimizing variance of error', default=0.85)

# Preprocessing
parser.add_argument('--do_random_rotate',                      help='if set, will perform random rotation for augmentation', action='store_true')
parser.add_argument('--degree',                    type=float, help='random rotation maximum degree', default=2.5)
parser.add_argument('--do_kb_crop',                            help='if set, crop input images as kitti benchmark images', action='store_true')
parser.add_argument('--use_right',                             help='if set, will randomly use right images when train on KITTI', action='store_true')

# Multi-gpu training
parser.add_argument('--num_threads',               type=int,   help='number of threads to use for data loading', default=1)
parser.add_argument('--world_size',                type=int,   help='number of nodes for distributed training', default=1)
parser.add_argument('--rank',                      type=int,   help='node rank for distributed training', default=0)
parser.add_argument('--dist_url',                  type=str,   help='url used to set up distributed training', default='tcp://127.0.0.1:1234')
parser.add_argument('--dist_backend',              type=str,   help='distributed backend', default='nccl')
parser.add_argument('--gpu',                       type=int,   help='GPU id to use.', default=None)
parser.add_argument('--multiprocessing_distributed',           help='Use multi-processing distributed training to launch '
                                                                    'N processes per node, which has N GPUs. This is the '
                                                                    'fastest way to use PyTorch for either single node or '
                                                                    'multi node data parallel training', action='store_true',)
# Online eval
parser.add_argument('--do_online_eval',                        help='if set, perform online eval in every eval_freq steps', action='store_true')
parser.add_argument('--min_depth_eval',            type=float, help='minimum depth for evaluation', default=1e-3)
parser.add_argument('--max_depth_eval',            type=float, help='maximum depth for evaluation', default=80)
parser.add_argument('--eigen_crop',                            help='if set, crops according to Eigen NIPS14', action='store_true')
parser.add_argument('--garg_crop',                             help='if set, crops according to Garg  ECCV16', action='store_true')
parser.add_argument('--eval_freq',                 type=int,   help='Online evaluation frequency in global steps', default=500)
parser.add_argument('--eval_summary_directory',    type=str,   help='output directory for eval summary,'
                                                                    'if empty outputs to checkpoint folder', default='')

if sys.argv.__len__() == 2:
    arg_filename_with_prefix = '@' + sys.argv[1]
    args = parser.parse_args([arg_filename_with_prefix])
else:
    args = parser.parse_args()

if args.mode == 'train'and not args.checkpoint_path:
    from depthnet import *

elif args.mode == 'train' and args.checkpoint_path:
    from depthnet import *
    print("load pretrained depth model")
    model_dir = os.path.dirname(args.checkpoint_path)
    model_name = os.path.basename(model_dir)
    import sys
    sys.path.append(model_dir)


inv_normalize = transforms.Normalize(
    mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
    std=[1/0.229, 1/0.224, 1/0.225]
)

eval_metrics_depth = ['silog', 'abs_rel', 'log10', 'rms', 'sq_rel', 'log_rms', 'd1', 'd2', 'd3']
eval_metrics_3d = ['EPE3D', 'acc3d_strict', 'acc3d_relax', 'outlier']
eval_metrics_2d = ['EPE2D', 'acc2d']


def compute_errors_depth(gt, pred):
    thresh = np.maximum((gt / pred), (pred / gt))
    d1 = (thresh < 1.25).mean()
    d2 = (thresh < 1.25 ** 2).mean()
    d3 = (thresh < 1.25 ** 3).mean()

    rms = (gt - pred) ** 2
    rms = np.sqrt(rms.mean())

    log_rms = (np.log(gt) - np.log(pred)) ** 2
    log_rms = np.sqrt(log_rms.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)
    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    err = np.log(pred) - np.log(gt)
    silog = np.sqrt(np.mean(err ** 2) - np.mean(err) ** 2) * 100

    err = np.abs(np.log10(pred) - np.log10(gt))
    log10 = np.mean(err)

    return [silog, abs_rel, log10, rms, sq_rel, log_rms, d1, d2, d3]


def sceneflow2flow(pc1, pc2, pc1_gt, pc2_gt):
    px1, py1 = project_3d_to_2d(pc1)
    px2, py2 = project_3d_to_2d(pc2)
    px1_gt, py1_gt = project_3d_to_2d(pc1_gt)
    px2_gt, py2_gt = project_3d_to_2d(pc2_gt)

    flow_x = px2 - px1
    flow_y = py2 - py1

    flow_x_gt = px2_gt - px1_gt
    flow_y_gt = py2_gt - py1_gt

    flow_pred = np.concatenate((flow_x[..., None], flow_y[..., None]), axis=-1)
    flow_gt = np.concatenate((flow_x_gt[..., None], flow_y_gt[..., None]), axis=-1)
    return flow_pred, flow_gt


def project_3d_to_2d(pc, f=-1050., cx=479.5, cy=269.5, constx=0, consty=0, constz=0):
    x = (pc[..., 0] * f + cx * pc[..., 2] + constx) / (pc[..., 2] + constz)
    y = (pc[..., 1] * f + cy * pc[..., 2] + consty) / (pc[..., 2] + constz)

    return x, y


def evaluate_3d(sf_pred, sf_gt):
    """
    sf_pred: (N, 3)
    sf_gt: (N, 3)
    """
    l2_norm = np.linalg.norm(sf_gt - sf_pred, axis=-1)
    EPE3D = l2_norm.mean()

    sf_norm = np.linalg.norm(sf_gt, axis=-1)
    relative_err = l2_norm / (sf_norm + 1e-4)

    acc3d_strict = (np.logical_or(l2_norm < 0.05, relative_err < 0.05)).astype(np.float).mean()
    acc3d_relax = (np.logical_or(l2_norm < 0.1, relative_err < 0.1)).astype(np.float).mean()
    outlier = (np.logical_or(l2_norm > 0.3, relative_err > 0.1)).astype(np.float).mean()

    return EPE3D, acc3d_strict, acc3d_relax, outlier


def evaluate_2d(flow_pred, flow_gt):
    """
    flow_pred: (N, 2)
    flow_gt: (N, 2)
    """

    epe2d = np.linalg.norm(flow_gt - flow_pred, axis=-1)
    epe2d_mean = epe2d.mean()

    flow_gt_norm = np.linalg.norm(flow_gt, axis=-1)
    relative_err = epe2d / (flow_gt_norm + 1e-5)

    acc2d = (np.logical_or(epe2d < 3., relative_err < 0.05)).astype(np.float).mean()

    return epe2d_mean, acc2d


def block_print():
    sys.stdout = open(os.devnull, 'w')


def enable_print():
    sys.stdout = sys.__stdout__


def get_num_lines(file_path):
    f = open(file_path, 'r')
    lines = f.readlines()
    f.close()
    return len(lines)


def colorize(value, vmin=None, vmax=None, cmap='Greys'):
    value = value.cpu().numpy()[:, :, :]
    value = np.log10(value)

    vmin = value.min() if vmin is None else vmin
    vmax = value.max() if vmax is None else vmax

    if vmin != vmax:
        value = (value - vmin) / (vmax - vmin)
    else:
        value = value*0.

    cmapper = matplotlib.cm.get_cmap(cmap)
    value = cmapper(value, bytes=True)

    img = value[:, :, :3]

    return img.transpose((2, 0, 1))


def normalize_result(value, vmin=None, vmax=None):
    value = value.cpu().numpy()[0, :, :]

    vmin = value.min() if vmin is None else vmin
    vmax = value.max() if vmax is None else vmax

    if vmin != vmax:
        value = (value - vmin) / (vmax - vmin)
    else:
        value = value * 0.

    return np.expand_dims(value, 0)


def set_misc(model):
    if args.bn_no_track_stats:
        print("Disabling tracking running stats in batch norm layers")
        model.apply(bn_init_as_tf)

    if args.fix_first_conv_blocks:
        if 'resne' in args.encoder:
            fixing_layers = ['base_model.conv1', 'base_model.layer1.0', 'base_model.layer1.1', '.bn']
        else:
            fixing_layers = ['conv0', 'denseblock1.denselayer1', 'denseblock1.denselayer2', 'norm']
        print("Fixing first two conv blocks")
    elif args.fix_first_conv_block:
        if 'resne' in args.encoder:
            fixing_layers = ['base_model.conv1', 'base_model.layer1.0', '.bn']
        else:
            fixing_layers = ['conv0', 'denseblock1.denselayer1', 'norm']
        print("Fixing first conv block")
    else:
        if 'resne' in args.encoder:
            fixing_layers = ['base_model.conv1', '.bn']
        else:
            fixing_layers = ['conv0', 'norm']
        print("Fixing first conv layer")

    for name, child in model.named_children():
        if not 'encoder' in name:
            continue
        for name2, parameters in child.named_parameters():
            if any(x in name2 for x in fixing_layers):
                parameters.requires_grad = False

                
                
def online_eval(model, model_scene, dataloader_eval, gpu, ngpus, args, depth_mask):
    eval_measures = torch.zeros(10).cuda(device=gpu)
    eval_measures_3d = torch.zeros(4).cuda(device=gpu)
    eval_measures_2d = torch.zeros(2).cuda(device=gpu)
    for _, sample_batched in enumerate(tqdm(dataloader_eval.data)):
        if sample_batched == 0:
                continue
        with torch.no_grad():
            imageL_0 = torch.autograd.Variable(sample_batched['image'].cuda(args.gpu, non_blocking=True))
            focal = torch.autograd.Variable(sample_batched['focal'].cuda(args.gpu, non_blocking=True))
            depth_gt_0 = torch.autograd.Variable(sample_batched['depth'].cuda(args.gpu, non_blocking=True))
            
            imageL_0 = imageL_0.permute(0,3,1,2)
            lpg8x8, lpg4x4, lpg2x2, reduc1x1, depth_est_0, all_feat_0 = model(imageL_0, focal)
       
            pred_depth = depth_est_0.cpu().numpy().squeeze()[:540, :]
            gt_depth = depth_gt_0.cpu().numpy().squeeze()
            
            
        pred_depth[pred_depth < args.min_depth_eval] = args.min_depth_eval
        pred_depth[pred_depth > args.max_depth_eval] = args.max_depth_eval
        pred_depth[np.isinf(pred_depth)] = args.max_depth_eval
        pred_depth[np.isnan(pred_depth)] = args.min_depth_eval

        valid_mask = np.logical_and(gt_depth > args.min_depth_eval, gt_depth < args.max_depth_eval)    
          
        measures = compute_errors_depth(gt_depth[valid_mask], pred_depth[valid_mask])
        eval_measures[:9] += torch.tensor(measures).cuda(device=gpu)
        eval_measures[9] += 1
        
        
        

    if args.multiprocessing_distributed:
        group = dist.new_group([i for i in range(ngpus)])
        dist.all_reduce(tensor=eval_measures, op=dist.ReduceOp.SUM, group=group)

    if not args.multiprocessing_distributed or gpu == 0:
        eval_measures_cpu = eval_measures.cpu()
        eval_measures_3d_cpu = eval_measures_3d.cpu()
        eval_measures_2d_cpu = eval_measures_2d.cpu()
        cnt = eval_measures_cpu[9].item()
        eval_measures_cpu /= cnt
        eval_measures_3d_cpu /= cnt
        eval_measures_2d_cpu /= cnt
        print('Computing errors for {} eval samples'.format(int(cnt)))
        print("{:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}".format('silog', 'abs_rel', 'log10', 'rms',
                                                                                     'sq_rel', 'log_rms', 'd1', 'd2',
                                                                                     'd3'))
        for i in range(8):
            print('{:7.3f}, '.format(eval_measures_cpu[i]), end='')
        print('{:7.3f}'.format(eval_measures_cpu[8]))

        return eval_measures_cpu
    return None


def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size, rank=args.rank)

    # Create model
    
    model = DepthModel(args)
    model.train()
    model.decoder.apply(weights_init_xavier)
    set_misc(model)

    num_params = sum([np.prod(p.size()) for p in model.parameters()])
    print("Total number of parameters: {}".format(num_params))

    num_params_update = sum([np.prod(p.shape) for p in model.parameters() if p.requires_grad])
    print("Total number of learning parameters: {}".format(num_params_update))
    
    if args.distributed:
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            args.batch_size = int(args.batch_size / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        else:
            model.cuda()
            model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)
    else:
        model = torch.nn.DataParallel(model)
        model.cuda()

    if args.distributed:
        print("Model Initialized on GPU: {}".format(args.gpu))
    else:
        print("Model Initialized")

        


    global_step = 0
    best_eval_measures_lower_better = torch.zeros(6).cpu() + 1e3
    best_eval_measures_higher_better = torch.zeros(3).cpu()
    best_eval_steps = np.zeros(9, dtype=np.int32)

    # Training parameters
    
    optimizer = torch.optim.Adam([{'params': model.parameters(), 'weight_decay': 0, 'lr': 1e-4}],                          
                                  lr=args.learning_rate, eps=args.adam_eps)
    
    
    if args.retrain:
        global_step = 0

    cudnn.benchmark = True

    dataloader = DepthDataLoader(args, 'train')
    dataloader_eval = DepthDataLoader(args, 'online_eval')

    # Logging
    if not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank % ngpus_per_node == 0):
        writer = SummaryWriter(args.log_directory + '/' + args.model_name + '/summaries', flush_secs=30)
        if args.do_online_eval:
            if args.eval_summary_directory != '':
                eval_summary_path = os.path.join(args.eval_summary_directory, args.model_name)
            else:
                eval_summary_path = os.path.join(args.log_directory, 'eval')
            eval_summary_writer = SummaryWriter(eval_summary_path, flush_secs=30)

    silog_criterion = silog_loss(variance_focus=args.variance_focus)

    start_time = time.time()
    duration = 0

    num_log_images = args.batch_size
    end_learning_rate = args.end_learning_rate if args.end_learning_rate != -1 else 0.001 * args.learning_rate

    var_sum = [var.sum() for var in model.parameters() if var.requires_grad]
    var_cnt = len(var_sum)
    var_sum = np.sum(var_sum)

    print("Initial variables' sum: {:.3f}, avg: {:.3f}".format(var_sum, var_sum/var_cnt))

    steps_per_epoch = len(dataloader.data)
    num_total_steps = args.num_epochs * steps_per_epoch
    epoch = global_step // steps_per_epoch
    
    #loading loss
    
    #loading camera parmameters
    count = 0
    scene_loss_total = 0
    depth_loss_total = 0
    loss_total = 0
    while epoch < args.num_epochs:
        if args.distributed:
            dataloader.train_sampler.set_epoch(epoch)

        for step, sample_batched in enumerate(dataloader.data):

            
            optimizer.zero_grad()
            before_op_time = time.time()

            imageL_0 = torch.autograd.Variable(sample_batched['image'].cuda(args.gpu, non_blocking=True))
            focal = torch.autograd.Variable(sample_batched['focal'].cuda(args.gpu, non_blocking=True))
            depth_gt_0 = torch.autograd.Variable(sample_batched['depth'].cuda(args.gpu, non_blocking=True))
            depth_mask = torch.autograd.Variable(sample_batched['depth_mask'].cuda(args.gpu, non_blocking=True))
            
            imageL_0 = imageL_0.permute(0,3,1,2)
            lpg8x8, lpg4x4, lpg2x2, reduc1x1, depth_est_0, all_feat_0 = model(imageL_0, focal)

            depth_gt_0 = depth_gt_0.permute(0,3,1,2)
            depth_mask = torch.unsqueeze(depth_mask, 1)
            depth_est_0 = depth_est_0[:,:,:540,:]
            lpg2x2 = lpg2x2[:,:,:540,:]
            lpg4x4 = lpg4x4[:,:,:540,:]
            lpg8x8 = lpg8x8[:,:,:540,:]
            
            depth_loss_temp = silog_criterion.forward(depth_est_0, lpg2x2, lpg4x4, lpg8x8, depth_gt_0, depth_mask.to(torch.bool))
            
            
            count += 1
            depth_loss = depth_loss_temp
            depth_loss.backward()

            for param_group in optimizer.param_groups:
                current_lr = (args.learning_rate - end_learning_rate) * (1 - global_step / num_total_steps) ** 0.9 + end_learning_rate
                param_group['lr'] = current_lr

            optimizer.step()

            if not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank % ngpus_per_node == 0):
                print("[epoch][s/s_per_e/gs]: [%d][%d/%d/%d], lr: %.4f, depth_loss:%.4f"%(epoch, step, steps_per_epoch, global_step, current_lr, depth_loss))
                if np.isnan(depth_loss.cpu().item()):
                    print('NaN in loss occurred. Aborting training.')
                    return -1

            duration += time.time() - before_op_time
            if global_step and global_step % args.log_freq == 0:# and not model_just_loaded:
                var_sum = [var.sum() for var in model_scene.parameters() if var.requires_grad]
                var_cnt = len(var_sum)
                var_sum = np.sum(var_sum)
                examples_per_sec = args.batch_size / duration * args.log_freq
                duration = 0
                time_sofar = (time.time() - start_time) / 3600
                training_time_left = (num_total_steps / global_step - 1.0) * time_sofar
                if not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank % ngpus_per_node == 0):
                    print("{}".format(args.model_name))
                print("GPU: %s | examples/s: %4.2f | depth_loss: %.5f | var sum: %.3f | avg: %.3f | time elasped: %.2f h | time left: %.2f h"%(args.gpu, examples_per_sec, depth_loss, var_sum.item(), var_sum.item()/var_cnt, time_sofar, training_time_left))

                if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                                                            and args.rank % ngpus_per_node == 0):
                    writer.add_scalar('depth_loss', depth_loss, global_step)
                    writer.add_scalar('learning_rate', current_lr, global_step)
                    writer.add_scalar('var average', var_sum.item()/var_cnt, global_step)
                    
                    for i in range(num_log_images):
                        writer.add_image('depth_gt/image/{}'.format(i), normalize_result(1/depth_gt_0[i, :, :, :].data), global_step)
                        writer.add_image('depth_est/image/{}'.format(i), normalize_result(1/depth_est_0[i, :, :, :].data), global_step)
                        writer.add_image('reduc1x1/image/{}'.format(i), normalize_result(1/reduc1x1[i, :, :, :].data), global_step)
                        writer.add_image('lpg2x2/image/{}'.format(i), normalize_result(1/lpg2x2[i, :, :, :].data), global_step)
                        writer.add_image('lpg4x4/image/{}'.format(i), normalize_result(1/lpg4x4[i, :, :, :].data), global_step)
                        writer.add_image('lpg8x8/image/{}'.format(i), normalize_result(1/lpg8x8[i, :, :, :].data), global_step)
                        writer.add_image('image/image/{}'.format(i), inv_normalize(imageL_0[i, :, :, :]).data, global_step)
                    writer.flush()

            if not args.do_online_eval and global_step and global_step % args.save_freq == 0:
                if not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank % ngpus_per_node == 0):
                    checkpoint = {'global_step': global_step,
                                  'model': model.state_dict(),
                                  'optimizer': optimizer.state_dict()}
                    torch.save(checkpoint, args.log_directory + '/' + args.model_name + '/model-{}'.format(global_step))

            if args.do_online_eval and global_step and global_step % args.eval_freq == 0: # and not model_just_loaded:
                time.sleep(0.1)
                model.eval()
                model_scene.eval()
                eval_measures = online_eval(model, model_scene, dataloader_eval, gpu, ngpus_per_node, args, depth_mask)
                
                if eval_measures is not None:
                    for i in range(9): 
                        eval_summary_writer.add_scalar(eval_metrics_depth[i], eval_measures[i].cpu(), int(global_step))
                        measure = eval_measures[i]
                        is_best = False
                        if i < 6 and measure < best_eval_measures_lower_better[i]:
                            old_best = best_eval_measures_lower_better[i].item()
                            best_eval_measures_lower_better[i] = measure.item()
                            is_best = True
                        elif i >= 6 and measure > best_eval_measures_higher_better[i-6]:
                            old_best = best_eval_measures_higher_better[i-6].item()
                            best_eval_measures_higher_better[i-6] = measure.item()
                            is_best = True
                        if is_best:
                            old_best_step = best_eval_steps[i]
                            old_best_name = '/model-{}-best_{}_{:.5f}'.format(old_best_step, eval_metrics_depth[i], old_best)
                            model_path = args.log_directory + '/' + args.model_name + old_best_name
                            if os.path.exists(model_path):
                                command = 'rm {}'.format(model_path)
                                os.system(command)
                            best_eval_steps[i] = global_step
                            model_save_name = '/model-{}-best_{}_{:.5f}'.format(global_step, eval_metrics_depth[i], measure)
                            print('New best for {}. Saving model: {}'.format(eval_metrics_depth[i], model_save_name))
                            checkpoint = {'global_step': global_step,
                                          'model': model.state_dict(),
                                          'optimizer': optimizer.state_dict(),
                                          'best_eval_measures_higher_better': best_eval_measures_higher_better,
                                          'best_eval_measures_lower_better': best_eval_measures_lower_better,
                                          'best_eval_steps': best_eval_steps
                                          }
                            torch.save(checkpoint, args.log_directory + '/' + args.model_name + model_save_name)
                    eval_summary_writer.flush()
                model.train()
                model_scene.train()
                block_print()
                enable_print()

            model_just_loaded = False
            global_step += 1

        epoch += 1


def main():
    if args.mode != 'train':
        print('bts_main.py is only for training. Use bts_test.py instead.')
        return -1
  
    model_filename = args.model_name + '.py'

    args_out_path = args.log_directory + '/' + args.model_name + '/' + sys.argv[1]

    if args.checkpoint_path == '':
        model_out_path = args.log_directory + '/' + args.model_name + '/' + model_filename
        aux_out_path = args.log_directory + '/' + args.model_name + '/.'
    else:
        loaded_model_dir = os.path.dirname(args.checkpoint_path)
        loaded_model_name = os.path.basename(loaded_model_dir)
        loaded_model_filename = loaded_model_name + '.py'

        model_out_path = args.log_directory + '/' + args.model_name + '/' + model_filename
        command = 'cp ' + loaded_model_dir + '/' + loaded_model_filename + ' ' + model_out_path
        os.system(command)

    torch.cuda.empty_cache()
    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if ngpus_per_node > 1 and not args.multiprocessing_distributed:
        print("This machine has more than 1 gpu. Please specify --multiprocessing_distributed, or set \'CUDA_VISIBLE_DEVICES=0\'")
        return -1

    if args.do_online_eval:
        print("You have specified --do_online_eval.")
        print("This will evaluate the model every eval_freq {} steps and save best models for individual eval metrics."
              .format(args.eval_freq))

    if args.multiprocessing_distributed:
        args.world_size = ngpus_per_node * args.world_size
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))

    else:
        main_worker(args.gpu, ngpus_per_node, args)


if __name__ == '__main__':
    main()
