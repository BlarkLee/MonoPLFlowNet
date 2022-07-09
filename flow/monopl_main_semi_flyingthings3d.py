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
from evaluation_monopl_flyingthings3d import evaluate


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
    print("args.arch", args.arch)
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
            model.load_state_dict(checkpoint['state_dict'], strict=True)
            min_epe3d = checkpoint["EPE3D"]
            logger.log("=> loaded checkpoint '{}' (start epoch {}, EPE3D {})"
                       .format(args.resume, checkpoint['epoch'], checkpoint['EPE3D']))
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

    # -------------------- optimizer --------------------
    optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, model.parameters()),
                                 lr=args.lr,
                                 weight_decay=0)
    if args.resume and (checkpoint is not None):
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("pre-optimizer loaded")

    if hasattr(args, 'reset_lr') and args.reset_lr:
        print('reset lr')
        reset_learning_rate(optimizer, args)

    # -------------------- main loop --------------------
    min_train_loss = None
    best_train_epoch = None
    best_val_epoch = None
    do_eval = True
    
    for epoch in range(args.start_epoch, args.epochs):
        old_lr = optimizer.param_groups[0]['lr']
        adjust_learning_rate(optimizer, epoch, args)

        lr = optimizer.param_groups[0]['lr']
        if old_lr != lr:
            print('Switch lr!')
        logger.log('lr: ' + str(optimizer.param_groups[0]['lr']))

        train_loss = train(train_loader, model, model_depth, criterion, optimizer, epoch, logger, gen_lattice)
        gc.collect()

        is_train_best = True if best_train_epoch is None else (train_loss < min_train_loss)
        if is_train_best:
            min_train_loss = train_loss
            best_train_epoch = epoch
        
        if do_eval:
            _, res_dict = evaluate(val_loader, model, model_depth, logger, args, gen_lattice)
            gc.collect()
            if epoch == 0:
                is_val_best = True
                best_val_epoch = epoch
                min_epe3d = res_dict["EPE3D"]
                logger.log("New min EPE!")
            else:
                is_val_best = (res_dict["EPE3D"] < min_epe3d)
                if is_val_best:
                    best_val_epoch = epoch
                    min_epe3d = res_dict["EPE3D"]
                    logger.log("New min EPE!") 
            
            
        is_best = is_val_best if do_eval else is_train_best
        save_checkpoint({
            'epoch': epoch + 1,  # next start epoch
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'EPE3D': res_dict["EPE3D"],
            'optimizer': optimizer.state_dict(),
        }, is_best, args.ckpt_dir)

    train_str = 'Best train loss: {:.5f} at epoch {:3d}'.format(min_train_loss, best_train_epoch)
    logger.log(train_str)

    if do_eval:
        val_str = 'Min val epe3d: {:.5f} at epoch {:3d}'.format(min_epe3d, best_val_epoch)
        logger.log(val_str)
        val_str_all = 'Epoch: {:3d} | EPE3D: {:.5f} | ACC3DS: {:.5f} | ACC3DR: {:.5f} | Outliers3D: {:.5f} | EPE2D: {:.5f} | ACC2D: {:.5f}'.format(res_dict["EPE3D"], res_dict["ACC3DS"], res_dict["ACC3DR"], res_dict["Outliers3D"], res_dict["EPE2D"], res_dict["ACC2D"],)

    logger.close()
    result_str = val_str if do_eval else train_str
    return result_str

    
def pixel2pc(depth, save_path=None, f=-1050., cx=479.5, cy=269.5):
    BASELINE = 1.0
    disparity = -1. * f * BASELINE / depth
    height, width = disparity.shape

    x = ((np.tile(np.arange(width, dtype=np.float32)[None, :], (height, 1)) - cx) * -1. / disparity)[:, :, None]
    y = ((np.tile(np.arange(height, dtype=np.float32)[:, None], (1, width)) - cy) * 1. / disparity)[:, :, None]
    pc = np.concatenate((-x, y, -depth[:, :, None]), axis=-1)

    if save_path is not None:
        np.save(save_path, pc)
    return pc

def pcl2pts(pcl, k, h, w, scale):
    if scale != 1:
        pcl_s= scale * pcl
        k_s = scale * k
        k_s[2,2] = 1
        h_max = scale * h
        w_max = scale * w
    else:
        pcl_s = pcl
        k_s = k
        h_max = h
        w_max = w
    temp = torch.matmul(k_s, pcl_s)
    temp[0, :] = temp[0, :]/(temp[2, :] + 1e-8)
    temp[1, :] = temp[1, :]/(temp[2, :]+ 1e-8)
    pts = temp[:2, :]
        
    x = temp[0, :].clone()
    y = temp[1, :].clone()
    x[x > w_max-1] = w_max-1
    y[y > h_max-1] = h_max-1
    x[x < 0] = 0
    y[y < 0] = 0
    pts[0, :] = y
    pts[1, :] = x
    pts = torch.round(pts).long()
    return pts

def down_feat_pcl(h, w, feat0_1, feat0_2, feat0_4, feat1_1, feat1_2, feat1_4, k, data):
        
    pcl0_1 = data[0]["pc1_barycentric_3d"].squeeze(0)#.cuda()
    pcl1_1 = data[0]["pc2_barycentric_3d"].squeeze(0)#.cuda()
    pcl0_2 = data[1]["pc1_barycentric_3d"].squeeze(0)#.cuda()
    pcl1_2 = data[1]["pc2_barycentric_3d"].squeeze(0)#.cuda()
    pcl0_4 = data[2]["pc1_barycentric_3d"].squeeze(0)#.cuda()
    pcl1_4 = data[2]["pc2_barycentric_3d"].squeeze(0)#.cuda()
        
    mask0_1 = pcl2pts(pcl0_1, k.cpu(), h, w, 1)
    mask1_1 = pcl2pts(pcl1_1, k.cpu(), h, w, 1)
    mask0_2 = pcl2pts(pcl0_2, k.cpu(), h, w, 0.5)
    mask1_2 = pcl2pts(pcl1_2, k.cpu(), h, w, 0.5)
    mask0_4 = pcl2pts(pcl0_4, k.cpu(), h, w, 0.25)
    mask1_4 = pcl2pts(pcl1_4, k.cpu(), h, w, 0.25)
    
    feat0_1 = torch.unsqueeze(feat0_1.cpu()[mask0_1[0, :], mask0_1[1, :], :].permute(1,0), 0)
    feat1_1 = torch.unsqueeze(feat1_1.cpu()[mask1_1[0, :], mask1_1[1, :], :].permute(1,0), 0)
    feat0_2 = torch.unsqueeze(feat0_2.cpu()[mask0_2[0, :], mask0_2[1, :], :].permute(1,0), 0)
    feat1_2 = torch.unsqueeze(feat1_2.cpu()[mask1_2[0, :], mask1_2[1, :], :].permute(1,0), 0)
    feat0_4 = torch.unsqueeze(feat0_4.cpu()[mask0_4[0, :], mask0_4[1, :], :].permute(1,0), 0)
    feat1_4 = torch.unsqueeze(feat1_4.cpu()[mask1_4[0, :], mask1_4[1, :], :].permute(1,0), 0)

    return feat0_1, feat0_2, feat0_4, feat1_1, feat1_2, feat1_4

def prepare_feat(all_feat_1, all_feat_2, generated_data, fu, fv, cx, cy):
    _, feat1_4, feat1_2, feat1_1 = all_feat_1
    _, feat2_4, feat2_2, feat2_1 = all_feat_2
    feat1_1 = feat1_1[:, :, :540, :]
    feat1_2 = feat1_2[:, :, :270, :]
    feat1_4 = feat1_4[:, :, :135, :]
    feat2_1 = feat2_1[:, :, :540, :]
    feat2_2 = feat2_2[:, :, :270, :]
    feat2_4 = feat2_4[:, :, :135, :]
    feat1_1 = torch.squeeze(feat1_1, 0).permute(1,2,0)
    feat2_1 = torch.squeeze(feat2_1, 0).permute(1,2,0)
    feat1_2 = torch.squeeze(feat1_2, 0).permute(1,2,0)
    feat2_2 = torch.squeeze(feat2_2, 0).permute(1,2,0)
    feat1_4 = torch.squeeze(feat1_4, 0).permute(1,2,0)
    feat2_4 = torch.squeeze(feat2_4, 0).permute(1,2,0)
    k = torch.tensor([[fu, 0, cx], [0, fv, cy], [0, 0, 1]])
    h = 540
    w = 960
    feat1_1, feat1_2, feat1_4, feat2_1, feat2_2, feat2_4 = down_feat_pcl(h, w, feat1_1, feat1_2, feat1_4, feat2_1, feat2_2, feat2_4, k, generated_data)
    return feat1_1.cuda(), feat1_2.cuda(), feat1_4.cuda(), feat2_1.cuda(), feat2_2.cuda(), feat2_4.cuda()

def train(train_loader, model, model_depth, criterion, optimizer, epoch, logger, gen_lattice):
    epe3d_losses = AverageMeter()
    cham_losses = AverageMeter()
    total_losses = AverageMeter()
    cham_loss_avg = 0
    epe3d_loss_avg = 0
    count_cham = 0
    count_epe3d = 0

    model.train()
    fu=-1050.
    fv=-1050.
    cx=479.5
    cy=269.5
    for i, (pc1_gt, pc2_gt, sf_gt, path, depth1_gt, depth2_gt, image1, image2, pc_mask, mask1, mask2, generated_data, generated_data_b) in enumerate(train_loader):
        if ((generated_data[0]['pc1_hash_cnt']<generated_data[1]['pc1_hash_cnt']) or (generated_data[0]['pc2_hash_cnt']<generated_data[1]['pc2_hash_cnt'])):
            print("forward max cnt low level larger than high level, skip!")
            continue
        if ((generated_data_b[0]['pc1_hash_cnt']<generated_data_b[1]['pc1_hash_cnt']) or (generated_data_b[0]['pc2_hash_cnt']<generated_data_b[1]['pc2_hash_cnt'])):
            print("backward max cnt low level larger than high level, skip!")
            continue
        try:
            cur_sf_gt = sf_gt.cuda(non_blocking=True)
            
            with torch.no_grad():
                image1 = image1.permute(0,3,1,2).cuda(non_blocking=True)
                image2 = image2.permute(0,3,1,2).cuda(non_blocking=True)
                pc1_gt = pc1_gt.cuda()
                pc2_gt = pc2_gt.cuda()
                sf_gt = sf_gt.cuda()
                focal = float(-1050)
                _, _, _, _, depth1_est, all_feat_1 = model_depth(image1, focal)
                _, _, _, _, depth2_est, all_feat_2 = model_depth(image2, focal)
                depth1_est = torch.squeeze(depth1_est, 0)[:, :540, :]
                depth2_est = torch.squeeze(depth2_est, 0)[:, :540, :]
                pc1_est = -pixel2pc(torch.squeeze(depth1_est,0).detach().cpu().numpy())
                pc2_est = -pixel2pc(torch.squeeze(depth2_est,0).detach().cpu().numpy())
                transformed_pc1_est = pc1_est[torch.squeeze(pc_mask,0)][torch.squeeze(mask1, 0)]
                transformed_pc2_est = pc2_est[torch.squeeze(pc_mask,0)][torch.squeeze(mask2, 0)]

            
            #0930 night from here
            feat0_1, feat0_2, feat0_4, feat1_1, feat1_2, feat1_4 = prepare_feat(all_feat_1, all_feat_2, generated_data, fu, fv, cx, cy)
            feat0_1_b, feat0_2_b, feat0_4_b, feat1_1_b, feat1_2_b, feat1_4_b = prepare_feat(all_feat_2, all_feat_1, generated_data_b, fu, fv, cx, cy)
            output = model(feat0_1, feat0_2, feat0_4, feat1_1, feat1_2, feat1_4, generated_data)
            output_b = model(feat0_1_b, feat0_2_b, feat0_4_b, feat1_1_b, feat1_2_b, feat1_4_b, generated_data_b)
            
            pc1_est = torch.unsqueeze(torch.from_numpy(transformed_pc1_est).permute(1,0), 0).cuda()
            pc2_est = torch.unsqueeze(torch.from_numpy(transformed_pc2_est).permute(1,0), 0).cuda()
            cham_loss = chamfer_loss(pc1_gt, pc2_gt, output) + chamfer_loss(pc2_gt, pc1_gt, output_b)
            cham_loss = cham_loss/8192
            
            loss_ratio_thresh = 5 * 0.99 ** (epoch+1)
            
            if i>0:
                if cham_loss>loss_ratio_thresh*cham_loss_avg:
                    print("cham_loss bad sample, skip!!!", cham_loss)
                    del pc1_gt, pc2_gt, sf_gt, path, depth1_gt, depth2_gt, image1, image2, pc_mask, mask1, mask2, generated_data, generated_data_b, depth1_est, all_feat_1, depth2_est, all_feat_2, pc1_est, pc2_est, transformed_pc1_est, transformed_pc2_est, feat0_1, feat0_2, feat0_4, feat1_1, feat1_2, feat1_4, feat0_1_b, feat0_2_b, feat0_4_b, feat1_1_b, feat1_2_b, feat1_4_b, output, output_b, cham_loss
                    torch.cuda.empty_cache()
                    gc.collect()
                    continue
            count_cham += 1
            cham_loss_avg = (cham_loss_avg * (count_cham-1) + cham_loss)/count_cham
            
            epe3d_loss = criterion(input=output, target=cur_sf_gt).mean()
            
            if i>0:
                if epe3d_loss>loss_ratio_thresh*epe3d_loss_avg:
                    print("epe3d_loss bad sample, skip!!!", epe3d_loss)
                    del pc1_gt, pc2_gt, sf_gt, path, depth1_gt, depth2_gt, image1, image2, pc_mask, mask1, mask2, generated_data, generated_data_b, depth1_est, all_feat_1, depth2_est, all_feat_2, pc1_est, pc2_est, transformed_pc1_est, transformed_pc2_est, feat0_1, feat0_2, feat0_4, feat1_1, feat1_2, feat1_4, feat0_1_b, feat0_2_b, feat0_4_b, feat1_1_b, feat1_2_b, feat1_4_b, output, output_b, cham_loss, epe3d_loss
                    torch.cuda.empty_cache()
                    gc.collect()
                    continue
            count_epe3d += 1
            epe3d_loss_avg = (epe3d_loss_avg * (count_epe3d-1) + epe3d_loss)/count_epe3d
            max_loss = torch.max(cham_loss, epe3d_loss)
            cham_weight = max_loss/cham_loss
            epe3d_weight = max_loss/epe3d_loss
            total_loss = cham_weight * cham_loss + epe3d_weight * epe3d_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            epe3d_losses.update(epe3d_loss.item(), sf_gt.size(0))  # batch size can only be 1 for now
            cham_losses.update(cham_loss.item(), sf_gt.size(0))
            total_losses.update(total_loss.item(), sf_gt.size(0))
            

            if i % args.print_freq == 0:
                logger.log('Epoch: [{0}][{1}/{2}]\t'
                           'EPE3D Loss {epe3d_losses_.val:.4f} ({epe3d_losses_.avg:.4f})\t'
                           'Cham Loss {cham_losses_.val:.4f} ({cham_losses_.avg:.4f})\t'
                           'Total Loss {total_losses_.val:.4f} ({total_losses_.avg:.4f})'
                            .format(
                            epoch + 1, i + 1, len(train_loader),
                            epe3d_losses_=epe3d_losses,
                            cham_losses_=cham_losses,
                            total_losses_=total_losses
                            ), end='')
                logger.log('')
                
            del pc1_gt, pc2_gt, sf_gt, path, depth1_gt, depth2_gt, image1, image2, pc_mask, mask1, mask2, generated_data, generated_data_b, depth1_est, all_feat_1, depth2_est, all_feat_2, pc1_est, pc2_est, transformed_pc1_est, transformed_pc2_est, feat0_1, feat0_2, feat0_4, feat1_1, feat1_2, feat1_4, feat0_1_b, feat0_2_b, feat0_4_b, feat1_1_b, feat1_2_b, feat1_4_b, output, output_b, cham_loss, epe3d_loss, max_loss, total_loss
            torch.cuda.empty_cache()
            gc.collect()
                
        
        except RuntimeError as ex:
            logger.log("in TRAIN, RuntimeError " + repr(ex))
            logger.log("batch idx: " + str(i) + ' path: ' + path[0])
            traceback.print_tb(ex.__traceback__, file=logger.out_fd)
            traceback.print_tb(ex.__traceback__)
            
            if "CUDA error: out of memory" in str(ex) or "cuda runtime error" in str(ex):
                logger.log("out of memory, continue")

                del pc1_gt, pc2_gt, sf_gt, path, depth1_gt, depth2_gt, image1, image2, pc_mask, mask1, mask2, generated_data, generated_data_b
                if 'output' in locals():
                    del output
                torch.cuda.empty_cache()
                gc.collect()
            else:
                logger.log("out of memory, continue")
                del pc1_gt, pc2_gt, sf_gt, path, depth1_gt, depth2_gt, image1, image2, pc_mask, mask1, mask2, generated_data, generated_data_b
                if 'depth1_est' in locals():
                    del depth1_est
                if 'depth2_est' in locals():
                    del depth2_est
                if 'all_feat_1' in locals():
                    del all_feat_1
                if 'all_feat_2' in locals():
                    del all_feat_2
                if 'pc1_est' in locals():
                    del pc1_est
                if 'pc2_est' in locals():
                    del pc2_est
                if 'transformed_pc1_est' in locals():
                    del transformed_pc1_est
                if 'transformed_pc2_est' in locals():
                    del transformed_pc2_est
                if 'feat0_1' in locals():
                    del feat0_1
                if 'feat0_2' in locals():
                    del feat0_2
                if 'feat0_4' in locals():
                    del feat0_4
                if 'feat1_1' in locals():
                    del feat1_1
                if 'feat1_2' in locals():
                    del feat1_2
                if 'feat1_4' in locals():
                    del feat1_4
                if 'feat0_1_b' in locals():
                    del feat0_1_b
                if 'feat0_2_b' in locals():
                    del feat0_2_b
                if 'feat0_4_b' in locals():
                    del feat0_4_b
                if 'feat1_1_b' in locals():
                    del feat1_1_b
                if 'feat1_2_b' in locals():
                    del feat1_2_b
                if 'feat1_4_b' in locals():
                    del feat1_4_b
                if 'output' in locals():
                    del output
                if 'output_b' in locals():
                    del output_b
                torch.cuda.empty_cache()
                gc.collect()

    logger.log(
        ' * Train EPE3D {epe3d_losses_.avg:.4f}'.format(epe3d_losses_=epe3d_losses))
    return epe3d_losses.avg


def validate(val_loader, model, model_depth, criterion, logger, gen_lattice):
    epe3d_losses = AverageMeter()

    model.eval()
    fu=-1050.
    fv=-1050.
    cx=479.5
    cy=269.5
    with torch.no_grad():
        for i, (sf_gt, path, depth1_gt, depth2_gt, image1, image2, pc_mask, mask1, mask2) in enumerate(val_loader):
            try:
                cur_sf = sf.cuda(non_blocking=True)
                output = model(pc1, pc2, generated_data)
                epe3d_loss = criterion(input=output, target=cur_sf)

                epe3d_losses.update(epe3d_loss.mean().item())

                if i % args.print_freq == 0:
                    logger.log('Test: [{0}/{1}]\t'
                               'EPE3D loss {epe3d_losses_.val:.4f} ({epe3d_losses_.avg:.4f})'
                               .format(i + 1, len(val_loader),
                                       epe3d_losses_=epe3d_losses))

            except RuntimeError as ex:
                logger.log("in VAL, RuntimeError " + repr(ex))
                traceback.print_tb(ex.__traceback__, file=logger.out_fd)
                traceback.print_tb(ex.__traceback__)

                if "CUDA error: out of memory" in str(ex) or "cuda runtime error" in str(ex):
                    logger.log("out of memory, continue")
                    del pc1, pc2, sf, generated_data
                    torch.cuda.empty_cache()
                    gc.collect()
                    print('remained objects after OOM crash')
                else:
                    sys.exit(1)

    logger.log(' * EPE3D loss {epe3d_loss_.avg:.4f}'.format(epe3d_loss_=epe3d_losses))
    return epe3d_losses.avg


if __name__ == '__main__':
    main()
