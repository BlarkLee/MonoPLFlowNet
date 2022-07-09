import os, sys
import os.path as osp
import numpy as np
import pickle

import torch
import torch.optim
import torch.utils.data

from main_utils import *
from utils import geometry
from evaluation_utils_relax import evaluate_2d, evaluate_3d
import cv2

TOTAL_NUM_SAMPLES = 142 #10

def pixel2pc(disparity, save_path=None, f=721.5, cx=609.5, cy=172.8):
    BASELINE = 0.54
    height, width = disparity.shape
    depth = -1. * f * BASELINE / disparity

    x = ((np.tile(np.arange(width, dtype=np.float32)[None, :], (height, 1)) - cx) * -1. / disparity)[:, :, None]
    y = ((np.tile(np.arange(height, dtype=np.float32)[:, None], (1, width)) - cy) * 1. / disparity)[:, :, None]
    pc = np.concatenate((-x, y, depth[:, :, None]), axis=-1)

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
    feat1_1 = feat1_1
    feat1_2 = feat1_2
    feat1_4 = feat1_4
    feat2_1 = feat2_1
    feat2_2 = feat2_2
    feat2_4 = feat2_4
    feat1_1 = torch.squeeze(feat1_1, 0).permute(1,2,0)
    feat2_1 = torch.squeeze(feat2_1, 0).permute(1,2,0)
    feat1_2 = torch.squeeze(feat1_2, 0).permute(1,2,0)
    feat2_2 = torch.squeeze(feat2_2, 0).permute(1,2,0)
    feat1_4 = torch.squeeze(feat1_4, 0).permute(1,2,0)
    feat2_4 = torch.squeeze(feat2_4, 0).permute(1,2,0)
    k = torch.tensor([[fu, 0, cx], [0, fv, cy], [0, 0, 1]])
    h = 352
    w = 1216
    feat1_1, feat1_2, feat1_4, feat2_1, feat2_2, feat2_4 = down_feat_pcl(h, w, feat1_1, feat1_2, feat1_4, feat2_1, feat2_2, feat2_4, k, generated_data)
    return feat1_1.cuda(), feat1_2.cuda(), feat1_4.cuda(), feat2_1.cuda(), feat2_2.cuda(), feat2_4.cuda()


def evaluate(val_loader, model, model_depth, logger, args, gen_lattice):
    save_idx = 0
    num_sampled_batches = TOTAL_NUM_SAMPLES // args.batch_size

    # sample data for visualization
    if TOTAL_NUM_SAMPLES == 0:
        sampled_batch_indices = []
    else:
        if len(val_loader) > num_sampled_batches:
            print('num_sampled_batches', num_sampled_batches)
            print('len(val_loader)', len(val_loader))

            sep = len(val_loader) // num_sampled_batches
            sampled_batch_indices = list(range(len(val_loader)))[::sep]
        else:
            sampled_batch_indices = range(len(val_loader))

    save_dir = osp.join(args.ckpt_dir, 'visu_' + osp.split(args.ckpt_dir)[-1])
    os.makedirs(save_dir, exist_ok=True)
    path_list = []
    epe3d_list = []

    epe3ds = AverageMeter()
    acc3d_stricts = AverageMeter()
    acc3d_relaxs = AverageMeter()
    outliers = AverageMeter()
    # 2D
    epe2ds = AverageMeter()
    acc2ds = AverageMeter()

    model.eval()
    fu=721.5
    fv=721.5
    cx=609.5
    cy=172.8
    with torch.no_grad():
        for i, items in enumerate(val_loader):
            pc1_gt, pc2_gt, sf_gt, path, image1, image2, pc_mask, mask1, mask2, generated_data, _, _ = items
            
            image1 = image1.permute(0,3,1,2)
            image2 = image2.permute(0,3,1,2)
            K = torch.Tensor([[fu, 0, cx], [0, fv, cy], [0, 0, 1]]).view(1,3,3)
            K_flip = torch.Tensor([[fu, 0, float(image1.shape[2]-cx)], [0, fv, cy], [0, 0, 1]]).view(1,3,3)
            input_dict = {}
            input_dict['aug_size'] = torch.Tensor([[float(image1.shape[2]), float(image1.shape[3])]])
            input_dict['input_l1_aug'] = image1
            input_dict['input_l2_aug'] = image2
            input_dict['input_k_l1_aug'] = K
            input_dict['input_k_l2_aug'] = K
            input_dict['input_k_l1_flip_aug'] = K_flip
            input_dict['input_k_l2_flip_aug'] = K_flip
            haha = None
            output, disp1_est, disp2_est = model(input_dict, haha)
            depth1_est = 721.5 * 0.54 / disp1_est
            depth1_est_np = torch.squeeze(torch.squeeze(depth1_est, 0), 0).detach().cpu().numpy()
            pc1_est = -pixel2pc(torch.squeeze(torch.squeeze(disp1_est,0), 0).detach().cpu().numpy())[:540, :, :]
            pc2_est = -pixel2pc(torch.squeeze(torch.squeeze(disp2_est,0), 0).detach().cpu().numpy())[:540, :, :]
            output = torch.squeeze(output, 0).permute(1,2,0)[:540, :, :]
            pc1_est = pc1_est[pc_mask[0]][mask1[0]]
            pc2_est = pc2_est[pc_mask[0]][mask2[0]]
            output = torch.unsqueeze(output[pc_mask[0]][mask1[0]], 0)
            ratio1_s=0
            ratio2_s=0
            for j in range(8192):
                ratio1 = (pc1_gt[0, 2, j]/pc1_est[j, 2]).detach().cpu().numpy()
                ratio2 = (pc2_gt[0, 2, j]/pc2_est[j, 2]).detach().cpu().numpy()
                ratio1_s += ratio1
                ratio2_s += ratio2
            ratio1 = ratio1_s/8192
            ratio2 = ratio2_s/8192
            pc1_est *= ratio1
            depth1_est_np *= ratio1
            pc1_np = pc1_gt.numpy()
            pc1_np = pc1_np.transpose((0,2,1))
            pc2_np = pc2_gt.numpy()
            pc2_np = pc2_np.transpose((0,2,1))
            pc1_est_np = np.expand_dims(pc1_est, axis=0)
            pc2_est_np = np.expand_dims(pc2_est, axis=0)
            sf_np = sf_gt.numpy()
            sf_np = sf_np.transpose((0,2,1))
            output_np = output.cpu().numpy()

            EPE3D, acc3d_strict, acc3d_relax, outlier = evaluate_3d(output_np, sf_np)

            epe3ds.update(EPE3D)
            acc3d_stricts.update(acc3d_strict)
            acc3d_relaxs.update(acc3d_relax)
            outliers.update(outlier)

            # 2D evaluation metrics
            flow_pred, flow_gt = geometry.get_batch_2d_flow_monopl(pc1_np,
                                                            pc1_np+sf_np,
                                                            pc1_est_np,
                                                            pc1_est_np+output_np,
                                                            path)
            EPE2D, acc2d = evaluate_2d(flow_pred, flow_gt)

            epe2ds.update(EPE2D)
            acc2ds.update(acc2d)

            if i % args.print_freq == 0:
                logger.log('Test: [{0}/{1}]\t'
                           'EPE3D {epe3d_.val:.4f} ({epe3d_.avg:.4f})\t'
                           'ACC3DS {acc3d_s.val:.4f} ({acc3d_s.avg:.4f})\t'
                           'ACC3DR {acc3d_r.val:.4f} ({acc3d_r.avg:.4f})\t'
                           'Outliers3D {outlier_.val:.4f} ({outlier_.avg:.4f})\t'
                           'EPE2D {epe2d_.val:.4f} ({epe2d_.avg:.4f})\t'
                           'ACC2D {acc2d_.val:.4f} ({acc2d_.avg:.4f})'
                           .format(i + 1, len(val_loader),
                                   epe3d_=epe3ds,
                                   acc3d_s=acc3d_stricts,
                                   acc3d_r=acc3d_relaxs,
                                   outlier_=outliers,
                                   epe2d_=epe2ds,
                                   acc2d_=acc2ds,
                                   ))

            if i in sampled_batch_indices:
                np.save(osp.join(save_dir, 'pc1_' + str(save_idx) + '.npy'), pc1_est_np)
                np.save(osp.join(save_dir, 'sf_' + str(save_idx) + '.npy'), sf_np)
                np.save(osp.join(save_dir, 'output_' + str(save_idx) + '.npy'), output_np)
                depth1_est_save = np.asarray(depth1_est_np, np.uint16)
                cv2.imwrite(osp.join(save_dir, 'depth_' + str(save_idx) + '.png'), depth1_est_save)
                epe3d_list.append(EPE3D)
                path_list.extend(path)
                save_idx += 1
            del pc1_est, pc2_est, sf_gt, generated_data, image1, image2, pc_mask, mask1, mask2
    
    if len(path_list) > 0:
        np.save(osp.join(save_dir, 'epe3d_per_frame.npy'), np.array(epe3d_list))
        with open(osp.join(save_dir, 'sample_path_list.pickle'), 'wb') as fd:
            pickle.dump(path_list, fd)

    res_str = (' * EPE3D {epe3d_.avg:.4f}\t'
               'ACC3DS {acc3d_s.avg:.4f}\t'
               'ACC3DR {acc3d_r.avg:.4f}\t'
               'Outliers3D {outlier_.avg:.4f}\t'
               'EPE2D {epe2d_.avg:.4f}\t'
               'ACC2D {acc2d_.avg:.4f}'
               .format(
                       epe3d_=epe3ds,
                       acc3d_s=acc3d_stricts,
                       acc3d_r=acc3d_relaxs,
                       outlier_=outliers,
                       epe2d_=epe2ds,
                       acc2d_=acc2ds,
                       ))
    logger.log(res_str)
    
    res_dict = {
    "EPE3D":EPE3D,
    "ACC3DS":acc3d_strict,
    "ACC3DR":acc3d_relax,
    "outlier":outlier,
    "epe2d":EPE2D,
    "acc2d":acc2d
    }
    
    return res_str, res_dict
