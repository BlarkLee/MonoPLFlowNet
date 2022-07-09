import torch
import torch.nn as nn
import torch.nn.functional as torch_nn_func
import math
import numpy as np

from collections import namedtuple

from .PL_model.models.bilateralNN_double import BilateralConvFlex
from .PL_model.models.bnn_flow_upsample import BilateralCorrelationFlex
from .PL_model.models.bnn_flow import BilateralCorrelationFlex as BilateralCorrelationFlex_no_upsample
from .PL_model.models.module_utils import Conv1dReLU

__all__ = ['PLSceneNet_shallow_lev1_lev2']

# This sets the batch norm layers in pytorch as if {'is_training': False, 'scale': True} in tensorflow
def bn_init_as_tf(m):
    if isinstance(m, nn.BatchNorm2d):
        m.track_running_stats = True  # These two lines enable using stats (moving mean and var) loaded from pretrained model
        m.eval()                      # or zero mean and variance of one if the batch norm layer has no pretrained values
        m.affine = True
        m.requires_grad = True


def weights_init_xavier(m):
    if isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)
            

class silog_loss(nn.Module):
    def __init__(self, variance_focus):
        super(silog_loss, self).__init__()
        self.variance_focus = variance_focus

    def forward(self, depth_est, depth_gt, mask):
        d = torch.log(depth_est[mask]) - torch.log(depth_gt[mask])
        return torch.sqrt((d ** 2).mean() - self.variance_focus * (d.mean() ** 2)) * 10.0


class atrous_conv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation, apply_bn_first=True):
        super(atrous_conv, self).__init__()
        self.atrous_conv = torch.nn.Sequential()
        if apply_bn_first:
            self.atrous_conv.add_module('first_bn', nn.BatchNorm2d(in_channels, momentum=0.01, affine=True, track_running_stats=True, eps=1.1e-5))
        
        self.atrous_conv.add_module('aconv_sequence', nn.Sequential(nn.ReLU(),
                                                                    nn.Conv2d(in_channels=in_channels, out_channels=out_channels*2, bias=False, kernel_size=1, stride=1, padding=0),
                                                                    nn.BatchNorm2d(out_channels*2, momentum=0.01, affine=True, track_running_stats=True),
                                                                    nn.ReLU(),
                                                                    nn.Conv2d(in_channels=out_channels * 2, out_channels=out_channels, bias=False, kernel_size=3, stride=1,
                                                                              padding=(dilation, dilation), dilation=dilation)))

    def forward(self, x):
        return self.atrous_conv.forward(x)
    

class upconv(nn.Module):
    def __init__(self, in_channels, out_channels, ratio=2):
        super(upconv, self).__init__()
        self.elu = nn.ELU()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, bias=False, kernel_size=3, stride=1, padding=1)
        self.ratio = ratio
        
    def forward(self, x):
        up_x = torch_nn_func.interpolate(x, scale_factor=self.ratio, mode='nearest')
        out = self.conv(up_x)
        out = self.elu(out)
        return out


class reduction_1x1(nn.Sequential):
    def __init__(self, num_in_filters, num_out_filters, max_depth, is_final=False):
        super(reduction_1x1, self).__init__()        
        self.max_depth = max_depth
        self.is_final = is_final
        self.sigmoid = nn.Sigmoid()
        self.reduc = torch.nn.Sequential()
        
        while num_out_filters >= 4:
            if num_out_filters < 8:
                if self.is_final:
                    self.reduc.add_module('final', torch.nn.Sequential(nn.Conv2d(num_in_filters, out_channels=1, bias=False,
                                                                                 kernel_size=1, stride=1, padding=0),
                                                                       nn.Sigmoid()))
                else:
                    self.reduc.add_module('plane_params', torch.nn.Conv2d(num_in_filters, out_channels=3, bias=False,
                                                                          kernel_size=1, stride=1, padding=0))
                break
            else:
                self.reduc.add_module('inter_{}_{}'.format(num_in_filters, num_out_filters),
                                      torch.nn.Sequential(nn.Conv2d(in_channels=num_in_filters, out_channels=num_out_filters,
                                                                    bias=False, kernel_size=1, stride=1, padding=0),
                                                          nn.ELU()))

            num_in_filters = num_out_filters
            num_out_filters = num_out_filters // 2
    
    def forward(self, net):
        net = self.reduc.forward(net)
        if not self.is_final:
            theta = self.sigmoid(net[:, 0, :, :]) * math.pi / 3
            phi = self.sigmoid(net[:, 1, :, :]) * math.pi * 2
            dist = self.sigmoid(net[:, 2, :, :]) * self.max_depth
            n1 = torch.mul(torch.sin(theta), torch.cos(phi)).unsqueeze(1)
            n2 = torch.mul(torch.sin(theta), torch.sin(phi)).unsqueeze(1)
            n3 = torch.cos(theta).unsqueeze(1)
            n4 = dist.unsqueeze(1)
            net = torch.cat([n1, n2, n3, n4], dim=1)
        
        return net

class local_planar_guidance(nn.Module):
    def __init__(self, upratio):
        super(local_planar_guidance, self).__init__()
        self.upratio = upratio
        self.u = torch.arange(self.upratio).reshape([1, 1, self.upratio]).float()
        self.v = torch.arange(int(self.upratio)).reshape([1, self.upratio, 1]).float()
        self.upratio = float(upratio)

    def forward(self, plane_eq, focal):
        plane_eq_expanded = torch.repeat_interleave(plane_eq, int(self.upratio), 2)
        plane_eq_expanded = torch.repeat_interleave(plane_eq_expanded, int(self.upratio), 3)
        n1 = plane_eq_expanded[:, 0, :, :]
        n2 = plane_eq_expanded[:, 1, :, :]
        n3 = plane_eq_expanded[:, 2, :, :]
        n4 = plane_eq_expanded[:, 3, :, :]
        
        u = self.u.repeat(plane_eq.size(0), plane_eq.size(2) * int(self.upratio), plane_eq.size(3)).cuda()
        u = (u - (self.upratio - 1) * 0.5) / self.upratio
        
        v = self.v.repeat(plane_eq.size(0), plane_eq.size(2), plane_eq.size(3) * int(self.upratio)).cuda()
        v = (v - (self.upratio - 1) * 0.5) / self.upratio

        return n4 / (n1 * u + n2 * v + n3)

class PLSceneNet_shallow_lev1_lev2(nn.Module):
    def __init__(self, args):
        super(PLSceneNet_shallow_lev1_lev2, self).__init__()
        
        self.scales_filter_map = [[1., 1, 1, 1],
                     [0.5, 1, 1, 1],
                     [0.25, 1, 1, 1],
                     [0.125, 1, 1, 1]]
        self.evaluate = False
        self.chunk_size = -1 if self.evaluate else 1024 * 1024 * 25
        self.last_relu = False
        self.bcn_use_bias = True
        self.use_leaky = True
        self.bcn_use_norm = True
        self.dim = 3
        self.DEVICE = 'cuda'
        
        conv_module = Conv1dReLU
        
        
        self.conv4_embed = nn.Sequential(
            conv_module(32, 64, use_leaky=self.use_leaky))
        
        self.conv2_embed = nn.Sequential(
            conv_module(32, 128, use_leaky=self.use_leaky))
        
        self.conv1_embed = nn.Sequential(
            conv_module(32, 256, use_leaky=self.use_leaky))
        
        self.conv4_scene = conv_module(512, 512, use_leaky=self.use_leaky)
        self.conv2_scene = conv_module(512, 256, use_leaky=self.use_leaky)
        self.conv1_scene = nn.Conv1d(256, 3, kernel_size=1)

            
        self.up_2_to_1 = BilateralConvFlex(self.dim, self.scales_filter_map[0][2],
                                      128 + self.dim + 1, [256, 256],
                                      self.DEVICE,
                                      use_bias=self.bcn_use_bias,
                                      use_leaky=self.use_leaky,
                                      use_norm=self.bcn_use_norm,
                                      do_splat=False,
                                      do_slice=True,
                                      last_relu=self.last_relu,
                                      chunk_size=self.chunk_size)
        
        self.up_1_to_1 = BilateralConvFlex(self.dim, self.scales_filter_map[0][1],
                                      256 + 256 + self.dim + 1, [512, 512],
                                      self.DEVICE,
                                      use_bias=self.bcn_use_bias,
                                      use_leaky=self.use_leaky,
                                      use_norm=self.bcn_use_norm,
                                      do_splat=False,
                                      do_slice=True,
                                      last_relu=self.last_relu,
                                      chunk_size=self.chunk_size)
        
        

        self.coor_2_to_1 = BilateralCorrelationFlex(self.dim,
                                              self.scales_filter_map[1][2], self.scales_filter_map[1][3],
                                              256, [32, 32], [256, 256],
                                              self.DEVICE,
                                              use_bias=self.bcn_use_bias,
                                              use_leaky=self.use_leaky,
                                              use_norm=self.bcn_use_norm,
                                              prev_corr_dim=0,
                                              last_relu=self.last_relu,
                                              chunk_size=self.chunk_size)
        
        self.coor_1_to_1 = BilateralCorrelationFlex(self.dim,
                                              self.scales_filter_map[0][2], self.scales_filter_map[0][3],
                                              512, [32, 32], [512, 512],
                                              self.DEVICE,
                                              use_bias=self.bcn_use_bias,
                                              use_leaky=self.use_leaky,
                                              use_norm=self.bcn_use_norm,
                                              prev_corr_dim=256,
                                              last_relu=self.last_relu,
                                              chunk_size=self.chunk_size)
        

    
    def forward(self, feat0_1, feat0_2, feat0_4, feat1_1, feat1_2, feat1_4, generated_data):  
        ## Embed, Upsample and correlation

        ### level 2 to level 1
        feat0_2 = self.conv2_embed(feat0_2)
        pcl0_2, pcl0_2_to_1 = self.up_2_to_1(torch.cat((generated_data[2]['pc1_el_minus_gr'], feat0_2), dim=1),
                               in_barycentric=None,
                               in_lattice_offset=None,
                               blur_neighbors=generated_data[1]['pc1_blur_neighbors'],
                               out_barycentric=generated_data[2]['pc1_barycentric'],
                               out_lattice_offset=generated_data[2]['pc1_lattice_offset'], 
                               out_barycentric_upsample=generated_data[1]['pc1_barycentric'], 
                               out_lattice_offset_upsample=generated_data[1]['pc1_lattice_offset'],
                              )

        feat1_2 = self.conv2_embed(feat1_2)
        pcl1_2, pcl1_2_to_1 = self.up_2_to_1(torch.cat((generated_data[2]['pc2_el_minus_gr'], feat1_2), dim=1),
                               in_barycentric=None,
                               in_lattice_offset=None,
                               blur_neighbors=generated_data[1]['pc2_blur_neighbors'],
                               out_barycentric=generated_data[2]['pc2_barycentric'],
                               out_lattice_offset=generated_data[2]['pc2_lattice_offset'],
                               out_barycentric_upsample=generated_data[1]['pc2_barycentric'],
                               out_lattice_offset_upsample=generated_data[1]['pc2_lattice_offset']
                              )

        pcl_corr_2_to_1 = self.coor_2_to_1(pcl0_2, pcl1_2, prev_corr_feat=None,
                               barycentric1=generated_data[2]['pc1_barycentric'],
                               lattice_offset1=generated_data[2]['pc1_lattice_offset'],
                               out_barycentric=generated_data[1]['pc1_barycentric'],
                               out_lattice_offset=generated_data[1]['pc1_lattice_offset'],
                               pc1_corr_indices=generated_data[1]['pc1_corr_indices'],
                               pc2_corr_indices=generated_data[1]['pc2_corr_indices'],
                               max_hash_cnt1=generated_data[1]['pc1_hash_cnt'].item(),
                               max_hash_cnt2=generated_data[1]['pc2_hash_cnt'].item(),
                              )
        
        ### level 1 to level 1
        feat0_1 = self.conv1_embed(feat0_1)
        pcl0_1 = self.up_1_to_1(torch.cat((generated_data[1]['pc1_el_minus_gr'], pcl0_2_to_1, feat0_1), dim=1),
                               in_barycentric=None,
                               in_lattice_offset=None,
                               blur_neighbors=generated_data[0]['pc1_blur_neighbors'],
                               out_barycentric=generated_data[1]['pc1_barycentric'],
                               out_lattice_offset=generated_data[1]['pc1_lattice_offset']
                              )
        
        feat1_1 = self.conv1_embed(feat1_1)
        pcl1_1 = self.up_1_to_1(torch.cat((generated_data[1]['pc2_el_minus_gr'], pcl1_2_to_1, feat1_1), dim=1),
                               in_barycentric=None,
                               in_lattice_offset=None,
                               blur_neighbors=generated_data[0]['pc2_blur_neighbors'],
                               out_barycentric=generated_data[1]['pc2_barycentric'],
                               out_lattice_offset=generated_data[1]['pc2_lattice_offset'],
                              )
        
        pcl_corr_1_to_1 = self.coor_1_to_1(pcl0_1, pcl1_1, pcl_corr_2_to_1,
                               barycentric1=generated_data[1]['pc1_barycentric'],
                               lattice_offset1=generated_data[1]['pc1_lattice_offset'],
                               out_barycentric=generated_data[0]['pc1_barycentric'],
                               out_lattice_offset=generated_data[0]['pc1_lattice_offset'],
                               pc1_corr_indices=generated_data[0]['pc1_corr_indices'],
                               pc2_corr_indices=generated_data[0]['pc2_corr_indices'],
                               max_hash_cnt1=generated_data[0]['pc1_hash_cnt'].item(),
                               max_hash_cnt2=generated_data[0]['pc2_hash_cnt'].item(),
                              )


        # tail
        res = self.conv4_scene(pcl_corr_1_to_1)
        res = self.conv2_scene(res)
        return self.conv1_scene(res)
    