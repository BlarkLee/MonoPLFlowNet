# This implementation is modified based on BTS https://github.com/cleinc/bts/tree/master/pytorch

import torch
import torch.nn as nn
import torch.nn.functional as torch_nn_func
import math
import numpy as np

from collections import namedtuple


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

    def forward(self, depth_est, depth_est_2x2, depth_est_4x4, depth_est_8x8, depth_gt, mask):
        d = torch.log(depth_est[mask]) - torch.log(depth_gt[mask])
        loss_1x1 = torch.sqrt((d ** 2).mean() - self.variance_focus * (d.mean() ** 2)) * 10.0
        d_2x2 = torch.log(depth_est_2x2[mask]) - torch.log(depth_gt[mask]/2)
        loss_2x2 = torch.sqrt((d_2x2 ** 2).mean() - self.variance_focus * (d_2x2.mean() ** 2)) * 10.0
        d_4x4 = torch.log(depth_est_4x4[mask]) - torch.log(depth_gt[mask]/4)
        loss_4x4 = torch.sqrt((d_4x4 ** 2).mean() - self.variance_focus * (d_4x4.mean() ** 2)) * 10.0
        d_8x8 = torch.log(depth_est_8x8[mask]) - torch.log(depth_gt[mask]/8)
        loss_8x8 = torch.sqrt((d_8x8 ** 2).mean() - self.variance_focus * (d_8x8.mean() ** 2)) * 10.0
        return (8 * loss_1x1 + 4 * loss_2x2 + 2 * loss_4x4 + loss_8x8)/15


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

def conv(in_planes, out_planes, kernel_size=3, stride=1, dilation=1, isReLU=True):
    if isReLU:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, dilation=dilation,
                      padding=((kernel_size - 1) * dilation) // 2, bias=True),
            nn.LeakyReLU(0.1, inplace=True)
        )
    else:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, dilation=dilation,
                      padding=((kernel_size - 1) * dilation) // 2, bias=True),
            nn.Sigmoid()
        )    
    
class depth_decoder(nn.Module):
    def __init__(self, ch_in):
        super(depth_decoder, self).__init__()

        self.convs = nn.Sequential(
            conv(ch_in, 128),
            #conv(128, 128),
            conv(128, 96),
            conv(96, 64),
            conv(64, 32)
        )
        self.conv_d1 = conv(32, 1, isReLU=False)

    def forward(self, x):
        x_out = self.convs(x)
        disp1 = self.conv_d1(x_out)

        return x_out, disp1    

class bts(nn.Module):
    def __init__(self, params, feat_out_channels, num_features=512):
        super(bts, self).__init__()
        self.params = params

        self.upconv5    = upconv(feat_out_channels[4], num_features)
        self.bn5        = nn.BatchNorm2d(num_features, momentum=0.01, affine=True, eps=1.1e-5)
        
        self.conv5      = torch.nn.Sequential(nn.Conv2d(num_features + feat_out_channels[3], num_features, 3, 1, 1, bias=False),
                                              nn.ELU())
        self.upconv4    = upconv(num_features, num_features // 2)
        self.bn4        = nn.BatchNorm2d(num_features // 2, momentum=0.01, affine=True, eps=1.1e-5)
        self.conv4      = torch.nn.Sequential(nn.Conv2d(num_features // 2 + feat_out_channels[2], num_features // 2, 3, 1, 1, bias=False),
                                              nn.ELU())
        self.bn4_2      = nn.BatchNorm2d(num_features // 2, momentum=0.01, affine=True, eps=1.1e-5)
        
        self.daspp_3    = atrous_conv(num_features // 2, num_features // 4, 3, apply_bn_first=False)
        self.daspp_6    = atrous_conv(num_features // 2 + num_features // 4 + feat_out_channels[2], num_features // 4, 6)
        self.daspp_12   = atrous_conv(num_features + feat_out_channels[2], num_features // 4, 12)
        self.daspp_18   = atrous_conv(num_features + num_features // 4 + feat_out_channels[2], num_features // 4, 18)
        self.daspp_24   = atrous_conv(num_features + num_features // 2 + feat_out_channels[2], num_features // 4, 24)
        self.daspp_conv = torch.nn.Sequential(nn.Conv2d(num_features + num_features // 2 + num_features // 4, num_features // 4, 3, 1, 1, bias=False),
                                              nn.ELU())
        self.depth_decoder8x8 = depth_decoder(128)
        
        self.upconv3    = upconv(num_features // 4, num_features // 4)
        self.bn3        = nn.BatchNorm2d(num_features // 4, momentum=0.01, affine=True, eps=1.1e-5)
        self.conv3      = torch.nn.Sequential(nn.Conv2d(num_features // 4 + feat_out_channels[1] + 1, num_features // 4, 3, 1, 1, bias=False),
                                              nn.ELU())
        self.depth_decoder4x4 = depth_decoder(128)
        
        self.upconv2    = upconv(num_features // 4, num_features // 8)
        self.bn2        = nn.BatchNorm2d(num_features // 8, momentum=0.01, affine=True, eps=1.1e-5)
        self.conv2      = torch.nn.Sequential(nn.Conv2d(num_features // 8 + feat_out_channels[0] + 1, num_features // 8, 3, 1, 1, bias=False),
                                              nn.ELU())
        
        self.depth_decoder2x2 = depth_decoder(64)
        
        self.upconv1    = upconv(num_features // 8, num_features // 16)
        self.depth_decoder1x1 = depth_decoder(32)
        self.conv1      = torch.nn.Sequential(nn.Conv2d(num_features // 16 + 4, num_features // 16, 3, 1, 1, bias=False),
                                              nn.ELU())
        self.get_depth  = torch.nn.Sequential(nn.Conv2d(num_features // 16, 1, 3, 1, 1, bias=False),
                                              nn.Sigmoid())

    def forward(self, features, focal):
        skip0, skip1, skip2, skip3 = features[1], features[2], features[3], features[4]
        dense_features = torch.nn.ReLU()(features[5])
        upconv5 = self.upconv5(dense_features) # H/16
        upconv5 = self.bn5(upconv5)
        concat5 = torch.cat([upconv5, skip3], dim=1)
        iconv5 = self.conv5(concat5)
        
        upconv4 = self.upconv4(iconv5) # H/8
        upconv4 = self.bn4(upconv4)
        concat4 = torch.cat([upconv4, skip2], dim=1)
        iconv4 = self.conv4(concat4)
        iconv4 = self.bn4_2(iconv4)
        
        daspp_3 = self.daspp_3(iconv4)
        concat4_2 = torch.cat([concat4, daspp_3], dim=1)
        daspp_6 = self.daspp_6(concat4_2)
        concat4_3 = torch.cat([concat4_2, daspp_6], dim=1)
        daspp_12 = self.daspp_12(concat4_3)
        concat4_4 = torch.cat([concat4_3, daspp_12], dim=1)
        daspp_18 = self.daspp_18(concat4_4)
        concat4_5 = torch.cat([concat4_4, daspp_18], dim=1)
        daspp_24 = self.daspp_24(concat4_5)
        concat4_daspp = torch.cat([iconv4, daspp_3, daspp_6, daspp_12, daspp_18, daspp_24], dim=1)
        daspp_feat = self.daspp_conv(concat4_daspp)
        
        feat_8x8, depth_8x8 = self.depth_decoder8x8(daspp_feat)
        depth_8x8 = depth_8x8.clone()
        depth_8x8_scaled_ds = torch_nn_func.interpolate(depth_8x8, scale_factor=2, mode='nearest') #* self.params.max_depth/8
        depth_8x8_scaled = torch_nn_func.interpolate(depth_8x8_scaled_ds, scale_factor=4, mode='nearest') * self.params.max_depth/8      
        
        upconv3 = self.upconv3(daspp_feat) # H/4
        upconv3 = self.bn3(upconv3)
        concat3 = torch.cat([upconv3, skip1, depth_8x8_scaled_ds], dim=1)
        iconv3 = self.conv3(concat3)
        
        feat_4x4,  depth_4x4 = self.depth_decoder4x4(iconv3)
        depth_4x4 = depth_4x4.clone()
        depth_4x4_scaled_ds = torch_nn_func.interpolate(depth_4x4, scale_factor=2, mode='nearest') #* self.params.max_depth/4
        depth_4x4_scaled = torch_nn_func.interpolate(depth_4x4_scaled_ds, scale_factor=2, mode='nearest') * self.params.max_depth/4
        
        upconv2 = self.upconv2(iconv3) # H/2
        upconv2 = self.bn2(upconv2)
        concat2 = torch.cat([upconv2, skip0, depth_4x4_scaled_ds], dim=1)
        iconv2 = self.conv2(concat2)
        feat_2x2,  depth_2x2 = self.depth_decoder2x2(iconv2)
        depth_2x2 = depth_2x2.clone()
        depth_2x2_scaled_ds = torch_nn_func.interpolate(depth_2x2, scale_factor=2, mode='nearest') #* self.params.max_depth/2
        depth_2x2_scaled = depth_2x2_scaled_ds * self.params.max_depth/2
        
        
        
        upconv1 = self.upconv1(iconv2)
        feat_1x1, depth_1x1 = self.depth_decoder1x1(upconv1)
        depth_1x1 = depth_1x1.clone()
        depth_1x1 *= self.params.max_depth
        final_depth = (depth_1x1 + 2 * depth_2x2_scaled + 4 * depth_4x4_scaled + 8 * depth_8x8_scaled)/4
        feat_all = [feat_8x8, feat_4x4, feat_2x2, feat_1x1]
        return depth_8x8_scaled, depth_4x4_scaled, depth_2x2_scaled, depth_1x1, final_depth, feat_all

class encoder(nn.Module):
    def __init__(self, params):
        super(encoder, self).__init__()
        self.params = params
        import torchvision.models as models
        if params.encoder == 'densenet121_bts':
            self.base_model = models.densenet121(pretrained=True).features
            self.feat_names = ['relu0', 'pool0', 'transition1', 'transition2', 'norm5']
            self.feat_out_channels = [64, 64, 128, 256, 1024]
        elif params.encoder == 'densenet161_bts':
            self.base_model = models.densenet161(pretrained=True).features
            self.feat_names = ['relu0', 'pool0', 'transition1', 'transition2', 'norm5']
            self.feat_out_channels = [96, 96, 192, 384, 2208]
        elif params.encoder == 'resnet50_bts':
            self.base_model = models.resnet50(pretrained=True)
            self.feat_names = ['relu', 'layer1', 'layer2', 'layer3', 'layer4']
            self.feat_out_channels = [64, 256, 512, 1024, 2048]
        elif params.encoder == 'resnet101_bts':
            self.base_model = models.resnet101(pretrained=True)
            self.feat_names = ['relu', 'layer1', 'layer2', 'layer3', 'layer4']
            self.feat_out_channels = [64, 256, 512, 1024, 2048]
        elif params.encoder == 'resnext50_bts':
            self.base_model = models.resnext50_32x4d(pretrained=True)
            self.feat_names = ['relu', 'layer1', 'layer2', 'layer3', 'layer4']
            self.feat_out_channels = [64, 256, 512, 1024, 2048]
        elif params.encoder == 'resnext101_bts':
            self.base_model = models.resnext101_32x8d(pretrained=True)
            self.feat_names = ['relu', 'layer1', 'layer2', 'layer3', 'layer4']
            self.feat_out_channels = [64, 256, 512, 1024, 2048]
        else:
            print('Not supported encoder: {}'.format(params.encoder))

    def forward(self, x):
        features = [x]
        skip_feat = [x]
        for k, v in self.base_model._modules.items():
            if 'fc' in k or 'avgpool' in k:
                continue
            feature = v(features[-1])
            features.append(feature)
            if any(x in k for x in self.feat_names):
                skip_feat.append(feature)
        
        return skip_feat
    

class DepthModel(nn.Module):
    def __init__(self, params):
        super(DepthModel, self).__init__()
        self.encoder = encoder(params)
        self.decoder = bts(params, self.encoder.feat_out_channels, params.bts_size)

    def forward(self, x, focal):
        skip_feat = self.encoder(x)
        return self.decoder(skip_feat, focal)
    