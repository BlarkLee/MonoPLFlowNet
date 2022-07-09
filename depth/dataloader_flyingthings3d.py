# This implementation is modified based on BTS https://github.com/cleinc/bts/tree/master/pytorch

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.utils.data.distributed
from torchvision import transforms
from PIL import Image
import os
import random
import IO



def _is_pil_image(img):
    return isinstance(img, Image.Image)


def _is_numpy_image(img):
    return isinstance(img, np.ndarray) and (img.ndim in {2, 3})


def preprocessing_transforms(mode):
    return transforms.Compose([
        ToTensor(mode=mode)
    ])


class DepthDataLoader(object):
    def __init__(self, args, mode):
        if mode == 'train':
            self.training_samples = DataLoadPreprocess(args, mode, transform=preprocessing_transforms(mode))
            if args.distributed:
                self.train_sampler = torch.utils.data.distributed.DistributedSampler(self.training_samples)
            else:
                self.train_sampler = None
    
            self.data = DataLoader(self.training_samples, args.batch_size,
                                   shuffle=False,
                                   num_workers=args.num_threads,
                                   pin_memory=True,
                                   sampler=self.train_sampler)

        elif mode == 'online_eval':
            self.testing_samples = DataLoadPreprocess(args, mode, transform=preprocessing_transforms(mode))
            if args.distributed:
                self.eval_sampler = DistributedSamplerNoEvenlyDivisible(self.testing_samples, shuffle=False)
            else:
                self.eval_sampler = None
            self.data = DataLoader(self.testing_samples, 1,
                                   shuffle=False,
                                   num_workers=1,
                                   pin_memory=True,
                                   sampler=self.eval_sampler)
        
        elif mode == 'test':
            self.testing_samples = DataLoadPreprocess(args, mode, transform=preprocessing_transforms(mode))
            self.data = DataLoader(self.testing_samples, 1, shuffle=False, num_workers=1)

        else:
            print('mode should be one of \'train, test, online_eval\'. Got {}'.format(mode))
            
            
class DataLoadPreprocess(Dataset):
    def __init__(self, args, mode, transform=None, is_for_online_eval=False):
        self.args = args
    
        self.mode = mode
        self.transform = transform
        self.to_tensor = ToTensor
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.is_for_online_eval = is_for_online_eval
        self.batch_size = args.batch_size
    
    def __getitem__(self, idx):
        focal = float(-1050)
        BASELINE = 1.0
        
        

        if self.mode == 'train':
            idx_train = 4*idx
            pc1_path = os.path.join(self.args.flyingthings3d_path, '../', "FlyingThings3D_subset_processed_35m", "train", "%07d"%(idx_train), "pc1.npy")
            
            while not os.path.exists(pc1_path):
                print("no corresponding mask")
                idx_train += 1
                pc1_path = os.path.join(self.args.flyingthings3d_path, '../', "FlyingThings3D_subset_processed_35m", "train", "%07d"%(idx_train), "pc1.npy")
            depth_mask_path = os.path.join(self.args.flyingthings3d_path, "../", "FlyingThings3D_subset_processed_35m", "train", "%07d"%(idx_train), "input_mask.npy")
            image_path_L_0 = os.path.join(self.args.flyingthings3d_path, "train", "image_clean", "left", "%07d"%(idx_train)+".png")
            disp_path_L_0 = os.path.join(self.args.flyingthings3d_path, "train","disparity", "left", "%07d"%(idx_train)+".pfm")
            image_path_L_1 = os.path.join(self.args.flyingthings3d_path, "train", "image_clean", "left", "%07d"%(idx_train+1)+".png")
            
            depth_mask = np.load(depth_mask_path)
            image_L_0 = Image.open(image_path_L_0)
            disp_gt_L_0 = IO.read(disp_path_L_0)
            image_L_1 = Image.open(image_path_L_1)
            
            depth_gt_L_0 = focal * BASELINE / disp_gt_L_0
            
            if self.args.do_kb_crop is True:
                canv0 = np.zeros((544, 960, 3))
                canv0[:540, :, :] = image_L_0
                canv1 = np.zeros((544, 960, 3))
                canv1[:540, :, :] = image_L_1
                
            
            image_L_0 = np.asarray(canv0, dtype=np.float32) / 255.0
            image_L_1 = np.asarray(canv1, dtype=np.float32) / 255.0
            depth_gt_L_0 = np.asarray(depth_gt_L_0, dtype=np.float32)
            depth_gt_L_0 = np.expand_dims(depth_gt_L_0, axis=2)

            sample = {'image': image_L_0, 'imageL_1': image_L_1, 'depth': depth_gt_L_0, 'focal': focal, 'depth_mask': depth_mask}
        
        else:
            idx_val = 4*idx
            image_path_L = os.path.join(self.args.flyingthings3d_path, "val", "image_clean", "left", "%07d"%(idx_val)+".png")
            disp_path_L = os.path.join(self.args.flyingthings3d_path, "val","disparity", "left", "%07d"%(idx_val)+".pfm")
            image_path_L_1 = os.path.join(self.args.flyingthings3d_path, "val", "image_clean", "left", "%07d"%(idx_val+1)+".png")

            image_L = Image.open(image_path_L)
            disp_gt_L = IO.read(disp_path_L)
            image_L_1 = Image.open(image_path_L_1)
            
            if self.args.do_kb_crop is True:
                canv0 = np.zeros((544, 960, 3))
                canv0[:540, :, :] = image_L
                canv1 = np.zeros((544, 960, 3))
                canv1[:540, :, :] = image_L_1


            
            
            depth_gt_L = focal * BASELINE / disp_gt_L
            
            image_L = np.asarray(canv0, dtype=np.float32) / 255.0
            image_L_1 = np.asarray(canv1, dtype=np.float32) / 255.0
            depth_gt_L = np.asarray(depth_gt_L, dtype=np.float32)
            depth_gt_L = np.expand_dims(depth_gt_L, axis=2)
            
            has_valid_depth = True
            sample = {'image': image_L, 'imageL_1': image_L_1, 'depth': depth_gt_L, 'focal': focal, 'has_valid_depth': has_valid_depth}        
        return sample
    
    
    def __len__(self):
        if self.mode == 'train':
            return int(len(os.listdir(os.path.join(self.args.flyingthings3d_path, "train", "flow", "left", "into_future")))/4)
        else:
            return int(len(os.listdir(os.path.join(self.args.flyingthings3d_path, "val", "flow", "left", "into_future")))/4)
        


class ToTensor(object):
    def __init__(self, mode):
        self.mode = mode
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    def __call__(self, sample):
        imageL_0 = sample['imageL_0']
        imageL_0 = self.normalize(imageL_0)
        imageL_1 = sample['imageL_1']
        imageL_1 = self.normalize(imageL_1)
        focal = sample["focal"]
        pc1 = sample['pc1']
        pc2 = sample['pc2']
        depth_mask = sample['depth_mask']
        

        depth = sample['depth']
        depth_gt_L_1 = sample['depth_1']
        if self.mode == 'train':
            return {'imageL_0': imageL_0, 'imageL_1':imageL_1, 'depth': depth, 'depth_1': depth_, 'focal': focal, 'pc1': pc1, 'pc2': pc2, 'depth_mask': depth_mask}
        else:
            return {'imageL_0': imageL_0, 'imageL_1':imageL_1, 'depth': depth, 'depth_1': depth_gt_L_1, 'focal': focal, 'has_valid_depth':sample['has_valid_depth'], 'pc1': pc1, 'pc2': pc2, 'depth_mask': depth_mask}
            
    
    def to_tensor(self, pic):
        if not (_is_pil_image(pic) or _is_numpy_image(pic)):
            raise TypeError(
                'pic should be PIL Image or ndarray. Got {}'.format(type(pic)))
        
        if isinstance(pic, np.ndarray):
            img = torch.from_numpy(pic.transpose((2, 0, 1)))
            return img
        
        # handle PIL Image
        if pic.mode == 'I':
            img = torch.from_numpy(np.array(pic, np.int32, copy=False))
        elif pic.mode == 'I;16':
            img = torch.from_numpy(np.array(pic, np.int16, copy=False))
        else:
            img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
        # PIL image mode: 1, L, P, I, F, RGB, YCbCr, RGBA, CMYK
        if pic.mode == 'YCbCr':
            nchannel = 3
        elif pic.mode == 'I;16':
            nchannel = 1
        else:
            nchannel = len(pic.mode)
        img = img.view(pic.size[1], pic.size[0], nchannel)
        
        img = img.transpose(0, 1).transpose(0, 2).contiguous()
        if isinstance(img, torch.ByteTensor):
            return img.float()
        else:
            return img
