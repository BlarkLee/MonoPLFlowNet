import sys, os
import os.path as osp
import numpy as np
import IO
from PIL import Image
import torch
import time

import torch.utils.data as data

__all__ = ['FlyingThings3DMonopl_self']


class FlyingThings3DMonopl_self(data.Dataset):
    """
    Args:
        train (bool): If True, creates dataset from training set, otherwise creates from test set.
        transform (callable):
        gen_func (callable):
        args:
    """
    def __init__(self,
                 train,
                 transform,
                 gen_func,
                 args):
        self.root = osp.join(args.data_root, 'FlyingThings3D_subset_processed_35m') #_real')
        self.root_others = osp.join(args.data_root, 'original')
        self.train = train
        self.transform = transform
        self.gen_func = gen_func
        self.num_points = args.num_points

        full = hasattr(args, 'full') and args.full
        self.samples = self.make_dataset(full)

        if len(self.samples) == 0:
            raise (RuntimeError("Found 0 files in subfolders of: " + self.root + "\n"))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        #print("index", index)
        #print("self.samples[index]", self.samples[index])
        #self.samples[index] = '/mnt/universe/DataSet/FlyingThings3D_subset/FlyingThings3D_subset_processed_35m/train/0013662'
        pc1_loaded, pc2_loaded, pc_mask, depth1_loaded, depth2_loaded, image1_loaded, image2_loaded, img1_path = self.loader(self.samples[index], self.root_others)
        time_lattice_start = time.time()
        pc1_transformed, pc2_transformed, sf_transformed_f, mask1, mask2 = self.transform([pc1_loaded, pc2_loaded])
        time_lattice_end = time.time()
        lattice_time = time_lattice_end - time_lattice_start
        if pc1_transformed is None:
            print('path {} get pc1 is None'.format(self.samples[index]), flush=True)
            index = np.random.choice(range(self.__len__()))
            return self.__getitem__(index)
        
        #print('pc1_trans, pc2_trans, sf_trans', pc1_transformed.shape, pc2_transformed.shape, sf_transformed.shape)
        '''
        pc1, pc2, sf, generated_data = self.gen_func([pc1_transformed,
                                                      pc2_transformed,
                                                      sf_transformed])'''
        
        pc1, pc2, generated_data_f = self.gen_func([pc1_transformed,
                                                  pc2_transformed])
        
        if self.train:
            _, _, generated_data_b = self.gen_func([pc2_transformed,
                                                  pc1_transformed])
        '''
        pc1 = torch.from_numpy(pc1_transformed.T)
        pc2 = torch.from_numpy(pc2_transformed.T)'''
        sf_f = torch.from_numpy(sf_transformed_f.T)
        #sf_b = torch.from_numpy(sf_transformed_b.T)

        #print("pc1, pc2, sf", pc1.shape, pc2.shape, sf.shape)
        '''print("generated_data:")
        for i in range (len(generated_data)):
            for key in generated_data[i].keys():
                print(key, generated_data[i][key].shape)'''
        if self.train:
            return pc1, pc2, sf_f, self.samples[index], depth1_loaded, depth2_loaded, image1_loaded, image2_loaded, pc_mask, mask1, mask2, generated_data_f, generated_data_b
        else:
            return pc1, pc2, sf_f, self.samples[index], depth1_loaded, depth2_loaded, image1_loaded, image2_loaded, pc_mask, mask1, mask2, generated_data_f, img1_path, lattice_time

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Number of points per point cloud: {}\n'.format(self.num_points)
        fmt_str += '    is training: {}\n'.format(self.train)
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str

    def make_dataset(self, full):
        root = osp.realpath(osp.expanduser(self.root))
        root = osp.join(root, 'train') if self.train else osp.join(root, 'val')

        all_paths = os.walk(root)
        useful_paths = sorted([item[0] for item in all_paths if len(item[1]) == 0])

        try:
            if self.train:
                assert (len(useful_paths) == 19640)
            else:
                assert (len(useful_paths) == 3824)
        except AssertionError:
            print('len(useful_paths) assert error', len(useful_paths))
            sys.exit(1)

        if not full:
            res_paths = useful_paths[::4]
        else:
            res_paths = useful_paths

        return res_paths

    def loader(self, path, root_path):
        """
        Args:
            path: path to a dir, e.g., home/xiuye/share/data/Driving_processed/35mm_focallength/scene_forwards/slow/0791
        Returns:
            pc1: ndarray (N, 3) np.float32
            pc2: ndarray (N, 3) np.float32
        """
        f = -1050
        BASELINE = 1.0
        #print("path", path)
        #print("path_split", osp.split(path))
        _, index = osp.split(path)
        root_path = osp.join(root_path, 'train') if self.train else osp.join(root_path, 'val')
        #print("disp1_path", osp.join(root_path, "disparity", "left", "%s.pfm"%index))
        #print("disp2_path", osp.join(root_path, "disparity", "left", "%07d.pfm"%(int(index)+1)))
        disp1 = IO.read(osp.join(root_path, "disparity", "left", "%s.pfm"%index))
        disp2 = IO.read(osp.join(root_path, "disparity", "left", "%07d.pfm"%(int(index)+1)))
        image1 = Image.open(osp.join(root_path, "image_clean", "left", "%s.png"%index))
        image2 = Image.open(osp.join(root_path, "image_clean", "left", "%07d.png"%(int(index)+1)))
        image1 = np.asarray(image1, dtype=np.float32) / 255.0
        image2 = np.asarray(image2, dtype=np.float32) / 255.0
        canv1 = np.zeros((544, 960, 3))
        canv1[:540, :, :] = image1
        canv2 = np.zeros((544, 960, 3))
        canv2[:540, :, :] = image2
        depth1 = -1. * f * BASELINE / disp1
        depth2 = -1. * f * BASELINE / disp2
        #print("depth1", depth1.shape)
        #print("depth2", depth2.shape)
        #print("image1", image1.size)
        #print("image2", image2.size)
        pc1 = np.load(osp.join(path, 'pc1.npy'))
        #print("pc1_loaded", pc1.shape)
        pc2 = np.load(osp.join(path, 'pc2.npy'))
        #print("pc2_loaded", pc2.shape)
        pc_mask = np.load(osp.join(path, 'input_mask.npy'))
        # multiply -1 only for subset datasets
        pc1[..., -1] *= -1
        pc2[..., -1] *= -1
        pc1[..., 0] *= -1
        pc2[..., 0] *= -1
        #print("pc1", pc1.shape)
        #print("pc2", pc2.shape)
        #print("pc_mask", pc_mask.shape)
        #depth1_masked = depth1[pc_mask]
        #depth2_masked = depth2[pc_mask]
        #print("depth1_masked", depth1_masked.shape)
        #print("depth2_masked", depth2_masked.shape)
        #print(stop)
        img1_path = osp.join(root_path, "image_clean", "left", "%s.png"%index)
        return pc1, pc2, pc_mask, depth1, depth2, canv1.astype(np.float32), canv2.astype(np.float32), img1_path
