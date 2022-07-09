import sys, os
import os.path as osp
import numpy as np
import IO
from PIL import Image
import torch
import torch.utils.data as data
import time

__all__ = ['KITTI_monopl_self']


class KITTI_monopl_self(data.Dataset):
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
                 args
                 ):
        self.root = osp.join(args.data_root, 'KITTI_processed_noground_crop')
        self.root_others = osp.join(args.data_root, 'training')
        assert train is False
        self.train = train
        self.transform = transform
        self.gen_func = gen_func
        self.num_points = args.num_points
        self.remove_ground = args.remove_ground

        self.samples = self.make_dataset()
        if len(self.samples) == 0:
            raise (RuntimeError("Found 0 files in subfolders of: " + self.root + "\n"))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        pc1_loaded, pc2_loaded, pc_mask, image1_loaded, image2_loaded, img1_path = self.pc_loader(self.samples[index], self.root_others)
        time_lattice_start = time.time()
        pc1_transformed, pc2_transformed, sf_transformed_f, mask1, mask2 = self.transform([pc1_loaded, pc2_loaded])
        time_lattice_end = time.time()
        lattice_time = time_lattice_end - time_lattice_start
        #print("pc1_transformed", pc1_transformed.shape)
        if pc1_transformed is None:
            print('path {} get pc1 is None'.format(self.samples[index]), flush=True)
            index = np.random.choice(range(self.__len__()))
            return self.__getitem__(index)
        '''
        pc1, pc2, sf, generated_data = self.gen_func([pc1_transformed,
                                                      pc2_transformed,
                                                      sf_transformed])'''
        
        pc1, pc2, generated_data_f = self.gen_func([pc1_transformed,
                                                  pc2_transformed])
        
        if self.train:
            _, _, generated_data_b = self.gen_func([pc2_transformed,
                                                  pc1_transformed])
        sf_f = torch.from_numpy(sf_transformed_f.T)

        if self.train:
            return pc1, pc2, sf_f, self.samples[index], image1_loaded, image2_loaded, pc_mask, mask1, mask2, generated_data_f, generated_data_b
        else:
            return pc1, pc2, sf_f, self.samples[index], image1_loaded, image2_loaded, pc_mask, mask1, mask2, generated_data_f, img1_path, lattice_time

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Number of points per point cloud: {}\n'.format(self.num_points)
        fmt_str += '    is removing ground: {}\n'.format(self.remove_ground)
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))

        return fmt_str

    def make_dataset(self):
        do_mapping = True
        root = osp.realpath(osp.expanduser(self.root))

        all_paths = sorted(os.walk(root))
        useful_paths = [item[0] for item in all_paths if len(item[1]) == 0]
        try:
            assert (len(useful_paths) == 200)
        except AssertionError:
            print('assert (len(useful_paths) == 200) failed!', len(useful_paths))

        if do_mapping:
            mapping_path = osp.join(osp.dirname(__file__), 'KITTI_mapping.txt')
            print('mapping_path', mapping_path)

            with open(mapping_path) as fd:
                lines = fd.readlines()
                lines = [line.strip() for line in lines]
            useful_paths = [path for path in useful_paths if lines[int(osp.split(path)[-1])] != '']

        res_paths = useful_paths

        return res_paths

    def pc_loader(self, path, root_path):
        """
        Args:
            path:
        Returns:
            pc1: ndarray (N, 3) np.float32
            pc2: ndarray (N, 3) np.float32
        """
        #print("path", path)
        #print(stop)
        _, index = osp.split(path)
        pc1 = np.load(osp.join(path, 'pc1.npy'))  #.astype(np.float32)
        pc2 = np.load(osp.join(path, 'pc2.npy'))  #.astype(np.float32)
        
        pc_mask = np.load(osp.join(path, 'pc_mask.npy'))
        pc1 = pc1[pc_mask]
        pc2 = pc2[pc_mask]
        
        image1 = Image.open(osp.join(root_path, "image_2", "%s_10.png"%index))
        image2 = Image.open(osp.join(root_path, "image_2", "%s_11.png"%index))
        image1 = np.asarray(image1, dtype=np.float32) / 255.0
        image2 = np.asarray(image2, dtype=np.float32) / 255.0
        image1 = image1[:352, :1216, :]
        image2 = image2[:352, :1216, :]
        img1_path = osp.join(root_path, "image_2", "%s_10.png"%index)
        #print("img1_path", img1_path)

        return pc1, pc2, pc_mask, image1, image2, img1_path
