# MonoPLFlowNet: Permutohedral Lattice FlowNet for Real-Scale 3D Scene Flow Estimation with Monocular Images
This is the offical repository for the implementation of our MonoPLFlowNet published on ECCV 2022. Preprint version is available at arXiv https://arxiv.org/abs/2111.12325 .

## Abstract
Real-scale scene flow estimation has become increasingly important for 3D computer vision. Some works successfully estimate real-scale 3D scene flow with LiDAR. However, these ubiquitous and expensive sensors are still unlikely to be equipped widely for real application. Other works use monocular images to estimate scene flow, but their scene flow estimations are normalized with scale ambiguity, where additional depth or point cloud ground truth are required to recover the real scale. Even though they perform well in 2D, these works do not provide accurate and reliable 3D estimates. We present a deep learning architecture on permutohedral lattice - MonoPLFlowNet. Different from all previous works, our MonoPLFlowNet is the first work where only two consecutive monocular images are used as input, while both depth and 3D scene flow are estimated in real scale. Our real-scale scene flow estimation outperforms all state-of-the-art monocular-image based works recovered to real scale by ground truth, and is comparable to LiDAR approaches. As a by-product, our real-scale depth estimation also outperforms other state-of-the-art works.


## Overview
 ![Image text](https://raw.githubusercontent.com/BlarkLee/MonoPLFlowNet/main/overview.png)
With only two consecutive monocular images (left) as input, our MonoPLFlowNet estimates
both the depth (middle) and 3D scene flow (right) in real scale. Right shows a zoom-in real-scale scene flow of the two vehicles from side
view with the pseudo point cloud generating from the estimated depth map ((middle), where blue points are from frame t, red and green
points are blue points translated to frame t+1 by ground truth and estimated 3D scene flow, respectively. The objective is to align green and
red points.

## Demo
With a perfect 3D Scene Flow estimation, points in red should be overlapped as much as possible by points in green. 
### KITTI Dataset:
With two consecutive monocular RGB images as the only input, MonoPLFlowNet jointly estimate depth (upper video) and 3D scene flow (lower video). Please watch carefully on the depth change, Scene flow visualization is rotated to the street-side view rather than original camera view to better visualize our estimation in **3D Real Scale**.

![](https://raw.githubusercontent.com/BlarkLee/MonoPLFlowNet/main/kitti_dynamic_3.gif)
### Flyingthings3D Dataset:
Scene flow visualization is rotated to the view better showing our estimation in **3D Real Scale**.

![](https://raw.githubusercontent.com/BlarkLee/MonoPLFlowNet/main/fly_dynamic_0.gif)

## Code will be available soon!

1. Prepare Environments
We highly recommend to use anaconda to prepare environments for this work. Our work is trained and tested under

Ubuntu 18.04
Python 3.7.3
Nvidia GPU + CUDA CuDNN 10.0
Pytorch 1.2.0
Numba 0.53.0
OpenCV 3.4.2
cffi 1.14.5


2. Dataset Preparation (check paper section 4 Experiments to see the details of datasets)
We use two datasets for both training and evaluation in our work, KITTI Dataset and Flyingthings3D Dataset. Make sure you have enough space to store the datasets. 

2.1 Prepare Flyingthings3D Dataset:
We use Flyingthings3D for training and evaluation of both depth and scene flow estimation.  please download  and unzip "RGB images (cleanpass)", "Disparity", "Disparity change", "Optical flow", "Disparity Occlusions", "Flow Occlusions" from "DispNet/FlowNet2.0 dataset subsets". They will be unzipped at the same directory, the disparity map is originally in the dataset, to prepare scene flow data, run

$ python data_preprocess/process_flyingthings3d_subset.py --raw_data_path RAW_DATA_PATH --save_path SAVE_PATH/FlyingThings3D_subset_processed_35m --only_save_near_pts


2.2Prepare KITTI Dataset:
We use KITTI Eigen's split for training and evaluation of depth estimation:
download from http://www.cvlibs.net/download.php?file=data_depth_annotated.zip, 

#Prepare depth data, run
$ cd ~/workspace/dataset/kitti_dataset
$ aria2c -x 16 -i ../../bts/utils/kitti_archives_to_download.txt
$ parallel unzip ::: *.zip

We use KITTI Flow 2015 split for training and evaluation of scene flow estimation:
download from https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html, and unzip to the directory RAW_DATA_PATH SAVE_PATH

# Prepare scene flow data, run
$ python data_preprocess/process_kitti.py RAW_DATA_PATH SAVE_PATH/KITTI_processed_noground_crop 


3. Training
Since the permutohedral lattice is written in hash table, please setup the model first:
cd models; python3 build_khash_cffi.py; cd ..

# Train DepthNet on Flyingthings3D: 
python main_train_fly.py arguments_train_fly.txt


#Train DepthNet on KITTI: 
python main_train_kitti.py arguments_train_kitti.txt

#Train MonoPLFlowNet on Flyingthings3D:
python monopl_main_semi_flyingthings3d.py configs/train_monopl_semi.yaml

Note that we don't train MonoPLFlowNet on KITTI, we only train it on Flyingthings3D while evaluating directly on KITTI. We train DepthNet on KITTI only for the sake of depth evaluation on KITTI, not for scene flow purpose.



4. Evaluation
We shared our trained models from anonymous cloud drive for your evaluation purposes, download our trained models and models for ablation study from https://drive.google.com/drive/folders/1MWX6ekn3k5JYeY3WIGo_QDo6thVTFX5B?usp=sharing. Find the .yaml file under /flow/configs/ and /depth/ to change the model path to your download directory. Specifically, the tag "resume" corresponds to  the scene flow model, while the tag "depth_checkpoint_path" corresponds to the depth model.


4.1 Evaluate Depth
python main_eval_kitti.py arguments_eval_kitti.txt
python main_eval_fly.py arguments_eval_fly.txt


4.1 Evaluate Scene Flow
python monopl_main_semi_flyingthings3d.py configs/test_monopl_flyingthings3d.yaml
python monopl_main_semi_kitti.py configs/test_monopl_kitti.yaml
python monopl_main_semi_kitti_ablation_expansion.py configs/test_monopl_kitti_ablation_expansion.yaml
python monopl_main_semi_flyingthings3d_ablation_expansion.py configs/test_monopl_flyingthings3d_ablation_expansion.yaml
python monopl_main_semi_flyingthings3d_ablation_monosf.py configs/test_monopl_flyingthings3d_ablation_monosf.yaml
python monopl_main_semi_kitti_ablation_monosf.py configs/test_monopl_kitti_ablation_monosf.yaml
python monopl_main_semi_kitti_ablation_monosf_multi.py configs/test_monopl_kitti_ablation_monosf_multi.yaml
python monopl_main_semi_flyingthings3d_ablation_monosf_multi.py configs/test_monopl_flyingthings3d_ablation_monosf_multi.yaml

Acknowledgments
Our MonoPLFlowNet implementation is based on "HPLFlowNet" - https://github.com/laoreja/HPLFlowNet and "BTS" - https://github.com/cleinc/bts. Thanks for their contribution.
