# MonoPLFlowNet: Permutohedral Lattice FlowNet for Real-Scale 3D Scene Flow Estimation with Monocular Images
This is the offical repository for the implementation MonoPLFlowNet https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136870316.pdf. https://arxiv.org/abs/2111.12325 .

## Overview
 ![Image text](https://raw.githubusercontent.com/BlarkLee/MonoPLFlowNet/main/figures/overview.png)
With only two consecutive monocular images (left) as input, our MonoPLFlowNet estimates
both the depth (middle) and 3D scene flow (right) in real scale. Right shows a zoom-in real-scale scene flow of the two vehicles from side
view with the pseudo point cloud generating from the estimated depth map ((middle), where blue points are from frame t, red and green
points are blue points translated to frame t+1 by ground truth and estimated 3D scene flow, respectively. The objective is to align green and
red points.

## Demo
With a perfect 3D Scene Flow estimation, points in red should be overlapped as much as possible by points in green. 
### on KITTI:
With two consecutive monocular RGB images as the only input, MonoPLFlowNet jointly estimate depth (upper video) and 3D scene flow (lower video). Please watch carefully on the depth change, Scene flow visualization is rotated to the street-side view rather than original camera view to better visualize our estimation in **3D Real Scale**.

![](https://raw.githubusercontent.com/BlarkLee/MonoPLFlowNet/main/figures/kitti_dynamic_3.gif)
### on Flyingthings3D:
Scene flow visualization is rotated to the view better showing our estimation in **3D Real Scale**.

![](https://raw.githubusercontent.com/BlarkLee/MonoPLFlowNet/main/figures/fly_dynamic_0.gif)

# Code

## Prepare Environments
We recommend to use anaconda to prepare environments for this work. Our work is trained and tested under
```
Ubuntu 18.04
Python 3.7.3
Nvidia GPU + CUDA CuDNN 10.0
Pytorch 1.2.0
Numba 0.53.0
OpenCV 3.4.2
cffi 1.14.5
```

## Dataset Preparation (check paper section 4 Experiments to see the details of datasets)
In this work, we use two datasets， KITTI and Flyingthings3D.

### Prepare Flyingthings3D Dataset:
We use Flyingthings3D for training and evaluation of both depth and scene flow estimation. Download  and unzip "RGB images (cleanpass)", "Disparity", "Disparity change", "Optical flow", "Disparity Occlusions", "Flow Occlusions" from "DispNet/FlowNet2.0 dataset subsets". They will be unzipped at the same directory, the disparity map is originally in the dataset, to prepare scene flow data, run

`python data_preprocess/process_flyingthings3d_subset.py --raw_data_path RAW_DATA_PATH --save_path SAVE_PATH/FlyingThings3D_subset_processed_35m --only_save_near_pts`


### Prepare KITTI Dataset:
We use KITTI Eigen's split for training and evaluation of depth estimation:
download from http://www.cvlibs.net/download.php?file=data_depth_annotated.zip, run
```
cd ~/workspace/dataset/kitti_dataset
aria2c -x 16 -i ../../bts/utils/kitti_archives_to_download.txt
parallel unzip ::: \*.zip
```

### Prepare scene flow data, run
We use KITTI Flow 2015 split for training and evaluation of scene flow estimation. Download from https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html, and unzip to the directory RAW_DATA_PATH SAVE_PATH

`python data_preprocess/process_kitti.py RAW_DATA_PATH SAVE_PATH/KITTI_processed_noground_crop`


## Training
Setup the model:
```
cd models; 
python3 build_khash_cffi.py; 
cd ..
```

### Train DepthNet on Flyingthings3D: 
`python main_train_fly.py arguments_train_fly.txt`


### Train DepthNet on KITTI: 
`python main_train_kitti.py arguments_train_kitti.txt`

### Train MonoPLFlowNet on Flyingthings3D:
`python monopl_main_semi_flyingthings3d.py configs/train_monopl_semi.yaml`

Note that we don't train MonoPLFlowNet on KITTI, we only train it on Flyingthings3D while evaluating directly on KITTI. We train DepthNet on KITTI only for the sake of depth evaluation on KITTI, not for scene flow purpose.



## Evaluation
Download our trained models for evaluation and ablation study from https://drive.google.com/drive/folders/1MWX6ekn3k5JYeY3WIGo_QDo6thVTFX5B?usp=sharing. Find the `.yaml` file under `/flow/configs/ and /depth/`, change the model path to your download path. Specifically, the tag "resume" corresponds to  the scene flow model, while the tag "depth_checkpoint_path" corresponds to the depth model.


### Evaluate Depth
```
python main_eval_kitti.py arguments_eval_kitti.txt
python main_eval_fly.py arguments_eval_fly.txt
```


### Evaluate Scene Flow
```
python monopl_main_semi_flyingthings3d.py configs/test_monopl_flyingthings3d.yaml
python monopl_main_semi_kitti.py configs/test_monopl_kitti.yaml
python monopl_main_semi_kitti_ablation_expansion.py configs/test_monopl_kitti_ablation_expansion.yaml
python monopl_main_semi_flyingthings3d_ablation_expansion.py configs/test_monopl_flyingthings3d_ablation_expansion.yaml
python monopl_main_semi_flyingthings3d_ablation_monosf.py configs/test_monopl_flyingthings3d_ablation_monosf.yaml
python monopl_main_semi_kitti_ablation_monosf.py configs/test_monopl_kitti_ablation_monosf.yaml
python monopl_main_semi_kitti_ablation_monosf_multi.py configs/test_monopl_kitti_ablation_monosf_multi.yaml
python monopl_main_semi_flyingthings3d_ablation_monosf_multi.py configs/test_monopl_flyingthings3d_ablation_monosf_multi.yaml
```

## Citation
```
@InProceedings{MonoPLFlowNet,
author={Li, Runfa and Nguyen, Truong},
title="MonoPLFlowNet: Permutohedral Lattice FlowNet for Real-Scale 3D Scene Flow Estimation with Monocular Images",
booktitle="Proceedings of the European Conference on Computer Vision (ECCV)",
year="2022",
publisher="Springer Nature Switzerland",
pages="322--339"
}
```

## Acknowledgments
Thanks for the contribution from "HPLFlowNet" - https://github.com/laoreja/HPLFlowNet and "BTS" - https://github.com/cleinc/bts. 
