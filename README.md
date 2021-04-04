# VoteNet - NPM3D Course Project, Master MVA
 
Project for the course "Nuages de Points et Modélisation 3D" of the Master MVA (2020-2021).

The goal of the project is to study the method described in the paper "Deep Hough Voting for 3D Object Detection in Point Clouds", test an implementation of the method, and reproduce some results. I chose to experiment with the official implementation of the method, rewrite some key elements, train a VoteNet architecture on the SUN RGB-D Dataset, evaluate the results and provide quantitative and qualitative evaluations in my report.

## Description

Most of the code comes from the original github repository: https://github.com/facebookresearch/votenet.  
The folders `models` and `losses` contain partial rewritings of the model's architecture and loss functions, with reorganized code and some additional comments. I did this rewriting to better understand the inner mechanisms of the method, and its behaviour is similar to the official implementation.  
The training scripts `main_train.py` and `main_eval.py` are also adapted from the original code.

## Run

To generate a point cloud dataset with bounding boxes from SUN RGB-D, please follow the instructions of the [README](https://github.com/pauljcb/deep_hough_voting/blob/main/sunrgbd/README.md) under the `sunrgbd` folder from Qi et al. (2019). At the end, you should have two dataset folders `sunrgbd_pc_bbox_votes_20k_v1_train` and `sunrgbd_pc_bbox_votes_20k_v1_val`.

Before training the network, please compile the CUDA layers of PointNet++: 

```
cd pointnet2
python setup.py install
```

Also make sure to have the following packages:

```
torch
torchvision
matplotlib
opencv-python
plyfile
'trimesh>=2.35.39,<2.35.40'
'networkx>=2.2,<2.3'
```

To train a VoteNet architecture, please start the script `main_train.py`. Feel free to change the trainings parameters inside the script itself. You should obtain a final mAP above 55.

To evaluate the performances, please start the script `main_eval.py`. Feel free to change the trainings parameters inside the script itself. The script will evaluate the mAP@0.25 and mAP@0.5 on the validation dataset, and store visual results with colored bounding boxes for the 50 first batches.

## Some visual results

![bedroom](https://github.com/pauljcb/deep_hough_voting/blob/main/figs/bedroom_detection.jpg)
![more_visual_results](https://github.com/pauljcb/deep_hough_voting/blob/main/figs/visual_results.jpg)
![more_visual_results_2](https://github.com/pauljcb/deep_hough_voting/blob/main/figs/visual_results_2.jpg)
