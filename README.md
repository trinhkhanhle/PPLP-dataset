# FCAV M-Air Pedestrian (FMP), CMU Panoptic, and Orientation Datasets for PPLP

This repository contains a complete FCAV M-Air Pedestrian (FMP) dataset, the code to generate a dataset that contains 2D LiDAR data and camera color images from CMU Panoptic dataset, and the code to generate Orientation dataset, which are used by PPLP for training, testing and validation, to detect 3D locations and orientations of pedestrians. We require datasets to be arranged in KITTI format and the folder structure should like the following:

```
Kitti
└── object
    ├── testing
    │   ├── calib
    │   │   ├── xxxxxx.txt
    │   │   └── xxxxxx.txt
    │   ├── rgb_images
    │   │   ├── xxxxxx.png
    │   │   └── xxxxxx.png
    │   └── lidar_ptclouds
    │       ├── xxxxxx.ply
    │       └── xxxxxx.ply
    ├── training
    │   ├── calib
    │   │   ├── xxxxxx.txt
    │   │   └── xxxxxx.txt
    │   ├── rgb_images
    │   │   ├── xxxxxx.png
    │   │   └── xxxxxx.png
    │   ├── label_2
    │   │   ├── xxxxxx.txt
    │   │   └── xxxxxx.txt
    │   ├── planes
    │   │   ├── xxxxxx.txt
    │   │   └── xxxxxx.txt
    │   └── lidar_ptclouds
    │       ├── xxxxxx.ply
    │       └── xxxxxx.ply
    ├── train.txt
    └── val.txt
```



The following are steps to generate the data:

### FMP dataset

It can be obtained via this [link](https://drive.google.com/open?id=13wOtYRuulcZwcItULLmrLc5jxNUIHoG_) (already in KITTI format).

The FMP dataset was collected from an HD camera and a Hokuyo UTM-30LX-EW planar LiDAR mounted on a ROS-enabled Segway mobile robot platform. The dataset was collected in an outdoor environment using the "M-Air" facility at the University of Michigan campus in Ann Arbor, MI, USA in January 2019. A Qualisys Motion Capture system was used to record shoulder key-points of pedestrians for ground truth data. 

The FMP dataset contains four short videos with a total recording time of 4 minutes and we used 3,934 frames with good quality ground truth. In each frame, there are up to two pedestrians walking in the scene, interacting with and sometimes occluding each other. The last video clip (810 frames) was selected as the test set. Similar to the CMU Panoptic Dataset, frames from the remaining three sequences (3,124 frames) are randomly shuffled for training and validation with 3:1 ratio.

### CMU dataset

It can be downloaded via this [link](http://domedb.perception.cs.cmu.edu/dataset.html). Please follow the instructions at that site to download following sequences: `160422_ultimatum1`, `160226_haggling1`, `160422_haggling1`, `160224_haggling1`, and `171204_pose3`. They are in raw format and need to be converted to KITTI format by following steps: 

- First, install Matlab CLI with Computer Vision Toolbox. Run `matlab/gen_train_val.m` then `matlab/gen_train_val_3D_lidar_ptclouds.m` to generate 2D LiDAR, 3D LiDAR data, and RGB images for each sequence. Please see the contents of `.m` files for detailed instructions of how to run them.
- Execute `python/panoptic_to_kitti.py` to generate data in KITTI format.  Please see `.py` file for detailed instructions for arguments.
- The generated dataset for each sequence will be under your specified output path passed as argument. Merge all datasets into one and place it under `kitti/object/testing` and `kitti/object/training` to conform with KITTI folder structure.

### Orientation dataset

This dataset annotates each pedestrian in the image with corresponding orientation angle value.
To generate this dataset, execute `python/orient/gen_orient_dataset.py`. Please see the `.py` file for detailed instructions of how to execute this file.

When the image is fed into the OrientNet, it first goes through the Mask R-CNN branch which detects each person in the image and associates pixels to the corresponding person (i.e., the mask of person). Unfortunately, the Mask R-CNN mask result is unordered so it is not possible to know which one of the masks (or which person in the image) corresponds to which orientation values in the label when there are more than one person in the image. As a result, the labelled orientation values of persons in the image need to be rearranged in the same sequence as the Mask R-CNN masks are in. The OrientNet can then link persons to the correct labelled values and, hence, learn properly and converge. The following steps were carried out to reorder the labelled values: 

1. Generate point clouds of entire persons (no 2D slicing)
2. Project all point clouds onto camera image
3. For each mask from Mask R-CNN:
    a. Extract only point clouds whose projections on the image overlap the mask (i.e. matching mask with point clouds)
	b. Among ground truth BEV (Bird-Eye-View) bounding boxes, match the point clouds with the box that has highest percentage of point cloud within.
4. After step 3, all the masks are matched with ground truth boxes, hence, with ground truth orientation values.

