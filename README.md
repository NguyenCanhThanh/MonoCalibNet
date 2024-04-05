# M-Calib: A Monocular 3D Object Localization using 2D Estimates for Industrial Robot Vision System
Implementation code for our paper [" M-Calib: A Monocular 3D Object Localization using 2D Estimates for Industrial Robot Vision System"]
 
## Requirements
* python 3.7
* torch 1.7.1
* tensorboard

## OGM-Datasets
There are two different datasets collected by the authors
The related datasets can be found at:

* 1. Object Detection dataset: (https://app.roboflow.com/uet-jvl1l/m-calib/1).
* 2. Object Segmentation dataset: (https://app.roboflow.com/uet-jvl1l/mcalibsegment/1).

## Usage: M-Calib (the inference)

Option 1: Use pre-trained model

 Option 2: Training new model
 
 +) model Mask-RCNN using by PaddleDetection
 
 +) model Yolov5
 
 Create dir: model,dataset
 
 Set path in cfg.py
 
 
Run run_chessboard_yolov5.py

## Citation
```
@article{canh2024m,
  title={M-Calib: A Monocular 3D Object Localization using 2D Estimates for Industrial Robot Vision System},
  author={Canh, Thanh Nguyen and HoangVan, Xiem and others},
  year={2024}
}
```
