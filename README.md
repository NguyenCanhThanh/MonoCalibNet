# M-Calib: A Monocular 3D Object Localization using 2D Estimates for Industrial Robot Vision System
Implementation code for our paper [" M-Calib: A Monocular 3D Object Localization using 2D Estimates for Industrial Robot Vision System"]

## Our proposed:
<img src="https://github.com/thanhnguyencanh/MonoCalibNet/blob/main/image/Overview.png" width="750px">
 
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

* 1. Step 1: Clone this repo
```
git clone https://github.com/thanhnguyencanh/MonoCalibNet
cd MonoCalibNet
```
* 2. Step 2: Creating a model

Option 1: Use a pre-trained model

Option 2: Training new model

* 3. Step 3: Testing

```
mkdir model
mkdir dataset
```
 + Set pat in cfg.py
```
python3 run_chessboard_yolov5.py
```


## Citation
```
@article{canh2024m,
  title={M-Calib: A Monocular 3D Object Localization using 2D Estimates for Industrial Robot Vision System},
  author={Canh, Thanh Nguyen and HoangVan, Xiem and others},
  year={2024}
}
```
