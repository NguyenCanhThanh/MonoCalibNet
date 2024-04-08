# M-Calib: A Monocular 3D Object Localization using 2D Estimates for Industrial Robot Vision System
Implementation code for our paper ["M-Calib: A Monocular 3D Object Localization using 2D Estimates for Industrial Robot Vision System"](https://assets.researchsquare.com/files/rs-4019542/v1_covered_5a75ac68-1bc8-4bdd-b2c5-8bbdb1eac8f1.pdf?c=1711473654)

## Our proposed:
<img src="https://github.com/thanhnguyencanh/MonoCalibNet/blob/main/image/Overview.png" width="750px">
 
## Requirements
* python 3.7
* torch 1.7.1
* tensorboard

## M-Calib Datasets
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

Option 1: Use a pre-trained model: [model](https://drive.google.com/drive/folders/1MS6DLxgKxo-FtC7TSTJN8WJCxhu8W3Fe?usp=sharing)

          Modify path in [cfg.py](https://github.com/thanhnguyencanh/MonoCalibNet/blob/main/cfg.py)

Option 2: Training new model using [Object_Detection](https://github.com/thanhnguyencanh/MonoCalibNet/blob/main/Object_Detection.ipynb) and [Instance_Segmentation](https://github.com/thanhnguyencanh/MonoCalibNet/blob/main/Instance_Segmentation.ipynb) 

          Convert to Onnx model

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
