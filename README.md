# yolov5-segmentation
yolov5 for semantic segmentation

## flexible-yolov5

Based on [flexible-yolov5](https://github.com/yl305237731/flexible-yolov5).


## Table of contents
* [Features](#features)
* [Prerequisites](#prerequisites)
* [Getting Started](#getting-started)
    * [Dataset Preparation](#dataset-preparation)
    * [Training and Testing](#Training-and-Testing)
* [Reference](#Reference)


## Features
- Provide model structure, such as backbone, neck, head, can modify the network flexibly and conveniently
- yolov5s, yolov5m, yolov5l, yolov5x
- segmentation head

## Prerequisites

please refer requirements.txt

## Getting Started

### Dataset Preparation

1. Download coco or cityscapes dataset.
2. Modify your dataset path in  `configs/data.yaml`.


### Training and Testing

For training and Testing, it's same like yolov5.

### Training

You can modify your setup in `train_cityscapes.sh`.
```shell script
$ ./train_cityscapes.sh
```
Tensorboard is automatically started while training. You can see the visualizition results in tensorboard.

## Reference

* [ultralytics/yolov5](https://github.com/ultralytics/yolov5)
