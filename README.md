# MLOps_project

This repository contains the group(Iris Jimenez, Nicolò Campagnoli, Meimingwei Li) project work for the MLOps course at LMU Munich. 


**1 Overall goal**: The goal is to perform object segmentation on images.\
**2 Framework:** As a framework [PyTorch Image Models](https://github.com/huggingface/pytorch-image-models) was used. It contains models, scripts and pre trained for a lot of state-of-the-art image models within computer vision.\
**3 Data:** We train our model on the [Pascal VOC 2012](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/index.html#devkit) dataset. It contains 20 foreground object classes and one background class. 1464 (train), 1449 (val), and 1456 (test) pixel-level annotated images.  \
**4 Deep Learning model used:**  We used the [Ultralytics YOLOv8 model](https://github.com/ultralytics/ultralytics).
