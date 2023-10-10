# -*- coding: utf-8 -*-
import torch
import torch_geometric.data
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torchvision.datasets import VOCDetection
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset
import os 
from pathlib import Path
import argparse  # Add this import

# Load VOC Dataset
voc_train = VOCDetection(root="data", year="2007", image_set="train", download=True)
voc_valid = VOCDetection(root="data", year="2007", image_set="val", download=True)
voc_test = VOCDetection(root="data", year="2007", image_set="test", download=True)

# Custom VOC Dataset Class
class CustomVOCDetection(Dataset):
    def __init__(self, voc_dataset, transforms=None):
        self.voc_dataset = voc_dataset
        self.transforms = transforms

    def __len__(self):
        return len(self.voc_dataset)

    def __getitem__(self, idx):
        img, target = self.voc_dataset[idx]
        boxes = target["annotation"]["object"]
        labels = []
        bboxes = []
        for box in boxes:
            xmin = float(box["bndbox"]["xmin"])
            ymin = float(box["bndbox"]["ymin"])
            xmax = float(box["bndbox"]["xmax"])
            ymax = float(box["bndbox"]["ymax"])
            bboxes.append([xmin, ymin, xmax, ymax])
            labels.append(box['name'])  # assuming all objects are of the same class

        if self.transforms:
            transformed = self.transforms(image=img, bboxes=bboxes, labels=labels)
            img = transformed["image"]
            bboxes = transformed["bboxes"]
            labels = transformed["labels"]

        return img, bboxes, labels

# Visualization Function
def visualize(img, bboxes):
    _, ax = plt.subplots(1)
    ax.imshow(img)
    for bbox in bboxes:
        rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2]-bbox[0], bbox[3]-bbox[1], linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
    plt.show()

# Example Usage of CustomVOCDetection
dataset = CustomVOCDetection(voc_train)
img, bboxes, labels = dataset[0] 

# Convert the PIL image to a numpy array
img_np = np.array(img) 

# This is not a tensor yet
# The shape is (H, W, C)
print(img_np.shape)
print(type(img_np))
visualize(img_np, bboxes)

# Data Conversion Function
def visdrone2yolo(dir):
    from PIL import Image
    from tqdm import tqdm

    def convert_box(size, box):
        # Convert VisDrone box to YOLO xywh box
        dw = 1. / size[0]
        dh = 1. / size[1]
        return (box[0] + box[2] / 2) * dw, (box[1] + box[3] / 2) * dh, box[2] * dw, box[3] * dh

    (dir / 'labels').mkdir(parents=True, exist_ok=True)  # make labels directory
    pbar = tqdm((dir / 'annotations').glob('*.txt'), desc=f'Converting {dir}')
    for f in pbar:
        img_size = Image.open((dir / 'images' / f.name).with_suffix('.jpg')).size
        lines = []
        with open(f, 'r') as file:  # read annotation.txt
            for row in [x.split(',') for x in file.read().strip().splitlines()]:
                if row[4] == '0':  # VisDrone 'ignored regions' class 0
                    continue
                cls = int(row[5]) - 1  # class_num - 1
                box = convert_box(img_size, tuple(map(int, row[:4])))
                lines.append(f"{cls} {' '.join(f'{x:.6f}' for x in box)}\n")
                with open(str(f).replace(os.sep + 'annotations' + os.sep, os.sep + 'labels' + os.sep), 'w') as fl:
                    fl.writelines(lines)  # write label.txt

# Example Usage of visdrone2yolo
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert VisDrone annotations to YOLO labels.")
    parser.add_argument("dir", type=str, help="The directory containing VisDrone annotations.")
    
    args = parser.parse_args()
    
    dataset_path = Path(args.dir)
    visdrone2yolo(dataset_path)

