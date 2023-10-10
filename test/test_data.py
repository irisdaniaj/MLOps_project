# Importing the required modules
import os 
import cv2
import pytest

@pytest.fixture
def dataset_path():
    return os.path.join(os.getcwd(), "mlops/data/Mydataset/VisDrone2019-DET-train/images/")

def load_data(dataset_path):
    images = []
    labels = []
    for filename in os.listdir(dataset_path):
        if filename.endswith(".jpg"):
            img = cv2.imread(os.path.join(dataset_path, filename))
            if img is not None:
                images.append(img)
        elif filename.endswith(".txt"):
            with open(os.path.join(dataset_path, filename), 'r') as file:
                labels.append(file.read().strip().split())
    return images, labels

def test_image_loading(dataset_path):
    images, _ = load_data(dataset_path)
    assert len(images) == 6471, f"Expected 6471 images, but got {len(images)} images"

def test_label_format(dataset_path):
    _, labels = load_data(dataset_path)
    for label in labels:
        assert len(label) ==10, "Label format is incorrect"
        assert 0 <= int(label[0]) <= 9, "First number in label is out of range"

def test_image_color(dataset_path):
    images, _ = load_data(dataset_path)
    for img in images:
        assert len(img.shape) == 3 and img.shape[2] == 3, "Image is not a colorful image"