# define the relative path for the images
train_images_directory = "./Mydataset/VisDrone2019-DET-train/images"
test_images_directory = "./Mydataset/VisDrone2019-DET-test-dev/images"
train_images_directory = "C:/Users/LMMISTA-WAP265/YOLOV8_mlops/mlops/data/Mydataset/VisDrone2019-DET-train"
test_images_directory = "C:/Users/LMMISTA-WAP265/YOLOV8_mlops/mlops/data/Mydataset/VisDrone2019-DET-test"

import os
import random
from PIL import Image
import torch
import pandas as pd

# requires transformers package: pip install transformers
from transformers import CLIPProcessor, CLIPModel
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# url = "http://images.cocodataset.org/val2017/000000039769.jpg"
# image = Image.open(requests.get(url, stream=True).raw)

# set either text=None or images=None when only the other is needed
# inputs = processor(text=["a photo of a cat", "a photo of a dog"], images=image, return_tensors="pt", padding=True)

# img_features = model.get_image_features(inputs['pixel_values'])
# text_features = model.get_text_features(inputs['input_ids'], inputs['attention_mask'])


def load_random_images(directory, num_images=10):
    all_images = [
        f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))
    ]
    random_images = random.sample(all_images, num_images)
    return [
        Image.open(os.path.join(directory, image_name)) for image_name in random_images
    ]


def extract_features(images, model, processor):
    inputs = processor(images=images, return_tensors="pt", padding=True)
    img_features = model.get_image_features(inputs["pixel_values"])
    return img_features


def convert_to_dataframe(tensor):
    return pd.DataFrame(tensor.detach().numpy())


def generate_drift_report(reference, current):
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference, current_data=current)
    report.save_html("report.html")


train_images = load_random_images(train_images_directory)
test_images = load_random_images(test_images_directory)

train_features = extract_features(train_images, model, processor)
test_features = extract_features(test_images, model, processor)

train_df = convert_to_dataframe(train_features)
test_df = convert_to_dataframe(test_features)

# Generate the report for data drift
generate_drift_report(train_df, test_df)
