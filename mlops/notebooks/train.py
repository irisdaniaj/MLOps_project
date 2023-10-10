from ultralytics import YOLO

# Initialize model
yolo_model = YOLO("yolov8n.pt")

costumdata = "C:/Users/ra78lof/yolov5/data/VisDrone.yaml"  # dataset.yaml
# Run Train
yolo_model.train(data=costumdata, epochs=50)

# customdata.yaml

# debug
print(yolo_model)


# This was found in the ultralytics github repo
from ultralytics.utils.benchmarks import ProfileModels

ProfileModels(['yolov8n.yaml', 'yolov8s.yaml'], imgsz=640).profile()

# Some other way to do it
import torch
from torch.profiler import profile, ProfilerActivity
from torch.profiler import profile, tensorboard_trace_handler

# A toy input for testing the model
inputs = torch.randn(5, 3, 224, 224)
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True, on_trace_ready=tensorboard_trace_handler("./log/yolov8")) as prof:
    yolo_model(inputs)

# Call this in command line to see the results
# tensorboard --logdir= ./log/yolov8
