# Some other way to do it
from ultralytics import YOLO

model_name = "yolov8n" #@param {type:"string"}

model = YOLO(f"{model_name}.pt")
import torch
from torch.profiler import profile, ProfilerActivity
from torch.profiler import profile, tensorboard_trace_handler

# A toy input for testing the model
inputs = torch.randn(1, 3, 224, 224)
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True, on_trace_ready=tensorboard_trace_handler("./log/yolov8_2")) as prof:
    model(inputs)

# Call this in command line to see the results
# tensorboard --logdir= ./log/yolov8
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
prof.export_chrome_trace("C:/Users/LMMISTA-WAP265/YOLOV8_mlopstrace_3.json")