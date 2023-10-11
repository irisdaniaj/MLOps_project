from ultralytics import YOLO

# Load a model
model = YOLO('yolov8n.pt')  # load a pretrained model (recommended for training)

# Train the model
results = model.train(data='VisDrone.yaml', epochs=5, imgsz=640) #since the goal of this project was not to have the best perfoming model and since the dataset was very heavy we only trained for five epochs, but feel free to modify the numbers of epochs
