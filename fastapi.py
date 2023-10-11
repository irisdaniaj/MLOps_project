from fastapi import FastAPI, UploadFile, HTTPException
import cv2
import numpy as np
from ultralytics import YOLO

# Load a model
model = YOLO('yolov8n.pt')  # load an official model
# model = YOLO('path/to/best.pt')  # load a custom trained model

# Export the model
model.export(format='onnx')


app = FastAPI()

ALLOWED_MIME_TYPES = ["image/jpeg", "image/png", "image/bmp"]


net = cv2.dnn.readNet("yolov8n.onnx")
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]


@app.post("/predict/")
async def predict(file: UploadFile = UploadFile(...)):
    # Ensure it's an image
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload an image file!")
    
    if file.content_type not in ALLOWED_MIME_TYPES:
        raise HTTPException(
            status_code=400, detail="Please upload jpg, png and bmp images only."
        )

    if file.content_type.startswith("image/"):
        data = await file.read()
        nparr = np.fromstring(data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

       
        blob = cv2.dnn.blobFromImage(
            img, 0.00392, (416, 416), (0, 0, 0), True, crop=False
        )
        net.setInput(blob)
        outs = net.forward(output_layers)

        
        class_ids = []
        confidences = []
        boxes = []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:  
                    center_x = int(detection[0] * img.shape[1])
                    center_y = int(detection[1] * img.shape[0])
                    w = int(detection[2] * img.shape[1])
                    h = int(detection[3] * img.shape[0])
                    boxes.append([center_x, center_y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
        return {"class_ids": class_ids, "confidences": confidences}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
