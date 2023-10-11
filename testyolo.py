from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
from ultralytics import YOLO
from PIL import Image

# Load the pretrained YOLOv8 model
model = YOLO("yolov8n.pt")

app = FastAPI()

@app.get('/')
async def main():
    return 'ok'

@app.post("/detect_objects")
async def detect_objects_endpoint(file: UploadFile):


    # Save image to a file
    image_path = "C:/Users/A/Downloads/cat.png"

    results = model(image_path)

    # Show the results
    for r in results:
        im_array = r.plot()  # plot a BGR numpy array of predictions
        im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
        im.show()  # show image
        im.save('results.jpg')  # save image
    return StreamingResponse(im_array, media_type="image/jpeg")
