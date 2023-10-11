from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
from ultralytics import YOLO
from PIL import Image
import io

# Load the pretrained YOLOv8 model
model = YOLO("yolov8n.pt")

app = FastAPI()

@app.get('/')
async def main():
    return 'ok'

@app.post("/detect_objects")
async def detect_objects_endpoint(file: UploadFile):
    # Read the image file directly into memory
    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data))

    # Now you can use the 'model' object to run inference on the image
    results = model(image)

    # Show the results
    for r in results:
        im_array = r.plot()  # plot a BGR numpy array of predictions
        im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
        im.show()  # show image
        im.save('results.jpg')  # save image

    # Convert the image array to a byte stream
    byte_io = io.BytesIO()
    im.save(byte_io, format='JPEG')

    # Seek to the beginning of the byte stream
    byte_io.seek(0)

    return StreamingResponse(byte_io, media_type="image/jpeg")