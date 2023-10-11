from fastapi import FastAPI, UploadFile
from fastapi.responses import StreamingResponse
from PIL import Image
import io
from ultralytics import YOLO
import numpy as np

# Load the pretrained YOLOv8 model
model = YOLO("yolov8n.pt")

app = FastAPI()

def detect_objects(image: Image.Image) -> Image.Image:
    """
    Detect objects in an image using the YOLOv8 model and return an image with bounding boxes.

    Args:
        image (PIL.Image.Image): The input image.

    Returns:
        PIL.Image.Image: The result image with bounding boxes.
    """
    # Convert PIL image to numpy array
    np_image = np.array(image)

    # Run object detection
    results = model(np_image)

    # Render result image with bounding boxes
    result_image = results.render()[0]

    # Convert result numpy array back to PIL image
    result_pil_image = Image.fromarray(result_image)

    return result_pil_image

@app.post("/detect_objects")
async def detect_objects_endpoint(file: UploadFile):
    """
    Endpoint to accept an image file, run object detection on it, and return an image with bounding boxes.

    Args:
        file (fastapi.UploadFile): The uploaded image file.

    Returns:
        fastapi.responses.StreamingResponse: The response containing the result image with bounding boxes.
    """
    # Read the image file
    image_bytes = await file.read()

    # Load image bytes into a PIL Image object
    image = Image.open(io.BytesIO(image_bytes))

    # Run object detection
    result_image = detect_objects(image)

    # Prepare the result image for streaming in the response
    result_bytes_io = io.BytesIO()
    result_image.save(result_bytes_io, format="JPEG")
    result_bytes_io.seek(0)

    return StreamingResponse(result_bytes_io, media_type="image/jpeg")

