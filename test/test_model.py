import pytest
import torch
from ultralytics import YOLO
from ultralytics.utils import ONLINE
MODEL = YOLO("yolov8n.pt")
# IS_TMP_WRITEABLE = is_tmp_writeble()

# This one also includes the profiling requirement
def test_model_profile():
    # Test profile=True model argument
    from ultralytics.nn.tasks import DetectionModel

    model = DetectionModel()  # build model
    im = torch.randn(1, 3, 64, 64)  # requires min imgsz=64
    _ = model.predict(im, profile=True)

@pytest.mark.skipif(not ONLINE, reason='environment is offline')
def test_track_stream():
    # Test YouTube streaming inference (short 10 frame video)
    # imgsz=96 is the image size
    model = MODEL
    model.predict('https://youtu.be/G17sBkb38XQ', imgsz=96, save=True)

# RUN THOSE IN COMMAND LINE
# pip install coverage
# coverage run -m pytest test/
# coverage report -i
# coverage html -i
# @pytest.mark.skipif(not IS_TMP_WRITEABLE, reason='directory is not writeable')