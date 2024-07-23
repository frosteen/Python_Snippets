import torch
import cv2

# Model
model = torch.hub.load(
    "ultralytics/yolov5",
    "yolov5s",
)  # or yolov5m, yolov5l, yolov5x, custom

model.to("cuda")
model.eval()
model.conf = 0.5


# Images
img = "https://ultralytics.com/images/zidane.jpg"  # or file, Path, PIL, OpenCV, numpy, list

# Inference
with torch.no_grad():
    results = model(img)

# Results
results.show()  # or .show(), .save(), .crop(), .pandas(), etc.
