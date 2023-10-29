import torchvision
import torch
import cv2
import RetinaNet.detect_utils as detect_utils
import numpy as np


class RetinaNet:
    def __init__(self, min_size=1200, threshold=0.5):
        # download or load the model from disk
        self.model = torchvision.models.detection.retinanet_resnet50_fpn(
            pretrained=True, min_size=min_size
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # load the model onto the computation device
        self.model.eval().to(self.device)
        self.threshold = threshold

    def detect_image(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_array = np.array(image)
        image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
        boxes, classes = detect_utils.predict(
            image, self.model, self.device, self.threshold
        )
        result = detect_utils.draw_boxes(boxes, classes, image_array)
        return boxes, classes, result
