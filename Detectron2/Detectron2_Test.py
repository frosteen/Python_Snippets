import cv2
from Detectron2 import Detector

detector = Detector(model_type="OD", threshold=0.5, use_gpu=True)
image = cv2.imread("Sample.jpg")
output, class_names_boxes = detector.on_image(image)

print(class_names_boxes)

cv2.imshow("Result", output)
cv2.waitKey(0)
