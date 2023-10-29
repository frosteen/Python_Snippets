import cv2
from RetinaNet.detect_images import RetinaNet

retina_net = RetinaNet(min_size=2000, threshold=0.5)

frame = cv2.imread("Sample.jpg")

boxes, classes, result = retina_net.detect_image(frame)

print(classes, boxes)

cv2.imshow("frame", result)
cv2.waitKey(0)
