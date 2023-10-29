# This program needs coco.names, yolov4.cfg and yolov4.weights

import time

import cv2
from imutils.video import VideoStream

# object detection settings
CONFIDENCE_THRESHOLD = 0.6
NMS_THRESHOLD = 0.4

# get all classes in coco.names
with open("YOLOv4_Files/coco.names", "r") as f:
    classnames = f.read().splitlines()

# YOLOv4 cfg and weights
net = cv2.dnn.readNetFromDarknet("YOLOv4_Files/yolov4-tiny.cfg", "YOLOv4_Files/yolov4-tiny.weights")

# create model
model = cv2.dnn_DetectionModel(net)

# default parameters (don't change)
model.setInputParams(scale=1 / 255, size=(416, 416), swapRB=True)

# Using VideoStream from imutils (fast because of threading)
cap = VideoStream(0).start()

while True:
    # Capture frame-by-frame
    frame = cap.read()

    if not cap.grabbed:
        continue

    # detection
    start = time.time()
    classes, scores, boxes = model.detect(
        frame, confThreshold=CONFIDENCE_THRESHOLD, nmsThreshold=NMS_THRESHOLD
    )
    end = time.time()

    # drawing
    start_drawing = time.time()
    for (classid, score, box) in zip(classes, scores, boxes):
        cv2.rectangle(
            frame,
            box,
            color=(0, 255, 0),
            thickness=2,
        )
        label = "%s: %.2f" % (classnames[classid], score)
        cv2.putText(
            frame,
            label,
            (box[0], box[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color=(0, 255, 0),
            thickness=2,
        )
    end_drawing = time.time()

    # show fps
    fps_label = "FPS: %.2f (excluding drawing time of %.2fms)" % (
        1 / (end - start),
        (end_drawing - start_drawing) * 1000,
    )
    cv2.putText(frame, fps_label, (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

    # Display the resulting frame
    cv2.imshow("frame", frame)

    # ESC to quit
    key = cv2.waitKey(1)
    if key & 0xFF == 27:
        break

# When everything done, stop the capture
cap.stop()
cv2.destroyAllWindows()
