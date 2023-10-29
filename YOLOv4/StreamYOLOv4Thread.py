# import the necessary packages
from threading import Thread

import cv2


class StreamYOLOv4:
    def __init__(
        self,
        stream,
        yolo_cfg="YOLOv4_Files/yolov4-tiny.cfg",
        yolo_weights="YOLOv4_Files/yolov4-tiny.weights",
        yolo_size=(416, 416),
        CONFIDENCE_THRESHOLD=0.6,
        NMS_THRESHOLD=0.4,
        name="StreamYOLOv4",
    ):
        # get the passed stream
        self.stream = stream
        self.frame = None

        # initialize the thread name
        self.name = name

        # initialize the variable used to indicate if the thread should
        # be stopped
        self.stopped = False

        # YOLOv4 cfg and weights
        self.net = cv2.dnn.readNetFromDarknet(yolo_cfg, yolo_weights)

        # create model
        self.model = cv2.dnn_DetectionModel(self.net)

        # input parameters
        self.model.setInputParams(scale=1 / 255, size=yolo_size, swapRB=True)

        # YOLOv4 confidence and nms thresholds
        self.CONFIDENCE_THRESHOLD = CONFIDENCE_THRESHOLD
        self.NMS_THRESHOLD = NMS_THRESHOLD

        # initialize detections
        self.classes, self.scores, self.boxes = None, None, None

    def start(self):
        # start the thread to read frames from the video stream
        t = Thread(target=self.update, name=self.name, args=())
        t.daemon = True
        t.start()
        return self

    def update(self):
        # keep looping infinitely until the thread is stopped
        while True:
            # if the thread indicator variable is set, stop the thread
            if self.stopped:
                return

            # check if frame exist else continue
            if not self.stream.grabbed:
                continue

            # otherwise, read the next frame from the stream
            self.frame = self.stream.read()

            # detect using yolov4
            self.classes, self.scores, self.boxes = self.model.detect(
                self.frame,
                confThreshold=self.CONFIDENCE_THRESHOLD,
                nmsThreshold=self.NMS_THRESHOLD,
            )

    def read(self):
        return self.frame

    def detect(self):
        return (self.classes, self.scores, self.boxes)

    def stop(self):
        # indicate that the thread should be stopped
        self.stopped = True


if __name__ == "__main__":
    from imutils.video import VideoStream

    # get all classes in coco.names
    with open("YOLOv4_Files/coco.names", "r") as f:
        classnames = f.read().splitlines()

    # Using VideoStream from imutils (fast because of threading)
    capture = VideoStream(src=0).start()

    # initialize YOLOv4 in a another thread
    StreamYOLOv4 = StreamYOLOv4(
        capture,
        yolo_cfg="YOLOv4_Files/yolov4.cfg",
        yolo_weights="YOLOv4_Files/yolov4.weights",
        yolo_size=(416, 416),
    ).start()

    while True:
        # get properties from StreamYOLOv4
        frame = StreamYOLOv4.read()
        classes, scores, boxes = StreamYOLOv4.detect()

        # continue loop if frame does not exist
        if frame is None:
            continue

        # draw boxes
        if boxes is not None:
            for (classid, score, box) in zip(classes, scores, boxes):
                cv2.rectangle(
                    frame,
                    box,
                    color=(0, 255, 0),
                    thickness=2,
                )
                cv2.putText(
                    frame,
                    "%s: %.2f" % (classnames[classid], score),
                    (box[0], box[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color=(0, 255, 0),
                    thickness=2,
                )

        # Display the resulting frame
        cv2.imshow("frame", frame)

        # ESC to quit
        key = cv2.waitKey(1)
        if key & 0xFF == 27:
            break

    # When everything done, stop the capture
    StreamYOLOv4.stop()
    capture.stop()
    cv2.destroyAllWindows()
