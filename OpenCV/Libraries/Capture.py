import cv2
import os


def capture(frame, directory, name=""):
    image_path = os.path.join(directory, f"Captured-{name}.jpg")
    cv2.imwrite(image_path, frame)
