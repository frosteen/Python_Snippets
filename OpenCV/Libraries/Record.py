import os
from datetime import datetime

import cv2


class Record:
    def __init__(self, directory):
        self.directory = directory
        self.is_recording = False

    def setup(self, height, width, fps=10.0, quality=True):
        if self.is_recording == False:
            dt_string = datetime.now().strftime("%d%m%Y%H%M%S")
            record_path = os.path.join(self.directory, f"Record-{dt_string}.avi")
            if quality:
                # High quality but higher filesize
                fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            else:
                # Low quality but lower filesize
                fourcc = cv2.VideoWriter_fourcc(*"XVID")
            self.video_writer = cv2.VideoWriter(
                record_path, fourcc, fps, (width, height)
            )
            self.is_recording = True

    def record(self, frame):
        if frame is not None and self.is_recording:
            self.video_writer.write(frame)

    def stop(self):
        if self.is_recording:
            self.video_writer.release()
            self.is_recording = False
