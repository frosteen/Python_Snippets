import os
import sys

import cv2
import numpy as np
from imutils.video import VideoStream
from PyQt5 import QtCore, QtGui, QtWidgets, uic


class VideoThread(QtCore.QThread):
    change_pixmap_signal = QtCore.pyqtSignal(np.ndarray)

    def __init__(self, label):
        super().__init__()
        self.label = label
        self._run_flag = True
        self.change_pixmap_signal.connect(self.update_image)

    def run(self):
        # capture from web cam
        if os.name == "nt":  # check if running on windows
            self.capture = VideoStream(usePiCamera=False).start()
        else:  # else run with pi camera
            self.capture = VideoStream(
                usePiCamera=True, resolution=(1000, 1000)
            ).start()
        while self._run_flag:
            self.cv_img = self.capture.read()
            if self.cv_img is None:
                continue
            self.cv_img = cv2.flip(self.cv_img, 1)
            self.change_pixmap_signal.emit(self.cv_img)
        # shut down capture system
        self.capture.stop()

    def stop(self):
        """Sets run flag to False and waits for thread to finish"""
        self._run_flag = False
        self.wait()

    @QtCore.pyqtSlot(np.ndarray)
    def update_image(self, cv_img):
        """Updates the image_label with a new opencv image"""
        qt_img = self.convert_cv_qt(cv_img)
        self.label.setPixmap(qt_img)

    def convert_cv_qt(self, cv_img):
        """Convert from an opencv image to QPixmap"""
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QtGui.QImage(
            rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888
        )
        return QtGui.QPixmap.fromImage(convert_to_Qt_format)


class Window(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()

    def center(self):
        frame_gm = self.frameGeometry()
        screen = QtWidgets.QApplication.desktop().screenNumber(
            QtWidgets.QApplication.desktop().cursor().pos()
        )
        center_point = QtWidgets.QApplication.desktop().screenGeometry(screen).center()
        frame_gm.moveCenter(center_point)
        self.move(frame_gm.topLeft())


class MainWindow(Window):
    def __init__(self):
        super().__init__()
        uic.loadUi("MainWindow.ui", self)
        self.setFixedSize(self.size())
        self.setWindowFlags(QtCore.Qt.FramelessWindowHint)
        self.center()
        self.show()

        # create the video capture thread
        self.thread = VideoThread(self.label)

        # start the thread
        self.thread.start()


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    w = MainWindow()
    app.exec_()
