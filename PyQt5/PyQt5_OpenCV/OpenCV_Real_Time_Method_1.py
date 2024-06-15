import os
import sys

import cv2
from imutils.video import VideoStream
from PyQt5 import QtCore, QtGui, QtWidgets, uic


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

        # start camera
        self.start_camera("label")

    # Camera Start #

    def start_camera(self, label_name):
        if os.name == "nt":  # check if running on windows
            self.capture = VideoStream(usePiCamera=False).start()
        else:  # else run with pi camera
            self.capture = VideoStream(
                usePiCamera=True, resolution=(1000, 1000)
            ).start()
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(lambda: self.__update_frame(label_name))
        self.timer.start()

    def stop_camera(self):
        self.capture.stop()
        self.timer.stop()

    def __update_frame(self, label_name):
        self.frame = self.capture.read()
        if self.frame is None:
            return
        self.frame = cv2.flip(self.frame, 1)
        self.__display_frames(self.frame, 1, label_name)

    def __display_frames(self, frame, window=1, label_name=None):
        self.out_frame = QtGui.QImage(
            frame,
            frame.shape[1],
            frame.shape[0],
            frame.strides[0],
            QtGui.QImage.Format_RGB888,
        )
        self.out_frame = self.out_frame.rgbSwapped()
        self.out_frame = QtGui.QPixmap.fromImage(self.out_frame)
        lbl = self.findChild(QtWidgets.QLabel, label_name)
        self.out_frame = self.out_frame.scaled(
            lbl.width(), lbl.height(), QtCore.Qt.KeepAspectRatio
        )
        lbl.setPixmap(self.out_frame)

    # Camera end #


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    w = MainWindow()
    app.exec_()
