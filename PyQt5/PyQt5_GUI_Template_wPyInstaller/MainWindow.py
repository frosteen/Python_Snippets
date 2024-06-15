import os
import sys

from PyQt5 import QtCore, QtWidgets, uic

from QTThreading import DoThreading
from CustomQtWidgets import CustomQtWidgets


class Window(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()

    def closeEvent(self, event):
        is_close = CustomQtWidgets.do_get_question(
            self, "Quit", "Are you sure want to quit?"
        )

        if is_close:
            event.accept()
        else:
            event.ignore()

    def center(self):
        qtRectangle = self.frameGeometry()
        centerPoint = QtWidgets.QDesktopWidget().availableGeometry().center()
        qtRectangle.moveCenter(centerPoint)
        self.move(qtRectangle.topLeft())

    def resource_path(self, relative_path):
        try:
            base_path = sys._MEIPASS
        except Exception:
            base_path = os.path.abspath(".")
        return os.path.join(base_path, relative_path)


class MainWindow(Window):
    def __init__(self):
        super().__init__()
        uic.loadUi(self.resource_path("MainWindow.ui"), self)
        self.setFixedSize(self.size())
        self.setWindowFlags(QtCore.Qt.FramelessWindowHint)
        self.center()
        self.show()


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    w = MainWindow()
    app.exec_()
