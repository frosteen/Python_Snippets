import os
import sys

import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from PyQt5 import QtCore, QtWidgets, uic


class Window(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()

    def closeEvent(self, event):
        close = QtWidgets.QMessageBox.question(
            self,
            "QUIT",
            "Are you sure want to quit?",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
        )
        if close == QtWidgets.QMessageBox.Yes:
            event.accept()
        else:
            event.ignore()

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

        # Instantiate FigureCanvas Start #
        self.figure = plt.figure()
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.verticalLayout.addWidget(self.toolbar)
        self.verticalLayout.addWidget(self.canvas)
        self.ax = self.figure.add_subplot(111)
        # Instantiate FigureCanvas End #

        # Plot Start #
        x_values = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        y_valeus = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        self.ax.clear()
        # self.ax.plot(x_values)
        self.ax.plot(x_values, y_valeus)
        self.figure.tight_layout()
        self.canvas.draw()
        # Plot End #


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    w = MainWindow()
    app.exec_()
