from PyQt5 import QtCore


class DoThreading(QtCore.QThread):
    def __init__(self, func, finished_func=None):
        QtCore.QThread.__init__(self)
        self.func = func
        self.daemon = True
        self.finished_func = finished_func
        if self.finished_func is not None:
            self.finished.connect(finished_func)

    def __del__(self):
        self.wait()

    def run(self):
        self.func()
