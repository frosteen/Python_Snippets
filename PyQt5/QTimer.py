from PyQt5.QtCore import QTimer


def set_timer(function=function):
    timer = QTimer()
    timer.timeout.connect(function)
    timer.start()
    return timer
