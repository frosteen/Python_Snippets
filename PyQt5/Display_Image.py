from PyQt5 import QtGui, QtWidgets, QtCore


def display_image(parent, image, label_name="Img_Label"):
    parent.output_image = QtGui.QImage(
        image,
        image.shape[1],
        image.shape[0],
        image.strides[0],
        QtGui.QImage.Format_RGB888,
    )
    parent.output_image = parent.output_image.rgbSwapped()
    parent.output_image = QtGui.QPixmap.fromImage(parent.output_image)
    img_label = parent.findChild(QtWidgets.QLabel, label_name)
    parent.output_image = parent.output_image.scaled(
        img_label.width(), img_label.height(), QtCore.Qt.KeepAspectRatio
    )
    img_label.setPixmap(parent.output_image)
    return parent.output_image
