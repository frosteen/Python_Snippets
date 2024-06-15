from PyQt5 import QtWidgets


class CustomQtWidgets:
    @staticmethod
    def do_message(parent, title="Information", text=None):
        QtWidgets.QMessageBox.information(
            parent,
            title,
            text,
            QtWidgets.QMessageBox.Ok,
        )

    @staticmethod
    def do_get_question(parent, title="Question", text=""):
        result = QtWidgets.QMessageBox.question(
            parent,
            title,
            text,
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            QtWidgets.QMessageBox.No,
        )
        if result == QtWidgets.QMessageBox.Yes:
            return True
        return False

    @staticmethod
    def do_get_item(parent, title="Input", label="", items=[]):
        inputted, ok_pressed = QtWidgets.QInputDialog.getItem(
            parent,
            title,
            label,
            items,
            0,
            False,
        )
        if inputted and ok_pressed:
            return inputted
        return None

    @staticmethod
    def do_get_filename(
        parent, caption="Open File", directory="", filter="All Files (*.*)"
    ):
        filename, _ = QtWidgets.QFileDialog.getOpenFileName(
            parent,
            caption,
            directory,
            filter,
        )
        return filename

    @staticmethod
    def do_save_filename(
        parent, caption="Save As", directory="", filter="All Files (*.*)"
    ):
        filename, _ = QtWidgets.QFileDialog.getSaveFileName(
            parent,
            caption,
            directory,
            filter,
        )
        return filename
