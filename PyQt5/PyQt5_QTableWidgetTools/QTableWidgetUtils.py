from PyQt5 import QtWidgets


class QTableWidgetUtils:
    @staticmethod
    def add_row(tableWidget, items):
        row = tableWidget.rowCount()
        tableWidget.setRowCount(row + 1)

        col = 0
        for y in items:
            cell = QtWidgets.QTableWidgetItem(str(y))
            tableWidget.setItem(row, col, cell)
            col += 1

    @staticmethod
    def get_rows(tableWidget):
        total_row = tableWidget.rowCount()
        total_columns = tableWidget.columnCount()

        row_items = []
        for row in range(0, total_row):
            column_items = []
            for column in range(0, total_columns):
                column_items.append(tableWidget.item(row, column).text())
            row_items.append(column_items)

        return row_items

    @staticmethod
    def remove_last_row(tableWidget):
        total_row = tableWidget.rowCount()
        tableWidget.removeRow(total_row - 1)

    @staticmethod
    def fill_table(tableWidget, items):
        total_row = tableWidget.rowCount()

        if total_row > 0:
            for _ in range(0, total_row):
                QTableWidgetUtils.remove_last_row(tableWidget)
            total_row = 0

        for item in items:
            QTableWidgetUtils.add_row(tableWidget, item)
