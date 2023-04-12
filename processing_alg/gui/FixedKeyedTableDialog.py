import os
import warnings
from typing import Any

from qgis.PyQt import uic
from qgis.PyQt.QtCore import Qt
from qgis.PyQt.QtGui import QStandardItemModel, QStandardItem
from qgis.PyQt.QtWidgets import QDialog, QAbstractItemView
from qgis.gui import QgsGui

pluginPath = os.path.split(os.path.dirname(__file__))[0]

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    WIDGET, BASE = uic.loadUiType(
        os.path.join(pluginPath, 'ui', 'DlgFixedTable.ui'))


class FixedKeyedTableDialog(BASE, WIDGET):
    KEY_COLUMN = 0

    def __init__(self, headers: list[str], table: dict[str, list[Any]], is_checkable=True):
        """
        Constructor for FixedTableDialog
        :param param: linked processing parameter
        :param table: initial table contents - squashed to 1-dimensional! <- NO!!!
        """
        super().__init__(None)

        self.setupUi(self)

        QgsGui.instance().enableAutoGeometryRestore(self)

        self.tblView.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.tblView.setSelectionMode(QAbstractItemView.ExtendedSelection)

        self.headers = headers
        self.is_checkable = is_checkable
        self.rettable = None

        self.checked_rows = set(range(len(table)))
        self.populateTable(table)

    def populateTable(self, table):
        cols = len(self.headers)
        rows = len(table)

        model = QStandardItemModel(rows, cols)
        model.itemChanged.connect(self.itemChanged)

        # Set headers
        model.setHorizontalHeaderLabels(self.headers)

        for row_idx, key in enumerate(table):
            key_item = QStandardItem(key)
            key_item.setCheckable(self.is_checkable)
            key_item.setCheckState(Qt.Checked)
            model.setItem(row_idx, self.KEY_COLUMN, key_item)

            for idx, val in enumerate(table[key]):
                val_item = QStandardItem(str(val))
                model.setItem(row_idx, idx + 1, val_item)

        self.tblView.setModel(model)

    def accept(self):
        cols = self.tblView.model().columnCount()
        rows = self.tblView.model().rowCount()
        # Table MUST BE 1-dimensional to match core QgsProcessingParameterMatrix expectations
        self.rettable = []
        for row in range(rows):
            if row not in self.checked_rows:
                continue
            for col in range(cols):
                self.rettable.append(str(self.tblView.model().item(row, col).text()))
        QDialog.accept(self)

    def reject(self):
        QDialog.reject(self)

    def itemChanged(self, item):
        if item.column() != self.KEY_COLUMN:
            return
        elif item.checkState() == Qt.Checked:
            self.checked_rows.add(item.row())
        else:
            self.checked_rows.discard(item.row())
