import os
import warnings
from typing import Any

from processing.gui.wrappers import WidgetWrapper
from qgis.PyQt import uic
from qgis.PyQt.QtCore import Qt
from qgis.PyQt.QtGui import QStandardItemModel, QStandardItem
from qgis.PyQt.QtWidgets import QDialog, QAbstractItemView, QHeaderView
from qgis.gui import QgsGui

from ...util import qgis_utils

pluginPath = os.path.split(os.path.dirname(__file__))[0]

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    PANEL_WIDGET, PANEL_BASE = uic.loadUiType(
        os.path.join(pluginPath, 'ui', 'widgetBaseSelector.ui'))
    DIALOG_WIDGET, DIALOG_BASE = uic.loadUiType(
        os.path.join(pluginPath, 'ui', 'DlgFixedTable.ui'))


class KeyedTableWidgetWrapper(WidgetWrapper):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def createWidget(self):
        return FixedTablePanel(self.parameterDefinition())

    def setValue(self, value):
        val_per_row = len(self.parameterDefinition().headers()) - 1
        table = qgis_utils.table_from_matrix_list(value, val_per_row)
        self.widget.setValue(table)

    def value(self):
        return qgis_utils.matrix_list_from_table(self.widget.table)

    def _get_table_dict(self):
        val_per_row = len(self.param.headers()) - 1
        return qgis_utils.table_from_matrix_list(self.table, val_per_row)


class FixedTablePanel(PANEL_BASE, PANEL_WIDGET):

    def __init__(self, param, parent=None):
        super(FixedTablePanel, self).__init__(parent)
        self.setupUi(self)

        self.leText.setEnabled(False)

        self.param = param

        self.table = dict()
        self.unchecked_table = dict()

        for row in range(param.numberRows()):
            self.table['0'] = ['0' for _ in range(1, len(param.headers()))]

        self.leText.setText(
            self.tr('Fixed table {0}x{1}').format(param.numberRows(), len(param.headers())))

        self.btnSelect.clicked.connect(self.showFixedTableDialog)

    def updateSummaryText(self, checked_keys):
        self.leText.setText(self.tr('{} option(s) selected').format(len(checked_keys)))

    def setValue(self, value):
        self.table = value
        self.updateSummaryText(self.table.keys())

    def showFixedTableDialog(self):
        dlg = FixedKeyedTableDialog(self.param.headers(), self.table, self.unchecked_table)
        dlg.exec_()
        if dlg.result_table is not None:
            checked_rows = {k: v for k, v in dlg.result_table.items() if k in dlg.checked_keys}
            self.unchecked_table = self._subtract_dict(dlg.result_table, checked_rows)
            self.setValue(checked_rows)
        dlg.deleteLater()

    def _subtract_dict(self, from_dict, to_dict):
        return {k: v for k, v in from_dict.items() if k not in to_dict}


class FixedKeyedTableDialog(DIALOG_BASE, DIALOG_WIDGET):
    KEY_COLUMN = 0

    def __init__(
            self,
            headers: list[str],
            table: dict[str, list[Any]],
            unchecked_table: dict[str, list[Any]],
            is_checkable=True):
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
        self.result_table = None

        self.checked_keys = set(table.keys())
        self._populate_table(table, unchecked_table)

    def _populate_table(self, table, unchecked_table):
        cols = len(self.headers)
        rows = len(table) + len(unchecked_table)

        model = QStandardItemModel(rows, cols)

        if self.is_checkable:
            model.itemChanged.connect(self._item_changed)

        # Set headers
        model.setHorizontalHeaderLabels(self.headers)

        start_row_idx = self._insert_checked_rows(model, table, Qt.Checked)
        self._insert_checked_rows(model, unchecked_table, Qt.Unchecked, start_row_idx)

        self.tblView.setModel(model)
        self.tblView.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)

    def _insert_checked_rows(
            self,
            model: QStandardItemModel,
            table: dict[str, list[Any]],
            check_state,
            start_row: int = 0):
        row_idx = start_row
        for key, values in table.items():
            key_item = QStandardItem(key)
            if self.is_checkable:
                key_item.setCheckable(self.is_checkable)
                key_item.setCheckState(check_state)

            model.setItem(row_idx, self.KEY_COLUMN, key_item)

            for idx, val in enumerate(values):
                val_item = QStandardItem(str(val))
                model.setItem(row_idx, idx + 1, val_item)
            row_idx += 1

        return row_idx

    def accept(self):
        cols = self.tblView.model().columnCount()
        rows = self.tblView.model().rowCount()

        self.result_table = dict()
        for row in range(rows):
            key = self._item(row)
            values = [self._item(row, col) for col in range(1, cols)]
            self.result_table[key] = values

        QDialog.accept(self)

    def reject(self):
        QDialog.reject(self)

    def _item(self, row, column=KEY_COLUMN):
        return self.tblView.model().item(row, column).text()

    def _item_changed(self, item):
        if item.column() != self.KEY_COLUMN or not self.is_checkable:
            return
        elif item.checkState() == Qt.Checked:
            self.checked_keys.add(item.text())
        else:
            self.checked_keys.discard(item.text())
