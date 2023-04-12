import os
import warnings

from qgis.PyQt import uic

from ...computation import qgis_utils
from .FixedKeyedTableDialog import FixedKeyedTableDialog

pluginPath = os.path.split(os.path.dirname(__file__))[0]

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    WIDGET, BASE = uic.loadUiType(
        os.path.join(pluginPath, 'ui', 'widgetBaseSelector.ui'))


class FixedTablePanel(BASE, WIDGET):

    def __init__(self, param, parent=None):
        super(FixedTablePanel, self).__init__(parent)
        self.setupUi(self)

        self.leText.setEnabled(False)

        self.param = param

        # NOTE - table IS squashed to 1-dimensional!
        self.table = []
        for row in range(param.numberRows()):
            for col in range(len(param.headers())):
                self.table.append('0')

        self.leText.setText(
            self.tr('Fixed table {0}x{1}').format(param.numberRows(), len(param.headers())))

        self.btnSelect.clicked.connect(self.showFixedTableDialog)

    def updateSummaryText(self):
        self.leText.setText(self.tr('Fixed table {0}x{1}').format(
            len(self.table) // len(self.param.headers()), len(self.param.headers())))

    def setValue(self, value):
        self.table = value
        self.updateSummaryText()

    def showFixedTableDialog(self):
        dlg = FixedKeyedTableDialog(self.param.headers(), self._get_table_dict())
        dlg.exec_()
        if dlg.rettable is not None:
            self.setValue(dlg.rettable)
        dlg.deleteLater()

    def _get_table_dict(self):
        val_per_row = len(self.param.headers()) - 1
        return qgis_utils.get_table_dict(self.table, val_per_row)
