from processing.gui.wrappers import WidgetWrapper

from qgis.PyQt import QtWidgets, QtGui

from .FixedTablePanel import FixedTablePanel


class NewFixedTableWidgetWrapper1(WidgetWrapper):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        """
        .. deprecated:: 3.4
        Do not use, will be removed in QGIS 4.0
        """

        from warnings import warn
        warn("FixedTableWidgetWrapper is deprecated and will be removed in QGIS 4.0", DeprecationWarning)

    def createWidget(self):
        return FixedTablePanel(self.parameterDefinition())

    def setValue(self, value):
        self.widget.setValue(value)

    def value(self):
        return self.widget.table


class FixedTablePanel1(QtWidgets.QWidget):

    def __init__(self, param, parent=None):
        super().__init__(parent=parent)
        self.setupUi(self)

        self.tableModel = QtGui.QStandardItemModel(self)
        self.tableModel.itemChanged.connect(self.itemChanged)

        item = QtGui.QStandardItem("Click me")
        item.setCheckable(True)
        self.tableModel.appendRow(item)

        self.mainLayout = QtWidgets.QVBoxLayout()
        self.setLayout(self.mainLayout)

        self.tableView = QtWidgets.QTableView()
        self.tableView.setModel(self.tableModel)
        self.mainLayout.addWidget(self.tableView)

    def setValue(self, value):
        pass

    def itemChanged(self, item):
        print("Item {!r} checkState: {}".format(item.text(), item.checkState()))
