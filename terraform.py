#!/usr/bin/env python
""" Terraform QGIS plugin.

This program is free software: you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free Software
Foundation, either version 3 of the License, or (at your option) any later
version.

This program is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with
this program. If not, see <http://www.gnu.org/licenses/>.
"""

__author__ = 'Tigran Manasyan'
__copyright__ = '(C) 2023 by Tigran Manasyan'
__license__ = "GPLv3"

from qgis.PyQt.QtWidgets import QMessageBox, QAction
from qgis.core import (
    QgsApplication
)

from .terraform_processing_provider import TerraformProcessingProvider


class TerraformTopoCorrectionPlugin(object):

    def __init__(self, iface):
        self.provider = None
        # save reference to the QGIS interface
        self.iface = iface

    def initProcessing(self):
        """Init Processing provider for QGIS >= 3.8."""
        self.provider = TerraformProcessingProvider()
        QgsApplication.processingRegistry().addProvider(self.provider)

    def initGui(self):
        self.initProcessing()

        self.key_action = QAction("Test Plugin", self.iface.mainWindow())
        self.iface.registerMainWindowAction(self.key_action, "Ctrl+I")  # action triggered by Ctrl+I
        self.iface.addPluginToMenu("&Test plugins", self.key_action)
        self.key_action.triggered.connect(self.key_action_triggered)

    def key_action_triggered(self):
        QMessageBox.information(self.iface.mainWindow(), "Ok", "You pressed Ctrl+I")

    def unload(self):
        QgsApplication.processingRegistry().removeProvider(self.provider)

        self.iface.unregisterMainWindowAction(self.key_action)
