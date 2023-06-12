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

from qgis.PyQt.QtCore import QCoreApplication
from qgis.core import (QgsProcessingAlgorithm, QgsProcessingParameterDefinition)


class TerraformProcessingAlgorithm(QgsProcessingAlgorithm):

    def tr(self, string):
        """
        Returns a translatable string with the self.tr() function.
        """
        return QCoreApplication.translate('Processing', string)

    def group(self):
        """
        Returns the name of the group this algorithm belongs to.
        """
        return self.tr('Topographic correction')

    def groupId(self):
        """
        Returns the unique ID of the group this algorithm belongs
        to.
        """
        return 'topocorrection'

    def _additional_param(self, param):
        param.setFlags(param.flags() | QgsProcessingParameterDefinition.FlagAdvanced)
        self.addParameter(param)
