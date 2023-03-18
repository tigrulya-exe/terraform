# -*- coding: utf-8 -*-

"""
/***************************************************************************
 TerraformTopoCorrection
                                 A QGIS plugin
 Topographically correct provided input layer.
 Generated by Plugin Builder: http://g-sherman.github.io/Qgis-Plugin-Builder/
                              -------------------
        begin                : 2023-02-27
        copyright            : (C) 2023 by Tigran Manasyan
        email                : t.manasyan@g.nsu.ru
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/
"""

__author__ = 'Tigran Manasyan'
__date__ = '2023-02-27'
__copyright__ = '(C) 2023 by Tigran Manasyan'

# This will get replaced with a git SHA1 when you do a git archive

__revision__ = '$Format:%H$'

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

