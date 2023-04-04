# -*- coding: utf-8 -*-
"""
/***************************************************************************
 TerraformTopoCorrection
                                 A QGIS plugin
 Topographically corrects provided input layer.
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
 This script initializes the plugin, making it known to QGIS.
"""

__author__ = 'Tigran Manasyan'
__date__ = '2023-02-27'
__copyright__ = '(C) 2023 by Tigran Manasyan'


# noinspection PyPep8Naming
def classFactory(iface):  # pylint: disable=invalid-name
    """Load TerraformTopoCorrection class from file TerraformTopoCorrection.

    :param iface: A QGIS interface instance.
    :type iface: QgsInterface
    """
    #
    from .dependencies import ensure_import
    ensure_import("numpy_groupies")
    ensure_import("tabulate")
    ensure_import("xlsxwriter")

    from .terraform import TerraformTopoCorrectionPlugin
    return TerraformTopoCorrectionPlugin(iface)
