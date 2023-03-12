# -*- coding: utf-8 -*-
"""
/***************************************************************************
 TerraformTopoCorrection
                                 A QGIS plugin
 Topographically corrects provided input layer.
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
 Utilities for handling meta information and procedures
"""

__author__ = 'Tigran Manasyan'
__date__ = '2023-02-27'
__copyright__ = '(C) 2023 by Tigran Manasyan'

import os
import sys


def ensure_import(package_name):
    """ Ensures that a dependency package could be imported. It is either already available in the QGIS environment or
    it is available in a subfolder `external` of this plugin and should be added to PATH
    """
    try:
        __import__(package_name)
    except ImportError:
        plugin_dir = _get_main_dir()
        external_path = os.path.join(plugin_dir, 'external')

        for wheel_name in os.listdir(external_path):
            if wheel_name.startswith(package_name):
                wheel_path = os.path.join(external_path, wheel_name)
                sys.path.append(wheel_path)
                return
        raise ImportError('Package {} not found'.format(package_name))


def _get_main_dir():
    """ Provides a path to the main plugin folder
    """
    utils_dir = os.path.dirname(__file__)
    return os.path.abspath(utils_dir)
