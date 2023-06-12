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
