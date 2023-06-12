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
