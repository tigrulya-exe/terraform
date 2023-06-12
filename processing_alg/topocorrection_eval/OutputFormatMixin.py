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

from enum import Enum

from qgis.core import QgsProcessingParameterEnum

from ..execution_context import QgisExecutionContext


class OutputFormatMixin:
    class OutputFormat(str, Enum):
        PNG = 'png'
        SVG = 'svg'

    def is_supported_by_qgis(self, fmt: OutputFormat):
        return fmt == self.OutputFormat.PNG

    def format_param_supported_by_qgis(self, ctx: QgisExecutionContext):
        fmt = self.get_output_format_param(ctx)
        return fmt == self.OutputFormat.PNG

    def get_output_format_param(self, ctx: QgisExecutionContext):
        return self.parameterAsEnumString(ctx.qgis_params, 'OUTPUT_FORMAT', ctx.qgis_context)

    def output_format_param(self):
        return QgsProcessingParameterEnum(
            'OUTPUT_FORMAT',
            'Output format',
            options=[f for f in self.OutputFormat],
            allowMultiple=False,
            defaultValue='png',
            usesStaticStrings=True
        )
