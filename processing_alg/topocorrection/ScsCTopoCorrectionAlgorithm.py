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

import numpy as np

from .CTopoCorrectionAlgorithm import CTopoCorrectionAlgorithm
from ..execution_context import QgisExecutionContext
from ...util.raster_calc import RasterInfo


class ScsCTopoCorrectionAlgorithm(CTopoCorrectionAlgorithm):
    @staticmethod
    def name():
        return "SCS+C"

    @staticmethod
    def description():
        return r'<a href="http://dx.doi.org/10.1109/TGRS.2005.852480">SCS+C</a>'

    def _process_band(self, ctx: QgisExecutionContext, band_idx: int):
        c = self._calculate_c(ctx, band_idx)

        def calculate(input_band, luminance, slope):
            denominator = luminance + c
            return input_band * np.divide(
                np.cos(slope) * ctx.sza_cosine() + c,
                denominator,
                out=input_band.astype('float32'),
                where=np.logical_and(denominator > 0, input_band > ctx.pixel_ignore_threshold)
            )

        return self._raster_calculate(
            ctx=ctx,
            calc_func=calculate,
            raster_infos=[
                RasterInfo("input_band", ctx.input_layer_path, band_idx + 1),
                RasterInfo("luminance", ctx.luminance_path, 1),
                RasterInfo("slope", ctx.slope_path, 1)
            ],
            out_file_postfix=band_idx
        )
