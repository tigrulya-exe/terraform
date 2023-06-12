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

from .LuminanceRegressionTopoCorrectionAlgorithm import LuminanceRegressionTopoCorrectionAlgorithm
from ..execution_context import QgisExecutionContext
from ...util import gdal_utils
from ...util.raster_calc import RasterInfo


class VecaTopoCorrectionAlgorithm(LuminanceRegressionTopoCorrectionAlgorithm):
    @staticmethod
    def name():
        return "VECA"

    @staticmethod
    def description():
        return r'<a href="https://ieeexplore.ieee.org/abstract/document/4423917/">VECA</a>'

    def init(self, ctx: QgisExecutionContext):
        super().init(ctx)
        self.raster_means = gdal_utils.compute_band_means(ctx.input_layer_path)

    def _process_band(self, ctx: QgisExecutionContext, band_idx: int):
        intercept, slope = self._get_linear_regression_coeffs(ctx, band_idx)

        def calculate(input_band, luminance):
            denominator = slope * luminance + intercept
            result = input_band * np.divide(
                self.raster_means[band_idx],
                denominator,
                out=input_band.astype('float32'),
                where=np.logical_and(denominator > 0, input_band > ctx.pixel_ignore_threshold)
            )
            result[result <= 0] = self._calculate_zero_noise()
            return result

        return self._raster_calculate(
            ctx=ctx,
            calc_func=calculate,
            raster_infos=[
                RasterInfo("input_band", ctx.input_layer_path, band_idx + 1),
                RasterInfo("luminance", ctx.luminance_path, 1),
            ],
            out_file_postfix=band_idx
        )
