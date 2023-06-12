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
from ...util.raster_calc import RasterInfo


def calculate(input, luminance, sza_cosine, c):
    denominator = luminance + c
    return input * np.divide(
        sza_cosine + c,
        denominator,
        input.astype('float32')
    )


class CTopoCorrectionAlgorithm(LuminanceRegressionTopoCorrectionAlgorithm):
    @staticmethod
    def name():
        return "C-correction"

    @staticmethod
    def description():
        return r'<a href="https://www.tandfonline.com/doi/abs/10.1080/07038992.1982.10855028">C-correction</a>'

    def _process_band(self, ctx: QgisExecutionContext, band_idx: int):
        c = self._calculate_c(ctx, band_idx)

        return self._raster_calculate(
            ctx=ctx,
            calc_func=calculate,
            raster_infos=[
                RasterInfo("input", ctx.input_layer_path, band_idx + 1),
                RasterInfo("luminance", ctx.luminance_path, 1),
            ],
            out_file_postfix=band_idx,
            sza_cosine=ctx.sza_cosine(),
            c=c
        )

    def _calculate_c(self, ctx: QgisExecutionContext, band_idx: int) -> float:
        intercept, slope = self._get_linear_regression_coeffs(ctx, band_idx)
        return intercept / slope
