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

from .TopoCorrectionAlgorithm import TopoCorrectionAlgorithm
from ..execution_context import QgisExecutionContext
from ...util.raster_calc import RasterInfo


class CosineCTopoCorrectionAlgorithm(TopoCorrectionAlgorithm):
    def __init__(self):
        super().__init__()
        self.luminance_mean = None

    @staticmethod
    def name():
        return "COSINE-C"

    @staticmethod
    def description():
        return '<a href="https://www.asprs.org/wp-content/uploads/pers/1989journal/sep/1989_sep_1303-1309.pdf">Cosine-C</a>'

    def init(self, ctx: QgisExecutionContext):
        super().init(ctx)
        self.luminance_mean = np.mean(ctx.luminance_bytes)

    def _process_band(self, ctx: QgisExecutionContext, band_idx: int):
        def calculate(input_band, luminance):
            return input_band * (1 + np.divide(
                self.luminance_mean - luminance,
                self.luminance_mean,
                out=input_band.astype('float32'),
                where=input_band > ctx.pixel_ignore_threshold
            ))

        return self._raster_calculate(
            ctx=ctx,
            calc_func=calculate,
            raster_infos=[
                RasterInfo("input_band", ctx.input_layer_path, band_idx + 1),
                RasterInfo("luminance", ctx.luminance_path, 1),
            ],
            out_file_postfix=band_idx
        )
