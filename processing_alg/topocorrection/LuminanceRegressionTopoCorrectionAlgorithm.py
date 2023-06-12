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

from random import randint

import numpy as np

from .TopoCorrectionAlgorithm import TopoCorrectionAlgorithm
from ..execution_context import QgisExecutionContext
from ...util.gdal_utils import read_band_flat


class LuminanceRegressionTopoCorrectionAlgorithm(TopoCorrectionAlgorithm):
    def _get_linear_regression_coeffs(self, ctx: QgisExecutionContext, band_idx: int) -> (float, float):
        luminance_bytes = ctx.luminance_bytes.ravel()
        band_bytes = read_band_flat(ctx.input_layer_path, band_idx=band_idx + 1)
        mask = band_bytes > ctx.pixel_ignore_threshold
        intercept, slope = np.polynomial.polynomial.polyfit(luminance_bytes[mask], band_bytes[mask], 1)
        return intercept, slope

    def _calculate_zero_noise(self):
        return 0.0001 + 0.000001 * randint(1, 99)
