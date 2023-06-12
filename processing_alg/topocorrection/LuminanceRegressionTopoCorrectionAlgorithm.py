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
