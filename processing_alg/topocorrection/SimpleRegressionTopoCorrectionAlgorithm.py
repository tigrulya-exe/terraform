import time
from random import randint

import numpy as np

from .TopoCorrectionAlgorithm import TopoCorrectionAlgorithm
from ..execution_context import QgisExecutionContext
from ...util.gdal_utils import read_band_flat


class SimpleRegressionTopoCorrectionAlgorithm(TopoCorrectionAlgorithm):
    def get_linear_regression_coeffs(self, ctx: QgisExecutionContext, band_idx: int) -> (float, float):
        luminance_bytes = ctx.luminance_bytes.ravel()

        read_start = time.process_time_ns()
        band_bytes = read_band_flat(ctx.input_layer_path, band_idx=band_idx+1)
        read_end = time.process_time_ns()

        ctx.log(f'read: {(read_end - read_start) / 1000000} sec')

        fit_start = time.process_time_ns()
        intercept, slope = np.polynomial.polynomial.polyfit(luminance_bytes, band_bytes, 1)
        fit_end = time.process_time_ns()

        ctx.log(f'{(intercept, slope)}: {(fit_end - fit_start) / 1000000} sec')
        return intercept, slope

    def _calculate_zero_noise(self):
        return 0.0001 + 0.000001 * randint(1, 99)
