from random import randint

from .TopoCorrectionAlgorithm import TopoCorrectionAlgorithm
from ..execution_context import QgisExecutionContext
from ...computation.gdal_utils import raster_linear_regression


class SimpleRegressionTopoCorrectionAlgorithm(TopoCorrectionAlgorithm):
    def get_linear_regression_coeffs(self, ctx: QgisExecutionContext, band_idx: int) -> (float, float):
        intercept, slope = raster_linear_regression(ctx.luminance, ctx.input_layer.source(), y_band=band_idx + 1)
        ctx.qgis_feedback.pushInfo(f'{(intercept, slope)}')
        return intercept, slope

    def _calculate_zero_noise(self):
        return 0.0001 + 0.000001 * randint(1, 99)
