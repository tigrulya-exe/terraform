from random import randint

from .TopoCorrectionAlgorithm import TopoCorrectionAlgorithm, TopoCorrectionContext
from ...computation.gdal_utils import raster_linear_regression


class SimpleRegressionTopoCorrectionAlgorithm(TopoCorrectionAlgorithm):
    def get_linear_regression_coeffs(self, ctx: TopoCorrectionContext, band_idx: int) -> (float, float):
        intercept, slope = raster_linear_regression(ctx.luminance_path, ctx.input_layer.source(), y_band=band_idx + 1)
        ctx.qgis_feedback.pushInfo(f'{(intercept, slope)}')
        return intercept, slope

    def _calculate_zero_noise(self):
        return 0.0001 + 0.000001 * randint(1, 99)
