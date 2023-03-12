import numpy as np

from .SimpleRegressionTopoCorrectionAlgorithm import SimpleRegressionTopoCorrectionAlgorithm
from .TopoCorrectionAlgorithm import TopoCorrectionContext
from ...computation import gdal_utils
from ...computation.raster_calc import RasterInfo


class VecaTopoCorrectionAlgorithm(SimpleRegressionTopoCorrectionAlgorithm):
    @staticmethod
    def get_name():
        return "VECA"

    def init(self, ctx: TopoCorrectionContext):
        self.raster_means = gdal_utils.compute_band_means(ctx.input_layer.source())

    def process_band(self, ctx: TopoCorrectionContext, band_idx: int):
        intercept, slope = self.get_linear_regression_coeffs(ctx, band_idx)

        def calculate(**kwargs):
            input_band = kwargs["input"]
            luminance = kwargs["luminance"]

            denominator = slope * luminance + intercept
            return input_band * np.divide(
                self.raster_means[band_idx],
                denominator,
                out=input_band.astype('float32'),
                where=np.logical_and(denominator > 0, input_band > 5)
            )

        return self.raster_calculate(
            calc_func=calculate,
            raster_infos=[
                RasterInfo("input", ctx.input_layer.source(), band_idx + 1),
                RasterInfo("luminance", ctx.luminance_path, 1),
            ]
        )
