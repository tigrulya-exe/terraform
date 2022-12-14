import numpy as np

from computation.my_simple_calc import RasterInfo
from topocorrection.TopoCorrectionAlgorithm import TopoCorrectionAlgorithm, TopoCorrectionContext
from computation import gdal_utils


class MinnaertTopoCorrectionAlgorithm(TopoCorrectionAlgorithm):
    @staticmethod
    def get_name():
        return "Minnaert"

    def init(self, ctx: TopoCorrectionContext):
        self.x_path = self.calculate_x(ctx)

    def process_band(self, ctx: TopoCorrectionContext, band_idx: int):
        k = self.calculate_k(ctx, band_idx)

        def calculate(**kwargs):
            input_band = kwargs["input"]
            luminance = kwargs["luminance"]

            quotient = np.divide(
                ctx.sza_cosine(),
                luminance,
                out=input_band.astype('float32'),
                where=np.logical_and(luminance > 0, input_band > 5)
            )
            return input_band * np.power(quotient, k)

        return self.raster_calculate(
            calc_func=calculate,
            raster_infos=[
                RasterInfo("input", ctx.input_layer.source(), band_idx + 1),
                RasterInfo("luminance", ctx.luminance_path, 1)
            ]
        )

    def calculate_k(self, ctx: TopoCorrectionContext, band_idx: int):
        y_path = self.calculate_y(ctx, band_idx)
        intercept, slope = gdal_utils.raster_linear_regression(self.x_path, y_path)
        ctx.qgis_feedback.pushInfo(f'{(intercept, slope)}')
        return slope

    def calculate_x(self, ctx: TopoCorrectionContext) -> str:
        def calculate(**kwargs):
            luminance = kwargs["luminance"]
            slope = kwargs["slope"]

            return np.log(
                np.cos(slope) * luminance,
                out=np.full_like(slope, -10),
                where=(luminance > 0)
            )

        return self.raster_calculate(
            calc_func=calculate,
            raster_infos=[
                RasterInfo("luminance", ctx.luminance_path, 1),
                RasterInfo("slope", ctx.slope_rad_path, 1)
            ]
        )

    def calculate_y(self, ctx: TopoCorrectionContext, band_idx: int) -> str:
        def calculate(**kwargs):
            input_raster = kwargs["input"]
            slope = kwargs["slope"]

            return np.log(
                np.cos(slope) * input_raster,
                out=np.full_like(slope, -10),
                where=(input_raster > 0)
            )

        return self.raster_calculate(
            calc_func=calculate,
            raster_infos=[
                RasterInfo("input", ctx.input_layer.source(), band_idx + 1),
                RasterInfo("slope", ctx.slope_rad_path, 1)
            ]
        )
