from math import pi, radians

import numpy as np

from .MinnaertTopoCorrectionAlgorithm import MinnaertTopoCorrectionAlgorithm
from .TopoCorrectionAlgorithm import TopoCorrectionContext
from ...computation import gdal_utils
from ...computation.raster_calc import RasterInfo


class PbcTopoCorrectionAlgorithm(MinnaertTopoCorrectionAlgorithm):
    @staticmethod
    def get_name():
        return "Pixel based C-correction"

    def init(self, ctx: TopoCorrectionContext):
        super().init(ctx)

        def calculate_h(**kwargs):
            return (1 - kwargs["slope"]) / pi

        self.h0 = (pi + 2 * radians(ctx.solar_zenith_angle)) / (2 * pi)
        self.h = self.raster_calculate(
            calc_func=calculate_h,
            raster_infos=[RasterInfo("slope", ctx.slope_rad_path, 1)]
        )

    def process_band(self, ctx: TopoCorrectionContext, band_idx: int):
        c = self.calculate_c(ctx, band_idx)

        def calculate(**kwargs):
            input_band = kwargs["input"]
            luminance = kwargs["luminance"]
            h = kwargs["h"]

            denominator = luminance + c * h
            return input_band * np.divide(
                ctx.sza_cosine() + c * self.h0,
                denominator,
                out=input_band.astype('float32'),
                where=np.logical_and(denominator > 0, input_band > 5)
            )

        return self.raster_calculate(
            calc_func=calculate,
            raster_infos=[
                RasterInfo("input", ctx.input_layer.source(), band_idx + 1),
                RasterInfo("luminance", ctx.luminance_path, 1),
                RasterInfo("h", self.h, 1),
            ]
        )

    def calculate_c(self, ctx: TopoCorrectionContext, band_idx: int) -> float:
        y_path = self.calculate_y(ctx, band_idx)
        intercept, slope = gdal_utils.raster_linear_regression(self.x_path, y_path)
        ctx.qgis_feedback.pushInfo(f'{(intercept, slope)}')
        return slope / intercept
