from math import pi, radians

import numpy as np

from .MinnaertTopoCorrectionAlgorithm import MinnaertTopoCorrectionAlgorithm
from ..execution_context import QgisExecutionContext
from ...computation.gdal_utils import read_band_flat
from ...computation.raster_calc import RasterInfo


class PbcTopoCorrectionAlgorithm(MinnaertTopoCorrectionAlgorithm):
    @staticmethod
    def get_name():
        return "Pixel based C-correction"

    def init(self, ctx: QgisExecutionContext):
        super().init(ctx)

        def calculate_h(**kwargs):
            return (1 - kwargs["slope"]) / pi

        self.h0 = (pi + 2 * radians(ctx.solar_azimuth_degrees)) / (2 * pi)
        self.h = self.raster_calculate(
            ctx=ctx,
            calc_func=calculate_h,
            raster_infos=[RasterInfo("slope", ctx.slope_path, 1)],
            out_file_postfix="pbc_h"
        )

    def process_band(self, ctx: QgisExecutionContext, band_idx: int):
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
            ctx=ctx,
            calc_func=calculate,
            raster_infos=[
                RasterInfo("input", ctx.input_layer.source(), band_idx + 1),
                RasterInfo("luminance", ctx.luminance_path, 1),
                RasterInfo("h", self.h, 1),
            ],
            out_file_postfix=band_idx
        )

    def calculate_c(self, ctx: QgisExecutionContext, band_idx: int) -> float:
        y_path = self.calculate_y(ctx, band_idx)

        x_bytes = self.x_bytes if ctx.keep_in_memory else read_band_flat(self.x_path)
        y_bytes = read_band_flat(y_path)

        intercept, slope = np.polynomial.polynomial.polyfit(x_bytes, y_bytes, 1)
        ctx.log(f'{(intercept, slope)}')
        return slope / intercept
