from math import pi, radians

import numpy as np

from .MinnaertTopoCorrectionAlgorithm import MinnaertTopoCorrectionAlgorithm
from ..execution_context import QgisExecutionContext
from ...util.raster_calc import RasterInfo


class PbcTopoCorrectionAlgorithm(MinnaertTopoCorrectionAlgorithm):
    def __init__(self):
        super().__init__()
        self.h = None
        self.h0 = None

    @staticmethod
    def name():
        return "Pixel based C-correction"

    @staticmethod
    def description():
        return r'<a href="https://www.tandfonline.com/doi/full/10.1080/01431160701881889">Pixel-based C correction</a>'

    def init(self, ctx: QgisExecutionContext):
        super().init(ctx)

        def calculate_h(slope):
            return (1 - slope) / pi

        self.h0 = (pi + 2 * radians(ctx.solar_azimuth_degrees)) / (2 * pi)
        self.h = self._raster_calculate(
            ctx=ctx,
            calc_func=calculate_h,
            raster_infos=[RasterInfo("slope", ctx.slope_path, 1)],
            out_file_postfix="pbc_h"
        )

    def _process_band(self, ctx: QgisExecutionContext, band_idx: int):
        c = self._calculate_c(ctx, band_idx)

        def calculate(input_band, luminance, h):
            denominator = luminance + c * h / self.h0
            return input_band * np.divide(
                ctx.sza_cosine() + c / self.h0,
                denominator,
                out=input_band.astype('float32'),
                where=np.logical_and(denominator > 0, input_band > ctx.pixel_ignore_threshold)
            )

        return self._raster_calculate(
            ctx=ctx,
            calc_func=calculate,
            raster_infos=[
                RasterInfo("input_band", ctx.input_layer_path, band_idx + 1),
                RasterInfo("luminance", ctx.luminance_path, 1),
                RasterInfo("h", self.h, 1),
            ],
            out_file_postfix=band_idx
        )

    def _calculate_c(self, ctx: QgisExecutionContext, band_idx: int) -> float:
        intercept, slope = self._calculate_intercept_slope(ctx, band_idx)
        return slope / intercept
