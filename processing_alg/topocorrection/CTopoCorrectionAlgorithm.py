import numpy as np
from numba import njit

from .SimpleRegressionTopoCorrectionAlgorithm import SimpleRegressionTopoCorrectionAlgorithm
from ..execution_context import QgisExecutionContext
from ...computation.raster_calc import RasterInfo


@njit
def calculate(input, luminance, sza_cosine, c):
    denominator = luminance + c
    return input * np.divide(
        sza_cosine + c,
        denominator,
        input.astype('float32'),
        # where=np.logical_and(denominator > 0, input > 5)
    )


class CTopoCorrectionAlgorithm(SimpleRegressionTopoCorrectionAlgorithm):
    @staticmethod
    def get_name():
        return "C-correction"

    def process_band(self, ctx: QgisExecutionContext, band_idx: int):
        c = self.calculate_c(ctx, band_idx)

        return self.raster_calculate(
            ctx=ctx,
            calc_func=calculate,
            raster_infos=[
                RasterInfo("input", ctx.input_layer.source(), band_idx + 1),
                RasterInfo("luminance", ctx.luminance_path, 1),
            ],
            out_file_postfix=band_idx,
            sza_cosine=ctx.sza_cosine(),
            c=c
        )

    def calculate_c(self, ctx: QgisExecutionContext, band_idx: int) -> float:
        intercept, slope = self.get_linear_regression_coeffs(ctx, band_idx)
        return intercept / slope
