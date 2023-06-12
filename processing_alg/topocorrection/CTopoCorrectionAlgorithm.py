import numpy as np

from .LuminanceRegressionTopoCorrectionAlgorithm import LuminanceRegressionTopoCorrectionAlgorithm
from ..execution_context import QgisExecutionContext
from ...util.raster_calc import RasterInfo


def calculate(input, luminance, sza_cosine, c):
    denominator = luminance + c
    return input * np.divide(
        sza_cosine + c,
        denominator,
        input.astype('float32')
    )


class CTopoCorrectionAlgorithm(LuminanceRegressionTopoCorrectionAlgorithm):
    @staticmethod
    def name():
        return "C-correction"

    @staticmethod
    def description():
        return r'<a href="https://www.tandfonline.com/doi/abs/10.1080/07038992.1982.10855028">C-correction</a>'

    def _process_band(self, ctx: QgisExecutionContext, band_idx: int):
        c = self._calculate_c(ctx, band_idx)

        return self._raster_calculate(
            ctx=ctx,
            calc_func=calculate,
            raster_infos=[
                RasterInfo("input", ctx.input_layer_path, band_idx + 1),
                RasterInfo("luminance", ctx.luminance_path, 1),
            ],
            out_file_postfix=band_idx,
            sza_cosine=ctx.sza_cosine(),
            c=c
        )

    def _calculate_c(self, ctx: QgisExecutionContext, band_idx: int) -> float:
        intercept, slope = self._get_linear_regression_coeffs(ctx, band_idx)
        return intercept / slope
