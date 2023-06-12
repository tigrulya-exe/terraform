import numpy as np

from .MinnaertTopoCorrectionAlgorithm import MinnaertTopoCorrectionAlgorithm
from ..execution_context import QgisExecutionContext
from ...util.raster_calc import RasterInfo


class MinnaertScsTopoCorrectionAlgorithm(MinnaertTopoCorrectionAlgorithm):
    @staticmethod
    def name():
        return "Minnaert-SCS"

    @staticmethod
    def description():
        return r'<a href="https://ui.adsabs.harvard.edu/abs/2002PhDT........92R/abstract">Minnaert-SCS</a>'

    def _process_band(self, ctx: QgisExecutionContext, band_idx: int):
        k = self._calculate_k(ctx, band_idx)

        def calculate(input_band, luminance, slope):
            quotient = np.divide(
                ctx.sza_cosine(),
                luminance,
                out=input_band.astype('float32'),
                where=np.logical_and(luminance > 0, input_band > ctx.pixel_ignore_threshold)
            )
            return input_band * np.cos(slope) * np.power(quotient, k)

        return self._raster_calculate(
            ctx=ctx,
            calc_func=calculate,
            raster_infos=[
                RasterInfo("input_band", ctx.input_layer_path, band_idx + 1),
                RasterInfo("luminance", ctx.luminance_path, 1),
                RasterInfo("slope", ctx.slope_path, 1)
            ],
            out_file_postfix=band_idx
        )
