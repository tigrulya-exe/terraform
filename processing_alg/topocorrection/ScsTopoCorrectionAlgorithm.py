import numpy as np

from .TopoCorrectionAlgorithm import TopoCorrectionAlgorithm
from ..execution_context import QgisExecutionContext
from ...util.raster_calc import RasterInfo


class ScsTopoCorrectionAlgorithm(TopoCorrectionAlgorithm):
    @staticmethod
    def name():
        return "SCS"

    @staticmethod
    def description():
        return r'<a href="http://dx.doi.org/10.1016/S0034-4257(97)00177-6">SCS</a>'

    def _process_band(self, ctx: QgisExecutionContext, band_idx: int):
        def calculate(input_band, luminance, slope):
            return input_band * np.divide(
                np.cos(slope) * ctx.sza_cosine(),
                luminance,
                out=input_band.astype('float32'),
                where=np.logical_and(luminance > 0, input_band > ctx.pixel_ignore_threshold)
            )

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
