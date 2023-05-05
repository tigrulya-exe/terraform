import numpy as np

from .MinnaertTopoCorrectionAlgorithm import MinnaertTopoCorrectionAlgorithm
from ..execution_context import QgisExecutionContext
from ...util.raster_calc import RasterInfo


class PbmTopoCorrectionAlgorithm(MinnaertTopoCorrectionAlgorithm):
    @staticmethod
    def get_name():
        return "Pixel based Minnaert"

    def process_band(self, ctx: QgisExecutionContext, band_idx: int):
        k = self.calculate_k(ctx, band_idx)

        def calculate(input_band, luminance, slope):
            slope_cos = np.cos(slope)

            return input_band * np.divide(
                slope_cos,
                np.power(slope_cos * luminance, k),
                out=input_band.astype('float32'),
                where=np.logical_and(luminance > 0, input_band > 5)
            )

        return self.raster_calculate(
            ctx=ctx,
            calc_func=calculate,
            raster_infos=[
                RasterInfo("input_band", ctx.input_layer_path, band_idx + 1),
                RasterInfo("luminance", ctx.luminance_path, 1),
                RasterInfo("slope", ctx.slope_path, 1)
            ],
            out_file_postfix=band_idx
        )
