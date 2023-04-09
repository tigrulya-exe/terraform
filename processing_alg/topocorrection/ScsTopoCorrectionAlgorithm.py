import numpy as np

from .TopoCorrectionAlgorithm import TopoCorrectionAlgorithm
from ..execution_context import QgisExecutionContext
from ...computation.raster_calc import RasterInfo


class ScsTopoCorrectionAlgorithm(TopoCorrectionAlgorithm):
    @staticmethod
    def get_name():
        return "SCS"

    def process_band(self, ctx: QgisExecutionContext, band_idx: int):
        def calculate(**kwargs):
            input_band = kwargs["input"]
            luminance = kwargs["luminance"]
            slope = kwargs["slope"]

            return input_band * np.divide(
                np.cos(slope) * ctx.sza_cosine(),
                luminance,
                out=input_band.astype('float32'),
                where=np.logical_and(luminance > 0, input_band > 5)
            )

        return self.raster_calculate(
            ctx=ctx,
            calc_func=calculate,
            raster_infos=[
                RasterInfo("input", ctx.input_layer_path, band_idx + 1),
                RasterInfo("luminance", ctx.luminance_path, 1),
                RasterInfo("slope", ctx.slope_path, 1)
            ],
            out_file_postfix=band_idx
        )
