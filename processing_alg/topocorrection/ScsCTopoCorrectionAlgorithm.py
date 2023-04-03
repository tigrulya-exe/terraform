import numpy as np

from .CTopoCorrectionAlgorithm import CTopoCorrectionAlgorithm
from ..execution_context import QgisExecutionContext
from ...computation.raster_calc import RasterInfo


class ScsCTopoCorrectionAlgorithm(CTopoCorrectionAlgorithm):
    @staticmethod
    def get_name():
        return "SCS+C"

    def process_band(self, ctx: QgisExecutionContext, band_idx: int):
        c = self.calculate_c(ctx, band_idx)

        def calculate(**kwargs):
            input_band = kwargs["input"]
            luminance = kwargs["luminance"]
            slope = kwargs["slope"]

            denominator = luminance + c
            return input_band * np.divide(
                np.cos(slope) * ctx.sza_cosine() + c,
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
                RasterInfo("slope", ctx.slope_path, 1)
            ],
            out_file_postfix=band_idx
        )
