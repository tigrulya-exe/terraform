import numpy as np

from .TopoCorrectionAlgorithm import TopoCorrectionAlgorithm
from ..execution_context import QgisExecutionContext
from ...util.raster_calc import RasterInfo


class CosineTTopoCorrectionAlgorithm(TopoCorrectionAlgorithm):
    @staticmethod
    def name():
        return "COSINE-T"

    @staticmethod
    def description():
        return r'<a href="https://www.tandfonline.com/doi/abs/10.1080/07038992.1982.10855028">Cosine-T</a>'

    def _process_band(self, ctx: QgisExecutionContext, band_idx: int):
        def calculate(input_band, luminance):
            return input_band * np.divide(
                ctx.sza_cosine(),
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
            ],
            out_file_postfix=band_idx
        )
