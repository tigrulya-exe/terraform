import numpy as np

from .TopoCorrectionAlgorithm import TopoCorrectionAlgorithm
from ..execution_context import QgisExecutionContext
from ...computation.raster_calc import RasterInfo


class CosineTTopoCorrectionAlgorithm(TopoCorrectionAlgorithm):
    @staticmethod
    def get_name():
        return "COSINE-T"

    def process_band(self, ctx: QgisExecutionContext, band_idx: int):
        def calculate(**kwargs):
            input_band = kwargs["input"]
            luminance = kwargs["luminance"]

            return input_band * np.divide(
                ctx.sza_cosine(),
                luminance,
                out=input_band.astype('float32'),
                where=np.logical_and(luminance > 0, input_band > 5)
            )

        return self.raster_calculate(
            calc_func=calculate,
            raster_infos=[
                RasterInfo("input", ctx.input_layer.source(), band_idx + 1),
                RasterInfo("luminance", ctx.luminance, 1),
            ],
            out_file_postfix=band_idx
        )
