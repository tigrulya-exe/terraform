import numpy as np

from computation.my_simple_calc import RasterInfo
from topocorrection.TopoCorrectionAlgorithm import TopoCorrectionAlgorithm, TopoCorrectionContext


class CosineTTopoCorrectionAlgorithm(TopoCorrectionAlgorithm):
    @staticmethod
    def get_name():
        return "[direct_calc] COSINE-T"

    def process_band(self, ctx: TopoCorrectionContext, band_idx: int):
        def calculate(**kwargs):
            input_band = kwargs["input"]
            luminance = kwargs["luminance"]

            return ctx.sza_cosine() * np.divide(
                input_band,
                luminance,
                out=input_band.astype('float32'),
                where=np.logical_and(luminance > 0, input_band > 5)
            )

        return self.calculate(
            calc_func=calculate,
            raster_infos=[
                RasterInfo("input", ctx.input_layer.source(), band_idx + 1),
                RasterInfo("luminance", ctx.luminance_path, 1),
            ]
        )
