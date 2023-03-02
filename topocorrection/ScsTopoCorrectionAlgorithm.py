import numpy as np

from computation.raster_calc import RasterInfo
from topocorrection.TopoCorrectionAlgorithm import TopoCorrectionAlgorithm, TopoCorrectionContext


class ScsTopoCorrectionAlgorithm(TopoCorrectionAlgorithm):
    @staticmethod
    def get_name():
        return "SCS"

    def process_band(self, ctx: TopoCorrectionContext, band_idx: int):
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
            calc_func=calculate,
            raster_infos=[
                RasterInfo("input", ctx.input_layer.source(), band_idx + 1),
                RasterInfo("luminance", ctx.luminance_path, 1),
                RasterInfo("slope", ctx.slope_rad_path, 1)
            ]
        )
