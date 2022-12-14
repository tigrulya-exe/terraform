import numpy as np

from computation.my_simple_calc import RasterInfo
from topocorrection.TopoCorrectionAlgorithm import TopoCorrectionAlgorithm, TopoCorrectionContext


class ScsTopoCorrectionAlgorithm(TopoCorrectionAlgorithm):
    @staticmethod
    def get_name():
        return "[direct_calc] SCS"

    def process_band(self, ctx: TopoCorrectionContext, band_idx: int):
        def calculate(**kwargs):
            input_band = kwargs["input"]
            luminance = kwargs["luminance"]
            slope = kwargs["slope"]

            return np.divide(
                slope * ctx.sza_cosine(),
                luminance,
                out=input_band.astype('float32'),
                where=luminance > 0
            )

        return self.calculate(
            calc_func=calculate,
            raster_infos=[
                RasterInfo("input", ctx.input_layer.source(), band_idx + 1),
                RasterInfo("luminance", ctx.luminance_path, 1),
                RasterInfo("slope", ctx.slope_rad_path, 1)
            ]
        )
