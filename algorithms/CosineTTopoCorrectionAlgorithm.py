import os
import random
import tempfile

import numpy as np

from algorithms.TopoCorrectionAlgorithm import TopoCorrectionAlgorithm, TopoCorrectionContext
from computation.my_simple_calc import RasterInfo


class CosineTTopoCorrectionAlgorithm(TopoCorrectionAlgorithm):
    @staticmethod
    def get_name():
        return "COSINE-T"

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

        out_path = os.path.join(
            tempfile.gettempdir(),
            f'cosine_t_{band_idx}_{random.randint(1, 100)}'
        )

        self.calc.calculate(
            func=calculate,
            output_path=out_path,
            raster_infos=[
                RasterInfo("input", ctx.input_layer.source(), band_idx + 1),
                RasterInfo("luminance", ctx.luminance_path, 1),
            ]
        )
        return out_path
