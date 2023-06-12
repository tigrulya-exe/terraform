import numpy as np

from .TopoCorrectionAlgorithm import TopoCorrectionAlgorithm
from ..execution_context import QgisExecutionContext
from ...util.raster_calc import RasterInfo


class CosineCTopoCorrectionAlgorithm(TopoCorrectionAlgorithm):
    def __init__(self):
        super().__init__()
        self.luminance_mean = None

    @staticmethod
    def name():
        return "COSINE-C"

    @staticmethod
    def description():
        return '<a href="https://www.asprs.org/wp-content/uploads/pers/1989journal/sep/1989_sep_1303-1309.pdf">Cosine-C</a>'

    def init(self, ctx: QgisExecutionContext):
        super().init(ctx)
        self.luminance_mean = np.mean(ctx.luminance_bytes)

    def _process_band(self, ctx: QgisExecutionContext, band_idx: int):
        def calculate(input_band, luminance):
            return input_band * (1 + np.divide(
                self.luminance_mean - luminance,
                self.luminance_mean,
                out=input_band.astype('float32'),
                where=input_band > ctx.pixel_ignore_threshold
            ))

        return self._raster_calculate(
            ctx=ctx,
            calc_func=calculate,
            raster_infos=[
                RasterInfo("input_band", ctx.input_layer_path, band_idx + 1),
                RasterInfo("luminance", ctx.luminance_path, 1),
            ],
            out_file_postfix=band_idx
        )
