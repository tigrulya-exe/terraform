import numpy as np

from .TopoCorrectionAlgorithm import TopoCorrectionAlgorithm
from ..execution_context import QgisExecutionContext
from ...util import gdal_utils
from ...util.raster_calc import RasterInfo


class CosineCTopoCorrectionAlgorithm(TopoCorrectionAlgorithm):
    @staticmethod
    def get_name():
        return "COSINE-C"

    def init(self, ctx: QgisExecutionContext):
        super().init(ctx)
        self.luminance_mean = np.mean(ctx.luminance_bytes)

    def process_band(self, ctx: QgisExecutionContext, band_idx: int):
        def calculate(input_band, luminance):
            return input_band * (1 + np.divide(
                self.luminance_mean - luminance,
                self.luminance_mean,
                out=input_band.astype('float32'),
                where=input_band > ctx.pixel_ignore_threshold
            ))

        return self.raster_calculate(
            ctx=ctx,
            calc_func=calculate,
            raster_infos=[
                RasterInfo("input_band", ctx.input_layer_path, band_idx + 1),
                RasterInfo("luminance", ctx.luminance_path, 1),
            ],
            out_file_postfix=band_idx
        )
