import numpy as np

from .TopoCorrectionAlgorithm import TopoCorrectionAlgorithm
from ..execution_context import QgisExecutionContext
from ...computation import gdal_utils
from ...computation.raster_calc import RasterInfo


class CosineCTopoCorrectionAlgorithm(TopoCorrectionAlgorithm):
    @staticmethod
    def get_name():
        return "COSINE-C"

    def init(self, ctx: QgisExecutionContext):
        super().init(ctx)
        # todo add validation
        self.luminance_mean = gdal_utils.compute_band_means(ctx.luminance_path)[0]

    def process_band(self, ctx: QgisExecutionContext, band_idx: int):
        def calculate(**kwargs):
            input_band = kwargs["input"]
            luminance = kwargs["luminance"]

            return input_band * (1 + np.divide(
                self.luminance_mean - luminance,
                self.luminance_mean,
                out=input_band.astype('float32'),
                where=input_band > 5
            ))

        return self.raster_calculate(
            ctx=ctx,
            calc_func=calculate,
            raster_infos=[
                RasterInfo("input", ctx.input_layer_path, band_idx + 1),
                RasterInfo("luminance", ctx.luminance_path, 1),
            ],
            out_file_postfix=band_idx
        )
