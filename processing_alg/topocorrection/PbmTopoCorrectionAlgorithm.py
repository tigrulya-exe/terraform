import numpy as np

from .MinnaertTopoCorrectionAlgorithm import MinnaertTopoCorrectionAlgorithm
from ..execution_context import QgisExecutionContext
from ...util.raster_calc import RasterInfo


class PbmTopoCorrectionAlgorithm(MinnaertTopoCorrectionAlgorithm):
    @staticmethod
    def name():
        return "Pixel based Minnaert"

    @staticmethod
    def description():
        return '<a href="https://www.researchgate.net/publication/235244169_Pixel-based_Minnaert_Correction_Method_for_Reducing_Topographic_Effects_on_a_Landsat_7_ETM_Image">Pixel-based Minnaert</a>'

    def _process_band(self, ctx: QgisExecutionContext, band_idx: int):
        k = self._calculate_k(ctx, band_idx)

        def calculate(input_band, luminance, slope):
            slope_cos = np.cos(slope)

            return input_band * np.divide(
                slope_cos,
                np.power(slope_cos * luminance, k),
                out=input_band.astype('float32'),
                where=np.logical_and(luminance > 0, input_band > ctx.pixel_ignore_threshold)
            )

        return self._raster_calculate(
            ctx=ctx,
            calc_func=calculate,
            raster_infos=[
                RasterInfo("input_band", ctx.input_layer_path, band_idx + 1),
                RasterInfo("luminance", ctx.luminance_path, 1),
                RasterInfo("slope", ctx.slope_path, 1)
            ],
            out_file_postfix=band_idx
        )
