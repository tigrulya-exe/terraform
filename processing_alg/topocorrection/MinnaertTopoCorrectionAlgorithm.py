import numpy as np

from .TopoCorrectionAlgorithm import TopoCorrectionAlgorithm
from ..execution_context import QgisExecutionContext
from ...util.gdal_utils import read_band_flat
from ...util.raster_calc import RasterInfo


class MinnaertTopoCorrectionAlgorithm(TopoCorrectionAlgorithm):
    def __init__(self):
        super().__init__()
        self.x_path = None
        self.x_bytes = None

    @staticmethod
    def name():
        return "Minnaert"

    @staticmethod
    def description():
        return r'<a href="https://www.asprs.org/wp-content/uploads/pers/1980journal/sep/1980_sep_1183-1189.pdf">Minnaert correction</a>'

    def init(self, ctx: QgisExecutionContext):
        super().init(ctx)

        x_path = self._calculate_x(ctx)
        self.x_bytes = read_band_flat(x_path, band_idx=1)

    def _process_band(self, ctx: QgisExecutionContext, band_idx: int):
        k = self._calculate_k(ctx, band_idx)

        def calculate(input_band, luminance):
            quotient = np.divide(
                ctx.sza_cosine(),
                luminance,
                out=input_band.astype('float32'),
                where=np.logical_and(luminance > 0, input_band > ctx.pixel_ignore_threshold)
            )
            return input_band * np.power(quotient, k)

        return self._raster_calculate(
            ctx=ctx,
            calc_func=calculate,
            raster_infos=[
                RasterInfo("input_band", ctx.input_layer_path, band_idx + 1),
                RasterInfo("luminance", ctx.luminance_path, 1)
            ],
            out_file_postfix=band_idx
        )

    def _calculate_k(self, ctx: QgisExecutionContext, band_idx: int):
        _, slope = self._calculate_intercept_slope(ctx, band_idx)
        return slope

    def _calculate_x(self, ctx: QgisExecutionContext) -> str:
        def calculate(luminance, slope):
            return np.log(
                np.cos(slope) * luminance,
                out=np.full_like(slope, -10),
                where=(luminance > 0)
            )

        return self._raster_calculate(
            ctx=ctx,
            calc_func=calculate,
            raster_infos=[
                RasterInfo("luminance", ctx.luminance_path, 1),
                RasterInfo("slope", ctx.slope_path, 1)
            ],
            out_file_postfix="minnaert_x"
        )

    def _calculate_y(self, ctx: QgisExecutionContext, band_idx: int) -> str:
        def calculate(input_raster, slope):
            return np.log(
                np.cos(slope) * input_raster,
                out=np.full_like(slope, -10),
                where=(input_raster > 0)
            )

        return self._raster_calculate(
            ctx=ctx,
            calc_func=calculate,
            raster_infos=[
                RasterInfo("input_raster", ctx.input_layer_path, band_idx + 1),
                RasterInfo("slope", ctx.slope_path, 1)
            ],
            out_file_postfix=f"minnaert_y_{band_idx}"
        )

    def _calculate_intercept_slope(self, ctx: QgisExecutionContext, band_idx: int) -> (float, float):
        y_path = self._calculate_y(ctx, band_idx)
        y_bytes = read_band_flat(y_path)

        intercept, slope = np.polynomial.polynomial.polyfit(self.x_bytes, y_bytes, 1)
        return intercept, slope
