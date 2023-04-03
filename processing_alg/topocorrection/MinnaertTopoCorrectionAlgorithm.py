import numpy as np

from ...computation.gdal_utils import read_band_flat
from .TopoCorrectionAlgorithm import TopoCorrectionAlgorithm
from ..execution_context import QgisExecutionContext
from ...computation.raster_calc import RasterInfo


class MinnaertTopoCorrectionAlgorithm(TopoCorrectionAlgorithm):
    @staticmethod
    def get_name():
        return "Minnaert"

    def init(self, ctx: QgisExecutionContext):
        super().init(ctx)

        x_path = self.calculate_x(ctx)
        if ctx.keep_in_memory:
            self.x_bytes = read_band_flat(x_path, band_idx=1)
        else:
            self.x_path = x_path

    def process_band(self, ctx: QgisExecutionContext, band_idx: int):
        k = self.calculate_k(ctx, band_idx)

        def calculate(**kwargs):
            input_band = kwargs["input"]
            luminance = kwargs["luminance"]

            quotient = np.divide(
                ctx.sza_cosine(),
                luminance,
                out=input_band.astype('float32'),
                where=np.logical_and(luminance > 0, input_band > 5)
            )
            return input_band * np.power(quotient, k)

        return self.raster_calculate(
            ctx=ctx,
            calc_func=calculate,
            raster_infos=[
                RasterInfo("input", ctx.input_layer.source(), band_idx + 1),
                RasterInfo("luminance", ctx.luminance_path, 1)
            ],
            out_file_postfix=band_idx
        )

    def calculate_k(self, ctx: QgisExecutionContext, band_idx: int):
        y_path = self.calculate_y(ctx, band_idx)

        x_bytes = self.x_bytes if ctx.keep_in_memory else read_band_flat(self.x_path)
        y_bytes = read_band_flat(y_path)

        intercept, slope = np.polynomial.polynomial.polyfit(x_bytes, y_bytes, 1)
        ctx.log(f'{(intercept, slope)}')
        return slope

    def calculate_x(self, ctx: QgisExecutionContext) -> str:
        def calculate(**kwargs):
            luminance = kwargs["luminance"]
            slope = kwargs["slope"]

            return np.log(
                np.cos(slope) * luminance,
                out=np.full_like(slope, -10),
                where=(luminance > 0)
            )

        return self.raster_calculate(
            ctx=ctx,
            calc_func=calculate,
            raster_infos=[
                RasterInfo("luminance", ctx.luminance_path, 1),
                RasterInfo("slope", ctx.slope_path, 1)
            ],
            out_file_postfix="minnaert_x"
        )

    def calculate_y(self, ctx: QgisExecutionContext, band_idx: int) -> str:
        def calculate(**kwargs):
            input_raster = kwargs["input"]
            slope = kwargs["slope"]

            return np.log(
                np.cos(slope) * input_raster,
                out=np.full_like(slope, -10),
                where=(input_raster > 0)
            )

        return self.raster_calculate(
            ctx=ctx,
            calc_func=calculate,
            raster_infos=[
                RasterInfo("input", ctx.input_layer.source(), band_idx + 1),
                RasterInfo("slope", ctx.slope_path, 1)
            ],
            out_file_postfix=f"minnaert_y_{band_idx}"
        )
