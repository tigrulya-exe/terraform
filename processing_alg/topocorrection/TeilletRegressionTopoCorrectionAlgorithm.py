from .SimpleRegressionTopoCorrectionAlgorithm import SimpleRegressionTopoCorrectionAlgorithm
from ..execution_context import QgisExecutionContext
from ...util import gdal_utils
from ...util.raster_calc import RasterInfo


class TeilletRegressionTopoCorrectionAlgorithm(SimpleRegressionTopoCorrectionAlgorithm):
    @staticmethod
    def get_name():
        return "Teillet regression"

    def init(self, ctx: QgisExecutionContext):
        super().init(ctx)
        self.raster_means = gdal_utils.compute_band_means(ctx.input_layer_path)

    def process_band(self, ctx: QgisExecutionContext, band_idx: int):
        intercept, slope = self.get_linear_regression_coeffs(ctx, band_idx)

        def calculate(input_band, luminance):
            result = input_band - slope * luminance - intercept + self.raster_means[band_idx]
            result[result <= 0] = self._calculate_zero_noise()
            return result

        return self.raster_calculate(
            ctx=ctx,
            calc_func=calculate,
            raster_infos=[
                RasterInfo("input_band", ctx.input_layer_path, band_idx + 1),
                RasterInfo("luminance", ctx.luminance_path, 1),
            ],
            out_file_postfix=band_idx
        )
