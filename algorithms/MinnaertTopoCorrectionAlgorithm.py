import processing

from algorithms.TopoCorrectionAlgorithm import TopoCorrectionAlgorithm, TopoCorrectionContext
from computation import gdal_utils

class MinnaertTopoCorrectionAlgorithm(TopoCorrectionAlgorithm):
    def init(self, ctx: TopoCorrectionContext):
        x = processing.run(
            'gdal:rastercalculator',
            {
                'INPUT_A': ctx.slope_rad_path,
                'BAND_A': 1,
                'INPUT_B': ctx.luminance_path,
                'BAND_B': 1,
                'FORMULA': f'log(cos(A) * B, full_like(A, -10), where=(B!=0))',
                'OUTPUT': 'TEMPORARY_OUTPUT',
            },
            feedback=ctx.qgis_feedback,
            context=ctx.qgis_context
        )
        self.x_path = x['OUTPUT']
        # add_layer_to_project(ctx.qgis_context, self.x_path, "x")

    @staticmethod
    def get_name():
        return "[old] Minnaert"

    def process_band(self, ctx: TopoCorrectionContext, band_idx: int):
        k = self.calculate_k(ctx, band_idx)

        result = processing.run(
            'gdal:rastercalculator',
            {
                # create layer from luminance_layer
                'INPUT_A': ctx.luminance_path,
                'BAND_A': 1,
                'INPUT_B': ctx.input_layer,
                'BAND_B': band_idx + 1,
                'FORMULA': f"B * power({self.safe_divide(ctx.sza_cosine(), 'A')}, {k})",
                'OUTPUT': 'TEMPORARY_OUTPUT',
            },
            feedback=ctx.qgis_feedback,
            context=ctx.qgis_context
        )
        return result['OUTPUT']

    def calculate_k(self, ctx: TopoCorrectionContext, band_idx: int):
        y_path = self.calculate_y(ctx, band_idx)
        # add_layer_to_project(ctx.qgis_context, y_path, f"y_{band_idx}")
        intercept, slope = gdal_utils.raster_linear_regression(self.x_path, y_path)
        ctx.qgis_feedback.pushInfo(f'{(intercept, slope)}')
        return slope

    def calculate_y(self, ctx: TopoCorrectionContext, band_idx: int) -> str:
        y = processing.run(
            'gdal:rastercalculator',
            {
                # create layer from luminance_layer
                'INPUT_A': ctx.slope_rad_path,
                'BAND_A': 1,
                'INPUT_B': ctx.input_layer,
                'BAND_B': band_idx + 1,
                'FORMULA': f'log(cos(A) * B, zeros_like(A), where=(B!=0))',
                'OUTPUT': 'TEMPORARY_OUTPUT',
            },
            feedback=ctx.qgis_feedback,
            context=ctx.qgis_context
        )
        return y['OUTPUT']
