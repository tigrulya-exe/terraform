import processing

from algorithms.TopoCorrectionAlgorithm import TopoCorrectionAlgorithm, TopoCorrectionContext
from computation import gdal_utils
from computation.qgis_utils import add_layer_to_project


class MinnaertTopoCorrectionAlgorithm(TopoCorrectionAlgorithm):
    def init(self, ctx: TopoCorrectionContext):
        x = processing.run(
            'gdal:rastercalculator',
            {
                'INPUT_A': ctx.slope_path,
                'BAND_A': 1,
                'INPUT_B': ctx.luminance_path,
                'BAND_B': 1,
                'FORMULA': f'log(cos(deg2rad(A)) * fmax(0.00001, B))',
                'OUTPUT': 'TEMPORARY_OUTPUT',
            },
            feedback=ctx.qgis_feedback,
            context=ctx.qgis_context
        )
        self.x_path = x['OUTPUT']
        # add_layer_to_project(ctx.qgis_context, self.x_path, "x")

    @staticmethod
    def get_name():
        return "Minnaert"

    def process_band(self, ctx: TopoCorrectionContext, band_idx: int):
        y = processing.run(
            'gdal:rastercalculator',
            {
                # create layer from luminance_layer
                'INPUT_A': ctx.slope_path,
                'BAND_A': 1,
                'INPUT_B': ctx.input_layer,
                'BAND_B': band_idx + 1,
                'FORMULA': f'log(cos(deg2rad(A)) * fmax(0.00001, B))',
                'OUTPUT': 'TEMPORARY_OUTPUT',
            },
            feedback=ctx.qgis_feedback,
            context=ctx.qgis_context
        )
        y_path = y['OUTPUT']
        # add_layer_to_project(ctx.qgis_context, y_path, f"y_{band_idx}")

        weights = gdal_utils.raster_linear_regression_full(self.x_path, y_path)
        ctx.qgis_feedback.pushInfo(f'{weights}')
        k = weights[0][1]

        result = processing.run(
            'gdal:rastercalculator',
            {
                # create layer from luminance_layer
                'INPUT_A': ctx.luminance_path,
                'BAND_A': 1,
                'INPUT_B': ctx.input_layer,
                'BAND_B': band_idx + 1,
                'FORMULA': f'B * power({ctx.sza_cosine()}/A, {k})',
                'OUTPUT': 'TEMPORARY_OUTPUT',
            },
            feedback=ctx.qgis_feedback,
            context=ctx.qgis_context
        )
        return result['OUTPUT']
