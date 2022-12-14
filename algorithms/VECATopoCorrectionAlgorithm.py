import processing

from algorithms.SimpleRegressionTopoCorrectionAlgorithm import SimpleRegressionTopoCorrectionAlgorithm
from algorithms.TopoCorrectionAlgorithm import TopoCorrectionContext


class VECATopoCorrectionAlgorithm(SimpleRegressionTopoCorrectionAlgorithm):
    @staticmethod
    def get_name():
        return "[old] VECA"

    def process_band(self, ctx: TopoCorrectionContext, band_idx: int):
        intercept, slope = self.get_linear_regression_coeffs(ctx, band_idx)

        result = processing.run(
            'gdal:rastercalculator',
            {
                'INPUT_A': ctx.luminance_path,
                'BAND_A': 1,
                'INPUT_B': ctx.input_layer,
                'BAND_B': band_idx + 1,
                'FORMULA': f'B * mean(B)/({slope} * A + {intercept})',
                'OUTPUT': 'TEMPORARY_OUTPUT',
            },
            feedback=ctx.qgis_feedback,
            context=ctx.qgis_context
        )
        return result['OUTPUT']
