import processing

from algorithms.SimpleRegressionTopoCorrectionAlgorithm import SimpleRegressionTopoCorrectionAlgorithm
from algorithms.TopoCorrectionAlgorithm import TopoCorrectionContext


class CTopoCorrectionAlgorithm(SimpleRegressionTopoCorrectionAlgorithm):
    @staticmethod
    def get_name():
        return "[direct_calc] C-correction"

    def process_band(self, ctx: TopoCorrectionContext, band_idx: int):
        c = self.calculate_c(ctx, band_idx)

        result = processing.run(
            'gdal:rastercalculator',
            {
                'INPUT_A': ctx.luminance_path,
                'BAND_A': 1,
                'INPUT_B': ctx.input_layer,
                'BAND_B': band_idx + 1,
                'FORMULA': f'B * ({ctx.sza_cosine()} + {c})/(A + {c})',
                'OUTPUT': 'TEMPORARY_OUTPUT',
            },
            feedback=ctx.qgis_feedback,
            context=ctx.qgis_context
        )
        return result['OUTPUT']

    def calculate_c(self, ctx: TopoCorrectionContext, band_idx: int) -> float:
        intercept, slope = self.get_linear_regression_coeffs(ctx, band_idx)
        return slope / intercept
