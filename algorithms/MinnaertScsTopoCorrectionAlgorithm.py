import processing

from algorithms.MinnaertTopoCorrectionAlgorithm import MinnaertTopoCorrectionAlgorithm
from algorithms.TopoCorrectionAlgorithm import TopoCorrectionContext


class MinnaertScsTopoCorrectionAlgorithm(MinnaertTopoCorrectionAlgorithm):
    @staticmethod
    def get_name():
        return "Minnaert-SCS"

    def process_band(self, ctx: TopoCorrectionContext, band_idx: int):
        k = self.calculate_k(ctx, band_idx)

        result = processing.run(
            'gdal:rastercalculator',
            {
                'INPUT_A': ctx.luminance_path,
                'BAND_A': 1,
                'INPUT_B': ctx.input_layer,
                'BAND_B': band_idx + 1,
                'INPUT_C': ctx.slope_rad_path,
                'BAND_C': 1,
                'FORMULA': f'B * cos(C) * power({ctx.sza_cosine()}/A, {k})',
                'OUTPUT': 'TEMPORARY_OUTPUT',
            },
            feedback=ctx.qgis_feedback,
            context=ctx.qgis_context
        )
        return result['OUTPUT']
