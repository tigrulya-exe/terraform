import processing
from computation import gdal_utils

from algorithms.TopoCorrectionAlgorithm import TopoCorrectionAlgorithm, TopoCorrectionContext


class CosineCTopoCorrectionAlgorithm(TopoCorrectionAlgorithm):
    @staticmethod
    def get_name():
        return "[old] COSINE-C"

    def process_band(self, ctx: TopoCorrectionContext, band_idx: int):
        # todo add validation
        luminance_mean = gdal_utils.compute_band_means(ctx.luminance_path)[0]

        result = processing.run(
            'gdal:rastercalculator',
            {
                'INPUT_A': ctx.luminance_path,
                'BAND_A': 1,
                'INPUT_B': ctx.input_layer,
                'BAND_B': band_idx + 1,
                'FORMULA': f'B*(1 + ({luminance_mean} - A)/{luminance_mean})',
                'OUTPUT': 'TEMPORARY_OUTPUT',
            },
            feedback=ctx.qgis_feedback,
            context=ctx.qgis_context
        )
        return result['OUTPUT']
