import processing

from algorithms.CTopoCorrectionAlgorithm import CTopoCorrectionAlgorithm
from algorithms.TopoCorrectionAlgorithm import TopoCorrectionContext


class ScsCTopoCorrectionAlgorithm(CTopoCorrectionAlgorithm):
    @staticmethod
    def get_name():
        return "[old] SCS+C"

    def process_band(self, ctx: TopoCorrectionContext, band_idx: int):
        c = self.calculate_c(ctx, band_idx)

        result = processing.run(
            'gdal:rastercalculator',
            {
                # create layer from luminance_layer
                'INPUT_A': ctx.luminance_path,
                'BAND_A': 1,
                'INPUT_B': ctx.input_layer,
                'BAND_B': band_idx + 1,
                'INPUT_C': ctx.slope_rad_path,
                'BAND_C': 1,
                'FORMULA': f'B*({ctx.sza_cosine()} * cos(C) + {c})/(A + {c})',
                'OUTPUT': 'TEMPORARY_OUTPUT',
                'NO_DATA': 0
            },
            feedback=ctx.qgis_feedback,
            context=ctx.qgis_context
        )
        return result['OUTPUT']
