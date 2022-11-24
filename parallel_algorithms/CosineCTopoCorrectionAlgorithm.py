from qgis._core import QgsProcessingAlgRunnerTask, QgsApplication

from algorithms.TopoCorrectionAlgorithm import TopoCorrectionContext
from parallel_algorithms.ParallelTopoCorrectionAlgorithm import ParallelTopoCorrectionAlgorithm


class CosineTParallelTopoCorrectionAlgorithm(ParallelTopoCorrectionAlgorithm):
    @staticmethod
    def get_name():
        return "[parallel]COSINE-T"

    def build_task(self, ctx: TopoCorrectionContext, band_idx: int):
        algorithm = QgsApplication.processingRegistry().algorithmById('gdal:rastercalculator')
        task = QgsProcessingAlgRunnerTask(
            algorithm=algorithm,
            parameters={
                # create layer from luminance_layer
                'INPUT_A': ctx.luminance_path,
                'BAND_A': 1,
                'INPUT_B': ctx.input_layer,
                'BAND_B': band_idx + 1,
                'FORMULA': f"{ctx.sza_cosine()} * {self.safe_divide('B', 'A')}",
                'OUTPUT': self.get_band_output_name(band_idx),
            },
            feedback=ctx.qgis_feedback,
            context=ctx.qgis_context
        )
        return task
