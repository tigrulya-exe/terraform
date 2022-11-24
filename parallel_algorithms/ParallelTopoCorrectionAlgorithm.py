import string, random
from typing import Dict, Any

import processing
from qgis._core import QgsTask, QgsApplication
from qgis.core import (QgsProcessingException)

from algorithms.TopoCorrectionAlgorithm import TopoCorrectionContext


class ParallelTopoCorrectionAlgorithm:

    @staticmethod
    def get_name():
        pass

    def init(self, ctx: TopoCorrectionContext):
        self.taskManager = QgsApplication.taskManager()
        self.salt = random_word(5)

    def build_task(self, ctx: TopoCorrectionContext, band_idx: int) -> QgsTask:
        pass

    def safe_divide(self, top: str, bottom: str) -> str:
        return self.safe_divide_check(top, bottom, bottom)

    def safe_divide_check(self, top: str, bottom: str, null_check: str) -> str:
        return f"divide({top}, {bottom}, out=zeros_like({bottom}, dtype='float32'), where={null_check}!=0)"


    def get_band_output(self, band_idx: int) -> QgsTask:
        pass

    def get_band_output_name(self, band_idx: int) -> str:
        return f'out-{self.salt}-{band_idx}'

    def process(self, ctx: TopoCorrectionContext) -> Dict[str, Any]:
        self.init(ctx)
        tasks_per_band = []

        for band_idx in range(ctx.input_layer.bandCount()):
            try:
                result = self.build_task(ctx, band_idx)
                tasks_per_band.append(result)
            except QgsProcessingException as exc:
                raise RuntimeError(f"Error during performing topocorrection: {exc}")

            if ctx.qgis_feedback.isCanceled():
                # todo
                return {}

        [self.taskManager.addTask(task) for task in tasks_per_band]

        for task in tasks_per_band:
            while task.isActive():
                if task.waitForFinished(1000):
                    continue
                if ctx.qgis_feedback.isCanceled():
                    return {}

        result_bands = [self.get_band_output_name(band) for band in range(ctx.input_layer.bandCount())]

        return processing.runAndLoadResults(
            "gdal:merge",
            {
                'INPUT': result_bands,
                'PCT': False,
                'SEPARATE': True,
                'NODATA_INPUT': None,
                'NODATA_OUTPUT': None,
                'OPTIONS': '',
                'EXTRA': '',
                'DATA_TYPE': 5,
                'OUTPUT': ctx.qgis_params['OUTPUT']
            },
            feedback=ctx.qgis_feedback,
            context=ctx.qgis_context
        )

def random_word(length):
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for _ in range(length))
