import os
import random
import tempfile
from math import cos, radians
from typing import Dict, Any

import processing
from qgis._core import QgsTaskManager
from qgis.core import (QgsRasterLayer, QgsProcessingContext, QgsProcessingFeedback, QgsProcessingException,
                       QgsTask)

from ...computation.raster_calc import SimpleRasterCalc, RasterInfo


class TopoCorrectionContext:
    def __init__(
            self,
            qgis_context: QgsProcessingContext,
            qgis_feedback: QgsProcessingFeedback,
            qgis_params: Dict[str, Any],
            input_layer: QgsRasterLayer,
            slope_rad_path: str,
            aspect_path: str,
            luminance_path: str,
            solar_zenith_angle: float,
            solar_azimuth: float,
            run_parallel: bool,
            task_timeout: int=10000):
        self.qgis_context = qgis_context
        self.qgis_params = qgis_params
        self.qgis_feedback = qgis_feedback
        self.input_layer = input_layer
        self.slope_rad_path = slope_rad_path
        self.aspect_path = aspect_path
        self.luminance_path = luminance_path
        self.solar_zenith_angle = solar_zenith_angle
        self.solar_azimuth = solar_azimuth
        self.run_parallel = run_parallel
        self.task_timeout = task_timeout

    def sza_cosine(self):
        return cos(radians(self.solar_zenith_angle))

    def azimuth_cosine(self):
        return cos(radians(self.solar_azimuth))


class TopoCorrectionAlgorithm:
    def __init__(self):
        self.calc = SimpleRasterCalc()
        self.task_manager = None
        self.salt = None

    @staticmethod
    def get_name():
        pass

    def init(self, ctx: TopoCorrectionContext):
        self.task_manager = QgsTaskManager()
        self.salt = random.randint(1, 100000)

    def process_band(self, ctx: TopoCorrectionContext, band_idx: int):
        pass

    def process(self, ctx: TopoCorrectionContext) -> Dict[str, Any]:
        self.init(ctx)

        result_bands = self._process_parallel(ctx) if ctx.run_parallel else self._process_sequentially(ctx)

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

    def _process_parallel(self, ctx: TopoCorrectionContext):
        result_bands = []

        def task_wrapper(task, _ctx, _band_idx):
            self.process_band(_ctx, _band_idx)

        for band_idx in range(ctx.input_layer.bandCount()):
            try:
                task = QgsTask.fromFunction(f'Task for band {band_idx}', task_wrapper,
                                            _ctx=ctx, _band_idx=band_idx)
                self.task_manager.addTask(task)
            except QgsProcessingException as exc:
                self.task_manager.cancelAll()
                raise RuntimeError(f"Error during performing topocorrection: {exc}")

            if ctx.qgis_feedback.isCanceled():
                self.task_manager.cancelAll()
                return None

            result_bands.append(self.output_file_path(str(band_idx)))

        for task in self.task_manager.tasks():
            if not task.waitForFinished(ctx.task_timeout):
                raise RuntimeError(f"Timeout exception for task {task.description()}")
            ctx.qgis_feedback.pushInfo(f"Task {task.description()} finished")
            if ctx.qgis_feedback.isCanceled():
                self.task_manager.cancelAll()
                return None

        return result_bands

    def _process_sequentially(self, ctx: TopoCorrectionContext):
        result_bands = []

        for band_idx in range(ctx.input_layer.bandCount()):
            try:
                result = self.process_band(ctx, band_idx)
                result_bands.append(result)
            except QgsProcessingException as exc:
                raise RuntimeError(f"Error during performing topocorrection: {exc}")

            if ctx.qgis_feedback.isCanceled():
                return None

        return result_bands

    def output_file_path(self, postfix=''):
        return os.path.join(
            tempfile.gettempdir(),
            f'{self.get_name()}_{self.salt}_{postfix}.tif'
        )

    def raster_calculate(self, calc_func, raster_infos: list[RasterInfo], out_file_postfix=''):
        out_path = self.output_file_path(out_file_postfix)
        self.calc.calculate(
            func=calc_func,
            output_path=out_path,
            raster_infos=raster_infos
        )
        return out_path
