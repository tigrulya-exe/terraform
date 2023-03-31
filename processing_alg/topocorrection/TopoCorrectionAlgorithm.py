import os
import random
import tempfile
from typing import Dict, Any

import processing
from qgis.core import (QgsProcessingException,
                       QgsTaskManager,
                       QgsTask)

from ..execution_context import QgisExecutionContext
from ...computation.raster_calc import SimpleRasterCalc, RasterInfo


class TopoCorrectionAlgorithm:
    def __init__(self):
        self.calc = SimpleRasterCalc()
        self.task_manager = None
        self.salt = None

    @staticmethod
    def get_name():
        pass

    def init(self, ctx: QgisExecutionContext):
        self.task_manager = QgsTaskManager()
        self.salt = random.randint(1, 100000)

    def process_band(self, ctx: QgisExecutionContext, band_idx: int):
        pass

    def process(self, ctx: QgisExecutionContext) -> Dict[str, Any]:
        self.init(ctx)

        ctx.log(f"{self.get_name()} correction started: parallel={ctx.run_parallel}")
        result_bands = self._process_parallel(ctx) if ctx.run_parallel else self._process_sequentially(ctx)

        # todo change to just run without load
        results = processing.runAndLoadResults(
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
                'OUTPUT': ctx.output_file_path
            },
            feedback=ctx.qgis_feedback,
            context=ctx.qgis_context
        )

        ctx.log(f"{self.get_name()} correction finished")
        return results

    def _process_parallel(self, ctx: QgisExecutionContext):
        result_bands = []

        def task_wrapper(task, _ctx, _band_idx):
            self.process_band(_ctx, _band_idx)

        # eagerly compute slope, aspect and luminance
        _ = ctx.luminance

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

    def _process_sequentially(self, ctx: QgisExecutionContext):
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
